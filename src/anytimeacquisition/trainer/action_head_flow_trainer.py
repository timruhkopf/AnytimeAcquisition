"""FlowMatchingActionHead behavior-cloning trainer -- the chunked-plan
counterpart to `trainer.action_head_imitation_trainer.ActionHeadImitationTrainer`,
trained against real multi-step targets (`trainer.exit_rollout`'s
`build_explore_chunk_buffer`/`build_exploit_chunk_buffer`) instead of
single points. See `docs/ROADMAP.md`'s VLA-architecture lens and
`docs/MILESTONES.md`'s prioritized experiment 4 for why this exists and
why it's deliberately NOT yet the default training path: it's architecture
groundwork, meant to be compared against the simple `ActionHeadImitationTrainer`
on the same `auc_improvement_vs_random` bar once that trainer's own open
label-quality/seeding questions are settled, not a replacement decided in
advance.

Deliberately narrower in scope than `ActionHeadImitationTrainer` -- same
DAgger-mixing/subsampling/checkpointing shape, but without that trainer's
full diagnostic surface (`exploit/target_distance`,
`explore/weighted_nll_reduction`, etc.) reproduced here; add those if/when
this path is promoted past groundwork, not preemptively."""
from pathlib import Path
from typing import Callable

import torch

from anytimeacquisition.callbacks.handler import Callback, CallbackHandler
from anytimeacquisition.models.action_head import build_rollout_aux_features
from anytimeacquisition.models.action_head_flow import FlowMatchingActionHead, flow_action_head_policy_fn
from anytimeacquisition.models.bar_distribution import BarDistribution
from anytimeacquisition.models.pfn import PFN
from anytimeacquisition.priors.bnn import BNNPrior
from anytimeacquisition.trainer.exit_rollout import (
    ImitationChunkExample,
    build_exploit_chunk_buffer,
    build_explore_chunk_buffer,
    label_branches,
    mixed_policy_fn,
    random_policy,
    rollout_episode,
)


class ActionHeadFlowTrainer:
    def __init__(
        self,
        pfn: PFN,
        bar_dist: BarDistribution,
        prior: BNNPrior,
        action_head: FlowMatchingActionHead,
        seed: int = 0,
        n_rollouts: int = 500,
        n_init: int = 5,
        n_steps: int = 20,
        lr: float = 1e-3,
        log_every: int = 10,
        # Same double-duty branch selection as ActionHeadImitationTrainer
        # (dict = on, None = off) -- see that trainer's own module
        # docstring for why, unchanged reasoning here.
        exploit_search_kwargs: dict | None = None,
        explore_search_kwargs: dict | None = None,
        build_interesting_points_kwargs: dict | None = None,
        # Must match action_head.chunk_len -- not re-derived automatically
        # (action_head is passed in already-constructed, same pattern as
        # ActionHeadImitationTrainer's own action_head), asserted in
        # __init__ instead of silently mismatching mid-run.
        chunk_len: int = 3,
        num_sample_steps: int = 10,
        require_improvement: bool = True,
        require_regret_improvement: bool = False,
        dagger_decay_rounds: int | None | str = "auto",
        dagger_beta_min: float = 0.05,
        max_explore_steps_per_rollout: int | None = None,
        fill_unselected_explore_steps_with_exploit: bool = False,
        checkpoint_path: str | Path | None = None,
        model_config: dict | None = None,
        on_log: Callable[[int, dict], None] | None = None,
        extra_checkpoint_metadata: dict | None = None,
        callbacks: list[Callback] | None = None,
    ):
        assert exploit_search_kwargs is not None or explore_search_kwargs is not None, \
            "at least one of exploit_search_kwargs/explore_search_kwargs must be set (not None) -- " \
            "a None value means that branch is off, and training needs at least one branch on"
        if explore_search_kwargs is not None:
            assert build_interesting_points_kwargs is not None, \
                "explore_search_kwargs is set (explore branch on) -- build_interesting_points_kwargs " \
                "is required to build x_int/y_int_true"
        assert action_head.chunk_len == chunk_len, (
            f"action_head.chunk_len ({action_head.chunk_len}) must match this trainer's chunk_len ({chunk_len})"
        )

        self.pfn = pfn
        self.bar_dist = bar_dist
        self.prior = prior
        self.action_head = action_head
        self.branches = [
            name for name, kwargs in (("exploit", exploit_search_kwargs), ("explore", explore_search_kwargs))
            if kwargs is not None
        ]
        self.seed = seed
        self.n_rollouts = n_rollouts
        self.n_init = n_init
        self.n_steps = n_steps
        self.lr = lr
        self.log_every = log_every
        self.exploit_search_kwargs = exploit_search_kwargs or {}
        self.explore_search_kwargs = explore_search_kwargs or {}
        self.build_interesting_points_kwargs = build_interesting_points_kwargs
        self.chunk_len = chunk_len
        self.num_sample_steps = num_sample_steps
        self.require_improvement = require_improvement
        self.require_regret_improvement = require_regret_improvement
        self.dagger_decay_rounds = dagger_decay_rounds
        self.dagger_beta_min = dagger_beta_min
        self.max_explore_steps_per_rollout = max_explore_steps_per_rollout
        self.fill_unselected_explore_steps_with_exploit = fill_unselected_explore_steps_with_exploit
        self.checkpoint_path = checkpoint_path
        self.model_config = model_config
        self.on_log = on_log
        self.extra_checkpoint_metadata = extra_checkpoint_metadata
        self.callback_handler = CallbackHandler(callbacks)

    def _collect_examples(self, rollout: dict) -> tuple[list[ImitationChunkExample], dict]:
        examples: list[ImitationChunkExample] = []
        extra: dict = {}

        if "exploit" in self.branches:
            examples += build_exploit_chunk_buffer(
                self.prior, rollout, self.n_init, self.chunk_len, self.exploit_search_kwargs,
            )

        if "explore" in self.branches:
            is_explore = ~label_branches(rollout["y_context"], self.n_init)
            eligible = [s for s in range(self.n_steps) if is_explore[:, s].any()]
            if self.max_explore_steps_per_rollout is not None and len(eligible) > self.max_explore_steps_per_rollout:
                perm = torch.randperm(len(eligible))[: self.max_explore_steps_per_rollout]
                selected = {eligible[i] for i in perm.tolist()}
            else:
                selected = set(eligible)
            unselected = set(eligible) - selected

            explore_examples = build_explore_chunk_buffer(
                self.prior, self.pfn, self.bar_dist, rollout, self.n_init, self.chunk_len,
                self.explore_search_kwargs, steps=selected if self.max_explore_steps_per_rollout is not None else None,
                require_improvement=self.require_improvement, require_regret_improvement=self.require_regret_improvement,
            )
            examples += explore_examples
            n_eligible_selected = sum(int(is_explore[:, s].sum().item()) for s in selected)
            extra["explore/signal_rate_train"] = (
                len(explore_examples) / n_eligible_selected if n_eligible_selected else float("nan")
            )

            if "exploit" in self.branches and self.fill_unselected_explore_steps_with_exploit and unselected:
                filler = build_exploit_chunk_buffer(
                    self.prior, rollout, self.n_init, self.chunk_len, self.exploit_search_kwargs,
                    steps=unselected, require_exploit_label=False,
                )
                examples += filler
                extra["n_examples/exploit_filler"] = float(len(filler))

        return examples, extra

    def _step_loss(self, step: int, rollout: dict, step_examples: list[ImitationChunkExample]) -> tuple[torch.Tensor, dict]:
        x_context = torch.stack([ex.x_context for ex in step_examples])
        y_context = torch.stack([ex.y_context for ex in step_examples])
        target_chunk = torch.stack([ex.target_chunk for ex in step_examples])
        aux = build_rollout_aux_features(rollout, step, self.n_steps)
        idx = torch.tensor([ex.instance_idx for ex in step_examples])
        aux = {k: v[idx] for k, v in aux.items()}

        per_example_loss = self.action_head.compute_loss(self.pfn, x_context, y_context, aux, target_chunk)
        branch_sums = {"exploit": torch.zeros(()), "explore": torch.zeros(())}
        for i, ex in enumerate(step_examples):
            branch_sums[ex.branch] = branch_sums[ex.branch] + per_example_loss[i]
        return per_example_loss.sum(), branch_sums

    def run(self) -> dict:
        torch.manual_seed(self.seed)
        opt = torch.optim.AdamW(self.action_head.parameters(), lr=self.lr)
        history = {"step": []}
        decay_rounds = self.n_rollouts if self.dagger_decay_rounds == "auto" else self.dagger_decay_rounds

        for rollout_idx in range(self.n_rollouts):
            build_ip_kwargs = self.build_interesting_points_kwargs if "explore" in self.branches else None

            usage_counter = None
            if decay_rounds is not None:
                beta = max(self.dagger_beta_min, 1.0 - rollout_idx / decay_rounds)
                usage_counter = {}
                policy_fn = mixed_policy_fn(
                    random_policy,
                    flow_action_head_policy_fn(self.action_head, self.pfn, self.n_steps, self.num_sample_steps),
                    beta, usage_counter=usage_counter,
                )
            else:
                beta = 1.0
                policy_fn = random_policy

            rollout = rollout_episode(
                self.prior, self.n_init, self.n_steps, policy_fn=policy_fn,
                build_interesting_points_kwargs=build_ip_kwargs,
            )
            examples, extra_metrics = self._collect_examples(rollout)
            n_exploit = sum(ex.branch == "exploit" for ex in examples)
            n_explore = sum(ex.branch == "explore" for ex in examples)

            by_step: dict[int, list[ImitationChunkExample]] = {}
            for ex in examples:
                by_step.setdefault(ex.step, []).append(ex)

            total_loss = torch.zeros(())
            branch_totals = {"exploit": torch.zeros(()), "explore": torch.zeros(())}
            for step, step_examples in by_step.items():
                step_loss, branch_sums = self._step_loss(step, rollout, step_examples)
                total_loss = total_loss + step_loss
                for k, v in branch_sums.items():
                    branch_totals[k] = branch_totals[k] + v

            if by_step:
                total_loss = total_loss / max(len(examples), 1)
                opt.zero_grad()
                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.action_head.parameters(), max_norm=10.0)
                opt.step()
            else:
                grad_norm = None

            if rollout_idx % self.log_every == 0 or rollout_idx == self.n_rollouts - 1:
                mean_loss = total_loss.item() if by_step else 0.0
                metrics = {
                    "policy_loss/train": mean_loss,
                    "n_examples/exploit": float(n_exploit),
                    "n_examples/explore": float(n_explore),
                    "dagger/beta": beta,
                }
                if grad_norm is not None:
                    metrics["grad_norm/action_head"] = grad_norm.item()
                if usage_counter:
                    total_actions = usage_counter.get("a", 0) + usage_counter.get("b", 0)
                    metrics["dagger/frac_self_generated"] = (
                        usage_counter.get("b", 0) / total_actions if total_actions else float("nan")
                    )
                metrics.update(extra_metrics)
                if "exploit" in self.branches:
                    metrics["policy_loss/train_exploit"] = (
                        branch_totals["exploit"].item() / n_exploit if n_exploit else float("nan")
                    )
                if "explore" in self.branches:
                    metrics["policy_loss/train_explore"] = (
                        branch_totals["explore"].item() / n_explore if n_explore else float("nan")
                    )
                metrics.update(self.callback_handler.run(rollout_idx, self, self.log_every))

                history["step"].append(rollout_idx)
                for k, v in metrics.items():
                    history.setdefault(k, []).append(v)

                print(f"rollout {rollout_idx:5d}  policy_loss/train={metrics['policy_loss/train']:.4f}  "
                      f"n_exploit={n_exploit}  n_explore={n_explore}")
                if self.on_log is not None:
                    self.on_log(rollout_idx, metrics)

        if self.checkpoint_path is not None:
            checkpoint_path = Path(self.checkpoint_path)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": self.action_head.state_dict(),
                    "config": self.model_config,
                    "history": history,
                    **(self.extra_checkpoint_metadata or {}),
                },
                checkpoint_path,
            )
            print("saved checkpoint to", checkpoint_path)

        return {"action_head": self.action_head, "prior": self.prior, "history": history}


if __name__ == "__main__":
    """Smoke demo: a tiny integrated (exploit+explore) run on a fresh
    x_dim=1 PFN checkpoint, showing policy_loss/train actually decreasing --
    same shape as ActionHeadImitationTrainer's own __main__ demo."""
    from anytimeacquisition.pipelines.train_pfn import load_pfn_checkpoint
    from anytimeacquisition.models.action_head import pfn_dims
    from anytimeacquisition.utils.paths import CHECKPOINT_DIR

    checkpoint_path = CHECKPOINT_DIR / "pfn_smoke_xdim1.pt"
    if not checkpoint_path.exists():
        raise SystemExit(
            f"No checkpoint at {checkpoint_path} -- train one first:\n"
            "  uv run python -m anytimeacquisition.pipelines.train_pfn "
            "experiment=pfn_smoke_xdim1 allow_dirty=true"
        )
    pfn, bar_dist, ckpt = load_pfn_checkpoint(checkpoint_path)
    x_dim = ckpt["config"]["max_x_dim"]
    d_model, n_layers = pfn_dims(pfn)

    torch.manual_seed(0)
    chunk_len = 3
    action_head = FlowMatchingActionHead(pfn_d_model=d_model, pfn_n_layers=n_layers, x_dim=x_dim, chunk_len=chunk_len)
    prior = BNNPrior(batch_size=4, x_dim=x_dim, seed=1)

    trainer = ActionHeadFlowTrainer(
        pfn=pfn, bar_dist=bar_dist, prior=prior, action_head=action_head,
        n_rollouts=20, n_init=4, n_steps=8, log_every=5, chunk_len=chunk_len,
        exploit_search_kwargs={"n_restarts": 4, "n_steps": 15},
        explore_search_kwargs={"n_restarts": 2, "n_steps": 10},
        build_interesting_points_kwargs={"n_sobol": 8, "n_random": 8, "n_basin_restarts": 4},
    )
    result = trainer.run()
    losses = result["history"]["policy_loss/train"]
    print(f"policy_loss/train: {losses[0]:.4f} -> {losses[-1]:.4f}")
