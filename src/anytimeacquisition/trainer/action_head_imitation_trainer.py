"""ActionHead behavior-cloning trainer (M5) -- the piece
`trainer/exit_rollout.py`'s own docstring lists first under "What's NOT
here yet": nothing in this repo, before this file, actually trains the
`ActionHead` against the privileged-search oracle tuples that module
builds. Same `_target_`-instantiable shape as `PFNTrainer`
(`trainer/pfn_trainer.py`) -- runtime objects (`pfn`, `bar_dist`, `prior`,
`action_head`) passed in at instantiate time, everything else a plain
config-driven hyperparameter on `self`, `run()` takes no arguments.

Scope, per the M5 plan (see docs/log/ and the plan this trainer was built
from): **round-0 only, no DAgger iteration** -- every rollout here uses
`trainer.exit_rollout.random_policy` (matching "learn to exploit/explore
from a random trajectory"); re-rolling out under the ActionHead being
trained and repeating is a separate, not-yet-built milestone. **No
persistent replay buffer across rollouts** -- each freshly-sampled
rollout's examples are grouped by step (same context length within a
step, no padding needed), trained on immediately via one optimizer step
per rollout (loss summed across all of that rollout's examples, not one
step per step-group), then discarded.

`branches: list[str]` (subset of `["exploit", "explore"]`) selects which
oracle(s) label this rollout's examples -- `["exploit"]`/`["explore"]` give
the two marginal behavior-cloning trainers, `["exploit", "explore"]` gives
the integrated one; all three are this same class with a different
config, not three different pieces of code.

`train_value_head`: off by default. `ImitationExample.y_star`'s units are
branch-dependent (true-y for exploit, weighted-NLL for explore --
`exit_rollout.py`'s own docstring is explicit about never conflating
them), so a value loss is only ever computed and logged per-branch, never
mixed across branches -- see `_value_loss` below.
"""
from pathlib import Path
from typing import Callable

import torch

from anytimeacquisition.callbacks.handler import Callback, CallbackHandler
from anytimeacquisition.models.action_head import ActionHead, build_rollout_aux_features, beta_mode
from anytimeacquisition.models.bar_distribution import BarDistribution
from anytimeacquisition.models.pfn import PFN
from anytimeacquisition.priors.bnn import BNNPrior
from anytimeacquisition.trainer.exit_rollout import (
    ImitationExample,
    build_exploit_buffer,
    build_explore_buffer,
    random_policy,
    rollout_episode,
)


def _beta_nll_loss(alpha: torch.Tensor, beta: torch.Tensor, target_x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """-> [B], summed over x_dim (product-of-marginals) -- same loss as
    `pipelines/action_head_posterior_distill.py::beta_nll_loss`, duplicated
    rather than imported since that module is a `pipelines/` entry point,
    not a shared library location, and this is the only line that would be
    shared."""
    target_x = target_x.clamp(eps, 1.0 - eps)
    return -torch.distributions.Beta(alpha, beta).log_prob(target_x).sum(dim=-1)


class ActionHeadImitationTrainer:
    def __init__(
        self,
        pfn: PFN,
        bar_dist: BarDistribution,
        prior: BNNPrior,
        action_head: ActionHead,
        branches: list[str],
        seed: int = 0,
        n_rollouts: int = 500,
        n_init: int = 5,
        n_steps: int = 20,
        lr: float = 1e-3,
        log_every: int = 10,
        exploit_search_kwargs: dict | None = None,
        explore_search_kwargs: dict | None = None,
        build_interesting_points_kwargs: dict | None = None,
        train_value_head: bool = False,
        checkpoint_path: str | Path | None = None,
        model_config: dict | None = None,
        on_log: Callable[[int, dict], None] | None = None,
        extra_checkpoint_metadata: dict | None = None,
        callbacks: list[Callback] | None = None,
    ):
        assert branches and set(branches) <= {"exploit", "explore"}, \
            f"branches must be a nonempty subset of ('exploit', 'explore'), got {branches}"
        if "explore" in branches:
            assert build_interesting_points_kwargs is not None, \
                "branches includes 'explore' -- build_interesting_points_kwargs is required to build x_int/y_int_true"

        self.pfn = pfn
        self.bar_dist = bar_dist
        self.prior = prior
        self.action_head = action_head
        self.branches = list(branches)
        self.seed = seed
        self.n_rollouts = n_rollouts
        self.n_init = n_init
        self.n_steps = n_steps
        self.lr = lr
        self.log_every = log_every
        self.exploit_search_kwargs = exploit_search_kwargs or {}
        self.explore_search_kwargs = explore_search_kwargs or {}
        self.build_interesting_points_kwargs = build_interesting_points_kwargs
        self.train_value_head = train_value_head
        self.checkpoint_path = checkpoint_path
        self.model_config = model_config
        self.on_log = on_log
        self.extra_checkpoint_metadata = extra_checkpoint_metadata
        self.callback_handler = CallbackHandler(callbacks)

    def _collect_examples(self, rollout: dict) -> list[ImitationExample]:
        examples: list[ImitationExample] = []
        if "exploit" in self.branches:
            examples += build_exploit_buffer(self.prior, rollout, self.n_init, self.exploit_search_kwargs)
        if "explore" in self.branches:
            examples += build_explore_buffer(
                self.prior, self.pfn, self.bar_dist, rollout, self.n_init, self.explore_search_kwargs,
            )
        return examples

    def _step_loss(self, step: int, rollout: dict, step_examples: list[ImitationExample]) -> tuple[torch.Tensor, dict]:
        """One step-group's worth of examples (same context length within
        a rollout step -> stackable with no padding). -> (loss summed over
        this group, {"exploit": n, "explore": n} loss sums for logging)."""
        x_context = torch.stack([ex.x_context for ex in step_examples])
        y_context = torch.stack([ex.y_context for ex in step_examples])
        x_star = torch.stack([ex.x_star for ex in step_examples])
        aux = build_rollout_aux_features(rollout, step, self.n_steps)
        idx = torch.tensor([ex.instance_idx for ex in step_examples])
        aux = {k: v[idx] for k, v in aux.items()}

        out = self.action_head(self.pfn, x_context, y_context, aux, blind=False)
        per_example_nll = _beta_nll_loss(out["alpha"], out["beta"], x_star)

        branch_sums = {"exploit": torch.zeros(()), "explore": torch.zeros(())}
        for i, ex in enumerate(step_examples):
            branch_sums[ex.branch] = branch_sums[ex.branch] + per_example_nll[i]
        loss = per_example_nll.sum()

        if self.train_value_head:
            y_star = torch.stack([ex.y_star for ex in step_examples])
            loss = loss + torch.nn.functional.mse_loss(out["value"], y_star, reduction="sum")

        return loss, branch_sums

    def run(self) -> dict:
        # BNNPrior owns its own seeded generator (self.prior's own `seed`
        # kwarg, set at construction) for reset()/sample_episode/evaluate --
        # this is what makes the rollout sequence reproducible across runs.
        # random_policy and the search restarts (exploit_search/
        # explore_search/build_interesting_points) draw from the *global*
        # torch RNG instead (same as exit_rollout.py's own __main__ demo) --
        # seeded once here, not reseeded per rollout, so the whole run's
        # randomness stream advances normally rather than resetting.
        torch.manual_seed(self.seed)
        opt = torch.optim.AdamW(self.action_head.parameters(), lr=self.lr)
        history = {"step": []}

        for rollout_idx in range(self.n_rollouts):
            build_ip_kwargs = self.build_interesting_points_kwargs if "explore" in self.branches else None
            rollout = rollout_episode(
                self.prior, self.n_init, self.n_steps, policy_fn=random_policy,
                build_interesting_points_kwargs=build_ip_kwargs,
            )
            examples = self._collect_examples(rollout)
            n_exploit = sum(ex.branch == "exploit" for ex in examples)
            n_explore = sum(ex.branch == "explore" for ex in examples)

            by_step: dict[int, list[ImitationExample]] = {}
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
                opt.zero_grad()
                total_loss.backward()
                opt.step()

            if rollout_idx % self.log_every == 0 or rollout_idx == self.n_rollouts - 1:
                n_examples = max(len(examples), 1)
                metrics = {
                    "policy_nll/train": total_loss.item() / n_examples,
                    "n_examples/exploit": float(n_exploit),
                    "n_examples/explore": float(n_explore),
                }
                if "exploit" in self.branches:
                    metrics["policy_nll/train_exploit"] = (
                        branch_totals["exploit"].item() / n_exploit if n_exploit else float("nan")
                    )
                if "explore" in self.branches:
                    metrics["policy_nll/train_explore"] = (
                        branch_totals["explore"].item() / n_explore if n_explore else float("nan")
                    )
                metrics.update(self.callback_handler.run(rollout_idx, self, self.log_every))

                history["step"].append(rollout_idx)
                for k, v in metrics.items():
                    history.setdefault(k, []).append(v)

                print(f"rollout {rollout_idx:5d}  policy_nll/train={metrics['policy_nll/train']:.4f}  "
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
    x_dim=1 PFN checkpoint, showing policy_nll/train actually decreasing."""
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
    action_head = ActionHead(pfn_d_model=d_model, pfn_n_layers=n_layers, x_dim=x_dim)
    prior = BNNPrior(batch_size=4, x_dim=x_dim, seed=1)

    trainer = ActionHeadImitationTrainer(
        pfn=pfn, bar_dist=bar_dist, prior=prior, action_head=action_head,
        branches=["exploit", "explore"], n_rollouts=20, n_init=4, n_steps=8, log_every=5,
        exploit_search_kwargs={"n_restarts": 4, "n_steps": 15},
        explore_search_kwargs={"n_restarts": 4, "n_steps": 10},
        build_interesting_points_kwargs={"n_sobol": 8, "n_random": 8, "n_basin_restarts": 4},
    )
    result = trainer.run()
    losses = result["history"]["policy_nll/train"]
    print(f"policy_nll/train: {losses[0]:.4f} -> {losses[-1]:.4f}")
