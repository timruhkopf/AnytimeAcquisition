"""ActionHead behavior-cloning trainer (M5) -- the piece
`trainer/exit_rollout.py`'s own docstring lists first under "What's NOT
here yet": nothing in this repo, before this file, actually trains the
`ActionHead` against the privileged-search oracle tuples that module
builds. Same `_target_`-instantiable shape as `PFNTrainer`
(`trainer/pfn_trainer.py`) -- runtime objects (`pfn`, `bar_dist`, `prior`,
`action_head`) passed in at instantiate time, everything else a plain
config-driven hyperparameter on `self`, `run()` takes no arguments.

Scope, per the M5 plan (see docs/log/ and the plan this trainer was built
from): **DAgger-style rollout mixing is the default** (`dagger_decay_rounds
= "auto"`, 2026-09-01, user-directed) -- round 0 rolls out under pure
`trainer.exit_rollout.random_policy` (matching "learn to exploit/explore
from a random trajectory"), then rollouts phase in the ActionHead's own
(stochastic) behavior via a linearly-decaying `beta = P(random_policy)`
(1.0 -> `dagger_beta_min`, spanning this run's own `n_rollouts` by
default), per `trainer.exit_rollout.mixed_policy_fn`. Pass
`dagger_decay_rounds=None` to disable mixing entirely (pure round-0-only
behavior, for ablations/back-compat). **Still no persistent replay
buffer across rollouts, deliberately** -- every rollout draws a brand-new
BNN instance regardless of which policy generates it, so an old buffer
entry's context would reflect an earlier, less-trained rollout policy's
behavior on a task nobody will ever see again; each round's oracle labels
stay computed fresh, against that round's own context and `x_int`
selection, then discarded, same as before. Each freshly-sampled rollout's
examples are grouped by step (same context length within a step, no
padding needed), trained on immediately via one optimizer step per
rollout (loss summed across all of that rollout's examples, not one step
per step-group), then discarded.

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
from anytimeacquisition.models.action_head import ActionHead, action_head_policy_fn, build_rollout_aux_features, beta_mode
from anytimeacquisition.models.bar_distribution import BarDistribution
from anytimeacquisition.models.pfn import PFN
from anytimeacquisition.priors.bnn import BNNPrior
from anytimeacquisition.search.explore import improvement_weights
from anytimeacquisition.trainer.exit_rollout import (
    ImitationExample,
    build_exploit_buffer,
    build_explore_buffer,
    label_branches,
    mixed_policy_fn,
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
        # DAgger-style rollout policy (2026-09-01, default flipped to ON
        # 2026-09-01 -- user-directed: mixed rollouts should be the default,
        # phased IN over random ones, not opt-in). "auto" (default) spans
        # the mix over this trainer's own n_rollouts; an int overrides the
        # span; None disables mixing entirely (pure round-0-only
        # random_policy throughout, for ablations/back-compat). Mixed
        # per-instance against random_policy via a linearly-decaying beta
        # (trainer.exit_rollout.mixed_policy_fn): beta = P(random_policy)
        # this rollout = max(dagger_beta_min, 1 - round/dagger_decay_rounds)
        # -- starts at 1.0 (round 0: pure random, matching the round-0
        # self-play seeding design) and decays toward dagger_beta_min
        # (mostly self-generated, floor of permanent random exploration
        # retained) as round -> dagger_decay_rounds. (Was inverted until
        # 2026-09-01: random_policy and the ActionHead's own rollout were
        # passed to mixed_policy_fn in the wrong order, so beta actually
        # controlled P(self-generated) decaying FROM 1.0 -- i.e. round 0
        # rolled out under an untrained ActionHead and phased OUT
        # self-play over training, backwards from both this docstring's
        # own stated intent and the round-0 self-play design.) Deliberately
        # NO persistent replay buffer alongside this -- every rollout still
        # draws a brand-new BNN instance (never revisited), so an old
        # buffer entry's context reflects an earlier, less-trained
        # rollout policy's behavior on a task nobody will ever see again;
        # each round's oracle labels are always computed fresh, against
        # that round's own context and x_int selection, then discarded
        # (same as before) -- see docs/log/ for the fuller reasoning.
        dagger_decay_rounds: int | None | str = "auto",
        dagger_beta_min: float = 0.05,
        # Explore-step subsampling (2026-09-01, compute-cost finding: a
        # real-scale (x_dim=6) rollout showed explore-labeled steps
        # vastly outnumbering exploit-labeled ones -- ~600 vs ~20 examples
        # per rollout -- making explore_search's PFN forward/backward the
        # dominant cost, ~2.3min/rollout at n_steps=40). None (default):
        # every explore-labeled step gets a target, unchanged. Set to an
        # int to randomly select at most that many DISTINCT STEP INDICES
        # (not instances) per rollout to actually run explore_search on --
        # cuts the number of explore_search calls (and their GD-step-count
        # x PFN forward/backward cost) roughly proportionally; the
        # unselected steps' explore-labeled instances simply get no
        # training example this rollout (0 loss contribution), same
        # skip-not-fabricate treatment as has_signal=False already gets.
        max_explore_steps_per_rollout: int | None = None,
        # If True (and max_explore_steps_per_rollout actually drops some
        # steps), those unselected steps' flat instances get an EXPLOIT
        # target instead of nothing -- exploit_search is far cheaper
        # (only touches prior.evaluate, no PFN forward/backward at all),
        # so this is close to free extra signal, not a new cost driver.
        # Semantically sound, not just convenient: exploit_search finds a
        # locally-better point given the CURRENT context regardless of
        # whether THIS trajectory's last step happened to already realize
        # an improvement -- label_branches is a which-oracle-to-consult
        # routing choice, not a claim the exploit oracle is only valid at
        # naturally-improving steps (see build_exploit_buffer's
        # `require_exploit_label` docstring). Never creates a second,
        # conflicting target for an instance that already got one this
        # step (filler only applies to steps NOT selected for explore,
        # and only to instances NOT already exploit-labeled there) --
        # False by default, opt-in for ablation against the plain
        # subsampling-only behavior.
        fill_unselected_explore_steps_with_exploit: bool = False,
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
        self.dagger_decay_rounds = dagger_decay_rounds
        self.dagger_beta_min = dagger_beta_min
        self.max_explore_steps_per_rollout = max_explore_steps_per_rollout
        self.fill_unselected_explore_steps_with_exploit = fill_unselected_explore_steps_with_exploit
        self.checkpoint_path = checkpoint_path
        self.model_config = model_config
        self.on_log = on_log
        self.extra_checkpoint_metadata = extra_checkpoint_metadata
        self.callback_handler = CallbackHandler(callbacks)

    def _collect_examples(self, rollout: dict) -> tuple[list[ImitationExample], dict]:
        """-> (examples, extra_metrics). `extra_metrics` carries this
        rollout's subsampling/filler bookkeeping (`explore/signal_rate_train`,
        `n_examples/exploit_filler`) -- computed here since this is the one
        place that already has the per-step eligible/selected/unselected
        step sets in hand."""
        examples: list[ImitationExample] = []
        extra: dict = {}

        if "exploit" in self.branches:
            examples += build_exploit_buffer(self.prior, rollout, self.n_init, self.exploit_search_kwargs)

        if "explore" in self.branches:
            is_explore = ~label_branches(rollout["y_context"], self.n_init)  # [B, n_steps]
            eligible = [s for s in range(self.n_steps) if is_explore[:, s].any()]

            if self.max_explore_steps_per_rollout is not None and len(eligible) > self.max_explore_steps_per_rollout:
                perm = torch.randperm(len(eligible))[: self.max_explore_steps_per_rollout]
                selected = {eligible[i] for i in perm.tolist()}
            else:
                selected = set(eligible)
            unselected = set(eligible) - selected

            explore_examples = build_explore_buffer(
                self.prior, self.pfn, self.bar_dist, rollout, self.n_init, self.explore_search_kwargs,
                steps=selected if self.max_explore_steps_per_rollout is not None else None,
            )
            examples += explore_examples
            n_eligible_selected = sum(int(is_explore[:, s].sum().item()) for s in selected)
            # Now reflects BOTH build_explore_buffer gates (2026-09-01):
            # has_signal (zero-weight x_int set) AND require_improvement
            # (correction didn't actually reduce weighted NLL) -- naturally
            # drops further than before that second gate existed, that's
            # the fix working as intended, not a regression.
            extra["explore/signal_rate_train"] = (
                len(explore_examples) / n_eligible_selected if n_eligible_selected else float("nan")
            )

            if "exploit" in self.branches and self.fill_unselected_explore_steps_with_exploit and unselected:
                filler = build_exploit_buffer(
                    self.prior, rollout, self.n_init, self.exploit_search_kwargs,
                    steps=unselected, require_exploit_label=False,
                )
                examples += filler
                extra["n_examples/exploit_filler"] = float(len(filler))

        return examples, extra

    def _step_loss(self, step: int, rollout: dict, step_examples: list[ImitationExample]) -> tuple[torch.Tensor, dict, dict]:
        """One step-group's worth of examples (same context length within
        a rollout step -> stackable with no padding). -> (loss summed over
        this group, {"exploit": n, "explore": n} loss sums for logging,
        diagnostic sums for policy/beta_entropy + exploit/target_distance)."""
        x_context = torch.stack([ex.x_context for ex in step_examples])
        y_context = torch.stack([ex.y_context for ex in step_examples])
        x_star = torch.stack([ex.x_star for ex in step_examples])
        aux = build_rollout_aux_features(rollout, step, self.n_steps)
        idx = torch.tensor([ex.instance_idx for ex in step_examples])
        aux = {k: v[idx] for k, v in aux.items()}

        out = self.action_head(self.pfn, x_context, y_context, aux, blind=False)
        per_example_nll = _beta_nll_loss(out["alpha"], out["beta"], x_star)

        # policy/beta_entropy: mean entropy of the predicted per-dimension
        # Beta distributions -- a standard BC/policy diagnostic that's
        # otherwise entirely invisible: detects premature collapse to
        # overconfident point predictions, or a policy that stays
        # uselessly close to uniform throughout training.
        entropy_sum = torch.distributions.Beta(out["alpha"], out["beta"]).entropy().sum()

        # exploit/target_distance: mean L2 distance between x_star and the
        # incumbent it was seeded from, for exploit-branch examples in this
        # group only -- confirms the incumbent-seeded, local-only redesign
        # (2026-09-01) is actually staying local, not drifting back toward
        # the "target may outrun context" failure mode
        # (docs/log/2026-08-28-exploit-search-target-may-outrun-context.md).
        target_distance_sum = torch.zeros(())
        n_exploit_examples = 0
        for i, ex in enumerate(step_examples):
            if ex.branch != "exploit":
                continue
            incumbent_x = ex.x_context[ex.y_context.argmin()]
            target_distance_sum = target_distance_sum + (ex.x_star - incumbent_x).norm()
            n_exploit_examples += 1

        branch_sums = {"exploit": torch.zeros(()), "explore": torch.zeros(())}
        for i, ex in enumerate(step_examples):
            branch_sums[ex.branch] = branch_sums[ex.branch] + per_example_nll[i]
        loss = per_example_nll.sum()

        if self.train_value_head:
            y_star = torch.stack([ex.y_star for ex in step_examples])
            loss = loss + torch.nn.functional.mse_loss(out["value"], y_star, reduction="sum")

        # explore/weighted_nll_reduction: how much the oracle correction
        # actually reduced the weighted NLL at x_int vs. doing nothing
        # (weighted_before - ex.y_star, ex.y_star already IS the achieved
        # "after" value from search.explore.explore_search) -- unlike
        # exploit_search, explore_search has no incumbent-style fallback
        # guarantee (see its own module docstring: "no privileged known-
        # good x to fall back to here"), so this can occasionally be small
        # or negative -- worth surfacing directly, not assuming it's
        # always positive. One batched extra PFN forward pass per
        # step-group (all this step's explore examples at once), not one
        # per example -- same idiom `search/explore.py`'s own `__main__`
        # demo uses to compute the "before" value.
        explore_reduction_sum = torch.zeros(())
        n_explore_examples = 0
        explore_idx = [i for i, ex in enumerate(step_examples) if ex.branch == "explore"]
        if explore_idx and "x_int" in rollout:
            explore_examples = [step_examples[i] for i in explore_idx]
            x_int_b = torch.stack([rollout["x_int"][ex.instance_idx] for ex in explore_examples])
            y_int_true_b = torch.stack([rollout["y_int_true"][ex.instance_idx] for ex in explore_examples])
            x_ctx_b = torch.stack([ex.x_context for ex in explore_examples])
            y_ctx_b = torch.stack([ex.y_context for ex in explore_examples])
            incumbent_b = y_ctx_b.min(dim=1).values
            weights_b = improvement_weights(incumbent_b, y_int_true_b)
            with torch.no_grad():
                nll_before = self.bar_dist(self.pfn(x_ctx_b, y_ctx_b, x_int_b), y_int_true_b)
                weighted_before = (weights_b * nll_before).sum(dim=-1)  # [n_explore_examples]
            y_star_b = torch.stack([ex.y_star for ex in explore_examples])
            explore_reduction_sum = (weighted_before - y_star_b).sum()
            n_explore_examples = len(explore_examples)

        diagnostics = {
            "entropy_sum": entropy_sum, "entropy_count": len(step_examples) * self.action_head.x_dim,
            "target_distance_sum": target_distance_sum, "n_exploit_examples": n_exploit_examples,
            "explore_reduction_sum": explore_reduction_sum, "n_explore_examples": n_explore_examples,
        }
        return loss, branch_sums, diagnostics

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
        # "auto" spans the phase-in over this run's own n_rollouts; an
        # explicit int overrides that span; None disables mixing entirely.
        decay_rounds = self.n_rollouts if self.dagger_decay_rounds == "auto" else self.dagger_decay_rounds

        for rollout_idx in range(self.n_rollouts):
            build_ip_kwargs = self.build_interesting_points_kwargs if "explore" in self.branches else None

            usage_counter = None
            if decay_rounds is not None:
                # beta = P(random_policy) this rollout, decaying from 1.0
                # (round 0: pure random, matching the round-0 self-play
                # seeding design) down to dagger_beta_min (mostly
                # self-generated, floor of permanent random exploration
                # retained) as rollout_idx -> dagger_decay_rounds -- i.e.
                # PHASES IN the ActionHead's own behavior over the run.
                # random_policy is policy_a (mixed_policy_fn's usage_counter
                # "a" key) so dagger/frac_self_generated below reads "b".
                beta = max(self.dagger_beta_min, 1.0 - rollout_idx / decay_rounds)
                usage_counter = {}
                # Fresh action_head_policy_fn every rollout: it tracks its
                # own step counter via closure state, which must start at 0
                # for each new episode (see models/action_head.py).
                # sample=True keeps the self-generated fraction stochastic,
                # not a greedy beta_mode -- avoids collapsing early
                # self-play into narrow, repetitive trajectories.
                policy_fn = mixed_policy_fn(
                    random_policy,
                    action_head_policy_fn(self.action_head, self.pfn, self.n_steps, sample=True),
                    beta, usage_counter=usage_counter,
                )
            else:
                beta = 1.0  # logged for consistency; round-0-only behavior is beta=1 (always random) throughout
                policy_fn = random_policy

            rollout = rollout_episode(
                self.prior, self.n_init, self.n_steps, policy_fn=policy_fn,
                build_interesting_points_kwargs=build_ip_kwargs,
            )
            examples, extra_metrics = self._collect_examples(rollout)
            n_exploit = sum(ex.branch == "exploit" for ex in examples)
            n_explore = sum(ex.branch == "explore" for ex in examples)

            by_step: dict[int, list[ImitationExample]] = {}
            for ex in examples:
                by_step.setdefault(ex.step, []).append(ex)

            total_loss = torch.zeros(())
            branch_totals = {"exploit": torch.zeros(()), "explore": torch.zeros(())}
            entropy_sum, entropy_count = torch.zeros(()), 0
            target_distance_sum, n_exploit_examples = torch.zeros(()), 0
            explore_reduction_sum, n_explore_examples = torch.zeros(()), 0
            for step, step_examples in by_step.items():
                step_loss, branch_sums, diag = self._step_loss(step, rollout, step_examples)
                total_loss = total_loss + step_loss
                for k, v in branch_sums.items():
                    branch_totals[k] = branch_totals[k] + v
                entropy_sum = entropy_sum + diag["entropy_sum"]
                entropy_count += diag["entropy_count"]
                target_distance_sum = target_distance_sum + diag["target_distance_sum"]
                n_exploit_examples += diag["n_exploit_examples"]
                explore_reduction_sum = explore_reduction_sum + diag["explore_reduction_sum"]
                n_explore_examples += diag["n_explore_examples"]

            if by_step:
                opt.zero_grad()
                total_loss.backward()
                grad_norm = torch.sqrt(sum(
                    p.grad.detach().square().sum() for p in self.action_head.parameters() if p.grad is not None
                ))
                opt.step()
            else:
                grad_norm = None

            if rollout_idx % self.log_every == 0 or rollout_idx == self.n_rollouts - 1:
                n_examples = max(len(examples), 1)
                metrics = {
                    "policy_nll/train": total_loss.item() / n_examples,
                    "n_examples/exploit": float(n_exploit),
                    "n_examples/explore": float(n_explore),
                    "dagger/beta": beta,  # P(random_policy) this rollout -- decays 1.0 -> dagger_beta_min
                    "policy/beta_entropy": (entropy_sum.item() / entropy_count) if entropy_count else float("nan"),
                }
                if grad_norm is not None:
                    metrics["grad_norm/action_head"] = grad_norm.item()
                if n_exploit_examples:
                    metrics["exploit/target_distance"] = target_distance_sum.item() / n_exploit_examples
                if n_explore_examples:
                    metrics["explore/weighted_nll_reduction"] = explore_reduction_sum.item() / n_explore_examples
                if usage_counter:
                    # "a" = random_policy, "b" = the ActionHead's own
                    # rollout (see the mixed_policy_fn call above) --
                    # frac_self_generated is the "b" share.
                    total_actions = usage_counter.get("a", 0) + usage_counter.get("b", 0)
                    metrics["dagger/frac_self_generated"] = (
                        usage_counter.get("b", 0) / total_actions if total_actions else float("nan")
                    )
                metrics.update(extra_metrics)
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
