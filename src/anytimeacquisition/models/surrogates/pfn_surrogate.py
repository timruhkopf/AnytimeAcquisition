"""Frozen-PFN surrogate wrapper (docs/ROADMAP.md §M2) — the eval-time API
the rest of the project (M4's Q-head features, M7's baseline comparisons)
reads a trained PFN checkpoint through, rather than every caller poking at
`PFN`/`BarDistribution` directly. Wraps the existing `models/pfn.py`
architecture and `models/bar_distribution.py` output head; doesn't
reimplement either.

Reuses `models/baselines/pfn_acquisition.py`'s `expected_improvement`
(already implemented and cross-checked against Monte Carlo there) rather
than duplicating it. `probability_of_improvement` is new here — the bar
distribution's own docstring notes EI/PI/UCB were deliberately dropped from
`BarDistribution` itself ("belongs to M6's [now M7's] classical baselines
instead, not the PFN's own output head"), so this lives alongside
`expected_improvement`, not on `BarDistribution`.

**Intra-step KV caching only (§2.11), by construction, not an explicit
cache object:** `PFN`'s train-side self-attention never depends on the test
tokens (`models/pfn.py`'s `PFNBlock`), so one batched `pfn(...)` call over
an entire candidate pool already computes the context representation once
and reuses it for every candidate — the same trick `pfn_ei_argmax` already
relies on. `predict()` below is written to always be called this way (one
call per whole candidate batch, never one call per candidate) — that's the
whole caching story; there is deliberately no cross-step cache (§2.11: the
context changes every step under bidirectional attention, so nothing from
a previous step's forward pass is reusable at the next one).
"""
import torch

from anytimeacquisition.models.bar_distribution import BarDistribution
from anytimeacquisition.models.baselines.pfn_acquisition import expected_improvement
from anytimeacquisition.models.pfn import PFN


def probability_of_improvement(bar_dist: BarDistribution, logits: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
    """Closed-form `P(Y < threshold)` under the piecewise-uniform bar
    density (this project minimizes throughout) — the CDF at `threshold`,
    same clamped-bucket trick as `expected_improvement`. logits: [...,
    n_bins]  threshold: broadcastable to `logits.shape[:-1]` -> [...]."""
    lo, hi = bar_dist.borders[:-1], bar_dist.borders[1:]
    thr = threshold.unsqueeze(-1)
    clamped = thr.clamp(lo, hi)
    bucket_cdf = (clamped - lo) / bar_dist.bucket_widths
    p = torch.softmax(logits, -1)
    return (p * bucket_cdf).sum(-1)


class PFNSurrogate:
    """no_grad, bf16-autocast forward pass by default (`use_bf16=True`) —
    kept off the fp32 default deliberately (this is the eval-time path
    M4+'s Q-head training loop will call at scale), but every derived
    scalar (EI/PI/mean/std) upcasts the raw logits to fp32 immediately
    after the forward pass, before any BarDistribution-style op runs on
    them (softmax/log-softmax near-ties in bf16 are numerically unreliable,
    and `BarDistribution`'s own ops were written/tested in fp32) — verify
    `torch.autocast`'s bf16 path is actually faster on whatever device
    you're running this on (CPU bf16 autocast works but isn't necessarily
    faster; confirm on `ulysses`'s actual GPU before assuming a speedup)."""

    def __init__(self, pfn: PFN, bar_dist: BarDistribution, use_bf16: bool = True):
        self.pfn = pfn.eval()
        self.bar_dist = bar_dist
        self.use_bf16 = use_bf16

    @torch.no_grad()
    def predict(self, x_context: torch.Tensor, y_context: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        """x_context: [B,Nt,d]  y_context: [B,Nt]  candidates: [B,C,d] (one
        whole candidate pool per call — see module docstring on caching)
        -> logits [B,C,n_bins], fp32."""
        n_features = x_context.new_full((x_context.shape[0],), x_context.shape[-1])
        device_type = x_context.device.type
        if self.use_bf16:
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits = self.pfn(x_context, y_context, candidates, n_features=n_features)
        else:
            logits = self.pfn(x_context, y_context, candidates, n_features=n_features)
        return logits.float()

    def expected_improvement(
        self, x_context: torch.Tensor, y_context: torch.Tensor, candidates: torch.Tensor,
        incumbent: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.predict(x_context, y_context, candidates)
        if incumbent is None:
            incumbent = y_context.min(dim=1).values
        return expected_improvement(self.bar_dist, logits, incumbent.unsqueeze(-1)), logits

    def probability_of_improvement(
        self, x_context: torch.Tensor, y_context: torch.Tensor, candidates: torch.Tensor,
        threshold: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.predict(x_context, y_context, candidates)
        if threshold is None:
            threshold = y_context.min(dim=1).values
        return probability_of_improvement(self.bar_dist, logits, threshold.unsqueeze(-1)), logits

    def mean_std(
        self, x_context: torch.Tensor, y_context: torch.Tensor, candidates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.predict(x_context, y_context, candidates)
        mean = self.bar_dist.mean(logits)
        std = self.bar_dist.variance(logits).clamp_min(0).sqrt()
        return mean, std


def pfn_surrogate_ei_policy(
    x_context: torch.Tensor, y_context: torch.Tensor, x_dim: int,
    surrogate: PFNSurrogate, n_candidates: int = 256, seed: int | None = None,
) -> torch.Tensor:
    """Same signature/contract as `metrics.rollout.random_policy` — drop-in
    `policy_fn` for `rollout_episode`. Unlike `pfn_acquisition.py`'s
    `pfn_acquisition_policy` (dense linspace grid, x_dim=1 only), this
    proposes a fresh Sobol candidate pool each call, so it works at any
    x_dim the underlying checkpoint supports."""
    B = x_context.shape[0]
    sob = torch.quasirandom.SobolEngine(dimension=x_dim, scramble=True, seed=seed)
    candidates = sob.draw(n_candidates).to(x_context.device).unsqueeze(0).expand(B, -1, -1)
    ei, _ = surrogate.expected_improvement(x_context, y_context, candidates)
    best_idx = ei.argmax(dim=1)
    return candidates[torch.arange(B), best_idx]


if __name__ == "__main__":
    """Loads pfn_variable_xdim_smoke.pt, runs its EI policy against a fresh
    2-D BNN draw, and reports the resulting log-incumbent AUC alongside a
    random baseline -- a fast sanity check, not the full M2 comparison
    (see notebooks/m2_pfn_surrogate_vs_ei.ipynb for that)."""
    from functools import partial

    from anytimeacquisition.metrics.inc_auc import log_incumbent_auc
    from anytimeacquisition.metrics.rollout import random_policy, rollout_episode
    from anytimeacquisition.pipelines.train_pfn import load_pfn_checkpoint
    from anytimeacquisition.priors.bnn import BNNPrior
    from anytimeacquisition.utils.paths import CHECKPOINT_DIR

    checkpoint_path = CHECKPOINT_DIR / "pfn_variable_xdim_smoke.pt"
    pfn, bar_dist, ckpt = load_pfn_checkpoint(checkpoint_path)
    print(f"loaded {checkpoint_path.name}, config={ckpt['config']}")
    surrogate = PFNSurrogate(pfn, bar_dist, use_bf16=False)  # bf16 autocast on CPU: correct but no speedup

    torch.manual_seed(0)
    x_dim, batch_size, n_init, n_steps = 2, 8, 3, 15

    prior = BNNPrior(batch_size=batch_size, x_dim=x_dim, seed=0)
    pfn_rollout = rollout_episode(
        prior, n_init=n_init, n_steps=n_steps,
        policy_fn=partial(pfn_surrogate_ei_policy, surrogate=surrogate, n_candidates=256, seed=1),
    )
    pfn_auc = log_incumbent_auc(pfn_rollout["y_context"]).mean().item()

    prior = BNNPrior(batch_size=batch_size, x_dim=x_dim, seed=0)
    random_rollout = rollout_episode(prior, n_init=n_init, n_steps=n_steps, policy_fn=random_policy)
    random_auc = log_incumbent_auc(random_rollout["y_context"]).mean().item()

    print(f"random_policy         mean log-incumbent AUC (lower is better): {random_auc:.4f}")
    print(f"pfn_surrogate_ei_policy mean log-incumbent AUC:                 {pfn_auc:.4f}")
