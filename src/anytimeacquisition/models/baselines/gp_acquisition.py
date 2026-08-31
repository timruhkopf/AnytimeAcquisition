"""GP + classical acquisition function baselines (M6) — BoTorch, per
`docs/OPEN_QUESTIONS.md` #3 (resolved 2026-08-28, user call). These are the
comparison points M5's EXIT policy gets evaluated against
(`docs/ROADMAP.md` Phase 6): does learning the acquisition function
end-to-end actually beat classical EI/PI/entropy-search on the same
instances, not just "does it run."

Sign convention: this project minimizes throughout (`y` from `BNNPrior`,
`metrics/inc_auc.py`, `search/exploit.py`, ...), but BoTorch's acquisition
functions are built to *maximize*. `fit_gp` fits on `-y_context`
internally so "improvement"/"best_f" mean the same thing BoTorch expects,
and `gp_acquisition_policy` never leaks that sign flip to its caller (its
inputs and outputs are both in this project's normal minimize/`[0,1]`
convention, matching `trainer.exit_rollout.random_policy`'s signature
exactly, so any of these can be dropped straight into `rollout_episode`'s
`policy_fn`).

Fits one independent GP per batch instance (Python loop over `B`), not a
batched multi-task GP — simpler and more robust for a first baseline, and
batch sizes here are small (a handful of parallel rollout instances, not a
training-scale batch). BoTorch/GPyTorch strongly prefer `float64` for the
GP fit and acquisition optimization (numerical stability of the Cholesky
factorization); inputs/outputs are cast back to `float32` at the boundary
to match the rest of this codebase.
"""
import torch
from botorch.acquisition import LogExpectedImprovement, ProbabilityOfImprovement
from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

ACQUISITIONS = ("EI", "PI", "ES")


def fit_gp(x_context: torch.Tensor, y_context: torch.Tensor) -> SingleTaskGP:
    """x_context: [Nt, x_dim]  y_context: [Nt] (this project's minimize
    convention) -> a SingleTaskGP fit on `-y_context` (so BoTorch's
    maximize-by-default acquisition functions target the right direction),
    in float64. `Standardize` outcome transform, standard BoTorch practice
    for numerical stability -- fit via `fit_gpytorch_mll` (type-II MLE,
    BoTorch's own default recipe, not a from-scratch training loop)."""
    train_x = x_context.to(torch.float64)
    train_y = (-y_context).to(torch.float64).unsqueeze(-1)  # [Nt, 1], maximize convention
    gp = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=1))
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    return gp


def _build_acquisition(
    gp: SingleTaskGP, acquisition: str, best_f: torch.Tensor, x_dim: int, mes_candidate_set_size: int,
):
    if acquisition == "EI":
        return LogExpectedImprovement(gp, best_f=best_f)
    if acquisition == "PI":
        return ProbabilityOfImprovement(gp, best_f=best_f)
    if acquisition == "ES":
        # Max-value Entropy Search needs a discretized candidate set to
        # estimate the distribution of the (unknown) maximum -- sampled
        # uniformly over the domain, matching this project's [0,1]^x_dim
        # convention (BNNPrior's own domain).
        candidate_set = torch.rand(mes_candidate_set_size, x_dim, dtype=torch.float64)
        return qMaxValueEntropy(gp, candidate_set=candidate_set)
    raise ValueError(f"unknown acquisition {acquisition!r}, expected one of {ACQUISITIONS}")


def gp_acquisition_policy(
    x_context: torch.Tensor, y_context: torch.Tensor, x_dim: int,
    acquisition: str = "EI", num_restarts: int = 10, raw_samples: int = 256, mes_candidate_set_size: int = 1000,
) -> torch.Tensor:
    """Same signature/contract as `trainer.exit_rollout.random_policy` --
    drop-in `policy_fn` for `rollout_episode`. x_context: [B, Nt, x_dim]
    y_context: [B, Nt] -> x_next: [B, x_dim].

    Fits an independent GP per batch instance and optimizes `acquisition`
    (`"EI"`/`"PI"`/`"ES"`) over `[0, 1]^x_dim` via BoTorch's
    `optimize_acqf` (multistart L-BFGS-B from `raw_samples` initial
    points, keeping `num_restarts` of them) -- same general shape as
    `search.exploit.exploit_search`'s multistart GD, but on a GP surrogate
    fit from the observed context only, not the true privileged surface;
    this is the whole point of a baseline.
    """
    B = x_context.shape[0]
    bounds = torch.stack([torch.zeros(x_dim, dtype=torch.float64), torch.ones(x_dim, dtype=torch.float64)])

    x_next = torch.empty(B, x_dim)
    for b in range(B):
        gp = fit_gp(x_context[b], y_context[b])
        best_f = (-y_context[b]).max().to(torch.float64)
        acqf = _build_acquisition(gp, acquisition, best_f, x_dim, mes_candidate_set_size)
        candidate, _ = optimize_acqf(acqf, bounds=bounds, q=1, num_restarts=num_restarts, raw_samples=raw_samples)
        x_next[b] = candidate.squeeze(0).to(torch.float32)
    return x_next


if __name__ == "__main__":
    """M6.md's required sanity check: each baseline should beat random
    search on a known easy instance -- compares log-incumbent AUC
    (`metrics/inc_auc.py`, already built for exactly this) for every
    acquisition here against `trainer.exit_rollout.random_policy`, on the
    identical rollout setup (same BNNPrior instances, same seed, same
    `rollout_episode` machinery -- any of these policies is a drop-in
    `policy_fn`)."""
    from functools import partial

    from anytimeacquisition.metrics.inc_auc import log_incumbent_auc
    from anytimeacquisition.priors.bnn import BNNPrior
    from anytimeacquisition.trainer.exit_rollout import random_policy, rollout_episode

    torch.manual_seed(0)
    x_dim = 2
    n_init, n_steps, batch_size = 3, 10, 4

    prior = BNNPrior(batch_size=batch_size, x_dim=x_dim, seed=0)
    random_rollout = rollout_episode(prior, n_init=n_init, n_steps=n_steps, policy_fn=random_policy)
    random_auc = log_incumbent_auc(random_rollout["y_context"]).mean().item()
    print(f"random_policy      mean log-incumbent AUC (lower is better): {random_auc:.4f}")

    for acquisition in ACQUISITIONS:
        torch.manual_seed(0)
        prior = BNNPrior(batch_size=batch_size, x_dim=x_dim, seed=0)
        policy_fn = partial(gp_acquisition_policy, acquisition=acquisition, num_restarts=5, raw_samples=64)
        rollout = rollout_episode(prior, n_init=n_init, n_steps=n_steps, policy_fn=policy_fn)
        auc = log_incumbent_auc(rollout["y_context"]).mean().item()
        beats_random = "beats" if auc < random_auc else "does NOT beat"
        print(f"gp_acquisition({acquisition})  mean log-incumbent AUC: {auc:.4f}  ({beats_random} random)")
