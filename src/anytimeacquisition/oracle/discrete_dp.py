"""Exact-DP oracle harness (docs/ROADMAP.md §M3) — "the only place
correctness can be verified rather than merely measured." Everything from
M4 onward trains a network and *measures* whether it looks reasonable (a
smooth loss curve, beating a baseline); none of that proves the training
machinery itself is correct. This module computes the actual, exact `Q*`
by backward dynamic programming, at a small enough scale (`d<=2`, a `k`-point
grid) to be tractable, so a learned `Q` can be checked against ground truth
instead of only against its own loss curve.

**What "the belief" means here.** §0's formal setting defines `p(y|x,D_t)`
as the posterior predictive — and the frozen PFN (`models/surrogates/pfn_surrogate.py`,
M2) is exactly what makes that computable. So this DP plans against the
**PFN's own predictive distribution**, not raw ground-truth function values:
it computes the same `Q*` the Q-head (M4/M5) is trying to approximate, just
exactly (backward induction) instead of via approximate branch-and-replay
rollouts. This is also why M3 depends on M2 — it needs a trusted frozen
surrogate to supply transitions.

**Why this doesn't stay small.** Exact backward induction needs a finite
state space, but observed outcomes are continuous. The fix here is the
standard one: discretize the PFN's own predictive into `n_outcome_bins`
representative (value, probability) pairs (`discretize_outcomes` below) by
pooling contiguous groups of its native (e.g. 64) bins — independent of
`k`, so it's a separate tractability knob. Every additional remaining step
multiplies the number of states to evaluate by `(k^d * n_outcome_bins)`,
so cost is genuinely exponential in `budget`, not just slow: at the
defaults below (`n_outcome_bins=6`), `d=1` (16 actions) stays tractable to
`budget` ~3-4; `d=2` (256 actions) only to `budget<=2`. This is inherent to
exact enumeration, not an implementation gap — reduce `n_outcome_bins`
and/or `budget` before going further, rather than expecting this to scale.

**Implementation shape.** `_solve_batched` is recursion-shaped (one call
per remaining-budget level, base case at `budget==1`) but every state at a
given level is processed as *one batch*, not one call per state — a single
`PFNSurrogate.predict()` call scores the whole frontier x the whole action
grid x the whole outcome-bin set at once. The frontier is chunked
(`_CHUNK_SIZE`) before hitting the PFN so memory stays bounded even once
the frontier itself is large. No cross-branch memoization (states from
different top-level actions are already-batched together, and the
benefit would be marginal — see the module's own cost analysis above; add
if a real need shows up)."""
import torch

from anytimeacquisition.models.surrogates.pfn_surrogate import PFNSurrogate
from anytimeacquisition.reward.tail_quantile_reward import g_reward_minimize

_CHUNK_SIZE = 4096  # cap on rows sent through PFNSurrogate.predict() in one call, for memory safety


def build_action_grid(d: int, k: int = 16) -> torch.Tensor:
    """Regular (not Sobol) grid over `[0,1]^d`, `k` points per dimension --
    a fixed, enumerable, deterministic action set, required for DP (Sobol's
    randomized low-discrepancy sampling is the right tool for M0/M1/M2's
    quadrature-style estimation, not for a tabular action space here).
    -> [k**d, d]."""
    axes = [torch.linspace(0.0, 1.0, k) for _ in range(d)]
    mesh = torch.meshgrid(*axes, indexing="ij")
    return torch.stack([m.reshape(-1) for m in mesh], dim=-1)


def discretize_outcomes(bar_dist, logits: torch.Tensor, n_outcome_bins: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Coarsens the bar distribution's native (fine) bins into
    `n_outcome_bins` representative (value, probability) pairs by pooling
    contiguous groups of fine bins -- keeps the DP's branching factor small
    regardless of the PFN's own bin resolution. logits: [..., n_bins]
    -> values [n_outcome_bins] (bin-count-weighted mean of each group's
    bucket midpoints, independent of `logits`), probs [..., n_outcome_bins]
    (probability-mass-summed per group, from `logits`)."""
    n_bins = bar_dist.num_bars
    assert n_outcome_bins <= n_bins, f"n_outcome_bins ({n_outcome_bins}) must be <= the bar distribution's own n_bins ({n_bins})"
    bucket_means = bar_dist.borders[:-1] + bar_dist.bucket_widths / 2  # [n_bins]
    group_ids = (torch.arange(n_bins) * n_outcome_bins) // n_bins  # [n_bins], in [0, n_outcome_bins)
    group_onehot = torch.nn.functional.one_hot(group_ids, n_outcome_bins).float()  # [n_bins, n_outcome_bins]

    counts = group_onehot.sum(0)  # [n_outcome_bins]
    values = (bucket_means @ group_onehot) / counts  # [n_outcome_bins]

    p = torch.softmax(logits, -1)  # [..., n_bins]
    probs = p @ group_onehot  # [..., n_outcome_bins]
    return values, probs


class DiscreteDPOracle:
    """Exact `Q*`/optimal-policy solver for one BNN-prior instance, planning
    against one frozen `PFNSurrogate`'s predictive distribution.

    `y_sorted`: this instance's own dense reference sample (`build_ecdf`,
    docs/ROADMAP.md §M1) -- the `g`-reward is computed relative to it, same
    convention as every other reward computation in this project."""

    def __init__(
        self, surrogate: PFNSurrogate, y_sorted: torch.Tensor, grid: torch.Tensor,
        n_outcome_bins: int = 6, max_score: float = 4.0,
    ):
        self.surrogate = surrogate
        self.y_sorted = y_sorted  # [N], sorted ascending
        self.grid = grid  # [n_actions, d]
        self.n_outcome_bins = n_outcome_bins
        self.max_score = max_score

    def _g(self, y: torch.Tensor) -> torch.Tensor:
        """g-reward of landing at value y (minimize convention), relative
        to this instance's own reference sample -- shape-agnostic (any
        shape in, same shape out). Goes through `g_reward_minimize`, not a
        raw `percentile` + `g_from_percentile` composition -- that
        composition silently computes the reward for *maximizing* y
        instead (see `g_reward_minimize`'s own docstring)."""
        shape = y.shape
        return g_reward_minimize(self.y_sorted.unsqueeze(0), y.reshape(1, -1), max_score=self.max_score).reshape(shape)

    def _predict_chunked(self, x_contexts: torch.Tensor, y_contexts: torch.Tensor) -> torch.Tensor:
        """PFNSurrogate.predict() over the whole action grid, batched over
        x_contexts/y_contexts in chunks of `_CHUNK_SIZE` rows -- memory
        safety valve as the DP's frontier grows with recursion depth."""
        n_actions = self.grid.shape[0]
        Bf = x_contexts.shape[0]
        chunks = []
        for start in range(0, Bf, _CHUNK_SIZE):
            end = min(start + _CHUNK_SIZE, Bf)
            candidates = self.grid.unsqueeze(0).expand(end - start, -1, -1)
            chunks.append(self.surrogate.predict(x_contexts[start:end], y_contexts[start:end], candidates))
        return torch.cat(chunks, dim=0)  # [Bf, n_actions, n_bins]

    def _solve_batched(self, x_contexts: torch.Tensor, y_contexts: torch.Tensor, budget: int) -> tuple[torch.Tensor, torch.Tensor]:
        """x_contexts: [Bf,Nt,d]  y_contexts: [Bf,Nt] -> (Q [Bf,n_actions], V [Bf]).
        One call per remaining-budget LEVEL, not per state -- see module
        docstring."""
        Bf, Nt, d = x_contexts.shape
        n_actions = self.grid.shape[0]

        logits = self._predict_chunked(x_contexts, y_contexts)  # [Bf, n_actions, n_bins]
        values, probs = discretize_outcomes(self.surrogate.bar_dist, logits, self.n_outcome_bins)  # values [O], probs [Bf,n_actions,O]

        incumbent = y_contexts.min(dim=1).values  # [Bf], minimize convention
        # the resulting incumbent after landing at a hypothetical outcome value
        # doesn't depend on which action produced it, only the value itself
        next_incumbent = torch.minimum(incumbent.view(Bf, 1, 1), values.view(1, 1, -1)).expand(Bf, n_actions, -1)
        step_reward = self._g(next_incumbent)  # [Bf, n_actions, O]

        if budget == 1:
            Q = (probs * step_reward).sum(-1)  # [Bf, n_actions]
        else:
            O = self.n_outcome_bins
            child_x = torch.cat([
                x_contexts.view(Bf, 1, 1, Nt, d).expand(Bf, n_actions, O, Nt, d),
                self.grid.view(1, n_actions, 1, 1, d).expand(Bf, n_actions, O, 1, d),
            ], dim=3).reshape(Bf * n_actions * O, Nt + 1, d)
            child_y = torch.cat([
                y_contexts.view(Bf, 1, 1, Nt).expand(Bf, n_actions, O, Nt),
                values.view(1, 1, O, 1).expand(Bf, n_actions, O, 1),
            ], dim=3).reshape(Bf * n_actions * O, Nt + 1)

            _, V_children = self._solve_batched(child_x, child_y, budget - 1)
            V_children = V_children.reshape(Bf, n_actions, O)

            # Ḡ_t = (Σ g_s)/m recursion (§1.11): this step's reward plus the
            # (already-normalized) continuation's own share, both weighted
            # by remaining budget so the whole sum stays divided by budget.
            Q = (probs * ((step_reward + (budget - 1) * V_children) / budget)).sum(-1)

        V = Q.max(dim=-1).values  # [Bf]
        return Q, V

    def solve(self, x_context: torch.Tensor, y_context: torch.Tensor, budget: int) -> tuple[torch.Tensor, torch.Tensor]:
        """x_context: [Nt,d]  y_context: [Nt] (one instance, no batch dim)
        -> (Q* [n_actions], V* scalar). `budget`: number of remaining
        decisions to plan over exactly (see module docstring for tractable
        ranges)."""
        assert budget >= 1
        Q, V = self._solve_batched(x_context.unsqueeze(0), y_context.unsqueeze(0), budget)
        return Q[0], V[0]


def score_candidate_q(
    oracle: DiscreteDPOracle, candidate_q: torch.Tensor, x_context: torch.Tensor, y_context: torch.Tensor, budget: int,
) -> dict:
    """Compares a candidate Q (e.g. a learned Q-head's own scores over the
    same action grid, or any other acquisition function's scores used as a
    stand-in before M4 exists) against the DP-exact Q* -- the exit
    criterion's required scoring: rank correlation and regret.
    candidate_q: [n_actions], same ordering as oracle.grid.

    -> {"q_star": [n_actions], "spearman_rho": float,
        "regret": float (V* - Q*[candidate's own argmax], >= 0, 0 = candidate
        picked the truly optimal action)}."""
    q_star, v_star = oracle.solve(x_context, y_context, budget)
    assert candidate_q.shape == q_star.shape

    rank_q_star = q_star.argsort().argsort().float()
    rank_candidate = candidate_q.argsort().argsort().float()
    rho = torch.corrcoef(torch.stack([rank_q_star, rank_candidate]))[0, 1].item()

    candidate_action = candidate_q.argmax()
    regret = (v_star - q_star[candidate_action]).item()

    return {"q_star": q_star, "spearman_rho": rho, "regret": regret}


if __name__ == "__main__":
    """Solves a tiny 1-D case exactly, then scores the PFN's own EI ranking
    against it as a stand-in "candidate Q" (M4's real Q-head doesn't exist
    yet) -- demonstrates the harness end to end. See
    notebooks/m3_discrete_dp_oracle.ipynb for the full milestone writeup."""
    from anytimeacquisition.pipelines.train_pfn import load_pfn_checkpoint
    from anytimeacquisition.priors.bnn import BNNPrior
    from anytimeacquisition.reward.tail_quantile_reward import build_ecdf
    from anytimeacquisition.utils.paths import CHECKPOINT_DIR

    torch.manual_seed(0)
    pfn, bar_dist, _ = load_pfn_checkpoint(CHECKPOINT_DIR / "pfn_variable_xdim_smoke.pt")
    surrogate = PFNSurrogate(pfn, bar_dist, use_bf16=False)

    x_dim, budget = 1, 3
    prior = BNNPrior(batch_size=1, x_dim=x_dim, seed=0)
    y_sorted = build_ecdf(prior, n_samples=100_000, seed=1)[0]
    x_context, y_context, _, _ = prior.sample_episode(n_train=3, n_test=0)

    grid = build_action_grid(d=x_dim, k=16)
    oracle = DiscreteDPOracle(surrogate, y_sorted, grid, n_outcome_bins=6)
    q_star, v_star = oracle.solve(x_context[0], y_context[0], budget=budget)
    print(f"Q* range: [{q_star.min().item():.4f}, {q_star.max().item():.4f}]  V*={v_star.item():.4f}")
    print(f"DP-optimal action: x={grid[q_star.argmax()].item():.4f}")

    ei, _ = surrogate.expected_improvement(x_context, y_context, grid.unsqueeze(0))
    result = score_candidate_q(oracle, ei[0], x_context[0], y_context[0], budget=budget)
    print(f"EI-as-candidate-Q vs DP-exact Q*: spearman_rho={result['spearman_rho']:.4f}  regret={result['regret']:.4f}")
