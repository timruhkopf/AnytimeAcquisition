"""Bar (Riemann) distribution output head — ported from PFNs4BO's
`pfns4bo/bar_distribution.py` (Müller et al., ICML 2023; vendored reference
at `archive/src/utils/bar_distribution.py`). NLL, mean/median/mode/variance,
`quantile`, `ucb`, closed-form `ei`/`pi` (2026-09-08: moved onto this class
from `models/baselines/pfn_acquisition.py`/`models/surrogates/pfn_surrogate.py`
— an earlier version of this docstring said EI/PI/UCB were deliberately
dropped here ("belongs to the classical baselines instead"); reversed by
user request, they live here now, matching PFNs4BO's own layout) plus
`entropy()`, which the original doesn't have but M5's explore-branch search
needs (closed-form, no Monte Carlo — see the design doc). `ei`/`pi`/`ucb`
are all mirrored for this project's minimize convention — the reference
assumes maximization throughout; see each method's own docstring for the
exact mirroring. Not ported: `smoothing`/`mean_prediction_logits` (both
`forward()`-only training-loss features, unused so far — see
`archive`'s own `forward()` for the shape if ever needed) and
`FullSupportBarDistribution`'s half-normal tail extrapolation (this
project uses fixed, bounded borders instead — see below).

Fixed `[0, 1]` borders (bounded, not PFNs4BO's FullSupportBarDistribution)
— matches M1's ECDF-normalized-to-[0,1] prior output. See
`docs/OPEN_QUESTIONS.md` #8 for the full reasoning and the full-support
alternative this deliberately isn't.
"""
import torch
from torch import nn


def uniform_bin_borders(n_bins: int, lo: float = 0.0, hi: float = 1.0) -> torch.Tensor:
    return torch.linspace(lo, hi, n_bins + 1)


class BarDistribution(nn.Module):
    def __init__(self, borders: torch.Tensor):
        """borders: 1D, sorted, ascending — bin edges over the support."""
        super().__init__()
        assert borders.dim() == 1 and (borders[1:] >= borders[:-1]).all(), "borders must be sorted"
        self.register_buffer("borders", borders)
        self.register_buffer("bucket_widths", borders[1:] - borders[:-1])
        self.num_bars = len(borders) - 1

    def map_to_bucket_idx(self, y: torch.Tensor) -> torch.Tensor:
        idx = torch.searchsorted(self.borders, y.contiguous()) - 1
        idx[y == self.borders[0]] = 0
        idx[y == self.borders[-1]] = self.num_bars - 1
        return idx.clamp(0, self.num_bars - 1)

    def compute_scaled_log_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """log p(y) of the piecewise-constant density, per bucket."""
        bucket_log_probs = torch.log_softmax(logits, -1)
        return bucket_log_probs - torch.log(self.bucket_widths)

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """NLL loss. logits: [..., num_bars], y: [...] (same leading shape)."""
        idx = self.map_to_bucket_idx(y)
        scaled_log_probs = self.compute_scaled_log_probs(logits)
        return -scaled_log_probs.gather(-1, idx[..., None]).squeeze(-1)

    def entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Closed-form differential entropy H = -sum_i p_i * log(p_i/width_i),
        no Monte Carlo — this is what makes the explore branch's
        entropy-gradient search (M5) a cheap gradient step."""
        p = torch.softmax(logits, -1)
        scaled_log_probs = self.compute_scaled_log_probs(logits)
        return -(p * scaled_log_probs).sum(-1)

    def mean(self, logits: torch.Tensor) -> torch.Tensor:
        bucket_means = self.borders[:-1] + self.bucket_widths / 2
        return torch.softmax(logits, -1) @ bucket_means

    def mean_of_square(self, logits: torch.Tensor) -> torch.Tensor:
        lo, hi = self.borders[:-1], self.borders[1:]
        bucket_mean_sq = (lo.square() + hi.square() + lo * hi) / 3.0
        return torch.softmax(logits, -1) @ bucket_mean_sq

    def variance(self, logits: torch.Tensor) -> torch.Tensor:
        return self.mean_of_square(logits) - self.mean(logits).square()

    def icdf(self, logits: torch.Tensor, left_prob: float) -> torch.Tensor:
        probs = torch.softmax(logits, -1)
        cumprobs = torch.cumsum(probs, -1)
        target = left_prob * torch.ones(*cumprobs.shape[:-1], 1, device=logits.device)
        idx = torch.searchsorted(cumprobs, target).squeeze(-1).clamp(0, self.num_bars - 1)
        cumprobs_padded = torch.cat([torch.zeros(*cumprobs.shape[:-1], 1, device=logits.device), cumprobs], -1)
        rest_prob = left_prob - cumprobs_padded.gather(-1, idx[..., None]).squeeze(-1)
        lo, hi = self.borders[idx], self.borders[idx + 1]
        return lo + (hi - lo) * rest_prob / probs.gather(-1, idx[..., None]).squeeze(-1)

    def median(self, logits: torch.Tensor) -> torch.Tensor:
        return self.icdf(logits, 0.5)

    def mode(self, logits: torch.Tensor) -> torch.Tensor:
        bucket_means = self.borders[:-1] + self.bucket_widths / 2
        return bucket_means[logits.argmax(-1)]

    def quantile(self, logits: torch.Tensor, center_prob: float = 0.682) -> torch.Tensor:
        """[lo, hi] interval containing `center_prob` probability mass
        around the median. Ported from PFNs4BO's own `BarDistribution.quantile`
        (`archive/src/utils/bar_distribution.py`) — direction-agnostic (an
        interval, not an improvement), no min/max mirroring needed.
        -> [..., 2] (logits.shape[:-1] + [2])."""
        side_prob = (1.0 - center_prob) / 2
        return torch.stack((self.icdf(logits, side_prob), self.icdf(logits, 1.0 - side_prob)), dim=-1)

    def ucb(self, logits: torch.Tensor, rest_prob: float = (1 - 0.682) / 2) -> torch.Tensor:
        """Optimistic-for-minimization confidence bound: the `rest_prob`
        lower quantile (the project minimizes throughout — this is the
        mirror of PFNs4BO's own `ucb`, which for *maximization* takes the
        `1 - rest_prob` *upper* quantile via the same `icdf`; swapping which
        tail is "optimistic" is the same mirroring `ei`/`pi` below use).
        `rest_prob=(1-0.682)/2` (the default, matching the reference) is the
        amount of density beyond the confidence bound being ignored,
        equivalent to GP-UCB/LCB with `beta=1`."""
        return self.icdf(logits, rest_prob)

    def ei(self, logits: torch.Tensor, best_f: torch.Tensor) -> torch.Tensor:
        """Closed-form `E[max(best_f - Y, 0)]` under the piecewise-uniform
        bar density (this project minimizes) — ported from PFNs4BO's own
        `BarDistribution.ei` (`archive/src/utils/bar_distribution.py`,
        which assumes maximization: `E[max(Y - best_f, 0)]`), algebraically
        mirrored for minimization (swap which border plays the "active"
        role: the reference's `borders[1:]` becomes `borders[:-1]` here).
        Cross-checked against a Monte Carlo estimate in
        `tests/test_bar_distribution.py` — not trusted on the algebra
        alone. logits: [..., n_bins]  best_f: broadcastable to
        `logits.shape[:-1]` -> [...]."""
        lo, hi = self.borders[:-1], self.borders[1:]
        inc = best_f.unsqueeze(-1)  # [..., 1], broadcasts against the n_bins axis
        clamped = inc.clamp(lo, hi)  # [..., n_bins]
        bucket_contributions = (inc * (clamped - lo) - (clamped**2 - lo**2) / 2) / self.bucket_widths
        p = torch.softmax(logits, -1)
        return (p * bucket_contributions).sum(-1)

    def pi(self, logits: torch.Tensor, best_f: torch.Tensor) -> torch.Tensor:
        """Closed-form `P(Y < best_f)` under the piecewise-uniform bar
        density (this project minimizes) — ported from PFNs4BO's own
        `BarDistribution.pi` (assumes maximization: `P(Y > best_f)`),
        mirrored the same way `ei` above is. Same clamped-bucket trick.
        logits: [..., n_bins]  best_f: broadcastable to
        `logits.shape[:-1]` -> [...]."""
        lo, hi = self.borders[:-1], self.borders[1:]
        thr = best_f.unsqueeze(-1)
        clamped = thr.clamp(lo, hi)
        bucket_cdf = (clamped - lo) / self.bucket_widths
        p = torch.softmax(logits, -1)
        return (p * bucket_cdf).sum(-1)


if __name__ == "__main__":
    torch.manual_seed(0)
    bd = BarDistribution(uniform_bin_borders(n_bins=64))

    # Analytic check: uniform logits -> uniform density over [0,1] ->
    # entropy should match the analytic differential entropy of U(0,1), 0.0.
    uniform_logits = torch.zeros(5, 64)
    h = bd.entropy(uniform_logits)
    print("entropy of a uniform bar distribution (expect ~0.0):", h.tolist())

    # A confident (near-delta) distribution should have much lower (very
    # negative) entropy than the uniform case.
    confident_logits = torch.full((1, 64), -10.0)
    confident_logits[0, 5] = 20.0
    print("entropy of a confident bar distribution (expect << 0):", bd.entropy(confident_logits).item())

    y = torch.rand(5)
    nll = bd(uniform_logits, y)
    print("NLL under uniform logits (expect ~0, since density=1 everywhere):", nll.tolist())

    print("mean under uniform logits (expect ~0.5):", bd.mean(uniform_logits).tolist())

    # ei/pi/ucb/quantile sanity, under a confident distribution centered
    # near bucket 5's midpoint (~0.5/64*5+... roughly 0.086) -- best_f well
    # above that point should have EI close to (best_f - mode) and PI close
    # to 1 (the distribution's mass is almost certainly below best_f);
    # best_f well below it should have both close to 0.
    mode_val = bd.mode(confident_logits).item()
    print(f"mode of the confident distribution: {mode_val:.4f}")
    print("EI at best_f=0.9 (expect ~0.9 - mode, large):", bd.ei(confident_logits, torch.tensor([0.9])).item())
    print("EI at best_f=0.01 (expect ~0.0, mode is above it):", bd.ei(confident_logits, torch.tensor([0.01])).item())
    print("PI at best_f=0.9 (expect ~1.0):", bd.pi(confident_logits, torch.tensor([0.9])).item())
    print("PI at best_f=0.01 (expect ~0.0):", bd.pi(confident_logits, torch.tensor([0.01])).item())
    print("68.2% quantile interval under uniform logits (expect ~[0.159, 0.841]):",
          bd.quantile(uniform_logits[:1]).tolist())
    print("ucb (optimistic-for-min lower quantile) under uniform logits (expect ~0.159):",
          bd.ucb(uniform_logits[:1]).tolist())
