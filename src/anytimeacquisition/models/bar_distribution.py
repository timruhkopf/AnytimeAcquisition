"""Bar (Riemann) distribution output head — ported from PFNs4BO's
`pfns4bo/bar_distribution.py` (Müller et al., ICML 2023), trimmed to what
M2 needs (NLL, mean/median/variance, mode) plus `entropy()`, which the
original doesn't have but M5's explore-branch search needs (closed-form,
no Monte Carlo — see the design doc). Dropped PFNs4BO's smoothing,
mean-prediction-loss, and EI/PI/UCB machinery — not needed here, EI/PI/UCB
belongs to M6's classical baselines instead, not the PFN's own output head.

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
