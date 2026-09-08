"""Per-instance quantile reference table for the tail-quantile reward (M0).

Given a batch of `B` independently-drawn `BNNPrior` functions, evaluates `N`
uniform points per function via chunked, batched forward passes -- a single
`[B, N, d]` tensor at `N=1e6` is memory- and compute-prohibitive at
realistic `B`/`d` (e.g. `B=512, d=18, N=1e6` is ~37GB just for `x`), so
"one batched forward" is interpreted here as one batched forward *per
chunk*, all B instances together each time, not a Python loop over
individual envs -- see `chunk_size` below.

The raw `N` samples are then compressed per `docs/MILESTONES.md`'s M0 spec:
a coarse grid of `n_grid` evenly rank-spaced order statistics for the bulk
of the distribution, plus the `n_exact_tail` smallest raw values stored
verbatim. "Smallest", not "largest": this project minimizes, so
`g_reward_minimize`'s (`reward/tail_quantile_reward.py`) precision-critical
region is the LOW end of the raw output -- its percentile rank tends to 0
there, and the reward reads `1 - percentile` (`tail_u -> 1`) exactly as the
incumbent approaches this function's true minimum. The grid alone would
under-resolve that region at realistic `n_grid`; the exact block fixes it
without paying for `n_grid` dense enough to cover the whole range that
finely.
"""
import torch

from anytimeacquisition.priors.bnn import BNNPrior


class QuantileTable:
    def __init__(
        self,
        prior: BNNPrior,
        n_samples: int = 1_000_000,
        n_grid: int = 4096,
        n_exact_tail: int = 1000,
        chunk_size: int = 50_000,
        generator: torch.Generator | None = None,
    ):
        """Builds the table against `prior`'s CURRENT draw -- call after
        `prior.reset()`, not before. `n_samples`/`n_grid`/`n_exact_tail`
        default to M0's spec; override with smaller values for fast tests,
        same "benchmark before assuming a default number is cheap" caution
        as `BNNPrior`'s own `ecdf_n_*` knobs (see that module's docstring).
        """
        assert n_exact_tail <= n_grid <= n_samples, (
            f"expected n_exact_tail ({n_exact_tail}) <= n_grid ({n_grid}) <= n_samples ({n_samples})"
        )
        dev = prior.device
        sorted_y = self.draw_sorted_raw(prior, n_samples, chunk_size, generator)

        # Exact tail: the n_exact_tail smallest (best, minimize-good) raw
        # values, stored verbatim, ascending -- ranks [0, n_exact_tail).
        self.exact_tail = sorted_y[:, :n_exact_tail].clone()

        # Grid: n_grid evenly rank-spaced order statistics across the FULL
        # range (including the exact-tail region again at coarser
        # resolution -- simplest to reason about; percentile() below always
        # prefers the exact tail when a value falls inside it).
        idx = torch.linspace(0, n_samples - 1, n_grid, device=dev).long()
        self.grid = sorted_y[:, idx].clone()

        self.n_samples = n_samples
        self.n_exact_tail = n_exact_tail
        self.B, self.d = prior.B, prior.d
        self.device = dev

    @staticmethod
    def draw_sorted_raw(
        prior: BNNPrior, n_samples: int, chunk_size: int = 50_000, generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """The raw (uncompressed) sorted sample this table is built from --
        exposed as a separate, reusable step (not just inlined in
        `__init__`) so a fidelity test can reconstruct the *exact same*
        raw 1e6 samples via an identically-seeded generator and compare
        `percentile()` against a brute-force lookup on them directly, per
        M0's acceptance criterion ("matches a brute-force lookup on the raw
        1e6 samples") -- comparing against a fresh, independent draw instead
        would conflate compression error with ordinary Monte Carlo sampling
        noise between two different draws, a much looser (and different)
        bar than the one M0 asks for. -> sorted [B, n_samples] ascending."""
        B, d, dev = prior.B, prior.d, prior.device
        gen = generator if generator is not None else prior.generator

        chunks = []
        remaining = n_samples
        with torch.no_grad():
            while remaining > 0:
                n = min(chunk_size, remaining)
                x = torch.rand(B, n, d, device=dev, generator=gen)
                x = x * prior.active_dim_mask.unsqueeze(1)
                chunks.append(prior.evaluate(x, noise=False))  # privileged/deterministic, matches build_ecdf
                remaining -= n
        all_y = torch.cat(chunks, dim=1)  # [B, n_samples]
        sorted_y, _ = torch.sort(all_y, dim=1)
        return sorted_y

    def percentile(self, v: torch.Tensor) -> torch.Tensor:
        """Per-instance percentile rank of `v` in [0,1] -- a plain CDF, LOW
        for a small/good `v` under this project's minimize convention (see
        `reward/tail_quantile_reward.py`'s `g_reward_minimize`/
        `g_reward_minimize_from_table` for the `1 - percentile` flip callers
        need to turn this into the actual reward). Interpolates linearly:
        within the exact-tail region between true adjacent order statistics
        (very fine, since consecutive exact_tail entries are literally
        ranks `k`/`k+1` out of `n_samples`), elsewhere between grid nodes.
        v: [B] or [B, ...] (broadcasts over trailing dims) -> same shape."""
        v_shape = v.shape
        v_flat = v.reshape(v_shape[0], -1).contiguous()

        def interp(table: torch.Tensor, rank_scale: float) -> torch.Tensor:
            n = table.shape[1]
            idx = torch.searchsorted(table, v_flat).clamp(1, n - 1)
            lo = torch.gather(table, 1, idx - 1)
            hi = torch.gather(table, 1, idx)
            frac = ((v_flat - lo) / (hi - lo + 1e-12)).clamp(0, 1)
            rank_lo = (idx - 1).float() * rank_scale
            rank_hi = idx.float() * rank_scale
            return (rank_lo + frac * (rank_hi - rank_lo)) / (self.n_samples - 1)

        u_tail = interp(self.exact_tail, rank_scale=1.0)
        u_grid = interp(self.grid, rank_scale=(self.n_samples - 1) / (self.grid.shape[1] - 1))

        in_tail = v_flat <= self.exact_tail[:, -1:]
        u = torch.where(in_tail, u_tail, u_grid)
        return u.reshape(v_shape)


if __name__ == "__main__":
    """Cross-checks percentile() against a brute-force lookup on the SAME
    raw samples the table was built from (M0's literal acceptance
    criterion -- an independent fresh draw would conflate compression
    error with ordinary Monte Carlo noise between two draws, see
    draw_sorted_raw's docstring). Runs at a scale small enough for seconds,
    not M0's literal N=1e6/4096/1000 (see the module docstring on why that
    scale isn't a single call), but the same construction proportionally
    scaled down, which is what the accuracy claim actually rests on.
    Exact smoke-test command: uv run python -m anytimeacquisition.priors.quantile_table
    """
    import time

    torch.manual_seed(0)
    B, d = 4, 3
    n_samples, n_grid, n_exact_tail = 200_000, 1024, 500

    prior = BNNPrior(batch_size=B, x_dim=d, seed=0, cache_dir=None, ecdf_n_draws=5, ecdf_samples_per_draw=100)
    prior.reset()

    build_gen = torch.Generator(device=prior.device).manual_seed(1)
    t0 = time.time()
    table = QuantileTable(
        prior, n_samples=n_samples, n_grid=n_grid, n_exact_tail=n_exact_tail,
        chunk_size=20_000, generator=build_gen,
    )
    print(f"built table for B={B}, n_samples={n_samples} in {time.time() - t0:.2f}s")

    # Reconstruct the EXACT same raw samples via an identically-seeded
    # generator (fresh instance, same seed, same call sequence) -- a
    # compression-fidelity check against the table's own build data.
    ref_gen = torch.Generator(device=prior.device).manual_seed(1)
    y_ref_sorted = QuantileTable.draw_sorted_raw(prior, n_samples, chunk_size=20_000, generator=ref_gen)

    def brute_force_percentile(y_sorted, v):
        idx = torch.searchsorted(y_sorted, v).clamp(0, y_sorted.shape[1] - 1)
        return idx.float() / (y_sorted.shape[1] - 1)

    # Query at a spread of true quantiles, including deep into the tail.
    probe_ranks = torch.tensor([0, 1, 5, 50, 500, 5000, n_samples // 4, n_samples // 2, n_samples - 1])
    v_probe = y_ref_sorted[:, probe_ranks]  # [B, len(probe_ranks)]

    u_table = table.percentile(v_probe)
    u_brute = brute_force_percentile(y_ref_sorted, v_probe)

    err = (u_table - u_brute).abs()
    tail_mask = probe_ranks < n_samples // 100  # "top 1%" in tail_u-space == bottom 1% in raw y-space here
    print(f"max |percentile error|, full range: {err.max().item():.2e}")
    print(f"max |percentile error|, bottom 1% (reward-critical tail): {err[:, tail_mask].max().item():.2e}")
