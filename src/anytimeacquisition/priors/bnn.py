"""Vectorized, ECDF-normalized BNN prior — synthetic data-generating process
for the PFN (M2) and, doubling as an environment, for EXIT rollouts (M5).

Each of the B batch elements is an independently sampled random-architecture
tanh MLP (random depth, random width, width-compensated init scale) — a
fresh "true function" draw per element. Init scale is NOT sampled
independently of width — see `log_amp_range` below and `docs/log/` for why
that used to make most draws flat, and how sampling
`depth * log(crit) = depth * log(init_std**2 * width)` directly (then
deriving `init_std`) instead fixes it, robustly across varying depth too.
Depth/width genuinely differ per element,
which is why this is NOT built on `torch.func.vmap`: vmap requires uniform
shapes/control flow across the mapped dimension, so a vmapped version would
still need every instance padded to the same (Lmax, Wmax) and masked down to
its actual (depth, width) — i.e. exactly what `_raw_forward` below already
does by hand via masked einsums. vmap would add an indirection layer without
buying either more expressiveness or measurably more speed here; it only
pays off if the whole batch shares one architecture (see
`archive/src/exit/prior/environment.py`'s `BatchedTaskFamily`, which does
exactly that — a simpler but less faithful setup, since the design calls for
each instance to be its own random architecture draw). Later, multistart GD
in the search (M5) doesn't need vmap either: restarts are just extra points
along `evaluate`'s existing N (points-per-instance) axis.

Preactivation/output noise, fan-in input scaling, sparseness, and spurious
(irrelevant) input dimensions below are deliberately aligned with PFNs4BO's
(Müller et al., ICML 2023) and ifBO's (Rakotoarison et al., ICML 2024) own
BNN priors — see `docs/log/2026-08-27-pfns4bo-bnn-prior-comparison.md` and
its 2026-08-28 addendum for the exact sourcing and what was deliberately
*not* carried over (input warping — shelved, see that addendum).
"""
import hashlib
import math
from pathlib import Path

import torch

DEFAULT_CACHE_DIR = Path(__file__).parent / "_ecdf_cache"


class BNNPrior:
    def __init__(
        self,
        batch_size: int,
        x_dim: int,
        device: str = "cpu",
        # PFNs4BO/ifBO never sample below depth 8 -- a depth-2/3 tanh MLP is
        # structurally close to incapable of multi-modal output regardless of
        # crit (see docs/log/). Raised the floor to match, and the ceiling
        # further for a deeper family by default; each _raw_forward layer
        # costs the same for every instance regardless of its real depth
        # (padded + masked), so this also raises the per-call compute cost
        # roughly linearly -- ~2x vs. the old (2,17) default.
        depth_range: tuple[int, int] = (8, 32),
        width_range: tuple[int, int] = (32, 128),
        # crit = init_std**2 * width is the standard mean-field/Xavier-style
        # per-layer variance-scaling factor for a tanh net: ~1 preserves
        # variance layer to layer, well below it the signal vanishes toward
        # a near-constant output (flat draws), well above it the network
        # enters the chaotic regime (rich, still-coherent multi-modal
        # structure up to a point, pure noise well past it). A FIXED crit
        # range doesn't stay in the "rich, not noise" band once depth varies
        # a lot: depth and crit compound (total log-amplification through
        # the network is ~ depth * log(crit)), so a crit that's fine at
        # depth=8 can be pure noise at depth=30. log_amp_range targets that
        # compounded quantity directly instead: per instance, sample
        # log_amp = depth * log(crit) uniformly from this range, then derive
        # log_crit = log_amp / depth, crit = exp(log_crit),
        # init_std = sqrt(crit / width). (8.0, 20.0) is picked from the same
        # depth x crit sweep as the original crit_range fix — see docs/log/
        # 2026-08-28's addendum — deep instances no longer need as much
        # per-layer crit to reach the same total amplification, so this
        # keeps both very shallow and very deep instances out of the boring
        # and noise-only regimes respectively.
        log_amp_range: tuple[float, float] = (8.0, 20.0),
        bias_std: float = 0.1,
        # Every hidden-to-hidden weight (W_h only, not W_in/W_out -- matches
        # PFNs4BO/ifBO applying this to `linears[1:-1]` only) is zeroed with
        # this probability, remaining weights rescaled by 1/sqrt(1-sparseness)
        # to preserve variance.
        sparseness: float = 0.145,
        # Preactivation noise (every layer, including the first) and output
        # noise, both per-instance. Set both to (0.0, 0.0) for a fully
        # deterministic prior. Also controllable per-call via evaluate(...,
        # noise=False) without changing the family config -- e.g. for M5's
        # exploit/explore search, which wants an exact, reproducible surface
        # to differentiate through, not a fresh noise draw every step.
        preactivation_noise_std_range: tuple[float, float] = (0.0003, 0.0014),
        output_noise_std_range: tuple[float, float] = (0.0004, 0.0013),
        # Spurious dimensions (PFNs4BO §5.2): each input dim is "relevant"
        # independently with this probability per instance; irrelevant dims
        # have their W_in row zeroed, so they're literally never fed to the
        # network (not just given a dummy value) -- 0.7 matches PFNs4BO's
        # reported 30% irrelevant.
        frac_relevant_features: float = 0.7,
        # Optional: train across a distribution of dimensionalities rather
        # than a single fixed x_dim, matching PFNs4BO's
        # sample_num_feaetures_get_batch (docs/log/). None (default): every
        # instance uses the full x_dim, i.e. today's behavior, unchanged.
        # Set e.g. 1: the whole batch shares one
        # active_dim ~ randint(variable_dim_min, x_dim+1), resampled fresh
        # every reset() (i.e. every training step) -- the PFN still sees a
        # fixed-size x_dim-wide input, but dims beyond active_dim are zeroed
        # (both the weight, via relevant_mask, and the value, in
        # sample_episode below), the fan-in scaling in _raw_forward divides
        # by sqrt(active_dim) not sqrt(x_dim), and nothing downstream needs
        # to know the count explicitly -- "zero" already means "not part of
        # this task." Batch-uniform (every instance in one reset() shares
        # the same active_dim), matching PFNs4BO/ifBO's own convention --
        # NOT per-instance: an earlier version sampled a distinct active_dim
        # per instance, reverted 2026-08-31 after a training run using it
        # stagnated and underperformed fixed-dim ("marginal") models; the
        # working hypothesis is that mixing differently-scaled instances
        # into the same batch/gradient step made optimization noisier than
        # necessary, on top of the genuine harder-in-high-dim signal you'd
        # expect either way -- see
        # docs/log/2026-08-31-variable-xdim-training-stagnation.md (not
        # confirmed via a controlled rerun yet).
        variable_dim_min: int | None = None,
        ecdf_n_samples: int = 1000,
        # ECDF-fit cost scales with n_draws * samples_per_draw (each draw is
        # a full reset() + forward pass) AND with the family's own cost per
        # forward pass (x_dim, depth_range, width_range, batch_size) --
        # there is no fixed "costs ~Ns" number that stays true across
        # config changes. Measured ~13s at x_dim=2 (2026-08-27) but ~70s at
        # x_dim=4 with the wider depth_range and sparseness/noise/spurious-
        # dims machinery added since (2026-08-28) -- pooling still only
        # needs 50 draws x 300 samples_per_draw (50*300*batch_size raw
        # samples, 960,000 at batch_size=64) to be statistically solid, but
        # *time* that pooling on your actual config before assuming it's
        # cheap, especially before bumping it up further (see
        # configs/experiment/pfn_ulysses_real.yaml's own history of this
        # exact mistake). The archived prototype's original defaults
        # (1000x1000) measured over 30 minutes uncached and were never
        # actually benchmarked before being copied here — don't reuse
        # numbers like that, or these, without benchmarking first.
        # This is a one-time cost per family config either way (see the
        # disk cache above) — raise these if the fitted ECDF's tails look
        # noisy for your chosen (x_dim, depth_range, width_range).
        ecdf_n_draws: int = 50,
        ecdf_samples_per_draw: int = 300,
        cache_dir: str | Path | None = DEFAULT_CACHE_DIR,
        seed: int | None = None,
        # Reuse an already-fit ECDF instead of fitting/loading this
        # instance's own -- [1, ecdf_n_samples] or [B, ecdf_n_samples]
        # (only the first row is used either way). For callbacks/
        # dim_validation.py's dedicated per-dimension validation priors,
        # which all share the TRAINING prior's own ecdf_sorted rather than
        # each fitting an independent one -- matching ifBO's own approach
        # (a single precomputed, universal calibration reference --
        # `ifbo/priors/ftpfn_prior.py`'s `output_sorted.npy`, used
        # regardless of which num_features a given sample happens to draw
        # -- not a separate normalization per dimensionality). Letting each
        # dimension's validation prior independently whiten its own raw-
        # output distribution to [0,1] would risk masking genuine
        # cross-dimension difficulty differences rather than revealing
        # them -- see docs/log/2026-08-31-variable-xdim-training-
        # stagnation.md. ecdf_n_samples/ecdf_n_draws/ecdf_samples_per_draw/
        # cache_dir are ignored when this is given (no fitting happens).
        ecdf_sorted: torch.Tensor | None = None,
    ):
        self.B, self.d, self.device = batch_size, x_dim, device
        self.depth_range = depth_range
        self.width_range = width_range
        self.log_amp_range = log_amp_range
        self.bias_std = bias_std
        self.sparseness = sparseness
        self.preactivation_noise_std_range = preactivation_noise_std_range
        self.output_noise_std_range = output_noise_std_range
        self.frac_relevant_features = frac_relevant_features
        self.variable_dim_min = variable_dim_min
        self.Lmax = depth_range[1]
        self.Wmax = width_range[1]
        self.ecdf_n_samples = ecdf_n_samples
        self.ecdf_n_draws = ecdf_n_draws
        self.ecdf_samples_per_draw = ecdf_samples_per_draw
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

        self.generator = torch.Generator(device=device)
        if seed is not None:
            self.generator.manual_seed(seed)

        self.reset()
        if ecdf_sorted is not None:
            self.ecdf_sorted = ecdf_sorted[:1].to(self.device).expand(self.B, -1).clone()
        else:
            self._load_or_fit_ecdf()

    def reset(self) -> None:
        """Draw a fresh batch of B independent architectures/weights."""
        B, d, Wmax, Lmax, dev, gen = self.B, self.d, self.Wmax, self.Lmax, self.device, self.generator
        depth_lo, depth_hi = self.depth_range
        width_lo, width_hi = self.width_range
        amp_lo, amp_hi = self.log_amp_range
        pn_lo, pn_hi = self.preactivation_noise_std_range
        on_lo, on_hi = self.output_noise_std_range

        self.depth = torch.randint(depth_lo, depth_hi, (B,), device=dev, generator=gen)
        self.width = torch.randint(width_lo, width_hi, (B,), device=dev, generator=gen)
        # Target the compounded quantity (depth * log(crit)) directly, not
        # crit alone, so deep and shallow instances both land in the "rich,
        # not noise" band instead of crit's effect scaling with depth by
        # accident (see docs/log/ 2026-08-28 addendum).
        log_amp = torch.empty(B, device=dev).uniform_(amp_lo, amp_hi, generator=gen)
        self.crit = (log_amp / self.depth.float()).exp()
        init_std = (self.crit / self.width.float()).sqrt()

        self.preactivation_noise_std = torch.empty(B, device=dev).uniform_(pn_lo, pn_hi, generator=gen)
        self.output_noise_std = torch.empty(B, device=dev).uniform_(on_lo, on_hi, generator=gen)

        unit_idx = torch.arange(Wmax, device=dev).unsqueeze(0).expand(B, -1)
        self.width_mask = (unit_idx < self.width.unsqueeze(1)).float()
        layer_idx = torch.arange(Lmax, device=dev).unsqueeze(0).expand(B, -1)
        self.layer_mask = (layer_idx < self.depth.unsqueeze(1)).float()

        # Variable dimensionality (optional, see __init__): dims >=
        # active_dim are "not part of this task" -- composed into
        # relevant_mask below. Disabled (variable_dim_min=None) -> active_dim
        # is always the full x_dim, active_dim_mask is all-ones, byte-
        # identical to not having this feature at all.
        #
        # Batch-uniform, not per-instance: one active_dim shared by all B
        # instances in this reset() (still resampled fresh every reset(),
        # i.e. every training step -- matches how n_train is already
        # resampled per step). An earlier version sampled a distinct
        # active_dim per instance; reverted 2026-08-31 -- see
        # docs/log/2026-08-31-variable-xdim-training-stagnation.md, matching
        # ifBO/PFNs4BO's own batch-level convention instead (their whole
        # batch shares one num_features per step) rather than mixing
        # differently-scaled instances into the same batch/gradient step.
        if self.variable_dim_min is not None:
            active_dim_scalar = torch.randint(self.variable_dim_min, d + 1, (1,), device=dev, generator=gen)
            self.active_dim = active_dim_scalar.expand(B).clone()
        else:
            self.active_dim = torch.full((B,), d, device=dev, dtype=torch.long)
        dim_idx = torch.arange(d, device=dev).unsqueeze(0).expand(B, -1)
        self.active_dim_mask = (dim_idx < self.active_dim.unsqueeze(1)).float()

        # Spurious dimensions: zero the W_in row for "irrelevant" dims below,
        # so they're literally not fed to the network. Composed with
        # active_dim_mask -- a dim beyond active_dim is trivially irrelevant
        # too.
        self.relevant_mask = torch.bernoulli(
            torch.full((B, d), self.frac_relevant_features, device=dev), generator=gen
        ) * self.active_dim_mask

        s = init_std.view(B, 1, 1)
        self.W_in = torch.randn(B, d, Wmax, device=dev, generator=gen) * s
        self.W_in = self.W_in * self.relevant_mask.unsqueeze(-1)
        self.b_in = torch.randn(B, Wmax, device=dev, generator=gen) * self.bias_std
        self.W_h = torch.randn(B, Lmax, Wmax, Wmax, device=dev, generator=gen) * init_std.view(B, 1, 1, 1)
        if self.sparseness > 0.0:
            keep_mask = torch.bernoulli(
                torch.full((B, Lmax, Wmax, Wmax), 1.0 - self.sparseness, device=dev), generator=gen
            )
            self.W_h = self.W_h / math.sqrt(1.0 - self.sparseness) * keep_mask
        self.b_h = torch.randn(B, Lmax, Wmax, device=dev, generator=gen) * self.bias_std
        self.W_out = torch.randn(B, Wmax, 1, device=dev, generator=gen) * s
        self.b_out = torch.randn(B, 1, device=dev, generator=gen) * self.bias_std

    def _raw_forward(self, x: torch.Tensor, noise: bool = True) -> torch.Tensor:
        """x: [B,N,d] -> raw (unnormalized) y: [B,N]. Differentiable w.r.t. x.
        noise=False disables preactivation/output noise for this call only
        (deterministic given the current weights) — e.g. for M5's
        exploit/explore search, which wants an exact, reproducible surface
        to differentiate through rather than a fresh noise draw every step."""
        B, N, d = x.shape
        assert B == self.B, (
            f"x's leading dim ({B}) must match the prior's batch_size ({self.B}) — "
            "each batch slot is its own drawn environment; evaluate N points per "
            "instance via x's middle dim, not by changing B. For a single-instance "
            "search, construct a BNNPrior with batch_size=1."
        )

        def add_noise(pre: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
            if not noise:
                return pre
            return pre + torch.randn(pre.shape, device=pre.device, generator=self.generator) * std.view(B, 1, 1)

        # Fan-in-aware input scaling: z-score a Uniform(0,1) input (mean .5,
        # std sqrt(1/12)) then divide by sqrt(active_dim) -- the number of
        # dims actually feeding the first layer this step, not the tensor's
        # structural width -- so first-layer preactivation variance doesn't
        # grow with x_dim. Matches PFNs4BO's sample_input() + get_batch()'s
        # `x_ / sqrt(num_features)`; active_dim generalizes it to variable
        # dimensionality (variable_dim_min above, batch-uniform -- see that
        # comment). active_dim == d for every instance when that's disabled,
        # so this is exactly the old `/ sqrt(d)` in that case.
        x_scaled = (x - 0.5) / math.sqrt(1 / 12) / self.active_dim.float().clamp(min=1).sqrt().view(B, 1, 1)

        pre = torch.einsum("bnd,bdw->bnw", x_scaled, self.W_in) + self.b_in.unsqueeze(1)
        pre = add_noise(pre, self.preactivation_noise_std)
        h = torch.tanh(pre) * self.width_mask.unsqueeze(1)
        for l in range(self.Lmax):
            pre = torch.einsum("bnw,bwv->bnv", h, self.W_h[:, l]) + self.b_h[:, l].unsqueeze(1)
            pre = add_noise(pre, self.preactivation_noise_std)
            h_new = torch.tanh(pre) * self.width_mask.unsqueeze(1)
            m = self.layer_mask[:, l].view(B, 1, 1)
            h = m * h_new + (1 - m) * h
        out = torch.einsum("bnw,bwo->bno", h, self.W_out).squeeze(-1) + self.b_out
        if noise:
            out = out + torch.randn(out.shape, device=out.device, generator=self.generator) * self.output_noise_std.view(B, 1)
        return out

    def _cache_key(self) -> str:
        fingerprint = (
            self.d, self.depth_range, self.width_range, self.log_amp_range,
            self.bias_std, self.sparseness, self.preactivation_noise_std_range,
            self.output_noise_std_range, self.frac_relevant_features, self.variable_dim_min,
            self.ecdf_n_samples, self.ecdf_n_draws, self.ecdf_samples_per_draw,
        )
        digest = hashlib.sha256(repr(fingerprint).encode()).hexdigest()[:16]
        return f"bnn_ecdf_d{self.d}_{digest}.pt"

    def _load_or_fit_ecdf(self) -> None:
        cache_file = self.cache_dir / self._cache_key() if self.cache_dir is not None else None
        if cache_file is not None and cache_file.exists():
            cached = torch.load(cache_file, map_location=self.device)  # shape [1, ecdf_n_samples]
            self.ecdf_sorted = cached.expand(self.B, -1).clone()
            return

        self._fit_ecdf()

        if cache_file is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            torch.save(self.ecdf_sorted[:1].clone(), cache_file)

    def _fit_ecdf(self) -> None:
        """Builds a marginal prior ECDF by sampling across randomized
        architectures, weights, and inputs — pooled across the *family*
        (many fresh `reset()` draws), not per-instance, so it approximates
        the ECDF of the whole BNN prior rather than one specific draw.
        """
        n_samples, n_draws, samples_per_draw = (
            self.ecdf_n_samples, self.ecdf_n_draws, self.ecdf_samples_per_draw,
        )
        raw_accum = []
        with torch.no_grad():
            for _ in range(n_draws):
                self.reset()
                probe_x = torch.rand(self.B, samples_per_draw, self.d, device=self.device, generator=self.generator)
                raw_accum.append(self._raw_forward(probe_x))

            all_raw = torch.cat(raw_accum, dim=1).flatten().unsqueeze(0)  # pooled across batch too
            sorted_raw, _ = torch.sort(all_raw, dim=1)
            total_samples = sorted_raw.shape[1]
            idx = torch.linspace(0, total_samples - 1, n_samples, device=self.device).long()
            decimated = sorted_raw[:, idx]
            self.ecdf_sorted = decimated.expand(self.B, -1).clone()

        self.reset()  # restore a fresh instance draw, don't leak the last ECDF-fitting draw

    def evaluate(self, x: torch.Tensor, noise: bool = True) -> torch.Tensor:
        """Differentiable ECDF-normalized evaluation ("step" — the env's
        dynamics). x: [B,N,d] -> y: [B,N] in [0, 1]. noise=False for a
        deterministic evaluation at the current weights (see _raw_forward)."""
        raw = self._raw_forward(x, noise=noise)
        ref = self.ecdf_sorted
        S = ref.shape[1]
        idx = torch.searchsorted(ref, raw.detach(), right=True).clamp(1, S - 1)
        lo = torch.gather(ref, 1, idx - 1)
        hi = torch.gather(ref, 1, idx)
        frac = ((raw - lo) / (hi - lo + 1e-8)).clamp(0, 1)  # differentiable w.r.t. raw
        y_norm = (idx.float() - 1 + frac) / (S - 1)
        return y_norm.clamp(0, 1)

    def sample_episode(
        self, n_train: int, n_test: int, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Uniform-random train/test points for PFN training (M2).
        Returns (x_train, y_train, x_test, y_test), batched over B. Inactive
        dims (variable_dim_min, if set) are zeroed in the returned x -- the
        BNN's own forward pass ignores them regardless (their W_in row is
        already zero), but a downstream model reading x directly needs an
        honest "not part of this task" signal, not leftover random values
        that happen not to matter."""
        gen = generator if generator is not None else self.generator
        x = torch.rand(self.B, n_train + n_test, self.d, device=self.device, generator=gen)
        x = x * self.active_dim_mask.unsqueeze(1)
        y = self.evaluate(x)
        return x[:, :n_train], y[:, :n_train], x[:, n_train:], y[:, n_train:]


def plot_1d_environments(
    n_envs: int = 3,
    n_random: int = 14,
    seed: int = 7,
    out_path: str | Path | None = None,
):
    """Plot `n_envs` independently drawn 1D BNN environments: the true curve
    on a dense grid, plus `n_random` random ECDF-normalized query points
    each. x_dim is fixed to 1 here — a visualization aid, not a general
    plotting utility for arbitrary dimensionality. Returns the Figure;
    saves to `out_path` if given."""
    import matplotlib.pyplot as plt

    torch.manual_seed(seed)
    prior = BNNPrior(
        batch_size=n_envs, x_dim=1,
        ecdf_n_draws=50, ecdf_samples_per_draw=300, ecdf_n_samples=1000,
        cache_dir=None, seed=seed,
    )

    x_grid = torch.linspace(0, 1, 400).view(1, -1, 1).expand(n_envs, -1, -1)
    with torch.no_grad():
        y_grid = prior.evaluate(x_grid)

    x_rand = torch.rand(n_envs, n_random, 1)
    with torch.no_grad():
        y_rand = prior.evaluate(x_rand)

    # Okabe-Ito colorblind-safe categorical hues, fixed order.
    colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9"]
    colors = (colors * (n_envs // len(colors) + 1))[:n_envs]
    ink, grid_color = "#1a1a1a", "#d9d9d9"

    fig, axes = plt.subplots(1, n_envs, figsize=(4.3 * n_envs, 4), sharey=True)
    axes = [axes] if n_envs == 1 else axes

    for i, ax in enumerate(axes):
        color = colors[i]
        depth, width = prior.depth[i].item(), prior.width[i].item()
        ax.plot(x_grid[i, :, 0].numpy(), y_grid[i].numpy(), color=color, linewidth=2, zorder=2)
        ax.scatter(
            x_rand[i, :, 0].numpy(), y_rand[i].numpy(),
            s=42, color=color, edgecolor="white", linewidth=0.8, zorder=3,
            label="random query points",
        )
        ax.set_title(f"Environment {i + 1}  (depth={depth}, width={width})", fontsize=11, color=ink)
        ax.set_xlabel("x", color=ink)
        if i == 0:
            ax.set_ylabel("y (ECDF-normalized)", color=ink)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, color=grid_color, linewidth=0.6, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(grid_color)
        ax.tick_params(colors=ink)
        ax.legend(loc="upper right", fontsize=8, frameon=False)

    fig.suptitle(f"BNNPrior — {n_envs} independently drawn 1D environments", fontsize=13, color=ink, y=1.03)
    fig.tight_layout()

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
        print("saved", out_path)

    return fig


def plot_2d_environments(
    n_envs: int = 3,
    n_random: int = 30,
    grid_res: int = 80,
    seed: int = 7,
    out_path: str | Path | None = None,
):
    """Plot `n_envs` independently drawn 2D BNN environments as heatmaps
    (evaluate() over a dense [0,1]^2 grid) with `n_random` random
    ECDF-normalized query points overlaid. x_dim is fixed to 2 here — a
    visualization aid, not a general plotting utility for arbitrary
    dimensionality. Returns the Figure; saves to `out_path` if given."""
    import matplotlib.pyplot as plt

    torch.manual_seed(seed)
    prior = BNNPrior(
        batch_size=n_envs, x_dim=2,
        ecdf_n_draws=50, ecdf_samples_per_draw=300, ecdf_n_samples=1000,
        cache_dir=None, seed=seed,
    )

    lin = torch.linspace(0, 1, grid_res)
    gx, gy = torch.meshgrid(lin, lin, indexing="xy")
    grid = torch.stack([gx.flatten(), gy.flatten()], dim=-1)
    x_grid = grid.unsqueeze(0).expand(n_envs, -1, -1)
    with torch.no_grad():
        y_grid = prior.evaluate(x_grid).view(n_envs, grid_res, grid_res)

    x_rand = torch.rand(n_envs, n_random, 2)
    with torch.no_grad():
        y_rand = prior.evaluate(x_rand)

    ink = "#1a1a1a"
    fig, axes = plt.subplots(1, n_envs, figsize=(4.6 * n_envs, 4.3), constrained_layout=True)
    axes = [axes] if n_envs == 1 else axes

    im = None
    for i, ax in enumerate(axes):
        depth, width = prior.depth[i].item(), prior.width[i].item()
        im = ax.imshow(
            y_grid[i].numpy(), origin="lower", extent=(0, 1, 0, 1),
            cmap="viridis", vmin=0, vmax=1, aspect="equal",
        )
        ax.scatter(
            x_rand[i, :, 0].numpy(), x_rand[i, :, 1].numpy(),
            s=42, c="white", edgecolor=ink, linewidth=1.0, zorder=3,
            label="random query points",
        )
        ax.set_title(f"Environment {i + 1}  (depth={depth}, width={width})", fontsize=11, color=ink)
        ax.set_xlabel("x1", color=ink)
        if i == 0:
            ax.set_ylabel("x2", color=ink)
        ax.tick_params(colors=ink)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.legend(loc="upper right", fontsize=8, frameon=True, facecolor="white", framealpha=0.85)

    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label("y (ECDF-normalized)", color=ink)
    cbar.ax.tick_params(colors=ink)

    fig.suptitle(f"BNNPrior — {n_envs} independently drawn 2D environments", fontsize=13, color=ink)

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
        print("saved", out_path)

    return fig


if __name__ == "__main__":
    torch.manual_seed(0)
    prior = BNNPrior(batch_size=4, x_dim=3, ecdf_n_draws=20, ecdf_samples_per_draw=200, ecdf_n_samples=200)

    x = torch.rand(4, 100, 3)
    y = prior.evaluate(x)
    print("evaluate() output range:", y.min().item(), y.max().item())

    x_tr, y_tr, x_te, y_te = prior.sample_episode(n_train=10, n_test=5)
    print("episode shapes:", x_tr.shape, y_tr.shape, x_te.shape, y_te.shape)

    x_grad = torch.rand(4, 8, 3, requires_grad=True)
    prior.reset()
    y_grad = prior.evaluate(x_grad)
    y_grad.sum().backward()
    print("d evaluate / dx nonzero entries:", (x_grad.grad.abs() > 0).sum().item(), "/", x_grad.grad.numel())

    prior.reset()
    y_before = prior.evaluate(x)
    prior.reset()
    y_after = prior.evaluate(x)
    print("batch elements differ after reset (new architectures):", not torch.allclose(y_before, y_after))

    # 3 fresh 1D environments, true curve + random query points each.
    # Play with n_envs / n_random / seed here.
    plot_1d_environments(
        n_envs=3, n_random=14, seed=7,
        out_path=Path(__file__).parent / "_demo_plots" / "bnn_environments_1d.png",
    )

    # Same idea in 2D — heatmap per environment, random query points overlaid.
    plot_2d_environments(
        n_envs=3, n_random=30, grid_res=80, seed=7,
        out_path=Path(__file__).parent / "_demo_plots" / "bnn_environments_2d.png",
    )
