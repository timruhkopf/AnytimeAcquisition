"""
tail_quantile_reward.py

Computes the environment reward for a Bayesian Optimization (BO) acquisition policy
trained on synthetically sampled Bayesian Neural Network (BNN) instances.

Rationale:
Why not raw regret?
Minimizing raw regret (argmin AUC = argmax Σ f_t*) requires knowing the global
optimum y*. Estimating y* via multi-start gradient ascent on complex landscapes
systematically undershoots, injecting per-function label bias. This bias acts
as reward noise that is most severe exactly on the hardest functions where the
policy needs the most accurate feedback. Additionally, raw regret requires
output-scale normalization to prevent functions with massive scales from dominating
the loss, which introduces further distortion.

The Solution: Tail Quantile Reward via Extreme Value Theory
Instead of raw function values, we score the incumbent f_t* by its percentile
rank against the function's own input distribution.
1. We draw N=10^6 uniform samples to form an empirical CDF (amortized once per BNN sample).
2. We isolate the top 1% of samples and fit a Generalized Pareto Distribution (GPD)
   using the Peaks-Over-Threshold (POT) method.
3. The reward is the log-scaled tail probability: g_t = -log10(1 - F(f_t*)) / scale.

Properties:
- Zero dependency on y*: Systematically eliminates optimization label bias.
- Scale invariant: Prior draws with wildly different output scales are naturally
  commensurable. No per-function power transforms needed in the reward.
- Log-scaled precision: Moving from the 99th to 99.9th percentile yields the
  same reward as moving from 90th to 99th.
- Smooth deep tails: The GPD smoothly extrapolates percentiles beyond the
  empirical limit of the 10^6 samples, eliminating the need for hard reward clipping.
"""

import numpy as np
import torch
from scipy.stats import genpareto

from anytimeacquisition.priors.bnn import BNNPrior


def g_from_percentile(u, max_score=4.0):
    """
    0. THE REWARD AS A FUNCTION OF AN ALREADY-KNOWN PERCENTILE (M0)
    Same clipped log-tail transform as `clipped_empirical_reward`, but takes
    the ECDF value u = F(f_t) directly (e.g. a rank computed once against a
    per-function reference grid) instead of re-deriving it from raw samples
    every call. Vectorized: u may be a numpy array.
    """
    tail_prob = np.clip(1.0 - np.asarray(u), 10**-max_score, 1.0)
    return np.clip(-np.log10(tail_prob), 0, max_score) / max_score


def clipped_empirical_reward(f_t, samples, max_score=4.0):
    """
    1. THE ORIGINAL CLIPPED REWARD (v1)
    Scores the incumbent strictly using the empirical CDF of the Monte Carlo samples.
    """
    N = len(samples)

    # 1 - u_t: The empirical tail probability (fraction of points better than f_t)
    tail_prob = np.sum(samples > f_t) / N

    # Guard against log(0) if f_t is better than all 1,000,000 samples
    tail_prob = max(tail_prob, 10 ** -max_score)

    # g_t = clip(-log10(1 - u_t), 0, max_score) / max_score
    raw_score = -np.log10(tail_prob)
    clipped_score = np.clip(raw_score, 0, max_score)

    return clipped_score / max_score


def fit_gpd_tail(samples, top_percentile=99.0):
    """
    2. THE GPD FIT
    Isolates the top 1% of samples and fits the Generalized Pareto Distribution.
    This only needs to be run once per function, amortized alongside the MC sampling.

    Deferred to M8 (docs/ROADMAP.md §8, gated on the clip-bind rate measured
    in M1) -- not on the v1 reward path (`clipped_empirical_reward`/
    `g_from_percentile`), but kept NaN-safe regardless (§2.3): `genpareto.fit`
    can fail to converge, and for xi < -0.5 the MLE is not a regular
    estimator (Smith's condition) -- exactly the regime bounded functions
    (tanh BNN draws on a compact domain) occupy. Falls back to a fixed
    exponential tail (xi=0) if the fit raises.
    """
    # Find the threshold u
    u = np.percentile(samples, top_percentile)

    # Isolate exceedances (x_i = y_i - u)
    exceedances = samples[samples > u] - u

    try:
        # Fit GPD via Maximum Likelihood using SciPy
        # floc=0 forces the location parameter to 0 since we already subtracted u
        # Returns: shape (xi), location (0), scale (sigma)
        xi, _, sigma = genpareto.fit(exceedances, floc=0)
    except Exception:
        xi, sigma = 0.0, float(exceedances.std()) or 1.0

    # P(Y > u): Empirical probability of exceeding the threshold
    p_u = len(exceedances) / len(samples)

    return u, xi, sigma, p_u


def _gpd_survival(x, xi, sigma):
    """P(X > x) under the fitted GPD (§2.3). BNN functions with tanh
    activations on a compact domain are bounded, so the fitted shape xi is
    almost always negative, giving a finite upper endpoint at
    u + sigma/|xi| -- past it, `(1 + xi*x/sigma)` goes negative and a naive
    `** (-1/xi)` returns nan (numpy) or a complex number (plain Python),
    which then propagates through reward -> return -> loss -> every
    parameter, and does so *more* often as training succeeds (pushing into
    the extreme tail is the whole point). Returns 0.0 past the endpoint --
    caller MUST clip the resulting tail_prob away from exactly 0 before
    `-log10`."""
    if abs(xi) < 1e-6:
        return float(np.exp(-x / sigma))
    z = 1.0 + xi * x / sigma
    if z <= 0.0:
        return 0.0
    return float(z ** (-1.0 / xi))


def unclipped_gpd_reward(f_t, samples, u, xi, sigma, p_u, scale_factor=4.0):
    """
    3. THE UNCLIPPED REWARD (POT Method)
    Uses empirical CDF for the body, and smooth GPD extrapolation for the deep tail.
    Returns a score that smoothly exceeds 1.0 if the agent pushes past 99.99%.
    """
    if f_t <= u:
        # Fall back to empirical CDF if the incumbent isn't in the top 1% yet
        N = len(samples)
        tail_prob = np.sum(samples > f_t) / N
        # Guard against absolute zero for safety, though mathematically impossible here
        tail_prob = max(tail_prob, 1e-10)

    else:
        # Calculate exceedance of the incumbent
        x = f_t - u

        # P(Y > f_t) = P(Y > u) * P(X > x | Y > u)
        tail_prob = p_u * _gpd_survival(x, xi, sigma)

    # Guard against float64 underflow (and the GPD's exact-0.0 past its
    # finite endpoint, §2.3) in spectacularly deep tails
    tail_prob = max(tail_prob, 1e-300)

    # Divide by scale_factor so 99.99th percentile still maps to exactly 1.0,
    # but 99.999th maps to 1.25, 99.9999th to 1.5, etc.
    return -np.log10(tail_prob) / scale_factor


# ---------------------------------------------------------------------------
# Batched (torch) pipeline (M1): build_ecdf / percentile / normalized_advantage
# / clip_bind_rate operate on a whole `BNNPrior` batch at once, for use
# directly against the trajectories a rollout produces, rather than one
# scalar f_t/samples pair at a time like the functions above.
# ---------------------------------------------------------------------------


def build_ecdf(prior: BNNPrior, n_samples: int = 100_000, seed: int | None = None) -> torch.Tensor:
    """Per-instance (not family-pooled -- unlike `BNNPrior`'s own internal
    `ecdf_sorted`) empirical reference distribution: a dense Sobol sweep of
    `prior`'s domain, evaluated on the *current* instance draw and sorted --
    the reward reference every incumbent's percentile rank is computed
    against. Sobol (low-discrepancy), not uniform pseudo-random, per
    docs/ROADMAP.md §M1. `n_samples=100_000` is chosen for *sufficiency*
    (5 decades of resolution >= the 4-decade clip in `g_from_percentile`),
    not cost.

    -> sorted [B, n_samples] (ascending), same batch/device as `prior`.
    """
    sob = torch.quasirandom.SobolEngine(dimension=prior.d, scramble=True, seed=seed)
    x = sob.draw(n_samples).to(prior.device)
    x = x.unsqueeze(0).expand(prior.B, -1, -1)
    y = prior.evaluate(x, noise=False)  # [B, n_samples], privileged/deterministic
    y_sorted, _ = torch.sort(y, dim=1)
    return y_sorted


def percentile(y_sorted: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Per-instance rank of `v` within `y_sorted` (e.g. from `build_ecdf`),
    as a fraction in [0, 1]. y_sorted: [B, N] ascending. v: [B] or [B, ...]
    (broadcasts over any trailing dims) -> same shape as `v`."""
    v_shape = v.shape
    idx = torch.searchsorted(y_sorted, v.reshape(v_shape[0], -1).contiguous())
    idx = idx.clamp(0, y_sorted.shape[1] - 1)
    return (idx.float() / (y_sorted.shape[1] - 1)).reshape(v_shape)


def normalized_advantage(
    G_bar: torch.Tensor, g_prev: torch.Tensor, sat_threshold: float = 1 - 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Â_t = (Ḡ_t - g_{t-1}) / (1 - g_{t-1}) (docs/ROADMAP.md §2.6). Blows up
    as g_{t-1} -> 1 (the incumbent has already saturated the reward's clip);
    masked out of the loss rather than epsilon-stabilized, since adding an
    epsilon to the denominator just divides noise by epsilon instead of
    fixing anything. `Ḡ_t >= g_{t-1}` always holds (the incumbent is
    monotone), so Â is never negative on unmasked steps.

    -> (value [same shape as inputs], mask [bool, True = valid/unmasked]).
    """
    mask = g_prev < sat_threshold
    denom = torch.where(mask, 1 - g_prev, torch.ones_like(g_prev))  # avoid /0 where masked out anyway
    value = (G_bar - g_prev) / denom
    return value, mask


def clip_bind_rate(u: torch.Tensor, max_score: float = 4.0) -> float:
    """Fraction of percentiles `u` where the reward's clip actually bound
    (the pre-clip score would have exceeded `max_score`) -- required
    instrumentation per docs/ROADMAP.md §M1, and the number that decides
    whether M8 reinstates the GPD tail (§2.3/§8)."""
    tail_prob = torch.clamp(1.0 - u, min=1e-300, max=1.0)
    raw_score = -torch.log10(tail_prob)
    return (raw_score > max_score).float().mean().item()


if __name__ == "__main__":
    """Rolls an EI baseline out against a fresh BNN draw, computes the
    tail-quantile reward and Â along the trajectory, and reports the two
    required instrumentation numbers -- clip-bind rate and advantage-mask
    rate -- per docs/ROADMAP.md §M1's exit criterion ("measured on EI
    trajectories. Record it -- it decides M8.")."""
    from functools import partial

    from anytimeacquisition.metrics.rollout import rollout_episode
    from anytimeacquisition.models.baselines.gp_acquisition import gp_acquisition_policy

    torch.manual_seed(0)
    x_dim, batch_size = 2, 8
    n_init, n_steps = 3, 15

    prior = BNNPrior(batch_size=batch_size, x_dim=x_dim, seed=0)
    y_sorted = build_ecdf(prior, n_samples=100_000, seed=1)

    policy_fn = partial(gp_acquisition_policy, acquisition="EI", num_restarts=5, raw_samples=64)
    rollout = rollout_episode(prior, n_init=n_init, n_steps=n_steps, policy_fn=policy_fn, reset=False)
    y_context = rollout["y_context"]  # [B, n_init+n_steps]

    incumbent = torch.cummax(y_context, dim=1).values  # [B, T]
    u = percentile(y_sorted, incumbent)  # [B, T]
    g = torch.as_tensor(g_from_percentile(u.numpy(), max_score=4.0))  # [B, T], running incumbent's own reward
    t = torch.arange(1, g.shape[1] + 1, dtype=g.dtype)
    g_bar = torch.cumsum(g, dim=1) / t  # [B, T], Ḡ_t = running mean reward through step t

    advantage, mask = normalized_advantage(g_bar[:, 1:], g[:, :-1])
    print(f"clip-bind rate on EI trajectories:      {clip_bind_rate(u):.4f}")
    print(f"advantage-mask rate (g_prev saturated):  {(~mask).float().mean().item():.4f}")
    print(f"mean final incumbent g-reward:           {g[:, -1].mean().item():.4f}")