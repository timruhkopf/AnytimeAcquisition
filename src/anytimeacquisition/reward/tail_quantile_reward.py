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
from scipy.stats import genpareto


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
    """
    # Find the threshold u
    u = np.percentile(samples, top_percentile)

    # Isolate exceedances (x_i = y_i - u)
    exceedances = samples[samples > u] - u

    # Fit GPD via Maximum Likelihood using SciPy
    # floc=0 forces the location parameter to 0 since we already subtracted u
    # Returns: shape (xi), location (0), scale (sigma)
    xi, _, sigma = genpareto.fit(exceedances, floc=0)

    # P(Y > u): Empirical probability of exceeding the threshold
    p_u = len(exceedances) / len(samples)

    return u, xi, sigma, p_u


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

        # P(X > x | Y > u): GPD survival function
        if abs(xi) < 1e-6:
            # Fallback to Exponential if shape parameter is effectively zero
            gpd_sf = np.exp(-x / sigma)
        else:
            # Standard GPD formula
            gpd_sf = (1 + (xi * x) / sigma) ** (-1.0 / xi)

        # P(Y > f_t) = P(Y > u) * P(X > x | Y > u)
        tail_prob = p_u * gpd_sf

    # Guard against float64 underflow in spectacularly deep tails
    tail_prob = max(tail_prob, 1e-300)

    # Divide by scale_factor so 99.99th percentile still maps to exactly 1.0,
    # but 99.999th maps to 1.25, 99.9999th to 1.5, etc.
    return -np.log10(tail_prob) / scale_factor