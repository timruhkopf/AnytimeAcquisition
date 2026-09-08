"""`BOEnv` -- stateful Bayesian-optimization environment (M0) wrapping a
`BNNPrior` and a `QuantileTable` into a `reset`/`step` interface, so
downstream rollout/training code gets `D_t` and the tail-quantile reward
`g_{t+1}` directly instead of assembling them from `BNNPrior` +
`QuantileTable` + `reward/tail_quantile_reward.py` by hand each time (as
`metrics/rollout.py`'s `rollout_episode` + its own `__main__` currently do).
`rollout_episode` isn't replaced by this -- it's a lighter policy-agnostic
harness already used elsewhere; `BOEnv` is the stateful, reward-emitting
object `docs/MILESTONES.md`'s M0 spec asks for.

GPD tail extrapolation is deliberately NOT wired in here (M0: "Do not
build"). `reward/tail_quantile_reward.py`'s `fit_gpd_tail`/`_gpd_survival`/
`unclipped_gpd_reward` exist but stay unused by this module; that's an M8
decision (`docs/MILESTONES.md`), gated on the clip-bind rate.
"""
import torch

from anytimeacquisition.priors.bnn import BNNPrior
from anytimeacquisition.priors.quantile_table import QuantileTable
from anytimeacquisition.reward.tail_quantile_reward import g_reward_minimize_from_table


class BOEnv:
    def __init__(
        self,
        prior: BNNPrior,
        budget: int,
        max_score: float = 4.0,
        quantile_table_kwargs: dict | None = None,
    ):
        """`prior` is owned by this env -- its `reset()` is called from
        `BOEnv.reset()`, don't reset it independently once wrapped.
        `budget` is the total number of observations (n_init + steps) an
        episode runs for; `step()` past it is a caller bug, not silently
        handled. `quantile_table_kwargs` forwards to `QuantileTable`
        (e.g. smaller `n_samples` for fast tests -- see that module)."""
        self.prior = prior
        self.budget = budget
        self.max_score = max_score
        self.quantile_table_kwargs = quantile_table_kwargs or {}
        self.quantile_table: QuantileTable | None = None
        self.x_context: torch.Tensor | None = None
        self.y_context: torch.Tensor | None = None
        self.incumbent: torch.Tensor | None = None
        self.t = 0

    def reset(self, n_init: int, seed: int | None = None) -> dict:
        """Fresh episode: resamples the prior (new architectures/weights),
        rebuilds the quantile table against the new draw, and seeds `D_t`
        with `n_init` Sobol points (low-discrepancy, matching M0's spec for
        the initial design specifically -- unlike `BNNPrior.sample_episode`,
        which uses uniform random and is shared PFN-pretraining machinery,
        not BOEnv-specific). -> {"x_context": [B, n_init, d], "y_context": [B, n_init]}."""
        assert n_init <= self.budget, f"n_init ({n_init}) must not exceed budget ({self.budget})"
        self.prior.reset()
        self.quantile_table = QuantileTable(self.prior, **self.quantile_table_kwargs)

        sob = torch.quasirandom.SobolEngine(dimension=self.prior.d, scramble=True, seed=seed)
        x0 = sob.draw(n_init).to(self.prior.device)
        x0 = x0.unsqueeze(0).expand(self.prior.B, -1, -1)
        x0 = x0 * self.prior.active_dim_mask.unsqueeze(1)
        y0 = self.prior.evaluate(x0)

        self.x_context, self.y_context = x0, y0
        self.incumbent = y0.min(dim=1).values
        self.t = n_init
        return {"x_context": self.x_context, "y_context": self.y_context}

    def step(self, x_next: torch.Tensor) -> tuple[dict, torch.Tensor, bool]:
        """`x_next`: `[B, x_dim]`, one query per env for this step (no
        Python loop over envs -- `prior.evaluate` is already batched over
        `B`). -> `(D_{t+1}, g_{t+1}, done)`: `D_{t+1}` is `{"x_context",
        "y_context"}` including the new point, `g_{t+1}` is the RUNNING
        INCUMBENT's reward (monotone non-decreasing across an episode by
        construction -- the incumbent only improves), not the immediate
        observation's own reward. `done` is a single bool shared across the
        batch (`B` envs share one fixed `budget`, no per-env early
        stopping in this design)."""
        assert self.x_context is not None, "call reset() before step()"
        assert self.t < self.budget, f"step() called at t={self.t} >= budget={self.budget}"

        y_next = self.prior.evaluate(x_next.unsqueeze(1)).squeeze(1)
        self.x_context = torch.cat([self.x_context, x_next.unsqueeze(1)], dim=1)
        self.y_context = torch.cat([self.y_context, y_next.unsqueeze(1)], dim=1)
        self.incumbent = torch.minimum(self.incumbent, y_next)
        self.t += 1

        g = g_reward_minimize_from_table(self.quantile_table, self.incumbent, max_score=self.max_score)
        done = self.t >= self.budget
        return {"x_context": self.x_context, "y_context": self.y_context}, g, done


if __name__ == "__main__":
    """Rolls a random policy through a full episode and reports the running
    incumbent reward -- a sanity check that reset()/step() compose into a
    monotone trajectory that terminates exactly at budget, and the exact
    smoke-test command to reproduce this:
    uv run python -m anytimeacquisition.priors.bo_env
    """
    torch.manual_seed(0)
    B, d, n_init, budget = 4, 2, 3, 12

    prior = BNNPrior(batch_size=B, x_dim=d, seed=0, cache_dir=None, ecdf_n_draws=5, ecdf_samples_per_draw=100)
    env = BOEnv(prior, budget=budget, quantile_table_kwargs=dict(n_samples=50_000, n_grid=512, n_exact_tail=100, chunk_size=10_000))

    D_t = env.reset(n_init=n_init, seed=0)
    print(f"reset: x_context {tuple(D_t['x_context'].shape)}, y_context {tuple(D_t['y_context'].shape)}")

    g_prev = torch.zeros(B)
    while env.t < env.budget:
        x_next = torch.rand(B, d)
        D_t, g, done = env.step(x_next)
        assert (g >= g_prev - 1e-6).all(), "incumbent reward regressed within an episode"
        g_prev = g
        print(f"t={env.t:2d}  g={g.mean().item():.4f}  done={done}")

    print(f"final x_context {tuple(D_t['x_context'].shape)}, y_context {tuple(D_t['y_context'].shape)}")
