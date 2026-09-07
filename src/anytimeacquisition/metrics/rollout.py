"""Generic self-play rollout against a `BNNPrior` instance — a `policy_fn`
plugs into `rollout_episode`'s loop the same way for any acquisition policy
(random, classical GP baselines, a learned Q-head), so this stays the one
shared harness M7's baseline comparisons and evaluation code roll episodes
through, rather than each baseline reimplementing its own loop.

Extracted from the retired `trainer/exit_rollout.py` (ActionHead/BC-EXIT
imitation-learning line, superseded by `docs/ROADMAP.md`'s Q-head design) —
this keeps only the policy-agnostic rollout mechanics, none of the
BC-EXIT-specific branch labeling/buffer building that lived alongside them.
"""
import torch

from anytimeacquisition.priors.bnn import BNNPrior


def random_policy(x_context: torch.Tensor, y_context: torch.Tensor, x_dim: int) -> torch.Tensor:
    """Uniform-random next query, ignoring context entirely — the standard
    baseline every acquisition policy should beat."""
    B = x_context.shape[0]
    return torch.rand(B, x_dim)


def rollout_episode(
    prior: BNNPrior, n_init: int, n_steps: int, policy_fn=random_policy, noise: bool = True, reset: bool = True,
) -> dict:
    """One self-play episode against `prior` — resets it once (fresh
    architecture/weights) unless `reset=False`, then queries `policy_fn`
    for `n_steps`, each time observing `prior.evaluate(x_next, noise=noise)`
    and appending to the running context.

    -> {"x_context": [B, n_init+n_steps, x_dim], "y_context": [B, n_init+n_steps]}
    """
    if reset:
        prior.reset()
    x_context, y_context, _, _ = prior.sample_episode(n_train=n_init, n_test=0)

    for _ in range(n_steps):
        x_next = policy_fn(x_context, y_context, prior.d).unsqueeze(1)  # [B,1,x_dim]
        y_next = prior.evaluate(x_next, noise=noise)  # [B,1]
        x_context = torch.cat([x_context, x_next], dim=1)
        y_context = torch.cat([y_context, y_next], dim=1)

    return {"x_context": x_context, "y_context": y_context}


if __name__ == "__main__":
    from anytimeacquisition.metrics.inc_auc import log_incumbent_auc

    torch.manual_seed(0)
    prior = BNNPrior(batch_size=4, x_dim=2, seed=0)
    rollout = rollout_episode(prior, n_init=3, n_steps=10, policy_fn=random_policy)
    auc = log_incumbent_auc(rollout["y_context"]).mean().item()
    print(f"x_context {tuple(rollout['x_context'].shape)}  y_context {tuple(rollout['y_context'].shape)}")
    print(f"random_policy mean log-incumbent AUC (lower is better): {auc:.4f}")
