import torch

from anytimeacquisition.metrics.rollout import random_policy, rollout_episode
from anytimeacquisition.priors.bnn import BNNPrior


def test_rollout_episode_shapes_and_incumbent_matches_context():
    torch.manual_seed(0)
    prior = BNNPrior(batch_size=3, x_dim=2, seed=0)
    n_init, n_steps = 4, 6
    rollout = rollout_episode(prior, n_init=n_init, n_steps=n_steps, policy_fn=random_policy)

    assert rollout["x_context"].shape == (3, n_init + n_steps, 2)
    assert rollout["y_context"].shape == (3, n_init + n_steps)


def test_random_policy_ignores_context_and_stays_in_domain():
    x_context = torch.rand(5, 3, 2)
    y_context = torch.rand(5, 3)
    x_next = random_policy(x_context, y_context, x_dim=2)

    assert x_next.shape == (5, 2)
    assert (x_next >= 0).all() and (x_next <= 1).all()


def test_rollout_episode_reset_false_keeps_same_instance():
    torch.manual_seed(0)
    prior = BNNPrior(batch_size=2, x_dim=1, seed=0)
    prior.reset()
    depth_before = prior.depth.clone()
    rollout_episode(prior, n_init=2, n_steps=2, policy_fn=random_policy, reset=False)
    assert torch.equal(prior.depth, depth_before)
