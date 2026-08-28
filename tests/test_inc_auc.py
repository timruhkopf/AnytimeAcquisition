import math

import torch

from anytimeacquisition.metrics.inc_auc import (
    incumbent_trajectory,
    log_incumbent_auc,
    log_incumbent_stepwise_reward,
)


def test_log_incumbent_auc_matches_hand_computed_value():
    y = torch.tensor([0.5, 0.3, 0.4, 0.1])
    # incumbents (minimize): 0.5, 0.3, 0.3, 0.1
    expected = math.log(0.5) + math.log(0.3) + math.log(0.3) + math.log(0.1)
    auc = log_incumbent_auc(y)
    assert torch.allclose(auc, torch.tensor(expected), atol=1e-6)


def test_stepwise_reward_telescopes_to_total_log_incumbent_improvement():
    torch.manual_seed(0)
    y = torch.rand(4, 20) + 1e-3  # batched trajectories, kept away from 0
    inc = incumbent_trajectory(y)
    r = log_incumbent_stepwise_reward(y)
    total = torch.log(inc[:, 0]) - torch.log(inc[:, -1])
    assert torch.allclose(r.sum(dim=-1), total, atol=1e-5)


def test_stepwise_reward_strictly_positive_under_strictly_improving_trajectory():
    y = torch.tensor([0.9, 0.7, 0.5, 0.3, 0.1])
    r = log_incumbent_stepwise_reward(y)
    assert (r > 0).all()


def test_log_incumbent_auc_is_lower_for_a_dominating_trajectory():
    # y_better is <= y_worse at every step, strictly less at some -- its
    # incumbent trajectory dominates, so its (lower-is-better) AUC must too.
    y_better = torch.tensor([0.5, 0.3, 0.2, 0.1])
    y_worse = torch.tensor([0.5, 0.4, 0.3, 0.2])
    assert log_incumbent_auc(y_better) < log_incumbent_auc(y_worse)


def test_stepwise_reward_is_zero_on_steps_that_dont_improve_the_incumbent():
    y = torch.tensor([0.5, 0.6, 0.7, 0.2])  # steps 1, 2 don't beat incumbent 0.5
    r = log_incumbent_stepwise_reward(y)
    assert r[0] == 0.0
    assert r[1] == 0.0
    assert r[2] > 0.0
