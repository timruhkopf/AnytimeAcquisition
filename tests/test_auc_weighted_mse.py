import pytest
from src.loss.auc import AUCWeightedMSELoss

import torch


def test_shifted_incumbents():
    loss_fn = AUCWeightedMSELoss()
    inc_indices = torch.tensor([[0, 0, 2, 2, 4, 4],
                                [0, 1, 1, 3, 4, 4]])
    shifted = loss_fn.shift_segments(inc_indices)
    expected = torch.tensor([[2, 2, 4, 4, 4, 4],
                             [1, 3, 3, 4, 4, 4]])
    assert torch.equal(shifted, expected), f"Expected {expected}, but got {shifted}"

    y = torch.tensor([[0.9, 1., 0.8, 0.9, 0.7, 0.8],
                      [0.6, 0.5, 0.55, 0.4, 0.35, 0.4]])

    target_y = torch.gather(y, 1, shifted)
