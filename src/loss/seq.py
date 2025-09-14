import torch
import torch.nn as nn


class SimpleSequenceReconstructionLoss:

    def __call__(self, predictions, X, targets):
        return nn.functional.mse_loss(predictions[..., :-1], X[..., :-1], reduction='mean')


class SequenceReconstructionLoss:

    def __call__(self, predictions, X, targets):
        # collect the target coordinates based on the element minimum y value of the
        # two sequences
        pred_y = targets
        y = X[..., -1]
        B, T, D = X.shape
        # Create mask where X_last <= Z_last (True where X is minimum)
        mask = y <= pred_y  # shape (B, T), bool

        # Expand mask to all dims for broadcasting
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, D)  # shape (B, T, D)
        min_sequence = torch.where(mask_expanded, X, predictions)  # shape (B, T, D)

        return nn.functional.mse_loss(
            predictions[..., :-1],
            min_sequence[..., :-1],
            reduction='mean'
        )
