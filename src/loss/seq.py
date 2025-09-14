import torch
import torch.nn as nn


class SimpleSequenceReconstructionLoss:

    def __call__(self, predictions, X, targets):
        return nn.functional.mse_loss(predictions[..., :-1], X[..., :-1], reduction='mean')


class ImproveSequenceReconstructionLoss:

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



class MinimaSeekingSequenceReconstructionLoss:
    """
    Given the predicted sequence and the alternative sequence X,
    we construct a new sequence that at each time step takes the element-wise minimum
    based on the last dimension (y value).
    We then compute the MSE loss between the predicted sequence and this new sequence,
    but weight the loss by the final outcome of the new sequence.
    """

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

        final_outcome = min_sequence[..., -1].min(dim=-1).values  # shape (B,)
        # standardize
        final_outcome = -(final_outcome - final_outcome.mean()) / (final_outcome.std() + 1e-9)

        return (nn.functional.mse_loss(
            predictions[..., :-1],
            min_sequence[..., :-1],
            reduction='none'
        ).sum(-1) * final_outcome.unsqueeze(-1)).mean()
