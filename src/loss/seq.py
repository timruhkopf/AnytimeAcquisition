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

        # get a normalized weighing on the batch items based on the "success" of that trajectory
        final_outcome = min_sequence[..., -1].min(dim=-1).values  # shape (B,)
        final_outcome = -(final_outcome - final_outcome.mean()) / (final_outcome.std() + 1e-9)

        diff =  nn.functional.mse_loss(
            predictions[:, 1:, :-1],
            min_sequence[:, :-1, :-1],
            reduction='none'
        ).sum(-1)

        if False:
            return (diff * final_outcome.unsqueeze(-1)).mean()
        else:
            return diff.mean()

class IncumbentLoss:
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
        inc_values, inc_indices = torch.cummin(min_sequence[..., -1], dim=1)

        exploration_mask = (inc_indices != torch.arange(
            y.size(1), device=y.device
        ).unsqueeze(0)).float()  # shape

        # Now that we know the incumbent trajectory, we can penalize the coordinates on the
        # incumbent position to be closer to that "optimal" position
        # Notice the shift by one index in the prediction --> we want the model to
        # be quicker than the "optimal" sequence that we have observed so far
        diff = nn.functional.mse_loss(
            torch.gather(predictions, dim=1, index=inc_indices.unsqueeze(
                -1).expand(-1, -1, D)),
            torch.gather(min_sequence, dim=1, index=inc_indices.unsqueeze(-1).expand(-1, -1, D)),
            reduction='none'
        )

        # we don't want to penalize the exploration steps, only the incumbent steps
        # deselect the y dimension!
        diff = (diff[..., :-1].sum(-1) * (1.0 - exploration_mask)).mean()

        return diff




