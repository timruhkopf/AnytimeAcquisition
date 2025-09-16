import torch
import torch.nn as nn


class SimpleSequenceReconstructionLoss:

    def __call__(self, predictions, X, targets):
        return nn.functional.mse_loss(predictions[..., :-1], X[..., :-1], reduction='mean')


class ImproveSequenceReconstructionLoss:

    def __call__(self, predictions, alternatives, predictions_y_true):
        # collect the target coordinates based on the element minimum y value of the
        # two sequences
        pred_y = predictions_y_true
        y = alternatives[..., -1]
        B, T, D = alternatives.shape
        # Create mask where X_last <= Z_last (True where X is minimum)
        mask = y <= pred_y  # shape (B, T), bool

        # Expand mask to all dims for broadcasting
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, D)  # shape (B, T, D)
        min_sequence = torch.where(mask_expanded, alternatives, predictions)  # shape (B, T, D)

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

    def __call__(self, predictions, predictions_y_true, alternatives):
        # collect the target coordinates based on the element minimum y value of the
        # two sequences
        pred_y = predictions_y_true
        y = alternatives[..., -1]
        B, T, D = alternatives.shape
        # Create mask where X_last <= Z_last (True where X is minimum)
        mask = y <= pred_y  # shape (B, T), bool

        # Expand mask to all dims for broadcasting
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, D)  # shape (B, T, D)
        min_sequence = torch.where(mask_expanded, alternatives, predictions)  # shape (B, T, D)

        # get a normalized weighing on the batch items based on the "success" of that trajectory
        final_outcome = min_sequence[..., -1].min(dim=-1).values  # shape (B,)
        final_outcome = 1 - final_outcome

        diff = nn.functional.mse_loss(
            predictions[:, 1:, :-1],
            min_sequence[:, :-1, :-1],
            reduction='none'
        ).sum(-1)

        if False:
            return (diff * final_outcome.unsqueeze(-1)).mean()
        else:
            return diff.mean()


class IncumbentLoss:
    def __init__(self, predict_future_incumbent=False):
        self.predict_future_incumbent = predict_future_incumbent

    def __call__(self, predictions, predictions_y_true, alternatives):
        """
        Given the predicted sequence and the alternative sequence X,
        we construct a new sequence that at each time step takes the element-wise minimum
        based on the last dimension (y value).
        We then compute the MSE loss between the predicted sequence and this new sequence,
        but only on the positions where the incumbent changes.
        This encourages the model to predict the incumbent positions more quickly.
        1. Find the incumbent trajectory from the element-wise minimum sequence.
        2. Create a mask for the positions where the incumbent changes.
        3. Compute the MSE loss between the predicted sequence and the incumbent sequence,
              but only on the positions where the incumbent changes.
        4. Optionally, shift the prediction indices by one to encourage predicting the
                next incumbent position.

        :param predictions: the model's predicted sequence (including hallucinated y values)
        :param alternatives: An alternative sequence (usually some generation strategy)
        :param predictions_y_true: the prediction's coordinates env evaluation (y values)
        :return:
        """
        # collect the target coordinates based on the element minimum y value of the
        # two sequences
        device = alternatives.device
        pred_y = predictions_y_true
        y = alternatives[..., -1]
        B, T, D = alternatives.shape

        # find the joint incumbent trajectory from
        mask = y <= pred_y  # shape (B, T), bool
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, D)  # shape (B, T, D)
        min_sequence = torch.where(mask_expanded, alternatives, predictions)  # shape (B, T, D)
        inc_values, inc_indices = torch.cummin(min_sequence[..., -1], dim=1)

        # create a mask for the incumbent positions
        inc_mask = torch.cat([
            torch.ones(B, 1, dtype=torch.bool, device=device),
            torch.diff(inc_values, dim=1) < 0
        ],
            dim=1
        ).to(device)

        exploration_mask = ~inc_mask

        if self.predict_future_incumbent:
            train_expl_mask = torch.roll(exploration_mask, shifts=-1, dims=1)
            train_expl_mask[:, -1] = 1  # last element cannot be trained
            target_expl_mask = exploration_mask
            target_expl_mask[:, 0] = 1  # first element cannot be target

            train_mask = inc_mask.roll(shifts=-1, dims=1)
            train_mask[
                :, -1] = 0  # last element cannot be trained, since we don't have a future incumbent for it
            target_mask = inc_mask
            target_mask[:, 0] = 0

            diff_collapse = nn.functional.mse_loss(
                predictions[..., :-1][train_mask],
                min_sequence[..., :-1][target_mask],
                reduction='none'
            )
            # if the prediction is close to the subsequent incumbent, we want them to collapse
            diff_collapse = (1 - diff_collapse)

            # cap the collapse loss to those subsequent predictions that are in a distance of 0.3
            # this way, we don't encourage collapsing when incumbents are far apart
            diff_collapse[diff_collapse < 0.95] = 0. # we want to squeeze them

            collapse_loss = diff_collapse.mean()



        else:
            collapse_loss = 1.


        train_expl_mask = exploration_mask
        target_expl_mask = exploration_mask

        train_mask = inc_mask
        target_mask = inc_mask

        diff = nn.functional.mse_loss(
            predictions[..., :-1][train_mask],
            min_sequence[..., :-1][target_mask],
            reduction='none'
        )

        # diff2 = - nn.functional.mse_loss(
        #     predictions[..., :-1][exploration_mask],
        #     X[..., :-1][exploration_mask],
        #     reduction='none'
        # )

        # Now that we know the incumbent trajectory, we can penalize the coordinates on the
        # incumbent position to be closer to that "optimal" position
        # Notice the shift by one index in the prediction --> we want the model to
        # be quicker than the "optimal" sequence that we have observed so far
        # diff = nn.functional.mse_loss(
        #     torch.gather(predictions, dim=1, index=prediction_idx.unsqueeze(-1).expand(-1, -1, D)),
        #     torch.gather(  # incumbent targets
        #         min_sequence, dim=1,
        #         index=target_idx.unsqueeze(-1).expand(-1, -1, D)
        #     ),
        #     reduction='none'
        # )

        # batch weighting; to encourage finding better minima by weighing examples
        # with better minima higher.
        # FIXME: Notice, that inbetween examples, some env instantiations may get lower weights,
        #  if their dynamic range (i.e. minima) are not as low as in other environments!
        #  we should maybe normalize this per batch (same env)?
        # weight = (1 - min_sequence[..., -1].min(dim=-1).values).unsqueeze(-1).expand(-1, T)
        # weight = weight[train_mask]

        return diff# + collapse_loss



class AUCIncumbentPenaltyLoss:
    def __call__(self, predictions, predictions_y_true, alternatives):
        device = alternatives.device
        pred_y = predictions_y_true
        y = alternatives[..., -1]
        B, T, D = alternatives.shape

        # find the joint incumbent trajectory from
        mask = y <= pred_y  # shape (B, T), bool
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, D)  # shape (B, T, D)
        min_sequence = torch.where(mask_expanded, alternatives, predictions)  # shape (B, T, D)
        inc_values, inc_indices = torch.cummin(min_sequence[..., -1], dim=1)

        # create a mask for the incumbent positions
        # inc_mask = torch.cat([
        #     torch.ones(B, 1, dtype=torch.bool, device=device),
        #     torch.diff(inc_values, dim=1) < 0
        # ],
        #     dim=1
        # ).to(device)
        #
        # exploration_mask = ~inc_mask

        # train_expl_mask = exploration_mask
        # target_expl_mask = exploration_mask
        #
        # train_mask = inc_mask
        # target_mask = inc_mask
        #
        # diff = nn.functional.mse_loss(
        #     predictions[..., :-1][train_mask],
        #     min_sequence[..., :-1][target_mask],
        #     reduction='none'
        # )

        # let us compute the weight based on the trapezoidal area difference
        # between the predicted incumbent and the actual incumbent curve
        pred_inc_values, _ = torch.cummin(predictions[..., -1], dim=1)
        contributions =  pred_inc_values - inc_values


        return contributions.mean()

