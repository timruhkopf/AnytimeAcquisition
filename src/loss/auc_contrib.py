import torch
import torch.nn as nn
from functools import lru_cache
from src.utils.bar_distribution import BarDistribution


class AUCContributionLoss(nn.Module):

    def __init__(self, env):
        super().__init__()
        self.env = env
        self.borders = torch.linspace(0, 1, 100).to(env.device)
        self.criterion = BarDistribution(borders=self.borders, )

    @lru_cache(maxsize=5)
    def get_causal_mask(self, B, T, device) -> torch.Tensor:
        # Creates a lower-triangular matrix mask of shape (T, T)
        # with True in permissible attention positions.
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool)).to(device)
        # Expand to (B, 1, T, T) or (B, T, T) depending on usage
        # Here just repeat mask for batch B
        return mask.view(1, T, T).expand(B, T, T)

    def get_inc_mask(self, inc_values):
        device = inc_values.device
        # create a mask for the incumbent positions
        B, A, T = inc_values.shape
        inc_mask = torch.cat([
            torch.ones(B, 1, 1, dtype=torch.bool, device=device),
            # torch.diff(inc_values, dim=-1) < 0
            inc_values[..., :-1] > inc_values[..., 1:]
        ],
            dim=-1
        ).to(device)
        return inc_mask

    def get_exploration_mask(self, inc_values):
        inc_mask = self.get_inc_mask(inc_values)
        exploration_mask = ~inc_mask
        return exploration_mask

    def get_inc_time_delta(self, inc_indices, inc_mask):
        """
        Computes the time spent between incumbent changes in a sequence.

        Args:
            inc_indices (torch.Tensor): Indices of incumbent changes of shape (B, A, T), where:
                B = batch size,
                A = number of alternatives,
                T = time steps.
            inc_mask (torch.Tensor): Mask indicating incumbent positions of shape (B, A, T).

        Returns:
            torch.Tensor: A tensor of shape (B, A, T-1) representing the time spent between
            incumbent changes, normalized by the number of time steps.
        """
        B, A, T = inc_mask.shape

        # penalize only incumbent change positions
        inc_change_pos = inc_indices * inc_mask
        inc_change_pos[..., 0] = 1
        inc_B, inc_A, inc_T = torch.nonzero(inc_change_pos, as_tuple=True)

        d = torch.diff(inc_T)
        d[d < 0] = 1  # we have negative distances on the next incumbent (first step)
        # because the idx will be 0 again! Since it costs one token to acquire this, we assign 1

        inc_distances = torch.zeros(B, A, T, device=inc_mask.device)
        inc_distances[inc_B[:-1], inc_A[:-1], inc_T[:-1]] = d.float()
        inc_distances = inc_distances[..., :-1]

        return inc_distances

    def find_instantaneous_regret(self, target_seq, pred_seq):
        """
        Compute instantaneous AUC difference between two sequences.

        Computes the instantaneous regret between two sequences by calculating the differences
        in area under the curve (AUC) between incumbent changes of the sequences.

        We can try to compute the rectangles that are the auc differences between
        incumbent changes of either sequence; where one edge is the improvement in y that we could
        have had, if at that moment in time we choose the better alternative for the coming
        steps up to the next incumbent change of either sequence.
        The other edge is the duration between the two incumbent changes.
        This is the regret of choosing the predicted incumbent over the best known incumbent at
        that point in time and sticking with it until we have the next (successful) exploitation.

        Args:
           target_seq (tuple): A tuple containing:
               - min_sequence (torch.Tensor): The element-wise optimal sequence of shape (B, A, T, D).
               - min_inc_values (torch.Tensor): The cumulative minimum values along the time axis of shape (B, A, T).
               - _: Placeholder for unused values.
           pred_seq (tuple): A tuple containing:
               - pred_sequence (torch.Tensor): The predicted sequence of shape (B, A, T, D).
               - pred_inc_values (torch.Tensor): The cumulative minimum values along the time axis of shape (B, A, T).
               - _: Placeholder for unused values.

        Returns:
           tuple: A tuple containing:
               - auc_diff (torch.Tensor): The regret values as AUC differences of shape (B, A, T-1).
               - joint_inc_mask (torch.Tensor): A mask indicating the joint incumbent positions of shape (B, A, T).
        """
        min_sequence, min_inc_values, _ = target_seq
        pred_sequence, pred_inc_values, _ = pred_seq
        B, A, T = pred_inc_values.shape

        # collect the joint set of incumbent positions as base for the rectangle_calc
        pred_inc_mask = self.get_inc_mask(pred_inc_values)
        min_inc_mask = self.get_inc_mask(min_inc_values)
        joint_inc_mask = torch.logical_or(pred_inc_mask, min_inc_mask)

        # based on the joint mask collect the actual indices
        a = torch.arange(0, T, device=pred_inc_values.device).reshape(1, 1, T).repeat(B, A, 1)
        joint_inc_non_zero = a * joint_inc_mask
        joint_inc_indices = (joint_inc_non_zero).cummax(dim=-1).indices

        # collect the y value differences
        upper_y = torch.gather(pred_inc_values, dim=-1, index=joint_inc_indices)
        lower_y = torch.gather(min_sequence[..., -1], dim=-1, index=joint_inc_indices)

        y_diff = upper_y - lower_y


        # collect the time difference between the incumbents and normalize by n time steps
        delta_t = self.get_inc_time_delta(joint_inc_indices, joint_inc_mask) / T

        # compute the rectangles
        auc_diff = y_diff * delta_t

        return auc_diff, joint_inc_mask

    def forward(self, predictions, predictions_y_true, alternatives, padding_mask=None):
        """
        Computes the regret loss by comparing predicted sequences with alternative sequences
        and calculating the differences in area under the curve (AUC) between incumbent changes.

        Args:
            predictions (torch.Tensor): Predicted sequences of shape (B, T, D), where:
                B = batch size,
                T = time steps,
                D = feature dimensions.
            predictions_y_true (torch.Tensor): Ground truth values for predictions of shape (B, T).
            alternatives (torch.Tensor): Alternative sequences of shape (B, A, T, D), where:
                A = number of alternatives.
            padding_mask (torch.Tensor, optional): Mask to ignore padded values. Defaults to None.

        Returns:
            torch.Tensor: The computed regret loss as a scalar value.
        """
        if len(alternatives.shape) == 3:
            alternatives = alternatives.unsqueeze(1)
        B, A, T, D = alternatives.shape

        # find the incumbents
        pred_inc = torch.cummin(predictions_y_true, dim=-1)  # (B,T)
        pred_inc_values = pred_inc.values.unsqueeze(1).repeat(1, A, 1)  # (B,A,T)
        # pred_inc_indices = pred_inc.indices.unsqueeze(1).repeat(1, A, 1)  # (B,A,T)

        predictions = predictions.unsqueeze(1).repeat(1, A, 1, 1)  # (B,A,T,D)

        # compute alternative incumbents
        # alt_inc = torch.cummin(alternatives[..., -1], dim=-1)  # (B,A,T)
        # alt_inc_values = alt_inc.values  # (B,A,T)
        # alt_inc_indices = alt_inc.indices  # (B,A,T)

        # find better alternatives
        min_sequence, min_inc_values, min_inc_indices = (
            find_element_wise_optimal_trajectory(
                predictions, predictions_y_true.unsqueeze(1), alternatives
            ))

        target_seq = (min_sequence, min_inc_values, min_inc_indices)
        pred_seq = (predictions, pred_inc_values, pred_inc.indices)

        auc, joint_mask = self.find_instantaneous_regret(target_seq, pred_seq)

        auc_steps = pred_inc_values - min_inc_values
        pred_inc_mask = self.get_inc_mask(pred_inc_values)
        min_inc_mask = self.get_inc_mask(min_inc_values)
        joint_inc_mask = torch.logical_or(pred_inc_mask, min_inc_mask)


        # self.exploration_bonus(target_seq, pred_seq)

        # compute the coordinate differences
        # TODO: check sign!
        coord_diff = torch.nn.functional.mse_loss(predictions, min_sequence, reduction='none')
        coord_diff *= joint_inc_mask.unsqueeze(-1).repeat(1, 1, 1, D)

        # deselect the y value difference here and multiply with the contribution
        # regret = (coord_diff[:, :, :T - 1, :-1] * auc_steps.unsqueeze(-1).repeat(1, 1, 1, D - 1))
        penalize_y=True
        if penalize_y:
            regret = coord_diff * auc_steps.unsqueeze(-1).repeat(1, 1, 1, D)
        else:
            regret = coord_diff[..., :-1] * auc_steps.unsqueeze(-1).repeat(1, 1, 1, D - 1)



        # fixme: instead of mean on the regret components, take the weighted average based on
        #  the relative final regret (Batch-wise weighing)
        # final_regret = 1- min_inc_values[..., -1].view(B, 1, 1, 1)
        # return (regret * (final_regret / B)).sum()

        # weigh by overall auc difference (i.e. improvement potential)
        return (regret * auc_steps.sum(dim=-1).view(B, A, 1,1).repeat(1,1,T, regret.shape[
            -1])).mean()

def find_element_wise_optimal_trajectory(
        predictions, predictions_y_true, alternatives
):
    """
       Finds the element-wise optimal trajectory by comparing predictions with alternatives
       and selecting the better option for each time step.

       Args:
           predictions (torch.Tensor): Predicted sequences of shape (B, A, T, D), where:
               B = batch size,
               A = number of alternatives,
               T = time steps,
               D = feature dimensions.
           predictions_y_true (torch.Tensor): Ground truth values for predictions of shape (B, A, T).
           alternatives (torch.Tensor): Alternative sequences of shape (B, A, T, D).
           penalize_y_pred (bool, optional): If True, penalizes the predicted y-values by explicitly
               including them in the loss calculation. Defaults to False.

       Returns:
           tuple: A tuple containing:
               - min_sequence (torch.Tensor): The element-wise optimal sequence of shape (B, A, T, D).
               - inc_values (torch.Tensor): The cumulative minimum values along the time axis of shape (B, A, T).
               - inc_indices (torch.Tensor): The indices of the cumulative minimum values along the time axis of shape (B, A, T).
       """
    device = predictions.device
    pred_y = predictions_y_true
    y = alternatives[..., -1]
    B, A, T, D = alternatives.shape

    assert predictions.shape == (B, A, T, D)
    assert pred_y.shape == (B, A, T)

    # find the joint incumbent trajectory from
    mask = pred_y <= y  # shape (B, A, T), bool
    # expand to collect the entire observation vector (D)
    mask_expanded = mask.unsqueeze(-1).repeat(1, 1, 1, D)  # shape (B, A, T, D)

    p = torch.cat([predictions[..., :-1], pred_y.unsqueeze(-1)], dim=-1)

    min_sequence = torch.where(mask_expanded, p, alternatives)  # shape (B, A, T, D)
    inc_values, inc_indices = torch.cummin(min_sequence[..., -1], dim=-1)

    return min_sequence, inc_values, inc_indices
