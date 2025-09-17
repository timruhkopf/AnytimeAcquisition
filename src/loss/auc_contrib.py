import torch
from torch import nn
# from functools import lru_cache

from src.loss.auc_alternatives import find_element_wise_optimal_trajectory


class AUCContributionLoss(nn.Module):

    # @lru_cache(maxsize=5)
    # def get_causal_mask(self, B, A, T, device) -> torch.Tensor:
    #     # Creates a lower-triangular matrix mask of shape (T, T)
    #     # with True in permissible attention positions.
    #     mask = torch.tril(torch.ones(T, T, dtype=torch.bool)).to(device)
    #     # Expand to (B, 1, T, T) or (B, T, T) depending on usage
    #     # Here just repeat mask for batch B
    #     return mask.view(1,1,T,T).expand(B, A, T, T)

    def get_inc_mask(self, inc_values):
        device = inc_values.device
        # create a mask for the incumbent positions
        B, A, T = inc_values.shape
        inc_mask = torch.cat([
            torch.ones(B, 1, 1, dtype=torch.bool, device=device),
            torch.diff(inc_values, dim=-1) < 0
        ],
            dim=-1
        ).to(device)
        return inc_mask

    def get_exploration_mask(self, inc_values):
        inc_mask = self.get_inc_mask(inc_values)
        exploration_mask = ~inc_mask
        return exploration_mask

    def forward(self, predictions, predictions_y_true, alternatives, padding_mask=None):
        if len(alternatives.shape) == 3:
            alternatives = alternatives.unsqueeze(1)
        B, A, T, D = alternatives.shape

        # find the incumbents
        pred_inc = torch.cummin(predictions_y_true, dim=-1)  # (B,T)
        pred_inc_values = pred_inc.values.unsqueeze(1).repeat(1, A, 1)  # (B,A,T)
        # pred_inc_indices = pred_inc.indices.unsqueeze(1).repeat(1, A, 1)  # (B,A,T)

        # TODO: consider using binned distribution and CE loss on y.
        predictions = predictions.unsqueeze(1).repeat(1, A, 1, 1)  # (B,A,T,D)

        # compute alternative incumbents
        alt_inc = torch.cummin(alternatives[..., -1], dim=-1)  # (B,A,T)
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

        # compute the coordinate differences
        # TODO: check sign!
        coord_diff = predictions - min_sequence
        coord_diff *= joint_mask.unsqueeze(-1).repeat(1, 1, 1, D)

        # fixme, the exploration bonus is intended to avoid collapse, but it likely is ill posed,
        #  because we try to fit a random sequence. instead, we should fixate on the degree to
        #  which we were able to reduce uncertainty in the predicted y; i.e. the CE/ KL / mse?
        #  on y in all future steps -- notice, that this is actually something we should consider
        #  for every single step
        # exploration_bonus = self.get_exploration_bonus(joint_mask, min_sequence, predictions).mean()
        # self.get_exploration_loss(predictions, predictions_y_true)

        # deselect the y value difference here and multiply with the contribution
        regret = (coord_diff[:, :, :T - 1, :-1] * auc.unsqueeze(-1).repeat(1, 1, 1, D - 1))

        # fixme: instead of mean on the regret components, take the weighted average based on
        #  the relative final regret
        final_regret = min_inc_values[..., -1].view(B, 1, 1, 1)

        return (regret * (final_regret / B)).sum()  # + exploration_bonus

    def get_exploration_bonus(self, inc_mask, min_sequence, pred_sequence):
        # TODO if we only have early incumbents, we may want to encourage denser signals, by
        #  adding ones to exploration steps, despite the lack of an incumbent change. This
        #  is just a safeguard against collapse when the exploration process is non-ideal
        exploration_mask = ~inc_mask
        _, _, T, D = pred_sequence.shape

        # todo check sign
        coord_diff = pred_sequence[..., :-1] - min_sequence[..., :-1]
        coord_diff *= exploration_mask.unsqueeze(-1).repeat(1, 1, 1, D - 1)
        return coord_diff / T

    # def get_exploration_loss(self, predictions, predictions_y_true):
    #     """
    #     Conceptually, we want to choose points that help us gain certainty for any future step.
    #     So we need to quantify the change in information gain for future steps and attribute it
    #     to the current position.
    #
    #     :return:
    #     """
    #     B, A, T, D = predictions.shape
    #     causal_mask = self.get_causal_mask(B, A, T, device=predictions.device)
    #     future = ~causal_mask
    #     pred_y = predictions[..., -1]
    #
    #     # FIXME: one would need to compute (lambda,y_train) (lambda,y_test) based on the causal
    #     #  sequence
    #     #  now the issue is that, while we can compute that for all elements in the batch;
    #     #  but we would need to do T-1 forward passes, because the pfn's attention would not work
    #     #  with the split varying splits
    #     #  load distribution would be even though due to constant size with train + test points
    #     # self.pfn()
    #
    #     # other than that this is computationally super inefficient,
    #     # it probably also will blow up, since there is an exponentially growing set of points to
    #     # choose from, making it excessively hard to interpret this loss signal.
    #
    #     # TODO make a separate experiment only using this -- a function that will give us the
    #     #  quickest path to gain the most information about the surface!
    #
    #     # TODO consider simplification, where we just look at the loss of the "past"'s ability
    #     #  to predict the immediate future

    #     # Maybe instead, we should just simply do a single step mse penalty for the y value
    #     # encouraging the model to be right about what it chooses in the future step; i.e. it
    #     # formulates its expectation. Maybe one can even try to tell the model about its failure?
    #     # which would go towards surprise exploration.

    def get_inc_time_delta(self, inc_indices, inc_mask):
        """
        Given a sequence, compute the time spent between incumbents
        :param inc_indices:
        :param inc_mask:
        :return:
        """
        B, A, T = inc_mask.shape

        # penalize only incumbent change positions
        inc_change_pos = inc_indices * inc_mask
        inc_change_pos[..., 0] = 1
        inc_B, inc_A, inc_T = torch.nonzero(inc_change_pos, as_tuple=True)

        d = torch.diff(inc_T)
        d[d < 0] = 1  # we have negative distances on the next incumbent (first step)
        # because the idx will be 0 again! Since it costs one token to acquire this, we assign 1

        inc_distances = torch.zeros(B, A, T)
        inc_distances[inc_B[:-1], inc_A[:-1], inc_T[:-1]] = d.float()
        inc_distances = inc_distances[..., :-1]

        return inc_distances

    def find_instantaneous_regret(self, target_seq, pred_seq):
        """
        Given two sequences, of which we know that one is smaller than the other,
        We can try to compute the rectangles that are the auc differences between
        incumbent changes of either sequence; where one edge is the improvement in y that we could
        have had, if at that moment in time we choose the better alternative for the coming
        steps up to the next incumbent change of either sequence.
        The other edge is the duration between the two incumbent changes.
        This is the regret of choosing the predicted incumbent over the best known incumbent at
        that point in time and sticking with it until we have the next (successful) exploitation.
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
        upper_y = torch.gather(pred_sequence[..., -1], dim=-1, index=joint_inc_indices)
        lower_y = torch.gather(min_sequence[..., -1], dim=-1, index=joint_inc_indices)
        upper_y_diff = torch.diff(upper_y, dim=-1)
        lower_y_diff = torch.diff(lower_y, dim=-1)
        y_diff = upper_y_diff - lower_y_diff

        # collect the time difference between the incumbents and normalize by n time steps
        delta_t = self.get_inc_time_delta(joint_inc_indices, joint_inc_mask) / T

        # compute the rectangles
        auc_diff = y_diff * delta_t

        return auc_diff, joint_inc_mask
