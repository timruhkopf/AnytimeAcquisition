import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from functools import lru_cache

from src.loss.auc_alternatives import find_element_wise_optimal_trajectory
from src.utils.bar_distribution import BarDistribution


class ECDF(torch.nn.Module):
    """
    CC: https://discuss.pytorch.org/t/cumulative-distribution-function-of-a-tensor-cdf/64613/2
    """

    def __init__(self, x, side='right'):
        super(ECDF, self).__init__()

        if side.lower() not in ['right', 'left']:
            msg = "side can take the values 'right' or 'left'"
            raise ValueError(msg)
        self.side = side

        if len(x.shape) != 1:
            msg = 'x must be 1-dimensional'
            raise ValueError(msg)

        x = x.sort()[0]
        nobs = len(x)
        y = torch.linspace(1. / nobs, 1, nobs, device=x.device)

        self.x = torch.cat((torch.tensor([-torch.inf], device=x.device), x))
        self.y = torch.cat((torch.tensor([0], device=y.device), y))
        self.n = self.x.shape[0]

    def forward(self, time):
        tind = torch.searchsorted(self.x, time, side=self.side) - 1
        return self.y[tind]


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
        # self.exploration_bonus(target_seq, pred_seq)

        # compute the coordinate differences
        # TODO: check sign!
        coord_diff = torch.nn.functional.mse_loss(predictions, min_sequence, reduction='none')
        coord_diff *= joint_mask.unsqueeze(-1).repeat(1, 1, 1, D)

        # fixme, the exploration bonus is intended to avoid collapse, but it likely is ill posed,
        #  because we try to fit a random sequence. instead, we should fixate on the degree to
        #  which we were able to reduce uncertainty in the predicted y; i.e. the CE/ KL / mse?
        #  on y in all future steps -- notice, that this is actually something we should consider
        #  for every single step
        # exploration_bonus = self.get_exploration_bonus(joint_mask, min_sequence, predictions).mean()
        # self.get_exploration_loss(predictions, predictions_y_true)
        exploration_failure = self.get_exploration_loss(
            target_seq, pred_seq, predictions_y_true
        )

        return (
                exploration_failure.unsqueeze(-1).repeat(1, 1, 1, D - 1) \
                * coord_diff[..., :-1]
        ).mean()

        # deselect the y value difference here and multiply with the contribution
        regret = (coord_diff[:, :, :T - 1, :-1] * auc.unsqueeze(-1).repeat(1, 1, 1, D - 1))

        # fixme: instead of mean on the regret components, take the weighted average based on
        #  the relative final regret
        final_regret = min_inc_values[..., -1].view(B, 1, 1, 1)
        return (regret * (final_regret / B)).sum()  # + exploration_bonus

    # def get_exploration_bonus(self, inc_mask, min_sequence, pred_sequence):
    #     # TODO if we only have early incumbents, we may want to encourage denser signals, by
    #     #  adding ones to exploration steps, despite the lack of an incumbent change. This
    #     #  is just a safeguard against collapse when the exploration process is non-ideal
    #     exploration_mask = ~inc_mask
    #     _, _, T, D = pred_sequence.shape
    #
    #     # todo check sign
    #     coord_diff = pred_sequence[..., :-1] - min_sequence[..., :-1]
    #     coord_diff *= exploration_mask.unsqueeze(-1).repeat(1, 1, 1, D - 1)
    #     return coord_diff / T

    def get_exploration_loss(self, target_seq, pred_seq, predictions_y_true):
        min_sequence, min_inc_values, _ = target_seq
        pred_sequence, pred_inc_values, _ = pred_seq
        B, A, T = pred_inc_values.shape
        device = predictions_y_true.device

        # meshgrid cover the space of the env, bounds (0,1) for all dimensions
        lower, upper = 0, 1
        n = 100

        # Collect the y values of the env on a grid, so we can compute the (empirical) cdf of
        # improvements over a given incumbent. This will also avoid the need to live in a
        # normalized environment (y \in [0,1])
        x = torch.linspace(lower, upper, n)
        y = torch.linspace(lower, upper, n)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        grid = torch.stack([X, Y], dim=-1).view(-1, 2)  # (n*n, 2)
        grid = grid.to(device)

        # collect the function values
        with torch.no_grad():
            z = self.env.evaluate(grid)

        # calculate the future probability of improvement;
        # How much area in X is still better than the incumbent?
        # i.e. if i sampled at x at random what was the likelihood of an improvement?
        #  this is basically the cdf of the empirical distribution of the grid function values
        #  evaluated at the current incumbent value
        pred_ecdf = ECDF(z)(predictions_y_true).unsqueeze(1).repeat(1, A, 1)
        min_ecdf = ECDF(z)(min_inc_values)

        step_regret_reduction = pred_ecdf - min_ecdf
        step_regret_reduction[step_regret_reduction < 0] = 0

        target_y = mean_below_thresholds(values=z, thresholds=min_inc_values.view(-1)).reshape(B, A,
                                                                                               -1)

        # We can calculate the estimated volume
        regret_dists = batched_binned_distributions(
            values=z,
            thresholds=predictions_y_true.view(-1),
            bin_edges=self.borders
        )
        regret_dists = regret_dists.reshape(B, A, T, -1).float()
        regret_dists /= regret_dists.sum(dim=-1).unsqueeze(-1).repeat(1, 1, 1, len(self.borders) - 1)  #
        # grid.shape[0] # dividing by grid will give us the actual volume.

        regret_dists += 1e-9
        regret_dist_logits = torch.log(regret_dists)
        pred_nll = self.criterion(logits=regret_dist_logits, y=target_y)

        # We can calculate the volume under the better alternative
        regret_dists = batched_binned_distributions(
            values=z,
            thresholds=min_inc_values.view(-1),
            bin_edges=self.borders
        )
        regret_dists = regret_dists.reshape(B, A, T, -1).float()
        regret_dists /= regret_dists.sum(dim=-1).unsqueeze(-1).repeat(1, 1, 1,
                                                                      len(self.borders) - 1)  #
        # grid.shape[0] # dividing by grid will give us the actual volume.


        # get the logits
        regret_dists += 1e-9
        regret_dist_logits = torch.log(regret_dists)
        target_nll = self.criterion(logits=regret_dist_logits, y=target_y)

        # loss = (target_nll - pred_nll)
        # loss[loss<0] *= 0

        # inc_mask = self.get_inc_mask(min_inc_values)
        # exploration_mask = ~inc_mask

        # now calculate the per calculate the per step nll improvement to decide on how much we
        # want to pull towards that solution for exploration steps

        # plot_distributions_colored(regret_dists[2, 0].cpu().numpy(), bin_centers=self.borders[
        #     :-1].cpu().numpy())

        # TODO consider expected regret; i.e. given the points that still satisfy the contour of
        #  an improvement, what is the expected improvement; i.e. the mean of the
        #  points that are better than the incumbent. This would allow us to quantify the
        #  expected regret. At every step we'd need to calculate the average over all points
        #  that are within the contour of the incumbent. This is probably expensive.

        # FIXME: notice how both the PI and EI are not reduced by an exploration step just yet,
        #  so the information is basically the same as the incumbent auc!

        # if we had a GP, knowing and knowing the lengthscale, we could compute the expected
        # improvement given the informational state; we kind of try to estimate the expected
        # improvement, which depends on the lengthscale and the uncertainty estimates that
        # we derive off of that. 

        return step_regret_reduction

        diff_y = torch.nn.functional.mse_loss(pred_sequence[:, 0, :, -1], predictions_y_true,
                                              reduction='none')

        # idx = torch.arange(0, T, device=device).view(1, 1, T ).repeat(B, T, 1)

        causal_mask = self.get_causal_mask(B, T, device=device)

        attributions = torch.zeros(B, T, T, device=device)
        attributions[causal_mask] = diff_y.unsqueeze(1).repeat(1, T, 1)[causal_mask]
        # batch losses
        return attributions.mean(dim=(1, 2))

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

        inc_distances = torch.zeros(B, A, T, device=inc_mask.device)
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


def mean_below_thresholds(values: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
    """
    Compute the mean of values below each threshold.
    :param values:
    :param thresholds:
    :return:
    """
    # values: shape (N,)
    # thresholds: shape (M,) descending
    values = values.unsqueeze(0)  # shape (1, N)
    thresholds = thresholds.unsqueeze(1)  # shape (M, 1)
    mask = values < thresholds  # shape (M, N)
    n_selected = mask.sum(dim=1)  # number of selections per threshold (M,)
    sums = (mask * values).sum(dim=1)  # sum of selected values (M,)
    means = torch.where(n_selected > 0, sums / n_selected, torch.zeros_like(sums))
    return means



def batched_binned_distributions(values, thresholds, bin_edges):
    """
    Compute histograms of values below each threshold, binned by bin_edges.
    :param values: shape (N,)
    :param thresholds: shape (M,)
    :param bin_edges: shape (B+1,)
    :return: histograms of shape (M, B)
    """

    # Broadcast values and thresholds
    values = values.unsqueeze(0)  # (1, N)
    thresholds = thresholds.unsqueeze(1)  # (M, 1)
    mask = values < thresholds  # (M, N)

    # Bucketize for bins (returns indices in 0...B-1)
    # Expand to (1, N) for batch, will be broadcast for thresholds
    bin_indices = torch.bucketize(values, bin_edges) - 1  # (1, N)

    num_bins = len(bin_edges) - 1
    num_thresholds = len(thresholds)

    # Broadcast mask and bin_indices
    hists = torch.zeros((num_thresholds, num_bins), dtype=torch.long, device=values.device)
    for b in range(num_bins):
        # For bin b, mask for elements assigned to bin b and values < threshold
        in_bin = (bin_indices == b)
        selected = mask & in_bin
        hists[:, b] = selected.sum(dim=1)
    return hists



def plot_distributions_colored(tensor, bin_centers=None, cmap_name='viridis', show_colorbar=True,
                               figsize=(10, 6)):
    tensor = np.array(tensor)  # Accept PyTorch or numpy
    num_distributions, num_bins = tensor.shape

    if bin_centers is None:
        bin_centers = np.arange(num_bins)

    cmap = plt.get_cmap(cmap_name)
    colors = [cmap(i / (num_distributions - 1)) for i in range(num_distributions)]

    fig, ax = plt.subplots(figsize=figsize)
    for i in range(num_distributions):
        ax.plot(bin_centers, tensor[i], color=colors[i], alpha=0.8)

    ax.set_xlabel('Bin')
    ax.set_ylabel('Probability')
    ax.set_title('All distributions (colored by index)')

    if show_colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=plt.Normalize(vmin=0, vmax=num_distributions - 1))
        sm.set_array([])  # To silence warning
        fig.colorbar(sm, ax=ax, label='Distribution index')

    plt.tight_layout()
    plt.show()
