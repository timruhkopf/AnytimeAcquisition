import torch
from torch import nn


class AUCAlternativesLoss(nn.Module):
    def __init__(self):
        super(AUCAlternativesLoss, self).__init__()

        self.penalize_y_pred = False

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

    def _calc_incumbent_auc(self, min_inc_values, min_inc_indices):
        """
        On the ideal trajectory, compute, what the AUC contributions of the incumbents are.
        Notice, that we don't have the individual contributions of the predicted incumbents as
        difference to this AUC
        :param min_inc_values:
        :param min_inc_indices:
        :return:
        """
        B, A, T = min_inc_values.shape
        # penalize only incumbent change positions to encourage exploration
        inc_mask = self.get_inc_mask(min_inc_values)
        inc_change_pos = min_inc_indices * inc_mask
        inc_change_pos[..., 0] = 1
        inc_B, inc_A, inc_T = torch.nonzero(inc_change_pos, as_tuple=True)
        # fixme: when incumbent is very early during training, we want the model to recover
        # so we add random positions with an auc step=1 respectively
        # fixme: check whether prepeding is correct here
        d = torch.diff(inc_T)
        d = torch.cat([torch.tensor([1]), d], dim=0)
        d[
            d < 0] = 1  # we have negative distances on the next incumbent (first step) because the idx
        # will be 0 again! Since it costs one token to acquire this, we assign 1

        inc_distances = torch.zeros(B, A, T)
        inc_distances[inc_B, inc_A, (inc_T)] = d.float()

        diff_auc = (-1) * torch.diff(min_inc_values, dim=-1)
        # prepend the initial condition cost (0-step generation)
        diff_auc = torch.cat([min_inc_values[..., 0].unsqueeze(-1), diff_auc], dim=-1)
        diff_auc *= inc_mask

        return diff_auc



    def forward(self, predictions, predictions_y_true, alternatives, padding_mask=None):
        """

        :param predictions: (B,T,D) tensor, where B is batch size, T is time steps, the first D-1
         dimensions are the x coordinates of the trajectory, the last dimension an unused one
        :param predictions_y_true: (B,T) tensor, with the ground truth y value associated to each
         coordinate in predictions
        :param alternatives: (B,A,T,D) tensor, which contains A alternative trajectories for each
         trajectory in predictions, with the same format as predictions, except that the last
         dimension is the true y coordinate associated to each coordinate of the trajectory.
        :param paddin_mask: (B,A,T) tensor, with 1 for valid coordinates and 0 for padded ones

        :return: scalar tensor.
        """
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

        diff_auc = self._calc_incumbent_auc(min_inc_values, min_inc_indices)

        inc_mask = self.get_inc_mask(min_inc_values)

        if padding_mask is None:
            padding_mask = torch.ones((B, A, T), device=predictions.device)

        diff_x = predictions[..., :-1] - min_sequence[..., :-1]  # alternatives[..., :-1]
        diff_x= diff_x * padding_mask.unsqueeze(-1).repeat(1, 1, 1, D - 1)  # (B,A,T,D-1)
        diff_x_inc = diff_x * inc_mask.unsqueeze(-1).repeat(1, 1, 1, D - 1)

        # calculate the incumbent differences at the incumbent position, just as single step cost
        # between incumbents.
        # diff_y = (pred_inc_values * padding_mask * inc_mask - min_inc_values * padding_mask *
        #           inc_mask)
        # diff_auc = diff_y.sum(dim=-1)  # (B,A)
        diff_x_inc = diff_x_inc.sum(dim=(-2, -1))

        # exploration penalty (one step = 1 token)
        exploration_penalty = diff_x * (~inc_mask).unsqueeze(-1).repeat(1, 1, 1, D - 1)
        return (diff_auc * diff_x_inc).mean() + exploration_penalty.mean()



def find_element_wise_optimal_trajectory(predictions, predictions_y_true, alternatives,
                                         penalize_y_pred=False):
    device = predictions.device
    pred_y = predictions_y_true
    y = alternatives[..., -1]
    B, A, T, D = alternatives.shape

    assert predictions.shape == (B, A, T, D)
    assert pred_y.shape == (B, A, T)

    # find the joint incumbent trajectory from
    mask = pred_y >= y  # shape (B, A, T), bool
    # expand to collect the entire observation vector (D)
    mask_expanded = mask.unsqueeze(-1).repeat(1, 1, 1, D)  # shape (B, A, T, D)

    if penalize_y_pred:
        # Consider Explicitly applying a loss on the predicted y value would mean instead of
        #  p we had predictions in here!
        p = torch.cat([predictions[..., :-1], pred_y.unsqueeze(-1)], dim=-1)
    else:
        p = predictions
    min_sequence = torch.where(mask_expanded, p, alternatives) # shape (B, A, T, D)
    inc_values, inc_indices = torch.cummin(min_sequence[..., -1], dim=-1)

    return min_sequence, inc_values, inc_indices

if __name__ == '__main__':
    # B, A, T, D = 2, 3, 4, 2
    # predictions = torch.randn((B, T, D))
    # predictions_y_true = torch.randn((B, T))
    # alternatives = torch.randn((B, A, T, D))
    # alternatives[..., -1] = torch.randn((B, A, T))
    # padding_mask = torch.ones((B, A, T), dtype=torch.bool)
    # for b in range(B):
    #     for a in range(A):
    #         pad_len = torch.randint(0, T // 2, (1,)).item()
    #         if pad_len > 0:
    #             padding_mask[b, a, -pad_len:] = 0
    #
    # criterion = AUCAlternativesLoss()
    # loss = criterion(predictions, predictions_y_true, alternatives, padding_mask)

    # ---- toy example ----

    import matplotlib.pyplot as plt

    B, A, T, D = 1, 1, 8, 2
    # predictions y ~ U-shaped
    pred_y_true = torch.tensor([[5., 4., 3., 4., 5., 6., 7., 8.]])  # shape (1,T)
    preds = torch.stack([torch.arange(T).float(), torch.zeros(T)], dim=-1).unsqueeze(0)  # x-axis
    alts = preds.unsqueeze(1).clone()
    alts[..., -1] = torch.tensor(
        [[[6., 2., 4., 3., 4., 7., 6., 5.]]])  # one alternative y trajectory

    criterion = AUCAlternativesLoss()
    min_seq, min_values, _, inc_mask = criterion.find_element_wise_optimal_trajectory(
        preds.unsqueeze(1), pred_y_true.unsqueeze(1), alts
    )

    true_min_seq = torch.tensor([[5, 2, 3, 3, 4, 6, 6, 5]], dtype=torch.float32).unsqueeze(-1)
    assert torch.equal(min_seq[..., -1].flatten(), true_min_seq[..., -1].flatten())
    # unwrap to numpy
    x = preds[0, :, 0].numpy()
    y_pred = pred_y_true[0].numpy()
    y_alt = alts[0, 0, :, -1].numpy()
    y_min = min_seq[0, 0, :, -1].detach().numpy()

    # ---- plot ----
    plt.figure(figsize=(8, 5))
    plt.plot(x, y_pred, marker="o", label="Prediction y_true")
    plt.plot(x, y_alt, marker="s", linestyle="--", label="Alternative")
    plt.plot(x, y_min, marker="x", linestyle="-.", color="k", label="Elementwise min sequence")
    plt.title("Trajectory vs Alternative with Elementwise Minimum")
    plt.xlabel("timestep")
    plt.ylabel("y value")
    plt.legend()
    plt.grid(True)
    plt.show()
