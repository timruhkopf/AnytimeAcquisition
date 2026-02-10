import torch

class AUICReward:
    """Computes Area Under Incumbent Curve reward.

    This reward is the most natural metric for black-box optimization, as it is abundantly used in the BBO literature.
    It measures

    But it ultimately is also just a proxy for regret volume reduction. It is also a very sparse reward, that will
    change only when the incumbent changes and only reward exploitative behaviour. Luck shots early on leave the rest
    of the sequence with zero token-level rewards, which can make learning difficult.
    """

    def __call__(self, obs_traj):
        # obs_traj: (Seq, Batch, 3) where [:, :, 2] is Y
        y_vals = obs_traj[..., -1]

        # Cumulative minimum along time dimension
        incumbents, _ = torch.cummin(y_vals, dim=0)

        # Reward is negative incumbent (minimizing Y = maximizing -Y)
        # We return the full trajectory of rewards
        return -incumbents