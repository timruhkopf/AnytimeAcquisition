import torch
import numpy as np

class AUICReward:
    """Computes Area Under Incumbent Curve reward.

    This reward is the most natural metric for black-box optimization, as it is abundantly used in the BBO literature.
    It measures

    But it ultimately is also just a proxy for regret volume reduction. It is also a very sparse reward, that will
    change only when the incumbent changes and only reward exploitative behaviour. Luck shots early on leave the rest
    of the sequence with zero token-level rewards, which can make learning difficult.
    """

    def __call__(self, obs_traj):
        # obs_traj: (Batch, Seq, hp_dim + 1) where [:, :, -1] is Y
        y_vals = obs_traj[..., -1]

        # Cumulative minimum along time dimension
        incumbents, _ = torch.cummin(y_vals, dim=1)

        # Reward is negative incumbent (minimizing Y = maximizing -Y)
        # We return the full trajectory of rewards
        return -incumbents

    def plot_reward(self, obs_traj, ax):
        """
        Plots the raw values vs incumbent over time for a single trajectory.
        obs_traj: (Seq, dim + 1)
        """
        y_vals = obs_traj[:, -1].cpu().numpy()
        # Calculate incumbent locally for plotting
        incumbents = np.minimum.accumulate(y_vals)
        steps = np.arange(len(y_vals))

        # Plot raw Y values (the search process)
        ax.plot(steps, y_vals, color='gray', alpha=0.3, label='Raw Y')

        # Plot the Incumbent Curve (the step function)
        ax.step(steps, incumbents, where='post', color='blue', lw=2, label='Incumbent')

        # Shade the Area Under the Incumbent Curve
        ax.fill_between(steps, incumbents, ax.get_ylim()[0], step='post', alpha=0.1, color='blue')

        ax.set_title("AUIC Progression")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Value (Y)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize='small')