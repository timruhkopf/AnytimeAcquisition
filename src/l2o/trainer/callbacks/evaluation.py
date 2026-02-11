import numpy as np
import matplotlib.pyplot as plt

from src.l2o.trainer.callbacks.abstract import AbstractCallback
from src.l2o.model.rewards.area_under_incumbent_curve import AUICReward


class ValidationCallback(AbstractCallback):
    def __init__(self, env, visualize_every=99, num_to_plot=4):
        self.env = env
        self.fixed_seeds = env.current_seeds.clone()
        self.visualize_every = visualize_every
        self.num_to_plot = min(num_to_plot, env.num_envs)

    def on_policy_epoch_end(self, **kwargs):
        self.env.reset(seeds=self.fixed_seeds)
        # Using trainer.policy to get the interaction engine
        traj, _, _ = self.trainer.policy._run_vectorized_episode(self.env)

        # 1. Visualization Logic
        if (self.trainer.epoch +1) % self.visualize_every == 0:
            # Create a 2-row grid: Top for Trajectory, Bottom for Reward
            fig, axes = plt.subplots(
                nrows=3 if not isinstance(self.trainer.reward_manager, AUICReward) else 2,
                ncols=self.num_to_plot,
                figsize=(5 * self.num_to_plot, 10)
            )

            # Handle indexing for single column cases
            if self.num_to_plot == 1:
                traj_axes = [axes[0]]
                reward_axes = [axes[1]]
            else:
                traj_axes = axes[0]
                reward_axes = axes[1]


            # 2. Delegate Spatial Drawing to Env
            self.env.plot_trajectories(traj["obs"], traj_axes)

            # 3. Delegate Reward Drawing to Reward Manager
            for i in range(self.num_to_plot):
                # Pass single trajectory: (Seq, Dim)
                self.trainer.reward_manager.plot_reward(traj["obs"][i], reward_axes[i])


            if not isinstance(self.trainer.reward_manager, AUICReward):
                auic_axes = axes[2]
                for i in range(self.num_to_plot):
                    # Pass single trajectory: (Seq, Dim)
                    AUICReward.plot_auic(traj["obs"][i], auic_axes[i])




            plt.tight_layout()
            plt.show()