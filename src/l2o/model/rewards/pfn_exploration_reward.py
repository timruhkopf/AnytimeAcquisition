import torch
import numpy as np

def sobol_monitor_generator(n, dimension):
    sobol = torch.quasirandom.SobolEngine(dimension=dimension, scramble=True)
    return sobol.draw(n)


def load_pfn_model():
    import pfns4bo

    # Import fix:
    #   File "/home/ruhkopf/PycharmProjects/AnytimeAcquisition/.venv/lib/python3.10/site-packages/torch/serialization.py", line 1549, in load
    #     return _load(
    #   File "/home/ruhkopf/PycharmProjects/AnytimeAcquisition/.venv/lib/python3.10/site-packages/torch/serialization.py", line 2143, in _load
    #     result = unpickler.load()
    #   File "/home/ruhkopf/PycharmProjects/AnytimeAcquisition/.venv/lib/python3.10/site-packages/torch/serialization.py", line 2132, in find_class
    #     return super().find_class(mod_name, name)
    #   File "/home/ruhkopf/.pycharm_helpers/pydev/_pydev_bundle/pydev_import_hook.py", line 21, in do_import
    #     module = self._system_import(name, *args, **kwargs)
    #   File "/home/ruhkopf/PycharmProjects/AnytimeAcquisition/.venv/lib/python3.10/site-packages/pfns4bo/transformer.py", line 9, in <module>
    #     from .layer import TransformerEncoderLayer, _get_activation_fn
    #   File "/home/ruhkopf/.pycharm_helpers/pydev/_pydev_bundle/pydev_import_hook.py", line 21, in do_import
    #     module = self._system_import(name, *args, **kwargs)
    #   File "/home/ruhkopf/PycharmProjects/AnytimeAcquisition/.venv/lib/python3.10/site-packages/pfns4bo/layer.py", line 5, in <module>
    #     from torch.nn.modules.transformer import _get_activation_fn, Module, Tensor, Optional, MultiheadAttention, Linear, Dropout, LayerNorm
    # ImportError: cannot import name 'Optional' from 'torch.nn.modules.transformer' (/home/ruhkopf/PycharmProjects/AnytimeAcquisition/.venv/lib/python3.10/site-packages/torch/nn/modules/transformer.py)

    import torch.nn.modules.transformer
    import typing
    import torch

    # Manually inject the missing names into the module pfns4bo is looking at
    torch.nn.modules.transformer.Optional = typing.Optional
    torch.nn.modules.transformer.Tensor = torch.Tensor

    # Now we can load the model without import errors
    return torch.load(pfns4bo.bnn_model, weights_only=False)


class PFNExplorationReward:
    """
    Core idea: Use the PFN under the current horizon to predict the current ppd estimate of test points
    (incl. e.g. Sobol anchor points). Doing so under varying horizons allows us to compute the
    information gain / variance reduction. Weighing the variance reduction by the quantile of the actual value for that location
    incentives targeted exploration over mere coverage. It also considers the current state of optimization, so
    the optimal action depends on the current horizon.

        Implementation Ideas:
        1. Use padding to batch parallelize the varying  with a fixed test set on the monitor points.
        This will however blow up the batch by T x T items The padding will require the compute only to be masked.
        Meaning this is computationally heavy, while factually still correct.
        TODO 2. Alternatively, We can do T key-value cached forward passes, that cache the training set up to the current horizon.
         Meaning, that the main computational cost lies in the T forward passes and the M monitor points
    """

    def __init__(self, pfn_model, device, monitor_sampler=None, **kwargs):
        self.pfn = pfn_model.to(device)
        self.pfn.eval()

        if monitor_sampler is None:
            monitor_sampler = lambda: sobol_monitor_generator(256, 2)  # Default: 10 Sobol points in 2D

        self.sample_monitor_points = monitor_sampler
        self.device = device

        self.env = None  # will be set externally on trainer init

    def __call__(self, obs_traj):
        B, T, D = obs_traj.shape

        # 1. (collect global support points) --------------------------------
        # Sample monitor points once per rollout call
        test_x = self.sample_monitor_points().to(self.device)  # (M, 2)
        test_x = test_x.unsqueeze(1).repeat(1, B, 1)
        test_y = self.env.evaluate(test_x)  # (Batch, M, 1)

        # permute to meet obs_traj (B, T, D) format
        # fixme: check that the PFN expects (Batch, Seq_Len, Dim) format and not (Seq_Len, Batch, Dim) - this is a common source of bugs
        # test_x = test_x.permute(1, 0, 2)  # (num_envs, M, 2)
        # test_y = test_y.permute(1, 0, 2)  # (num_envs, M, 1)


        # 2. (collect nll differences per horizon) --------------------------------

        obs_traj = obs_traj.permute(1, 0, 2) # now (T, B, D)
        # Let us for now do the expensive version:
        nlls = []
        for horizon in range(0, T):

            train_x = obs_traj[:horizon, :, :-1]
            train_y = obs_traj[:horizon, :,  -1:]


            with torch.no_grad():
                output = self.pfn(
                    (
                        # TODO Key-value caching on train_x and train_y will save a lot of compute
                        torch.cat([train_x, test_x], dim=0),  # (T + M, B, D-1)
                        train_y # (T, B, 1)
                    ),
                    single_eval_pos=horizon,
                )
                nll = self.pfn.criterion(output, test_y) # M, B
                nlls.append(nll)

        nlls = torch.stack(nlls, dim=0)  # T, M, B
        nll_improvement = -torch.diff(nlls, dim=0) # (T-1, M, B)

        # 3. (weigh by y quantiles) --------------------------------
        # Weigh the nll_diff by the ecdf of the actual test_y values for the monitor points.
        # This incentivizes targeted exploration over mere coverage
        # 2. Per-Environment Ranking
        # test_y.squeeze(-1) gives us (M, B)
        # TODO check that smaller y values get a higher reward!
        y_values = test_y.squeeze(-1)

        # Argsort twice along dim=0 (the M dimension) to get ranks
        # Ranks will be (M, B), where values are 0 to M-1
        ranks = y_values.argsort(dim=0).argsort(dim=0).float()

        # 3. Create Weights (Inverted & Normalized)
        # Smallest y gets weight 1.0, largest gets weight 1/M
        weights = 1.0 - (ranks / y_values.size(0))

        # 4. Apply Weights (T-1, M, B)
        # Broadcasting automatically handles the T-1 dimension
        weighted_improvements = nll_improvement * weights
        weighted_improvements = weighted_improvements.sum(dim=1)

        # Create a zero-row for the initial state where no action has been taken yet
        zero_reward = torch.zeros(1, B, device=nlls.device)

        # Concat to get back to length T
        # rewards[t] is the improvement gained by having observation t
        stepwise_rewards = torch.cat([weighted_improvements, zero_reward], dim=0)

        # TODO Normalize the rewards

        return stepwise_rewards.permute(1, 0)  # (B, T)



    def plot_reward(self, obs_traj, axes):
        """
        Plots the step-wise and cumulative exploration reward for a single trajectory.

        Args:
            reward_provider: Instance of PFNExplorationReward
            obs_traj: Tensor of shape (B, T, D) representing (x, y) for one environment (B)
            ax: Matplotlib axes object
        """
        # 1. Prepare data (Add batch dimension as the __call__ expects B, T, D)

        # 2. Compute rewards
        with torch.no_grad():
            rewards = self(obs_traj).cpu().numpy()

        steps = np.arange(rewards.shape[1])
        cum_rewards = np.cumsum(rewards, axis=1) # Cumulative reward for the single trajectory (B=1)

        # 3. Plotting
        # Primary axis: Step-wise reward
        for i, ax in enumerate(axes):
            color_step = 'tab:blue'
            ax.set_xlabel('Step (Horizon)')
            ax.set_ylabel('Step-wise Information Gain', color=color_step)
            ax.bar(steps, rewards[i], color=color_step, alpha=0.3, label='Step Reward')
            ax.tick_params(axis='y', labelcolor=color_step)

            # Secondary axis: Cumulative reward
            ax2 = ax.twinx()
            color_cum = 'tab:red'
            ax2.set_ylabel('Cumulative Reward', color=color_cum)
            ax2.plot(steps, cum_rewards[i], color=color_cum, marker='o', linewidth=2, label='Total Reward')
            ax2.tick_params(axis='y', labelcolor=color_cum)

            ax.set_title("PFN Exploration Reward over Trajectory")
            ax.grid(True, alpha=0.3)

            # Combined legend
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc='upper left')


if __name__ == '__main__':
    from src.l2o.environment.sinusoid import VectorizedSinusoidEnv

    env = VectorizedSinusoidEnv(num_envs=1, max_steps=50, device=torch.device('cpu'), dim=2)

    trajectory = []
    for _ in range(env.max_steps):
        action = torch.rand(env.num_envs, 2)
        obs = env.step(action)
        trajectory.append(obs)

    trajectory = torch.cat(trajectory)


    pfn_model = load_pfn_model()
    reward_provider = PFNExplorationReward(pfn_model, device='cpu')
    reward_provider.env=env


    # Plot the reward for this dummy trajectory
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    reward_provider.plot_reward(trajectory.unsqueeze(0), [ax])
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))
    env.plot_trajectories(trajectory.unsqueeze(0), [ax])
    plt.show()


