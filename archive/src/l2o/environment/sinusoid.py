import torch
import numpy as np


class VectorizedSinusoidEnv:
    def __init__(self, num_envs, max_steps, device, dim=2):
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.device = device
        self.dim = dim  # Configurable dimensionality
        _, self.seed = self.reset()

    def _get_params_from_seed(self, seeds):
        """Vectorized reconstruction of freq and phase using seeds."""
        # Moving seeds to the correct device for computation
        seeds = seeds.to(self.device).unsqueeze(1)

        # Use a simple deterministic hash to generate freq and phase without loops
        # This scales much better for high dimensions than torch.Generator loops
        def get_random_tensor(offset, scale):
            # Prime numbers used to create distinct "channels" for freq and phase
            state = (seeds * 1103515245 + offset) & 0x7FFFFFFF
            noise = (state.float() / 0x7FFFFFFF).repeat(1, self.dim)
            # Add dimension-specific jitter so each dimension is different
            dim_offsets = torch.arange(self.dim, device=self.device).unsqueeze(0)
            noise = (noise + dim_offsets * 0.618033) % 1.0
            return noise * scale

        freq = get_random_tensor(12345, 4 * torch.pi)
        phase = get_random_tensor(67890, 2 * torch.pi)

        return freq, phase

    def reset(self, seeds=None):
        self.t = 0
        if seeds is not None:
            self.current_seeds = seeds
        else:
            self.current_seeds = torch.randint(0, 2 ** 31, (self.num_envs,), device='cpu')

        self.freq, self.phase = self._get_params_from_seed(self.current_seeds)
        # Observation is [x_1, ..., x_n, y]
        x = torch.rand(self.num_envs, self.dim, device=self.device)
        return torch.cat([x, self.evaluate(x)], dim=1), self.current_seeds

    def evaluate(self, x):
        # f(x) = sum(sin(w*x + phi))
        val = torch.sin(x * self.freq + self.phase)
        return val.sum(dim=-1, keepdim=True) / self.dim

    def step(self, action):
        x = torch.clamp(action, 0.0, 1.0)
        self.t += 1
        # Returns (x_1, ..., x_n, y)
        return torch.cat([x, self.evaluate(x)], dim=1)

    def plot_trajectories(self, trajectories, axes):
        """
        Renders the reward surface, trajectory, and 'incumbent' markers.
        trajectories: Tensor (N, T, 3)
        axes: List of matplotlib axes or a single axis
        """
        num_to_plot = len(axes)

        for i in range(num_to_plot):
            ax = axes[i]
            # 1. Create Heatmap Grid
            grid_res = 100
            x_range = torch.linspace(0, 1, grid_res, device=self.device)
            grid_x, grid_y = torch.meshgrid(x_range, x_range, indexing='ij')
            grid_flat = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)

            # Temporarily override params for index 'i' to get the correct landscape
            original_freq, original_phase = self.freq, self.phase
            self.freq, self.phase = original_freq[i:i + 1], original_phase[i:i + 1]

            with torch.no_grad():
                z = self.evaluate(grid_flat).reshape(grid_res, grid_res).cpu().numpy()

            # Restore original params
            self.freq, self.phase = original_freq, original_phase

            # 2. Plot Heatmap
            cont = ax.contourf(grid_x.cpu(), grid_y.cpu(), z, levels=50, cmap='viridis')
            # Note: We skip the colorbar here to keep the ax clean,
            # or you can add it back using plt.colorbar(cont, ax=ax)

            # 3. Plot Trajectory
            states = trajectories[i, :, :2].cpu().numpy()
            ax.plot(states[:, 0], states[:, 1], color='white', lw=1.5, alpha=0.8)
            ax.scatter(states[0, 0], states[0, 1], color='cyan', label='Start', s=50, zorder=5)
            ax.scatter(states[-1, 0], states[-1, 1], color='magenta', label='End', s=50, zorder=5)

            # 4. Find Incumbents (Minimization)
            z_vals = trajectories[i, :, 2]  # Keep as tensor for vector ops
            # Calculate running minimum
            running_min, _ = torch.cummin(z_vals, dim=0)

            # An 'incumbent' occurs where the value is equal to the running min
            # AND it is strictly less than the previous running min (or it's the first step)
            is_incumbent = (z_vals == running_min)
            # To avoid repeats of the same minimum value, we find where the running min changes
            shifted_min = torch.cat([torch.tensor([float('inf')], device=self.device), running_min[:-1]])
            change_mask = (running_min < shifted_min).cpu().numpy()

            # Plot all incumbents at once
            incumbent_idx = np.where(change_mask)[0]
            ax.scatter(
                states[incumbent_idx, 0],
                states[incumbent_idx, 1],
                color='red', marker='x', s=100, zorder=6, label='Incumbent'
            )

            ax.set_title(f"Env {i}")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Create an environment with 10-dimensional inputs
    env = VectorizedSinusoidEnv(num_envs=64, max_steps=100, device='cpu', dim=10)

    # (Check the seeding and reset behavior) ---------------
    obs0, seeds0 = env.reset()

    # collect reference y values :
    x = torch.rand(100, env.num_envs, env.dim, device=env.device)
    y = env.evaluate(x)

    obs, seeds = env.reset()

    assert not torch.allclose(seeds, seeds0), "Seeds should be different on each reset if not provided."

    obs, seeds = env.reset(seeds=seeds0)

    y1 = env.evaluate(x)
    assert not torch.allclose(obs,
                              obs0), "The initial x values should be different on each reset, even with the same seeds."
    assert torch.allclose(y, y1), "The same seeds should produce the same landscape and thus the same evaluations."

    # (Check the incumbent performance vs dimensionality) ---------------
    # We can run a simple random search and see how the incumbent evolves in higher dimensions.
    env_high_dim = VectorizedSinusoidEnv(num_envs=4, max_steps=100, device='cpu', dim=30)
    obs, seeds = env_high_dim.reset()

    trajectory = [obs]
    for _ in range(400):
        action = torch.rand(env_high_dim.num_envs, env_high_dim.dim)  # Random action for testing
        obs = env_high_dim.step(action)
        trajectory.append(obs)

    trajectory = torch.stack(trajectory, dim=0).to('cpu')  # Move to CPU for plotting

    # from l2o.model.rewards.area_under_incumbent_curve import AUICReward

    reward_manager = AUICReward()

    fig, axes = plt.subplots(1, env_high_dim.num_envs, figsize=(15, 5), sharey=True)

    for i, ax in enumerate(axes):
        tr = trajectory[:, i]  # (T, dim + 1)
        reward_manager.plot_reward(tr, ax)

    plt.show()

    # (Plotting example with 2D inputs for visualization) ---------------
    env2 = VectorizedSinusoidEnv(num_envs=4, max_steps=100, device='cpu', dim=2)
    obs, seeds = env2.reset()

    trajectory = [obs]
    for _ in range(50):
        action = torch.rand(env2.num_envs, env2.dim)  # Random action for testing
        obs = env2.step(action)
        trajectory.append(obs)

    trajectory = torch.stack(trajectory, dim=0).to('cpu')  # Move to CPU for plotting

    fig, axes = plt.subplots(1, env2.num_envs, figsize=(15, 5))
    env2.plot_trajectories(trajectory.permute(1, 0, 2), axes)
    plt.show()
