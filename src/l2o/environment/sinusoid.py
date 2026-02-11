import torch
import numpy as np

class VectorizedSinusoidEnv:
    def __init__(self, num_envs, max_steps, device):
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.device = device
        self.current_seeds = torch.randint(0, 2 ** 31, (num_envs,), device='cpu')
        self.reset()

    def _get_params_from_seed(self, seeds):
        """Reconstructs freq and phase from a tensor of seeds."""
        batch_size = seeds.shape[0]
        freq = torch.zeros((batch_size, 2), device=self.device)
        phase = torch.zeros((batch_size, 2), device=self.device)

        for i, seed in enumerate(seeds):
            # Use a local generator for total isolation
            gen = torch.Generator(device=self.device).manual_seed(int(seed))
            freq[i] = torch.rand(2, generator=gen, device=self.device) * 4 * 3.14159
            phase[i] = torch.rand(2, generator=gen, device=self.device) * 2 * 3.14159

        return freq, phase

    def reset(self, seeds=None):
        self.t = 0
        if seeds is not None:
            self.current_seeds = seeds
        else:
            # Generate new random seeds if none provided
            self.current_seeds = torch.randint(0, 2 ** 31, (self.num_envs,), device='cpu')

        self.freq, self.phase = self._get_params_from_seed(self.current_seeds)
        x = torch.rand(self.num_envs, 2, device=self.device)
        return torch.cat([x, self.evaluate(x)], dim=1), self.current_seeds

    def evaluate(self, x):
        val = torch.sin(x * self.freq + self.phase)
        return val.sum(dim=-1, keepdim=True)


    def step(self, action):
        x = torch.clamp(action, 0.0, 1.0)
        self.t += 1

        # Returns (x1, x2, y)
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
    env = VectorizedSinusoidEnv(num_envs=4, max_steps=50, device='cpu')
    obs, seeds0 = env.reset()

    # collect reference y values :
    x = torch.rand(100, env.num_envs, 2, device=env.device)
    y = env.evaluate(x)

    print(env.freq, env.phase, seeds0)

    obs, seeds = env.reset()

    print(env.freq, env.phase, seeds)

    obs, seeds = env.reset(seeds=seeds0)

    y1 = env.evaluate(x)

    assert torch.allclose(y, y1), "The same seeds should produce the same landscape and thus the same evaluations."

    print(env.freq, env.phase, seeds)

    trajectory = [obs]
    for _ in range(50):
        action = torch.rand(env.num_envs, 2)  # Random action for testing
        obs = env.step(action)
        trajectory.append(obs)

    trajectory = torch.stack(trajectory, dim=0)


    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, env.num_envs, figsize=(15, 5))
    env.plot_trajectories(trajectory.permute(1, 0, 2), axes)
    plt.show()