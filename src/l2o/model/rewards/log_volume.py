class GroundTruthVolumeReward:
    def __init__(self, num_grid_points=1024, device='cuda', use_log=True):
        self.grid = torch.linspace(0, 1, num_grid_points, device=device).unsqueeze(0)
        self.use_log = use_log
        self.device = device

    def __call__(self, obs_traj, env):

        # obs_traj: (Seq, Batch, 3)
        # env: Must allow functional evaluation

        # 1. Get Current Incumbent (The Ceiling)
        y_vals = obs_traj[..., 2]
        incumbents, _ = torch.cummin(y_vals, dim=0)  # (Seq, Batch)

        # 2. Get True Function Shape (The Floor)
        # We evaluate the TRUE function on a dense grid
        # env.freq, env.phase: (Batch, 2)
        true_y_grid = env.functional_evaluate(self.grid, env.freq, env.phase)  # (Batch, G, 1)
        true_y_grid = true_y_grid.squeeze(-1).unsqueeze(0)  # (1, Batch, G)

        # 3. Calculate True Regret Volume
        # The volume is the integral of the gap between the incumbent and the true function
        # wherever the true function is better.
        # Vol = Integral( max(0, Incumbent - True_Y) )

        # Volume = Area where the true function is below our best
        gaps = torch.clamp(incumbents.unsqueeze(-1) - true_y_grid, min=1e-8)
        volumes = gaps.mean(dim=-1)  # (Seq, Batch)

        if self.use_log:
            # Reward is the reduction in log-volume (relative improvement)
            log_v = torch.log(volumes)
            reward = log_v[:-1] - log_v[1:]
            return torch.cat([torch.zeros(1, obs_traj.size(1), device=self.device), reward], dim=0)

        return torch.roll(volumes, 1, 0) - volumes



class BudgetAwareVolumeReward:
    def __init__(self, num_grid_points=1024, step_penalty=0.01, device='cuda'):
        self.grid = torch.linspace(0, 1, num_grid_points, device=device).unsqueeze(0)
        self.step_penalty = step_penalty
        self.device = device

    def __call__(self, obs_traj, env):
        # 1. Setup dimensions
        T, B, _ = obs_traj.shape

        # 2. Get Ground Truth Function on Grid
        # true_y_grid: (1, B, G)
        true_y_grid = env.functional_evaluate(self.grid, env.freq, env.phase)
        true_y_grid = true_y_grid.squeeze(-1).unsqueeze(0)

        # 3. Get Incumbents (The Ceiling)
        y_vals = obs_traj[..., 2]  # (T, B)
        incumbents, _ = torch.cummin(y_vals, dim=0)
        incumbents = incumbents.unsqueeze(-1)  # (T, B, 1)

        # 4. Calculate Potential Regret Volume
        # Gap = max(0, Incumbent - True_Function)
        # We add a small epsilon to avoid log(0)
        gaps = torch.clamp(incumbents - true_y_grid, min=1e-7)
        volumes = gaps.mean(dim=-1)  # (T, B)

        # 5. Reward = (Log Volume Reduction) - Step Penalty
        # We use log so that reducing volume from 0.01 to 0.001
        # is as rewarding as 1.0 to 0.1.
        log_v = torch.log(volumes)

        # Volume improvement (positive is good)
        vol_reduction = log_v[:-1] - log_v[1:]

        # Pad the first step
        first_step = torch.zeros(1, B, device=self.device)
        total_vol_reward = torch.cat([first_step, vol_reduction], dim=0)

        # Apply Penalty: Every step costs something
        # This encourages the agent to stop exploring once the
        # 'Information Gain' is less than the 'Step Cost'.
        combined_reward = total_vol_reward - self.step_penalty

        return combined_reward