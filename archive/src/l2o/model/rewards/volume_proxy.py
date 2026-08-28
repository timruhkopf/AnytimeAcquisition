"""
The idea here is, that we can devide the space into cells based on the current train set.
the larger the cell, the more "potential" it has to reduce the incumbent.
we want to reward for closing gaps but even more so, if those gaps produce a good y value.

Consider. if we use this, then we may also incorporate local curvature into the mix ; which will encourage the model
to find its own exploration exploitation tradeoff in order to maximize incumbent / regret volume  reduction
"""

import torch


class VoronoiVolumeReward:
    def __init__(self, num_stardust=1024, step_penalty=0.01, device='cuda'):
        self.device = device
        self.step_penalty = step_penalty
        # Fixed 'stardust' points to approximate the geometry of the space
        self.stardust = torch.quasirandom.SobolEngine(2).draw(num_stardust).to(device)  # (M, 2)

    def __call__(self, obs_traj, env):
        """
        obs_traj: (T, B, 3) where [..., :2] is X and [..., 2] is Y
        """
        T, B, _ = obs_traj.shape
        M = self.stardust.size(0)

        # 1. Get X trajectory: (T, B, 2)
        x_history = obs_traj[..., :2]
        y_history = obs_traj[..., 2]

        # 2. Track Incumbent
        incumbents, _ = torch.cummin(y_history, dim=0)  # (T, B)

        # 3. Compute Voronoi Membership for all time steps
        # This is the "expensive" part, but we can vectorize it.
        # For each Batch and each Step t, which stardust point 'm' is closest to which x_i?

        rewards = torch.zeros(T, B, device=self.device)

        # We iterate through time because Voronoi is naturally sequential
        # (Optimization: You could vectorize this with a massive distance matrix)

        # Initial 'Occupancy' - how many stardust points belong to each cell
        # At t=0, first point 'owns' everything.
        # But we want to measure REDUCTION in territory.

        # Pre-calculate distances between all stardust and all possible X
        # x_history: (T, B, 2), stardust: (M, 2)
        # distances: (T, B, M)

        # To make it efficient, let's use the "Current Volume" state
        for b in range(B):
            # dist_matrix: (T, M) distance from each sampled x to each stardust
            dists = torch.cdist(x_history[:, b, :], self.stardust)

            # Cumulative minimum distance from any sampled point to each stardust point
            # running_min_dists: (T, M)
            running_min_val, running_indices = torch.min(
                torch.stack([dists[:t + 1].min(dim=0)[0] for t in range(T)]), dim=0
            )


            # For each stardust point m, which x_i (i <= t) is the closest?
            for t in range(T):
                # x_so_far: (t+1, 2)
                # distances from stardust to all points sampled so far: (t+1, M)
                d_to_samples = dists[:t + 1, :]

                # closest_sample_idx: (M,) - which of the (t+1) points owns each stardust point
                nearest_sample_val, nearest_sample_idx = torch.min(d_to_samples, dim=0)

                # 4. Weight the Voronoi Cells by "Regret Potential"
                # Cell Volume = (number of stardust points owned)
                # Potential = max(0, Incumbent_t - Y_of_owning_sample)
                # But that's zero for the incumbent!
                # Instead, we weight by the "Potential of the territory"

                # A territory is "dangerous" (high regret) if it's large AND
                # near a low Y value.

                # Simplified: Total Volume = Sum over stardust of (dist to nearest sample)
                # This is the 'Average Distance to nearest neighbor' across the domain.
                volume_t = nearest_sample_val.mean()

                rewards[t, b] = volume_t

        # 5. Delta Volume
        # reward = Vol_{t-1} - Vol_t
        vol_reduction = rewards[:-1] - rewards[1:]
        first_step = torch.zeros(1, B, device=self.device)

        return torch.cat([first_step, vol_reduction], dim=0) - self.step_penalty