import torch


class StepPenaltyReward:
    def __init__(self, step_penalty=0.01, time_decay=0.1, device='cuda'):
        """
        step_penalty: The base cost per step.
        time_decay: How much the penalty increases as time t increases.
        """
        self.step_penalty = step_penalty
        self.time_decay = time_decay
        self.device = device

    def __call__(self, obs_traj, env=None):
        T, B, _ = obs_traj.shape

        # Create step indices: [0, 1, 2, ..., T-1]
        step_indices = torch.arange(T, device=self.device).view(T, 1)

        # Penalty grows: base_penalty * (1 + decay * t)
        # We return it as a negative value (cost)
        penalty = -(self.step_penalty * (1.0 + self.time_decay * step_indices))

        # Broadcast to match Batch size: (T, B)
        return penalty.expand(-1, B)