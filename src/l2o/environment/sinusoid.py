import torch

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
        self.state_x = torch.rand(self.num_envs, 2, device=self.device)
        return torch.cat([self.state_x, self.y], dim=1), self.current_seeds

    @property
    def y(self):
        val = torch.sin(self.state_x * self.freq + self.phase)
        return val.sum(dim=1, keepdim=True)


    def step(self, action):
        self.state_x = torch.clamp(action, 0.0, 1.0)
        self.t += 1

        # Returns (x1, x2, y)
        return torch.cat([self.state_x, self.y], dim=1)
