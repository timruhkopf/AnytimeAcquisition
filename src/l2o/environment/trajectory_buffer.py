import torch


class CumulativeEpisodeBuffer:
    def __init__(self, total_episodes, max_steps, obs_dim, act_dim, device):
        self.total_episodes = total_episodes
        self.max_steps = max_steps
        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim


        self.reset()

    def reset(self):
        total_episodes, max_steps, obs_dim, act_dim, device = \
            self.total_episodes, self.max_steps, self.obs_dim, self.act_dim, self.device

        # Metadata to reinstantiate envs where necessary
        self.seeds = torch.zeros((total_episodes,), dtype=torch.long, device='cpu')

        # Core Trajectory Data
        self.obs = torch.zeros((total_episodes, max_steps, obs_dim), device=device)
        self.acts = torch.zeros((total_episodes, max_steps, act_dim), device=device)
        self.logprobs = torch.zeros((total_episodes, max_steps), device=device)
        self.values = torch.zeros((total_episodes, max_steps), device=device)
        self.rewards = torch.zeros((total_episodes, max_steps), device=device)
        self.dones = torch.zeros((total_episodes, max_steps), device=device)

        self.episode_ptr = 0

    def store_batch(self, obs, acts, logprobs, values, rewards, dones, seeds):
        """Stores a chunk of parallel episodes from the vectorized env."""
        num_envs = obs.shape[0]
        end_ptr = self.episode_ptr + num_envs

        if end_ptr > self.total_episodes:
            raise ValueError("Buffer overflow! Decrease num_envs or increase total_episodes.")

        self.obs[self.episode_ptr:end_ptr] = obs
        self.acts[self.episode_ptr:end_ptr] = acts
        self.logprobs[self.episode_ptr:end_ptr] = logprobs
        self.values[self.episode_ptr:end_ptr] = values
        self.dones[self.episode_ptr:end_ptr] = dones
        self.seeds[self.episode_ptr:end_ptr] = seeds.to('cpu')

        self.episode_ptr = end_ptr

    def compute_gae(self, last_values, gamma=0.99, gae_lambda=0.95):
        """
        Computes GAE and Returns based on stored buffer data.
        # FIXME: this should probably best be situated in the trainer since it is part of the training loop
        """
        advantages = torch.zeros_like(self.rewards)
        last_gae_lam = 0

        for t in reversed(range(self.max_steps)):
            if t == self.max_steps - 1:
                next_values = last_values
            else:
                next_values = self.values[:, t + 1]

            # TD Error (delta)
            delta = self.rewards[:, t] + gamma * next_values * (1 - self.dones[:, t]) - self.values[:, t]

            # Recursive GAE calculation
            last_gae_lam = delta + gamma * gae_lambda * (1 - self.dones[:, t]) * last_gae_lam
            advantages[:, t] = last_gae_lam

        returns = advantages + self.values
        return advantages, returns

    def get_loader(self, last_values, batch_size, gamma=0.99, gae_lambda=0.95):
        """Yields mini-batches of full sequences after computing advantages."""

        # Calculate GAE using the factored method
        advantages, returns = self.compute_gae(last_values, gamma, gae_lambda)

        # Flatten/Shuffle logic
        indices = torch.randperm(self.total_episodes)

        for i in range(0, self.total_episodes, batch_size):
            batch_idx = indices[i: i + batch_size]
            yield (
                self.obs[batch_idx],
                self.acts[batch_idx],
                self.logprobs[batch_idx],
                returns[batch_idx],
                advantages[batch_idx],
                self.values[batch_idx]
            )

    def to(self, device):
        """Moves all tensors to the specified device."""
        self.obs = self.obs.to(device)
        self.acts = self.acts.to(device)
        self.logprobs = self.logprobs.to(device)
        self.values = self.values.to(device)
        self.rewards = self.rewards.to(device)
        self.dones = self.dones.to(device)
        self.seeds = self.seeds.to(device)