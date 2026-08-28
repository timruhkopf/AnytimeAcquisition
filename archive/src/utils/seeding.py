import random
import numpy as np
import torch

class SeededRandomContext:
    """
    Context manager for setting random seeds across random, numpy, and torch (CPU & CUDA).
    Restores original states on exit. Optionally enforces torch deterministic mode.
    """
    def __init__(self, seed, torch_deterministic=True):
        self.seed = seed
        self.torch_deterministic = torch_deterministic
        self.original_states = {}

    def __enter__(self):
        # Save current RNG states
        self.original_states['random'] = random.getstate()
        self.original_states['numpy'] = np.random.get_state()
        self.original_states['torch'] = torch.get_rng_state()
        self.original_states['torch_deterministic'] = torch.backends.cudnn.deterministic
        self.original_states['torch_benchmark'] = torch.backends.cudnn.benchmark
        if torch.cuda.is_available():
            self.original_states['torch_cuda'] = torch.cuda.get_rng_state_all()

        # Set seeds
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        # Set deterministic behavior
        torch.backends.cudnn.deterministic = self.torch_deterministic
        torch.backends.cudnn.benchmark = not self.torch_deterministic

    def __exit__(self, exc_type, exc_value, traceback):
        # Restore RNG states
        random.setstate(self.original_states['random'])
        np.random.set_state(self.original_states['numpy'])
        torch.set_rng_state(self.original_states['torch'])
        torch.backends.cudnn.deterministic = self.original_states['torch_deterministic']
        torch.backends.cudnn.benchmark = self.original_states['torch_benchmark']
        if torch.cuda.is_available() and 'torch_cuda' in self.original_states:
            torch.cuda.set_rng_state_all(self.original_states['torch_cuda'])
