from .utils import print_once

import torch
from torch import nn
import psutil


def get_borders(lower=0.0, upper=1.0, num_bars=128):
    return torch.linspace(lower, upper, num_bars + 1)


class BarDistribution(nn.Module):
    """Taken from pfns4BO"""

    def __init__(self, borders: torch.Tensor, smoothing=.0,
                 ignore_nan_targets=True):  # here borders should start with min and end with max, where all values lie in (min,max) and are sorted
        '''
        :param borders:
        :param smoothing:
        :param append_mean_pred: Whether to predict the mean of the other positions as a last output in forward,
        is enabled when additionally y has a sequence length 1 shorter than logits, i.e. len(logits) == 1 + len(y)
        '''
        super().__init__()
        assert len(borders.shape) == 1
        self.register_buffer('borders', borders)
        self.register_buffer('smoothing', torch.tensor(smoothing))
        self.register_buffer('bucket_widths', self.borders[1:] - self.borders[:-1])
        full_width = self.bucket_widths.sum()

        assert (1 - (full_width / (self.borders[-1] - self.borders[
            0]))).abs() < 1e-2, f'diff: {full_width - (self.borders[-1] - self.borders[0])} with {full_width} {self.borders[-1]} {self.borders[0]}'
        assert (self.bucket_widths >= 0.0).all(), "Please provide sorted borders!"  # This also allows size zero buckets
        self.num_bars = len(borders) - 1
        self.ignore_nan_targets = ignore_nan_targets
        self.to(borders.device)

    def __setstate__(self, state):
        super().__setstate__(state)
        self.__dict__.setdefault('append_mean_pred', False)
