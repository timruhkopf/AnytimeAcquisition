import torch
import torch.nn as nn


class SimpleSequenceReconstructionLoss:

    def __call__(self, predictions, X, targets):
        return nn.functional.mse_loss(predictions[..., :-1], X[..., :-1], reduction='mean')

class SequenceReconstructionLoss:

    def __call__(self, predictions, X, targets):
        weight =  X[..., -1] - predictions[..., -1]
        weight = torch.abs(weight)
        diff= nn.functional.mse_loss(predictions[..., :-1], X[..., :-1], reduction='none').sum(dim=-1)
        return torch.mean(weight*diff)
