import torch
import torch.nn as nn

import math

class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        return (x-self.mean)/self.std

class MLP(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_outputs,
        num_layers,
        num_hidden,
        preactivation_noise_std,
        output_noise,
        init_std, 
        sparseness,
    ):
        super(MLP, self).__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.preactivation_noise_std = preactivation_noise_std
        self.output_noise = output_noise
        self.init_std = init_std
        self.sparseness = sparseness

        activation = "tanh"

        # (x - m) / sigma to normalize BNN inputs
        self.normalizer = Normalize(0.5, math.sqrt(1 / 12))

        self.linears = torch.nn.ModuleList(
            [torch.nn.Linear(num_inputs, num_hidden)]
            + [torch.nn.Linear(num_hidden, num_hidden) for _ in range(num_layers - 2)]
            + [torch.nn.Linear(num_hidden, num_outputs)]
        )

        self.reset_parameters()

        self.activation = {
            "tanh": torch.nn.Tanh(),
            "relu": torch.nn.ReLU(),
            "elu": torch.nn.ELU(),
            "identity": torch.nn.Identity(),
        }[activation]

    def reset_parameters(self, init_std=None, sparseness=None):
        init_std = init_std if init_std is not None else self.init_std
        sparseness = sparseness if sparseness is not None else self.sparseness
        for linear in self.linears:
            linear.reset_parameters()

        with torch.no_grad():
            if init_std is not None:
                for linear in self.linears:
                    linear.weight.normal_(0, init_std)
                    linear.bias.normal_(0, init_std)

            if sparseness > 0.0:
                for linear in self.linears[1:-1]:
                    linear.weight /= (1.0 - sparseness) ** (1 / 2)
                    linear.weight *= torch.bernoulli(
                        torch.ones_like(linear.weight) * (1.0 - sparseness)
                    )

    def forward(self, x):
        self.normalizer(x)
        for linear in self.linears[:-1]:
            x = linear(x)
            x = x + torch.randn_like(x) * self.preactivation_noise_std
            x = torch.tanh(x)
        x = self.linears[-1](x)
        return x + torch.randn_like(x) * self.output_noise
