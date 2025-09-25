from timeit import timeit

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

from src.callbacks.abstract_callback import AbstractCallback


from collections import deque, defaultdict

class DormancyTracker:
    def __init__(self, model, layers_to_track=None, tau=0.05, logger=None, max_capacity=20):
        self.model = model
        self.handles = []
        # Use deque with maxlen for memory-bounded activation storage
        self.activations = defaultdict(lambda: deque(maxlen=max_capacity))
        self.tau = tau
        self.logger = logger
        self.layers_to_track = layers_to_track
        self.add_hooks(layers_to_track)

    def add_hooks(self, layers_to_track=None):
        if layers_to_track is None:
            layers_to_track = [blk.mlp.c_fc for blk in self.model.transformer.h]

        for idx, layer in enumerate(layers_to_track):
            handle = layer.register_forward_hook(self._make_hook(idx))
            self.handles.append(handle)

    def _make_hook(self, idx):
        def hook(module, inp, out):
            act = out.detach().cpu()
            self.activations[idx].append(act)  # deque automatically discards oldest when full
        return hook

    def clear(self):
        # Clear all deques by resetting them
        self.activations = defaultdict(lambda: deque(maxlen=next(iter(self.activations.values())).maxlen if self.activations else 100))

    def compute_dormancy_scores(self, step):
        dormancy_scores = {}
        for layer_idx, acts_deque in self.activations.items():

            if len(acts_deque) == 0:
                continue

            acts = torch.cat(list(acts_deque), dim=0)
            acts = acts.view(-1, acts.shape[-1])
            avg_abs = acts.abs().mean(dim=0)
            normalized = avg_abs / avg_abs.mean()
            dormancy_scores[layer_idx] = normalized

        if self.logger is not None:
            log = {'epoch': step}
            for layer_idx, scores in dormancy_scores.items():
                fraction_dormant = (scores <= self.tau).float().mean().item()
                log[f'dormant/layer_{layer_idx}_fraction'] = fraction_dormant
            self.logger.log(log)

        return dormancy_scores

    def remove_hooks(self):
        for h in self.handles:
            h.remove()


class ReDo:
    def __init__(self, tracker: DormancyTracker, tau=0.05, frequency=1000):
        self.model = tracker.model
        self.tracker = tracker
        self.tau = tau
        self.frequency = frequency
        self.step = 0

        # Save initial input weight and bias initialization functions to reuse
        # For demonstrating re-init: sample from current distribution's mean/std
        # More precise: save original init fn used by nn.Linear layers

    def _reinitialize_input_weights(self, layer):
        # Reinitialize input weights and bias of a layer with same shape and dtype
        with torch.no_grad():
            # Assume nn.Linear weights and bias
            fan_in, fan_out = layer.weight.shape[1], layer.weight.shape[0]
            stdv = 1.0 / np.sqrt(fan_in)
            layer.weight.data.uniform_(-stdv, stdv)
            if layer.bias is not None:
                layer.bias.data.uniform_(-stdv, stdv)

    def _zero_outgoing_weights(self, outgoing_layer, neuron_idx):
        # For outgoing weights, zero the weights corresponding to the neuron index
        with torch.no_grad():
            # Example: outgoing_layer.weight shape (out_features, in_features)
            # We zero all weights whose column corresponds to neuron_idx (incoming neuron)
            # So zero column neuron_idx of weights matrix
            outgoing_layer.weight.data[:, neuron_idx] = 0.0
            # Often no bias for outgoing weights, but if bias dimension corresponds, optionally zero

    def apply_redo_if_due(self, step):
        self.step = step
        if step % self.frequency != 0:
            return

        # Compute dormant neuron scores
        dormancy_scores = self.tracker.compute_dormancy_scores(step)

        # Iterate over tracked layers
        for layer_idx, scores in dormancy_scores.items():
            # Identify dormant neurons below threshold tau
            dormant_neurons = (scores <= self.tau).nonzero(as_tuple=True)[0]

            if len(dormant_neurons) == 0:
                continue

            # Find input and outgoing layers
            input_layer = self.tracker.model.transformer.h[
                layer_idx].mlp.c_fc  # input weights of neuron
            outgoing_layer = self.tracker.model.transformer.h[
                layer_idx].mlp.c_proj  # outgoing weights

            for neuron_idx in dormant_neurons:
                neuron_idx = neuron_idx.item()

                # Reinitialize input weights and bias of dormant neuron
                self._reinitialize_single_neuron_input_weights(input_layer, neuron_idx)
                # Zero out outgoing weights connected to this neuron
                self._zero_outgoing_weights(outgoing_layer, neuron_idx)

        # Clear accumulated activations to start fresh after recycling
        self.tracker.clear()

    def _reinitialize_single_neuron_input_weights(self, linear_layer, neuron_idx):
        """
        Reinit input weights only for one neuron (row neuron_idx) in linear_layer.
        """
        with torch.no_grad():
            fan_in = linear_layer.weight.size(1)
            stdv = 1.0 / np.sqrt(fan_in)

            # Uniform init for that neuron row
            linear_layer.weight.data[neuron_idx].uniform_(-stdv, stdv)

            if linear_layer.bias is not None:
                # also reinit bias for that neuron
                linear_layer.bias.data[neuron_idx].uniform_(-stdv, stdv)


class DormancyCallback(AbstractCallback):
    def __init__(self, tau=0.05, frequency=100, **kwargs):
        super().__init__(**kwargs)
        self.tau = tau
        self.frequency = frequency

    def on_train_begin(self):
        self.tracker = DormancyTracker(self.trainer.model, tau=self.tau, logger=self.trainer.logger)
        self.redo = ReDo(self.tracker, tau=self.tau, frequency=self.frequency)


    def on_epoch_end(self, **kwargs):
        self.redo.apply_redo_if_due(self.epoch)

    def on_train_end(self):
        self.tracker.remove_hooks()


if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import torch.optim as optim

    # Assuming DormancyTracker and ReDo classes are defined as before and imported
    from src.model.tiny_causal import TinyCausalTransformer

    model = TinyCausalTransformer(n_layers=2)  # your model instantiation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Example loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Initialize dormancy tracker and ReDo manager
    tracker = DormancyTracker(model, tau=0.05)
    redo = ReDo(model, tracker, tau=0.05, frequency=1000)

    num_epochs = 10
    step = 0

    # create dummy dataloader for demonstration
    X = torch.randn(100, 20, 3)  # 100 samples, sequence length 20, input dim 3
    y = torch.randn(100, 20, 3)  # dummy targets
    dataset = torch.utils.data.TensorDataset(X, y)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    for epoch in range(num_epochs):
        model.train()
        for X, y in train_dataloader:  # Your dataloader yielding batches of shape (B, S, d_in)
            step += 1
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            outputs, _ = model(X)

            # Replace targets accordingly if supervised setup
            targets = y  # Tensor matching outputs shape, prepared outside loop.
            loss = loss_fn(outputs, targets)

            loss.backward()
            optimizer.step()

            # The tracker hooks collect activations during forward pass automatically

            # Periodically apply ReDo neuron recycling
            redo.apply_redo_if_due(step)

        # Optionally evaluate and log dormancy stats
        dormancy_scores = tracker.compute_dormancy_scores()
        for layer_idx, scores in dormancy_scores.items():
            fraction_dormant = (scores <= tracker.tau).float().mean().item()
            print(
                f"Epoch {epoch} Step {step} Layer {layer_idx}: {fraction_dormant * 100:.2f}% dormant neurons")

        # Clear activations after epoch or at your desired interval
        tracker.clear()

    tracker.remove_hooks()
