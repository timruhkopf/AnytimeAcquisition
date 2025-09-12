
import math


import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.env.sinenv import SinEnv


class ToyMultiFidelityEnv(SinEnv):
    # FIXME before using this env, the Loss needs to be made fidelity aware,
    #  and we no longer want to maintain the fidleity_traces argument provided by perplexity.
    # TODO make the plot more pretty :)
    def __init__(self, device='cpu', seed=None,  fidelity_step=0.1):
        """
        Environment with:
        - x: free hyperparameter in [0,1]
        - y: fidelity parameter in [0,1]

        The function is f(x,y) = g(x) * h(y,x)
        where g(x) = amplitude function and h(y,x) is a smooth increasing curve in y resembling learning curves.

        :param device: 'cpu' or 'cuda'
        :param seed: optional seed for reproducibility
        """
        if seed is not None:
            torch.manual_seed(seed)
        self.device = device

        # Parameters for base function g(x)
        self.a1 = torch.tensor(0.9, device=device)
        self.b1 = torch.tensor(2.0, device=device)
        self.phi1 = torch.tensor(0.0, device=device)
        self.a2 = torch.tensor(0.3, device=device)
        self.b2 = torch.tensor(1.0, device=device)
        self.phi2 = torch.tensor(0.7, device=device)
        self.m = torch.tensor(-0.6, device=device)

        # FIXME: remove me:
        self.fidelity_traces = {}
        self.fidelity_step = fidelity_step

    def _g(self, x):
        # Base function g(x) represented as sinusoidal + linear trend in one dimension x
        # Use a 1D sinusoid plus linear combined in a simple way
        val = (self.a1 * torch.sin(self.b1 * 2 * math.pi * x + self.phi1) +
               self.a2 * torch.sin(self.b2 * 2 * math.pi * x + self.phi2) +
               self.m * x)
        return val

    def _h(self, y, x):
        # Learning curve shape: smooth monotone increasing function saturating at 1 for fidelity y in [0,1]
        # Use a shape parameter c(x) controlling speed of learning saturation
        c = 5.0 + 3.0 * torch.sin(2 * math.pi * x)  # shape varies with x
        curve = 1 - torch.exp(-c * y)
        return curve

    def evaluate(self, X):
        """
        Evaluate the environment at points X of shape (..., 2) where
        X[...,0] = x (hyperparameter) ∈ [0,1]
        X[...,1] = y (fidelity) ∈ [0,1]

        Returns tensor of shape X[...,0].shape with values f(x,y).
        """
        x = X[..., 0].to(self.device) if isinstance(X, torch.Tensor) else torch.tensor(X[..., 0], device=self.device)
        y = X[..., 1].to(self.device) if isinstance(X, torch.Tensor) else torch.tensor(X[..., 1], device=self.device)

        g_val = self._g(x)
        h_val = self._h(y, x)

        return g_val * h_val

    def update_trace(self, x, y):
        x_key = round(float(x), 6)
        if x_key not in self.fidelity_traces:
            self.fidelity_traces[x_key] = []
        if not self.fidelity_traces[x_key] or y > self.fidelity_traces[x_key][-1]:
            self.fidelity_traces[x_key].append(y)

    def update_trace(self, x, y):
        x_key = round(float(x), 6)
        if x_key not in self.fidelity_traces:
            self.fidelity_traces[x_key] = []
        if not self.fidelity_traces[x_key] or y > self.fidelity_traces[x_key][-1]:
            self.fidelity_traces[x_key].append(y)

    def expand_fidelity_trace(self, trace):
        expanded_points = []
        for i in range(len(trace) - 1):
            start_fid, x_val = trace[i, 0].item(), trace[i, 1].item()
            end_fid = trace[i + 1, 0].item()
            fid_steps = np.arange(start_fid, end_fid, self.fidelity_step)
            for f in fid_steps:
                expanded_points.append([x_val, f])
        # add the last point explicitly
        last_fid, last_x = trace[-1, 0].item(), trace[-1, 1].item()
        expanded_points.append([last_x, last_fid])
        return np.array(expanded_points)

    def plot3d(self, resolution=200, traces=None):
        # Surface grid
        X = torch.linspace(0, 1, resolution, device=self.device)
        Y = torch.linspace(0, 1, resolution, device=self.device)
        XX, YY = torch.meshgrid(X, Y, indexing='xy')
        flat_in = torch.stack([XX.flatten(), YY.flatten()], dim=-1)
        ZZ = self.evaluate(flat_in).reshape(XX.shape)
        XX_np, YY_np, ZZ_np = XX.cpu().numpy(), YY.cpu().numpy(), ZZ.cpu().numpy()

        fig = make_subplots(rows=1, cols=2,
                            specs=[[{'type': 'surface'}, {'type': 'xy'}]],
                            subplot_titles=("Environment Surface", "Learning Curves"))

        # Surface plot
        surface = go.Surface(z=ZZ_np, x=XX_np, y=YY_np, colorscale='Viridis', opacity=0.85)
        fig.add_trace(surface, row=1, col=1)

        # Plot expanded traces on surface with color by total budget used
        if traces is not None:
            traces = traces.cpu() if isinstance(traces, torch.Tensor) else torch.tensor(traces)
            for b in range(traces.shape[0]):
                trace = traces[b].numpy()
                expanded = self.expand_fidelity_trace(trace)
                xs, ys = expanded[:, 0], expanded[:, 1]
                zs = self.evaluate(torch.tensor(expanded, device=self.device)).cpu().numpy()

                # Color based on cumulative fidelity budget spent along trace
                cum_budget = np.linspace(0, 1, len(xs))

                fig.add_trace(go.Scatter3d(
                    x=xs, y=ys, z=zs,
                    mode='lines+markers',
                    marker=dict(color=cum_budget, colorscale='Viridis', size=4),
                    line=dict(color='black', width=2),
                    name=f"Trace {b}"
                ), row=1, col=1)

        # Plot stored fidelity traces as learning curves on right panel
        for x_key, fidelities in self.fidelity_traces.items():
            fidelities = np.array(fidelities)
            x_inputs = np.full_like(fidelities, x_key)
            points = torch.tensor(np.stack([x_inputs, fidelities], axis=-1), device=self.device)
            performances = self.evaluate(points).cpu().numpy()
            fig.add_trace(go.Scatter(
                x=fidelities, y=performances,
                mode='lines+markers', name=f"x={x_key:.3f}"), row=1, col=2
            )

        fig.update_layout(
            height=600, width=1000,
            # scene=dict(
            #     xaxis_title="Hyperparameter (x)",
            #     yaxis_title="Fidelity (y)",
            #     zaxis_title="Performance f(x,y)"
            # ),
            # xaxis2_title="Fidelity (y)",
            # yaxis2_title="Performance",
            title_text="Multi-Fidelity Environment Surface and Learning Curves"
        )
        fig.show()

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = ToyMultiFidelityEnv(device=device, seed=42)

    # Test evaluation at some points
    test_points = torch.tensor([
        [0.0, 0.0],
        [0.0, 0.5],
        [0.0, 1.0],
        [0.5, 0.0],
        [0.5, 0.5],
        [0.5, 1.0],
        [1.0, 0.0],
        [1.0, 0.5],
        [1.0, 1.0]
    ], device=device)

    values = env.evaluate(test_points)
    print("Test Points:\n", test_points)
    print("Evaluated Values:\n", values)

    env.plot3d(traces=torch.rand(3, 10, 3))