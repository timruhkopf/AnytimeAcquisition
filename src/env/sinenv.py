import torch
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.loss.auc import AUCRegretLoss


class SinEnvTorch:
    def __init__(self, a=1.0, b=3, phi=0.0, m=0.0, theta=0.0, seed=None, device='cpu'):
        """
        Sinusoidal + linear trend + rotation in 2D with PyTorch tensors.
        Domain normalized to [0,1] x [0,1].

        f(x, y) = a * sin(b * 2π * x' + phi) + m * x' (with rotation)
        x', y' = rotated coordinates

        :param a: Amplitude of the sine wave.
        :param b: Frequency of the sine wave.
        :param phi: Phase shift of the sine wave.
        :param m: Slope of the linear trend.
        :param theta: Rotation angle in radians.
        :param seed: Random seed for reproducibility.
        :param device: 'cpu' or 'cuda' device for tensors.
        """
        if seed is not None:
            torch.manual_seed(seed)
        self.device = device
        self.a = torch.tensor(a, device=device)
        self.b = torch.tensor(b, device=device)
        self.phi = torch.tensor(phi, device=device)
        self.m = torch.tensor(m, device=device)
        self.theta = torch.tensor(theta, device=device)

    def resample(self, a_range=(0.5, 2.0), b_range=(1, 10), phi_range=(0, 2*math.pi),
                 m_range=(-1, 1), theta_range=(0, 2*math.pi)):
        """Randomly sample new environment parameters from given ranges (torch.uniform)."""
        self.a = (a_range[1] - a_range[0]) * torch.rand(1, device=self.device) + a_range[0]
        self.b = (b_range[1] - b_range[0]) * torch.rand(1, device=self.device) + b_range[0]
        self.phi = (phi_range[1] - phi_range[0]) * torch.rand(1, device=self.device) + phi_range[0]
        self.m = (m_range[1] - m_range[0]) * torch.rand(1, device=self.device) + m_range[0]
        self.theta = (theta_range[1] - theta_range[0]) * torch.rand(1, device=self.device) + theta_range[0]

    def rotate(self, x, y):
        # x, y: tensors of same shape
        x_r = torch.cos(self.theta) * x + torch.sin(self.theta) * y
        y_r = -torch.sin(self.theta) * x + torch.cos(self.theta) * y
        return x_r, y_r

    def evaluate(self, x, y):
        """
        Evaluate the function at (x, y). x,y should be tensors or convertible to tensors.
        Values should be in [0,1].
        """
        x = x.to(self.device) if isinstance(x, torch.Tensor) else torch.tensor(x, device=self.device)
        y = y.to(self.device) if isinstance(y, torch.Tensor) else torch.tensor(y, device=self.device)
        x_r, y_r = self.rotate(x, y)
        val = self.a * torch.sin(self.b * 2 * math.pi * x_r + self.phi) + self.m * x_r
        return val

    def normalized(self, xs, ys):
        vals = self.evaluate(xs, ys)
        vmin = vals.min()
        vmax = vals.max()
        return (vals - vmin) / (vmax - vmin + 1e-9)

    # def plot3d(self, resolution=200, trace=None, plot_points_only=False, auc=False):
    #     # Create grid tensors
    #     X = torch.linspace(0, 1, resolution, device=self.device)
    #     Y = torch.linspace(0, 1, resolution, device=self.device)
    #     XX, YY = torch.meshgrid(X, Y, indexing='xy')
    #     ZZ = self.normalized(XX, YY)
    #
    #     # Convert to numpy arrays for Plotly
    #     XX_np = XX.cpu().numpy()
    #     YY_np = YY.cpu().numpy()
    #     ZZ_np = ZZ.cpu().numpy()
    #
    #     # Create surface plot
    #     surface = go.Surface(z=ZZ_np, x=XX_np, y=YY_np, colorscale='Viridis', opacity=0.85)
    #
    #     data = [surface]
    #
    #     # Add trace if given
    #     if trace is not None:
    #         X_trace, Y_trace = trace
    #         # Convert trace to tensor for normalized evaluation
    #         Z_trace = self.normalized(torch.tensor(X_trace, device=self.device),
    #                                   torch.tensor(Y_trace, device=self.device)).cpu().numpy()
    #         if plot_points_only:
    #             trace_points = go.Scatter3d(
    #                 x=X_trace, y=Y_trace, z=Z_trace,
    #                 mode='markers',
    #                 marker=dict(size=5, color='red')
    #             )
    #             data.append(trace_points)
    #         else:
    #             trace_line = go.Scatter3d(
    #                 x=X_trace, y=Y_trace, z=Z_trace,
    #                 mode='lines+markers',
    #                 line=dict(color='red', width=4),
    #                 marker=dict(size=3, color='red')
    #             )
    #             data.append(trace_line)
    #
    #     fig = go.Figure(data=data)
    #     fig.update_layout(
    #         title="Interactive 2D Sinusoidal Environment",
    #         scene=dict(
    #             xaxis_title='x',
    #             yaxis_title='y',
    #             zaxis_title='f(x, y)'
    #         ),
    #         width=800,
    #         height=600
    #     )
    #
    #     fig.show()



    def plot3d(self, resolution=200, trace=None, plot_points_only=False, trajectories=None):
        """
        Plots the environment surface and optionally trajectories with their AUC loss side by side.

        Params:
        - resolution: surface grid resolution.
        - trace: tuple/list of (X_trace, Y_trace) for trajectory path on surface.
        - plot_points_only: if True, plot only points along trace.
        - trajectories: torch tensor (batch_size, seq_len) for AUC loss plotting.
        """
        # Create surface grid
        X = torch.linspace(0, 1, resolution, device=self.device)
        Y = torch.linspace(0, 1, resolution, device=self.device)
        XX, YY = torch.meshgrid(X, Y, indexing='xy')
        ZZ = self.normalized(XX, YY)

        # Numpy for plotting
        XX_np, YY_np, ZZ_np = XX.cpu().numpy(), YY.cpu().numpy(), ZZ.cpu().numpy()

        # Setup subplot figure: 1 row, 2 cols if trajectories given, else just 1 col
        if trajectories is not None:
            fig = make_subplots(rows=1, cols=2,
                                specs=[[{'type': 'surface'}, {'type': 'xy'}]],
                                subplot_titles=("Environment Surface", "Trajectories AUC"))
        else:
            fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'surface'}]])

        # Surface plot on left
        surface = go.Surface(z=ZZ_np, x=XX_np, y=YY_np, colorscale='Viridis', opacity=0.85)
        fig.add_trace(surface, row=1, col=1)

        # Add trace path on surface if given
        if trace is not None:
            X_trace, Y_trace = trace
            Z_trace = self.normalized(torch.tensor(X_trace, device=self.device),
                                      torch.tensor(Y_trace, device=self.device)).cpu().numpy()
            if plot_points_only:
                trace_points = go.Scatter3d(x=X_trace, y=Y_trace, z=Z_trace, mode='markers',
                                            marker=dict(size=5, color='red'))
                fig.add_trace(trace_points, row=1, col=1)
            else:
                trace_line = go.Scatter3d(x=X_trace, y=Y_trace, z=Z_trace, mode='lines+markers',
                                          line=dict(color='red', width=4),
                                          marker=dict(size=3, color='red'))
                fig.add_trace(trace_line, row=1, col=1)

        # If trajectories provided, create AUC plot on right
        if trajectories is not None:
            auc_loss = AUCRegretLoss()
            auc_fig = auc_loss.plotly_plot(trajectories)

            # Add all traces from auc_fig to subplot col 2
            for trace_obj in auc_fig.data:
                fig.add_trace(trace_obj, row=1, col=2)

            # Update right subplot layout
            fig.update_xaxes(title_text="Timestep", row=1, col=2)
            fig.update_yaxes(title_text="Value", row=1, col=2)

        # General layout updates for 3D scene and figure size
        fig.update_layout(
            title="Interactive 2D Sinusoidal Environment and Trajectories",
            scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='f(x, y)'),
            width=1200,
            height=600,
            template='plotly_white'
        )
        fig.show()


if __name__ == '__main__':

    env = SinEnvTorch(seed=42, device='cpu')
    env.resample()
    env.plot3d(trace=(
        [0.1, 0.3, 0.5, 0.7, 0.9],  # example trace x
        [0.2, 0.4, 0.6, 0.8, 1.0]   # example trace y
    ), trajectories=torch.tensor([
        [0.5, 0.4, 0.6, 0.3, 0.2, 0.25],
        [0.6, 0.5, 0.55, 0.4, 0.35, 0.3]
    ]))