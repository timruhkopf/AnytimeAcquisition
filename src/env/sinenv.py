import torch
import math
import numpy as np
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
from plotly.offline import plot

from src.loss.auc import AUCRegretLoss


class SinEnv:
    def __init__(self, a1=0.9, b1=2, phi1=0.0, a2=0.3, b2=1, phi2=0.7, m=-0.6, theta=0.3,
                 seed=None, device='cpu', logger=None):
        """
        Sinusoidal + linear trend + rotation in 2D with PyTorch tensors.
        Domain normalized to [0,1] x [0,1].

        f(x, y) = a1 * sin(b1 * 2π * x' + phi1) + a2 * sin(b2 * 2π * y' + phi2)
                  + m * ((x' + y')/2)
        x', y' = rotated coordinates

        :param a1: Amplitude of the sine wave for the first component.
        :param b1: Frequency of the sine wave for the first component.
        :param phi1: Phase shift of the sine wave for the first component.
        :param a2: Amplitude of the sine wave for the second component.
        :param m: Slope of the linear trend.
        :param theta: Rotation angle in radians.
        :param seed: Random seed for reproducibility.
        :param device: 'cpu' or 'cuda' device for tensors.
        """
        if seed is not None:
            torch.manual_seed(seed)
        self.device = device
        self.a1 = torch.tensor(a1, device=device)
        self.a2 = torch.tensor(a2, device=device)
        self.b1 = torch.tensor(b1, device=device)
        self.b2 = torch.tensor(b2, device=device)
        self.phi1 = torch.tensor(phi1, device=device)
        self.phi2 = torch.tensor(phi2, device=device)
        self.m = torch.tensor(m, device=device)
        self.theta = torch.tensor(theta, device=device)

        self.min, self.max = self.get_bounds()

    def get_bounds(self):

        # determine min and max on a grid for normalization
        X, Y, Z = self.grid(n=1000)

        return Z.min().item(), Z.max().item()

    def resample(self, a_range=(0.5, 2.0), b_range=(1, 10), phi_range=(0, 2 * math.pi),
                 m_range=(-1, 1), theta_range=(0, 2 * math.pi)):
        """Randomly sample new environment parameters from given ranges (torch.uniform)."""
        self.a = (a_range[1] - a_range[0]) * torch.rand(1, device=self.device) + a_range[0]
        self.b = (b_range[1] - b_range[0]) * torch.rand(1, device=self.device) + b_range[0]
        self.phi = (phi_range[1] - phi_range[0]) * torch.rand(1, device=self.device) + phi_range[0]
        self.m = (m_range[1] - m_range[0]) * torch.rand(1, device=self.device) + m_range[0]
        self.theta = (theta_range[1] - theta_range[0]) * torch.rand(1, device=self.device) + \
                     theta_range[0]

        self.min, self.max = self.get_bounds()

    def rotate(self, x, y):
        # x, y: tensors of same shape
        x_r = torch.cos(self.theta) * x + torch.sin(self.theta) * y
        y_r = -torch.sin(self.theta) * x + torch.cos(self.theta) * y
        return x_r, y_r

    def _evaluate(self, X):
        """
        Evaluate the function at (x, y). x,y should be tensors or convertible to tensors.
        Values should be in [0,1].
        """

        x, y = X[..., 0], X[..., 1]
        x = x.to(self.device) if isinstance(x, torch.Tensor) else torch.tensor(x,
                                                                               device=self.device)
        y = y.to(self.device) if isinstance(y, torch.Tensor) else torch.tensor(y,
                                                                               device=self.device)
        x_r, y_r = self.rotate(x, y)
        val = (self.a1 * torch.sin(self.b1 * 2 * math.pi * x_r + self.phi1)
               + self.a2 * torch.sin(self.b2 * 2 * math.pi * y_r + self.phi2)
               + self.m * ((x_r + y_r) / 2.0))

        return val

    def evaluate(self, X):
        y = self._evaluate(X)

        # normalize & clamp to [0,1]
        y = (y - self.min) / (self.max - self.min + 1e-9)
        y = y.clamp(0, 1)
        return y

    def grid(self, n=201):
        xs = np.linspace(0, 1, n)
        ys = np.linspace(0, 1, n)
        X, Y = np.meshgrid(xs, ys)
        X, Y = torch.tensor(X, device=self.device), torch.tensor(Y, device=self.device)
        Z = self._evaluate(torch.stack([X, Y], axis=-1)).cpu().numpy()
        return X, Y, Z

    def approx_global_min(self, n=401):
        X, Y, Z = self.grid(n)
        idx = Z.argmin()
        i, j = np.unravel_index(idx, Z.shape)
        return (X[i, j], Y[i, j], Z[i, j])

    def sample_initial_condition(self, B=16):
        """
        Samples B random initial conditions (x,y,f(x,y)) uniformly in [0,1]^2.
        """
        X = torch.rand(B, 1, 3, device=self.device)

        X[..., -1] = self.evaluate(X[..., :-1])
        return X


    def plot3d_matplotlib(self, resolution=200, traces=None, plot_points_only=False):
        """
        Simplified 3D surface plot with optional trajectory paths in matplotlib.

        Params:
        - resolution: grid resolution for surface.
        - traces: torch tensor (batch_size, seq_len, 2) of (x, y) trajectory points.
        - plot_points_only: if True, plot only points for trajectories.
        """

        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # needed for 3d projection

        # Generate surface grid
        X = torch.linspace(0, 1, resolution, device=self.device)
        Y = torch.linspace(0, 1, resolution, device=self.device)
        XX, YY = torch.meshgrid(X, Y, indexing='xy')
        flat_in = torch.stack([XX.flatten(), YY.flatten()], dim=-1)
        ZZ = self.evaluate(flat_in).reshape(XX.shape)

        X_np, Y_np, Z_np = XX.cpu().numpy(), YY.cpu().numpy(), ZZ.cpu().numpy()

        fig, (ax3d, ax2d) = plt.subplots(1, 2, figsize=(14, 6),
                                         gridspec_kw={'width_ratios': [2, 1]})

        ax3d.axis('off')

        # 3D Surface plot without axes/bounding box
        ax3d = fig.add_subplot(1, 2, 1, projection='3d')
        surf = ax3d.plot_surface(X_np, Y_np, Z_np, cmap='viridis', alpha=0.85, edgecolor='none')

        # Set angled top view
        ax3d.view_init(elev=80, azim=45)

        # Plot trajectories on surface if provided
        if traces is not None:
            traces_np = traces.cpu().numpy()
            batch_size, seq_len, _ = traces_np.shape

            for traj in traces_np:
                traj_torch = torch.tensor(traj, device=self.device)
                Z_traj = self.evaluate(traj_torch[..., :-1]).cpu().numpy()

                if plot_points_only:
                    ax3d.scatter(traj[:, 0], traj[:, 1], Z_traj, color='red', s=20)
                else:
                    ax3d.plot(traj[:, 0], traj[:, 1], Z_traj, marker='o', linewidth=1, color='red')

        # fig.colorbar(surf, ax=ax3d, shrink=0.5, aspect=10)

        # 2D incumbent plot - only raw curve with markers on incumbent changes
        if traces is not None:
            traces_np = traces.cpu().numpy()
            batch_size, seq_len, _ = traces_np.shape

            # Evaluate performance sequences
            performance = np.zeros((batch_size, seq_len))
            for i in range(batch_size):
                traj_torch = torch.tensor(traces_np[i], device=self.device)
                performance[i] = self.evaluate(traj_torch[..., :-1]).cpu().numpy()

            for i in range(batch_size):
                seq = performance[i]
                ax2d.plot(range(seq_len), seq, label=f'Seq {i}')

                # Find points where incumbent (min so far) changes
                incumbent = np.minimum.accumulate(seq)
                changed_indices = np.where(np.diff(incumbent, prepend=np.inf) < 0)[0]
                ax2d.scatter(changed_indices, seq[changed_indices], color='black', marker='x', s=50)

            ax2d.set_xlabel('Timestep')
            ax2d.set_ylabel('Performance (lower better)')
            ax2d.set_title('Raw Trajectories with Incumbent Change Markers')
            ax2d.legend(fontsize='small', loc='upper right')

        plt.tight_layout()
        plt.show()

    def plot3d(self, fig=None, resolution=200, traces=None, plot_points_only=False,
               trajectories=None,
               static=True):
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

        flat_in = torch.stack([XX.flatten(), YY.flatten()], dim=-1)
        ZZ = self.evaluate(flat_in).reshape(XX.shape)
        # # ZZ_raw = self.evaluate(XX, YY)
        # vmin = ZZ_raw.min()
        # vmax = ZZ_raw.max()
        # ZZ = (ZZ_raw - vmin) / (vmax - vmin + 1e-9)

        # Convert to numpy arrays for Plotly
        # ZZ = self.normalized(XX, YY)

        # Numpy for plotting
        XX_np, YY_np, ZZ_np = XX.cpu().numpy(), YY.cpu().numpy(), ZZ.cpu().numpy()

        # Setup subplot figure: 1 row, 2 cols if trajectories given, else just 1 col
        if fig is None:
            if traces is not None:
                fig = make_subplots(
                    rows=1, cols=2,
                    specs=[[{'type': 'surface'}, {'type': 'xy'}]],
                    subplot_titles=("Environment Surface", "Trajectories AUC")
                )
            else:
                fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'surface'}]])


        # Surface plot on left
        surface = go.Surface(z=ZZ_np, x=XX_np, y=YY_np, colorscale='Viridis', opacity=0.85)
        fig.add_trace(surface, row=1, col=1)

        # Add trace path on surface if given
        if traces is not None:
            X_trace, Y_trace = traces[..., :2].cpu().numpy(), traces[
                ..., -1].flatten().cpu().numpy()
            Z_trace = self.evaluate(X_trace).cpu().numpy()
            # Z_trace = ((Z_trace_raw - vmin) / (vmax - vmin + 1e-9)).cpu().numpy()
            if plot_points_only:
                trace_points = go.Scatter3d(
                    x=X_trace, y=Y_trace, z=Z_trace, mode='markers',
                    marker=dict(size=5, color='red')
                )
                fig.add_trace(trace_points, row=1, col=1)
            else:
                colors = np.linspace(0, 1, X_trace.shape[0])

                for b in range(X_trace.shape[0]):
                    trace_line = go.Scatter3d(
                        x=X_trace[b, :, 0], y=X_trace[b, :, 1], z=Z_trace[b],
                        mode='lines+markers',
                        line=dict(color=colors[b], width=2),
                        marker=dict(color=colors[b], size=2)
                    )
                    fig.add_trace(trace_line, row=1, col=1)

        # If trajectories provided, create AUC plot on right
        if traces is not None:
            auc_loss = AUCRegretLoss()
            y = self.evaluate(traces[..., :2])
            auc_loss.plotly_plot(y, fig=fig, row=1, col=2)

            # Add all traces from auc_fig to subplot col 2
            # for trace_obj in auc_fig.data:
            #     fig.add_trace(trace_obj, row=1, col=2)

            # Update right subplot layout
            fig.update_xaxes(title_text="Timestep", row=1, col=2)
            fig.update_yaxes(title_text="Value", row=1, col=2)

        # General layout updates for 3D scene and figure size
        fig.update_layout(
            title="Interactive 2D Sinusoidal Environment and Trajectories",
            scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='f(x, y)'),
            width=1200,
            height=600,
            template='plotly_white',
            yaxis_range=[0, 1]
        )
        return fig


if __name__ == '__main__':
    env = SinEnv(seed=42, device='cpu')
    env.resample()

    print("Approx global min (grid):", env.approx_global_min())

    env.plot3d(traces=torch.rand(3, 10, 3))
