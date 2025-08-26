import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt


class AUCRegretLoss(torch.nn.Module):

    def forward(self, trajectories):
        # trajectories: tensor of shape (batch_size, seq_len)
        min_val = trajectories.min()
        # Each trajectory AUC above min
        aucs = torch.trapz(trajectories - min_val, dim=1)
        return aucs.mean()

    def plot(self, trajectories, ax):
        min_val = trajectories.min().item()
        losses = torch.trapz(trajectories - min_val,
                             dim=1)  # same loss as forward

        for i, traj in enumerate(trajectories.cpu().numpy()):
            ax.plot(traj, alpha=0.3, label=f"Batch item {i}, loss: {losses[i]:.2f}")
            ax.fill_between(range(len(traj)), min_val, traj, alpha=0.2)
        ax.axhline(min_val, color='red', linestyle='--', label=f'Batch Min [{min_val:.2f}]')
        ax.legend()
        ax.set_title("Trajectory AUC Above Min Value")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Value")

        return ax

    def plotly_plot(self, trajectories):
        min_val = trajectories.min().item()
        losses = torch.trapz(trajectories - min_val, dim=1).cpu().numpy()

        fig = go.Figure()

        for i, traj in enumerate(trajectories.cpu().numpy()):
            x_vals = list(range(len(traj)))
            # Plot trajectory line
            fig.add_trace(go.Scatter(
                x=x_vals, y=traj,
                mode='lines',
                name=f"Trajectory {i} (Loss: {losses[i]:.3f})",
                line=dict(width=2),
                opacity=0.5
            ))
            # Fill area between min_val and trajectory
            fig.add_trace(go.Scatter(
                x=x_vals + x_vals[::-1],
                y=list(traj) + [min_val] * len(traj),
                fill='toself',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(color='rgba(255,0,0,0)'),
                showlegend=False,
                hoverinfo='skip'
            ))

        # Horizontal line for min_val
        fig.add_trace(go.Scatter(
            x=[0, len(trajectories[0]) - 1],
            y=[min_val, min_val],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name=f"Batch Min [{min_val:.3f}]"
        ))

        fig.update_layout(
            title="Trajectory AUC Above Min Value",
            xaxis_title="Timestep",
            yaxis_title="Value",
            showlegend=True,
            template='plotly_white'
        )
        return fig


if __name__ == '__main__':
    loss = AUCRegretLoss()
    traj = torch.tensor([[0.5, 0.4, 0.6, 0.3, 0.2, 0.25],
                         [0.6, 0.5, 0.55, 0.4, 0.35, 0.3]], dtype=torch.float32)
    print("AUC Regret Loss:", loss(traj).item())
    # fig, ax = plt.subplots(figsize=(8, 5))
    # loss.plot(traj, ax)
    # plt.show()

    fig = loss.plotly_plot(traj)
    fig.show()


