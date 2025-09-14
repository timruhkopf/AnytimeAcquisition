import torch


class AUCRegretLoss(torch.nn.Module):

    def forward(self, X, y):
        # trajectories: tensor of shape (batch_size, seq_len)
        min_val = y.min()  # best discovered value
        # Each trajectory AUC above min
        aucs = torch.trapz(y - min_val, dim=1)


        # FIXME: We actually want the model to pull towards better X --> i.e. we need to do a MSE
        #  loss on X values where each x token is weighted by its AUC regret of the trajectory
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

    def plotly_plot(self, traces, fig=None, row=1, col=1):

        import plotly.graph_objects as go

        min_val = traces.min().item()
        losses = torch.trapz(traces, dim=1).cpu().numpy()

        if fig is None:
            fig = go.Figure()
            kwargs = {}
        else:
            kwargs = {'row': row, 'col': col}

        for i, traj in enumerate(traces.cpu().numpy()):
            x_vals = list(range(len(traj)))
            # Plot trajectory line
            fig.add_trace(go.Scatter(
                x=x_vals, y=traj,
                mode='lines',
                # name=f"Trajectory {i} (Loss: {losses[i]:.3f})",
                line=dict(width=2),
                opacity=0.5
            ), **kwargs)
            # Fill area between min_val and trajectory
            fig.add_trace(go.Scatter(
                x=x_vals + x_vals[::-1],
                y=list(traj) + [min_val] * len(traj),
                fill='toself',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(color='rgba(255,0,0,0)'),
                showlegend=False,
                hoverinfo='skip'
            ), **kwargs)

        # Horizontal line for min_val
        fig.add_trace(go.Scatter(
            x=[0, len(traces[0]) - 1],
            y=[min_val, min_val],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name=f"Batch Min [{min_val:.3f}]"
        ), **kwargs)

        if fig is None:
            fig.update_layout(
                title="Trajectory AUC Above Min Value",
                xaxis_title="Timestep",
                yaxis_title="Value",
                showlegend=True,
                template='plotly_white'
            )
            fig.show()


class AUCWeightedMSELoss(AUCRegretLoss):

    def __init__(self, soft_penalize_exploration=True, **kwargs):
        super().__init__()
        self.soft_penalize_exploration = soft_penalize_exploration
        self.kwargs = kwargs

    @staticmethod
    def shift_segments(x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, L) with repeated segments of integers.
        Returns:
            y: Tensor of shape (B, L) where each segment is replaced
               by the next distinct value in that row (last segment stays as is).
        """
        B, L = x.shape
        y = torch.empty_like(x)
        for b in range(B):
            row = x[b]
            # Find where value changes
            change_points = torch.nonzero(row[1:] != row[:-1], as_tuple=False).squeeze(-1) + 1
            starts = torch.cat([torch.tensor([0], device=x.device), change_points])
            ends = torch.cat([change_points, torch.tensor([L], device=x.device)])
            # Fill targets
            for i in range(len(starts)):
                if i < len(starts) - 1:
                    y[b, starts[i]:ends[i]] = row[starts[i + 1]]
                else:
                    y[b, starts[i]:ends[i]] = row[starts[i]]  # last segment stays as is
        return y

    def linear_penalty(self, y, inc_indices, cap=5):
        # PENALTY SCHEME:
        # define a penalty factor (always non-zero!). this penalty factor is a slack variable,
        #  that will allow us to explore unpenalized; specifically, we don't want to penalize
        #  early steps after an incumbent change, because these are exploratory steps.
        # but we don't want to penalize too much, so it will have to be capped to a max value of 1.

        # find the number of steps between incumbent changes: ; basically a counter of steps
        # between incumbents (increasing for any steps between incumbents, reset to 0 at
        # incumbent change)
        B, T = y.shape
        n_steps = torch.zeros_like(y)
        for b in range(B):
            count = 0
            for t in range(1, T):
                if inc_indices[b, t] != inc_indices[b, t - 1]:
                    count = 0
                else:
                    count += 1
                n_steps[b, t] = count

        if isinstance(cap, int):
            n_steps = torch.clamp(n_steps, max=cap)
            penalty = (n_steps.float() / cap)

        elif cap is None:
            # sample cap per batch item uniformly
            caps = torch.randint(1, T, (B,1), device=y.device).float()
            n_steps = torch.clamp(n_steps, max=caps).float()
            penalty = (n_steps / torch.min(caps, torch.max(n_steps)))

        else:
            raise ValueError("cap must be int or None")

        return penalty

    def forward(self, X, y):
        """
        Technically, we do not compute auc here, but it is inspired by the incumbent auc
        We want to pull the x values towards the next incumbent x values, weighted by the regret
        of the y values over that incumbents performance.
        The main idea is, that we want to "chip away" unnecessary steps and pull the incumbents
        together, slimming down the trajectory towards the optimal path.
        However, without exploration this will collapse in local optima,
        so we add a damping factor, which does not penalize exploration steps as much,
        if they are after an incumbent change. This dampener is capped to a max number of steps,
        after which we penalize fully again.
        Since this is an important hyperparameter, and we don't know its optimal value apriori,
        we sample the cap

        :param X:
        :param y:
        :return:
        """
        # compute the incumbent curve :
        inc_values, inc_indices = torch.cummin(y, dim=1)

        # get the X sequence of incumbent values (shifted by one incumbent)
        # i.e. we want to pull the y values towards the next incumbent.
        target_indices = self.shift_segments(inc_indices)

        # FIXME: we will have zero loss for the last incumbent segment. can we maybe try local
        #  sampling around it

        # collect the y_values of the shifted
        # fixme: we probably want to inversely penalize with the regret here:
        #  this means that incumbent changes with small difference in y should be penalized more,
        #  encouraging to collapse small incumbent changes
        # regret = distance from best-so-far value
        regret = (y - inc_values).clamp(min=0)  # (B, L)

        # exploration mask = 1 if not incumbent, 0 if incumbent
        exploration_mask = (inc_indices != torch.arange(
            y.size(1), device=y.device
        ).unsqueeze(0)).float()  # shape (B,L)

        # # normalize regret to [0, 1]
        # regret_norm = regret / (regret.max(dim=1, keepdim=True).values + 1e-9)
        #
        # # invert regret: small regret -> large weight
        # inv_regret = 1.0 - regret_norm

        # regret = regret / (regret.max(dim=1, keepdim=True).values + 1e-9)  # Normalize to [0,1]

        X = X[..., :-1]  # remove y values from input
        target_X = torch.gather(X, 1, target_indices.unsqueeze(-1).expand(-1, -1, X.size(-1)))

        if self.soft_penalize_exploration:
            penalty = self.linear_penalty(y, inc_indices, cap=self.kwargs.get('penalty_cap', None))
        else:
            penalty = 1.0

        loss = ((target_X -X) ** 2).sum(-1)  # * penalty # *  inv_regret
        return -loss.mean()

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
