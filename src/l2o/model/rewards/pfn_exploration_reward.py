import torch


def sobol_monitor_generator(n, dimension):
    sobol = torch.quasirandom.SobolEngine(dimension=dimension, scramble=True)
    return sobol.draw(n)


class PFNExplorationReward:
    def __init__(self, pfn_model, device, monitor_sampler=None):
        self.pfn = pfn_model.to(device)
        self.pfn.eval()

        if monitor_sampler is None:
            monitor_sampler = lambda: sobol_monitor_generator(100, 2)  # Default: 10 Sobol points in 2D

        self.sample_monitor_points = monitor_sampler
        self.device = device

    def __call__(self, obs_traj, env):
        # 1. Concern: Global Information Query
        # Sample monitor points once per rollout call
        mon_x = self.sample_monitor_points().to(self.device)  # (M, 2)
        mon_y = env.functional_evaluate(mon_x, env.freq, env.phase)  # (Batch, M, 1)

        # 2. Concern: Triangular Batch Construction
        # We transform (Seq, Batch, Dim) -> (Seq * Batch, Seq + M, Dim)
        pfn_input_x, pfn_input_y, eval_pos = self._create_triangular_batch(obs_traj, mon_x)

        # 3. Concern: Batched Inference
        # pfn_bnn expects (Batch, Total_Len, Dim)
        with torch.no_grad():
            # The PFN predicts for everything after eval_pos (the monitor points)
            output = self.pfn((pfn_input_x, pfn_input_y), single_eval_pos=eval_pos)

            # Compute NLL of the monitor points given the varying horizons
            # target_y must be repeated to match the Seq * Batch expansion
            target_y = mon_y.repeat_interleave(obs_traj.size(0), dim=0)
            nll = self.pfn.criterion(output, target_y)  # (Seq * Batch,)

        # 4. Concern: Extracting Delta-Gain
        nll = nll.view(obs_traj.size(1), obs_traj.size(0))  # (Batch, Seq)
        nll = nll.permute(1, 0)  # (Seq, Batch)

        # Reward_t = NLL_{t-1} - NLL_t
        info_gain = nll[:-1] - nll[1:]
        # Pad first step with zero or a constant novelty
        first_step_gain = torch.zeros(1, obs_traj.size(1), device=self.device)
        return torch.cat([first_step_gain, info_gain], dim=0)

    # TODO: this might be a more efficient version :

    # def __call__(self, obs_traj, env):
    #         T, B, _ = obs_traj.shape
    #         mon_x = self.sample_monitor_points().to(self.device) # (M, 2)
    #         M = mon_x.size(0)
    #
    #         # 1. Ground Truth Eval (Once per env)
    #         mon_y = env.functional_evaluate(mon_x, env.freq, env.phase) # (B, M, 1)
    #
    #         # 2. Sequence Construction
    #         # We expand the batch to (T * B) to represent every horizon
    #         # But we use VIEWS to keep memory low.
    #         x_history = obs_traj[..., :2].permute(1, 0, 2) # (B, T, 2)
    #         y_history = obs_traj[..., 2:].permute(1, 0, 2) # (B, T, 1)
    #
    #         # Repeat history and monitor points for the "Horizon Batch"
    #         # full_x shape: (B * T, T + M, 2)
    #         full_x_hist = x_history.repeat_interleave(T, dim=0)
    #         full_y_hist = y_history.repeat_interleave(T, dim=0)
    #         full_x_mon = mon_x.unsqueeze(0).repeat(B * T, 1, 1)
    #
    #         full_x = torch.cat([full_x_hist, full_x_mon], dim=1)
    #
    #         # 3. Concern: The Horizon Mask
    #         # This is the "logic" the model is missing.
    #         # For each batch element 'i' in T, we mask history points > i.
    #         mask = self._create_horizon_mask(T, M, B).to(self.device)
    #
    #         with torch.no_grad():
    #             # Pass the mask into the PFN.
    #             # Note: This assumes your PFN forward accepts an 'attn_mask'
    #             # or a 'key_padding_mask' argument.
    #             logits = self.pfn(
    #                 (full_x, full_y_hist),
    #                 single_eval_pos=T,
    #                 attn_mask=mask
    #             )
    #
    #             # 4. Score the Monitor Points
    #             # logits: (B*T, M, Num_Classes)
    #             target_y = mon_y.repeat_interleave(T, dim=0) # (B*T, M, 1)
    #             nll = self.pfn.criterion(logits, target_y) # (B*T,)
    #
    #         # 5. Reshape and compute Delta
    #         nll = nll.view(B, T).permute(1, 0) # (T, B)
    #         info_gain = nll[:-1] - nll[1:]
    #         return torch.cat([torch.zeros(1, B, device=self.device), info_gain], dim=0) * self.weight
    #
    #     def _create_horizon_mask(self, T, M, B):
    #         """
    #         Creates a mask of shape (B*T, T+M, T+M)
    #         For a batch representing horizon 'h', points in history > h are masked.
    #         Monitor points (T:T+M) can see history (0:h) but NOT history (h:T).
    #         """
    #         # Create a single T x T causal mask for history
    #         history_mask = torch.tril(torch.ones(T, T))
    #
    #         # Create the mask for one set of horizons (T, T+M, T+M)
    #         full_mask = torch.zeros(T, T + M, T + M)
    #
    #         for h in range(T):
    #             # History part: horizon 'h' sees history up to 'h'
    #             full_mask[h, :T, :T] = history_mask[h].unsqueeze(0) * history_mask
    #
    #             # Monitor part: monitor points see history up to 'h'
    #             full_mask[h, T:, :h+1] = 1.0
    #
    #         # Convert to boolean mask (False means masked for many Transformer impls)
    #         # or additive mask (large negative)
    #         return full_mask.repeat(B, 1, 1) == 0

    def _create_triangular_batch(self, obs_traj, mon_x):
        T, B, _ = obs_traj.shape
        M = mon_x.size(0)

        # Repeat the trajectory T times to create horizons
        # We want a batch where:
        # Batch 0 has 1 point, Batch 1 has 2 points...
        x_raw = obs_traj[..., :2]  # (T, B, 2)
        y_raw = obs_traj[..., 2:]  # (T, B, 1)

        # Expand to (T, T, B, Dim) then flatten to (T*B, T, Dim)
        x_expanded = x_raw.unsqueeze(0).expand(T, -1, -1, -1)
        y_expanded = y_raw.unsqueeze(0).expand(T, -1, -1, -1)

        # Apply triangular mask to history
        # For horizon 'h', we only keep points 0...h
        mask = torch.tril(torch.ones(T, T, device=self.device)).view(T, T, 1, 1)
        x_masked = x_expanded * mask
        y_masked = y_expanded * mask

        # Concatenate monitor points to every horizon
        # x_masked is (T, T, B, 2), mon_x is (M, 2)
        mon_x_expanded = mon_x.view(1, 1, M, 2).expand(T, T, B, -1)
        # FIXME: this is unused !
        # Note: PFNs usually take x_train and x_test concatenated
        # We only pass Y for the train part

        # Reshape to (T*B, T+M, 2) for PFN
        # This is the "Triangular Matrix" concern
        final_x = torch.cat([
            x_masked.permute(0, 2, 1, 3),
            mon_x.view(1, 1, M, 2).expand(T, B, -1, -1)
        ],
            dim=2
        )
        final_x = final_x.reshape(T * B, T + M, 2)

        final_y = y_masked.permute(0, 2, 1, 3).reshape(T * B, T, 1)

        return final_x, final_y, T  # eval_pos is at the end of history
