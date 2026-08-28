import torch


class BNNPrior:

    def __init__(
            self, batch_size, x_input_dim, device="cpu",
            n_draws=1000, samples_per_draw=1000, ecdf_n_samples=800,
            seed=None
    ):
        self.B, self.d, self.device = batch_size, x_input_dim, device
        if seed is not None:
            torch.manual_seed(seed)

        # FIXME: magic numbers should be configuralble!
        self.Lmax, self.Wmax = 16, 150
        self._sample()
        self._fit_ecdf(
            n_samples=ecdf_n_samples,  # this is more like a quantile thing
            n_draws=n_draws,
            samples_per_draw=samples_per_draw,
            pool_across_batch=True,
        )

    def _sample(self):
        B, d, Wmax, Lmax, dev = self.B, self.d, self.Wmax, self.Lmax, self.device
        self.depth = torch.randint(2, 17, (B,), device=dev)
        self.width = torch.randint(32, 128, (B,), device=dev)
        init_std = torch.empty(B, device=dev).uniform_(0.089, 0.193)
        unit_idx = torch.arange(Wmax, device=dev).unsqueeze(0).expand(B, -1)
        self.width_mask = (unit_idx < self.width.unsqueeze(1)).float()
        layer_idx = torch.arange(Lmax, device=dev).unsqueeze(0).expand(B, -1)
        self.layer_mask = (layer_idx < self.depth.unsqueeze(1)).float()

        s = init_std.view(B, 1, 1)
        self.W_in = torch.randn(B, d, Wmax, device=dev) * s
        self.b_in = torch.randn(B, Wmax, device=dev) * 0.1
        self.W_h = torch.randn(B, Lmax, Wmax, Wmax, device=dev) * init_std.view(B, 1, 1, 1)
        self.b_h = torch.randn(B, Lmax, Wmax, device=dev) * 0.1
        self.W_out = torch.randn(B, Wmax, 1, device=dev) * s
        self.b_out = torch.randn(B, 1, device=dev) * 0.1

    def _raw_forward(self, x):
        """x: [B,N,d] -> raw (unnormalized) y: [B,N]. Differentiable w.r.t. x
        (this is the 'privileged, known, differentiable dynamics' the search
        step and the pathwise RL term rely on)."""
        B, N, d = x.shape
        h = torch.tanh(torch.einsum("bnd,bdw->bnw", x, self.W_in) + self.b_in.unsqueeze(1)) * self.width_mask.unsqueeze(
            1)
        for l in range(self.Lmax):
            h_new = torch.tanh(torch.einsum("bnw,bwv->bnv", h, self.W_h[:, l]) + self.b_h[:, l].unsqueeze(1))
            h_new = h_new * self.width_mask.unsqueeze(1)
            m = self.layer_mask[:, l].view(B, 1, 1)
            h = m * h_new + (1 - m) * h
        return torch.einsum("bnw,bwo->bno", h, self.W_out).squeeze(-1) + self.b_out

    def _get_param_state(self):
        """Snapshots current architecture masks and weights."""
        keys = [
            "depth",
            "width",
            "width_mask",
            "layer_mask",
            "W_in",
            "b_in",
            "W_h",
            "b_h",
            "W_out",
            "b_out",
        ]
        return {k: getattr(self, k).clone() for k in keys}

    def _set_param_state(self, state):
        """Restores architecture masks and weights."""
        for k, v in state.items():
            setattr(self, k, v)

    def _fit_ecdf(
            self,
            n_samples,
            n_draws=50,
            samples_per_draw=200,
            pool_across_batch=True,  # should always be true
    ):
        """Builds a marginal prior ECDF cache by sampling across randomized

        architectures, weights, and inputs without blowing up GPU memory.

        :param n_samples. total number of evaluations Batch_size * n_draws * samples_per_draw
        :param n_draws. the number of complete Monte Carlo draws to perform (each draw samples a fresh architecture and weights)
        :param samples_per_draw. the number of random input samples (X) to evaluate per draw
        :param pool_across_batch. whether to pool across the batch dimension (B) when computing the ECDF.
        If True, all B batch slots share the same ECDF.
        This is exactly what we want! But in theory, one could also per-task normalize the outputs by computing a
         separate ECDF for each batch slot. But we want to have ecdf of the BNN prior, not of one instance!
        """
        # 1. Snapshot active weights so we don't clobber the instance's current state
        active_state = self._get_param_state()

        raw_accum = []
        with torch.no_grad():
            for _ in range(n_draws):
                # Sample fresh architectures and weights theta ~ p(theta)
                self._sample()

                # Sample random inputs x ~ U(0, 1) of shape [B, samples_per_draw, d]
                probe_x = torch.rand(
                    self.B, samples_per_draw, self.d, device=self.device
                )

                # Collect unnormalized forward outputs: [B, samples_per_draw]
                raw = self._raw_forward(probe_x)
                raw_accum.append(raw)

            # Concatenate all Monte Carlo draws along the sample dimension
            all_raw = torch.cat(
                raw_accum, dim=1
            )  # [B, n_draws * samples_per_draw]

            if pool_across_batch:
                # Pool across batch slots since all share the identical prior p(theta)
                # This yields (B * n_draws * samples_per_draw) total Monte Carlo samples
                all_raw = all_raw.flatten().unsqueeze(0)  # [1, total_samples]

            # 2. Sort along the sample dimension
            sorted_raw, _ = torch.sort(all_raw, dim=1)
            total_samples = sorted_raw.shape[1]

            # 3. Decimate (downsample) to exactly `n_samples` evenly-spaced quantiles
            idx = torch.linspace(
                0, total_samples - 1, n_samples, device=self.device
            ).long()
            decimated = sorted_raw[:, idx]

            if pool_across_batch:
                # Broadcast global sorted quantiles to shape [B, n_samples]
                self.ecdf_sorted = decimated.expand(self.B, -1).clone()
            else:
                self.ecdf_sorted = decimated  # [B, n_samples]

        # 4. Restore the original network weights for actual forward inference
        self._set_param_state(active_state)

    def evaluate(self, x):
        """Differentiable ECDF-normalized evaluation via linear interpolation
        against the per-env sorted reference (a smooth surrogate for the true
        step-function ECDF, so gradients through evaluate() are well-defined)."""
        raw = self._raw_forward(x)  # [B,N], differentiable
        ref = self.ecdf_sorted  # [B,S]
        S = ref.shape[1]
        idx = torch.searchsorted(ref, raw.detach(), right=True).clamp(1, S - 1)
        lo = torch.gather(ref, 1, idx - 1)
        hi = torch.gather(ref, 1, idx)
        frac = ((raw - lo) / (hi - lo + 1e-8)).clamp(0, 1)  # differentiable w.r.t. raw
        y_norm = (idx.float() - 1 + frac) / (S - 1)
        return y_norm.clamp(0, 1)


if __name__ == '__main__':
    # Test the BNNPrior class
    prior = BNNPrior(batch_size=4, x_input_dim=3, device="cpu", ecdf_n_samples=100)
    x_test = torch.rand(4, 10, 3)  # [B,N,d] = [4,10,3]
    y_norm = prior.evaluate(x_test)
    print("Normalized outputs:", y_norm)
