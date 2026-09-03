"""FlowMatchingActionHead — a chunked, flow-matching action-expert head,
built as groundwork for the option `docs/ROADMAP.md`'s VLA-architecture
lens leaves open: does representing/committing to a multi-step plan (not
just finding a given argmax, already confirmed possible by
`pipelines/action_head_ei_diagnostic.py`) earn its place once genuine
multi-step targets exist (`search/kstep_explore.py`)? See
`docs/MILESTONES.md`'s prioritized-experiments list for why this is
deliberately sequenced *after*, not instead of, validating the simpler
single-shot `ActionHead` with better labels — this module is buildable and
testable now without that validation being finished, but training it
end-to-end against real k-step chunk labels and comparing it to the simple
head on `auc_improvement_vs_random` is not done here.

Reuses, rather than ports, pi0.5's actual mechanism where the reuse
assessment in `docs/ROADMAP.md` says to: the *rectified-flow* loss/sampler
shape (`x_t = t*noise + (1-t)*data`, constant target velocity `u_t = noise
- data`, `MSE(v_t, u_t)` training loss, Euler-integration sampling) is
small, generic, and not backbone-specific, so it's reproduced here near-
verbatim. pi0's own *attention* mechanism (one joint self-attention op over
concatenated expert-weighted streams) is NOT reused — this module keeps
`models.action_head.ActionHeadBlock`'s existing, genuine cross-attention
into the frozen PFN's per-layer hidden states instead, since that's already
a cleaner match to "cross-attend into a frozen backbone" than pi0 itself
is (see `docs/ROADMAP.md`'s VLA-architecture lens for the full argument).

Token design: one token per chunk step (`chunk_len` of them, each carrying
that step's current noisy value `x_t[:, i, :]` plus a shared flow-time
embedding and a learned per-position embedding to break the symmetry
between chunk steps -- unlike `search.kstep_explore`'s *interchangeable*
companion points, these tokens represent genuinely different, ordered
positions in the plan and must NOT be treated as symmetric), plus the same
4 aux-feature tokens `models.action_head.ActionHead` already uses. Velocity
is read out per chunk-step token (not from one summary token), so the head
predicts a full `[chunk_len, x_dim]` velocity field per forward call.

**Efficiency point this module exists to get right, not just mention**:
`compute_pfn_hidden_states` (`models/action_head.py`) is called exactly
ONCE per `compute_loss`/`sample` call, not once per denoising iteration --
`forward_velocity` below takes already-computed hidden states and is the
only thing that runs repeatedly inside the sampling loop. Skipping this
would mean paying a full PFN forward pass on every one of `num_steps`
denoising iterations, an avoidable multiplicative cost `docs/ROADMAP.md`
flags explicitly as a real prerequisite, not a style choice.
"""
import torch
import torch.nn as nn

from anytimeacquisition.models.action_head import AUX_FEATURE_NAMES, ActionHeadBlock, compute_pfn_hidden_states
from anytimeacquisition.models.pfn import PFN


class FlowMatchingActionHead(nn.Module):
    def __init__(
        self, pfn_d_model: int, pfn_n_layers: int, x_dim: int, chunk_len: int = 3,
        d_model: int = 64, n_heads: int = 4, d_ff: int = 128, dropout: float = 0.0,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.chunk_len = chunk_len
        self.x_embed = nn.Linear(x_dim, d_model)
        self.time_embed = nn.Linear(1, d_model)
        self.chunk_pos_embed = nn.Parameter(torch.randn(1, chunk_len, d_model) * 0.02)
        self.aux_embed = nn.ModuleDict({name: nn.Linear(1, d_model) for name in AUX_FEATURE_NAMES})
        self.blocks = nn.ModuleList([
            ActionHeadBlock(d_model, pfn_d_model, n_heads, d_ff, dropout) for _ in range(pfn_n_layers)
        ])
        self.out_ln = nn.LayerNorm(d_model)
        self.velocity_head = nn.Linear(d_model, x_dim)

    def forward_velocity(
        self, hidden_states: list[torch.Tensor], aux_features: dict, x_t: torch.Tensor, t: torch.Tensor,
    ) -> torch.Tensor:
        """The CHEAP part -- no PFN call in here, safe to call repeatedly in
        a denoising loop against the same `hidden_states`.
        x_t: [B, chunk_len, x_dim] -- the current noisy chunk.
        t: [B] -- current flow time (1 = pure noise, 0 = data), one scalar
        per batch instance, broadcast across chunk steps (matches pi0's own
        convention: one timestep per denoising step, shared across the
        whole chunk, not per-chunk-step).
        -> v_t [B, chunk_len, x_dim], the predicted denoising velocity.
        """
        B = x_t.shape[0]
        chunk_tokens = (
            self.x_embed(x_t) + self.time_embed(t.view(B, 1, 1).float()) + self.chunk_pos_embed.expand(B, -1, -1)
        )  # [B, chunk_len, d_model]
        aux_tokens = [self.aux_embed[name](aux_features[name].view(B, 1, 1).float()) for name in AUX_FEATURE_NAMES]
        h = torch.cat([chunk_tokens] + aux_tokens, dim=1)  # [B, chunk_len + len(AUX_FEATURE_NAMES), d_model]
        self_mask = torch.ones(h.shape[1], h.shape[1], dtype=torch.bool, device=h.device)

        for block, train_hidden in zip(self.blocks, hidden_states):
            h = block(h, train_hidden, self_mask)

        h = self.out_ln(h)
        chunk_repr = h[:, : self.chunk_len, :]  # only the chunk-step tokens' own final states are read out
        return self.velocity_head(chunk_repr)  # [B, chunk_len, x_dim]

    def compute_loss(
        self, pfn: PFN, x_train: torch.Tensor, y_train: torch.Tensor, aux_features: dict,
        target_chunk: torch.Tensor, blind: bool = False,
    ) -> torch.Tensor:
        """Rectified-flow training loss against `target_chunk` [B, chunk_len,
        x_dim] -- e.g. a `search.kstep_explore.kstep_explore_search` plan
        (`plan_star`), or `x_star` broadcast/padded for a simpler
        single-point-repeated target while chunk labels aren't wired in yet.
        One PFN forward pass total (via `compute_pfn_hidden_states`), same
        cost class as `ActionHead.forward`'s own single call.
        -> per-example loss [B] (mean over chunk_len and x_dim, matching
        pi0's own `Pi0.compute_loss` reduction)."""
        B = x_train.shape[0]
        hidden_states = compute_pfn_hidden_states(pfn, x_train, y_train, blind=blind)

        noise = torch.randn_like(target_chunk)
        # time ~ Beta(1.5, 1) * 0.999 + 0.001, skewed toward t=1 (pure
        # noise) -- same non-uniform sampling openpi's Pi0.compute_loss
        # uses, spends more training signal on the harder, noisier end of
        # the denoising trajectory.
        time = torch.distributions.Beta(1.5, 1.0).sample((B,)).to(target_chunk.device) * 0.999 + 0.001
        t_broadcast = time.view(B, 1, 1)
        x_t = t_broadcast * noise + (1.0 - t_broadcast) * target_chunk
        u_t = noise - target_chunk  # constant target velocity, doesn't depend on t

        v_t = self.forward_velocity(hidden_states, aux_features, x_t, time)
        return ((v_t - u_t) ** 2).mean(dim=(1, 2))  # [B]

    @torch.no_grad()
    def sample(
        self, pfn: PFN, x_train: torch.Tensor, y_train: torch.Tensor, aux_features: dict,
        num_steps: int = 10, blind: bool = False,
    ) -> torch.Tensor:
        """Euler-integration sampling from t=1 (pure noise) to t=0 (data),
        `num_steps` denoising iterations (decoupled from `chunk_len` --
        planning-horizon length and denoising-step count are independent
        knobs, see module docstring). ONE PFN forward pass total, not one
        per step (`compute_pfn_hidden_states` called once, outside the
        loop). Clamped to `[0,1]^x_dim` only at the end, not every step --
        a deliberate simplification (rectified-flow's ODE isn't guaranteed
        to stay in-domain mid-integration; only the final chunk needs to be
        valid).
        -> x_0 [B, chunk_len, x_dim], the sampled action chunk. `x_0[:, 0]`
        is the only point ever meant to be deployed -- see
        `search.kstep_explore`'s own module docstring for why the rest is
        planning leverage, not something to execute blindly."""
        B = x_train.shape[0]
        hidden_states = compute_pfn_hidden_states(pfn, x_train, y_train, blind=blind)

        x_t = torch.randn(B, self.chunk_len, self.x_dim, device=x_train.device)
        dt = -1.0 / num_steps
        t = torch.ones(B, device=x_train.device)
        for _ in range(num_steps):
            v_t = self.forward_velocity(hidden_states, aux_features, x_t, t)
            x_t = x_t + dt * v_t
            t = t + dt
        return x_t.clamp(0.0, 1.0)


def flow_action_head_policy_fn(action_head: "FlowMatchingActionHead", pfn: PFN, n_steps: int, num_sample_steps: int = 10):
    """`FlowMatchingActionHead`'s analogue of
    `models.action_head.action_head_policy_fn` -- same
    `policy_fn(x_context, y_context, x_dim) -> [B, x_dim]` contract
    `trainer.exit_rollout.rollout_episode` expects, same closure-tracked
    step counter/incumbent history for the aux features. The one
    difference: `sample()` returns a full `[B, chunk_len, x_dim]` plan;
    only `[:, 0]` is ever returned as the action actually played, matching
    every other k-step-flavored mechanism in this repo (only the first
    planned point is real, the rest is discarded, replanned fresh next
    call)."""
    trend_window = 3
    state = {"step": 0, "incumbent_history": []}

    def policy_fn(x_context: torch.Tensor, y_context: torch.Tensor, x_dim: int) -> torch.Tensor:
        step = state["step"]
        incumbent = y_context.min(dim=1).values  # [B]
        state["incumbent_history"].append(incumbent.detach())

        if step < trend_window:
            improvement_trend = torch.zeros_like(incumbent)
        else:
            incumbent_prev = state["incumbent_history"][step - trend_window]
            improvement_trend = (
                torch.log(incumbent_prev.clamp_min(1e-12)) - torch.log(incumbent.clamp_min(1e-12))
            ).clamp_min(0.0)

        aux = {
            "step_count": torch.full_like(incumbent, float(step)),
            "remaining_budget": torch.full_like(incumbent, (n_steps - step) / n_steps),
            "incumbent_value": incumbent,
            "improvement_trend": improvement_trend,
        }
        chunk = action_head.sample(pfn, x_context, y_context, aux, num_steps=num_sample_steps)  # [B, chunk_len, x_dim]
        state["step"] += 1
        return chunk[:, 0]

    return policy_fn


if __name__ == "__main__":
    from anytimeacquisition.pipelines.train_pfn import load_pfn_checkpoint
    from anytimeacquisition.priors.bnn import BNNPrior
    from anytimeacquisition.models.action_head import pfn_dims
    from anytimeacquisition.utils.paths import CHECKPOINT_DIR

    checkpoint_path = CHECKPOINT_DIR / "pfn_smoke_xdim1.pt"
    if not checkpoint_path.exists():
        raise SystemExit(
            f"No checkpoint at {checkpoint_path} -- train one first:\n"
            "  uv run python -m anytimeacquisition.pipelines.train_pfn "
            "experiment=pfn_smoke_xdim1 allow_dirty=true"
        )
    pfn, bar_dist, ckpt = load_pfn_checkpoint(checkpoint_path)
    print(f"loaded PFN checkpoint: {checkpoint_path.name}, config={ckpt['config']}")

    d_model, n_layers = pfn_dims(pfn)
    x_dim = ckpt["config"]["max_x_dim"]
    chunk_len = 3
    torch.manual_seed(0)
    head = FlowMatchingActionHead(pfn_d_model=d_model, pfn_n_layers=n_layers, x_dim=x_dim, chunk_len=chunk_len)

    B = 4
    prior = BNNPrior(batch_size=B, x_dim=x_dim, seed=1)
    x_train, y_train, _, _ = prior.sample_episode(n_train=10, n_test=0)
    aux_features = {
        "step_count": torch.arange(B).float(),
        "remaining_budget": torch.full((B,), 0.8),
        "incumbent_value": y_train.min(dim=1).values,
        "improvement_trend": torch.zeros(B),
    }

    # Stand-in chunk target (not a real kstep_explore_search plan -- this
    # demo is about the head's own mechanics, not label quality): the
    # context's own incumbent point, repeated chunk_len times, is at least
    # a valid, in-domain [0,1] target to smoke-test the loss/sampler shapes
    # against.
    incumbent_idx = y_train.argmin(dim=1)
    incumbent_x = x_train[torch.arange(B), incumbent_idx]  # [B, x_dim]
    target_chunk = incumbent_x.unsqueeze(1).expand(B, chunk_len, x_dim).clone()

    loss = head.compute_loss(pfn, x_train, y_train, aux_features, target_chunk)
    print("compute_loss output:", loss.shape, loss.detach())
    assert loss.shape == (B,)

    loss.sum().backward()
    pfn_grads = [p.grad for p in pfn.parameters()]
    head_grads = [p.grad for p in head.parameters()]
    print("PFN params with a gradient after backward (expect 0):", sum(g is not None for g in pfn_grads))
    print("FlowMatchingActionHead params with a gradient after backward (expect all):",
          sum(g is not None for g in head_grads), "/", len(head_grads))
    assert all(g is None for g in pfn_grads), "PFN must receive zero gradient -- it's frozen"
    assert all(g is not None for g in head_grads), "every head param should get a gradient"

    sampled = head.sample(pfn, x_train, y_train, aux_features, num_steps=10)
    print("sampled chunk:", sampled.shape, sampled)
    assert sampled.shape == (B, chunk_len, x_dim)
    assert (sampled >= 0.0).all() and (sampled <= 1.0).all(), "sampled chunk must stay in [0,1]^x_dim"

    print("smoke forward + backward + sample OK: rectified-flow loss trains, "
          "PFN gradient-isolated, sampler produces valid in-domain chunks.")
