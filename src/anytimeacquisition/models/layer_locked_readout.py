"""`LayerLockedReadout` -- a VLA-action-expert-style readout (the π0.5
precedent already cited in `docs/PROBLEM_SETTING.md`) into a frozen PFN's
per-layer train-token hidden states, built as an experimental alternative
to `docs/ROADMAP.md` §3.2's flat design ("concatenate all `L` layers into
one `L·t`-token sequence, read with a single shallow cross-attention
block"). Instead: a same-depth (`n_layers` matches the frozen PFN),
smaller-width (`d_expert < d_model_pfn`) expert stack that cross-attends
into ONE layer's worth of train-token hidden states per expert layer,
sequentially -- giving the readout itself genuine depth (nonlinear
processing *between* layer-reads) rather than a single flat attention pool
over the concatenation, at the same total cross-attention FLOP count (both
designs do `n_query * L * t` key comparisons in total; this one just
structures them sequentially instead of as one `L·t`-wide softmax).

See `notebooks/vla_readout_ei_probe.ipynb` for the experiment this was
built for: can this readout recover EI (closed-form, known exactly from the
frozen PFN's own bar distribution at the query) using ONLY the frozen
train-token hidden states as input -- the same information budget a real
future policy would have, unlike reading the frozen model's own
already-computed query-conditioned PPD directly. That result is what
should decide whether §3.2 actually changes, not this module's existence
on its own.

Every candidate/query token is scored independently (no candidate-candidate
self-attention) -- matches §3.4's pointwise-scorer decision; this module
isolates the readout-mechanism question, not a second architecture change.

**Incumbent anchor: an additive marker, not a synthetic token (revised
2026-09-08, flagged by user review).** The first version encoded the
incumbent's y-VALUE through a fresh, untrained `Linear(1, d_model_pfn)`
per layer, concatenated as an extra key/value pair. That re-derives
information the frozen PFN already computed: the incumbent is literally
ONE of the `t` train tokens, so `layer_hidden_states[l][b, incumbent_idx]`
already encodes its y-value richly through the frozen PFN's own pretrained
`train_embed`. Marking that existing token (a small learned per-layer
`nn.Parameter`, ADDED to its hidden state -- not deriving a new one from
its raw value) reuses that pretrained representation instead of asking a
fresh linear map to reconstruct it from a bare scalar, and doesn't grow
the KV sequence. `incumbent_idx` (`y_train.argmin(dim=1)`) is the caller's
responsibility, same division of labor as before: this module only ever
sees the frozen PFN's own representations, never raw `x`/`y`.
"""
import torch
import torch.nn as nn


class CrossAttnBlock(nn.Module):
    """One expert layer: `q_input` (`[B,Q,d_q]`) cross-attends into
    `kv_input` (`[B,Tk,d_kv]`) -- `k`/`v` project `kv_input` into the
    expert's own `d_q` space, so `d_kv` (the frozen PFN's `d_model`) may
    differ freely from `d_q` (the expert's own, smaller width). Residual +
    LayerNorm + FFN, all in the expert's own `d_q` space -- `kv_input` is
    read, never written to."""

    def __init__(self, d_q: int, d_kv: int, n_heads: int, d_ff: int):
        super().__init__()
        assert d_q % n_heads == 0
        self.h = n_heads
        self.dk = d_q // n_heads
        self.ln_q = nn.LayerNorm(d_q)
        self.q_proj = nn.Linear(d_q, d_q)
        self.k_proj = nn.Linear(d_kv, d_q)
        self.v_proj = nn.Linear(d_kv, d_q)
        self.out_proj = nn.Linear(d_q, d_q)
        self.ln_ff = nn.LayerNorm(d_q)
        self.ff = nn.Sequential(nn.Linear(d_q, d_ff), nn.GELU(), nn.Linear(d_ff, d_q))

    def forward(self, q_input: torch.Tensor, kv_input: torch.Tensor) -> torch.Tensor:
        B, Q, _ = q_input.shape
        Tk = kv_input.shape[1]
        q_normed = self.ln_q(q_input)
        q = self.q_proj(q_normed).view(B, Q, self.h, self.dk).transpose(1, 2)
        k = self.k_proj(kv_input).view(B, Tk, self.h, self.dk).transpose(1, 2)
        v = self.v_proj(kv_input).view(B, Tk, self.h, self.dk).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / (self.dk**0.5)
        attn = scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, Q, -1)
        q_input = q_input + self.out_proj(out)
        q_input = q_input + self.ff(self.ln_ff(q_input))
        return q_input


class LayerLockedReadout(nn.Module):
    def __init__(
        self, d_model_pfn: int, d_expert: int, n_heads: int, n_layers: int, d_ff: int,
        max_x_dim: int, n_out: int,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.query_embed = nn.Linear(max_x_dim, d_expert)
        # One marker per layer, not shared across layers: the frozen hidden
        # states at different depths live in genuinely different
        # representational spaces, so "this is the incumbent" needs a
        # matching per-layer marker to stay meaningful at that layer's kv.
        self.incumbent_marker = nn.Parameter(torch.zeros(n_layers, d_model_pfn))
        nn.init.normal_(self.incumbent_marker, std=0.02)
        self.cross_blocks = nn.ModuleList(
            [CrossAttnBlock(d_expert, d_model_pfn, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.out_ln = nn.LayerNorm(d_expert)
        self.head = nn.Linear(d_expert, n_out)

    def forward(
        self, x_query: torch.Tensor, layer_hidden_states: list[torch.Tensor], incumbent_idx: torch.Tensor,
    ) -> torch.Tensor:
        """x_query: `[B,Q,max_x_dim]`. `layer_hidden_states`: `n_layers`
        tensors, each `[B,t,d_model_pfn]` -- the FROZEN PFN's train-token
        hidden state at that layer (caller's responsibility to run the
        frozen forward under `no_grad`; this module doesn't assume
        anything about where they came from, just that they don't require
        grad). `incumbent_idx`: `[B]` long, the index into the `t` axis of
        the incumbent train point (`y_train.argmin(dim=1)`) -- NOT its
        value; the marked hidden state already carries that. -> logits
        `[B,Q,n_out]`."""
        assert len(layer_hidden_states) == self.n_layers, (
            f"expected {self.n_layers} layer hidden states, got {len(layer_hidden_states)}"
        )
        B = x_query.shape[0]
        batch_idx = torch.arange(B, device=x_query.device)

        h = self.query_embed(x_query)
        for l in range(self.n_layers):
            kv = layer_hidden_states[l].clone()  # don't mutate the caller's tensor
            kv[batch_idx, incumbent_idx] = kv[batch_idx, incumbent_idx] + self.incumbent_marker[l]
            h = self.cross_blocks[l](h, kv)
        return self.head(self.out_ln(h))


if __name__ == "__main__":
    torch.manual_seed(0)
    B, t, Q, max_x_dim = 3, 7, 5, 2
    d_model_pfn, d_expert, n_heads, n_layers, d_ff, n_out = 16, 8, 2, 4, 32, 1

    readout = LayerLockedReadout(d_model_pfn, d_expert, n_heads, n_layers, d_ff, max_x_dim, n_out)
    layer_hidden = [torch.randn(B, t, d_model_pfn) for _ in range(n_layers)]
    x_query = torch.rand(B, Q, max_x_dim)
    incumbent_idx = torch.randint(0, t, (B,))

    out = readout(x_query, layer_hidden, incumbent_idx)
    print("output shape:", out.shape)  # expect [3, 5, 1]

    # Query independence: perturbing query k's x must not change any OTHER
    # query's output -- no candidate-candidate self-attention, mirroring
    # models/pfn.py's own test-test independence invariant.
    x_query_pert = x_query.clone()
    x_query_pert[:, 0, :] += 1.0
    out_pert = readout(x_query_pert, layer_hidden, incumbent_idx)
    other_diff = (out[:, 1:] - out_pert[:, 1:]).abs().max().item()
    print("max diff at OTHER queries after perturbing query 0 (~0 expected):", other_diff)

    # Incumbent anchor is actually used: moving WHICH index is marked as
    # the incumbent (hidden states and x_query held fixed) must change the
    # output.
    incumbent_idx_alt = (incumbent_idx + 1) % t
    out_alt_incumbent = readout(x_query, layer_hidden, incumbent_idx_alt)
    print("max diff from moving the incumbent marker alone (>0 expected):",
          (out - out_alt_incumbent).abs().max().item())

    # The caller's hidden-state tensors must not be mutated in place.
    layer_hidden_before = [h.clone() for h in layer_hidden]
    readout(x_query, layer_hidden, incumbent_idx)
    unchanged = all(torch.equal(a, b) for a, b in zip(layer_hidden, layer_hidden_before))
    print("caller's layer_hidden_states left untouched:", unchanged)
