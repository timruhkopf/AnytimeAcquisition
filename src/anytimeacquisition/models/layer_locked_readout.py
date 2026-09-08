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

**Incumbent anchor token.** Many acquisition criteria (EI, PI, UCB) are
functionals evaluated AT the incumbent `y*_t` specifically, not just "some
function of the context broadly." Rather than making the expert rediscover
which of the `t` train tokens is the incumbent from context alone, a small
per-layer learned embedding of `y*_t` is concatenated as an EXTRA key/value
pair into that layer's cross-attention KV set. This never touches the
frozen PFN's own forward pass (fully additive, expert-side only) and
mirrors §3.3's own incumbent-relative philosophy (resampling the candidate
descriptor's bin grid so `y*_t` sits at a fixed index) on the input/readout
side instead of the output/descriptor side.
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
        # One small per-layer embedding, not one shared across layers: the
        # frozen hidden states at different depths live in genuinely
        # different representational spaces (each layer's own attention +
        # FFN reshapes them), so the incumbent anchor needs a matching
        # per-layer re-embedding to stay comparable to that layer's kv, not
        # a single embedding reused verbatim everywhere.
        self.incumbent_embed = nn.ModuleList([nn.Linear(1, d_model_pfn) for _ in range(n_layers)])
        self.cross_blocks = nn.ModuleList(
            [CrossAttnBlock(d_expert, d_model_pfn, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.out_ln = nn.LayerNorm(d_expert)
        self.head = nn.Linear(d_expert, n_out)

    def forward(
        self, x_query: torch.Tensor, layer_hidden_states: list[torch.Tensor], y_incumbent: torch.Tensor,
    ) -> torch.Tensor:
        """x_query: `[B,Q,max_x_dim]`. `layer_hidden_states`: `n_layers`
        tensors, each `[B,t,d_model_pfn]` -- the FROZEN PFN's train-token
        hidden state at that layer (caller's responsibility to run the
        frozen forward under `no_grad`; this module doesn't assume
        anything about where they came from, just that they don't require
        grad). `y_incumbent`: `[B]`. -> logits `[B,Q,n_out]`."""
        assert len(layer_hidden_states) == self.n_layers, (
            f"expected {self.n_layers} layer hidden states, got {len(layer_hidden_states)}"
        )
        h = self.query_embed(x_query)
        for l in range(self.n_layers):
            inc_tok = self.incumbent_embed[l](y_incumbent.view(-1, 1, 1))  # [B,1,d_model_pfn]
            kv = torch.cat([layer_hidden_states[l], inc_tok], dim=1)  # [B,t+1,d_model_pfn]
            h = self.cross_blocks[l](h, kv)
        return self.head(self.out_ln(h))


if __name__ == "__main__":
    torch.manual_seed(0)
    B, t, Q, max_x_dim = 3, 7, 5, 2
    d_model_pfn, d_expert, n_heads, n_layers, d_ff, n_out = 16, 8, 2, 4, 32, 1

    readout = LayerLockedReadout(d_model_pfn, d_expert, n_heads, n_layers, d_ff, max_x_dim, n_out)
    layer_hidden = [torch.randn(B, t, d_model_pfn) for _ in range(n_layers)]
    x_query = torch.rand(B, Q, max_x_dim)
    y_incumbent = torch.rand(B)

    out = readout(x_query, layer_hidden, y_incumbent)
    print("output shape:", out.shape)  # expect [3, 5, 1]

    # Query independence: perturbing query k's x must not change any OTHER
    # query's output -- no candidate-candidate self-attention, mirroring
    # models/pfn.py's own test-test independence invariant.
    x_query_pert = x_query.clone()
    x_query_pert[:, 0, :] += 1.0
    out_pert = readout(x_query_pert, layer_hidden, y_incumbent)
    other_diff = (out[:, 1:] - out_pert[:, 1:]).abs().max().item()
    print("max diff at OTHER queries after perturbing query 0 (~0 expected):", other_diff)

    # Incumbent anchor is actually used: varying y_incumbent alone (hidden
    # states and x_query held fixed) must change the output.
    y_incumbent_alt = y_incumbent + 0.3
    out_alt_incumbent = readout(x_query, layer_hidden, y_incumbent_alt)
    print("max diff from varying y_incumbent alone (>0 expected):",
          (out - out_alt_incumbent).abs().max().item())
