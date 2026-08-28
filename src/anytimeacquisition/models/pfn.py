"""The PFN itself — ported near-verbatim from the tested prototype at
`archive/src/exit/claude/pfn-explore-exploit-repo/repo/src/model/pfn.py`.

Attention pattern (the whole point of a PFN, not an implementation detail —
see `archive/src/exit/PFN_ActionHead_ExpertIteration_Design.md`):
  * train tokens self-attend to train tokens only (bidirectional) -- this
    makes the train-token representations a permutation-invariant summary of
    the observed context, which is what a Bayesian posterior should be.
  * test tokens cross-attend to train tokens only -- never to each other,
    never to themselves. This prevents test-test leakage and makes every
    test point's prediction conditionally independent given the context,
    required for the PPD to be a valid predictive distribution point-by-point.

Checked against PFNs4BO's own `TransformerModel.generate_D_q_matrix` (their
reference implementation, via a single mask fed into one homogeneous
transformer stack rather than separate self-/cross-attention modules): same
train-train / test-train pattern, except PFNs4BO's mask also lets each test
token attend to itself (an `eye` OR'd in, likely just to avoid an
all-masked row edge case in their more general code path). We deliberately
don't include that self-attention term — matches the design doc's "never to
themselves" more literally, and n_train >= 1 always holds here so there's no
degenerate all-masked-row risk to guard against.

Implemented with one custom masked multi-head attention rather than
`nn.TransformerEncoderLayer`, because the encoder layer's API doesn't cleanly
express "two token types with an asymmetric, non-square attention pattern"
without fighting it. No positional encoding anywhere: train tokens must stay
permutation-invariant, and test tokens don't attend to each other so there's
nothing for a position to disambiguate. No TabPFN feature-wise attention —
see `docs/log/2026-08-27-pfns4bo-bnn-prior-comparison.md`.
"""
import torch
import torch.nn as nn


class MaskedMHA(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dk = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """x: [B,T,d_model]; attn_mask: [T,T] bool, True = allowed to attend."""
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.h, self.dk).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.h, self.dk).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.h, self.dk).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / (self.dk**0.5)
        neg_inf = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~attn_mask.view(1, 1, T, T), neg_inf)
        attn = scores.softmax(dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)  # guard an all-masked row, shouldn't occur here

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)


class PFNBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MaskedMHA(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.ff(self.ln2(x))
        return x


def build_pfn_attn_mask(n_train: int, n_test: int, device) -> torch.Tensor:
    """[T,T] bool, True = allowed. T = n_train + n_test, layout
    [0:n_train)=train, [n_train:T)=test.
      train->train: allowed (bidirectional self-attn)
      train->test:  NOT allowed (train must never see test / leak)
      test->train:  allowed (the cross-attention that produces the PPD)
      test->test:   NOT allowed, including self (no leakage, see module docstring)
    """
    T = n_train + n_test
    mask = torch.zeros(T, T, dtype=torch.bool, device=device)
    mask[:n_train, :n_train] = True
    mask[n_train:, :n_train] = True
    return mask


class PFN(nn.Module):
    def __init__(
        self, x_dim: int, d_model: int = 128, n_heads: int = 4, n_layers: int = 4,
        d_ff: int = 256, n_bins: int = 64, dropout: float = 0.0,
    ):
        super().__init__()
        self.x_dim = x_dim
        # Train tokens see (x, y); test tokens see (x,) with a learned
        # "no-y-yet" placeholder embedding instead of a real y.
        self.train_embed = nn.Linear(x_dim + 1, d_model)
        self.test_x_embed = nn.Linear(x_dim, d_model)
        self.test_placeholder = nn.Parameter(torch.zeros(d_model))

        self.blocks = nn.ModuleList([PFNBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.out_ln = nn.LayerNorm(d_model)
        self.out_head = nn.Linear(d_model, n_bins)

    def forward(
        self, x_train: torch.Tensor, y_train: torch.Tensor, x_test: torch.Tensor,
        return_hidden: bool = False,
    ):
        """x_train: [B,Ntr,x_dim]  y_train: [B,Ntr]  x_test: [B,Nte,x_dim]
        -> logits: [B,Nte,n_bins] (bar-distribution logits per test point).
        If return_hidden=True, also returns the list of per-layer hidden
        states for the FULL sequence (train++test) — the "KV cache" surface
        the ActionHead (M4) taps into."""
        B, Ntr, _ = x_train.shape
        Nte = x_test.shape[1]

        train_tok = self.train_embed(torch.cat([x_train, y_train.unsqueeze(-1)], dim=-1))
        test_tok = self.test_x_embed(x_test) + self.test_placeholder.view(1, 1, -1)
        x = torch.cat([train_tok, test_tok], dim=1)

        mask = build_pfn_attn_mask(Ntr, Nte, x.device)

        hidden_states = []
        for block in self.blocks:
            x = block(x, mask)
            if return_hidden:
                hidden_states.append(x)

        x = self.out_ln(x)
        logits = self.out_head(x[:, Ntr:, :])

        if return_hidden:
            return logits, hidden_states
        return logits


if __name__ == "__main__":
    torch.manual_seed(0)
    B, Ntr, Nte, d = 2, 5, 3, 2
    model = PFN(x_dim=d, d_model=32, n_heads=4, n_layers=2, d_ff=64, n_bins=16)
    x_train = torch.rand(B, Ntr, d)
    y_train = torch.rand(B, Ntr)
    x_test = torch.rand(B, Nte, d)
    logits = model(x_train, y_train, x_test)
    print("logits shape:", logits.shape)  # expect [2, 3, 16]

    perm = torch.randperm(Ntr)
    logits_perm = model(x_train[:, perm], y_train[:, perm], x_test)
    print("max abs diff under train-set permutation (~0 expected):",
          (logits - logits_perm).abs().max().item())

    # No test-test leakage: perturbing one test point's x must not change
    # any OTHER test point's logits.
    x_test_perturbed = x_test.clone()
    x_test_perturbed[:, 0, :] += 1.0
    logits_pert = model(x_train, y_train, x_test_perturbed)
    other_diff = (logits[:, 1:, :] - logits_pert[:, 1:, :]).abs().max().item()
    print("max diff at OTHER test points after perturbing test point 0 (~0 expected):", other_diff)
