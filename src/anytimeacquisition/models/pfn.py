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

**No physical `[T,T]` mask is materialized.** An earlier version of this
module concatenated train++test into one `[B,T,D]` sequence and ran a single
masked attention over the full `T x T` score matrix, relying on the mask to
zero out the train->test and test->test blocks. Those blocks are pure waste
-- always masked to -inf, never contributing a gradient -- but still cost
`O(T^2)` compute and (worse, memory-wise) an `O(B*h*T^2)` score tensor, which
is what was blowing up context/activation memory as n_train/n_test grew (the
reported symptom this rewrite responds to). Instead, train and test tokens
are kept as two physically separate tensors all the way through the block
stack, and each block runs two attention calls that share the SAME
`MaskedMHA` weights: `attn(train, train)` (bidirectional self-attention) and
`attn(test, kv_input=train)` (cross-attention). Structural separation makes
train->test and test->test attention impossible to compute at all, rather
than computed-then-discarded -- matching ifBO's
`ifbo/layer.py::TransformerEncoderLayer.forward`'s `isinstance(src_mask,
int)` branch (`src_left = self_attn(train,train,train)`, `src_right =
self_attn(test, train, train)`, both through the same `self.self_attn`
module), which does the same split for the same reason. One thing ifBO's
split path does NOT support: per-batch-item padding (it asserts
`src_key_padding_mask is None` whenever it takes this branch, since ifBO
always uses a single scalar `single_eval_position` shared across the whole
batch). We add that back in via `train_key_padding_mask` (see `MaskedMHA`
and `PFNBlock` below) so a batch can mix episodes with different n_train by
padding the train section out to a common width instead of forcing one
n_train onto the whole batch.

Implemented with one custom masked multi-head attention rather than
`nn.TransformerEncoderLayer`, because the encoder layer's API doesn't cleanly
express "two token types with an asymmetric, non-square attention pattern"
without fighting it. No positional encoding anywhere: train tokens must stay
permutation-invariant, and test tokens don't attend to each other so there's
nothing for a position to disambiguate. No TabPFN feature-wise attention —
see `docs/log/2026-08-27-pfns4bo-bnn-prior-comparison.md`.

`self.bar_dist` (a `BarDistribution` submodule, built from `n_bins`) is
owned by the model itself, not constructed separately alongside it --
matches how other PFN implementations bundle the output distribution into
the model for deployment, and means `state_dict()`/checkpoint save-load
carries it automatically instead of every call site reconstructing a
(today, always uniform-border) `BarDistribution` by hand from `n_bins` and
hoping it stays in sync. `bar_dist.borders`/`bucket_widths` are buffers, not
parameters, so this doesn't add anything to `self.parameters()` or change
what the optimizer sees.

**Variable x_dim, still with no feature-wise attention.** `max_x_dim` is a
ceiling, not a fixed requirement: `train_embed`/`test_x_embed` are built
`max_x_dim`-wide, but any one episode may use fewer real dims
(`n_features[b] <= max_x_dim`, per batch item). This was a real design
choice, not a default: TabPFN's own variable-feature-count support
(`architectures/tabpfn_v2.py::AlongRowAttention`/`AlongColumnAttention`,
PriorLabs/TabPFN) is structural -- every (example, feature) pair is its own
token on a 2D grid, with a whole second attention axis between features at
every block, plus a persisted per-column embedding to break feature-order
symmetry -- because TabPFN is a zero-shot foundation model for *arbitrary*
tabular schemas, where column identity itself carries no fixed meaning
across datasets. That problem doesn't apply here: a BO run's `x_dim` is a
real, fixed hyperparameter space throughout that run, only varying *across*
episodes/tasks during training -- exactly PFNs4BO's and ifBO's own setting,
and neither of them use feature attention either. Both wrap their ordinary
dense per-example encoder in a `VariableNumFeaturesEncoder`
(`automl/PFNs4BO` and `automl/ifBO`, `encoders.py`, byte-for-byte identical
between the two): rescale by `num_features / x.shape[-1]` (a fan-in
correction -- a fixed-width Linear's fan-in shouldn't shrink just because
some of its input slots are zero-padded rather than real), then zero-pad up
to the fixed width their encoder was built for. `_pad_and_rescale_features`
below is that same encoder, generalized from their single batch-wide
`x.shape[-1]` to a per-batch-item `n_features` tensor, so one batch can mix
episodes with different real x_dim -- same idea as `train_key_padding_mask`
generalizing ifBO's own batch-wide `single_eval_position` to per-item
`n_train`. `priors/bnn.py`'s `BNNPrior` already draws per-instance variable
dimensionality this way (`active_dim`/`active_dim_mask`, its own
`variable_dim_min` option) — `PFNTrainer` passes that straight through as
`n_features`.
"""
import torch
import torch.nn as nn

from anytimeacquisition.models.bar_distribution import BarDistribution, uniform_bin_borders


def _pad_and_rescale_features(
    x: torch.Tensor, max_x_dim: int, n_features: torch.Tensor
) -> torch.Tensor:
    """x: [B,N,D] (D <= max_x_dim). n_features: [B] int/long, each batch
    item's real feature count. Real values must be packed into the first
    n_features[b] columns; anything at column index >= n_features[b] --
    including within D itself, not just the D..max_x_dim extension -- is
    zeroed here rather than trusted to already be zero, so this is correct
    regardless of caller discipline (BNNPrior's `active_dim_mask` already
    zeroes it, but this function doesn't rely on that). Rescales the real
    columns by max_x_dim/n_features[b] (see module docstring), then
    zero-extends D up to max_x_dim."""
    B, N, D = x.shape
    feature_idx = torch.arange(D, device=x.device).view(1, 1, D)
    real_mask = feature_idx < n_features.view(B, 1, 1)
    scale = (max_x_dim / n_features.to(x.dtype)).view(B, 1, 1)
    x = torch.where(real_mask, x * scale, torch.zeros_like(x))
    if D < max_x_dim:
        x = torch.cat([x, x.new_zeros(B, N, max_x_dim - D)], dim=-1)
    return x


class MaskedMHA(nn.Module):
    """Shared-weight multi-head attention: `q_input` and `kv_input` may be two
    different physical tensors (a plain, non-square query/key-value split
    rather than one square mask over a joint sequence), but there's only ever
    one set of q/k/v/out projections, so calling this module twice with
    different (q_input, kv_input) pairs -- as `PFNBlock` does below for
    train-train self-attention and test-train cross-attention -- reuses the
    exact same weights both times, matching ifBO's `layer.py` reuse of one
    `self.self_attn` module across its `src_left`/`src_right` calls.

    Two independent, composable masking mechanisms, either or both optional:
      * `attn_mask`: [Tq,Tk] bool, True = allowed, shared across the batch --
        for a full square mask over one physical sequence (e.g. ActionHead's
        own small self-attention block, ported forward unchanged).
      * `key_padding_mask`: [B,Tk] bool, True = real token / False = padding,
        one mask per batch item -- for a ragged kv sequence, e.g. PFN's train
        section when different batch items have different n_train padded out
        to a common width. Only masks the key/value side: a padded row on the
        query side is harmless (nobody reads its output; see `PFNBlock`)."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dk = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        q_input: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        kv_input: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """q_input: [B,Tq,d_model]. kv_input: [B,Tk,d_model], defaults to
        q_input (self-attention) when omitted."""
        if kv_input is None:
            kv_input = q_input
        B, Tq, D = q_input.shape
        Tk = kv_input.shape[1]
        q = self.q_proj(q_input).view(B, Tq, self.h, self.dk).transpose(1, 2)
        k = self.k_proj(kv_input).view(B, Tk, self.h, self.dk).transpose(1, 2)
        v = self.v_proj(kv_input).view(B, Tk, self.h, self.dk).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / (self.dk**0.5)
        if attn_mask is not None or key_padding_mask is not None:
            neg_inf = torch.finfo(scores.dtype).min
            if attn_mask is not None:
                scores = scores.masked_fill(~attn_mask.view(1, 1, Tq, Tk), neg_inf)
            if key_padding_mask is not None:
                scores = scores.masked_fill(~key_padding_mask.view(B, 1, 1, Tk), neg_inf)
        attn = scores.softmax(dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)  # guard an all-masked row, shouldn't occur here

        out = (attn @ v).transpose(1, 2).reshape(B, Tq, D)
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

    def forward(
        self,
        x_train: torch.Tensor,
        x_test: torch.Tensor,
        train_key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """x_train: [B,Ntr,D]; x_test: [B,Nte,D] -- kept as two separate
        tensors (never concatenated) so train-train and test-train attention
        are two physically distinct matmuls, per the module docstring.
        train_key_padding_mask: [B,Ntr] bool, True = real train token, passed
        straight through to both attention calls' key/value side (train is
        the kv side of both: itself for train-train, the only side for
        test-train)."""
        train_normed = self.ln1(x_train)
        test_normed = self.ln1(x_test)
        x_train = x_train + self.attn(train_normed, key_padding_mask=train_key_padding_mask)
        x_test = x_test + self.attn(
            test_normed, kv_input=train_normed, key_padding_mask=train_key_padding_mask
        )
        x_train = x_train + self.ff(self.ln2(x_train))
        x_test = x_test + self.ff(self.ln2(x_test))
        return x_train, x_test


class PFN(nn.Module):
    def __init__(
        self, max_x_dim: int, d_model: int = 128, n_heads: int = 4, n_layers: int = 4,
        d_ff: int = 256, n_bins: int = 64, dropout: float = 0.0,
    ):
        super().__init__()
        self.max_x_dim = max_x_dim
        # Train tokens see (x, y); test tokens see (x,) with a learned
        # "no-y-yet" placeholder embedding instead of a real y. Built
        # max_x_dim-wide, not necessarily every episode's real x_dim -- see
        # module docstring / forward()'s n_features.
        self.train_embed = nn.Linear(max_x_dim + 1, d_model)
        self.test_x_embed = nn.Linear(max_x_dim, d_model)
        self.test_placeholder = nn.Parameter(torch.zeros(d_model))

        self.blocks = nn.ModuleList([PFNBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.out_ln = nn.LayerNorm(d_model)
        self.out_head = nn.Linear(d_model, n_bins)
        self.bar_dist = BarDistribution(uniform_bin_borders(n_bins))

    def forward(
        self, x_train: torch.Tensor, y_train: torch.Tensor, x_test: torch.Tensor,
        n_features: torch.Tensor | None = None,
        train_key_padding_mask: torch.Tensor | None = None,
        return_hidden: bool = False,
    ):
        """x_train: [B,Ntr,D]  y_train: [B,Ntr]  x_test: [B,Nte,D], D <=
        max_x_dim. n_features: [B] int, each batch item's real feature count
        (see module docstring / `_pad_and_rescale_features`). Optional; None
        (the default) means every item in the batch has D real features --
        i.e. ifBO/PFNs4BO's own batch-wide convention, and today's only
        usage pattern before `BNNPrior.variable_dim_min` is turned on.
        train_key_padding_mask: [B,Ntr] bool, True = real train token / False
        = padding. Optional; None (the default) means every train token in
        the batch is real, i.e. today's only usage pattern. Lets a batch mix
        episodes with different n_train by padding the train section out to
        a common width instead of forcing one n_train onto the whole batch --
        see the module docstring for why this needed adding (ifBO's own
        split-attention path doesn't support it).
        -> logits: [B,Nte,n_bins] (bar-distribution logits per test point).
        If return_hidden=True, also returns the list of per-layer hidden
        states for the FULL sequence (train++test) — the "KV cache" surface
        the ActionHead (M4) taps into."""
        if n_features is None:
            n_features = x_train.new_full((x_train.shape[0],), x_train.shape[-1])
        x_train = _pad_and_rescale_features(x_train, self.max_x_dim, n_features)
        x_test = _pad_and_rescale_features(x_test, self.max_x_dim, n_features)

        train_tok = self.train_embed(torch.cat([x_train, y_train.unsqueeze(-1)], dim=-1))
        test_tok = self.test_x_embed(x_test) + self.test_placeholder.view(1, 1, -1)

        hidden_states = []
        for block in self.blocks:
            train_tok, test_tok = block(train_tok, test_tok, train_key_padding_mask)
            if return_hidden:
                hidden_states.append(torch.cat([train_tok, test_tok], dim=1))

        test_tok = self.out_ln(test_tok)
        logits = self.out_head(test_tok)

        if return_hidden:
            return logits, hidden_states
        return logits


if __name__ == "__main__":
    torch.manual_seed(0)
    B, Ntr, Nte, d = 2, 5, 3, 2
    model = PFN(max_x_dim=d, d_model=32, n_heads=4, n_layers=2, d_ff=64, n_bins=16)
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

    # Ragged train sets via train_key_padding_mask: batch item 1 only has
    # n_valid real train points, padded out to Ntr with the rest ignored.
    # Its logits must match a separate, unpadded forward using just its real
    # points -- i.e. the padding is truly invisible, not just "small".
    n_valid = Ntr - 2
    train_key_padding_mask = torch.ones(B, Ntr, dtype=torch.bool)
    train_key_padding_mask[1, n_valid:] = False
    logits_ragged = model(x_train, y_train, x_test, train_key_padding_mask=train_key_padding_mask)
    logits_item1_unpadded = model(x_train[1:2, :n_valid], y_train[1:2, :n_valid], x_test[1:2])
    pad_diff = (logits_ragged[1:2] - logits_item1_unpadded).abs().max().item()
    print("max diff, padded vs. unpadded forward for the ragged batch item (~0 expected):", pad_diff)

    # Variable x_dim via n_features: a model built at max_x_dim=4 accepting
    # episodes that actually only use 2 real dims, with item 1 using even
    # fewer (1) than item 0 (2) -- per-instance, like BNNPrior's own
    # active_dim/active_dim_mask (see module docstring).
    max_x_dim = 4
    var_model = PFN(max_x_dim=max_x_dim, d_model=32, n_heads=4, n_layers=2, d_ff=64, n_bins=16)
    x_dim_actual = 2
    x_tr_var = torch.rand(B, Ntr, x_dim_actual)
    x_te_var = torch.rand(B, Nte, x_dim_actual)
    n_features = torch.tensor([2, 1])
    x_tr_var[1, :, 1:] = 0.0  # item 1's inactive dim, zeroed like active_dim_mask would
    x_te_var[1, :, 1:] = 0.0
    logits_var = var_model(x_tr_var, y_train, x_te_var, n_features=n_features)
    print("variable-x_dim logits shape (max_x_dim=4, actual x_dim=2):", logits_var.shape)

    # Garbage in a "padding" column beyond n_features[b] must be invisible --
    # _pad_and_rescale_features zeroes it regardless of what the caller put
    # there, rather than trusting it's already zero.
    x_tr_var_garbage = x_tr_var.clone()
    x_tr_var_garbage[1, :, 1:] = 99.0  # item 1's inactive dim: garbage, not zero
    logits_var_garbage = var_model(x_tr_var_garbage, y_train, x_te_var, n_features=n_features)
    print("max diff from garbage in a masked-out feature column (~0 expected):",
          (logits_var[1] - logits_var_garbage[1]).abs().max().item())

    # bar_dist is owned by the model itself now (see module docstring) --
    # sanity-check it round-trips through state_dict like any other buffer.
    print("model.bar_dist.num_bars:", model.bar_dist.num_bars)
    print("'bar_dist.borders' in state_dict():", "bar_dist.borders" in model.state_dict())
    print("mean predicted y at test points:", model.bar_dist.mean(logits)[0].tolist())
