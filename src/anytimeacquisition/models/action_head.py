"""ActionHead — VLA-style cross-attention into a frozen PFN, informed by
pi0/pi0.5's action-expert design (Physical Intelligence). See Phase 4 of
`docs/ROADMAP.md` and `docs/milestones/M4.md`.

Design inspiration (cited, not reproduced verbatim -- our setting differs
from theirs in ways that matter, see below):

  Black et al., "pi_0: A Vision-Language-Action Flow Model for General Robot
  Control" (arXiv:2410.24164), Sec. III-IV. The action expert is "a second,
  smaller set of weights for the robotics-specific tokens"; images, text,
  and action tokens are processed by one transformer whose attention matrix
  A(x_{1:N}) "indicates if a token can attend to another token", and "each
  x_i can be processed not only by a different encoder, but also by
  different expert weights within the transformer" -- one shared attention
  operation per layer, computed jointly over concatenated token streams, but
  with expert-specific (not shared) Q/K/V/FFN weights per token type. That's
  the idea we borrow: a separate weight set per stream, tapped at *every*
  layer, not just the final one.

  Black et al., "pi_0.5: a Vision-Language-Action Model with Open-World
  Generalization" (Physical Intelligence, pi.website/blog/pi05) builds the
  same action-expert mechanism on top of pi_0, adding a post-training stage
  and flow matching for the continuous-action head.

Two deliberate departures from both papers -- already decided in
`docs/ROADMAP.md`, not oversights:

  1. **This PFN is genuinely frozen** (`requires_grad_(False)`, forward
     pass under `torch.no_grad()`), not fine-tuned jointly the way pi_0.5's
     VLM backbone is (initialized from a pretrained VLM but still updated
     throughout both pre- and post-training). Our setting is closer to
     frozen-backbone-probing VLA work than to pi_0.5's own joint-training
     recipe. Because the backbone is truly frozen, there's no need for the
     PFN's own internal computation to be re-run jointly with ours behind a
     shared block-mask -- it's already fixed by the time the ActionHead
     runs. So this module does genuine cross-attention (ActionHead tokens
     as queries; the frozen PFN's per-layer train-token hidden states as
     keys/values, via this module's *own* per-layer K/V projections) rather
     than one joint self-attention op over a concatenated sequence.
  2. **No flow matching.** pi_0/pi_0.5 need it because their action targets
     come from many different human demonstrators (genuinely multimodal).
     Our imitation targets come from a single deterministic
     privileged-search oracle per call (`docs/ROADMAP.md`, Goal point 2) --
     a distributional head (Beta per dimension) is enough; flow matching's
     complexity wouldn't earn its place here.

Multi-layer tapping follows both pi_0.5's own recipe and this project's own
prior art: `archive/src/exit/model/asdf.py`'s `VLAAcquisitionHead`, a
standalone toy demo of the same per-layer K/V-projection idea against a
`DemoPFN` stand-in ("ITS OWN K/V projection heads (a Linear(d_pfn -> d_act)
per layer)") -- reference only, not ported verbatim. `docs/milestones/M4.md`
called for porting `archive/.../repo/src/model/action_head.py` directly, but
that file did not survive the accidental `rm -rf` this repo was recovered
from (it was never read into the Claude Code session the recovery replayed,
so no copy of it exists anywhere in this repo's history -- see
`docs/log/`); this is a fresh implementation against the design doc
(`archive/src/exit/PFN_ActionHead_ExpertIteration_Design.md`) and the
surviving `asdf.py` fragment, not a literal port.

The policy/value heads below are a **provisional single-Beta-per-dimension
placeholder**, gated as such in `docs/milestones/M4.md`: the real
single-vs-mixture decision needs the multimodality diagnostic (design doc
Sec. 4.2/8, stage 1), which has not been run yet (explicit user call,
2026-08-28). Alpha/beta are clamped via `softplus(x) + alpha_beta_floor`, matching the
NaN-gradient-instability fix `archive/src/prototype/other_diff/README.md`
already found necessary for a Beta head. `alpha_beta_floor` defaults to
2.0, not 1.0 (2026-09-01, see
`docs/log/2026-09-01-explore-branch-beta-nll-uniform-collapse.md`'s
"Reopening the loss itself" section): at floor=1.0, `Beta(1,1)` is the
*exact* uniform distribution, whose density is exactly 1 everywhere on
`[0,1]` -- meaning `-log_prob(target) = 0` for literally any target, a
zero-cost escape hatch the explore-branch imitation trainer's Beta-NLL
loss (`trainer/action_head_imitation_trainer.py::_beta_nll_loss`) was
found collapsing into even when its training target was genuinely
context-dependent. At floor=2.0, `Beta(2,2)` is NOT uniform (density
varies with the target), so "give up and predict the floor" no longer
gives exactly-zero loss -- there's a real, non-negligible cost to
retreating there, not just a cheaper-than-informative one. Good enough to
smoke-test the cross-attention pathway; not the final head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from anytimeacquisition.models.pfn import MaskedMHA, PFN

AUX_FEATURE_NAMES = ("step_count", "remaining_budget", "incumbent_value", "improvement_trend")


def pfn_dims(pfn: PFN) -> tuple[int, int]:
    """(d_model, n_layers) introspected from a PFN instance -- PFN doesn't
    store these as public attributes, so this is the one place that reaches
    into its submodules instead of every caller doing so independently."""
    return pfn.train_embed.out_features, len(pfn.blocks)


class CrossAttention(nn.Module):
    """Unmasked cross-attention: queries from this module's own tokens,
    keys/values already projected into this module's d_model by the caller
    (the per-layer K/V projection heads live in `ActionHeadBlock`, not
    here -- this class only does the attention op itself, mirroring
    `models.pfn.MaskedMHA`'s split of concerns for the train/test case)."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dk = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """x: [B,Tq,d]; k,v: [B,Tk,d]."""
        B, Tq, D = x.shape
        Tk = k.shape[1]
        q = self.q_proj(x).view(B, Tq, self.h, self.dk).transpose(1, 2)
        k_ = k.view(B, Tk, self.h, self.dk).transpose(1, 2)
        v_ = v.view(B, Tk, self.h, self.dk).transpose(1, 2)
        attn = ((q @ k_.transpose(-2, -1)) / (self.dk**0.5)).softmax(dim=-1)
        out = (attn @ v_).transpose(1, 2).reshape(B, Tq, D)
        return self.out_proj(out)


class ActionHeadBlock(nn.Module):
    """One layer: self-attend among the ActionHead's own tokens, then
    cross-attend into that layer's frozen PFN train-token activations via
    this block's own K/V projections, then FFN. Pre-LN residual style,
    matching `models.pfn.PFNBlock`."""

    def __init__(self, d_model: int, pfn_d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = MaskedMHA(d_model, n_heads)
        self.k_proj = nn.Linear(pfn_d_model, d_model)
        self.v_proj = nn.Linear(pfn_d_model, d_model)
        self.ln_cross = nn.LayerNorm(d_model)
        self.cross_attn = CrossAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, h: torch.Tensor, pfn_train_hidden: torch.Tensor, self_mask: torch.Tensor) -> torch.Tensor:
        h = h + self.self_attn(self.ln1(h), self_mask)
        h2 = self.ln_cross(h)
        k, v = self.k_proj(pfn_train_hidden), self.v_proj(pfn_train_hidden)
        h = h + self.cross_attn(h2, k, v)
        h = h + self.ff(self.ln2(h))
        return h


class ActionHead(nn.Module):
    def __init__(
        self, pfn_d_model: int, pfn_n_layers: int, x_dim: int,
        d_model: int = 64, n_heads: int = 4, d_ff: int = 128, dropout: float = 0.0,
        alpha_beta_floor: float = 2.0,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.alpha_beta_floor = alpha_beta_floor
        self.action_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.aux_embed = nn.ModuleDict({name: nn.Linear(1, d_model) for name in AUX_FEATURE_NAMES})
        self.blocks = nn.ModuleList([
            ActionHeadBlock(d_model, pfn_d_model, n_heads, d_ff, dropout) for _ in range(pfn_n_layers)
        ])
        self.out_ln = nn.LayerNorm(d_model)
        self.policy_head = nn.Linear(d_model, 2 * x_dim)  # -> (alpha_raw, beta_raw) per dim
        self.value_head = nn.Linear(d_model, 1)

    def forward(
        self, pfn: PFN, x_train: torch.Tensor, y_train: torch.Tensor, aux_features: dict,
        blind: bool = False,
    ) -> dict:
        """x_train: [B,Ntr,x_dim]  y_train: [B,Ntr]
        aux_features: dict of the 4 AUX_FEATURE_NAMES -> [B] tensors.
        -> {"alpha": [B,x_dim], "beta": [B,x_dim], "value": [B]}, alpha/beta
        both >= `self.alpha_beta_floor` (see module docstring). Gradients never reach `pfn`'s
        parameters -- frozen + forward under `torch.no_grad()` below; see
        `tests/test_action_head.py::test_pfn_gradients_are_none_after_backward`.

        `blind`: the "blind ActionHead" ablation flagged as not-yet-done in
        `docs/milestones/M4.md` -- zeroes the PFN's per-layer train-token
        hidden states before cross-attention, aux tokens untouched. Cross-
        attending into all-zero K/V still runs (same op, same shapes; the
        K/V projections' biases give a nonzero-but-context-independent
        result), so this isolates "does the ActionHead's fit depend on the
        specific PFN context" without changing the architecture or
        parameter count between the real and blind runs -- see
        `pipelines/action_head_posterior_distill.py`, which is what this
        flag exists for.
        """
        B = x_train.shape[0]
        pfn.eval()
        for p in pfn.parameters():
            p.requires_grad_(False)
        x_test_empty = x_train.new_zeros(B, 0, x_train.shape[-1])
        with torch.no_grad():
            _, hidden_states = pfn(x_train, y_train, x_test_empty, return_hidden=True)
        if blind:
            hidden_states = [torch.zeros_like(hs) for hs in hidden_states]
        n_train = x_train.shape[1]

        # The cross-attention query is the FULL 5-token sequence below (the
        # action_query param + all 4 aux-feature tokens), not action_query
        # alone -- every ActionHeadBlock self-attends all 5 among themselves
        # first, then cross-attends all 5 (as queries) into this layer's PFN
        # train-token hidden states (as keys/values). Only action_query's
        # own final state (h[:, 0, :] below) is ever read out by
        # policy_head/value_head -- the 4 aux tokens' own final states are
        # discarded, they only matter through how they shaped action_query
        # via self-attention across blocks.
        tokens = [self.action_query.expand(B, -1, -1)]
        for name in AUX_FEATURE_NAMES:
            tokens.append(self.aux_embed[name](aux_features[name].view(B, 1, 1).float()))
        h = torch.cat(tokens, dim=1)  # [B, 1 + len(AUX_FEATURE_NAMES), d_model]
        self_mask = torch.ones(h.shape[1], h.shape[1], dtype=torch.bool, device=h.device)

        for block, layer_hidden in zip(self.blocks, hidden_states):
            train_hidden = layer_hidden[:, :n_train, :]
            h = block(h, train_hidden, self_mask)

        h = self.out_ln(h)
        action_repr = h[:, 0, :]  # the action-query token's own final state, read out here -- see note above
        alpha_beta = self.policy_head(action_repr).view(B, self.x_dim, 2)
        alpha = F.softplus(alpha_beta[..., 0]) + self.alpha_beta_floor
        beta = F.softplus(alpha_beta[..., 1]) + self.alpha_beta_floor
        value = self.value_head(action_repr).squeeze(-1)
        return {"alpha": alpha, "beta": beta, "value": value}


def beta_mode(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """alpha, beta >= 1 always holds for `ActionHead`'s own output
    (`softplus(x) + alpha_beta_floor`, `alpha_beta_floor >= 1` by
    construction/default), so the mode is always defined:
    (alpha-1)/(alpha+beta-2), with the alpha=beta=1 (uniform) edge case
    mapped to 0.5 -- unreachable in practice when `alpha_beta_floor > 1`
    (the default, 2.0), kept as a defensive fallback for `alpha_beta_floor=1.0`
    callers/older checkpoints. Shared here (not just
    `pipelines/action_head_posterior_distill.py`, the first caller) since
    `action_head_policy_fn` below needs it too."""
    denom = alpha + beta - 2.0
    return torch.where(denom > 1e-6, (alpha - 1.0) / denom.clamp_min(1e-6), torch.full_like(denom, 0.5))


def build_rollout_aux_features(rollout: dict, step: int, n_steps: int, trend_window: int = 3) -> dict:
    """Real aux features off actual rollout state -- for the imitation
    trainer (`trainer/action_head_imitation_trainer.py`), replacing
    `pipelines/action_head_posterior_distill.py`'s toy `canonical_aux_features`
    (which hardcodes 3 of 4 fields at canonical values) with the genuine
    per-step quantities a trained policy should condition on.

    `rollout`: as returned by `trainer.exit_rollout.rollout_episode`. Reads
    `rollout["pre_step_contexts"][step]` (context as it stood immediately
    before this step, matching what a policy/oracle actually conditions on
    at that state) and, for `improvement_trend`, also `[step - trend_window]`
    when available.

    `improvement_trend = max(0, log(incumbent_{step-trend_window}) -
    log(incumbent_step))` -- same log-improvement idiom as
    `search.explore.improvement_weights`, applied to the realized trajectory
    instead of a candidate set, for consistency. 0 when `step < trend_window`
    (no history yet).

    -> {name: [B] tensor} for each of AUX_FEATURE_NAMES.
    """
    _, y_ctx = rollout["pre_step_contexts"][step]
    incumbent = y_ctx.min(dim=1).values  # [B]
    B = incumbent.shape[0]

    if step < trend_window:
        improvement_trend = torch.zeros(B)
    else:
        _, y_prev = rollout["pre_step_contexts"][step - trend_window]
        incumbent_prev = y_prev.min(dim=1).values
        improvement_trend = (
            torch.log(incumbent_prev.clamp_min(1e-12)) - torch.log(incumbent.clamp_min(1e-12))
        ).clamp_min(0.0)

    return {
        "step_count": torch.full((B,), float(step)),
        "remaining_budget": torch.full((B,), (n_steps - step) / n_steps),
        "incumbent_value": incumbent,
        "improvement_trend": improvement_trend,
    }


def action_head_policy_fn(action_head: "ActionHead", pfn: PFN, n_steps: int, sample: bool = False):
    """Factory wrapping `ActionHead.forward` into the
    `policy_fn(x_context, y_context, x_dim) -> [B, x_dim]` signature
    `trainer.exit_rollout.rollout_episode` expects -- this glue doesn't
    exist anywhere else. Tracks a step counter via closure state,
    incremented once per call -- safe because `rollout_episode` calls
    `policy_fn` exactly once per step, in order (0..n_steps-1).

    Aux features are built directly from `(x_context, y_context)` at call
    time, not via `build_rollout_aux_features` above (that needs a
    finished rollout's full `pre_step_contexts` list, which doesn't exist
    yet while a rollout is still being generated) -- but reproduces the
    same quantities: `step_count` from the closure counter,
    `remaining_budget` from `(n_steps - step) / n_steps`,
    `incumbent_value` from `y_context` directly, and `improvement_trend`
    from this call's own running incumbent history (the only history
    available at call time).

    `sample=False` (default -- evaluation/rollout-for-metrics): the Beta
    distribution's mode (`beta_mode`). `sample=True`: an actual
    `Beta(alpha, beta).sample()` draw -- kept for future DAgger-round
    exploration, not exercised by the round-0-only trainer built here.
    """
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
        with torch.no_grad():
            out = action_head(pfn, x_context, y_context, aux, blind=False)
        state["step"] += 1
        if sample:
            return torch.distributions.Beta(out["alpha"], out["beta"]).sample()
        return beta_mode(out["alpha"], out["beta"])

    return policy_fn


if __name__ == "__main__":
    from anytimeacquisition.utils.paths import CHECKPOINT_DIR
    from anytimeacquisition.pipelines.train_pfn import load_pfn_checkpoint
    from anytimeacquisition.priors.bnn import BNNPrior

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
    torch.manual_seed(0)
    action_head = ActionHead(pfn_d_model=d_model, pfn_n_layers=n_layers, x_dim=x_dim)

    prior = BNNPrior(batch_size=4, x_dim=x_dim, seed=1)
    x_train, y_train, _, _ = prior.sample_episode(n_train=10, n_test=0)
    aux_features = {
        "step_count": torch.arange(4).float(),
        "remaining_budget": torch.full((4,), 40.0),
        "incumbent_value": y_train.min(dim=1).values,
        "improvement_trend": torch.zeros(4),
    }

    out = action_head(pfn, x_train, y_train, aux_features)
    print("alpha:", out["alpha"].shape, out["alpha"].detach())
    print("beta: ", out["beta"].shape, out["beta"].detach())
    print("value:", out["value"].shape, out["value"].detach())
    assert (out["alpha"] >= 1.0).all() and (out["beta"] >= 1.0).all(), "Beta params must stay >= 1 (stability clamp)"

    loss = out["alpha"].sum() + out["beta"].sum() + out["value"].sum()
    loss.backward()
    pfn_grads = [p.grad for p in pfn.parameters()]
    action_head_grads = [p.grad for p in action_head.parameters()]
    print("PFN params with a gradient after backward (expect 0):", sum(g is not None for g in pfn_grads))
    print("ActionHead params with a gradient after backward (expect all):",
          sum(g is not None for g in action_head_grads), "/", len(action_head_grads))
    assert all(g is None for g in pfn_grads), "PFN must receive zero gradient -- it's frozen"
    assert all(g is not None for g in action_head_grads), "every ActionHead param should get a gradient"
    print("smoke forward + backward pass OK: cross-attention pathway is connected, PFN gradient-isolated.")
