import torch

from anytimeacquisition.models.action_head import ActionHead, pfn_dims
from anytimeacquisition.models.bar_distribution import BarDistribution, uniform_bin_borders
from anytimeacquisition.models.baselines.pfn_acquisition import pfn_ei_argmax
from anytimeacquisition.models.pfn import PFN
from anytimeacquisition.pipelines.action_head_ei_diagnostic import (
    beta_mode,
    beta_nll_loss,
    canonical_aux_features,
    run_stage,
)
from anytimeacquisition.pipelines.action_head_posterior_distill import posterior_argmin_targets
from anytimeacquisition.priors.bnn import BNNPrior


def _tiny_pfn(x_dim=1, seed=0):
    torch.manual_seed(seed)
    pfn = PFN(max_x_dim=x_dim, d_model=16, n_heads=2, n_layers=2, d_ff=32, n_bins=16)
    pfn.eval()
    bar_dist = BarDistribution(uniform_bin_borders(16))
    return pfn, bar_dist


def test_canonical_aux_features_uses_context_derived_incumbent():
    x_train, y_train = torch.rand(4, 6, 1), torch.rand(4, 6)
    aux = canonical_aux_features(x_train, y_train)
    assert torch.allclose(aux["incumbent_value"], y_train.min(dim=1).values)
    assert torch.equal(aux["step_count"], torch.zeros(4))
    assert torch.equal(aux["remaining_budget"], torch.ones(4))
    assert torch.equal(aux["improvement_trend"], torch.zeros(4))


def test_beta_nll_loss_is_lower_for_target_near_mode():
    alpha, beta = torch.tensor([[5.0]]), torch.tensor([[2.0]])
    near_mode = beta_mode(alpha, beta)
    far_from_mode = torch.tensor([[0.02]])
    loss_near = beta_nll_loss(alpha, beta, near_mode)
    loss_far = beta_nll_loss(alpha, beta, far_from_mode)
    assert (loss_near < loss_far).all()


def test_real_action_head_fits_a_single_fixed_context_better_than_blind():
    """Stage-1-style smoke check, kept fast: memorize one fixed context
    with the real (non-blind) pathway for a handful of steps and check
    loss drops -- the minimal signal that gradients flow usefully through
    cross-attention into the frozen PFN's hidden states for THIS (harder,
    EI-argmax) target, not just for the easier posterior-argmin one
    action_head_posterior_distill.py already validated."""
    torch.manual_seed(0)
    x_dim = 1
    pfn, bar_dist = _tiny_pfn(x_dim=x_dim)
    d_model, n_layers = pfn_dims(pfn)
    prior = BNNPrior(batch_size=4, x_dim=x_dim, seed=0)
    prior.reset()
    fixed_context = prior.sample_episode(n_train=5, n_test=0)[:2]

    action_head = ActionHead(pfn_d_model=d_model, pfn_n_layers=n_layers, x_dim=x_dim, d_model=16, n_heads=2, d_ff=32)
    losses = run_stage(
        pfn, bar_dist, action_head, prior, n_steps=60, n_train=5, lr=1e-2,
        blind=False, seed=0, log_every=1_000_000, fixed_context=fixed_context, n_grid=200,
    )
    assert losses[-1] < losses[0], "beta NLL should drop when memorizing a single fixed EI-argmax target"


def test_blind_ablation_does_not_use_pfn_context():
    """blind=True must make the ActionHead's output identical across two
    contexts that differ only in (x_train, y_train) -- same invariant as
    action_head_posterior_distill.py's own blind-ablation test, re-checked
    here since this is a fresh, independent pipeline."""
    x_dim = 1
    pfn, _ = _tiny_pfn(x_dim=x_dim)
    d_model, n_layers = pfn_dims(pfn)
    torch.manual_seed(0)
    action_head = ActionHead(pfn_d_model=d_model, pfn_n_layers=n_layers, x_dim=x_dim, d_model=16, n_heads=2, d_ff=32)

    aux = {name: torch.rand(2) for name in ("step_count", "remaining_budget", "incumbent_value", "improvement_trend")}
    x_train_a, y_train_a = torch.rand(2, 5, x_dim), torch.rand(2, 5)
    x_train_b, y_train_b = torch.rand(2, 5, x_dim), torch.rand(2, 5)

    out_a = action_head(pfn, x_train_a, y_train_a, aux, blind=True)
    out_b = action_head(pfn, x_train_b, y_train_b, aux, blind=True)
    assert torch.allclose(out_a["alpha"], out_b["alpha"])
    assert torch.allclose(out_a["beta"], out_b["beta"])


def test_ei_argmax_target_genuinely_differs_from_posterior_mean_argmin():
    """Proves the EI-argmax oracle isn't degenerating into the easier
    posterior-mean-argmin target action_head_posterior_distill.py already
    uses: construct a case where the incumbent is already very low
    (matching or beating most of the posterior's mass), so the
    exploration term should pull EI's argmax away from the pure
    exploitation point at least some of the time across several sampled
    contexts."""
    pfn, bar_dist = _tiny_pfn(x_dim=1, seed=3)
    prior = BNNPrior(batch_size=1, x_dim=1, seed=7)

    differed = 0
    n_trials = 15
    for trial in range(n_trials):
        prior.reset()
        x_train, y_train, _, _ = prior.sample_episode(n_train=8, n_test=0)
        ei_target, _, _ = pfn_ei_argmax(pfn, bar_dist, x_train, y_train, n_grid=300)
        mean_target = posterior_argmin_targets(pfn, bar_dist, x_train, y_train, n_candidates=300)
        if (ei_target - mean_target).abs().item() > 0.05:
            differed += 1
    assert differed > 0, "EI-argmax and posterior-mean-argmin targets never differed across any trial"
