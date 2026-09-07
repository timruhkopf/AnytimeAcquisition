import torch

from anytimeacquisition.metrics.rollout import rollout_episode
from anytimeacquisition.models.pfn import PFN
from anytimeacquisition.models.surrogates.pfn_surrogate import PFNSurrogate, pfn_surrogate_ei_policy
from anytimeacquisition.priors.bnn import BNNPrior

# ei()/pi() themselves (BarDistribution methods) are tested against Monte
# Carlo in tests/test_bar_distribution.py; these tests cover the
# PFNSurrogate wrapper's own contract (shapes, calling through correctly).


def _tiny_pfn(x_dim=2, seed=0):
    torch.manual_seed(seed)
    pfn = PFN(max_x_dim=x_dim, d_model=16, n_heads=2, n_layers=2, d_ff=32, n_bins=16)
    pfn.eval()
    return pfn, pfn.bar_dist


def test_predict_shape_and_wrapper_matches_calling_pfn_directly():
    pfn, bar_dist = _tiny_pfn(x_dim=2)
    surrogate = PFNSurrogate(pfn, bar_dist, use_bf16=False)
    x_context, y_context = torch.rand(3, 4, 2), torch.rand(3, 4)
    candidates = torch.rand(3, 5, 2)

    logits = surrogate.predict(x_context, y_context, candidates)
    assert logits.shape == (3, 5, 16)

    with torch.no_grad():
        n_features = x_context.new_full((3,), 2)
        expected_logits = pfn(x_context, y_context, candidates, n_features=n_features)
    assert torch.allclose(logits, expected_logits, atol=1e-5)


def test_expected_improvement_matches_bar_dist_ei_directly():
    pfn, bar_dist = _tiny_pfn(x_dim=1)
    surrogate = PFNSurrogate(pfn, bar_dist, use_bf16=False)
    x_context, y_context = torch.rand(2, 3, 1), torch.rand(2, 3)
    candidates = torch.rand(2, 7, 1)

    ei, logits = surrogate.expected_improvement(x_context, y_context, candidates)
    ei_direct = bar_dist.ei(logits, y_context.min(dim=1).values.unsqueeze(-1))
    assert torch.allclose(ei, ei_direct)
    assert (ei >= 0).all()  # EI is non-negative by construction


def test_probability_of_improvement_matches_bar_dist_pi_directly():
    pfn, bar_dist = _tiny_pfn(x_dim=1)
    surrogate = PFNSurrogate(pfn, bar_dist, use_bf16=False)
    x_context, y_context = torch.rand(2, 3, 1), torch.rand(2, 3)
    candidates = torch.rand(2, 7, 1)

    pi, logits = surrogate.probability_of_improvement(x_context, y_context, candidates)
    pi_direct = bar_dist.pi(logits, y_context.min(dim=1).values.unsqueeze(-1))
    assert torch.allclose(pi, pi_direct)
    assert (pi >= 0).all() and (pi <= 1).all()


def test_pfn_surrogate_ei_policy_is_a_drop_in_rollout_policy_fn():
    from functools import partial

    pfn, bar_dist = _tiny_pfn(x_dim=2)
    surrogate = PFNSurrogate(pfn, bar_dist, use_bf16=False)
    prior = BNNPrior(batch_size=3, x_dim=2, seed=0)

    policy_fn = partial(pfn_surrogate_ei_policy, surrogate=surrogate, n_candidates=32, seed=1)
    rollout = rollout_episode(prior, n_init=3, n_steps=4, policy_fn=policy_fn)
    assert rollout["x_context"].shape == (3, 7, 2)
    assert (rollout["x_context"] >= 0).all() and (rollout["x_context"] <= 1).all()
