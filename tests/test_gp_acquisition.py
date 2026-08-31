from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from anytimeacquisition.models.baselines.gp_acquisition import ACQUISITIONS, fit_gp, gp_acquisition_policy
from anytimeacquisition.priors.bnn import BNNPrior

CONFIG_DIR = Path(__file__).parent.parent / "configs" / "models" / "baselines"


def _context(batch_size=2, x_dim=2, n_train=4, seed=0):
    torch.manual_seed(seed)
    prior = BNNPrior(batch_size=batch_size, x_dim=x_dim, seed=seed)
    prior.reset()
    return prior.sample_episode(n_train=n_train, n_test=0)[:2]


def test_fit_gp_shapes_and_maximize_sign_flip():
    x_context, y_context = _context(batch_size=1)
    gp = fit_gp(x_context[0], y_context[0])
    assert gp.train_inputs[0].shape == (4, 2)
    # fit_gp negates y internally -- the largest (maximize-convention)
    # training target must correspond to the SMALLEST original y.
    train_y = gp.outcome_transform.untransform(gp.train_targets.unsqueeze(-1))[0].squeeze(-1)
    best_idx = train_y.argmax()
    assert torch.allclose(-train_y[best_idx], y_context[0].min().double(), atol=1e-4)


def test_gp_acquisition_policy_shapes_for_every_acquisition():
    x_context, y_context = _context(batch_size=2, x_dim=2, n_train=4)
    for acquisition in ACQUISITIONS:
        x_next = gp_acquisition_policy(
            x_context, y_context, x_dim=2, acquisition=acquisition, num_restarts=3, raw_samples=32,
        )
        assert x_next.shape == (2, 2)
        assert (x_next >= 0.0).all() and (x_next <= 1.0).all()


def test_baseline_configs_compose_and_instantiate_as_a_policy_fn():
    x_context, y_context = _context(batch_size=2, x_dim=2, n_train=4)
    for name in ("ei", "pi", "es"):
        cfg = OmegaConf.load(CONFIG_DIR / f"{name}.yaml")
        policy_fn = instantiate(cfg, num_restarts=3, raw_samples=32)
        x_next = policy_fn(x_context, y_context, 2)
        assert x_next.shape == (2, 2)


def test_gp_acquisition_policy_rejects_unknown_acquisition():
    x_context, y_context = _context(batch_size=1, x_dim=1, n_train=3)
    try:
        gp_acquisition_policy(x_context, y_context, x_dim=1, acquisition="bogus")
        assert False, "expected a ValueError for an unrecognized acquisition"
    except ValueError:
        pass
