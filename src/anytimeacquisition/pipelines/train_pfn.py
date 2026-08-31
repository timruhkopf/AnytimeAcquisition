"""PFN-only training pipeline (M2) — trains the PFN (models/pfn.py) against
BNNPrior (M1) instances via bar-distribution NLL, producing a checkpoint
that then never gets fine-tuned again. A fresh synthetic task (BNNPrior
`reset()`) is drawn every step, with a randomized train/test split size
each step too — matches PFNs4BO's own training loop (variable
`single_eval_pos`, AdamW + cosine-warmup schedule; see
`docs/log/2026-08-27-pfns4bo-bnn-prior-comparison.md`).

The actual loop lives in `trainer/pfn_trainer.py`'s `PFNTrainer`, a
`_target_`-instantiable class — same shape as `trainer/dummy.py`'s
`DummyTrainer`. This module has two entry points:
- `train_pfn(...)` — plain function, builds its own `BNNPrior`/`PFN`/
  `PFNTrainer` from scalar kwargs. What `tests/test_train_pfn.py` and the
  notebooks call; no Hydra/MLflow involved.
- `main(cfg)` — the Hydra pipeline (`configs/train_pfn.yaml`): instantiates
  the prior/model/trainer via `hydra.utils.instantiate` and wraps it with
  the provenance + MLflow logging convention every other pipeline in this
  repo uses (see `pipelines/train.py`). Named, reproducible training
  configs live under `configs/experiment/` (Hydra's standard
  `# @package _global_` pattern) — run one via `experiment=<name>`, e.g.
  `experiment=pfn_smoke_xdim2`.
"""
import logging
import os
from pathlib import Path

# MLflow 3.x deprecated the plain filesystem tracking store in favor of DB
# backends. We deliberately stay on the file store — see pipelines/train.py.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import hydra
import mlflow
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from anytimeacquisition.deployment.provenance import record_provenance
from anytimeacquisition.models.bar_distribution import BarDistribution
from anytimeacquisition.models.pfn import PFN
from anytimeacquisition.priors.bnn import BNNPrior
from anytimeacquisition.trainer.pfn_trainer import PFNTrainer
from anytimeacquisition.utils.flatten import flatten

log = logging.getLogger(__name__)


def train_pfn(
    x_dim: int = 2,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 3,
    d_ff: int = 128,
    n_bins: int = 64,
    batch_size: int = 32,
    min_train: int = 3,
    max_train: int = 20,
    n_test: int = 10,
    n_steps: int = 500,
    lr: float = 1e-3,
    warmup_steps: int = 50,
    device: str = "cpu",
    seed: int = 0,
    prior_kwargs: dict | None = None,
    checkpoint_path: str | Path | None = None,
    log_every: int = 50,
    mixed_precision: bool = False,
) -> dict:
    """Plain-function entry point — builds its own BNNPrior/PFN/PFNTrainer
    from scalar kwargs. No Hydra/MLflow. See `main()` for the Hydra
    pipeline, which instantiates the same pieces from config instead.
    `mixed_precision`: CUDA-only, no-ops on `device="cpu"` — see
    `trainer/pfn_trainer.py`'s docstring, untested on real GPU hardware."""
    torch.manual_seed(seed)
    prior_kwargs = prior_kwargs or {}
    prior = BNNPrior(batch_size=batch_size, x_dim=x_dim, device=device, seed=seed, **prior_kwargs)
    model = PFN(max_x_dim=x_dim, d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff, n_bins=n_bins).to(device)
    model_config = dict(max_x_dim=x_dim, d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff, n_bins=n_bins)

    trainer = PFNTrainer(
        prior=prior, model=model, seed=seed, n_steps=n_steps, min_train=min_train,
        max_train=max_train, n_test=n_test, lr=lr, warmup_steps=warmup_steps, log_every=log_every,
        checkpoint_path=checkpoint_path, model_config=model_config, mixed_precision=mixed_precision,
    )
    return trainer.run()


def load_pfn_checkpoint(checkpoint_path: str | Path, device: str = "cpu") -> tuple[PFN, BarDistribution, dict]:
    """`bar_dist` in the returned tuple is `model.bar_dist` -- the PFN owns
    its own BarDistribution as a submodule (see models/pfn.py), so it loads
    from `model_state` along with everything else. Checkpoints saved before
    that (missing `bar_dist.*` keys) still load: those buffers are a
    deterministic function of `n_bins` alone under today's uniform-border
    design (docs/OPEN_QUESTIONS.md #8), which `PFN.__init__` above already
    reconstructs correctly from `ckpt["config"]` before `load_state_dict`
    runs -- so a *missing* `bar_dist.*` key is expected/harmless for an old
    checkpoint, but any other missing or unexpected key is a real mismatch
    and still raises."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = PFN(**ckpt["config"]).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    ignorable_missing = {"bar_dist.borders", "bar_dist.bucket_widths"}
    bad_missing = set(missing) - ignorable_missing
    if bad_missing or unexpected:
        raise RuntimeError(
            f"checkpoint at {checkpoint_path} doesn't match PFN(**config)'s state_dict — "
            f"missing={sorted(bad_missing)}, unexpected={sorted(unexpected)}"
        )
    model.eval()
    return model, model.bar_dist, ckpt


def plot_prior_data_and_pfn_1d(
    model: PFN,
    bar_dist: BarDistribution,
    prior_kwargs: dict | None = None,
    n_train: int = 10,
    grid_res: int = 200,
    seed: int = 7,
    out_path: str | Path | None = None,
):
    """1D diagnostic: samples a fresh BNNPrior instance, plots its true
    function + the sampled train data, and overlays the PFN's predictive
    density (softmax(logits)/bucket_width) as a heatmap over the same
    (x, y) grid. x_dim fixed to 1 -- the heatmap needs y as its own axis.
    `model`/`bar_dist` must have been built for x_dim=1. Returns the Figure;
    saves to `out_path` if given."""
    import matplotlib.pyplot as plt

    torch.manual_seed(seed)
    prior_kwargs = prior_kwargs or {}
    prior = BNNPrior(batch_size=1, x_dim=1, seed=seed, **prior_kwargs)

    x_tr, y_tr, _, _ = prior.sample_episode(n_train=n_train, n_test=0)

    x_grid = torch.linspace(0, 1, grid_res).view(1, -1, 1)
    with torch.no_grad():
        y_true = prior.evaluate(x_grid, noise=False)[0]
        logits = model(x_tr, y_tr, x_grid)[0]  # [grid_res, n_bins]
        density = torch.softmax(logits, -1) / bar_dist.bucket_widths  # [grid_res, n_bins]

    ink, grid_color = "#1a1a1a", "#d9d9d9"
    fig, ax = plt.subplots(figsize=(8, 5))

    im = ax.imshow(
        density.T.numpy(), origin="lower", extent=(0, 1, 0, 1), aspect="auto",
        cmap="viridis", interpolation="nearest",
    )
    ax.plot(x_grid[0, :, 0].numpy(), y_true.numpy(), color="white", linewidth=2.5, label="true function", zorder=2)
    ax.plot(x_grid[0, :, 0].numpy(), y_true.numpy(), color="#D55E00", linewidth=1.2, zorder=3)
    ax.scatter(
        x_tr[0, :, 0].numpy(), y_tr[0].numpy(), s=55, color="white", edgecolor=ink,
        linewidth=1.2, zorder=4, label=f"train data (n={n_train})",
    )

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("PFN predictive density", color=ink)
    cbar.ax.tick_params(colors=ink)

    ax.set_xlabel("x", color=ink)
    ax.set_ylabel("y", color=ink)
    ax.set_title("BNNPrior true function + data, overlaid with the PFN's predictive density", color=ink, fontsize=11)
    ax.tick_params(colors=ink)
    ax.legend(loc="upper right", fontsize=9, frameon=True, facecolor="white", framealpha=0.85)
    fig.tight_layout()

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
        print("saved", out_path)

    return fig


def plot_prior_data_and_pfn_2d_heatmaps(
    model: PFN,
    bar_dist: BarDistribution,
    prior_kwargs: dict | None = None,
    n_train: int = 15,
    grid_res: int = 60,
    seed: int = 7,
    out_path: str | Path | None = None,
):
    """2D diagnostic (x_dim=2): compares the BNNPrior's true function against
    the PFN's posterior over the same [0,1]^2 grid, conditioned on a fresh
    sampled train context -- a heatmap-panel gut check ("is the posterior
    even in the right shape/place"), not the `plot_prior_data_and_pfn_1d`
    density heatmap, which doesn't generalize past x_dim=1 since y would
    need its own axis on top of x1/x2. Four panels: true function, posterior
    mean, |mean - true| (calibration), posterior std (confidence -- should
    shrink near train points, stay wide in gaps). `model`/`bar_dist` must
    have been built for x_dim=2. Returns the Figure; saves to `out_path` if
    given."""
    import matplotlib.pyplot as plt

    torch.manual_seed(seed)
    prior_kwargs = prior_kwargs or {}
    prior = BNNPrior(batch_size=1, x_dim=2, seed=seed, **prior_kwargs)

    x_tr, y_tr, _, _ = prior.sample_episode(n_train=n_train, n_test=0)

    lin = torch.linspace(0, 1, grid_res)
    gx, gy = torch.meshgrid(lin, lin, indexing="xy")
    x_grid = torch.stack([gx.flatten(), gy.flatten()], dim=-1).unsqueeze(0)  # [1, grid_res^2, 2]

    with torch.no_grad():
        y_true = prior.evaluate(x_grid, noise=False)[0].view(grid_res, grid_res)
        logits = model(x_tr, y_tr, x_grid)[0]  # [grid_res^2, n_bins]
        mean = bar_dist.mean(logits).view(grid_res, grid_res)
        std = bar_dist.variance(logits).clamp(min=0).sqrt().view(grid_res, grid_res)
    error = (mean - y_true).abs()

    ink = "#1a1a1a"
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 9), constrained_layout=True)

    def panel(ax, data, title, cmap, vmin=None, vmax=None):
        im = ax.imshow(
            data.numpy(), origin="lower", extent=(0, 1, 0, 1),
            cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal",
        )
        ax.scatter(
            x_tr[0, :, 0].numpy(), x_tr[0, :, 1].numpy(),
            s=36, c="white", edgecolor=ink, linewidth=1.0, zorder=3,
        )
        ax.set_title(title, fontsize=11, color=ink)
        ax.tick_params(colors=ink)
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)

    panel(axes[0, 0], y_true, "true function", "viridis", 0, 1)
    panel(axes[0, 1], mean, "PFN posterior mean", "viridis", 0, 1)
    panel(axes[1, 0], error, "|posterior mean - true|", "magma")
    panel(axes[1, 1], std, "PFN posterior std (confidence)", "magma")

    axes[1, 0].set_xlabel("x1", color=ink)
    axes[1, 1].set_xlabel("x1", color=ink)
    axes[0, 0].set_ylabel("x2", color=ink)
    axes[1, 0].set_ylabel("x2", color=ink)

    fig.suptitle(
        f"BNNPrior (x_dim=2) vs. PFN posterior, conditioned on {n_train} train points",
        fontsize=13, color=ink,
    )

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
        print("saved", out_path)

    return fig


def plot_pfn_posterior_2d_surface(
    model: PFN,
    bar_dist: BarDistribution,
    prior_kwargs: dict | None = None,
    n_train: int = 15,
    grid_res: int = 40,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    seed: int = 7,
    out_path: str | Path | None = None,
):
    """2D diagnostic (x_dim=2), alternate view: the PFN's posterior as a 3D
    surface over [0,1]^2 -- the middle quantile (default median) as an
    opaque colored surface, the outer quantiles as translucent surfaces
    above/below it (a "confidence tube"), train data scattered at their
    observed (x1, x2, y). Complements `plot_prior_data_and_pfn_2d_heatmaps`'s
    flatter panel view, which is the more reliable "is this reasonable"
    check at a glance -- a static PNG here is a single fixed camera angle
    (rotatable only in a live matplotlib backend), so treat this as the
    illustrative/presentation view, not the primary diagnostic. `quantiles`
    must be sorted with an odd length (the middle one is the opaque
    surface). `model`/`bar_dist` must have been built for x_dim=2. Returns
    the Figure; saves to `out_path` if given."""
    import matplotlib.pyplot as plt

    assert len(quantiles) % 2 == 1 and list(quantiles) == sorted(quantiles), (
        "quantiles must be sorted with an odd length (middle one is the opaque surface)"
    )

    torch.manual_seed(seed)
    prior_kwargs = prior_kwargs or {}
    prior = BNNPrior(batch_size=1, x_dim=2, seed=seed, **prior_kwargs)

    x_tr, y_tr, _, _ = prior.sample_episode(n_train=n_train, n_test=0)

    lin = torch.linspace(0, 1, grid_res)
    gx, gy = torch.meshgrid(lin, lin, indexing="xy")
    x_grid = torch.stack([gx.flatten(), gy.flatten()], dim=-1).unsqueeze(0)

    with torch.no_grad():
        logits = model(x_tr, y_tr, x_grid)[0]
        surfaces = [bar_dist.icdf(logits, q).view(grid_res, grid_res) for q in quantiles]

    ink = "#1a1a1a"
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    mid = len(quantiles) // 2
    ax.plot_surface(
        gx.numpy(), gy.numpy(), surfaces[mid].numpy(),
        cmap="viridis", vmin=0, vmax=1, alpha=0.95, edgecolor="none",
    )
    for i, surf in enumerate(surfaces):
        if i == mid:
            continue
        ax.plot_surface(gx.numpy(), gy.numpy(), surf.numpy(), color="#56B4E9", alpha=0.18, edgecolor="none")

    ax.scatter(
        x_tr[0, :, 0].numpy(), x_tr[0, :, 1].numpy(), y_tr[0].numpy(),
        s=45, color="white", edgecolor=ink, linewidth=1.0, depthshade=False,
    )

    ax.set_xlabel("x1", color=ink)
    ax.set_ylabel("x2", color=ink)
    ax.set_zlabel("y", color=ink)
    ax.set_zlim(0, 1)
    ax.set_title(
        f"PFN posterior surface (x_dim=2): q={quantiles} band + train data (n={n_train})",
        fontsize=11, color=ink,
    )

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
        print("saved", out_path)

    return fig


@hydra.main(config_path="../../../configs", config_name="train_pfn", version_base=None)
def main(cfg: DictConfig) -> dict:
    """Hydra entry point. Select a named, reproducible config via
    `experiment=<name>` (see configs/experiment/), e.g.:
      uv run python -m anytimeacquisition.pipelines.train_pfn experiment=pfn_smoke_xdim2
    """
    overrides = HydraConfig.get().overrides.task
    provenance = record_provenance(list(overrides), allow_dirty=cfg.get("allow_dirty", False))

    mlflow.set_tracking_uri(cfg.callbacks.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.callbacks.mlflow.experiment_name)

    with mlflow.start_run():
        mlflow.set_tags(provenance.as_mlflow_tags())
        mlflow.log_params(flatten(OmegaConf.to_container(cfg, resolve=True)))

        prior = instantiate(cfg.priors, seed=cfg.seed, device=cfg.device)
        model = instantiate(cfg.models.surrogates).to(cfg.device)
        model_config = OmegaConf.to_container(cfg.models.surrogates, resolve=True)
        model_config.pop("_target_")

        trainer = instantiate(
            cfg.trainer, prior=prior, model=model, seed=cfg.seed, model_config=model_config,
            on_log=lambda step, metrics: mlflow.log_metrics(metrics, step=step),
        )
        result = trainer.run()

        log.info("run complete, final metrics: %s", {k: v[-1] for k, v in result["history"].items() if k != "step"})
        return result["history"]


if __name__ == "__main__":
    main()
