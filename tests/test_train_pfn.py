from anytimeacquisition.pipelines.train_pfn import (
    plot_pfn_posterior_2d_surface,
    plot_prior_data_and_pfn_2d_heatmaps,
    train_pfn,
)


def test_training_nll_decreases(tmp_path):
    result = train_pfn(
        x_dim=1,
        d_model=16, n_layers=1, n_heads=2, d_ff=32, n_bins=16,
        batch_size=8, min_train=3, max_train=6, n_test=4,
        n_steps=60, log_every=10,
        prior_kwargs=dict(ecdf_n_draws=5, ecdf_samples_per_draw=50, cache_dir=None),
        checkpoint_path=tmp_path / "smoke.pt",
    )
    nlls = result["history"]["nll/train"]
    assert len(nlls) >= 2
    assert nlls[-1] < nlls[0]
    assert (tmp_path / "smoke.pt").exists()


def test_mixed_precision_request_on_cpu_is_a_safe_no_op(tmp_path, capsys):
    # AMP here targets CUDA only (see trainer/pfn_trainer.py) -- on CPU it
    # must fall back cleanly, not error, and training should behave exactly
    # as without it. This is the one thing about mixed_precision we can
    # actually verify without a GPU; the CUDA path itself is untested here.
    result = train_pfn(
        x_dim=1,
        d_model=16, n_layers=1, n_heads=2, d_ff=32, n_bins=16,
        batch_size=8, min_train=3, max_train=6, n_test=4,
        n_steps=20, log_every=10,
        prior_kwargs=dict(ecdf_n_draws=5, ecdf_samples_per_draw=50, cache_dir=None),
        checkpoint_path=tmp_path / "amp_smoke.pt",
        mixed_precision=True,
    )
    assert len(result["history"]["nll/train"]) >= 2
    assert "ignoring" in capsys.readouterr().out


def test_checkpoint_round_trip(tmp_path):
    from anytimeacquisition.pipelines.train_pfn import load_pfn_checkpoint
    import torch

    train_pfn(
        x_dim=1, d_model=16, n_layers=1, n_heads=2, d_ff=32, n_bins=16,
        batch_size=8, min_train=3, max_train=6, n_test=4, n_steps=5, log_every=5,
        prior_kwargs=dict(ecdf_n_draws=5, ecdf_samples_per_draw=50, cache_dir=None),
        checkpoint_path=tmp_path / "ckpt.pt",
    )
    model, bar_dist, ckpt = load_pfn_checkpoint(tmp_path / "ckpt.pt")

    x_train, y_train = torch.rand(1, 4, 1), torch.rand(1, 4)
    x_test = torch.rand(1, 3, 1)
    logits = model(x_train, y_train, x_test)
    assert logits.shape == (1, 3, 16)
    assert bar_dist.num_bars == 16
    # bar_dist is the PFN's own submodule now, not a fresh reconstruction --
    # see models/pfn.py's docstring.
    assert bar_dist is model.bar_dist


def test_checkpoint_state_dict_contains_bar_dist_buffers(tmp_path):
    import torch

    train_pfn(
        x_dim=1, d_model=16, n_layers=1, n_heads=2, d_ff=32, n_bins=16,
        batch_size=8, min_train=3, max_train=6, n_test=4, n_steps=5, log_every=5,
        prior_kwargs=dict(ecdf_n_draws=5, ecdf_samples_per_draw=50, cache_dir=None),
        checkpoint_path=tmp_path / "ckpt.pt",
    )
    ckpt = torch.load(tmp_path / "ckpt.pt", weights_only=False)
    assert "bar_dist.borders" in ckpt["model_state"]
    assert "bar_dist.bucket_widths" in ckpt["model_state"]


def test_load_pfn_checkpoint_accepts_old_format_missing_bar_dist_keys(tmp_path):
    # Checkpoints saved before PFN owned bar_dist as a submodule have no
    # `bar_dist.*` keys at all -- must still load, since those buffers are a
    # deterministic function of n_bins (see load_pfn_checkpoint's docstring).
    import torch

    from anytimeacquisition.pipelines.train_pfn import load_pfn_checkpoint

    train_pfn(
        x_dim=1, d_model=16, n_layers=1, n_heads=2, d_ff=32, n_bins=16,
        batch_size=8, min_train=3, max_train=6, n_test=4, n_steps=5, log_every=5,
        prior_kwargs=dict(ecdf_n_draws=5, ecdf_samples_per_draw=50, cache_dir=None),
        checkpoint_path=tmp_path / "ckpt.pt",
    )
    ckpt = torch.load(tmp_path / "ckpt.pt", weights_only=False)
    old_format_state = {k: v for k, v in ckpt["model_state"].items() if not k.startswith("bar_dist.")}
    torch.save({**ckpt, "model_state": old_format_state}, tmp_path / "old_format_ckpt.pt")

    model, bar_dist, _ = load_pfn_checkpoint(tmp_path / "old_format_ckpt.pt")
    assert bar_dist.num_bars == 16


def test_extra_checkpoint_metadata_round_trips_as_sibling_keys(tmp_path):
    """Checkpoint lineage (train_pfn.py's main(): mlflow_run_id, git_commit)
    -- PFNTrainer's own extra_checkpoint_metadata, not reachable through the
    plain train_pfn() entry point (no MLflow run to reference there), so
    tested directly against PFNTrainer. Must land as sibling top-level keys,
    not merged into "config" -- that dict gets **-unpacked straight into
    PFN(**ckpt["config"]) at load time (load_pfn_checkpoint), so anything
    else in it would break that call with an unexpected kwarg."""
    import torch

    from anytimeacquisition.models.pfn import PFN
    from anytimeacquisition.priors.bnn import BNNPrior
    from anytimeacquisition.trainer.pfn_trainer import PFNTrainer

    prior = BNNPrior(
        batch_size=4, x_dim=1, seed=0,
        ecdf_n_draws=5, ecdf_samples_per_draw=50, cache_dir=None,
    )
    model = PFN(max_x_dim=1, d_model=16, n_heads=2, n_layers=1, d_ff=32, n_bins=16)
    checkpoint_path = tmp_path / "lineage_ckpt.pt"
    trainer = PFNTrainer(
        prior=prior, model=model, n_steps=2, n_test=4, log_every=1,
        checkpoint_path=checkpoint_path,
        model_config={"max_x_dim": 1, "d_model": 16, "n_heads": 2, "n_layers": 1, "d_ff": 32, "n_bins": 16},
        extra_checkpoint_metadata={"mlflow_run_id": "abc123", "git_commit": "deadbeef"},
    )
    trainer.run()

    ckpt = torch.load(checkpoint_path, weights_only=False)
    assert ckpt["mlflow_run_id"] == "abc123"
    assert ckpt["git_commit"] == "deadbeef"
    assert "mlflow_run_id" not in ckpt["config"]
    assert "git_commit" not in ckpt["config"]
    # PFN(**ckpt["config"]) (load_pfn_checkpoint's own reconstruction) must
    # still work -- the lineage keys must not have leaked into "config".
    PFN(**ckpt["config"])


def test_load_pfn_checkpoint_rejects_genuine_state_dict_mismatch(tmp_path):
    import pytest
    import torch

    from anytimeacquisition.pipelines.train_pfn import load_pfn_checkpoint

    train_pfn(
        x_dim=1, d_model=16, n_layers=1, n_heads=2, d_ff=32, n_bins=16,
        batch_size=8, min_train=3, max_train=6, n_test=4, n_steps=5, log_every=5,
        prior_kwargs=dict(ecdf_n_draws=5, ecdf_samples_per_draw=50, cache_dir=None),
        checkpoint_path=tmp_path / "ckpt.pt",
    )
    ckpt = torch.load(tmp_path / "ckpt.pt", weights_only=False)
    broken_state = {k: v for k, v in ckpt["model_state"].items() if k != "out_head.weight"}
    torch.save({**ckpt, "model_state": broken_state}, tmp_path / "broken_ckpt.pt")

    with pytest.raises(RuntimeError, match="doesn't match"):
        load_pfn_checkpoint(tmp_path / "broken_ckpt.pt")


def _tiny_2d_model_and_bar_dist():
    from anytimeacquisition.models.bar_distribution import BarDistribution, uniform_bin_borders
    from anytimeacquisition.models.pfn import PFN

    model = PFN(max_x_dim=2, d_model=16, n_heads=2, n_layers=1, d_ff=32, n_bins=16)
    bar_dist = BarDistribution(uniform_bin_borders(16))
    return model, bar_dist


def test_plot_prior_data_and_pfn_2d_heatmaps_smoke():
    import matplotlib

    matplotlib.use("Agg")
    model, bar_dist = _tiny_2d_model_and_bar_dist()
    prior_kwargs = dict(ecdf_n_draws=5, ecdf_samples_per_draw=50, cache_dir=None)
    fig = plot_prior_data_and_pfn_2d_heatmaps(
        model, bar_dist, prior_kwargs=prior_kwargs, n_train=6, grid_res=10, seed=3,
    )
    assert len(fig.axes) == 8  # 4 panels + 4 colorbars


def test_plot_pfn_posterior_2d_surface_smoke():
    import matplotlib

    matplotlib.use("Agg")
    model, bar_dist = _tiny_2d_model_and_bar_dist()
    prior_kwargs = dict(ecdf_n_draws=5, ecdf_samples_per_draw=50, cache_dir=None)
    fig = plot_pfn_posterior_2d_surface(
        model, bar_dist, prior_kwargs=prior_kwargs, n_train=6, grid_res=10, seed=3,
    )
    assert len(fig.axes) == 1


def test_plot_pfn_posterior_2d_surface_rejects_bad_quantiles():
    import pytest

    model, bar_dist = _tiny_2d_model_and_bar_dist()
    with pytest.raises(AssertionError):
        plot_pfn_posterior_2d_surface(model, bar_dist, quantiles=(0.1, 0.9))
