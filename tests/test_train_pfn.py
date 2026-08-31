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
    nlls = result["history"]["train_nll"]
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
    assert len(result["history"]["train_nll"]) >= 2
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

    model = PFN(x_dim=2, d_model=16, n_heads=2, n_layers=1, d_ff=32, n_bins=16)
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
