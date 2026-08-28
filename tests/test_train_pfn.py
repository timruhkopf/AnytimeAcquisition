from anytimeacquisition.pipelines.train_pfn import train_pfn


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
