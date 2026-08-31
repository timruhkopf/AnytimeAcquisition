from anytimeacquisition.callbacks.handler import Callback, CallbackHandler


def test_callback_runs_at_default_cadence():
    calls = []
    handler = CallbackHandler([Callback(name="probe", fn=lambda step, t: calls.append(step) or {"seen": step})])

    for step in range(6):
        handler.run(step, trainer=None, default_every_n_steps=2)

    assert calls == [0, 2, 4]


def test_callback_own_cadence_overrides_default():
    calls = []
    handler = CallbackHandler([
        Callback(name="coarse", fn=lambda step, t: calls.append(step) or {}, every_n_steps=3),
    ])

    for step in range(9):
        handler.run(step, trainer=None, default_every_n_steps=1)

    assert calls == [0, 3, 6]


def test_metrics_are_namespaced_under_callback_name():
    handler = CallbackHandler([Callback(name="real_benchmark", fn=lambda step, t: {"regret": 0.5})])

    metrics = handler.run(step=0, trainer=None, default_every_n_steps=1)

    assert metrics == {"real_benchmark/regret": 0.5}


def test_empty_or_none_result_logs_nothing():
    handler = CallbackHandler([
        Callback(name="prints_only", fn=lambda step, t: None),
        Callback(name="empty_dict", fn=lambda step, t: {}),
    ])

    assert handler.run(step=0, trainer=None, default_every_n_steps=1) == {}


def test_multiple_callbacks_merge_without_collision():
    handler = CallbackHandler([
        Callback(name="a", fn=lambda step, t: {"x": 1}),
        Callback(name="b", fn=lambda step, t: {"x": 2}),  # same inner key "x", different namespace
    ])

    metrics = handler.run(step=0, trainer=None, default_every_n_steps=1)

    assert metrics == {"a/x": 1, "b/x": 2}


def test_callback_receives_the_trainer_instance():
    class _FakeTrainer:
        value = 42

    seen = {}

    def probe(step, t):
        seen["value"] = t.value
        return {}

    handler = CallbackHandler([Callback(name="probe", fn=probe)])
    handler.run(step=0, trainer=_FakeTrainer(), default_every_n_steps=1)

    assert seen["value"] == 42


def test_pfn_trainer_wires_callback_metrics_into_history_and_on_log(tmp_path):
    """End-to-end: a Callback registered on PFNTrainer shows up namespaced
    in both the returned history and the on_log-facing metrics dict,
    alongside the loop's own built-in train_nll/eval_mse."""
    from anytimeacquisition.models.pfn import PFN
    from anytimeacquisition.priors.bnn import BNNPrior
    from anytimeacquisition.trainer.pfn_trainer import PFNTrainer

    prior = BNNPrior(
        batch_size=4, x_dim=1, seed=0,
        ecdf_n_draws=5, ecdf_samples_per_draw=50, cache_dir=None,
    )
    model = PFN(max_x_dim=1, d_model=16, n_heads=2, n_layers=1, d_ff=32, n_bins=16)

    logged = []
    trainer = PFNTrainer(
        prior=prior, model=model, n_steps=2, n_test=4, log_every=1,
        on_log=lambda step, metrics: logged.append((step, metrics)),
        callbacks=[Callback(name="edge_case", fn=lambda step, t: {"flag": 1.0})],
    )
    result = trainer.run()

    assert result["history"]["edge_case/flag"] == [1.0, 1.0]
    assert all("edge_case/flag" in metrics for _, metrics in logged)
    assert all("train/nll" in metrics and "eval/mse" in metrics for _, metrics in logged)


def test_dummy_trainer_merges_callback_metrics_into_result():
    from anytimeacquisition.trainer.dummy import DummyTrainer

    class _Prior:
        def sample(self, n):
            return list(range(n))

    class _Benchmark:
        def evaluate(self, x):
            return float(x[0])

    class _Surrogate:
        def fit(self, xs, ys):
            pass

    trainer = DummyTrainer(
        benchmark=_Benchmark(), prior=_Prior(), surrogate=_Surrogate(), n_steps=3,
        callbacks=[Callback(name="probe", fn=lambda step, t: {"n_steps_seen": t.n_steps})],
    )
    result = trainer.run()

    assert result["probe/n_steps_seen"] == 3
    assert result["n_steps"] == 3
    assert "best_y" in result
