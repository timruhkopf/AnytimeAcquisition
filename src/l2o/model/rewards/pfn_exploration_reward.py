import torch


def sobol_monitor_generator(n, dimension):
    sobol = torch.quasirandom.SobolEngine(dimension=dimension, scramble=True)
    return sobol.draw(n)


def load_pfn_model():
    import pfns4bo

    # Import fix:
    #   File "/home/ruhkopf/PycharmProjects/AnytimeAcquisition/.venv/lib/python3.10/site-packages/torch/serialization.py", line 1549, in load
    #     return _load(
    #   File "/home/ruhkopf/PycharmProjects/AnytimeAcquisition/.venv/lib/python3.10/site-packages/torch/serialization.py", line 2143, in _load
    #     result = unpickler.load()
    #   File "/home/ruhkopf/PycharmProjects/AnytimeAcquisition/.venv/lib/python3.10/site-packages/torch/serialization.py", line 2132, in find_class
    #     return super().find_class(mod_name, name)
    #   File "/home/ruhkopf/.pycharm_helpers/pydev/_pydev_bundle/pydev_import_hook.py", line 21, in do_import
    #     module = self._system_import(name, *args, **kwargs)
    #   File "/home/ruhkopf/PycharmProjects/AnytimeAcquisition/.venv/lib/python3.10/site-packages/pfns4bo/transformer.py", line 9, in <module>
    #     from .layer import TransformerEncoderLayer, _get_activation_fn
    #   File "/home/ruhkopf/.pycharm_helpers/pydev/_pydev_bundle/pydev_import_hook.py", line 21, in do_import
    #     module = self._system_import(name, *args, **kwargs)
    #   File "/home/ruhkopf/PycharmProjects/AnytimeAcquisition/.venv/lib/python3.10/site-packages/pfns4bo/layer.py", line 5, in <module>
    #     from torch.nn.modules.transformer import _get_activation_fn, Module, Tensor, Optional, MultiheadAttention, Linear, Dropout, LayerNorm
    # ImportError: cannot import name 'Optional' from 'torch.nn.modules.transformer' (/home/ruhkopf/PycharmProjects/AnytimeAcquisition/.venv/lib/python3.10/site-packages/torch/nn/modules/transformer.py)

    import torch.nn.modules.transformer
    import typing
    import torch

    # Manually inject the missing names into the module pfns4bo is looking at
    torch.nn.modules.transformer.Optional = typing.Optional
    torch.nn.modules.transformer.Tensor = torch.Tensor

    # Now we can load the model without import errors
    return torch.load(pfns4bo.bnn_model, weights_only=False)


class PFNExplorationReward:
    """
    Core idea: Use the PFN under the current horizon to predict the current ppd estimate of test points
    (incl. e.g. Sobol anchor points). Doing so under varying horizons allows us to compute the
    information gain / variance reduction. Weighing the variance reduction by the quantile of the actual value for that location
    incentives targeted exploration over mere coverage. It also considers the current state of optimization, so
    the optimal action depends on the current horizon.

        Implementation Ideas:
        1. Use padding to batch parallelize the varying  with a fixed test set on the monitor points.
        This will however blow up the batch by T x T items The padding will require the compute only to be masked.
        Meaning this is computationally heavy, while factually still correct.
        TODO 2. Alternatively, We can do T key-value cached forward passes, that cache the training set up to the current horizon.
         Meaning, that the main computational cost lies in the T forward passes and the M monitor points
    """

    def __init__(self, pfn_model, device, monitor_sampler=None, **kwargs):
        self.pfn = pfn_model.to(device)
        self.pfn.eval()

        if monitor_sampler is None:
            monitor_sampler = lambda: sobol_monitor_generator(100, 2)  # Default: 10 Sobol points in 2D

        self.sample_monitor_points = monitor_sampler
        self.device = device

        self.env = None  # will be set externally on trainer init

    def __call__(self, obs_traj):
        B, T, D = obs_traj.shape

        # 1. Concern: Global Information Query
        # Sample monitor points once per rollout call
        test_x = self.sample_monitor_points().to(self.device)  # (M, 2)
        test_x = test_x.unsqueeze(1).repeat(1, B, 1)
        test_y = self.env.evaluate(test_x)  # (Batch, M, 1)

        # permute to meet obs_traj (B, T, D) format
        # fixme: check that the PFN expects (Batch, Seq_Len, Dim) format and not (Seq_Len, Batch, Dim) - this is a common source of bugs
        test_x = test_x.permute(1, 0, 2)  # (num_envs, M, 2)
        test_y = test_y.permute(1, 0, 2)  # (num_envs, M, 1)

        x_train = obs_traj.permute(1, 0, 2)  # (T, B, D)

        # 3. Concern: Batched Inference
        # pfn_bnn expects (Batch, Total_Len, Dim)
        with torch.no_grad():
            # The PFN predicts for everything after eval_pos (the monitor points)
            output = self.pfn(
                (
                    x_train,  # TODO (1, B, D) in KV-caching
                    x_test, # Consider make it possible to split the x_test? maybe overly complicated changes in the PFN?
                    y_train # TODO only (1, B) is needed in kv-caching
                ),
                single_eval_pos=eval_pos, # TODO check how this interacts with kv-caching

            )

            # Compute NLL of the monitor points given the varying horizons
            # target_y must be repeated to match the Seq * Batch expansion
            target_y = mon_y.repeat_interleave(obs_traj.size(0), dim=0)
            nll = self.pfn.criterion(output, target_y)  # (Seq * Batch,)

        # 4. Concern: Extracting Delta-Gain
        nll = nll.view(obs_traj.size(1), obs_traj.size(0))  # (Batch, Seq)
        nll = nll.permute(1, 0)  # (Seq, Batch)

        # Reward_t = NLL_{t-1} - NLL_t
        info_gain = nll[:-1] - nll[1:]
        # Pad first step with zero or a constant novelty
        first_step_gain = torch.zeros(1, obs_traj.size(1), device=self.device)
        return torch.cat([first_step_gain, info_gain], dim=0)
