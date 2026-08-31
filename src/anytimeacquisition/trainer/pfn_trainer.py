"""PFN pretraining loop (M2), as a `_target_`-instantiable trainer class --
same shape as `trainer/dummy.py`'s `DummyTrainer`: runtime objects (prior,
model) and the seed/logging hook are passed in at instantiate time
(`instantiate(cfg.trainer, prior=prior, model=model, seed=cfg.seed)`),
everything else is a plain config-driven hyperparameter on `self`. `run()`
takes no arguments. This replaces an earlier version that threaded every
hyperparameter through a `_train_loop(...)` helper function with a matching
call in `main()` that manually unpacked `cfg.xxx` for each one -- pure
duplication of the config schema, not an abstraction over it.

No separate `bar_dist` argument -- `model.bar_dist` is the one, since
`models/pfn.py` owns it as a submodule (see that module's docstring); a
second, independently-constructed `BarDistribution` here could silently
drift from the model's own if their `n_bins` ever disagreed.

`mixed_precision` (AMP): ifBO's own `train.py` uses this
(`torch.cuda.amp.autocast` + `GradScaler`, gated behind a
`train_mixed_precision` flag that defaults to False even there) --
see docs/log/. Implemented here with the modern, device-generic
`torch.amp` API rather than the deprecated `torch.cuda.amp` alias.
**Untested on real GPU hardware** -- this environment has no CUDA device
(local dev is deliberately pinned to CPU-only torch, see
`docs/OPEN_QUESTIONS.md` #7). The disabled/no-CUDA path (`GradScaler`/
`autocast` with `enabled=False`) is verified safe -- that's the path every
CPU run, including all of M2's current smoke checkpoints, actually takes.
The CUDA path itself needs validating for real once training moves to
hardware that has a GPU, before trusting it for a real run.
"""
import math
import time
from pathlib import Path
from typing import Callable

import torch

from anytimeacquisition.models.pfn import PFN
from anytimeacquisition.priors.bnn import BNNPrior


def _cosine_warmup_lr(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


class PFNTrainer:
    def __init__(
        self,
        prior: BNNPrior,
        model: PFN,
        seed: int = 0,
        n_steps: int = 500,
        min_train: int = 3,
        max_train: int = 100,
        n_test: int = 1000,
        lr: float = 1e-3,
        warmup_steps: int = 50,
        log_every: int = 50,
        checkpoint_path: str | Path | None = None,
        model_config: dict | None = None,
        on_log: Callable[[int, dict], None] | None = None,
        mixed_precision: bool = False,
    ):
        self.prior = prior
        self.model = model
        self.bar_dist = model.bar_dist
        self.seed = seed
        self.n_steps = n_steps
        self.min_train = min_train
        self.max_train = max_train
        self.n_test = n_test
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.log_every = log_every
        self.checkpoint_path = checkpoint_path
        self.model_config = model_config
        self.on_log = on_log
        self.mixed_precision = mixed_precision

    def run(self) -> dict:
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: _cosine_warmup_lr(step, self.warmup_steps, self.n_steps)
        )

        device_type = next(self.model.parameters()).device.type
        use_amp = self.mixed_precision and device_type == "cuda"
        if self.mixed_precision and not use_amp:
            print(f"mixed_precision=True requested but model is on '{device_type}', not 'cuda' -- "
                  "AMP here only targets CUDA GPUs (see trainer/pfn_trainer.py docstring); ignoring.")
        # enabled=False makes every GradScaler/autocast call below a no-op,
        # so the loop doesn't need a separate AMP/non-AMP code path.
        scaler = torch.amp.GradScaler(device_type, enabled=use_amp)

        generator = torch.Generator().manual_seed(self.seed)
        history = {"step": [], "train_nll": [], "eval_mse": []}
        t0 = time.perf_counter()

        for step in range(self.n_steps):
            self.prior.reset()
            n_train = int(torch.randint(self.min_train, self.max_train + 1, (1,), generator=generator).item())
            x_tr, y_tr, x_te, y_te = self.prior.sample_episode(n_train, self.n_test)

            with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                logits = self.model(x_tr, y_tr, x_te)
                loss = self.bar_dist(logits, y_te).mean()

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if step % self.log_every == 0 or step == self.n_steps - 1:
                with torch.no_grad():
                    # logits came out of the autocast block above, so under
                    # AMP it's float16 -- bar_dist's buffers (borders etc.)
                    # are float32, and outside of autocast torch.matmul
                    # refuses to mix the two ("expected scalar type Half but
                    # found Float"). Eval logging isn't perf-critical, so
                    # just force full precision here rather than re-entering
                    # autocast.
                    eval_mse = (self.bar_dist.mean(logits.float()) - y_te).square().mean().item()
                metrics = {"train_nll": loss.item(), "eval_mse": eval_mse}
                history["step"].append(step)
                history["train_nll"].append(metrics["train_nll"])
                history["eval_mse"].append(metrics["eval_mse"])
                print(f"step {step:5d}  train_nll={metrics['train_nll']:.4f}  eval_mse={metrics['eval_mse']:.4f}  "
                      f"n_train={n_train}  ({time.perf_counter() - t0:.1f}s elapsed)")
                if self.on_log is not None:
                    self.on_log(step, metrics)

        if self.checkpoint_path is not None:
            checkpoint_path = Path(self.checkpoint_path)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model_state": self.model.state_dict(), "config": self.model_config, "history": history},
                checkpoint_path,
            )
            print("saved checkpoint to", checkpoint_path)

        return {"model": self.model, "bar_dist": self.bar_dist, "prior": self.prior, "history": history}
