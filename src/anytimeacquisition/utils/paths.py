"""Repo-root-anchored paths, computed from this file's own location rather
than the process's current working directory.

Hydra/MLflow both default to cwd-relative interpolation
(`${hydra:runtime.cwd}`) for run/tracking/checkpoint dirs -- fine when a
pipeline is launched from the repo root (the documented case), but silently
re-nests `outputs/`, `mlruns/`, and checkpoints under wherever it actually
gets launched from otherwise (e.g. an IDE run config that defaults to the
script's own directory), instead of the intended top-level, gitignored
siblings of `src/`. Same anchoring problem `deployment/provenance.py`'s
`_REPO_DIR` already works around, there for git commands specifically.

Registers the `aa_root` OmegaConf resolver (`${aa_root:}` in configs/*.yaml,
used for `hydra.run.dir`/`hydra.sweep.dir`, the MLflow tracking URI, and
checkpoint_path defaults) as an import side effect below -- one hardcoded
`Path(__file__)`-relative hop up to the repo root, done here once, rather
than every `@hydra.main` pipeline threading an env var into Hydra's config
resolution itself. `anytimeacquisition/__init__.py` imports this module
unconditionally, so the resolver is registered before any pipeline's
`hydra.main`-decorated `main()` runs, without that pipeline needing its own
import for it. `AA_PROJECT_ROOT` still overrides it (e.g. to point a cluster
run at shared storage), same override shape as `AA_MLFLOW_DIR`. Non-Hydra
call sites (demo `__main__` blocks, the plain `train_pfn()` function) import
`PROJECT_ROOT`/`CHECKPOINT_DIR` directly instead.

Lives in `utils/` (not directly under `anytimeacquisition/`) alongside
`utils/flatten.py` -- both are small, dependency-free helpers used across
multiple unrelated subpackages, not owned by any one Hydra config group.
"""
import os
from pathlib import Path

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_DIR = PROJECT_ROOT / "models"

if not OmegaConf.has_resolver("aa_root"):
    OmegaConf.register_new_resolver("aa_root", lambda: os.environ.get("AA_PROJECT_ROOT", str(PROJECT_ROOT)))
