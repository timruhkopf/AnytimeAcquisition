"""Run provenance: ties every training run to a specific, clean git commit."""
import subprocess
from dataclasses import dataclass


class DirtyRepoError(RuntimeError):
    pass


@dataclass
class Provenance:
    commit: str
    dirty: bool
    overrides: list[str]

    def as_mlflow_tags(self) -> dict[str, str]:
        return {
            "git_commit": self.commit,
            "git_dirty": str(self.dirty),
            "hydra_overrides": " ".join(self.overrides),
        }


def _run_git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def record_provenance(overrides: list[str], allow_dirty: bool = False) -> Provenance:
    """Capture the current commit and Hydra overrides for a run.

    Raises DirtyRepoError if the working tree has uncommitted changes,
    unless allow_dirty=True (e.g. for local debugging runs).
    """
    commit = _run_git("rev-parse", "HEAD")
    dirty = bool(_run_git("status", "--porcelain"))
    if dirty and not allow_dirty:
        raise DirtyRepoError(
            "Working tree has uncommitted changes. Commit before launching a "
            "tracked run, or pass allow_dirty=True for local debugging."
        )
    return Provenance(commit=commit, dirty=dirty, overrides=overrides)
