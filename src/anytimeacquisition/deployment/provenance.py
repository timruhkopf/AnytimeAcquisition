"""Run provenance: ties every training run to a specific, clean git commit."""
import subprocess
from dataclasses import dataclass
from pathlib import Path

# Default: the repo containing this file, not the process's current working
# directory. Pipelines run under Hydra's legacy chdir behavior (version_base=
# None), which moves cwd into the run's output dir before record_provenance
# is called -- a bare `git ...` there fails with "not a git repository" (and
# on clusters with automounted home dirs, git's own upward directory search
# can additionally hit a filesystem-boundary check and refuse to cross it).
_REPO_DIR = Path(__file__).resolve().parent


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


def _run_git(repo_dir: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_dir), *args], text=True
    ).strip()


def record_provenance(
    overrides: list[str], allow_dirty: bool = False, repo_dir: Path | None = None
) -> Provenance:
    """Capture the current commit and Hydra overrides for a run.

    Raises DirtyRepoError if the working tree has uncommitted changes,
    unless allow_dirty=True (e.g. for local debugging runs).

    repo_dir defaults to this package's own repo, independent of the
    process's current working directory (see module docstring above).
    """
    repo_dir = repo_dir or _REPO_DIR
    commit = _run_git(repo_dir, "rev-parse", "HEAD")
    dirty = bool(_run_git(repo_dir, "status", "--porcelain"))
    if dirty and not allow_dirty:
        raise DirtyRepoError(
            "Working tree has uncommitted changes. Commit before launching a "
            "tracked run, or pass allow_dirty=True for local debugging."
        )
    return Provenance(commit=commit, dirty=dirty, overrides=overrides)
