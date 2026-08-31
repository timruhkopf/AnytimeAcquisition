import subprocess

import pytest

from anytimeacquisition.deployment.provenance import DirtyRepoError, record_provenance


def _init_repo(path):
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=path, check=True)
    (path / "f.txt").write_text("x")
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=path, check=True, capture_output=True)


def test_record_provenance_clean_repo(tmp_path):
    _init_repo(tmp_path)

    prov = record_provenance(["seed=1"], repo_dir=tmp_path)

    assert not prov.dirty
    assert len(prov.commit) == 40
    assert prov.overrides == ["seed=1"]


def test_record_provenance_dirty_raises(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "f.txt").write_text("changed")

    with pytest.raises(DirtyRepoError):
        record_provenance([], repo_dir=tmp_path)


def test_record_provenance_allow_dirty(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "f.txt").write_text("changed")

    prov = record_provenance([], allow_dirty=True, repo_dir=tmp_path)

    assert prov.dirty


def test_record_provenance_ignores_cwd(tmp_path, monkeypatch, tmp_path_factory):
    """Regression test: Hydra chdirs into its output dir before calling
    record_provenance, so the result must not depend on cwd."""
    _init_repo(tmp_path)
    unrelated_dir = tmp_path_factory.mktemp("not_a_repo")
    monkeypatch.chdir(unrelated_dir)

    prov = record_provenance([], repo_dir=tmp_path)

    assert not prov.dirty
    assert len(prov.commit) == 40
