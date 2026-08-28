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


def test_record_provenance_clean_repo(tmp_path, monkeypatch):
    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)

    prov = record_provenance(["seed=1"])

    assert not prov.dirty
    assert len(prov.commit) == 40
    assert prov.overrides == ["seed=1"]


def test_record_provenance_dirty_raises(tmp_path, monkeypatch):
    _init_repo(tmp_path)
    (tmp_path / "f.txt").write_text("changed")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(DirtyRepoError):
        record_provenance([])


def test_record_provenance_allow_dirty(tmp_path, monkeypatch):
    _init_repo(tmp_path)
    (tmp_path / "f.txt").write_text("changed")
    monkeypatch.chdir(tmp_path)

    prov = record_provenance([], allow_dirty=True)

    assert prov.dirty
