from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from raq_kv.experiments.provenance import collect_provenance, git_state


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    try:
        _git(tmp_path, "init", "-q")
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git not available")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "a.md").write_text("başlangıç\n", encoding="utf-8")
    _git(tmp_path, "add", "a.md")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def test_clean_and_dirty_states_with_utf8_diff(repo: Path):
    clean = git_state(repo)
    assert clean["commit"] and clean["dirty"] is False and clean["diff_sha256"] is None

    (repo / "a.md").write_text("değişiklik × ≤ 12 ölçüm\n", encoding="utf-8")  # non-ASCII diff must not crash
    dirty = git_state(repo)
    assert dirty["dirty"] is True and dirty["describe"].endswith("-dirty") and dirty["diff_sha256"]

    (repo / "a.md").write_text("başka değişiklik\n", encoding="utf-8")
    assert git_state(repo)["diff_sha256"] != dirty["diff_sha256"]


def test_collect_provenance_has_required_fields():
    prov = collect_provenance()
    for key in ("python", "packages", "torch_cuda_build", "git", "timestamp_utc", "gpus_at_start"):
        assert key in prov
    assert prov["packages"]["transformers"] == "5.17.0"
