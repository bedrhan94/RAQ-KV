"""Runner guarantees: resume without duplicates, no model load when finished, conflicts detected."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from raq_kv.config.schema import ExperimentConfig
from raq_kv.experiments.runner import ExperimentConflictError, run_experiment
from raq_kv.utils.io import read_json, read_jsonl, write_json_atomic


class FakeTask:
    def __init__(self, fail_at: str | None = None, fingerprint: str = "fp") -> None:
        self.fail_at = fail_at
        self.fingerprint = fingerprint
        self.setup_calls = 0
        self.ran: list[str] = []

    def unit_keys(self) -> list[str]:
        return [f"u{i}" for i in range(5)]

    def data_fingerprint(self) -> dict[str, Any]:
        return {"fp": self.fingerprint}

    def setup(self) -> dict[str, Any]:
        self.setup_calls += 1
        return {}

    def run_unit(self, unit_key: str) -> dict[str, Any]:
        if unit_key == self.fail_at:
            raise RuntimeError("simulated crash")
        self.ran.append(unit_key)
        return {"value": int(unit_key[1:]) ** 2}


def make_cfg(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate({
        "name": "fake", "seed": 0,
        "model": {"hf_id": "org/m", "revision": "abcdef123", "short_name": "m", "device": "cpu"},
        "dataset": {"name": "wikitext2", "revision": "1234567abc", "n_chunks": 1, "prefix_len": 4,
                    "continuation_len": 2},
        "task": {"kind": "cache_ppl"},
        "output": {"root": str(tmp_path)},
    })


def test_resume_after_crash_has_no_duplicates(tmp_path: Path):
    cfg = make_cfg(tmp_path)
    crashing = FakeTask(fail_at="u3")
    with pytest.raises(RuntimeError):
        run_experiment(cfg, task_factory=lambda _c: crashing, log=lambda _m: None)
    assert crashing.ran == ["u0", "u1", "u2"]

    resumed = FakeTask()
    summary = run_experiment(cfg, task_factory=lambda _c: resumed, log=lambda _m: None)
    assert resumed.ran == ["u3", "u4"] and summary.units_skipped == 3

    records = read_jsonl(summary.exp_dir / "results.jsonl")
    assert [r["unit_key"] for r in records] == ["u0", "u1", "u2", "u3", "u4"]
    assert len({r["unit_id"] for r in records}) == 5
    manifest = read_json(summary.exp_dir / "manifest.json")
    assert [r["status"] for r in manifest["runs"]] == ["failed at u3", "completed"]
    assert len(read_jsonl(summary.exp_dir / "errors.jsonl")) == 1


def test_unit_seeds_are_independent_of_resume(tmp_path: Path):
    cfg = make_cfg(tmp_path)
    with pytest.raises(RuntimeError):
        run_experiment(cfg, task_factory=lambda _c: FakeTask(fail_at="u2"), log=lambda _m: None)
    summary = run_experiment(cfg, task_factory=lambda _c: FakeTask(), log=lambda _m: None)
    resumed_seeds = {r["unit_key"]: r["unit_seed"] for r in read_jsonl(summary.exp_dir / "results.jsonl")}

    fresh_cfg = make_cfg(tmp_path / "fresh")
    fresh = run_experiment(fresh_cfg, task_factory=lambda _c: FakeTask(), log=lambda _m: None)
    assert resumed_seeds == {r["unit_key"]: r["unit_seed"] for r in read_jsonl(fresh.exp_dir / "results.jsonl")}


def test_finished_experiment_does_not_call_setup(tmp_path: Path):
    cfg = make_cfg(tmp_path)
    run_experiment(cfg, task_factory=lambda _c: FakeTask(), log=lambda _m: None)
    again = FakeTask()
    summary = run_experiment(cfg, task_factory=lambda _c: again, log=lambda _m: None)
    assert again.setup_calls == 0 and summary.units_run == 0


def test_conflicts_are_detected(tmp_path: Path):
    cfg = make_cfg(tmp_path)
    with pytest.raises(RuntimeError):
        run_experiment(cfg, task_factory=lambda _c: FakeTask(fail_at="u1"), log=lambda _m: None)
    with pytest.raises(ExperimentConflictError, match="fingerprint"):
        run_experiment(cfg, task_factory=lambda _c: FakeTask(fingerprint="other"), log=lambda _m: None)

    manifest_path = next(tmp_path.glob("raw/*/manifest.json"))
    manifest = read_json(manifest_path)
    manifest["runs"][0]["provenance"]["git"]["commit"] = "0000000"
    write_json_atomic(manifest_path, manifest)
    with pytest.raises(ExperimentConflictError, match="code state"):
        run_experiment(cfg, task_factory=lambda _c: FakeTask(), log=lambda _m: None)
    run_experiment(cfg, allow_code_change=True, task_factory=lambda _c: FakeTask(), log=lambda _m: None)

    manifest = read_json(manifest_path)
    manifest["config_hash"] = "tampered"
    write_json_atomic(manifest_path, manifest)
    with pytest.raises(ExperimentConflictError, match="config hash"):
        run_experiment(cfg, task_factory=lambda _c: FakeTask(), log=lambda _m: None)


def test_a_change_to_the_executed_source_mid_process_is_refused(tmp_path, monkeypatch):
    """A sweep that edits executed source between its experiments would record a code state that never ran."""
    from raq_kv.experiments import runner as runner_mod

    monkeypatch.setattr(runner_mod, "_PROCESS_CODE_STATE", None)
    first = {"commit": "abc", "diff_sha256": "d1", "executed_code_sha256": "e1"}
    runner_mod._check_process_code_state(first)
    runner_mod._check_process_code_state(dict(first))  # same state again is fine
    with pytest.raises(runner_mod.ExperimentConflictError, match="executed source changed"):
        runner_mod._check_process_code_state({**first, "executed_code_sha256": "e2"})


def test_editing_something_the_campaign_never_executes_does_not_abort_it(monkeypatch):
    """The point of the change: D24's rule is about `src/`, and the guard used to be about the whole repository.

    Editing a decision record, a test or a report script mid-campaign aborted a run that D24 permits -- it cost a rung
    of D76 plus a supersede. So did `git commit`, because a diff is relative to HEAD while the executing files are
    unchanged, and committing a record between runs is the normal workflow here. Both now show up as a moved
    `diff_sha256` or `commit` with an unchanged `executed_code_sha256`, and both are accepted.
    """
    from raq_kv.experiments import runner as runner_mod

    monkeypatch.setattr(runner_mod, "_PROCESS_CODE_STATE", None)
    runner_mod._check_process_code_state({"commit": "abc", "diff_sha256": "d1", "executed_code_sha256": "e1"})
    runner_mod._check_process_code_state({"commit": "abc", "diff_sha256": "d2", "executed_code_sha256": "e1"})
    runner_mod._check_process_code_state({"commit": "def", "diff_sha256": None, "executed_code_sha256": "e1"})


def test_the_executed_hash_ignores_exempt_modules_and_notices_the_rest(tmp_path):
    """Measured on a real tree rather than asserted: the hash must move for one file and not for the other."""
    import shutil

    TOUCH = chr(10) + "# touched" + chr(10)

    from raq_kv.experiments.provenance import EXECUTED_CODE_EXEMPT, executed_code_sha256
    from raq_kv.utils.paths import repo_root

    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    shutil.copytree(repo_root() / "src" / "raq_kv", root / "src" / "raq_kv")
    base = executed_code_sha256(root)
    assert base is not None

    exempt = root / "src" / EXECUTED_CODE_EXEMPT[1]          # an analysis module a campaign never imports
    assert exempt.exists(), exempt
    exempt.write_text(exempt.read_text(encoding="utf-8") + TOUCH, encoding="utf-8")
    assert executed_code_sha256(root) == base, "editing an exempt module must not move the executed hash"

    executed = root / "src" / "raq_kv" / "faults" / "storage.py"   # squarely in the campaign's graph
    executed.write_text(executed.read_text(encoding="utf-8") + TOUCH, encoding="utf-8")
    assert executed_code_sha256(root) != base, "editing an executed module must move it"

    # And a brand-new module counts, which a git diff could not see if it were untracked.
    (root / "src" / "raq_kv" / "faults" / "brand_new.py").write_text("x = 1" + chr(10), encoding="utf-8")
    assert executed_code_sha256(root) != base
