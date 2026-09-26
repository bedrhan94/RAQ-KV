"""Every live experiment on disk must still be loadable under the current schema.

A schema change that silently orphans stored results is only discovered when an analysis script crashes
mid-campaign. Superseded runs are exempt: they are kept as history, not as live data.
"""

from __future__ import annotations

import pytest

from raq_kv.config.schema import ExperimentConfig
from raq_kv.experiments.ids import experiment_id
from raq_kv.utils.io import read_json
from raq_kv.utils.paths import resolve_output_root


def _manifests():
    raw = resolve_output_root("outputs") / "raw"
    if not raw.exists():
        pytest.skip("no local results")
    return [(p.name, read_json(p / "manifest.json")) for p in sorted(raw.glob("*")) if (p / "manifest.json").exists()]


def test_live_manifests_parse_and_keep_their_ids():
    problems = []
    for name, manifest in _manifests():
        if manifest.get("superseded_by"):
            continue
        try:
            cfg = ExperimentConfig.model_validate(manifest["config"])
        except Exception as exc:  # noqa: BLE001 - the point is to report, not to raise here
            problems.append(f"{name}: config no longer valid ({type(exc).__name__}: {exc})")
            continue
        if experiment_id(cfg) != manifest["exp_id"]:
            problems.append(f"{name}: identity drifted, config now hashes to {experiment_id(cfg)}")
    assert not problems, "stored results orphaned by a schema change:\n" + "\n".join(problems)


def test_superseded_runs_explain_themselves():
    for name, manifest in _manifests():
        superseded = manifest.get("superseded_by")
        if superseded:
            assert superseded.get("reason"), f"{name}: superseded without a reason"


def _fake_run(root, exp_name, config_name, n_units, superseded=None):
    import json

    d = root / exp_name
    d.mkdir(parents=True)
    manifest = {"exp_id": exp_name, "config": {"name": config_name}, "unit_keys_total": n_units}
    if superseded:
        manifest["superseded_by"] = superseded
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (d / "results.jsonl").write_text(
        "\n".join(json.dumps({"sample_id": f"s{i}", "golden_correct": True, "faults": []})
                  for i in range(n_units)), encoding="utf-8")
    return d


def test_two_live_runs_of_one_arm_name_are_refused_instead_of_picked_by_hash_order(tmp_path):
    """The defect this guards was live for one commit and would have published a 2-sample result as a 320-trial one.

    `--set task.n_samples=2` produces a smoke run with the SAME config name as the real arm and a different
    identity hash. `load_arm` returned the first match in directory order -- hash order, so arbitrary -- and
    nothing in the report would have said which run it read.
    """
    import pytest

    from raq_kv.analysis.arms import load_arm

    raw = tmp_path / "raw"
    _fake_run(raw, "arm_aaaa1111", "d61_qwen_tail", 2)
    _fake_run(raw, "arm_bbbb2222", "d61_qwen_tail", 40)
    with pytest.raises(SystemExit, match="live complete runs"):
        load_arm(raw, "d61_qwen_tail")

    # Marking the smoke run superseded is the resolution, and then the real arm loads unambiguously.
    import json

    m = json.loads((raw / "arm_aaaa1111" / "manifest.json").read_text(encoding="utf-8"))
    m["superseded_by"] = {"exp_id": "arm_bbbb2222", "reason": "smoke run"}
    (raw / "arm_aaaa1111" / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    assert len(load_arm(raw, "d61_qwen_tail")) == 40


def test_an_incomplete_run_is_refused_rather_than_partially_reported(tmp_path):
    import pytest

    from raq_kv.analysis.arms import load_arm

    raw = tmp_path / "raw"
    d = _fake_run(raw, "arm_cccc3333", "some_arm", 13)
    import json

    m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    m["unit_keys_total"] = 120  # the resumed-campaign case: died at 13 of 120
    (d / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(SystemExit, match=r"incomplete \(13/120\)"):
        load_arm(raw, "some_arm")


def test_the_supersede_helper_writes_the_shape_the_integrity_test_reads(tmp_path):
    """`superseded_by` is an object with `reason`, not a bare string.

    The first version of the helper wrote a string, which made `test_superseded_runs_explain_themselves` raise
    AttributeError rather than fail -- a test that cannot report the problem it exists to catch. The shape is now
    asserted from both ends.
    """
    import importlib.util
    import json
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "supersede_run", Path(__file__).resolve().parents[1] / "scripts" / "supersede_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    raw = tmp_path / "raw"
    d = _fake_run(raw, "arm_dddd4444", "an_arm", 2)
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    renamed = mod.supersede(d, manifest, "a 2-sample smoke run", "the_full_arm")

    assert ".superseded-" in renamed.name, "the rename is what makes it visible in a listing"
    assert not d.exists() and (renamed / "results.jsonl").exists(), "records are kept, never deleted"
    stored = json.loads((renamed / "manifest.json").read_text(encoding="utf-8"))["superseded_by"]
    assert stored["reason"] == "a 2-sample smoke run" and stored["exp_id"] == "the_full_arm"
