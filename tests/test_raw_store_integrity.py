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
