"""Reading a completed fault arm off disk, and the sample-level rates a contrast is computed over.

Shared for the reason `contrasts.py` and `groups.py` are: D61 needs the same arm access as
`scripts/report_fault_retention.py`, and this project has repaired three cases of one quantity computed twice.
The rules encoded here are not conveniences -- each one is a decision:

* **An incomplete run is refused, not partially reported.** A resumed campaign that died at 13 of 120 would
  otherwise be summarised as though it were the whole thing.
* **A superseded run is skipped.** Records are renamed and marked, never deleted (the project's provenance rule),
  so the reader has to know to ignore them.
* **Rates are per SAMPLE, never pooled over trials.** Eight draws on one sample are not eight independent
  observations, and every contrast in this project is paired at the sample level.
* **The shared-sample set is the intersection of what every arm answers correctly with no faults.** An arm cannot
  be charged for a sample its own clean cache already gets wrong.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from raq_kv.utils.io import read_json, read_jsonl


def load_arm(raw_root: Path, name: str) -> dict[str, dict]:
    for exp in sorted(raw_root.glob("*")):
        if ".superseded-" in exp.name or not (exp / "manifest.json").exists():
            continue
        manifest = read_json(exp / "manifest.json")
        if manifest.get("superseded_by") or manifest["config"]["name"] != name:
            continue
        records = read_jsonl(exp / "results.jsonl")
        if len(records) != manifest["unit_keys_total"]:
            raise SystemExit(f"{name}: incomplete ({len(records)}/{manifest['unit_keys_total']})")
        return {r["sample_id"]: r for r in records}
    raise SystemExit(f"no complete run found for arm {name!r}")


def shared_samples(arms: dict[str, dict[str, dict]]) -> list[str]:
    """Samples every arm answers correctly with no faults, and that the full cache answers too."""
    keep: set[str] | None = None
    for data in arms.values():
        good = {s for s, r in data.items()
                if r["golden_correct"] and r.get("unpruned_correct") is not False}
        keep = good if keep is None else (keep & good)
    return sorted(keep or ())


def _trials(record: dict) -> list[dict]:
    return [t for t in record["faults"] if t["trial"] >= 0]


def _controls_clean(data: dict[str, dict]) -> tuple[int, int]:
    """No-fault scorings must show exactly zero damage, or the arm's own reference is unstable."""
    total = changed = 0
    for r in data.values():
        for t in r["faults"]:
            if t["trial"] < 0:
                total += 1
                changed += int(bool(t["answer_changed"]))
    return changed, total


def rates(data: dict[str, dict], samples: list[str], field: str) -> np.ndarray:
    out = []
    for s in samples:
        trials = _trials(data[s])
        if field == "task_failure":
            out.append(sum(1 for t in trials if not t["faulty_correct"]) / len(trials))
        else:
            out.append(sum(1 for t in trials if t["answer_changed"]) / len(trials))
    return np.array(out)
