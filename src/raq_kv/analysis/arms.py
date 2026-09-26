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
    """The one live, complete run of the arm called ``name``.

    **Two live runs of one name is an error, not a choice.** This used to return the first match in directory
    order, which is hash order and therefore arbitrary. A `--set task.n_samples=2` smoke run of a D61 arm carries
    the *same config name* as the real 40-sample arm and a different identity hash, so the reader could silently
    summarise the smoke run -- a 2-sample result reported as a 320-trial one, with nothing in the output saying so.
    Found while smoke-testing D61's wrapping tail cell, before any report had been written against it.

    Marking the older run superseded is the project's own mechanism for this and remains the right fix; the raise
    is here because a reader that picks arbitrarily cannot tell the operator that the mechanism was not used.
    """
    found: list[tuple[Path, dict]] = []
    for exp in sorted(raw_root.glob("*")):
        if ".superseded-" in exp.name or not (exp / "manifest.json").exists():
            continue
        manifest = read_json(exp / "manifest.json")
        if manifest.get("superseded_by") or manifest["config"]["name"] != name:
            continue
        records = read_jsonl(exp / "results.jsonl")
        if len(records) != manifest["unit_keys_total"]:
            raise SystemExit(f"{name}: incomplete ({len(records)}/{manifest['unit_keys_total']})")
        found.append((exp, {r["sample_id"]: r for r in records}))
    if not found:
        raise SystemExit(f"no complete run found for arm {name!r}")
    if len(found) > 1:
        listing = "\n  ".join(f"{e.name} ({len(d)} units)" for e, d in found)
        raise SystemExit(
            f"{name!r} has {len(found)} live complete runs and no way to choose between them:\n  {listing}\n"
            f"mark the ones that are not the result superseded (manifest `superseded_by` + the `.superseded-` "
            f"rename) rather than leaving a reader to pick by hash order")
    return found[0][1]


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
