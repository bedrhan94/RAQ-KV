"""Mark a raw run superseded: `superseded_by` in its manifest, plus the `.superseded-` rename. Never deletes.

The project's provenance rule is that raw JSONL is never edited or deleted, so a run that turns out to be wrong,
a duplicate or a smoke test is *marked* and the readers skip it. That was done by hand until D61, where a
`--set task.n_samples=2` smoke run landed in `outputs/raw/` carrying the **same config name** as the real 40-sample
arm. `load_arm` now refuses an ambiguous name rather than picking by hash order, which turns that situation into an
error the operator has to resolve -- and this is how it gets resolved.

    python scripts/supersede_run.py --name d61_qwen_tail --units-below 40 \
        --by "2-sample smoke run" --replaced-by d61_qwen_tail
    python scripts/supersede_run.py --exp-id qwen3-...-3035b740 \
        --by "duplicate" --replaced-by d61_qwen_tail

Either select by `--exp-id` (exact) or by `--name` plus a filter, because the whole point is that a name alone is
ambiguous. `--units-below N` marks only runs with fewer than N units, which is the smoke-run case; without a
filter, a name matching more than one live run is refused.

The rename is what makes the marking visible in a directory listing, and the manifest field is what makes it
visible to a reader that globs. Both, because this project has had a reader ignore each of them separately.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from raq_kv.utils.paths import resolve_output_root


def live_runs(raw_root: Path) -> list[tuple[Path, dict]]:
    out = []
    for exp in sorted(raw_root.glob("*")):
        if ".superseded-" in exp.name or not (exp / "manifest.json").exists():
            continue
        manifest = json.loads((exp / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("superseded_by"):
            continue
        out.append((exp, manifest))
    return out


def n_units(exp: Path) -> int:
    results = exp / "results.jsonl"
    if not results.exists():
        return 0
    return sum(1 for line in results.read_text(encoding="utf-8").splitlines() if line.strip())


def supersede(exp: Path, manifest: dict, reason: str, replaced_by: str) -> Path:
    """Write the marker first, then rename. In that order, because a crash between them must not lose the reason.

    ``superseded_by`` is an object with ``exp_id`` and ``reason``, matching every marking already in the store --
    `tests/test_raw_store_integrity.py::test_superseded_runs_explain_themselves` reads `.reason` off it and a bare
    string makes that test raise rather than fail, which is how the shape was found.
    """
    manifest["superseded_by"] = {"exp_id": replaced_by, "reason": reason}
    manifest["superseded_at"] = datetime.now(UTC).isoformat()
    (exp / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    target = exp.with_name(f"{exp.name}.superseded-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}")
    exp.rename(target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp-id", help="Exact experiment directory name")
    parser.add_argument("--name", help="Config name; needs a filter, because a name alone can match several runs")
    parser.add_argument("--units-below", type=int, help="Only runs holding fewer than this many result records")
    parser.add_argument("--by", required=True, help="Why this run is superseded; goes in the manifest")
    parser.add_argument("--replaced-by", required=True,
                        help="exp_id (or a description) of the run that supersedes this one")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not (args.exp_id or args.name):
        raise SystemExit("select a run with --exp-id or --name")

    raw_root = resolve_output_root(args.output_root) / "raw"
    picked = []
    for exp, manifest in live_runs(raw_root):
        if args.exp_id and exp.name != args.exp_id:
            continue
        if args.name and manifest["config"]["name"] != args.name:
            continue
        if args.units_below is not None and n_units(exp) >= args.units_below:
            continue
        picked.append((exp, manifest))

    if not picked:
        raise SystemExit("nothing matched; no run was changed")
    if args.name and not args.exp_id and args.units_below is None and len(picked) > 1:
        listing = "\n  ".join(f"{e.name} ({n_units(e)} units)" for e, _ in picked)
        raise SystemExit(f"--name {args.name!r} matches {len(picked)} live runs:\n  {listing}\n"
                         f"add --units-below or use --exp-id; marking the wrong one is not recoverable by rerun")

    for exp, manifest in picked:
        if args.dry_run:
            print(f"would supersede {exp.name} ({n_units(exp)} units): {args.by}")
            continue
        renamed = supersede(exp, manifest, args.by, args.replaced_by)
        print(f"superseded {exp.name} ({n_units(exp)} units) -> {renamed.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
