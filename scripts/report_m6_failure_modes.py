"""How do the agent arms fail -- by picking the wrong tool, or by filling its arguments wrong?

`evaluation/scoring.py` has reported `named_gold_function` since M6 precisely so a failed tool call could be split
into those two modes, and D50 recorded that the split "was built for this and is not yet analysed". This analyses
it. No GPU: every number here comes off records already on disk.

The split matters because the two modes have different implications for the whole project. A **wrong function** means
the cache lost the gold schema -- a retention failure, and the compression axis is at fault. **Right function, wrong
arguments** means the schema survived well enough to be selected but not well enough to be transcribed, which is a
fidelity failure and points at quantization or at how much of the schema the span unit actually keeps. The same
overall task-failure rate can be either, and the allocator's levers differ.

    python scripts/report_m6_failure_modes.py

Conditioned, as everywhere in this project, on the samples the arm's own **golden** answers correctly: an arm cannot
be charged for a tool call its unpruned cache already gets wrong. `m6_bfcl_golden` supplies the no-compression
baseline, so the modes it fails in are the vehicle's floor rather than any arm's doing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from raq_kv.analysis.stats import wilson
from raq_kv.utils.io import read_json, read_jsonl
from raq_kv.utils.paths import resolve_output_root

# Golden first, so its floor is read before the arms are judged. Parameterised rather than fixed because D70
# replicates the whole set on a second family, and a hardcoded arm list is how a report ends up able to describe only
# the model it was written for.
QWEN = ("m6_bfcl_golden", "m6_bfcl_joint", "m6_bfcl_prefixcache", "m6_bfcl_magnitude", "m6_bfcl_recency")
LLAMA = ("m9l_bfcl_golden", "m9l_bfcl_joint", "m9l_bfcl_prefixcache", "m9l_bfcl_magnitude", "m9l_bfcl_recency")
FAMILIES = {"qwen": QWEN, "llama": LLAMA}


def live_runs(raw_root: Path, name: str) -> list[list[dict]]:
    """Every live complete run of ``name``. A list, because a sweep config shares its name across experiments.

    Deliberately not `analysis.arms.sole_run`: that refuses a name matching several runs, which is right for a
    single-experiment arm and wrong here -- these are read for their contents, not selected between.
    """
    out = []
    for exp in sorted(raw_root.glob("*")):
        if ".superseded-" in exp.name or not (exp / "manifest.json").exists():
            continue
        manifest = read_json(exp / "manifest.json")
        if manifest.get("superseded_by") or manifest["config"]["name"] != name:
            continue
        records = read_jsonl(exp / "results.jsonl")
        if len(records) == manifest["unit_keys_total"]:
            out.append(records)
    return out


def split(records: list[dict], prefix: str) -> dict[str, int]:
    """Failures by mode, over the samples this arm's own golden answers correctly.

    ``prefix`` is "pruned_" on a retention arm and "" on the golden arm, because the golden arm's own scoring *is*
    the outcome. `named_gold_function` is True when the model emitted a call naming a gold function, so among
    failures it separates "the schema was found and mis-transcribed" from "the schema was not found".
    """
    ok = n = wrong_fn = wrong_args = 0
    for r in records:
        # The golden arm has no separate golden; for it the record's own correctness is the quantity.
        if prefix and not r.get("golden_correct"):
            continue
        correct = r.get(f"{prefix}correct")
        named = r.get(f"{prefix}named_gold_function")
        if correct is None or named is None:
            continue
        n += 1
        if correct:
            ok += 1
        elif named:
            wrong_args += 1
        else:
            wrong_fn += 1
    return {"n": n, "correct": ok, "wrong_function": wrong_fn, "wrong_arguments": wrong_args}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--family", default="qwen", choices=sorted(FAMILIES), help="which arm set to read")
    parser.add_argument("--arms", help="comma-separated arm names, golden first; overrides --family")
    args = parser.parse_args()
    raw_root = resolve_output_root(args.output_root) / "raw"
    arm_names = tuple(a.strip() for a in args.arms.split(",")) if args.arms else FAMILIES[args.family]
    golden_name = arm_names[0]

    print("M6 agent failure modes -- wrong tool against wrong arguments\n")
    print(f"{'arm':22s}{'n':>5s}{'correct':>9s}{'fail':>6s}{'wrong fn':>10s}{'wrong args':>12s}"
          f"{'args share of failures':>25s}")
    out: dict[str, dict] = {}
    for name in arm_names:
        runs = live_runs(raw_root, name)
        if not runs:
            print(f"{name:22s}   no complete live run found")
            continue
        prefix = "" if name == golden_name else "pruned_"
        for records in runs:
            s = split(records, prefix)
            if not s["n"]:
                continue
            fails = s["wrong_function"] + s["wrong_arguments"]
            share = s["wrong_arguments"] / fails if fails else 0.0
            lo, hi = wilson(s["wrong_arguments"], fails) if fails else (0.0, 1.0)
            print(f"{name:22s}{s['n']:5d}{s['correct']:9d}{fails:6d}{s['wrong_function']:10d}"
                  f"{s['wrong_arguments']:12d}      {share:6.3f} [{lo:.3f}, {hi:.3f}]")
            out[name] = s | {"n_failures": fails, "wrong_arguments_share": share, "ci": (lo, hi)}

    if golden_name in out:
        g = out[golden_name]
        gf = g["n_failures"]
        print(f"\nThe vehicle's floor: with NO compression the model already fails {gf} of {g['n']} calls, "
              f"{g['wrong_arguments']} of them\nwith the right function named. Every arm below is measured only on "
              f"the calls the golden gets right, so\nthat floor is excluded from their counts rather than charged "
              f"to compression.")

    arms = [(k, v) for k, v in out.items() if k != golden_name and v["n_failures"]]
    if arms:
        print("\nreading:")
        for k, v in arms:
            dominant = "arguments" if v["wrong_arguments_share"] > 0.5 else "function choice"
            print(f"  {k:22s} failures are mostly **{dominant}** "
                  f"({v['wrong_arguments']} args / {v['wrong_function']} fn)")
        print("\n  A wrong FUNCTION means the cache lost the gold schema -- a retention failure, and the "
              "compression\n  axis is at fault. RIGHT function with wrong ARGUMENTS means the schema survived "
              "selection but not\n  transcription, which is a fidelity failure and points at quantization or at "
              "how much of a schema\n  the span unit keeps. The same task-failure rate can be either, and the "
              "allocator's levers differ.")

    dest = resolve_output_root(args.output_root) / "reports" / f"m6_failure_modes_{args.family}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
