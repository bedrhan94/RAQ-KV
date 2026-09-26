"""Damage by target class -- metadata against codes -- for any fault-injection campaign.

This is the finding every protection decision in the project descends from: M3's arm design, the +0.075% metadata
guard, the allocator's protection stage. It has been computed by hand twice, in D17 and again in D55 for the second
family, and the two disagreed (2.4x against 2.845x on the same data) purely because D17 pooled the two metadata
classes differently. That disagreement is the argument for this file existing: a number quoted in three records
should come from one definition.

**The definition, fixed here.** Per fault site the campaign records `delta_nll_max`, the largest per-token NLL
increase that site caused. Sites are grouped by `target` -- `code`, `scale`, `zero` -- and the metadata figure is the
**count-weighted** mean over scale and zero together, because the two are drawn in proportion to their share of the
metadata and pooling them unweighted would let the rarer class dominate. That is D55's choice, and it is the one
carried forward.

    python scripts/report_target_class_damage.py --arms m2_fault_campaign,m9l_fault_campaign,d68_fault_campaign_4b

Reported beside the ratio: worst-case damage per class (the tail is what the guard is really bought for) and the
LSB -> MSB bit-position factor, which H1c says is near 1 while the stored value changes 128x.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from raq_kv.utils.io import read_json, read_jsonl
from raq_kv.utils.paths import resolve_output_root

META = ("scale", "zero")


def sites(raw_root: Path, name: str) -> list[dict]:
    """Every fault site of the one live complete run of ``name``."""
    for exp in sorted(raw_root.glob("*")):
        if ".superseded-" in exp.name or not (exp / "manifest.json").exists():
            continue
        manifest = read_json(exp / "manifest.json")
        if manifest.get("superseded_by") or manifest["config"]["name"] != name:
            continue
        records = read_jsonl(exp / "results.jsonl")
        if len(records) != manifest["unit_keys_total"]:
            raise SystemExit(f"{name}: incomplete ({len(records)}/{manifest['unit_keys_total']})")
        return [f for r in records for f in r["faults"]]
    raise SystemExit(f"no complete live run found for arm {name!r}")


def by_class(faults: list[dict]) -> dict[str, dict[str, float]]:
    """Mean and worst damage per target class, with the non-finite sites counted rather than absorbed.

    **Some metadata flips drive the NLL non-finite**, because a corrupted fp16 word can make the dequantized cache
    infinite and the model then produces NaN (D62 measured this at ~1.3% of metadata flips and found the collapses
    concentrated on zero-points). A plain `mean` over a column containing one NaN returns NaN, which is how the
    first version of this script reported every metadata figure -- and D55's original computation must have dropped
    them silently, since its numbers match the finite-only mean here exactly.

    So the mean is taken over the **finite** sites and `n_nonfinite` is reported beside it. That makes every ratio a
    **lower bound**: the excluded sites are not missing data, they are the worst outcomes there are.
    """
    out: dict[str, dict[str, float]] = {}
    for target in ("code", *META, "metadata"):
        group = [f for f in faults if (f["target"] in META if target == "metadata" else f["target"] == target)]
        if not group:
            continue
        vals = np.array([float(f["delta_nll_max"]) for f in group])
        finite = vals[np.isfinite(vals)]
        out[target] = {"n": int(vals.size), "n_nonfinite": int(vals.size - finite.size),
                       "mean_finite": float(finite.mean()) if finite.size else float("nan"),
                       "max_finite": float(finite.max()) if finite.size else float("nan")}
    return out


def bit_position_factor(faults: list[dict]) -> float | None:
    """Mean damage at the highest code bit over the lowest -- H1c's quantity.

    Code sites only: a metadata bit index means something entirely different (an fp16 field, not a magnitude), so
    mixing them would average two unrelated scales.
    """
    codes = [f for f in faults if f["target"] == "code"]
    if not codes:
        return None
    bits = sorted({int(f["bit"]) for f in codes})
    if len(bits) < 2:
        return None
    lo = np.mean([float(f["delta_nll_max"]) for f in codes if int(f["bit"]) == bits[0]])
    hi = np.mean([float(f["delta_nll_max"]) for f in codes if int(f["bit"]) == bits[-1]])
    return float(hi / lo) if lo else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", required=True, help="Comma-separated fault_injection config names")
    parser.add_argument("--output-root", default="outputs")
    args = parser.parse_args()
    raw_root = resolve_output_root(args.output_root) / "raw"

    print("damage by target class (delta_nll_max per site, nats)\n")
    print(f"{'arm':26s}{'code mean':>11s}{'meta mean':>11s}{'ratio':>9s}{'worst meta':>12s}"
          f"{'worst code':>12s}{'tail':>7s}{'LSB->MSB':>10s}{'meta NaN':>12s}{'code NaN':>12s}")
    out: dict = {}
    for name in (a.strip() for a in args.arms.split(",") if a.strip()):
        faults = sites(raw_root, name)
        cls = by_class(faults)
        if "code" not in cls or "metadata" not in cls:
            print(f"{name:26s}  no code or metadata sites")
            continue
        c, m = cls["code"], cls["metadata"]
        ratio = m["mean_finite"] / c["mean_finite"] if c["mean_finite"] else float("inf")
        tail = m["max_finite"] / c["max_finite"] if c["max_finite"] else float("inf")
        bf = bit_position_factor(faults)
        print(f"{name:26s}{c['mean_finite']:11.4f}{m['mean_finite']:11.4f}{ratio:8.3f}x"
              f"{m['max_finite']:12.2f}{c['max_finite']:12.3f}{tail:6.0f}x"
              f"{(bf if bf else float('nan')):10.3f}"
              f"{m['n_nonfinite']:8d}/{m['n']:<3d}{c['n_nonfinite']:8d}/{c['n']:<3d}")
        out[name] = {"classes": cls, "metadata_over_code_lower_bound": ratio, "tail_separation": tail,
                     "bit_position_factor": bf}
        # The destructive subclass, reported because it is consistent across models and actionable: a corrupted
        # zero-point collapses the model more often than a corrupted scale does.
        for sub in META:
            if sub in cls:
                s = cls[sub]
                print(f"{'':26s}  {sub:6s} non-finite {s['n_nonfinite']:3d}/{s['n']:<4d} "
                      f"= {s['n_nonfinite'] / s['n']:6.2%}   finite mean {s['mean_finite']:.4f}")

    print("\n  metadata mean is count-weighted over scale and zero together (D55's definition, fixed here after "
          "D17 and\n  D55 disagreed at 2.4x against 2.845x on the same data through a different pooling). Means are "
          "over\n  FINITE sites with the non-finite count shown: those flips made the NLL NaN, which is the worst "
          "outcome\n  there is, so every ratio is a LOWER BOUND. tail = worst metadata / worst code. LSB->MSB is "
          "H1c's\n  factor over CODE sites only, against a 128x change in the stored value.")

    dest = resolve_output_root(args.output_root) / "reports" / "target_class_damage.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
