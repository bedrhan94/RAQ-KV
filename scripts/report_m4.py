"""M4 report: what does dropping cached tokens cost, and does the choice of which tokens matter?

Three things this report refuses to do, each because an earlier milestone was bitten by it.

* **It never compares arms on pooled trials.** Retention has one outcome per sample, so clustering is not an
  issue here the way it was in M3 (D26) — but arms are still compared *paired over the samples every arm
  answers correctly without pruning*, which is the denominator the claim is about.
* **It breaks every number out by target depth.** The probe places its fact at nominal depth 0.1, 0.5 or 0.9,
  and recency keeps the deep ones for free. An aggregate accuracy hides that entirely, and a scorer that only
  beats recency on shallow targets is a different (and more interesting) claim than one that beats it overall.
* **It prints `kept_fraction_target` next to accuracy, always.** A scorer that holds accuracy while discarding
  the answer tokens has not found importance — it has found that the model did not need them. That is the
  D19–D21/D32 redundancy question and it must not be read as success.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from raq_kv.utils.io import read_json, read_jsonl
from raq_kv.utils.paths import resolve_output_root

DEPTH_BANDS = ((0.0, 0.3, "shallow"), (0.3, 0.7, "middle"), (0.7, 1.01, "deep"))


def load(raw_root: Path, names: list[str] | None) -> pd.DataFrame:
    rows = []
    for exp in sorted(raw_root.glob("*")):
        if ".superseded-" in exp.name or not (exp / "manifest.json").exists():
            continue
        manifest = read_json(exp / "manifest.json")
        if manifest.get("superseded_by") or manifest["config"]["task"]["kind"] != "retention":
            continue
        name = manifest["config"]["name"]
        if names and name not in names:
            continue
        records = read_jsonl(exp / "results.jsonl")
        if len(records) != manifest["unit_keys_total"]:
            print(f"  [skipped incomplete] {manifest['exp_id']} ({len(records)}/{manifest['unit_keys_total']})")
            continue
        cfg = manifest["config"].get("compression") or {}
        for r in records:
            meta = r.get("meta") or {}
            rows.append({
                "arm": name, "scorer": r.get("scorer"), "ratio": cfg.get("ratio"),
                "protect_initial": cfg.get("protect_initial"), "protect_recent": cfg.get("protect_recent"),
                "sample_id": r["sample_id"], "golden_correct": bool(r["golden_correct"]),
                "task_failure": bool(r["task_failure"]), "answer_changed": bool(r["answer_changed"]),
                "kept_fraction_target": r.get("kept_fraction_target"),
                "kept_fraction_distractor": r.get("kept_fraction_distractor"),
                "retention_ratio": r.get("retention_ratio"), "avg_kv_bits": r.get("avg_kv_bits"),
                "tokens_original": r.get("tokens_original"), "tokens_retained": r.get("tokens_retained"),
                "depth": meta.get("depth_actual"), "nominal_len": meta.get("nominal_len"),
                "fact_type": meta.get("fact_type"),
            })
    return pd.DataFrame(rows)


def _band(depth: float | None) -> str:
    if depth is None or not np.isfinite(depth):
        return "unknown"
    for lo, hi, label in DEPTH_BANDS:
        if lo <= depth < hi:
            return label
    return "unknown"


def arm_table(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["golden_correct"]]
    rows = []
    for arm, sub in ok.groupby("arm"):
        rows.append({
            "arm": arm, "scorer": sub["scorer"].iloc[0], "ratio": sub["ratio"].iloc[0],
            "samples": len(sub),
            "kv_bits_per_original_token": float((sub["avg_kv_bits"] * sub["tokens_retained"]
                                                 / sub["tokens_original"]).mean()),
            "retained": float(sub["retention_ratio"].mean()),
            "kept_target": float(sub["kept_fraction_target"].mean()),
            "task_failure": float(sub["task_failure"].mean()),
            "answer_changed": float(sub["answer_changed"].mean()),
        })
    return pd.DataFrame(rows).sort_values("task_failure").reset_index(drop=True)


def depth_table(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["golden_correct"]].copy()
    ok["band"] = ok["depth"].map(_band)
    out = (ok.groupby(["arm", "band"])
             .agg(n=("task_failure", "size"), task_failure=("task_failure", "mean"),
                  kept_target=("kept_fraction_target", "mean"))
             .reset_index())
    order = {label: i for i, (_, _, label) in enumerate(DEPTH_BANDS)}
    out["_o"] = out["band"].map(lambda b: order.get(b, 9))
    return out.sort_values(["arm", "_o"]).drop(columns="_o").reset_index(drop=True)


def paired(df: pd.DataFrame, a: str, b: str) -> dict | None:
    """Paired over the samples both arms answer correctly without pruning."""
    sub = df[df["golden_correct"]]
    x = sub[sub["arm"] == a].set_index("sample_id")["task_failure"]
    y = sub[sub["arm"] == b].set_index("sample_id")["task_failure"]
    shared = sorted(set(x.index) & set(y.index))
    if not shared:
        return None
    xv = x.reindex(shared).to_numpy(dtype=float)
    yv = y.reindex(shared).to_numpy(dtype=float)
    diff = xv - yv
    discordant = int((diff != 0).sum())
    rng = np.random.default_rng(0)
    boot = [float(np.mean(rng.choice(diff, diff.size, replace=True))) for _ in range(10000)]
    p = 1.0 if discordant == 0 else float(stats.wilcoxon(xv, yv).pvalue)
    return {"arm_a": a, "arm_b": b, "samples": len(shared), "rate_a": float(xv.mean()),
            "rate_b": float(yv.mean()), "diff": float(diff.mean()),
            "ci_lo": float(np.quantile(boot, 0.025)), "ci_hi": float(np.quantile(boot, 0.975)),
            "discordant": discordant, "p": p}


def _fmt(df: pd.DataFrame) -> str:
    return df.to_string(index=False, float_format=lambda v: f"{v:.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--arms", default=None, help="Comma-separated config names; default is every complete one")
    parser.add_argument("--compare", default=None, help="Two arm names to pair, comma-separated")
    args = parser.parse_args()

    raw_root = resolve_output_root(args.output_root) / "raw"
    names = args.arms.split(",") if args.arms else None
    print(f"M4 report — reading {raw_root}")
    df = load(raw_root, names)
    if df.empty:
        print("no completed retention runs found")
        return 1
    print(f"loaded {len(df)} records from {df['arm'].nunique()} arm(s)")

    print("\nPer arm, on the samples answered correctly without pruning")
    print(_fmt(arm_table(df)))
    print("\n  kept_target is the diagnostic: accuracy held while it is low means the model did not need those")
    print("  tokens, which is a redundancy result and not a scorer's success (D32).")

    print("\nBy target depth — recency keeps deep targets for free, so an aggregate hides the real comparison")
    print(_fmt(depth_table(df)))

    if args.compare:
        a, b = args.compare.split(",")
        result = paired(df, a, b)
        if result is None:
            print(f"\nno shared samples between {a} and {b}")
        else:
            print(f"\nPaired over {result['samples']} shared samples, task failure "
                  f"(Wilcoxon, bootstrap CI over samples)")
            print(f"  {a} = {result['rate_a']:.4f}   {b} = {result['rate_b']:.4f}")
            print(f"  difference {result['diff']:+.4f} [{result['ci_lo']:+.4f}, {result['ci_hi']:+.4f}]  "
                  f"p = {result['p']:.4g}  discordant = {result['discordant']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
