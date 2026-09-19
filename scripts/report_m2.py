"""M2 report: what a single stored bit does, and whether the two sensitivity signals agree.

Three questions, each with its own guard:

1. **Damage profile** — by fault class (code vs metadata), bit significance, part and layer. The campaign's floor
   is the no-fault control, which must be exactly zero; if it is not, nothing else here is readable.
2. **Q48** — does quantization sensitivity predict fault sensitivity? Both maps are per (layer, part) and were
   measured on the same storage scheme. Reported as Spearman with a bootstrap CI, and, when KV magnitude
   statistics are available, as a partial correlation controlling for magnitude — because both signals could be
   proxies for "this region holds big numbers".
3. **Q49** — do faults in the answer-bearing tokens hurt more than faults in filler? Odds ratio on task-level
   failure, with same-type distractor tokens as the content-matched control.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from raq_kv.utils.io import read_json, read_jsonl
from raq_kv.utils.paths import resolve_output_root


def load_flat(raw_root: Path, name: str, inner: str) -> pd.DataFrame:
    """Flatten per-unit records that carry a list of per-injection rows."""
    frames = []
    for exp in sorted(raw_root.glob("*")):
        manifest_path = exp / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = read_json(manifest_path)
        if manifest["config"]["name"] != name or manifest.get("superseded_by"):
            continue
        records = read_jsonl(exp / "results.jsonl")
        if len(records) != manifest["unit_keys_total"]:
            print(f"  [skipped incomplete] {manifest['exp_id']} ({len(records)}/{manifest['unit_keys_total']})")
            continue
        rows = []
        for r in records:
            base = {k: v for k, v in r.items() if k not in (inner, "faults")}
            for row in r[inner]:
                rows.append({**base, **row})
        frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _bootstrap_spearman(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, n: int = 4000) -> tuple[float, float]:
    rhos = []
    idx = np.arange(len(x))
    for _ in range(n):
        pick = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(x[pick])) > 2 and len(np.unique(y[pick])) > 2:
            rhos.append(stats.spearmanr(x[pick], y[pick]).statistic)
    return (float(np.quantile(rhos, 0.025)), float(np.quantile(rhos, 0.975))) if rhos else (float("nan"),) * 2


def _partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Spearman correlation of x and y after regressing both on the control z (rank-based)."""
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    design = np.column_stack([np.ones_like(rz), rz])
    resid_x = rx - design @ np.linalg.lstsq(design, rx, rcond=None)[0]
    resid_y = ry - design @ np.linalg.lstsq(design, ry, rcond=None)[0]
    return float(stats.spearmanr(resid_x, resid_y).statistic)


def damage_section(df: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    control = df[df["stratum"] == "no_fault"]
    faults = df[df["stratum"] != "no_fault"]
    clean = bool((control["delta_nll_abs_max"] == 0).all()) if not control.empty else None
    by_stratum = faults.groupby("stratum").agg(
        n=("delta_nll_max", "size"), mean_worst=("delta_nll_max", "mean"),
        p95_worst=("delta_nll_max", lambda s: s.quantile(0.95)), max_worst=("delta_nll_max", "max"),
        tokens_changed=("tokens_changed", "mean")).round(4)
    codes = faults[faults["target"] == "code"]
    by_sig = codes.groupby("significance").agg(n=("delta_nll_max", "size"), mean_worst=("delta_nll_max", "mean"),
                                               tokens_changed=("tokens_changed", "mean")).round(4)
    by_part = codes.groupby("part").agg(n=("delta_nll_max", "size"),
                                        mean_worst=("delta_nll_max", "mean")).round(4)
    meta = faults[faults["target"] != "code"]
    by_meta = meta.groupby("significance").agg(n=("delta_nll_max", "size"), mean_worst=("delta_nll_max", "mean"),
                                               max_worst=("delta_nll_max", "max")).round(4) if not meta.empty else None
    body = [
        f"No-fault controls: {len(control)}, all exactly zero: **{clean}**.", "",
        "Damage by fault class (worst scored token, nats):", "", "```", by_stratum.to_string(), "```", "",
        "Code faults by bit significance:", "", "```", by_sig.to_string(), "```", "",
        "Code faults by part:", "", "```", by_part.to_string(), "```", "",
    ]
    if by_meta is not None:
        body += ["Metadata faults by float16 field:", "", "```", by_meta.to_string(), "```", ""]
    facts = {"control_clean": clean, "n_faults": len(faults),
             "metadata_over_code": float(meta["delta_nll_max"].mean() / codes["delta_nll_max"].mean())
             if not meta.empty else None,
             "msb_over_lsb": float(by_sig.loc["msb", "mean_worst"] / by_sig.loc["lsb", "mean_worst"])
             if {"msb", "lsb"} <= set(by_sig.index) else None}
    return "\n".join(body), facts


def fault_map(df: pd.DataFrame) -> pd.DataFrame:
    codes = df[(df["stratum"] != "no_fault") & (df["target"] == "code")]
    return codes.groupby(["layer", "part"])["delta_nll_max"].mean().rename("fault_damage").reset_index()


def quant_map(raw_root: Path, name: str) -> pd.DataFrame:
    rows = []
    for exp in sorted(raw_root.glob("*")):
        manifest_path = exp / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = read_json(manifest_path)
        if manifest["config"]["name"] != name or manifest.get("superseded_by"):
            continue
        for record in read_jsonl(exp / "results.jsonl"):
            rows += [r for r in record["regions"] if not r.get("is_null")]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.groupby(["layer", "part"])["delta_nll_mean"].mean().rename("quant_sensitivity").reset_index()


def q48_section(raw_root: Path, faults: pd.DataFrame, quant_name: str, rng: np.random.Generator):
    qm = quant_map(raw_root, quant_name)
    if qm.empty or faults.empty:
        return "_missing one of the two maps_\n", {}
    merged = fault_map(faults).merge(qm, on=["layer", "part"])
    magnitudes = raw_root.parent / "reports" / "kv_magnitude.json"
    rho = float(stats.spearmanr(merged["quant_sensitivity"], merged["fault_damage"]).statistic)
    lo, hi = _bootstrap_spearman(merged["quant_sensitivity"].to_numpy(), merged["fault_damage"].to_numpy(), rng)
    facts = {"n_cells": len(merged), "spearman": rho, "ci_low": lo, "ci_high": hi}
    lines = [f"{len(merged)} (layer, part) cells measured on the same storage scheme.", "",
             f"- Spearman ρ(quantization sensitivity, fault damage) = **{rho:.3f}** [95% CI {lo:.3f}, {hi:.3f}]"]
    if magnitudes.exists():
        mag = pd.DataFrame(read_json(magnitudes)["per_region"])
        merged = merged.merge(mag, on=["layer", "part"], how="left")
        if merged["mean_abs"].notna().all():
            partial = _partial_spearman(merged["quant_sensitivity"].to_numpy(), merged["fault_damage"].to_numpy(),
                                        merged["mean_abs"].to_numpy())
            facts["partial_spearman_given_magnitude"] = partial
            lines.append(f"- Partial ρ controlling for mean |KV| = **{partial:.3f}** "
                         "(if this collapses, both signals were proxies for magnitude)")
    else:
        lines.append("- _KV magnitude statistics not found; run `scripts/kv_stats.py` for the partial correlation_")
    by_part = merged.groupby("part")[["quant_sensitivity", "fault_damage"]].mean().round(4)
    lines += ["", "Means by part:", "", "```", by_part.to_string(), "```", ""]
    return "\n".join(lines), facts


def q49_section(df: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    if df.empty:
        return "_no probe fault run_\n", {}
    faults = df[(df["stratum"] != "no_fault") & df["golden_correct"]].copy()
    faults["failed"] = ~faults["faulty_correct"].astype(bool)
    by_role = faults.groupby("stratum").agg(n=("failed", "size"), failure_rate=("failed", "mean"),
                                            prediction_changed=("prediction_changed", "mean")).round(4)
    facts: dict[str, Any] = {"by_role": by_role["failure_rate"].to_dict(), "n": int(len(faults))}
    lines = ["Task-level failure rate by the role of the corrupted token (samples correct at golden only):", "",
             "```", by_role.to_string(), "```", ""]
    if {"target_value", "filler"} <= set(by_role.index):
        a = faults[faults["stratum"] == "target_value"]["failed"]
        b = faults[faults["stratum"] == "filler"]["failed"]
        table = [[int(a.sum()), int((~a).sum())], [int(b.sum()), int((~b).sum())]]
        odds, p = stats.fisher_exact(table)
        facts |= {"odds_ratio_target_vs_filler": float(odds), "fisher_p": float(p)}
        lines.append(f"- Answer tokens vs filler: odds ratio **{odds:.2f}**, Fisher exact p = {p:.2e}")
    if {"target_value", "distractor_value"} <= set(by_role.index):
        a = faults[faults["stratum"] == "target_value"]["failed"]
        c = faults[faults["stratum"] == "distractor_value"]["failed"]
        odds, p = stats.fisher_exact([[int(a.sum()), int((~a).sum())], [int(c.sum()), int((~c).sum())]])
        facts |= {"odds_ratio_target_vs_distractor": float(odds), "fisher_p_distractor": float(p)}
        lines.append(f"- Answer tokens vs same-type distractor tokens (content-matched): odds ratio "
                     f"**{odds:.2f}**, p = {p:.2e}")
    if "meta.depth_nominal" in faults:
        depth = faults.groupby(["stratum", "meta.depth_nominal"])["failed"].mean().unstack().round(3)
        lines += ["", "Failure rate by role × nominal depth (depth is the obvious confound):", "",
                  "```", depth.to_string(), "```", ""]
    return "\n".join(lines), facts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--campaign", default="m2_fault_campaign")
    parser.add_argument("--quant-map", default="m1_quant_sensitivity_blocks")
    parser.add_argument("--probe", default="m2_fault_qa_probe")
    args = parser.parse_args()
    root = resolve_output_root(args.output_root)
    raw = root / "raw"
    rng = np.random.default_rng(0)

    body = ["# M2: single-bit faults in the stored KV cache", "", __doc__.split("\n\n", 1)[1], ""]
    campaign = load_flat(raw, args.campaign, "faults")
    for title, (text, facts) in (
        (f"Damage profile ({args.campaign})", damage_section(campaign) if not campaign.empty else ("_no run_", {})),
        ("Q48 — do the two sensitivities agree?", q48_section(raw, campaign, args.quant_map, rng)),
        ("Q49 — does the corrupted token's role matter?", q49_section(load_flat(raw, args.probe, "faults"))),
    ):
        body += [f"## {title}", "", text, ""]
        print(f"[{title}] {facts}")

    out = root / "reports" / "m2_fault_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(body), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
