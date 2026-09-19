"""M1 sensitivity report: is per-region quantization damage heterogeneous, and is the ranking stable?

For each region (layer × keys/values) the profiler recorded the paired per-token NLL increase on every chunk.
This script asks three questions the allocation policy depends on (plan §I, H1):

1. **Size** - mean damage per region with a 95% t-CI over chunks.
2. **Concentration** - what share of total damage the most sensitive 10%/25% of regions carry.
3. **Stability** - split-half reliability: rank the regions on two disjoint halves of the chunks and correlate.
   Without stability, a ranking measured on calibration data cannot transfer, and an allocation policy built on
   it is fitting noise.

A one-way ANOVA across regions (chunks as replicates) gives a heterogeneity test: F >> 1 means regions differ by
more than chunk-to-chunk variation.
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


def load_profile(raw_root: Path, name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    for exp in sorted(raw_root.glob("*quant_sensitivity*")):
        manifest = read_json(exp / "manifest.json")
        if manifest["config"]["name"] != name or manifest.get("superseded_by"):
            continue
        records = read_jsonl(exp / "results.jsonl")
        if not records:
            continue
        rows = [{"chunk": r["sample_id"], "golden_nll": r["golden_nll_mean"], **region}
                for r in records for region in r["regions"]]
        meta = {"exp_id": manifest["exp_id"], "n_chunks": len(records),
                "key_spec": records[0]["key_spec"], "value_spec": records[0]["value_spec"],
                "null_bits": manifest["config"]["task"].get("null_bits")}
        return pd.DataFrame(rows), meta
    raise SystemExit(f"no completed profile named '{name}'")


def _region_keys(df: pd.DataFrame) -> list[str]:
    """Group by head as well when the profile is head-level; otherwise a head profile collapses to layer means."""
    return ["layer", "part", "head"] if df["head"].notna().any() else ["layer", "part"]


def _region_table(df: pd.DataFrame, null_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-region mean with a CI, plus a Welch test against the null distribution with Holm correction.

    Comparing a region's *mean* to the spread of individual null measurements is over-conservative; the right
    question is whether the region's measurements come from a different distribution than the null ones.
    """
    keys = _region_keys(df)
    grouped = df.groupby(keys)["delta_nll_mean"]
    table = grouped.agg(["mean", "std", "count"]).reset_index()
    table["sem"] = table["std"] / np.sqrt(table["count"])
    table["ci95_half"] = stats.t.ppf(0.975, table["count"] - 1) * table["sem"]
    table["significant"] = table["mean"] - table["ci95_half"] > 0

    if null_df is not None and not null_df.empty:
        null_values = null_df["delta_nll_mean"].to_numpy()
        p_values = []
        for _, group in df.groupby(keys):
            result = stats.ttest_ind(group["delta_nll_mean"].to_numpy(), null_values, equal_var=False)
            p_values.append(float(result.pvalue))
        order = np.argsort(p_values)
        n = len(p_values)
        holm = np.empty(n)
        running = 0.0
        for rank, idx in enumerate(order):
            running = max(running, (n - rank) * p_values[idx])
            holm[idx] = min(1.0, running)
        table["p_vs_null"] = p_values
        table["p_holm"] = holm
        table["differs_from_null"] = (holm < 0.05) & (table["mean"] > null_values.mean())
    return table.sort_values("mean", ascending=False).reset_index(drop=True)


def _concentration(values: np.ndarray) -> dict[str, float]:
    positive = np.clip(values, 0, None)
    order = np.sort(positive)[::-1]
    total = order.sum()
    if total <= 0:
        return {"top10_share": float("nan"), "top25_share": float("nan")}
    cum = order.cumsum() / total
    return {"top10_share": float(cum[max(1, round(0.1 * len(order))) - 1]),
            "top25_share": float(cum[max(1, round(0.25 * len(order))) - 1])}


def _split_half(df: pd.DataFrame, rng: np.random.Generator, n_splits: int = 200) -> dict[str, float]:
    chunks = sorted(df["chunk"].unique())
    wide = df.pivot_table(index=_region_keys(df), columns="chunk", values="delta_nll_mean")
    rhos = []
    for _ in range(n_splits):
        perm = rng.permutation(chunks)
        a, b = perm[: len(perm) // 2], perm[len(perm) // 2 :]
        ra, rb = wide[list(a)].mean(axis=1), wide[list(b)].mean(axis=1)
        rho = stats.spearmanr(ra, rb).statistic
        if np.isfinite(rho):
            rhos.append(float(rho))
    if not rhos:
        return {"split_half_rho_median": float("nan"), "spearman_brown": float("nan")}
    median = float(np.median(rhos))
    return {"split_half_rho_median": median,
            # Spearman-Brown steps the half-sample reliability up to the full sample.
            "spearman_brown": float(2 * median / (1 + median)) if median > -1 else float("nan")}


def _noise_band(null_df: pd.DataFrame) -> dict[str, float]:
    """Spread of deltas when the 'perturbation' is lossless: the measurement's own floor."""
    if null_df.empty:
        return {}
    values = null_df["delta_nll_mean"].to_numpy()
    return {"null_n": int(len(values)), "null_mean": float(values.mean()), "null_sd": float(values.std(ddof=1)),
            "null_p2.5": float(np.quantile(values, 0.025)), "null_p97.5": float(np.quantile(values, 0.975)),
            "null_abs_max": float(np.abs(values).max())}


def analyse(df: pd.DataFrame, meta: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if "is_null" in df:
        null_df = df[df["is_null"].astype(bool)]
        df = df[~df["is_null"].astype(bool)]
    else:
        null_df = df.iloc[0:0]
    band = _noise_band(null_df)
    table = _region_table(df, null_df)
    groups = [g["delta_nll_mean"].to_numpy() for _, g in df.groupby(_region_keys(df))]
    f_stat, p_value = stats.f_oneway(*groups)
    rng = np.random.default_rng(0)
    stability = _split_half(df, rng)
    conc = _concentration(table["mean"].to_numpy())
    by_part = df.groupby("part")["delta_nll_mean"].mean().round(5).to_dict()

    clears_band = int((table["mean"] - table["ci95_half"] > band.get("null_p97.5", 0.0)).sum()) if band else None
    differs = int(table["differs_from_null"].sum()) if "differs_from_null" in table else None
    facts = {**meta, **conc, **stability, **band, "anova_F": float(f_stat), "anova_p": float(p_value),
             "n_regions": len(table), "significant_regions": int(table["significant"].sum()),
             "max_mean_delta": float(table["mean"].max()), "min_mean_delta": float(table["mean"].min()),
             "regions_clearing_noise_band": clears_band, "regions_above_null_holm": differs, "by_part": by_part}
    pivot = df.pivot_table(index="layer", columns="part", values="delta_nll_mean").round(4)
    body = [
        f"### `{meta['exp_id']}`", "",
        f"Keys stressed with {meta['key_spec']['bits']}b/{meta['key_spec']['axis']}, "
        f"values with {meta['value_spec']['bits']}b/{meta['value_spec']['axis']}; "
        f"{meta['n_chunks']} chunks × {len(table)} regions.", "",
        f"- Region damage spans **{table['mean'].min():+.4f} … {table['mean'].max():+.4f} nats** "
        f"(largest: {table['part'].iloc[0]} of layer {int(table['layer'].iloc[0])}); "
        f"{int(table['significant'].sum())}/{len(table)} regions are significantly > 0. A negative mean is "
        "physically meaningless and bounds the noise from below.",
        (f"- Noise band from {band['null_n']} lossless ({meta['null_bits']}-bit) region measurements: "
         f"mean {band['null_mean']:+.5f}, sd {band['null_sd']:.5f}, 95% span "
         f"[{band['null_p2.5']:+.5f}, {band['null_p97.5']:+.5f}]. "
         f"**{clears_band}/{len(table)} regions clear it** (lower CI above the band's upper edge, conservative); "
         f"**{differs}/{len(table)} differ from the null distribution** (Welch test, Holm-corrected p < 0.05)."
         if band else "- No null condition in this run: the noise floor is unknown."),
        f"- Mean damage: keys {by_part.get('keys', float('nan')):.5f} vs values "
        f"{by_part.get('values', float('nan')):.5f} nats.",
        f"- Concentration: top 10% of regions carry {conc['top10_share']:.1%} of positive damage, "
        f"top 25% carry {conc['top25_share']:.1%}.",
        f"- Heterogeneity across regions: one-way ANOVA F = {f_stat:.2f} (p = {p_value:.2e}).",
        f"- Ranking stability (split-half Spearman, median of 200 splits): "
        f"{stability['split_half_rho_median']:.3f}; Spearman-Brown corrected {stability['spearman_brown']:.3f}.",
        "", "Top 10 regions (mean ± 95% CI over chunks):", "",
        "```", table.head(10)[["layer", "part", "mean", "ci95_half", "significant"]].round(4).to_string(index=False),
        "```", "", "Per-layer means:", "", "```", pivot.to_string(), "```", "",
    ]
    return "\n".join(body), facts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--names", nargs="*", default=["m1_quant_sensitivity", "m1_quant_sensitivity_mismatched"])
    args = parser.parse_args()
    root = resolve_output_root(args.output_root)

    body = ["# M1: quantization-sensitivity profile", "", __doc__.split("\n\n", 1)[1], ""]
    for name in args.names:
        try:
            df, meta = load_profile(root / "raw", name)
        except SystemExit as exc:
            body += [f"### {name}", "", f"_{exc}_", ""]
            continue
        section, facts = analyse(df, meta)
        body.append(section)
        df.to_csv(root / "csv" / f"{name}_regions.csv", index=False, encoding="utf-8")
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in facts.items()})

    out = root / "reports" / "m1_sensitivity_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(body), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
