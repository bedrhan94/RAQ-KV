"""M1 report: KV precision/granularity versus cache-conditioned perplexity.

Every variant scores the same chunks and the same target tokens, so quality differences are computed as *paired*
per-token NLL deltas against the full-precision passthrough run, with a bootstrap CI over chunks. Memory comes
from the recorded bit budget (codes + fp16 scales), not from tensor sizes (decisions D6).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raq_kv.evaluation.metrics import corpus_ppl
from raq_kv.utils.io import read_json, read_jsonl
from raq_kv.utils.paths import resolve_output_root

BASELINE_LABEL = "bf16 passthrough"


def _variant_label(cfg: dict[str, Any]) -> str:
    q = cfg.get("quantization")
    if q is None:
        return BASELINE_LABEL
    def part(spec: dict[str, Any] | None) -> str:
        if spec is None:
            return "fp"
        axis = {"per_tensor": "tensor", "per_head": "head", "per_channel": "chan", "per_token": "token",
                "group_channel": f"g{spec['group_size']}"}[spec["axis"]]
        return f"{spec['bits']}b/{axis}{'/sym' if spec['symmetric'] else ''}"
    label = f"K {part(q['keys'])} · V {part(q['values'])}"
    return label + (f" · resid {q['residual_window']}" if q.get("residual_window") else "")


def _load(raw_root: Path, name: str) -> list[dict[str, Any]]:
    runs = []
    for exp in sorted(raw_root.glob("*")):
        manifest_path = exp / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = read_json(manifest_path)
        if manifest["config"]["name"] != name or manifest.get("superseded_by"):
            continue
        records = read_jsonl(exp / "results.jsonl")
        if records:
            runs.append({"exp_id": manifest["exp_id"], "config": manifest["config"], "records": records})
    return runs


def _paired_delta(baseline: list[dict], variant: list[dict], rng: np.random.Generator, n_boot: int = 2000):
    """Mean per-token NLL increase over the baseline, with a bootstrap CI resampling chunks."""
    base = {r["unit_key"]: np.asarray(r["token_nll"], dtype=float) for r in baseline}
    var = {r["unit_key"]: np.asarray(r["token_nll"], dtype=float) for r in variant}
    keys = sorted(set(base) & set(var))
    per_chunk = np.array([float((var[k] - base[k]).mean()) for k in keys])
    if not len(per_chunk):
        return float("nan"), (float("nan"), float("nan"))
    boot = rng.choice(per_chunk, size=(n_boot, len(per_chunk)), replace=True).mean(axis=1)
    return float(per_chunk.mean()), (float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975)))


def build_table(raw_root: Path, name: str, baseline_name: str = "m0a_golden_ppl") -> pd.DataFrame:
    runs = _load(raw_root, name)
    if not runs:
        raise SystemExit(f"no results for experiment name '{name}'")
    baseline = next((r for r in runs if r["config"].get("quantization") is None), None)
    if baseline is None:
        # A sweep without its own full-precision arm (e.g. the causality comparison) is paired against the golden
        # run: same chunks, same tokens, same model, so per-token deltas remain exactly paired.
        external = [r for r in _load(raw_root, baseline_name) if r["config"].get("quantization") is None]
        if external:
            baseline = external[0]
            runs = [baseline, *runs]
    rng = np.random.default_rng(0)
    rows = []
    for run in runs:
        recs = run["records"]
        df = pd.DataFrame(recs)
        kv_bits_per_token = df["kv_budget"].apply(lambda b: b["total_bits"]).mean() / df["tokens_retained"].mean()
        row: dict[str, Any] = {
            "variant": _variant_label(run["config"]),
            "exp_id": run["exp_id"],
            "chunks": len(df),
            "effective_bits_per_value": df["avg_kv_bits"].mean(),
            "kv_bits_per_token": kv_bits_per_token,
            "kv_mib_per_1k_tokens": kv_bits_per_token * 1000 / 8 / 2**20,
            "corpus_ppl": corpus_ppl(df["nll_sum"].tolist(), df["n_scored_tokens"].tolist()),
        }
        quant = recs[0].get("quantization")
        if quant:
            row["key_sqnr_db"] = quant["keys"]["sqnr_db_mean"]
            row["value_sqnr_db"] = quant["values"]["sqnr_db_mean"]
        if baseline is not None:
            mean, (lo, hi) = _paired_delta(baseline["records"], recs, rng)
            row["delta_nll_per_token"] = mean
            row["delta_nll_ci95_low"] = lo
            row["delta_nll_ci95_high"] = hi
            row["delta_ppl"] = row["corpus_ppl"] - corpus_ppl(
                pd.DataFrame(baseline["records"])["nll_sum"].tolist(),
                pd.DataFrame(baseline["records"])["n_scored_tokens"].tolist())
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(["effective_bits_per_value", "corpus_ppl"], ascending=[False, True])
    return table.reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--name", default="m1_kv_quant_ppl")
    parser.add_argument("--baseline-name", default="m0a_golden_ppl",
                        help="experiment whose full-precision arm is the pairing baseline, if the sweep has none")
    args = parser.parse_args()
    root = resolve_output_root(args.output_root)
    table = build_table(root / "raw", args.name, args.baseline_name)

    shown = ["variant", "effective_bits_per_value", "kv_mib_per_1k_tokens", "corpus_ppl", "delta_ppl",
             "delta_nll_per_token", "delta_nll_ci95_low", "delta_nll_ci95_high", "key_sqnr_db", "value_sqnr_db"]
    shown = [c for c in shown if c in table.columns]
    body = [
        "# M1: KV quantization vs cache-conditioned perplexity", "",
        "Paired per-token NLL deltas against the bf16 passthrough run on identical chunks; CI from a bootstrap over "
        "chunks. Memory is the recorded bit budget (codes + fp16 scales), not tensor sizes.", "",
        "```", table[shown].round(4).to_string(index=False), "```", "",
    ]
    base_row = table[table["variant"] == BASELINE_LABEL]
    if not base_row.empty:
        golden = _load(root / "raw", "m0a_golden_ppl")
        line = f"Passthrough control: corpus PPL {float(base_row['corpus_ppl'].iloc[0]):.6f}"
        if golden:
            g = pd.DataFrame(golden[0]["records"])
            golden_ppl = corpus_ppl(g["nll_sum"].tolist(), g["n_scored_tokens"].tolist())
            same = abs(golden_ppl - float(base_row["corpus_ppl"].iloc[0])) < 1e-9
            line += f"; M0 golden {golden_ppl:.6f} — {'identical' if same else 'DIFFERS, investigate'}."
        body.append(line)
    out = root / "reports" / f"{args.name}_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(body), encoding="utf-8")
    (root / "csv").mkdir(parents=True, exist_ok=True)
    table.to_csv(root / "csv" / f"{args.name}.csv", index=False, encoding="utf-8")
    print("\n".join(body))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
