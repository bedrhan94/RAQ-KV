"""Raw JSONL -> summary CSV.

Two tables are produced:
* per-experiment: one row per (experiment, condition), e.g. one row per prompt length for latency;
* across-seeds:   rows sharing ``group_hash`` (identical config except seed) with mean, std, n and a 95%
  Student-t confidence half-width.
Summaries are always regenerated from raw records; nothing here writes into ``outputs/raw``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
from scipy import stats

from raq_kv.evaluation.metrics import corpus_ppl
from raq_kv.utils.io import read_json, read_jsonl


def _summarize_cache_ppl(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [{
        "condition": "all",
        "n_units": len(df),
        "corpus_ppl": corpus_ppl(df["nll_sum"].tolist(), df["n_scored_tokens"].tolist()),
        "nll_mean_tokenweighted": df["nll_sum"].sum() / df["n_scored_tokens"].sum(),
        "chunk_ppl_mean": df["ppl"].mean(),
        "chunk_ppl_std": df["ppl"].std(ddof=1) if len(df) > 1 else float("nan"),
        "tokens_retained_mean": df["tokens_retained"].mean(),
        "compression_ratio_mean": df["compression_ratio"].mean(),
        "kv_bytes_actual_mean": df["kv_bytes_actual"].mean(),
        "kv_bytes_analytic_mean": df["kv_bytes_analytic"].mean(),
        "kv_analytic_equals_actual": bool((df["kv_bytes_actual"] == df["kv_bytes_analytic"]).all()),
        "prefill_cuda_allocated_delta_mean": pd.to_numeric(df["prefill_cuda_allocated_delta_bytes"]).mean(),
    }]


def _summarize_latency(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for prompt_tokens, g in df.groupby("prompt_tokens", sort=True):
        row: dict[str, Any] = {"condition": f"prompt_tokens={prompt_tokens}", "n_units": len(g)}
        for col in ("ttft_s", "tpot_mean_s", "e2e_s", "prefill_tokens_per_s", "decode_tokens_per_s"):
            row[f"{col}_mean"] = g[col].mean()
            row[f"{col}_std"] = g[col].std(ddof=1) if len(g) > 1 else float("nan")
        rows.append(row)
    return rows


def _qa_row(condition: str, g: pd.DataFrame) -> dict[str, Any]:
    row: dict[str, Any] = {
        "condition": condition,
        "n_units": len(g),
        "accuracy": g["correct"].mean(),
        "em_strict": g["em_strict"].mean(),
        "generated_tokens_mean": g["generated_tokens"].mean(),
        "stopped_on_eos_rate": g["stopped_on_eos"].mean(),
        "tokens_retained_mean": g["tokens_retained"].mean(),
    }
    optional = {
        "answer_margin_min_mean": "answer_margin_min",
        "answer_nll_mean": "answer_nll_mean",
        "answer_first_token_nll": "answer_first_token_nll",
        "mentions_distractor_rate": "mentions_distractor",
    }
    for out_col, col in optional.items():
        if col in g:
            row[out_col] = pd.to_numeric(g[col], errors="coerce").mean()
    if "extraction_method" in g:
        row["extraction_marker_rate"] = (g["extraction_method"] == "marker").mean()
    return row


def _summarize_qa(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = [_qa_row("all", df)]
    for col in ("meta.fact_type", "meta.nominal_len", "meta.depth_nominal"):
        if col in df:
            for value, g in df.groupby(col, sort=True):
                rows.append(_qa_row(f"{col[5:]}={value}", g))
    if "meta.fact_type" in df and "meta.depth_nominal" in df:  # the M0 golden-band cells (type x depth)
        for (ftype, depth), g in df.groupby(["meta.fact_type", "meta.depth_nominal"], sort=True):
            rows.append(_qa_row(f"fact_type={ftype}|depth_nominal={depth}", g))
    return rows


_SUMMARIZERS = {"cache_ppl": _summarize_cache_ppl, "latency": _summarize_latency, "qa_generate": _summarize_qa}


def summarize_experiment(exp_dir: Path) -> list[dict[str, Any]]:
    manifest = read_json(exp_dir / "manifest.json")
    records = read_jsonl(exp_dir / "results.jsonl")
    if not records:
        return []
    cfg = manifest["config"]
    base = {
        "exp_id": manifest["exp_id"],
        "group_hash": manifest["group_hash"],
        "name": cfg["name"],
        "model": cfg["model"]["short_name"],
        "dtype": cfg["model"]["dtype"],
        "device": cfg["model"]["device"],
        "dataset": cfg["dataset"]["name"],
        "task": cfg["task"]["kind"],
        "seed": cfg["seed"],
        "units_complete": len(records) == manifest["unit_keys_total"],
        "git_commits": ";".join(sorted({str(r["provenance"]["git"]["describe"]) for r in manifest["runs"]})),
    }
    df = pd.json_normalize(records, max_level=1)  # nested dicts (meta, kv_budget) become "meta.x" columns
    return [{**base, **row} for row in _SUMMARIZERS[cfg["task"]["kind"]](df)]


def summarize_all(raw_root: Path, name_filter: str | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for exp_dir in sorted(p for p in Path(raw_root).iterdir() if (p / "manifest.json").exists()):
        if name_filter and name_filter not in exp_dir.name:
            continue
        rows.extend(summarize_experiment(exp_dir))
    return pd.DataFrame(rows)


def across_seeds(per_exp: pd.DataFrame, confidence: float = 0.95) -> pd.DataFrame:
    """Replicate statistics over seeds for every numeric column, grouped by (group_hash, condition)."""
    if per_exp.empty:
        return per_exp
    id_cols = ["group_hash", "condition", "name", "model", "dtype", "device", "dataset", "task"]
    numeric = [c for c in per_exp.select_dtypes("number").columns if c not in ("seed", "n_units")]
    rows = []
    for keys, g in per_exp.groupby(id_cols, sort=True, dropna=False):
        row: dict[str, Any] = dict(zip(id_cols, keys, strict=True))
        row["n_seeds"] = g["seed"].nunique()
        row["all_units_complete"] = bool(g["units_complete"].all())
        for col in numeric:
            values = g[col].dropna()
            n = len(values)
            mean = values.mean() if n else float("nan")
            std = values.std(ddof=1) if n > 1 else float("nan")
            half = stats.t.ppf(0.5 + confidence / 2, n - 1) * std / math.sqrt(n) if n > 1 else float("nan")
            row[f"{col}__mean"], row[f"{col}__std"], row[f"{col}__ci95_half"] = mean, std, half
        rows.append(row)
    return pd.DataFrame(rows)
