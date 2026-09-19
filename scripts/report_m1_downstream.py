"""M1 downstream report: what the KV operating points cost on tasks, next to what perplexity said they cost.

Perplexity is measured on the cached prefix of WikiText; the probe measures whether a fact survives in the cache
well enough to be retrieved. Putting the two side by side tests this project's own premise — that token-level
metrics understate task damage — before any of it is claimed in M5/M6.

Guards: superseded runs are skipped, and so are incomplete ones (a partial run silently compared on whatever
samples it happens to contain is how a stopped job turns into a wrong number).
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


def load_runs(raw_root: Path, name: str, require_complete: bool = True) -> list[tuple[dict[str, Any], pd.DataFrame]]:
    runs, skipped = [], []
    for exp in sorted(raw_root.glob("*")):
        manifest_path = exp / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = read_json(manifest_path)
        if manifest["config"]["name"] != name or manifest.get("superseded_by"):
            continue
        records = read_jsonl(exp / "results.jsonl")
        if not records:
            continue
        if require_complete and len(records) != manifest["unit_keys_total"]:
            skipped.append(f"{manifest['exp_id']} ({len(records)}/{manifest['unit_keys_total']} units)")
            continue
        runs.append((manifest["config"], pd.json_normalize(records, max_level=1)))
    for s in skipped:
        print(f"  [skipped incomplete] {s}")
    return runs


def op_label(cfg: dict[str, Any]) -> str:
    q = cfg.get("quantization")
    if not q:
        return "bf16"
    k, v = q["keys"], q["values"]
    axis = f"blk{k['group_size']}" if k["axis"] == "group_token" else k["axis"]
    return f"K{k['bits']}b/{axis} · V{v['bits']}b/token"


def _boot_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    draws = rng.choice(values, size=(4000, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def ppl_by_operating_point(raw_root: Path, names: tuple[str, ...]) -> dict[str, float]:
    """Corpus perplexity of each operating point from the WikiText sweeps, keyed by the same label."""
    out: dict[str, float] = {}
    for name in names:
        for cfg, df in load_runs(raw_root, name):
            out[op_label(cfg)] = corpus_ppl(df["nll_sum"].tolist(), df["n_scored_tokens"].tolist())
    return out


def compare(golden: pd.DataFrame, run: pd.DataFrame, rng: np.random.Generator, with_margin: bool) -> dict[str, Any]:
    g = golden.set_index("sample_id").sort_index()
    q = run.set_index("sample_id").sort_index()
    common = g.index.intersection(q.index)
    g, q = g.loc[common], q.loc[common]
    gc, qc = g["correct"].astype(bool), q["correct"].astype(bool)
    row: dict[str, Any] = {
        "n": len(common),
        "effective_bits": float(q["avg_kv_bits"].mean()),
        "accuracy": float(qc.mean()),
        "golden_accuracy": float(gc.mean()),
        "correct_to_wrong": int((gc & ~qc).sum()),
        "wrong_to_correct": int((~gc & qc).sum()),
        "truncated": int((~q["stopped_on_eos"].astype(bool)).sum()),
    }
    if with_margin and "gen_margin_min_value" in q:
        delta = (q["gen_margin_min_value"] - g["gen_margin_min_value"]).dropna().to_numpy()
        lo, hi = _boot_ci(delta, rng)
        row |= {"margin_delta": float(delta.mean()), "margin_ci_low": lo, "margin_ci_high": hi}
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", default="outputs")
    args = parser.parse_args()
    root = resolve_output_root(args.output_root)
    raw = root / "raw"
    rng = np.random.default_rng(0)

    ppl = ppl_by_operating_point(raw, ("m1_kv_quant_ppl", "m1_causality_ppl", "m1_causality_isobit"))
    golden_ppl = ppl.get("bf16", float("nan"))
    body = ["# M1 downstream: operating points on tasks vs on perplexity", "", __doc__.split("\n\n", 1)[1], ""]

    for task, golden_name, with_margin in (("m1_downstream_probe", "m0b_golden_probe_margins", True),
                                           ("m1_downstream_gsm8k", "m0b_golden_gsm8k", False)):
        golden_runs = load_runs(raw, golden_name)
        runs = load_runs(raw, task)
        if not golden_runs or not runs:
            body += [f"## {task}", "", "_no complete runs_", ""]
            continue
        golden = golden_runs[0][1]
        rows = []
        for cfg, df in runs:
            row = {"operating point": op_label(cfg), **compare(golden, df, rng, with_margin)}
            point_ppl = ppl.get(row["operating point"])
            row["wikitext_ppl"] = point_ppl
            row["ppl_increase_pct"] = (100 * (point_ppl - golden_ppl) / golden_ppl) if point_ppl else None
            row["accuracy_drop_pct"] = 100 * (row["golden_accuracy"] - row["accuracy"]) / row["golden_accuracy"]
            rows.append(row)
        table = pd.DataFrame(rows).sort_values("effective_bits", ascending=False)
        cols = ["operating point", "effective_bits", "wikitext_ppl", "ppl_increase_pct", "accuracy",
                "accuracy_drop_pct", "correct_to_wrong", "wrong_to_correct", "truncated"]
        if with_margin:
            cols += ["margin_delta", "margin_ci_low", "margin_ci_high"]
        body += [f"## {task} (golden accuracy {golden['correct'].mean():.4f}, n={len(golden)})", "", "```",
                 table[cols].round(4).to_string(index=False), "```", ""]
        print(table[cols].round(4).to_string(index=False))

        if with_margin:
            detail = []
            for cfg, df in runs:
                d = df.set_index("sample_id").sort_index()
                by_type = d.groupby("meta.fact_type")["correct"].mean().round(3).to_dict()
                by_len = d.groupby("meta.nominal_len")["correct"].mean().round(3).to_dict()
                detail += [f"**{op_label(cfg)}** — by fact type: {by_type}", "", f"by context length: {by_len}", ""]
            body += ["### Probe accuracy breakdown", "", *detail]

    out = root / "reports" / "m1_downstream_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(body), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
