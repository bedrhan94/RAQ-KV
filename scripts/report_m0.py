"""Build the M0 golden-baseline report from raw results.

Reads every completed experiment under ``outputs/raw`` whose name matches the M0 golden runs, checks the exit
criteria that can be checked from data, and writes a Markdown report to ``outputs/reports/m0_golden_report.md``.
Nothing is recomputed from the model: the report is a pure function of the raw records.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from raq_kv.analysis.aggregate import across_seeds, summarize_all
from raq_kv.evaluation.metrics import corpus_ppl
from raq_kv.utils.io import read_json, read_jsonl
from raq_kv.utils.paths import resolve_output_root


def _records(
    raw_root: Path, pattern: str, names: tuple[str, ...] | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Records plus manifests of experiments matching ``pattern``.

    ``names`` selects experiment configs by name in order of preference: the first name that has results wins, so a
    superseding rerun (e.g. the probe recorded with decode-path margins) is used instead of the earlier one rather
    than pooled with it.
    """
    by_name: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    for exp in sorted(raw_root.glob(pattern)):
        if not (exp / "manifest.json").exists():
            continue
        manifest = read_json(exp / "manifest.json")
        rows = read_jsonl(exp / "results.jsonl")
        if not rows or manifest.get("superseded_by"):
            continue
        records, manifests = by_name.setdefault(manifest["config"]["name"], ([], []))
        manifests.append(manifest)
        records.extend({**r, "seed": manifest["config"]["seed"]} for r in rows)
    if names:
        for name in names:
            if name in by_name:
                return by_name[name]
        return [], []
    all_records = [r for records, _ in by_name.values() for r in records]
    all_manifests = [m for _, manifests in by_name.values() for m in manifests]
    return all_records, all_manifests


def _fmt(x: float, digits: int = 3) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x:,.{digits}f}"


def _ppl_section(raw_root: Path) -> tuple[str, dict[str, Any]]:
    records, manifests = _records(raw_root, "*wikitext2_cache_ppl_bf16*cuda*")
    if not records:
        return "_no GPU perplexity run found_\n", {}
    df = pd.DataFrame(records)
    lines = ["| seed | chunks | corpus PPL | mean NLL | KV bytes (analytic) | allocator delta |",
             "|---|---|---|---|---|---|"]
    per_seed_nll = {}
    for seed, g in df.groupby("seed"):
        lines.append(
            f"| {seed} | {len(g)} | {_fmt(corpus_ppl(g['nll_sum'].tolist(), g['n_scored_tokens'].tolist()))} | "
            f"{_fmt(g['nll_sum'].sum() / g['n_scored_tokens'].sum(), 4)} | {int(g['kv_bytes_analytic'].iloc[0]):,} | "
            f"{int(pd.to_numeric(g['prefill_cuda_allocated_delta_bytes']).iloc[0]):,} |"
        )
        per_seed_nll[seed] = g.set_index("unit_key")["token_nll"].to_dict()
    seeds = sorted(per_seed_nll)
    identical = all(per_seed_nll[seeds[0]] == per_seed_nll[s] for s in seeds[1:]) if len(seeds) > 1 else None
    exact_kv = bool((df["kv_bytes_actual"] == df["kv_bytes_analytic"]).all())
    alloc = pd.to_numeric(df["prefill_cuda_allocated_delta_bytes"], errors="coerce")
    ratios = (alloc / df["kv_bytes_analytic"]).dropna()
    # The first measured unit of a process also pays one-off allocations (kernel workspaces, rotary buffers), so the
    # steady-state statistics are the meaningful ones; tests/slow measures a clean prefill separately.
    steady = ratios.drop(index=df.groupby("exp_id").head(1).index, errors="ignore")
    alloc_ratio = float(steady.max()) if len(steady) else float(ratios.max())
    facts = {
        "determinism_identical_token_nll": identical,
        "kv_analytic_equals_tensor_bytes": exact_kv,
        "allocator_ratio_median": float(ratios.median()),
        "allocator_ratio_max_steady_state": alloc_ratio,
        "allocator_ratio_max_including_first_unit": float(ratios.max()),
        "n_units": len(df),
        "corpus_ppl": corpus_ppl(df["nll_sum"].tolist(), df["n_scored_tokens"].tolist()),
        "git": sorted({r["provenance"]["git"]["describe"] for m in manifests for r in m["runs"]}),
    }
    lines += [
        "",
        f"- Identical per-token NLL across seeds: **{identical}** (greedy teacher forcing has no stochasticity)",
        f"- Analytic KV bytes == sum of tensor bytes: **{exact_kv}**",
        f"- Allocator delta / analytic KV bytes: median **{ratios.median():.4f}**, steady-state max "
        f"**{alloc_ratio:.4f}** (criterion: within 2%); including each run's first unit the max is "
        f"{ratios.max():.4f}, which carries one-off warm-up allocations",
    ]
    return "\n".join(lines) + "\n", facts


def _latency_section(raw_root: Path) -> tuple[str, dict[str, Any]]:
    records, _ = _records(raw_root, "*latency_bf16*cuda*")
    if not records:
        return "_no GPU latency run found_\n", {}
    df = pd.DataFrame(records)
    lines = ["| prompt tokens | runs | TTFT (ms) | TPOT (ms) | decode tok/s | prefill tok/s |",
             "|---|---|---|---|---|---|"]
    for n, g in df.groupby("prompt_tokens"):
        lines.append(
            f"| {int(n):,} | {len(g)} | {_fmt(g['ttft_s'].mean() * 1000, 1)} ± {_fmt(g['ttft_s'].std() * 1000, 1)} | "
            f"{_fmt(g['tpot_mean_s'].mean() * 1000, 2)} ± {_fmt(g['tpot_mean_s'].std() * 1000, 2)} | "
            f"{_fmt(g['decode_tokens_per_s'].mean(), 1)} | {_fmt(g['prefill_tokens_per_s'].mean(), 0)} |"
        )
    return "\n".join(lines) + "\n", {"decode_tokens_per_s": float(df["decode_tokens_per_s"].mean())}


def _probe_section(raw_root: Path) -> tuple[str, dict[str, Any]]:
    records, manifests = _records(raw_root, "*raq_probev0*bf16*cuda*",
                                  names=("m0b_golden_probe_margins", "m0b_golden_probe"))
    if not records:
        return "_no GPU probe run found_\n", {}
    df = pd.json_normalize(records, max_level=1)
    df["correct"] = df["correct"].astype(bool)
    run_name = manifests[0]["config"]["name"]
    cells = df.groupby(["meta.fact_type", "meta.depth_nominal"])["correct"].mean()
    worst_type, worst_depth = cells.idxmin()
    worst_cell = f"{worst_type} @ depth {float(worst_depth):g}", float(cells.min())
    lines = [
        f"Run `{run_name}`. Overall accuracy **{_fmt(df['correct'].mean())}** over {len(df)} samples "
        f"(strict EM {_fmt(df['em_strict'].mean())}, distractor mentioned {_fmt(df['mentions_distractor'].mean())}).",
        "",
        "Accuracy by fact type × nominal context length:",
        "",
        "```",
        df.pivot_table(index="meta.fact_type", columns="meta.nominal_len", values="correct").round(3).to_string(),
        "```",
        "",
        "Golden answer margin (min over value tokens, correct samples only) by fact type × length:",
        "",
        "```",
        df[df["correct"]].pivot_table(index="meta.fact_type", columns="meta.nominal_len",
                                      values="answer_margin_min").round(2).to_string(),
        "```",
        "",
        "Accuracy and margin by nominal depth:",
        "",
        "```",
        df.pivot_table(index="meta.depth_nominal", values=["correct", "answer_margin_min"]).round(3).to_string(),
        "```",
        "",
        f"- Weakest (fact type, depth) cell: **{worst_cell[0]}** at accuracy {worst_cell[1]:.3f} "
        f"(criterion 7: every cell in 0.85-1.00)",
        f"- Answer NLL (float64) median {df['answer_nll_mean'].median():.3e}, "
        f"max {df['answer_nll_mean'].max():.3e}",
    ]
    if "gen_margin_min_value" in df:
        gen = pd.to_numeric(df["gen_margin_min_value"], errors="coerce")
        ties = int((pd.to_numeric(df["gen_margin_min_all"], errors="coerce") == 0).sum())
        ok = df["correct"]
        lines += [
            f"- Exact logit ties during generation: {ties}/{len(df)} samples (bf16 logits are coarse enough for two "
            "tokens to tie; the chosen token then depends on the tie-break rule, not on the model)",
            f"- Decode-path margin on the answer value (correct samples): median {gen[ok].median():.2f}, "
            f"10th pct {gen[ok].quantile(0.1):.2f}, min {gen[ok].min():.2f}; located in "
            f"{gen.notna().mean():.1%} of samples",
            "- Teacher-forced margins can be negative on correct answers the model tokenizes differently; the "
            "decode-path margin is the tokenization-independent version (decisions D9).",
        ]
    facts = {
        "accuracy": float(df["correct"].mean()),
        "run_name": run_name,
        "min_cell_accuracy": worst_cell[1],
        "min_cell": worst_cell[0],
        "margin_min_q10_correct": float(df[df["correct"]]["answer_margin_min"].quantile(0.1)),
        "n_samples": len(df),
    }
    return "\n".join(lines) + "\n", facts


def _gsm8k_section(raw_root: Path) -> tuple[str, dict[str, Any]]:
    records, _ = _records(raw_root, "*gsm8k*bf16*cuda*")
    if not records:
        return "_no GPU GSM8K run found_\n", {}
    df = pd.DataFrame(records)
    rates = df["extraction_method"].value_counts(normalize=True).round(3).to_dict()
    truncated = int((~df["stopped_on_eos"].astype(bool)).sum())
    lines = [
        f"Accuracy **{_fmt(df['correct'].mean())}** over {len(df)} problems; "
        f"mean generated tokens {_fmt(df['generated_tokens'].mean(), 1)}; "
        f"stopped on EOS {_fmt(df['stopped_on_eos'].mean())}.",
        f"Answer extraction: {rates}.",
        f"**Truncated at the token cap: {truncated}** — these score as wrong, so any comparison that changes "
        "generation length must report truncation next to accuracy (degraded runs tend to ramble).",
    ]
    return "\n".join(lines) + "\n", {"accuracy": float(df["correct"].mean()), "extraction": rates,
                                     "truncated": truncated}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", default="outputs")
    args = parser.parse_args()
    root = resolve_output_root(args.output_root)
    raw = root / "raw"

    sections = [
        ("Cache-conditioned perplexity (WikiText-2)", _ppl_section(raw)),
        ("Serving latency (full precision)", _latency_section(raw)),
        ("RAQ-Probe v0", _probe_section(raw)),
        ("GSM8K-200", _gsm8k_section(raw)),
    ]
    facts: dict[str, Any] = {}
    body = ["# M0 golden baseline report", "",
            "Generated from raw records by `scripts/report_m0.py`; every number is a function of "
            "`outputs/raw/*/results.jsonl`.", ""]
    for title, (text, section_facts) in sections:
        body += [f"## {title}", "", text, ""]
        facts[title] = section_facts

    per_exp = summarize_all(raw)
    csv_dir = root / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    per_exp.to_csv(csv_dir / "m0_per_experiment.csv", index=False, encoding="utf-8")
    across_seeds(per_exp).to_csv(csv_dir / "m0_across_seeds.csv", index=False, encoding="utf-8")

    out = root / "reports" / "m0_golden_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(body), encoding="utf-8")
    print("\n".join(body))
    print(f"\nwrote {out} and {csv_dir / 'm0_per_experiment.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
