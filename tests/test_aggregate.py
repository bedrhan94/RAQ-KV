from __future__ import annotations

import math
from pathlib import Path

from raq_kv.analysis.aggregate import across_seeds, summarize_experiment
from raq_kv.utils.io import JsonlAppender, write_json_atomic


def _write_exp(tmp_path: Path, records: list[dict], task: str = "qa_generate", seed: int = 0) -> Path:
    exp = tmp_path / f"exp-s{seed}"
    write_json_atomic(exp / "manifest.json", {
        "exp_id": exp.name, "group_hash": "g", "unit_keys_total": len(records),
        "config": {"name": "t", "seed": seed, "model": {"short_name": "m", "dtype": "bfloat16", "device": "cuda"},
                   "dataset": {"name": "raq_probe"}, "task": {"kind": task}},
        "runs": [{"provenance": {"git": {"describe": "abc"}}}],
    })
    with JsonlAppender(exp / "results.jsonl") as w:
        for r in records:
            w.append(r)
    return exp


def _qa(correct: bool, margin: float, fact_type: str, depth: float) -> dict:
    return {"correct": correct, "em_strict": correct, "generated_tokens": 5, "stopped_on_eos": True,
            "tokens_retained": 500, "answer_margin_min": margin, "answer_nll_mean": 0.0,
            "mentions_distractor": not correct, "meta": {"fact_type": fact_type, "depth_nominal": depth}}


def test_qa_summary_separates_margins_by_correctness(tmp_path: Path):
    records = [_qa(True, 20.0, "id", 0.1), _qa(True, 10.0, "id", 0.9), _qa(False, -6.0, "id", 0.9)]
    rows = {r["condition"]: r for r in summarize_experiment(_write_exp(tmp_path, records))}
    overall = rows["all"]
    assert math.isclose(overall["accuracy"], 2 / 3)
    assert overall["answer_margin_min_mean_if_correct"] == 15.0
    assert overall["answer_margin_min_mean_if_incorrect"] == -6.0 and overall["n_incorrect"] == 1
    assert rows["fact_type=id|depth_nominal=0.9"]["n_units"] == 2


def test_superseded_runs_are_excluded(tmp_path: Path):
    from raq_kv.analysis.aggregate import mark_superseded, summarize_all

    old = _write_exp(tmp_path, [_qa(True, 10.0, "id", 0.1)], seed=0)
    _write_exp(tmp_path, [_qa(True, 12.0, "id", 0.1)], seed=1)
    mark_superseded(old, "exp-s1", "rerun with decode-path margins")
    assert summarize_experiment(old) == []
    assert set(summarize_all(tmp_path)["exp_id"]) == {"exp-s1"}
    assert set(summarize_all(tmp_path, include_superseded=True)["exp_id"]) == {"exp-s0", "exp-s1"}


def test_across_seeds_ci(tmp_path: Path):
    import pandas as pd

    rows = []
    for seed, margin in ((0, 10.0), (1, 12.0), (2, 14.0)):
        rows += summarize_experiment(_write_exp(tmp_path, [_qa(True, margin, "id", 0.1)], seed=seed))
    agg = across_seeds(pd.DataFrame(rows))
    overall = agg[agg["condition"] == "all"].iloc[0]
    assert overall["n_seeds"] == 3
    assert overall["answer_margin_min_mean_if_correct__mean"] == 12.0
    assert overall["answer_margin_min_mean_if_correct__ci95_half"] > 0
