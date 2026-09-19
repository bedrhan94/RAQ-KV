"""M0's golden records are M1's regression fixture.

With no transform configured, the pipeline must reproduce the stored per-token NLL exactly. This catches any
silent change in the read path (cache handling, positions, scoring) introduced by later milestones, without
re-deriving the baseline. Skipped when the golden records or the GPU are unavailable.
"""

from __future__ import annotations

import pytest
import torch

from raq_kv.config.schema import ExperimentConfig
from raq_kv.experiments.tasks import build_task
from raq_kv.utils.io import read_json, read_jsonl
from raq_kv.utils.paths import resolve_output_root

pytestmark = [pytest.mark.slow, pytest.mark.gpu]

GOLDEN_PPL = "qwen3-1p7b_wikitext2_cache_ppl_bf16_noecc_nofault_p0_cuda_s0-f7481787"
GOLDEN_PROBE = "qwen3-1p7b_raq_probev0_qa_generate_bf16_noecc_nofault_p0_cuda_s0-a078288c"


def _golden(exp_id: str, n_units: int):
    exp = resolve_output_root("outputs") / "raw" / exp_id
    if not (exp / "results.jsonl").exists():
        pytest.skip(f"golden records missing: {exp_id}")
    if not torch.cuda.is_available():
        pytest.skip("golden records were produced on CUDA")
    cfg = ExperimentConfig.model_validate(read_json(exp / "manifest.json")["config"])
    records = {r["unit_key"]: r for r in read_jsonl(exp / "results.jsonl")}
    keys = sorted(records)[:n_units]
    task = build_task(cfg)
    task.setup()
    return task, records, keys


def test_perplexity_pipeline_reproduces_golden_records():
    task, records, keys = _golden(GOLDEN_PPL, 3)
    for key in keys:
        got = task.run_unit(key)
        want = records[key]
        assert got["token_nll"] == want["token_nll"], f"{key}: per-token NLL changed"
        assert got["nll_sum"] == pytest.approx(want["nll_sum"], rel=0, abs=1e-9)
        assert got["kv_bytes_analytic"] == want["kv_bytes_analytic"]


def test_probe_pipeline_reproduces_golden_records():
    task, records, keys = _golden(GOLDEN_PROBE, 4)
    for key in keys:
        got = task.run_unit(key)
        want = records[key]
        assert got["prediction"] == want["prediction"], f"{key}: generation changed"
        assert got["answer_margin_min"] == want["answer_margin_min"]
        assert got["answer_nll_mean"] == want["answer_nll_mean"]
        assert got["correct"] == want["correct"]
