# M0 exit checklist

M0 is closed only when every row is ✅ with evidence (file path, commit, value). Criteria from `00_research_plan.md` §G.

| # | Criterion | Status | Evidence / how to check |
|---|---|---|---|
| 1 | Two-phase path equals single pass on the real model: identical argmax; KL recorded (gate < 1e-4 only in float32) | CPU fp32 ✅ · GPU bf16 ⏳ | `tests/slow` on CPU fp32: argmax identical, max KL = 6.95e-10 (2026-09-17). GPU bf16: record the printed max KL here |
| 2 | Determinism: repeated reads identical; seeds 0/1/2 of the golden run give identical `token_nll` (seed does not enter cache_ppl) | CPU ✅ (same-process) · GPU ⏳ | `test_repeated_reads_are_bitwise_identical`; after the golden run compare `token_nll` across the three seed experiments (`*_ci95_half` of `corpus_ppl` must be 0) |
| 3 | Analytic KV bytes: equal to tensor nbytes (✅ tautological sanity) and within 2% of the CUDA allocator delta at 2K tokens | nbytes ✅ · allocator ⏳ | `test_kv_bytes_match_cuda_allocator` (GPU only) |
| 4 | Interrupted runner resumes without duplicate units | ✅ | `tests/test_runner_resume.py`; rerun of `m0a_cpu_smoke` skipped 3/3 units |
| 5 | Every record has provenance; CSV regenerable from raw | ✅ | `outputs/raw/qwen3-1p7b_wikitext2_cache_ppl_fp32_noecc_nofault_p0_cpu_s0-f96bb32e/manifest.json`, `scripts/aggregate.py` |
| 6 | Decode rate measured on an idle GPU (sizes the M2 fault campaign) | ⏳ | `m0a_latency` → `decode_tokens_per_s_mean` |
| 7 | RAQ-Probe golden accuracy (`correct`) in 85–100% per fact type × depth; answer NLL recorded | CPU pilot ⏳ · GPU ⏳ | `m0b_golden_probe` → `outputs/csv/m0b_*` rows `fact_type=…|depth_nominal=…` |
| 8 | GSM8K golden run completes; `####` extraction rate reported | CPU pilot ⏳ · GPU ⏳ | `m0b_golden_gsm8k` → `extraction_marker_rate` |

## GPU session (run only when `python scripts/check_env.py` reports no busy-GPU warning)

```bash
python scripts/check_env.py
RAQ_TEST_DEVICE=cuda python -m pytest -m slow tests/slow -s        # criteria 1-3; copy printed diagnostics above
python scripts/run_experiment.py --config configs/experiments/m0a_golden_ppl.yaml
python scripts/run_experiment.py --config configs/experiments/m0a_latency.yaml
python scripts/run_experiment.py --config configs/experiments/m0b_golden_probe.yaml
python scripts/run_experiment.py --config configs/experiments/m0b_golden_gsm8k.yaml
python scripts/aggregate.py --name m0 --filter _cuda_
```

Order matters only for convenience: criteria 1-3 first (minutes), then golden perplexity and latency (minutes), then
the probe (540 samples) and GSM8K (200 problems, up to 512 new tokens each; the longest step). Golden QA runs use a
single runtime seed because greedy decoding is deterministic; replicate variance comes from dataset resampling.

PowerShell: set the device with `$env:RAQ_TEST_DEVICE = "cuda"` before the pytest line.

## Open question for the golden run

`store_token_nll: true` keeps 512 floats per unit (40 units × 3 seeds ≈ 0.2 MB of JSONL). Kept on purpose: per-token
NLL is needed for paired per-token comparisons against quantized/faulted runs in M1-M3.
