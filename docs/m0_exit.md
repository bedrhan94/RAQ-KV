# M0 exit checklist

M0 is closed only when every row is ✅ with evidence (file path, commit, value). Criteria from `00_research_plan.md` §G.

| # | Criterion | Status | Evidence / how to check |
|---|---|---|---|
| 1 | Two-phase path equals single pass on the real model: identical argmax; KL recorded (gate < 1e-4 only in float32) | ✅ | CPU fp32: argmax identical, max KL 6.95e-10. **GPU bf16 (2026-09-19): argmax identical, max KL 3.18e-5, mean 1.38e-6** |
| 2 | Determinism: repeated reads identical; seeds 0/1/2 of the golden run give identical `token_nll` | ✅ | `test_repeated_reads_are_bitwise_identical` passes on GPU; **the three golden perplexity seeds produced identical per-token NLL (corpus PPL 14.392 in all three)** |
| 3 | Analytic KV bytes: equal to tensor nbytes and within 2% of the CUDA allocator delta at 2K tokens | ✅ | `test_kv_bytes_match_cuda_allocator` on GPU: allocator delta 234,881,024 B = analytic exactly (ratio 1.0000). In the golden runs the steady-state ratio is 1.0000; a run's first unit also carries one-off warm-up allocations (~1.19) |
| 4 | Interrupted runner resumes without duplicate units | ✅ | `tests/test_runner_resume.py`; rerun of `m0a_cpu_smoke` skipped 3/3 units |
| 5 | Every record has provenance; CSV regenerable from raw | ✅ | `outputs/raw/qwen3-1p7b_wikitext2_cache_ppl_fp32_noecc_nofault_p0_cpu_s0-f96bb32e/manifest.json`, `scripts/aggregate.py` |
| 6 | Decode rate measured on an idle GPU (sizes the M2 fault campaign) | ✅ | BF16, RTX 5060 Ti: TTFT 89/218/601 ms and 39.6/38.2/32.7 decode tok/s at 512/1024/2048 prompt tokens; prefill 6.0k/4.9k/3.5k tok/s. RAQ-Probe golden unit time 0.44/0.60/1.02/2.34 s at 512/1K/2K/4K |
| 7 | RAQ-Probe golden accuracy (`correct`) in 85–100% per fact type × depth; answer margin and float64 NLL recorded | CPU pilots ✅ · GPU ⏳ | CPU fp32 pilots: 512 tokens 36/36 correct (commit 33be27a, margins not yet recorded); 2048 tokens 17/18 correct, golden `answer_margin_min` 9.9–27.5, the one failure (constraint, depth 0.9) chose the nearest later distractor (margin −6.3) (commit aa7cd7b); 4096 tokens 18/18 correct, margins 1.4–28.5 (commit cf9a47c). GPU (720 samples): `m0b_golden_probe` rows `fact_type=…|depth_nominal=…` |
| 8 | GSM8K golden run completes; answer extraction rates reported | CPU pilot ✅ · GPU ⏳ | CPU pilot 4 problems: 3 correct; extraction marker 2, boxed-only 1 (boxed extraction added afterwards), 145–324 new tokens. GPU: `m0b_golden_gsm8k` → `extraction_*_rate` |

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
