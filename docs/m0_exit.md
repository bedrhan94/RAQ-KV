# M0 exit checklist

**Status: M0 CLOSED on 2026-09-19** — all eight criteria met on an idle RTX 5060 Ti (BF16, Qwen3-1.7B @ 70d244cc).
Full numbers: `outputs/reports/m0_golden_report.md`; raw records under `outputs/raw/`.

Golden baselines every later milestone measures against:

| Quantity | Value |
|---|---|
| WikiText-2 cache-conditioned perplexity (1536 cached + 512 scored, 40 chunks × 3 seeds) | **14.392** (identical in all seeds) |
| RAQ-Probe v0 accuracy (720 samples) | **0.994**; weakest cell 0.975 |
| RAQ-Probe decode-path margin on correct answers | median **19.25**, 10th pct 9.56 |
| GSM8K-200 accuracy (non-thinking) | **0.825** |
| KV cache | 112 KiB/token; 450 MiB at 4K tokens; analytic = measured |
| Latency (512/1024/2048-token prompts) | TTFT 89/218/601 ms; decode 39.6/38.2/32.7 tok/s |
| bf16 noise floor for cache-geometry changes (D10) | ~0.3 logits, KL ~1e-4 |

First observation from the golden probe (input to H1, to be tested properly in M2): the decode-path margin depends on
content type even after controlling for answer length — OLS `margin ~ n_tokens + type` gives a token-count coefficient
of −0.04 ± 0.68 (no effect) while entity is −9.3 ± 3.1 and tool_arg +6.7 ± 2.9 relative to constraint; at a fixed
8 answer tokens entity averages 11.1 versus 25.7 for other types. Invented names sit closest to a flip, tool-call
arguments furthest.

M0 is closed only when every row is ✅ with evidence (file path, commit, value). Criteria from `00_research_plan.md` §G.

| # | Criterion | Status | Evidence / how to check |
|---|---|---|---|
| 1 | Two-phase path equals single pass on the real model: identical argmax; KL recorded (gate < 1e-4 only in float32) | ✅ | CPU fp32: argmax identical, max KL 6.95e-10. **GPU bf16 (2026-09-19): argmax identical, max KL 3.18e-5, mean 1.38e-6** |
| 2 | Determinism: the same measurement repeated gives identical numbers | ✅ | Strongest evidence: **the two golden RAQ-Probe runs — separate processes, different code revisions — produced byte-identical teacher-forced NLL and margins for all 720 samples**, diverging only in 3 generations sitting on exact bf16 ties (D11). Also `test_repeated_reads_are_bitwise_identical` on GPU. *Note:* the 3-seed perplexity sweep agrees trivially — `cache_ppl` has no stochastic component, so seeds cannot change it; it is not evidence about kernel-level nondeterminism |
| 3 | Analytic KV bytes: equal to tensor nbytes and within 2% of the CUDA allocator delta at 2K tokens | ✅ | `test_kv_bytes_match_cuda_allocator` on GPU: allocator delta 234,881,024 B = analytic exactly (ratio 1.0000). In the golden runs the steady-state ratio is 1.0000; a run's first unit also carries one-off warm-up allocations (~1.19) |
| 4 | Interrupted runner resumes without duplicate units | ✅ | `tests/test_runner_resume.py`; rerun of `m0a_cpu_smoke` skipped 3/3 units |
| 5 | Every record has provenance; CSV regenerable from raw | ✅ | `outputs/raw/qwen3-1p7b_wikitext2_cache_ppl_fp32_noecc_nofault_p0_cpu_s0-f96bb32e/manifest.json`, `scripts/aggregate.py` |
| 6 | Decode rate measured on an idle GPU (sizes the M2 fault campaign) | ✅ | BF16, RTX 5060 Ti: TTFT 89/218/601 ms and 39.6/38.2/32.7 decode tok/s at 512/1024/2048 prompt tokens; prefill 6.0k/4.9k/3.5k tok/s. RAQ-Probe golden unit time 0.44/0.60/1.02/2.34 s at 512/1K/2K/4K |
| 7 | RAQ-Probe golden accuracy (`correct`) in 85–100% per fact type × depth; answer margin and float64 NLL recorded | ✅ | **GPU BF16, 720 samples (`m0b_golden_probe_margins`): accuracy 0.994, weakest cell 0.975 (constraint @ depth 0.5), strict EM 0.985, distractor mentioned 0.4%. Decode-path margin on correct answers: median 19.25, 10th pct 9.56, min 0.12; teacher-forced margin by type 10.0 (entity) to 26.5 (tool_arg). 3/720 generations hit an exact bf16 logit tie (D11).** Earlier CPU fp32 pilots: 512 tokens 36/36; 2048 tokens 17/18 (the failure chose the nearest later distractor); 4096 tokens 18/18 |
| 8 | GSM8K golden run completes; answer extraction rates reported | ✅ | **GPU BF16, 200 problems: accuracy 0.825**; extraction marker 173 / boxed 23 / last-number 4 (accuracy 0.83 / 0.87 / 0.25); 4 runs hit the 512-token cap; mean 250 generated tokens, 6.4 s per problem; cached problem text averages 62 tokens |

## Handover to M1

- **Passthrough regression fixture.** M1's BF16 passthrough path must reproduce the golden *records*, not merely a
  fresh run: assert that per-token NLL equals `outputs/raw/qwen3-1p7b_wikitext2_cache_ppl_bf16_..._s0-f7481787/
  results.jsonl` exactly, and that the probe reproduces `..._s0-a078288c`. This turns M0's output into M1's
  regression test and catches silent changes in the read path.
- **Always report next to accuracy:** truncation rate (`stopped_on_eos`), tie rate (`gen_margin_min_all == 0`), and
  the bf16 noise floor (D10). Degradation lengthens generations, so accuracy alone can hide a truncation artefact.
- **Superseded runs** carry `superseded_by` in their manifest and are excluded from summaries
  (`analysis.aggregate.mark_superseded`); raw records are never deleted.

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
