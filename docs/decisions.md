# Decision log

Each entry: context, decision, evidence, consequences. Newest last.

## D1 — Primary model: Qwen3-1.7B @ `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` (2026-09-17)

- **Status:** explicitly approved by the project owner on 2026-09-17.

- **Why:** 8 KV heads (usable head axis; Qwen2.5-1.5B has 2), uniform full attention (no sliding-window layers),
  native tool-calling template and a verified non-thinking switch (so M6 does not force a model change), ungated.
- **Evidence:** HF config fields read on 2026-09-17; chat template probe (`<tool_call>{json}</tool_call>`).
- **Consequences:** Qwen3 uses QK-Norm, so key-outlier behaviour may differ from Llama; a second family
  (Llama-3.2, access granted 2026-09-17) and Qwen3-4B are required for generalization claims.
- **Addendum (2026-09-17):** Llama-3.2-3B-Instruct (rev `0cb88a4f764b7a12671c53f0838cd831a0843b95`) has exactly the
  same KV geometry as Qwen3-1.7B (28 layers × 8 KV heads × head_dim 128, 112 KiB/token in BF16). It is the preferred
  cross-family replicate: family effects can be separated from cache geometry. Llama-3.2-1B-Instruct
  (rev `9213176726f574b556790deb65791e0c5aa438b6`): 16 × 8 × 64.

## D2 — transformers pinned to 5.17.0 (2026-09-17)

- **Why:** the project depends on cache internals, which changed substantially between 4.x and 5.x.
- **Evidence:** `DynamicCache.layers[i].keys/.values` of shape `[B, H_kv, T, d_head]`; `DynamicCache(ddp_cache_data=...)`
  rebuilds a cache by concatenation (no aliasing); `Qwen3Model.forward` has no `cache_position` argument.
- **Consequences:** `tests/test_protocol_equivalence.py` and `tests/test_kv_snapshot.py` are the API probe; run them
  before any upgrade.

## D3 — Two-phase measurement protocol (2026-09-17)

- **Why:** in a single forward pass K/V are produced and consumed together, so cache transforms are invisible.
- **Decision:** prefill -> snapshot -> transform -> read through the cache (see conventions §3).
- **Consequences:** WikiText numbers are "cache-conditioned perplexity" and are not number-for-number comparable
  with published sliding-window perplexities; only trends are compared.

## D4 — Only `position_ids` is controlled; the mask follows the physical cache (2026-09-17)

- **Context:** an early probe passed `cache_position`, which transformers 5.17 silently swallowed via `**kwargs`.
- **Evidence:** masking source (`get_query_offset() == get_seq_length()`); tiny-model probe: middle pruning +
  5-token chunk + 3 greedy steps equals masked full cache (max |Δ| = 4.2e-7, sdpa and eager); omitting
  `position_ids` changes logits by 0.30.
- **Consequences:** `model.generate()` is banned on modified caches; our own greedy loop is used everywhere.

## D5 — `sdpa` does not return attention weights (2026-09-17)

- **Evidence:** `output_attentions=True` under `sdpa` logs a warning and returns an empty list.
- **Consequences:** attention-based importance (M4) computes Q·K explicitly over an observation window and asserts
  non-empty results; eager attention over long contexts is avoided (O(T²) memory).

## D6 — Memory is reported analytically; quantization/ECC latency is not measured in Python (2026-09-17)

- **Why:** fake-quantized tensors do not shrink GPU memory, and Python bit manipulation is slower than the baseline.
- **Decision:** KV memory comes from `budget/bit_accounting.py`, validated at full precision against real tensor bytes
  (exact) and CUDA allocator deltas (M0 exit criterion). Latency is measured only for real execution paths
  (full-precision baseline, token pruning).

## D7 — Experiment infrastructure choices (2026-09-17)

- pydantic v2 with `extra="forbid"` + PyYAML (includes, dotted overrides, sweeps); no Hydra.
- Resume keyed by `unit_id`; the code state (commit, `git diff HEAD` hash) must match previous runs unless
  `--allow-code-change`.
- A crash-truncated JSONL line is kept on disk and followed by a marker record on resume (raw data is never edited).
- GPU state is read with `nvidia-smi`, so provenance collection never creates a CUDA context on a busy GPU.

## D9 — RAQ-Probe v0 design (2026-09-17)

- **Primary metric = minimum gold-token logit margin over the answer value** (`answer_margin_min`:
  logit(gold) − max logit(other), teacher-forced through the cache; for `tool_arg` only the argument value inside the
  canonical `<tool_call>`). It is the distance to an argmax flip, so it links directly to SDC (flip ⇔ margin < 0).
  Secondary: float64 answer NLL, then EM (`correct`, `em_strict`). The golden EM band (85-100%) remains an M0 sanity
  criterion.
  - *Why not NLL as primary (revised 2026-09-17):* the 512-token CPU pilot (36 samples, commit 33be27a) had 100%
    accuracy and float32 answer NLL of exactly 0.0 (median; max 2.6e-5): confident log-probabilities round to 0 in
    float32, so NLL sits on its floor at golden and hides small degradations. NLL is now computed in float64, and the
    non-saturating margin is primary. EM alone is at ceiling at 512 tokens, as expected for a 1.7B model on
    single-needle retrieval with 2 distractors.
- **Distractors** (same type and template, different invented keys/values) are critical-looking but task-irrelevant
  content: they separate content type from task relevance in Q49.
- **number vs constraint** share the surface type (integers above 2100, not year-like) and differ in role.
- **Paired depth variants:** filler text and distractor slots are identical across depths; target slots are chosen by
  filler-token fraction. Observed actual depth ranges for nominal 0.1/0.5/0.9: 0.06-0.21 / 0.39-0.62 / 0.80-0.98
  (spread from long WikiText sentences in 512-token contexts); `meta.depth_actual` is recorded for use as a covariate.
- **Filler:** WikiText-2 raw train split with tokenization artefacts cleaned; keys/values must not occur in the filler.
- **Correctness rules:** listing a distractor value makes an answer incorrect; numbers must match token-bounded.
- Full set: 540 samples, built in ~4 s; nominal 512/1024/2048 give mean cache lengths 533/1044/2066 tokens.
- **2K CPU pilot (18 samples, commit aa7cd7b):** 17/18 correct; golden `answer_margin_min` 9.9–27.5 on correct
  answers. The failure (constraint, nominal depth 0.9) returned the distractor placed 3 sentences after the target
  (margin −6.3, `mentions_distractor=True`): distractors produce the intended retrieval confusion at long context.
  At 512 tokens accuracy is at ceiling (36/36), so length is the main difficulty knob; no redesign before the GPU run.

## D8 — M0 split and GPU-free validation (2026-09-17)

- **Context:** the only GPU was busy (10.8/16.3 GiB, 98% utilization) when implementation started.
- **Decision:** M0a code is validated on CPU with the real model in float32 (`configs/experiments/m0a_cpu_smoke.yaml`,
  `tests/slow`). The golden BF16 baseline and latency runs wait for an idle GPU. CPU numbers are never used as the
  baseline (different device and dtype yield a different `exp_id`).
- **Deviation from plan §H:** no separate `scripts/probe_cache_api.py`; the protocol and snapshot tests serve that role.
