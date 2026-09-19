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

## D15 — Causal token-block key scales replace offline per-channel scales (2026-09-19)

`per_channel` computes one scale per (head, channel) over **every token in the context**, so a streaming decoder
cannot produce it; it is a prefill-time scheme. `group_token` closes a block of tokens at a time and is causal
(asserted by `test_group_token_is_causal_and_per_channel_is_not`). The question is what causality costs.

At matched budget (40 WikiText-2 chunks, values 2-bit/4-bit per-token in both arms):

| Budget | Offline per-channel | Causal block-512 | Paired difference (offline − causal) |
|---|---|---|---|
| ~4.15 bits/value | PPL 14.5301 | PPL 14.4991 | +0.0021 nats [−0.0020, +0.0063] — **tie** |
| ~2.15 bits/value | PPL 23.0484 | PPL 18.5788 | **+0.2156 nats [+0.1871, +0.2447]**, offline worse on 40/40 chunks |

- **Causality is not a cost here; at low precision it is a gain.** A context-wide scale must span a channel's full
  dynamic range across all positions, while a block scale adapts to local magnitude.
- Smaller blocks buy more quality for more metadata: at 2 bits, block 512 / 256 / 128 / 32 give PPL 18.58 / 17.31 /
  16.56 / 15.47 at 2.156 / 2.188 / 2.250 / 2.625 effective bits. The token-block family is the Pareto frontier;
  offline per-channel sits below it.
- **Revises D12's recommendation.** The rule is not "keys per channel" but "key scales must be per channel *within
  a token block*"; a single per-channel scale over the whole context is both non-causal and (at 2 bits) worse.
  D12's other findings stand: per-token key scales are catastrophic, and grouping *channels* within a token does
  not fix the outlier structure.
- **Operating points from here on** (all causal): 4-bit block-128 (4.25 bits, ΔNLL 0.0025, CI includes 0),
  2-bit block-32 (2.625 bits, ΔNLL 0.072), 2-bit block-512 (2.156 bits, ΔNLL 0.255).
- Open: the sensitivity profile (D13/D14) was measured with the offline per-channel stress. Its keys-vs-values and
  layer structure should be re-checked under a block scheme before M5 relies on it.

## D14 — Q48 is computed at layer×part; head level stays exploratory (2026-09-19)

Head-level profile (448 regions = 28 layers × 8 KV heads × {keys, values}, 40 chunks, 640 null measurements):

| | layer×part (56) | head×part (448) |
|---|---|---|
| Regions differing from null (Welch, Holm p<0.05) | **24 (43%)** | **37 (8%)** |
| Regions clearing the conservative band | 18 | 10 |
| Split-half rank stability | 0.901 (SB 0.948) | 0.725 (SB 0.841) |
| Largest region damage | 0.0291 nats | 0.0181 nats |

- A single KV head is 1/8 of a part, and at deployment precision its quantization damage mostly falls inside the
  noise band: only 8% of head regions are distinguishable from lossless re-quantization. **Q48 (QS vs fault
  sensitivity) is therefore computed at layer×part**, where 43% of regions carry measurable signal and the ranking
  is stable (0.90). The head-level map is kept as an exploratory layer, not as a correlation input.
- The head-level concentration statistic (top 10% carry 58%) must not be read as support for H1: when most cells
  are noise, top-k share is inflated by the noise tail. Concentration is only meaningful over cells that clear the
  null, which is why the Welch/Holm count is the headline number.
- **Statistical convention fixed here and inherited by M2:** a region counts as sensitive when its measurements
  differ from the null distribution by a Welch test with Holm correction across regions — not when its mean clears
  the spread of individual null measurements (that comparison mixes a mean with single-measurement scatter and is
  over-conservative). Both numbers are reported.

## D13 — Quantization sensitivity is real but modest at deployment precision (2026-09-19)

Profile: one region (layer × keys/values) at a time forced to 2 bits with the deployment granularity
(keys per-channel, values per-token), 40 WikiText-2 chunks, 56 regions, plus 320 null measurements at a lossless
8 bits that define the measurement's noise band.

| Quantity | Deployment granularity (2-bit) | Mismatched granularity (keys per-token, positive control) |
|---|---|---|
| Region damage range | −0.0211 … +0.0291 nats | −0.0247 … +7.18 nats |
| Noise band (null, 95%) | [−0.0048, +0.0038] | not measured |
| Regions clearing the band | 18 / 56 | — (all large effects) |
| Top 10% share of damage | 40.2% | 76.6% |
| Split-half rank stability | 0.901 (Spearman-Brown 0.948) | 0.929 (0.963) |
| ANOVA across regions | F = 18.1, p ≈ 1e-138 | F = 573 |
| Keys vs values (mean) | 0.0085 vs 0.0008 nats (11×) | 0.458 vs 0.0007 nats (640×) |

- **Power matters more than it looks.** The same profile on 8 chunks gave no stable ranking (chunk-to-chunk
  Spearman −0.05); at 40 chunks split-half stability is 0.90. Region-level claims need tens of chunks.
- **Against the pre-registered H1 thresholds** (top 10% ≥ 50% supports, ≤ 20% refutes): 40.2% is *intermediate*.
  The honest statement is that sensitivity is heterogeneous and stably ranked, but not extremely concentrated at
  deployment precision. Thresholds stay as registered; the result is reported as partial.
- **K/V asymmetry is the robust part:** keys dominate under both granularities (11× and 640×). Allocation should
  split by part first and by layer second.
- **Some regions clear the band in the negative direction** (min −0.0211 vs band −0.0048): quantizing them
  slightly *improves* NLL. Named, so M2 can check them rather than rediscover them: **layer 15 values
  (−0.0211 ± 0.0033), layer 19 keys (−0.0158 ± 0.0054), layer 15 keys (−0.0104 ± 0.0055)** — layer 15 appears on
  both parts. Treated as a regularization-like fluke for now; if the *same* regions come out protective under
  fault injection, "damage" as a metric needs rethinking.
- The 18 regions above the band are 13 keys and 5 values, spread over layers 4-27 with no monotone depth trend.
- The mismatched-granularity profile stays in the record as a **positive control**: it shows the profiler detects
  heterogeneity when it is present, which is what makes the modest deployment-precision numbers meaningful.

## D12 — KV precision/granularity: per-channel keys, per-token values (2026-09-19)

M1 sweep, 13 variants × 40 WikiText-2 chunks, paired per-token NLL against the bf16 passthrough (which reproduces
the M0 golden corpus perplexity exactly: 14.391535 both).

| Variant | eff. bits/value | KV MiB / 1K tok | corpus PPL | paired ΔNLL/token [95% CI] |
|---|---|---|---|---|
| bf16 passthrough | 16.00 | 109.4 | 14.3915 | 0 |
| K 8b/chan · V 8b/token | 8.14 | 55.6 | 14.3910 | −0.0000 [−0.0009, 0.0008] |
| K 4b/chan · V 4b/token | 4.14 | 28.3 | 14.5301 | +0.0096 [0.0047, 0.0142] |
| K 4b/g32 · V 4b/g32 | 5.00 | 34.2 | 15.4895 | +0.0735 |
| K 4b/token · V 4b/token | 4.25 | 29.1 | **3226.8** | +5.41 |
| K 2b/chan · V 2b/token | 2.14 | 14.6 | 23.0484 | +0.4710 |
| K 2b/g32 · V 2b/g32 | 3.00 | 20.5 | **10782.9** | +6.62 |
| K 2b/token · V 2b/token | 2.25 | 15.4 | **639252.7** | +10.70 |

- **Keys must be quantized per channel.** Per-token key scales are catastrophic even at 4 bits (PPL 3227), because
  key channels carry persistent outliers; per-channel scales are nearly lossless at the same budget. Grouping
  channels *within* a token (`group_channel`) does not fix it (2-bit g32: PPL 10783): the outlier structure lives
  along the channel axis across tokens. This reproduces KIVI's observation on Qwen3, which was not guaranteed —
  Qwen3 applies QK-Norm, so key statistics differ from the Llama models the original observation came from.
- **Per-channel keys are also cheaper in metadata** at long context: one scale per (head, channel) instead of per
  (head, token), so 4-bit K/V costs 4.14 effective bits per value, not 4.25.
- **Defaults from here on:** keys per-channel, values per-token. 8-bit is lossless (CI includes 0), 4-bit costs
  +0.14 PPL at 3.9× less KV memory, 2-bit is degraded but functional (PPL 23.0) — the interesting operating point
  for M5's mixed-precision allocation.
  - **Superseded by D15 (2026-09-19):** key scales are per channel *within a token block*, not over the whole
    context. The 2-bit number above (PPL 23.0) belongs to the offline scheme; the causal block scheme reaches
    15.47–18.58 at the same or slightly higher budget.
- Symmetric 4-bit per-token (PPL 18.9) beats asymmetric per-token (3227) but both are far worse than per-channel,
  so symmetry is a second-order choice compared with the axis.

## D11 — Exact logit ties in bfloat16 make greedy output tie-break dependent (2026-09-19)

- **Measurement:** the two golden RAQ-Probe runs (same data, same model, code differing only in how the next token is
  selected — `argmax` vs `topk(2)`) produced byte-identical teacher-forced scores for all 720 samples but different
  generations for 3. All three had an exact logit tie (decode-path margin 0.0) at the step after the answer value
  ("3711" vs "3711 euros"); all three remained correct.
- **Consequences:**
  - ~0.4% of greedy steps here sit on an exact tie; at a tie the emitted token reflects the tie-break rule, not the
    model. A token-level output change at a tie is not evidence of corruption, so M2's SDC definition uses
    task-level outcomes and reports the tie rate alongside token-level change rates.
  - Ties are recorded: `gen_margin_min_all == 0` flags them per sample.
  - Generation uses one tie-break rule everywhere (`torch.topk`), so golden and perturbed runs are comparable.

## D10 — bfloat16 numerical noise floor for cache-geometry changes (2026-09-19)

- **Measurement:** physically pruning cache entries versus hiding the same entries with a 2D attention mask is
  mathematically identical. On GPU: float32 gives max |Δlogit| = 0.0000 (max KL 9e-9, both sdpa and eager); bfloat16
  gives max |Δlogit| = 0.25 (sdpa) / 0.41 (eager), max KL ~1e-4, argmax unchanged. Re-running the same path twice
  gives exactly 0 (kernels are deterministic), and a mask that hides nothing also gives exactly 0 — so the gap comes
  from changing the set of accumulated keys, not from the mask path.
- **Consequences:**
  - In bf16, any effect below ~0.3 logits / ~1e-4 KL cannot be attributed to an injected fault or to quantization.
    M2 must report this floor next to its effect sizes, and threshold-based SDC definitions must sit above it.
  - Conditions are always compared through the *same* code path (same cache geometry and kernels), so the floor
    largely cancels in paired comparisons; cross-geometry comparisons (e.g. pruned vs unpruned) carry it.
  - The pruning-equivalence test gates on argmax equality plus a KL bound in bf16, and on exact logits in float32.

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
  At 512 tokens accuracy is at ceiling (36/36), so length is the main difficulty knob.
- **Golden float64 answer NLL on correct samples (2K pilot, CPU fp32):** per-type means 9e-13 (tool_arg) to 2.3e-6
  (entity); i.e. resolvable in float64 where float32 returned exactly 0. Reference for Saturday's bf16 values.
- **4K CPU pilot (18 samples, commit cf9a47c):** 18/18 correct; golden `answer_margin_min` 1.4–28.5 (median 16.9;
  the 1.4 case is an `id` at depth 0.1). Across pilots EM is near ceiling at every length (512: 36/36, 2K: 17/18,
  4K: 18/18) and correct-answer margins are similar at 2K and 4K, so length is not a strong difficulty knob for this
  model; the only failure was distractor confusion.
- **Length decision:** keep 512/1024/2048 and add 4096 (720 samples). 4096 represents the KV-memory-dominant regime
  that motivates the project; shorter lengths anchor the length/position axis. Margins (not EM) carry the
  fault-sensitivity analyses, so EM ceiling does not remove headroom.
- **Summaries never pool margins across correctness:** `answer_margin_min_mean_if_correct` (distance to a flip) and
  `..._if_incorrect` (distance from being right) are reported separately, with golden accuracy alongside.

## D8 — M0 split and GPU-free validation (2026-09-17)

- **Context:** the only GPU was busy (10.8/16.3 GiB, 98% utilization) when implementation started.
- **Decision:** M0a code is validated on CPU with the real model in float32 (`configs/experiments/m0a_cpu_smoke.yaml`,
  `tests/slow`). The golden BF16 baseline and latency runs wait for an idle GPU. CPU numbers are never used as the
  baseline (different device and dtype yield a different `exp_id`).
- **Deviation from plan §H:** no separate `scripts/probe_cache_api.py`; the protocol and snapshot tests serve that role.
