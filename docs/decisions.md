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

## D21 — Q49 resolved: the role effect exists, but only above a corruption scale (2026-09-19)

Same probe, same roles, corruption enlarged to the token's **entire row in one layer** (all 8 KV heads,
2048 bits at 2-bit precision), 120 samples, 2071 bursts, no-fault controls exactly zero.

| Corrupted role | Failure rate | Bits flipped |
|---|---|---|
| answer-bearing token, whole row | **2.14%** (14/654) | 2048 |
| same-type distractor token, whole row | 0.46% | 2048 |
| filler token, whole row | 0.31% | 2048 |
| quantization metadata | **3.21%** | **1** |

- **Target vs filler: OR 7.13 (p = 0.004); target vs content-matched distractor: OR 4.75 (p = 0.036).** The
  pre-registered Q49 criterion (OR > 1, CI excluding 1) is **met at this scale**.
- **Scale decides.** The same roles showed OR 1.00 when the burst was one memory word (16 bits, 1/224 of the
  token) and nothing at all for single bits (D19, D20). Semantic role only becomes visible once a token's
  representation in a layer is actually destroyed.
- **Depth reverses between regimes, and the split is significant:** destroying the answer token's row costs the
  answer in late layers only — **14/338 failures in layers 14-27 vs 0/316 in layers 0-13, Fisher p = 1.2e-4**
  (14 events in total; the zero in early layers carries the test). Single-bit damage to WikiText continuation
  concentrated in *early* layers instead (D18). Different task, metric and fault size — the depth profile is
  regime-dependent and a protection policy cannot assume one transfers to the other.
- **Per bit, metadata wins by ~3000×:** one flipped scale bit fails **7/218 = 3.21%** [95% CI 1.30-6.50%] of
  answers, while **14/654 = 2.14%** [1.18-3.57%] fail when 2048 bits inside the answer token's row are flipped —
  a per-bit ratio of 3072×. The event counts are small, so the ratio's order of magnitude is the claim, not its
  third digit.
- **Conditioning caveat:** rates are computed on samples the model answers correctly *at that operating point*.
  At 2-bit block-32 only 109 of 120 samples qualify (vs 147/150 at 4 bits), so the role comparison runs on the
  easier subset, and the 2-bit point is probably too degraded to be M5's primary vehicle.

## D20 — Q49: the corrupted token's role does not matter; what it *governs* does (2026-09-19)

Bursts of 16 bits inside one token's (layer, head) slot — one corrupted memory word — on 150 probe samples at
both operating points, 4050 bursts each, no-fault controls exactly zero.

| Role of the corrupted token | 4-bit point | 2-bit point |
|---|---|---|
| answer-bearing tokens | 0 / 1176 | 0.37% (4/1072) |
| same-type distractor tokens | 0 / 882 | 0.25% |
| filler tokens | 0 / 1176 | 0.37% (4/1072) |
| quantization metadata | **1.36%** | **2.24%** |

- **Target vs filler: odds ratio 1.00 (p = 1).** Identical failure rates. Against the pre-registered Q49 rule
  (OR > 1 with a CI excluding 1) the hypothesis is **not supported**; on the more sensitive prediction-change
  measure the answer tokens are also indistinguishable from filler.
- **Metadata vs filler: OR 6.11 (p = 0.0007) at 2 bits and infinite at 4 bits** (all failures, none in filler).
- Depth and fact type show no role structure either; failures at 2 bits concentrate slightly at depth 0.9 and on
  `number` facts, in every role equally.
- **The emerging rule is not semantic but structural:** damage follows *how much data a bit governs* — a scale
  covers a whole group, an early-layer entry feeds every later layer — not how meaningful the token is to the
  task. That contradicts this project's initial framing, in which semantic importance was expected to predict
  fault vulnerability, and it is the finding M5's allocator must be built on.
- Scope: a burst hits one (layer, head) slot, i.e. 1/224 of that token's cache. The stricter test — the token's
  entire row in one layer, 2048 bits — is `m2_fault_qa_row`.

## D19 — Single-bit faults are harmless at the task level; only metadata isn't (2026-09-19)

Role-targeted single-bit faults on the probe (120 samples, 3240 injections, 4-bit block-128 operating point,
no-fault controls exactly zero):

| Corrupted token role | Task failures | Prediction changed |
|---|---|---|
| answer-bearing tokens | **0 / 960** | 0.31% |
| same-type distractor tokens | 0 / 720 | 0.69% |
| filler tokens | 0 / 960 | 0.73% |
| quantization metadata | **6 / 480 (1.25%)** | 2.08% |

- **Every task-level failure came from a float16 exponent bit of a scale or zero point.** No single-bit code
  fault changed an answer, in any role. Metadata is also the only stratum whose prediction-change rate beats
  filler (odds ratio 2.90 vs filler, p = 0.036).
- **Q49 cannot be answered in this regime:** with zero failures in all three code strata there is nothing to
  compare. The target-vs-filler odds ratio is undefined, and on the more sensitive prediction-change measure the
  answer tokens are, if anything, *less* fragile than filler (OR 0.43, p = 0.34) — consistent with no role
  effect at single-bit strength, not with the hypothesis.
- **Consequence:** the fault unit for the role question becomes a burst — 16 bits inside one token's head, which
  is what a corrupted memory word looks like — run at both the safe (4-bit) and fragile (2-bit) operating points.
- **A sampling bug was found and fixed:** the 120-sample subset was taken with a fixed stride, and because the
  generator emits a group's three depths consecutively, every selected sample had depth 0.1. Subsets are now
  drawn with a seeded random sample. The single-bit numbers above therefore describe depth 0.1 only.

## D18 — Q48 answered: the two sensitivities are *anti*-correlated and point at opposite ends of the network (2026-09-19)

Both maps at (layer, part), same storage scheme (2-bit stress for quantization sensitivity, single-bit faults at
the 4-bit operating point), 56 cells: quantization profile 40 chunks × 56 regions, fault campaign 40 chunks ×
4800 injections (no-fault controls exactly zero).

| Correlation | ρ |
|---|---|
| quantization sensitivity ↔ fault damage | **−0.377** [95% CI −0.588, −0.128] |
| … partialling out mean \|KV\| | **−0.282** |
| within keys only / within values only | −0.271 (p=0.16) / −0.461 (p=0.014) |
| replication on the 8-bit campaign | −0.388, partial −0.277 |

**Against the pre-registered rule** (partial ρ ≥ 0.6 → one shared signal; ≤ 0.3 → separate signals): the result
falls in the *separate signals* branch, and is in fact mildly negative. Quantization-guided protection would
allocate ECC to the wrong regions.

Where each signal lives:

| Rank | Most quantization-sensitive | Most fault-sensitive |
|---|---|---|
| 1-5 | layers 26, 20, 22 (keys), 19 (values), 9 (values) | layers 1, 0, 1, 0, 2 (values and keys) |

- Fault damage concentrates in the **earliest layers** and where mean \|KV\| is *small*
  (ρ(fault damage, mean \|KV\|) = −0.385), while quantization sensitivity sits in **late-layer keys** and grows
  with magnitude (+0.311). A single corrupted early-layer entry feeds every later layer, whereas whole-region
  quantization error matters most where the representation is read out — that reading is a hypothesis; the
  depth profile itself is the measurement.
- Consequence for M5: **allocate precision and protection from different maps.** A single "importance" score
  cannot serve both, which is the joint-allocation problem this project set out to test — now with evidence that
  it is a real problem rather than an assumed one.

## D17 — Fault pilot: quantization metadata is the vulnerable part, and there is no inert bit (2026-09-19)

Pilot: 8 chunks × 122 single-bit injections at the safe operating point (4-bit block-128 keys, 4-bit per-token
values), plus 2 no-fault controls per chunk.

- **The measurement floor is exactly zero.** All 16 no-fault controls produced identical logits (0 tokens changed,
  0.000000 NLL delta), and re-dequantizing an untouched layer also changes nothing. Every observed effect is
  therefore attributable to the injected bit — unlike M1's region profiling, no statistical floor is needed.
- **There is no inert bit at 4 bits.** The original plan had a "null" stratum flipping the LSB of a mid-range
  code, assumed harmless. Measured: one LSB step moves the stored value by 0.168 where the tensor's mean
  magnitude is 0.316 — about half. The stratum was renamed `lsb_midrange` ("the smallest possible storage
  error") and the floor is established by no-fault controls instead.
- **Metadata dominates.** Mean worst-token damage: metadata 0.618 nats vs codes 0.260; the six most damaging
  injections are all float16 **exponent** bits of a scale or zero point, the worst costing 23.6 nats on one token
  and changing 237 of 512 predictions. Within metadata, mantissa and sign flips are minor (≈0.000 mean).
  - Mechanistically unsurprising — one scale covers a whole group — but decision-relevant: metadata is ~6% of the
    stored bits at this operating point, so protecting it is cheap, and it carries the entire damage tail.
- **Code bits are nearly flat in significance:** worst-token damage 0.261 (MSB) vs 0.253 (LSB), and keys ≈ values
  (0.256 vs 0.265). At 4 bits every code step is large relative to the value distribution, so the classical
  MSB/LSB hierarchy barely appears.
- **Confirmed at 8 bits (full campaigns, 4880 + 2840 injections):** damage is strongly *sublinear* in the size of
  the perturbation. From LSB to MSB the stored value changes by 8× (4-bit) and 128× (8-bit), but worst-token
  damage changes only 1.03× and 1.30×, and changed-token counts 1.10× and 1.52×. What bit position really moves
  is the *probability of any effect at all*: the share of injections that change nothing goes 28.1% → 5.6% across
  the 8-bit positions (8.2% → 4.1% at 4 bits).
  - Consequence: **MSB-only protection, the classic selective-ECC move, buys much less than its bit savings
    suggest** for KV caches. Metadata protection is the high-value target instead (2.85× code damage at 4 bits,
    1.97× at 8 bits, while metadata is ~6% of stored bits).
- **Metric choice:** a single bit perturbs a few positions strongly rather than all positions slightly, so the
  campaign reports the **worst scored token** (`delta_nll_max`) and the count of changed predictions; the mean
  over 512 tokens is near zero for every stratum and hides the effect.

## D16 — Perplexity badly understates task damage; "2-bit is usable" is withdrawn (2026-09-19)

Operating points measured on the probe (720 samples) and GSM8K (200), paired per sample against the bf16 golden
runs. WikiText perplexity for the same configurations comes from the M1 sweeps.

| Operating point | bits/value | WikiText PPL | PPL increase | Probe accuracy | Accuracy drop | Decode margin Δ | Truncated |
|---|---|---|---|---|---|---|---|
| K4b/blk128 · V4b/token | 4.26 | 14.43 | +0.3% | 0.9944 | 0% | −0.03 [−0.18, +0.12] | 0 |
| K2b/blk32 · V2b/token | 2.63 | 15.47 | +7.5% | 0.867 | −12.8% | −5.58 [−6.00, −5.13] | 1 |
| K2b/blk512 · V2b/token | 2.17 | 18.58 | +29.1% | 0.285 | **−71.4%** | −11.66 [−12.63, −10.68] | 113 |

GSM8K (cached prefix is only ~62 tokens, so it is expected to be insensitive): 0.825 → 0.800 at 4 bits and 0.735
at 2-bit block-32, i.e. a −11% relative drop where the probe loses −13%.

- **The project's premise now has its first direct measurement.** A 29% perplexity increase corresponds to a 71%
  collapse in retrieval accuracy; a 7.5% perplexity increase already costs 13% of accuracy. The relationship is
  strongly amplifying, so perplexity cannot be used to certify an operating point.
- **D12's "2-bit is degraded but functional" is withdrawn.** It was based on perplexity alone. At 2 bits the
  usable configuration is block-32 (0.867 accuracy), and the cheap 2-bit variant (block-512) destroys retrieval
  while looking merely mediocre on perplexity.
- **4-bit block-128 is lossless downstream** (identical accuracy, margin CI spans zero, no truncations) at 4.26
  bits per value — a 3.8× KV reduction. This is the operating point M2 should treat as the "safe" reference.
- **Truncation tracks damage:** 113 of 720 generations at the worst point ran to the token cap, versus 0 at
  4 bits. Degraded runs ramble, which is why truncation is reported next to accuracy.
- Reporting guard added: `report_m1_downstream.py` refuses incomplete runs. A partial 75/720 run (aborted when the
  operating points changed) had briefly appeared as if it were a complete result.

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

## D22 — Precision and protection are not exchangeable at bit-width granularity (2026-09-20)

- **Context:** `docs/m3_plan.md` specified the decisive M3 comparison as **4-bit + uniform SECDED** versus
  **2-bit + metadata-only SECDED** "at ±1% equal budget". Before booking GPU time the two arms were costed with
  the closed-form predictor (`src/raq_kv/budget/iso.py`, pinned bit-for-bit against the real encoders in
  `tests/test_iso_budget.py`), on the Qwen3-1.7B geometry at 768 cached tokens.
- **Measurement** (bits per cached scalar, `group_token` g=128, asymmetric, SECDED over 64-bit words):

  | arm | data | quant metadata | ECC data | ECC metadata | total |
  |---|---|---|---|---|---|
  | 4-bit + uniform SECDED + metadata SECDED | 4.000 | 0.250 | 0.500 | 0.094 | **4.844** |
  | 2-bit + metadata SECDED only | 2.000 | 0.250 | 0.000 | 0.094 | **2.344** |

- **Finding:** the planned pair differs by **107%**, not ±1%. The reason is structural, not a parameter slip:
  SECDED over 64-bit words costs 8/64 = **12.5%** of the data bits, while one step of the bit-width ladder
  (2→4, 4→8) costs **100%**. The exchange rate between protection and precision is about **8:1**, so *no*
  protection policy on this codec family can ever pay for a bit-width step. §6 of the brief assumed these three
  budget axes were mutually tradeable at a common granularity; for precision-vs-protection they are not.
- **Decision:** the tradeable margin against ECC is **quantization metadata (group size)**, not bit width. Group
  size moves the budget in the same 0.1–1.0 bits/value range as ECC does (g=128 → 0.25, g=64 → 0.50,
  g=32 → 1.00 bits/value), so an iso-budget family can be built by exchanging group size for check bits at fixed
  bit width. M3's arms are rebuilt on that axis; the 2-bit arm is dropped from the iso-budget set.
- **Consequences:** (a) the M5 allocator's decision variables become (bits, group size, codec, protected layer
  set), and *bits* is a coarse knob that mostly selects the operating point rather than participating in the
  fine-grained trade; (b) `iso_budget_report()` enforces the ±1% tolerance in code, and every arm's predicted
  budget is recorded before it runs; (c) this correction of the project's own premise is a result in its own
  right and belongs in the paper, not only in the plan.
- **Falsification:** the 8:1 rate is codec-specific. A code with much larger overhead (e.g. BCH over short words,
  or replication) would close the gap; if such a codec is added, this entry must be re-measured, not assumed.

## D23 — The M3 fault model is a bit error rate over the *declared* representation (2026-09-20)

- **Context:** check bits occupy storage and are counted in the budget, but `faults/plan.py` can only address
  codes, scales and zero points. A protected arm would therefore have been exposed on a *smaller* fraction of the
  bits it claims to store than an unprotected one: the extra 0.5 bits/value of an arm's SECDED would have been
  immune by construction. Any "equal bits" comparison built that way measures the model's blind spot.
- **Decision:** an M3 fault model draws uniformly over **every bit the arm declares** — codes + quantization
  metadata + ECC check bits — in proportion to each region's real size (`src/raq_kv/faults/storage.py`). A
  check-bit flip is a first-class site (`target="check:code"` etc.); it makes the decoder compute a syndrome for
  an error that did not happen and repair intact data, i.e. a miscorrection, which the codec already classifies.
  `flips_for_rate()` converts a bit error rate into a flip count per arm, so "the same rate" means the same thing
  in arms of different shape.
- **Evidence:** `tests/test_storage_faults.py` pins the inventory against the bit budget
  (codes = `kv_data_bits`, metadata = `quant_metadata_bits`, check = `ProtectedKV.ecc_bits`), asserts every drawn
  site is addressable and exactly reversible, and checks that drawn region shares match storage shares to within
  four standard errors. On the test geometry check bits are 5–25% of a protected arm's storage: not negligible.
- **Consequence — the M2 replay rule no longer generalises.** A plan is a list of (layer, part, index, bit)
  addresses, which denote the same *physical* bits only within one layout. Replay therefore stays valid for
  **protected vs unprotected at the same quantization** (the paired comparison that isolates the codec) and is
  **invalid across arms with different group sizes or protection maps**; those are matched on rate instead, with
  the trial count set by power rather than by pairing. `docs/m3_plan.md` inherited the unqualified M2 rule and is
  corrected.
- **Spatial correlation is the axis that decides whether ECC can help at all.** Measured word occupancy of the
  existing generators (Qwen3-1.7B geometry, 176.2 Mbit of codes = 2.75 M 64-bit words, SECDED per word):

  | generator | words with >1 flip | max flips in a word | regime |
  |---|---|---|---|
  | `generate_plan(n_uniform=2048)` | 0.0% | 1 | every fault correctable |
  | `generate_plan(n_uniform=8192)` | 0.1% | 2 | first coincidences appear |
  | `generate_bursts_in_tokens(64 b, scope="layer_token")` | 36.3% | 6 | partly correctable — the crossover |
  | `generate_bursts_in_tokens(512 b, scope="head")` | 100% | 64 | word failure, nothing correctable |

  No new generator is needed: the two physical regimes (uniform SEU, contiguous word/row failure) and the
  crossover between them are already reachable from config. A "≤1 flip per word by construction" stratum was
  considered and **rejected**: SECDED corrects it with probability 1 by definition, so it would be a restatement
  of the code's datasheet dressed as a measurement.
- **Rate calibration (pre-registered).** At D21's 4% base damage rate, halving it needs ~1113 trials per arm
  (two-proportion, 80% power, α=0.05); at a 35–50% base rate, 58–99. The campaign therefore first sweeps the
  rate on the *unprotected* arm alone to find the decade where damage is ~50%, then runs the full arm set at
  three rates spanning it. The rate is a stated experimental condition, never a claim; what is compared is the
  ordering of arms at a fixed rate, and the rate at which that ordering changes.
- **Cost:** a full protected read is 1.3 s and a single layer-part 18 ms on CPU at 768 tokens (SECDED/64 +
  metadata SECDED, 26.1 Mbit of check bits = 14.84% of code bits), so per-burst re-reads dominate nothing;
  5 arms × 3 fault models × 1000 trials ≈ 1.5–3.3 h of decoding.

## D24 — Provenance records the tree, not the process: a guard, and four retired pilot runs (2026-09-20)

- **Context:** during the M3 rate-calibration sweep (four rates in one process) the source tree was edited between
  experiments. The runner collects provenance per experiment, so the four manifests recorded **three different**
  `diff_sha256` values (`41f01749c0`, `d8d02693f0`, `6e6da51d81`) for one campaign. Python had already imported the
  modules, so every rate actually ran the *first* version: two of the manifests describe code that never executed.
- **Why this matters more than it looks:** the damage those runs measured was probably unaffected (the edits were
  reports, docs, and an explicit-zero field on a path the unprotected arm does not use). But "probably unaffected"
  is not a reproducibility claim, and a manifest whose code state is wrong cannot be checked by anyone later. The
  whole point of §28 is that a reader can reconstruct what produced a number.
- **Decision (guard):** `runner._check_process_code_state` remembers the code state of the first experiment in a
  process and refuses any later experiment whose state differs, with the reason stated. A silent inconsistency
  becomes a loud refusal. Pinned by `tests/test_runner_resume.py`.
- **Decision (the pilot runs):** all four `protection_arm` pilot directories were marked `superseded_by` with the
  reason above and renamed to `<exp_id>.superseded-<diff>`; **no record was deleted** (§27). The calibration was
  rerun from a committed tree.
- **What the retired pilot still tells us** (used only to choose an experimental condition, never reported as a
  result): on the unprotected arm, uniform SEU over the stored representation gives 5.0% of answers changed at
  BER 1e-7 (25 flips) and 37.5% at 1e-6 (~555 flips), with task failure 3.75% and 33.75%; no-fault controls
  exactly zero at both. **1e-6 is therefore the campaign's main operating rate**: a ~35% base rate needs ~99
  trials per arm for 80% power against a halving, and the arms provide 480. A first protected-arm probe at 1e-6
  corrected 286 of 286 flips, check-bit flips included, and changed no answer — as expected in a regime where
  every fault lands in its own word.
- **Process rule:** never edit a file that a running process has already imported, and never change code between
  the arms of one comparison. In practice that means `src/` is frozen while a campaign runs.
- **Addendum (2026-09-20):** the rule as first written also forbade editing docs and analysis scripts, on the
  grounds that the diff hash covers the whole tree. That was over-broad and cost waiting time. A changing hash is
  not the hazard; a **mismatch** between the recorded state and the executed code is, and neither docs nor report
  scripts are imported by the runner, so editing them mid-campaign leaves every manifest accurate for its own
  start time. Leaving a number known to be wrong sitting in `docs/` for hours to preserve a uniform hash is the
  worse trade.

## D25 — H2a answered: at equal bits, protecting the quantization metadata is ~100x more cost-effective than uniform data ECC (2026-09-20)

- **Setup.** RAQ-Probe, 60 samples, 8 storage-wide fault draws per sample (480 trials per arm), BER 1e-6 over each
  arm's declared storage (codes + quantization metadata + its own ECC check bits, D23). No-fault controls exactly
  zero in every arm. Iso-budget confirmed from the runs' own recorded budgets: spread 0.11% across B/C/D/E.
  Commit `ea9f711`, arms `m3_arm_*`.

| arm | bits/value | flips/trial | corrected words | task failure | answer changed |
|---|---|---|---|---|---|
| A unprotected (cost reference, −12.4%) | 4.2604 | 524 | 0 | **31.78%** (150/472) | 36.23% |
| B uniform data+metadata SECDED | 4.8581 | 597 | 596.9 | **0.00%** (0/472) | 0.00% |
| C precision, **no data ECC** | 4.8705 | 599 | 106.4 | **0.64%** (3/472) | 1.69% |
| D SECDED layers 9–27 | 4.8670 | 598 | 440.3 | **0.00%** (0/464) | 2.50% |

- **The pre-registered discriminator.** Before the arms ran, two accounts of M2's evidence predicted arm C at
  ~32% (every stored bit equally damaging) or ~0.2% (metadata dominant). Measured: **0.64%**. The metadata account
  wins, and the mechanism is isolated cleanly: arms A and C leave the *same* number of code flips uncorrected
  (492.3 vs 492.5 per trial) and differ only in whether the ~32 metadata flips per trial are corrected. Same code
  damage, 50x different failure rate.
- **Per-bit damage, properly estimated.** From that contrast: **1.29e-5 failures per code flip, 9.83e-3 per
  metadata bit — a factor of 762** *(superseded by D28: the per-code-flip rate was three events in one non-shared sample, i.e. noise. The supported statement is a bound — at most 6.5e-6 per code flip against about 1.0e-2 per metadata bit, so at least 1543x)*, and metadata carries **98.0% of the damage while being 6.0% of the flips**.
  D17/D21 estimated ~3072x from a handful of events; the order of magnitude survives, the point estimate does not,
  and 762x is now backed by 472 trials per arm with failures spread over **51 distinct samples** (not one fragile
  case). D17/D21's ratio should be cited as this number from here on.
- **Cost-effectiveness, the headline.** Metadata SECDED costs 0.236 bits/value in arm C and removes 31.1 points of
  failure (132 points per bit/value). Adding uniform data ECC costs a further 0.500 bits/value and removes the
  remaining 0.64 points (1.3 points per bit/value). **~100x difference.** On task failure B vs C is **not
  significant** (0/472 vs 3/472, Fisher, Holm-corrected p = 0.25); on the stricter `answer_changed` it is
  (0.00% vs 1.69%, p = 0.0076). So uniform data ECC does buy something real and small, at twice the price.
- **H2a supported**, with the mechanism named: the protection budget belongs on the metadata. **H2 as originally
  phrased is supported in a way that changes its meaning** — "adaptive beats uniform" turns out not to be about
  placing ECC on important *data* regions but about not protecting data at all.
- **H2b (placement) is not answerable at this rate, and the reason is more interesting than the failure.** Arm D
  leaves 9 early layers of codes unprotected and still recorded 0/464 failures, because code flips are nearly
  harmless. The placement of data ECC barely matters *because data ECC barely matters*. The question is only
  meaningful where data faults dominate, i.e. D21's whole-row regime, which is the pre-registered crossover test
  and needs the burst fault model rather than uniform SEU.
- **Arm G (added after seeing this).** Arm C's finer key groups are not what protects it; the metadata SECDED is.
  So the deployable configuration keeps arm A's quantization and adds metadata SECDED only: **4.3615 bits/value,
  +2.3% over unprotected, −10.3% under arm B.** If it reproduces C's ~0.64%, the practical claim is *+2.3% storage
  for a ~50x reduction in fault-induced task failure*. Reported as cost-benefit, never as an equal-budget arm.
- **Engagement with prior work.** Published fault studies conclude KV caches tolerate error rates below ~1e-3 with
  minimal perplexity increase. We measure 31.78% of answers broken at **1e-6**, three decades lower. Not a
  contradiction — a metric artefact of the kind D16 already quantified (+29% perplexity coexisting with −71% probe
  accuracy). **Caveat: the 1e-3 figure comes from a search summary and the primary source is paywalled and
  unverified; the instruction-level GPU study we could read explicitly assumes faults do *not* occur in the KV
  cache. The comparison must not be made in the paper until the rate's definition is read in the original.**
- **Consequence for M5.** The allocator's protection dimension collapses to something nearly trivial — protect all
  metadata, it is cheap — which frees the budget for the precision and retention axes. The interesting joint
  problem is therefore (retention x precision) under a metadata-protection floor, not a three-way trade.
## D26 — Trials within a sample are not independent: every M3 comparison must be sample-level (2026-09-20)

- **How it surfaced.** Arm D recorded 12 changed answers and arm E 7 (OR 1.73, Fisher p = 0.35 on 464 trials each),
  which looked like weak support for D18's early-layer sensitivity. Checking *where* those events sat killed it:
  **D's 12 events came from 3 distinct samples (3, 3 and 6 of that sample's 8 trials), all at context 4096; E's 7
  came from 2 samples, both at 2048.** A Fisher test on pooled trials treats 464 trials as 464 independent
  observations; the effective n is the number of fragile *samples*, which is 3 versus 2. The same clustering is in
  arm C: its 3 task failures were all one sample.
- **Decision.** The unit of analysis is the **sample**, not the trial. Each arm gets a per-sample failure rate over
  its 8 draws; arms are compared **paired over the samples every arm answers correctly without faults** (58 of 60
  here) with a Wilcoxon signed-rank test and a paired bootstrap CI over samples. Pooled-trial Fisher tests are kept
  only as a descriptive cross-check and are never the basis of a claim.
- **Corrected results** (58 shared samples, BER 1e-6, 8 draws per sample):

| arm | task failure | answer changed | samples with >= 1 failure |
|---|---|---|---|
| A unprotected | 0.3211 | 0.3664 | **50/58** |
| B uniform data+metadata SECDED | 0.0000 | 0.0000 | 0/58 |
| C finer groups, **no data ECC** | **0.0000** | 0.0108 | 0/58 |
| D data ECC layers 9-27 | 0.0000 | 0.0259 | 0/58 |
| E data ECC layers 0-18 | 0.0000 | 0.0151 | 0/58 |

| paired contrast | task failure | answer changed |
|---|---|---|
| A − B | **+0.3211 [+0.2629, +0.3815]**, p = 6.0e-10, 50 discordant | +0.3664 [+0.3082, +0.4267], p = 2.0e-10 |
| B − C | **identical on all 58 samples**, p = 1 | −0.0108 [−0.0237, −0.0022], p = 0.059, 4 discordant |
| D − E | **identical on all 58 samples**, p = 1 | +0.0108 [−0.0237, +0.0517], p = 0.68, 5 discordant |

- **What changes.** (a) Arm C's 0.64% task failure came from a sample outside the shared set; on the fair
  denominator **C equals B exactly**, so uniform data ECC buys *no* measurable task-level benefit for its
  0.5 bits/value — H2a is supported more cleanly than the pooled analysis suggested. (b) The D-vs-E direction claim
  is **withdrawn**: the 1.73x was a clustering artefact, and the paired CI spans zero widely. H2b remains
  unanswered, not weakly supported.
- **What survives untouched.** The protection effect itself: 32.1 points of task failure removed, 50 of 58 samples
  affected without protection and 0 with it. No clustering correction can touch an effect of that size.
- **Consequence.** `scripts/report_m3.py` must compute sample-level paired statistics as its primary output. The
  trial count still matters for precision *within* a sample, but the degrees of freedom come from samples, so future
  campaigns buy power by adding **samples**, not draws per sample. The earlier power calculations (which assumed
  independent trials) overstate the power of every design that adds draws instead of samples — including the 1e-4
  placement test as sized. Re-size it on samples before running it.

## D27 — The campaign found a bug the tests could not: metadata erasure at higher fault rates (2026-09-20)

- **What happened.** The BER 1e-5 placement run crashed on its third sample with
  `ValueError: bits must be in 1..8, got 16`. `ProtectedKV._decode_region` built its erasure fill with
  `codes_to_bits(..., region.value_bits)`, which is correct for 2-8 bit codes and illegal for the 16-bit float16
  metadata regions. The branch is only reached when a metadata word is **detected-uncorrectable**, i.e. when two
  flips land in the same 16-bit scale word — expected 2.5e-4 times per trial at 1e-6 and ~100x more at 1e-5. The
  unit tests covered metadata *correction* and code *erasure*, but not their intersection.
- **Fix.** `_erasure_fill` now branches on what the region stores: the mid-code for codes, **float16 zero** for
  metadata. An erased scale collapses its group to the zero point (the group's offset) and an erased zero point
  centres the group at 0 — visibly neutral rather than a plausible-looking wrong number, which is what an erasure
  is for. Pinned by two tests, including one asserting the group reads back as its stored offset.
- **Consequence.** Both 1e-5 runs were retired with `superseded_by` and renamed (2 valid units each, kept). The
  lesson is not "add a test" but *which* test: the gap was an untested **interaction** between two features that
  each had good coverage. Rare-branch interactions of this kind are what a rate sweep exposes and a unit test
  suite does not, so the campaign ladder (calibrate, then climb rates) is also a test strategy.
- **Related measurement.** Chasing the crash produced the mechanism behind D25's metadata dominance. Flipping each
  bit of a float16 scale changes it by: mantissa bits 0-9, x1.00 to x1.36; exponent bits 10-12, x0.5 / x4 / x0.06;
  **bit 13, x0.004; bit 14, x65536**; bit 15 flips the sign. NaN is *not* the mechanism (produced at a rate of
  0.0009, and only when one value is hit twice, which does not happen at campaign rates). **Metadata is damaging
  because six of its sixteen bits can rescale an entire quantization group by orders of magnitude**, while a code
  bit moves one value by at most a few quantization steps.
- **Open question this raises, to be tested before metadata SECDED is recommended.** If the damage comes from
  implausible magnitudes, a **range check** on the stored scales — two float16 bounds per region, about 3.5 kbit
  against 596 Mbit, i.e. free — would catch exactly the bits that matter (13, 14, 15 certainly; 12 partly) without
  any error-correcting code. Arm H must test it: if a free validity check matches metadata SECDED, the paper's
  recommendation changes from "protect the metadata with ECC" to "check that the metadata is plausible".
## D28 — Three corrections from the 1e-5 runs, and a much tighter bound (2026-09-20)

The BER 1e-5 placement runs were meant to check that damage stays linear in uncorrected code flips before that
linearity was used to size a 1e-4 campaign. It does not, and chasing why corrected two earlier numbers.

### 1. The linear extrapolation fails, and the per-code-flip rate was noise

Arm D at 1e-5 carries ~1525 uncorrected code flips per trial and recorded **0 task failures in 448 trials**. The
rate extrapolated from 1e-6 (1.29e-5 failures per code flip) predicted 1.97%. That rate came from arm C's three
events, all in one sample that was not even in the shared denominator — i.e. it was noise given a decimal point.

**Withdraw the 315x figure from D25.** Replace it with a bound, which is what the data supports:

* largest exposure measured: arm D at 1e-5, longest context band, 152 trials x ~3037 uncorrected code flips =
  **4.6e5 flip-exposures, 0 task failures** -> 95% upper bound (rule of three) **6.5e-6 per code flip**;
* per metadata bit, from the clean A-vs-G contrast: **1.0e-2**;
* therefore metadata is **at least 1543x** more damaging per bit, and the point estimate is unbounded because code
  damage is consistent with zero at every exposure measured.

The sentence to use: *a code bit flip has at most a 6.5e-6 chance of costing an answer; a metadata bit flip has
about a 1.0% chance.*

### 2. The "protection fails through its own erasure policy" finding was Simpson's paradox — withdrawn

Arm D at 1e-5 showed 5.3% of DUE-bearing trials changing their answer against 1.2% of the rest (Fisher OR 4.79,
p = 0.012), which looked like the erasure policy being a damage source. Stratifying by cache length destroys it:

| band (tokens) | n | DUE rate | changed given DUE | changed given none |
|---|---|---|---|---|
| 0–700 | 88 | 0.170 | 0/15 | 0/73 |
| 700–1400 | 136 | 0.228 | 0/31 | 0/105 |
| 1400–2800 | 104 | 0.567 | 0/59 | 0/45 |
| 2800–4200 | 152 | 0.697 | 11/106 | 3/46 |

Long contexts have both four times the flips (so far more DUEs) and all of the damage. Inside the only band with
any events the odds ratio falls to **1.66** on 11 versus 3 events — no effect. **Withdrawn.** The lesson is the
same one D26 taught in a different guise: this campaign's records vary in cache length by a factor of eight, so
*any* pooled comparison is exposed to it, and length must be a stratum or a covariate in every analysis.

### 3. What replaces both: a within-run dose-response, and protection that holds where damage is worst

Cache length sets the flip count, so a single run contains its own dose-response. Task failure by band:

| flips per trial | A unprotected | G metadata SECDED | H range guard |
|---|---|---|---|
| ~130 | 5.7% | 0.0% | 0.0% |
| ~255 | 18.4% | 0.0% | 0.0% |
| ~504 | 40.4% | 0.0% | 0.0% |
| ~1005 | **51.3%** | **0.0%** | **0.0%** |

This is the robustness check the headline needed. "31.8% -> 0%" is an average over lengths, and the obvious
objection is that easy short contexts carry it. They do not: at the longest contexts the unprotected cache loses
**half** its answers and both protected arms lose none. It also reproduces the calibration curve's shape from an
independent axis — varying flips through context length rather than through the error rate — which is a stronger
form of the same measurement than either alone.
## D29 — The range guard, measured end to end: what it catches, where it breaks, and how it fails (2026-09-20)

Arm H protects the quantization metadata with two stored float16 bounds per region and no error-correcting code
at all: **+0.075% of storage** (3584 bits against 596 Mbit). Three measurements describe it completely.

### 1. It catches exactly the bits the mechanism predicts

A float16 scale has 16 bits. Three of them — 13 (x0.004), 14 (x65536) and 15 (sign) — push the value outside the
range the clean tensor occupied; the other thirteen move it by at most 4x and stay inside. So a range check should
catch **3/16 = 18.75%** of metadata flips and be blind to the rest.

Measured, on trials where the guard's own bounds were not hit: **20.1%** at BER 1e-5 and **20%** at 1e-6. The
prediction and the measurement agree at both rates, which is what makes this a mechanism rather than a
correlation.

### 2. It is enough at 1e-6 and not at 1e-5 — the engineering boundary

| | BER 1e-6 | BER 1e-5 |
|---|---|---|
| flips per trial | 524 | 5240 |
| metadata flips per trial | 31.7 | 315.5 |
| caught by the guard | 20% | 20% |
| **task failure, arm H** | **0.00%** (0/472) | **5.93%** (28/472) |

The ~13 of 16 metadata bits the guard cannot see are individually mild, so at 31.7 metadata flips per trial they
cost nothing measurable; at 315 per trial they accumulate to 5.9%. **Below roughly 1e-6 the free check is
sufficient; above it, correction earns its bits.** Arm H still removes most of the damage even where it is no
longer sufficient — an unprotected cache at 1e-5 would be near-saturated, against arm H's 5.9%.

### 3. Its failure mode is its own bounds, and the failure is false positives

Because the bounds are themselves stored bits exposed to the fault model, a flip can land on one. Over the 1e-5
run, 24 of 472 trials had a bound flip, and in those trials the guard flagged **1733 values against 277 real
metadata flips — a 625% "catch" rate.** A corrupted bound narrows the range, so the guard clamps hundreds of
perfectly valid scales. That is the defect worth naming: *a plausibility check whose notion of plausibility is
corrupted attacks the data it defends.*

At this scale the false positives were not measurably harmful — task failure was 4.17% in bound-flip trials
against 6.03% in the rest, i.e. if anything lower, with 24 trials and no significance either way. Clamping a
valid scale to a bound near the edge of its real distribution is a small perturbation; the damage that does get
through comes from the mantissa flips the guard cannot see. But "not harmful at the rates measured" is the claim,
not "harmless".

**This is why the bounds had to be exposed.** With the guard invulnerable, as arm H was first run, the
false-positive mode is invisible by construction and the arm reports a coverage it would not have in a real
memory. The exposure cost nothing in the headline — 0/480 either way — and bought the only honest account of how
the mechanism fails.

### Consequence for the recommendation

The deployable configuration is a range check, not a code, **at rates around 1e-6 and below**, with two caveats
stated rather than buried: the bounds need their own integrity (the ordered-and-finite self-check is free and
caught one case; a checksum over 3584 bits would cost nothing and close the rest), and the sufficiency is
rate-dependent in a way a single-rate table would hide.

## D30 — The crossover, measured: where a free check stops matching a code (2026-09-20)

Arm H (two float16 bounds per metadata region, **+0.075%** of storage, no error-correcting code) against arm G
(metadata SECDED, **+2.3%**), paired over the 59 samples both answer correctly without faults:

| | BER 1e-6 | BER 1e-5 |
|---|---|---|
| H, range check | **0.0000** | **0.0593** |
| G, metadata SECDED | **0.0000** | **0.0000** |
| H − G, paired | +0.0000 [0.0000, 0.0000], p = 1, **0 discordant samples** | **+0.0593 [+0.0381, +0.0847], p = 1.1e-5, 22 discordant** |

- **At 1e-6 neither arm failed at all**: 0/472 each, so the paired test has 0 discordant samples. That is an
  *uninformative* comparison, not a demonstration of equality — two arms pinned at the floor cannot be told
  apart by construction. What the rate establishes is an upper bound on the code's advantage: with 472 trials
  and no events, at most ~0.6% by the rule of three. Thirty times the storage buys at most that much, which is
  still the useful claim, but it is a bound and is written as one. (The same discipline as the D-vs-E null:
  0 discordant means the test could not discriminate.)
- **At 1e-5 the code wins decisively.** The guard sees only the 3 of 16 float16 bits that push a scale out of its
  clean range (measured coverage 20.1%, D29); the other thirteen are individually mild and, at ten times the
  rate, accumulate. SECDED corrects all of them — 434 words per trial — and stays at zero.
- **Arm G's zero at 1e-5 is not an absence of hard cases.** 28 detected-uncorrectable metadata words occurred
  across the 480 trials, one per affected trial, split over scale and zero-point regions — and the erasure
  policy absorbed every one of them at no task cost. This is the same path that crashed the campaign in D27,
  now exercised 28 times in anger and correct. The coincidence arithmetic that predicted it also lands:
  0.051 DUEs per trial predicted from two flips meeting in one 16-bit word, 0.058 measured (and at 1e-6,
  0.001 predicted, 0.000 measured).
- **So the recommendation has a boundary, and the boundary is the result.** Below roughly 1e-6, spend 0.075% on a
  plausibility check. Above it, spend 2.3% on metadata SECDED. In neither regime is uniform data ECC (14%) worth
  its bits: arm B matched arm G at 1e-6 and neither differs from it in any measurement so far.
- **Why this needed two rates.** A single-rate table would have supported "a free check is as good as a code",
  which is true at 1e-6 and false one decade up. Reporting the rate at which an ordering changes, rather than an
  ordering, is the form this kind of claim has to take — the same discipline D23 set for the fault model and D26
  for the statistics.

**Consequence for M5.** The protection axis reduces to a rate-dependent floor with two settings, both cheap. The
allocator should take the deployment's expected error rate as an input and pick between them, rather than
searching a protection dimension: there is nothing to search.

## D31 — H2b: placement does not matter for task outcomes, and the proxy effect will not resolve (2026-09-20)

The properly sized placement test: arms D and E, bit-identical, 19 of 28 layers protected each, differing only
in *which* nine are left exposed — D the early layers 0–8, E the late layers 19–27. 180 samples of the
4096-token condition at BER 1e-5, 8 draws each, no-fault controls exactly zero, 177 samples paired.

| endpoint | D (early exposed) | E (late exposed) | D − E | p | discordant |
|---|---|---|---|---|---|
| task failure (secondary) | **0.0000** | **0.0000** | 0.0000 | 1 | 0 of 177 |
| answer changed (primary, pre-registered) | 0.0177 | 0.0064 | +0.0113 [+0.0007, +0.0247] | **0.074** | 7 of 177 |

- **Against the pre-registered rule, this is not support.** `docs/prereg_h2b.md` required Wilcoxon p < 0.05
  **and** a bootstrap CI excluding zero. The CI excludes zero; p = 0.074 does not clear the threshold. The
  pre-registration also said that anything else "is a third underpowered null and will be reported as one rather
  than dressed up", so that is what this is.
- **The sizing was wrong again, and for the same reason.** The run was built for ~130 events per arm from a 9.2%
  proxy rate measured on 19 samples with 3 discordant — itself noise. The real rate at 4096 tokens is 1.77%, so
  the run produced 25 and 9 events. A sign test needs ~13 discordant samples for 80% power here; there were 7.
  **Twice now a power calculation has been built on an effect size estimated from a handful of events.** The rule
  to take forward: size from a rate measured on a *different* set than the one that suggested the effect.
- **What is nevertheless true.** The direction D18 predicts — exposing early layers is worse — has now appeared
  in four independent tests (ratios 1.72×, 2.34×, 3.50×, 2.77×) and in 6 of the 7 discordant samples here. Four
  replications of a sign is not nothing. It is also not the pre-registered threshold, and it will be reported as
  a consistent direction below significance, not as a finding.
- **The answer that *is* established, and it is the more useful one.** Task failure is **0 of 1416 trials in both
  arms**. Protecting 19 of 28 layers is sufficient at this rate and length **regardless of which 19**. Placement
  shows up only on a proxy that does not translate into a single lost answer. For the allocator, H2b's practical
  answer is therefore: **do not spend search effort on where to put data ECC** — spend it on whether to buy data
  ECC at all, and D25–D30 already answered that with "no".
- **Where the question could still be answered.** Only in the regime where data faults dominate outcomes rather
  than a proxy: D21's whole-row bursts, where late layers carried all 14 failures and early layers none. That is
  the opposite sign to D18's, so a crossover remains the live hypothesis — but it needs the burst fault model,
  not more samples of uniform SEU. Deferred to M5 with the fault model as an allocator input, not re-run here.

## D32 — The attention sink is one token, and it is worth every answer (2026-09-20)

- **How it surfaced.** The first M4 gate ran two arms: drop only the answer-bearing tokens (99% retained), and
  keep only them (5% retained). Both failed **~100%** of answers. The second was built as a sanity control and
  should have been near ceiling, so its failure said the pruning path was broken, not that targets matter.
- **A one-minute diagnostic separated the causes**, on 12 probe samples at 512 tokens, each condition against
  the same golden:

  | dropped | task failures |
  |---|---|
  | nothing (control) | **0/12** |
  | ten middle filler tokens | **1/12 = 8.3%** |
  | **token 0 alone** | **12/12 = 100%** |
  | tokens 0–3 | 12/12 = 100% |
  | the answer-bearing tokens | **12/12 = 100%** |

- **Position 0 is an attention sink.** Dropping that one token out of ~1000 fails every answer. This is the
  StreamingLLM phenomenon and every practical retention method keeps an initial block for it; our design did
  not, so *both* gate arms dropped the sink in most samples and measured it instead of what they were built to
  measure. Both runs were retired with `superseded_by` and rerun-marked; no records deleted.
- **The target tokens are load-bearing too**, which the confounded gate could not establish: 12/12 against
  8.3% for the same number of filler tokens. So the probe *can* carry retention research, and the gate's
  conclusion survives — but only because the diagnostic separated it from the sink, not because the gate showed
  it.
- **Decision.** `select_keep` protects an initial block by default (`protect_initial=4`), alongside the recent
  tail it already protected. A default that silently changes results is normally bad practice; here the
  alternative is that every future config must remember a rule whose violation costs 100% of the outcome, and
  the default is tested rather than assumed.
- **The finding that outlives the bug.** Combined with D21 this resolves the M2 redundancy puzzle. Destroying
  the answer token's entire row **in one layer** cost 2.14% of answers; **removing the same token from the
  cache** costs 100%. The information lives in that token and nowhere else, but it is replicated across the 28
  layers' representations of it, so losing one layer's copy is survivable and losing the token is not.
- **Consequence for M5.** The two axes have different granularity, and it is now measured rather than assumed:
  **retention is all-or-nothing across layers** — there is no such thing as keeping a token in some layers —
  while protection *can* be layer-selective and, per D31, does not even need to be. An allocator that treats
  both as per-(layer, token) decisions is modelling a choice that does not exist on one of the axes.

## D33 — The answer tokens are necessary and not sufficient; the unit of retention is under test (2026-09-20)

First real M4 measurement, at 25% retention on 117 probe samples answered correctly without pruning, with the
attention sink protected (D32). Broken out by the depth at which the probe places its fact, because recency
keeps deep targets for free and an aggregate would hide the whole comparison:

| arm | shallow | middle | deep | kept_target (shallow/middle/deep) |
|---|---|---|---|---|
| recency | 1.000 | 1.000 | 0.027 | 0.00 / 0.00 / 1.00 |
| oracle_keep_target | **0.773** | **0.861** | 0.027 | 1.00 / 1.00 / 1.00 |
| oracle_drop_target (99% retained) | 0.977 | 0.972 | 1.000 | ~0.05 |

- **Recency behaves exactly as the probe's construction predicts**, which is the check that the pruning path is
  doing what it claims: it keeps the deep targets and drops the shallow ones, and its accuracy follows.
- **The finding is the oracle.** Retaining **100% of the answer-value tokens** still fails **77% of shallow and
  86% of middle-depth** questions, while dropping those same tokens fails 98%. The value tokens are
  **necessary and not sufficient** — a pair of statements that a single arm could not have produced.
- **Consequence: token-level importance may have a low ceiling on retrieval tasks.** If perfect identification
  of the answer tokens does not rescue accuracy, then no token-level scorer can, and the literature's
  token-scoring framing inherits that ceiling on this kind of task. That is a claim about the *unit* of
  importance, not about any particular scorer, and it is testable rather than rhetorical.
- **Next test, already running:** `oracle_keep_sentence` keeps the whole sentence the fact sits in. If it
  rescues the shallow and middle questions, importance is a property of spans and token-level scoring is the
  wrong granularity. If it does not, something larger than the sentence is load-bearing and the next candidate
  is the document structure itself. Either answer redirects M4; neither leaves it where it was.
- **Reporting rule adopted here.** Every retention number is broken out by target depth, and
  `kept_fraction_target` and `kept_fraction_sentence` print next to accuracy always. A scorer that holds
  accuracy while discarding the answer tokens has found redundancy, not importance (D32), and the only way to
  see that is to keep the diagnostic beside the outcome.

## D34 — Importance is a property of spans, not tokens (2026-09-20)

The test D33 set up. Same 25% retention budget, same protected attention sink, same 117 probe samples answered
correctly without pruning, and **the answer-value tokens fully retained in both oracle arms**. The only
difference is the *unit* that retention operates on.

| arm at 25% retention | overall task failure | shallow | middle | deep |
|---|---|---|---|---|
| recency (trivial baseline) | 0.6923 | 1.000 | 1.000 | 0.027 |
| `oracle_keep_target` — a **perfect token-level** scorer | 0.5641 | 0.773 | 0.861 | 0.027 |
| `oracle_keep_sentence` — the fact's **whole sentence** | **0.1538** | **0.227** | **0.194** | 0.027 |

Paired over the 117 shared samples: **+0.4103 [+0.3248, +0.5043], Wilcoxon p = 4.3e-12, 48 of 117 samples
discordant.** No floor effect, no clustering caveat, no underpowered null — the cleanest comparison in the
project so far.

**The confound is excluded, and it resolves in favour of locality.** Both oracle arms keep the value tokens
and hit the same budget, but the span arm spends part of that budget on a contiguous block *at the fact* while
the token arm spends it elsewhere — so "the value tokens need their local context" and "a contiguous block of
context anywhere helps" both survived the table above. `oracle_keep_random_span` places a same-sized
contiguous window elsewhere in the document, matched on budget, value tokens, contiguity and width:

| arm at 25%, value tokens fully retained in all three | task failure |
|---|---|
| contiguous window **at the fact** (`oracle_keep_sentence`) | **0.1538** |
| same-sized contiguous window **elsewhere** (`oracle_keep_random_span`) | **0.5812** |
| no context at all, value tokens only (`oracle_keep_target`) | 0.5641 |

**Contiguity per se buys nothing**: a contiguous block in the wrong place performs like keeping no context at
all (0.5812 against 0.5641). Paired sentence against random span: **+0.4274 [+0.3333, +0.5214],
p = 1.5e-12, 50 of 117 discordant.** So the mechanism sentence stands as written — the value tokens are not
interpretable without *their own* local context, not merely without some context.

- **The sharpest form of the result.** A *perfect* token-level scorer buys **13 points** over the trivial
  recency baseline. Changing the unit of retention to spans buys **54**. The headroom that the KV-compression
  literature is competing over — better token scores — is a quarter of the headroom sitting in the choice of
  unit, which that literature does not vary.
- **Why it is not a tautology.** The span arm keeps the same answer-value tokens as the token arm; it adds only
  the rest of the sentence around them. So the gain is not "the oracle found the answer" — both oracles found
  it — but "the answer's value tokens are not interpretable without their local context". Necessary and not
  sufficient (D33) becomes: *sufficient once the span is intact*.
- **The deep column carries no information.** All four arms read 0.027 there — the same single sample failing
  everywhere — because a deep target is kept by *every* arm including recency, by construction of the probe.
  It is not evidence that the methods agree on deep targets; it is evidence that deep targets were never at
  risk. Only the shallow and middle bands discriminate.
- **It is not the whole story either.** 15.4% of questions still fail at span level, so something beyond the
  sentence is load-bearing for those. That residual is the next question, not a rounding error, and the next
  candidate is document structure rather than a bigger window.
- **Consequences.** (a) M4's scorers should be evaluated at span granularity, and a token-level scorer should
  be reported against the span ceiling rather than against recency alone, or it will look better than it is.
  (b) M5's retention decision variable is a **span**, not a token — which, with D32 (retention is all-or-nothing
  across layers), fixes the allocator's decision unit on both axes from measurement rather than convenience.
  (c) The probe labels sentences, so this is measurable here; on a natural corpus the span boundary becomes a
  modelling choice and that limitation must be stated with the result. **Stated more strongly:** the probe's
  facts *are* single sentences by construction, so this experiment cannot distinguish "the sentence" from
  "whatever unit the generator used" — they are the same object. On a natural corpus the load-bearing span
  might be a paragraph, a table row, or discontiguous, and the finding as phrased would need re-testing.
- **Scope, stated because it is easy to overclaim.** This is one model, one synthetic retrieval probe, and
  oracles rather than deployable scorers. What is established is a *ceiling comparison* between two units, not
  that any practical span scorer achieves it. A real span-level method still has to be built and beaten against
  KIVI's own optimum (`docs/literature.md`).

## D35 — The unit transfers to real scorers; locating the fact does not (2026-09-20)

D34 established the span result with oracles. This is the deployable version: punctuation segmentation from the
tokenizer's own pieces, no ground truth, same 25% budget, 117 samples, sink protected.

| arm | kept_target | task failure |
|---|---|---|
| `oracle_keep_sentence` (ceiling) | 1.000 | **0.154** |
| `oracle_keep_target` | 1.000 | 0.564 |
| `oracle_keep_random_span` | 1.000 | 0.581 |
| **`window_attention` @ span** | 0.419 | **0.624** |
| `recency` (trivial baseline) | 0.316 | 0.692 |
| `key_norm` @ span | 0.248 | 0.778 |
| **`window_attention` @ token** | 0.203 | 0.855 |
| `key_norm` @ token | 0.216 | 0.957 |

Paired over the 117 shared samples:

| contrast | difference | p | discordant |
|---|---|---|---|
| `window_attention` token → span | **+0.2308 [+0.1624, +0.3077]** | 2.0e-07 | 27 |
| `key_norm` token → span | **+0.1795 [+0.1111, +0.2479]** | 4.6e-06 | 21 |
| `recency` → `window_attention` @ span | +0.0684 [+0.0256, +0.1197] | 0.0047 | 8 |
| `window_attention` @ span → oracle | **+0.4701 [+0.3761, +0.5556]** | 1.2e-13 | 55 |
| `recency` → `window_attention` @ **token** | **−0.1624 [−0.2479, −0.0769]** | 0.00064 | 31 |

- **The unit effect is not an artefact of oracles.** Span expansion improves both real scorers by 18–23 points,
  which is most of what it gave the oracle. It costs nothing but a punctuation pass.
- **At token granularity both "smart" scorers lose to keeping the tail.** `window_attention` is 16 points
  *worse* than recency and `key_norm` is 27 points worse. Token-level importance scoring is not merely weak
  here; it is actively harmful relative to the trivial baseline. *(Scoped by D36: this holds for
  **query-agnostic** scoring only. A query-aware scorer at the same token granularity beats recency by 41
  points, so the separating factor is query-awareness, not token-level scoring as such.)*
- **The problem decomposes cleanly, and the two halves have different prospects.** The unit is worth ~20 points
  and is free. Locating the fact is worth the remaining ~47 points and is unsolved: `kept_target` is 0.42 for
  the best query-agnostic scorer against 1.00 for the oracle. Improving the *score* is the hard half; changing
  the *unit* is the free half that the literature does not vary. *(D36 solved the hard half for the case where
  the query is known — a query-aware scorer reaches kept_target 1.00 — at the cost of no longer working on a
  prefix cache shared across queries.)*
- **A fairness caveat that changes the reading, and the arm that settles it.** `window_attention` here is
  **query-agnostic**, while SnapKV and H2O place their observation window at the end of a prompt that contains
  the question — their scorer is effectively query-aware. Reporting only the agnostic variant as
  "attention-based importance" would compare the literature against a weaker version of itself, which is the
  baseline failure the review protocol treats as a red line. `query_attention` (running) scores cached tokens
  by the attention the **query** pays them. Its cost is real and stated: the scores depend on the query, so
  they cannot be computed at prefill time and a prefix cache shared across queries cannot use them — which
  `window_attention` can. Until it lands, the "token-level scoring loses to recency" statement is established
  for the query-agnostic case only.

## D36 — Query-awareness dominates, the unit compounds with it, and my "oracle" was not a ceiling (2026-09-20)

The fairness arm from D35 landed and reversed two of that entry's readings. Same 25% budget, 117 samples.

| arm | kept_target | task failure |
|---|---|---|
| **`query_attention` @ span** | 1.000 | **0.0256** |
| `oracle_keep_sentence` | 1.000 | 0.1538 |
| **`query_attention` @ token** | 0.867 | 0.2821 |
| `oracle_keep_target` | 1.000 | 0.5641 |
| `window_attention` @ span | 0.419 | 0.6239 |
| `recency` | 0.316 | 0.6923 |
| `window_attention` @ token | 0.203 | 0.8547 |

| contrast | difference | p | discordant |
|---|---|---|---|
| query-awareness at token granularity (`wa` to `qa`) | **+0.5726 [+0.4701, +0.6752]** | 1.0e-14 | 75 |
| the unit, for the query-aware scorer (`qa` token to span) | **+0.2564 [+0.1795, +0.3333]** | 4.3e-08 | 30 |
| the full deployable stack against recency | **+0.6667 [+0.5812, +0.7521]** | 1.0e-18 | 78 |
| `oracle_keep_sentence` to `query_attention` @ span | **+0.1282 [+0.0684, +0.1966]** | 1.1e-04 | 15 |

- **Retraction 1: `oracle_keep_sentence` is not a ceiling, and D34/D35 should not have called it one.** A real
  method beats it by 13 points. The reason is plain in hindsight: that arm keeps the fact's sentence and fills
  the *rest* of its budget by recency, so it spends a fifth of the cache on the tail regardless of relevance.
  An oracle is an upper bound only for the policy class it represents; this one is "the ground-truth-sentence
  policy", not "the best possible use of 25%". Every use of the word ceiling for it is withdrawn.
- **Retraction 2: "token-level importance scoring is actively harmful" holds only for query-agnostic scoring.**
  D35 measured `window_attention` losing to recency by 16 points and generalised too far. `query_attention` at
  the same token granularity *beats* recency by 41 points. The correct statement is that **query-awareness,
  not the score family, is what separates a useful token scorer from a harmful one** — and it is worth 57
  points, the largest single effect measured in M4.
- **What survives, strengthened.** The unit effect is real for every scorer tried and compounds with
  query-awareness: +23 points for `window_attention`, +18 for `key_norm`, **+26 for `query_attention`**, whose
  span arm reaches 0.0256 — a 96% cut in cached tokens costing 2.6% of answers.
- **The cost that makes this a trade rather than a free win.** `query_attention` cannot be computed at prefill
  time: its scores depend on the query, so a prefix cache shared across queries cannot use it, while
  `window_attention` can. M5 therefore has two regimes — **query known at compression time**, where query-aware
  span retention is extremely strong, and **shared prefix caches**, where the best available is
  `window_attention` @ span at 0.624 and the gap to 0.026 is the open problem.
- **Method note.** This is the seventh finding in the project withdrawn or re-scoped after a control, and the
  control existed only because reporting a query-agnostic scorer as "attention-based importance" would have
  compared the literature against a weaker version of itself. Without that arm, M4 would have concluded
  "attention-based importance loses to recency".

## D37 — With a good retention method the token-precision trade goes away at 4 bits (2026-09-20)

"More Tokens, Lower Precision" (arXiv 2412.12706) compares 1x tokens at 16 bits, 2x at 8 and 4x at 4 at equal
memory and finds more tokens at lower precision wins. Their retention is not a strong query-aware method. Ours
is (D36), so the trade was re-run rather than inherited, with query-aware span retention throughout.

| arm | bits / ORIGINAL token | task failure |
|---|---|---|
| 100% of tokens @ 4-bit | 4.260 | **0.0000** (0/117) |
| 50% of tokens @ 8-bit | 4.096 | **0.0085** (1/117) |
| 25% of tokens @ bf16 | 3.947 | **0.0256** (3/117) |

| contrast | difference | p | discordant |
|---|---|---|---|
| 25% bf16 vs 100% 4-bit | +0.0256 [+0.0000, +0.0598] | 0.083 | 3 |
| 25% bf16 vs 50% 8-bit | +0.0171 [+0.0000, +0.0427] | 0.157 | 2 |
| 50% 8-bit vs 100% 4-bit | +0.0085 [+0.0000, +0.0256] | 0.317 | 1 |

- **Not a replication, and not a refutation: a floor.** The ordering points the same way the published finding
  does, but with 0, 1 and 3 failures out of 117 **no comparison is significant** and the project's own rule
  (D30) says three arms this close to zero support a bound, not an ordering. What is established is that all
  three are near-lossless.
- **The finding is that the trade stops mattering.** The published comparison found a gap because its retention
  was weak enough for token count to matter. With a retention method that keeps the right tokens, **any** of
  the three splits costs at most 3 of 117 answers at ~4 bits per original token.
- **So the practical advice inverts.** At the same 25% budget, the *method* is worth 67 points
  (recency 0.6923 against query-aware span 0.0256) while the *split* is worth at most 3. Optimising where to
  put bits between token count and precision is second-order next to choosing how tokens are selected —
  which is the axis the compression literature varies least.
- **Where the trade should bite, and the next family.** At ~1 bit per original token the three matched points
  are 25% @ 4-bit (1.070), 12.5% @ 8-bit (1.039) and 6.25% @ bf16 (1.000). That budget is four times tighter
  and the arms should separate; running it is the way to find the trade rather than assert its absence from a
  floor.
- **Scope.** One model, one synthetic retrieval probe, and a query-aware method that cannot run on a shared
  prefix cache (D36). On a task where retention is harder — or with the query-agnostic scorer, whose best arm
  sits at 0.624 — the trade may well reappear, and that is a prediction rather than a caveat.

## D38 — At a tight budget the token-precision trade returns, and the published direction holds (2026-09-20)

D37 found the trade invisible at ~4 bits per original token: all three splits were within 3 failures of 117 and
no comparison was significant. That was a floor, so the family was re-run four times tighter, where the arms can
separate. Query-aware span retention throughout, 117 samples, sink protected.

| arm | bits / ORIGINAL token | kept_target | task failure |
|---|---|---|---|
| 25% of tokens @ 4-bit | 1.054 | 1.000 | **0.0085** (1/117) |
| 12.5% of tokens @ 8-bit | 1.016 | 1.000 | **0.0684** (8/117) |
| 6.25% of tokens @ bf16 | 0.967 | 0.915 | **0.1966** (23/117) |

| contrast | difference | p | discordant |
|---|---|---|---|
| 6.25% bf16 vs 25% 4-bit | **+0.1880 [+0.1197, +0.2564]** | 2.7e-06 | 22 |
| 6.25% bf16 vs 12.5% 8-bit | **+0.1282 [+0.0684, +0.1880]** | 1.1e-04 | 15 |
| 12.5% 8-bit vs 25% 4-bit | **+0.0598 [+0.0171, +0.1026]** | 0.0082 | 7 |

- **The published direction replicates, cleanly and monotonically.** More tokens at lower precision wins at
  every step, all three comparisons significant. "More Tokens, Lower Precision" (arXiv 2412.12706) holds with a
  retention method far stronger than the one it was established with, which is the useful form of a
  replication: the conclusion was not an artefact of weak retention.
- **One arm carries a confound, and the clean comparison survives without it.** The 6.25% arm has
  `kept_target` 0.915 — at that budget even a query-aware scorer cannot fit the whole answer span, so part of
  its disadvantage is lost answer tokens rather than precision. The 12.5% and 25% arms both keep **100%** of
  the answer tokens and differ only in count against precision, and that comparison still favours more tokens
  (+0.0598, p = 0.008). The finding does not rest on the confounded arm.
- **Where the trade appears, and why it vanished before.** At ~4 bits every split is near-lossless, so there is
  nothing to trade; at ~1 bit the budget is scarce enough that how it is divided matters. The practical reading
  is a two-regime one: **above roughly 4 bits per original token, choose the split for engineering convenience
  and spend the effort on the retention method; below it, prefer more tokens at lower precision.**
- **The joint operating point M4 delivers.** 25% of tokens at 4 bits is **1.054 bits per original token — a
  15.2x compression of the bf16 cache — for 0.85% task failure** (1 of 117). For comparison, recency at 25%
  and bf16 (4.0 bits per original token, i.e. four times the storage) fails 69.2%.
- **Scope, unchanged from D36 and still binding.** One model, one synthetic retrieval probe, and a query-aware
  scorer that cannot run on a prefix cache shared across queries. The query-agnostic best sits at 0.624 at 25%
  and bf16, so none of these compression numbers transfer to the shared-cache regime as they stand.
