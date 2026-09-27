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

## D39 — KIVI at its own configuration, and what its residual window costs at moderate context (2026-09-21)

Every earlier comparison in this project ran against KIVI-*like* settings with `residual_window: 0`, which is a
weakened baseline. This arm runs KIVI as published — keys `group_token` 32 (the streaming variant, not the
context-wide `per_channel` D15 found worse and which KIVI does not use), values `group_channel` 32, residual
window 128 — on the same probe, samples and metric as every M4 arm.

| arm | bits / ORIGINAL token | task failure |
|---|---|---|
| **ours, joint: 25% of tokens @ 4-bit** | **1.054** | **0.0085** (1/117) |
| ours, 4-bit, all tokens | 4.260 | **0.0000** |
| **KIVI faithful, 2-bit** | **4.331** | **0.0513** (6/117) |
| ours, query-aware span 25% @ bf16 | 3.947 | 0.0256 |
| KIVI faithful, 4-bit | 6.127 | 0.0000 |

- **The joint point dominates the faithful baseline outright**: 4.1x less storage *and* six times fewer
  failures. This is the first comparison in the project made against another method at its own optimum rather
  than a convenient version of it, and it is the one the M5 baseline table should carry.
- **KIVI's fixed 128-token window makes its cost strongly context-dependent**, which a benchmark of long
  documents would not surface:

  | nominal context | KIVI 2-bit bits/orig | KIVI 2-bit failure | ours 4-bit bits/orig | ours 4-bit failure |
  |---|---|---|---|---|
  | 512 | **6.137** | 4.55% | 4.274 | 0% |
  | 1024 | **4.603** | 7.14% | 4.262 | 0% |
  | 2048 | 3.810 | 5.56% | 4.256 | 0% |
  | 4096 | 3.406 | 3.23% | 4.253 | 0% |

  **Below roughly 1500 tokens, KIVI 2-bit is strictly dominated by plain 4-bit with no window at all** — more
  expensive *and* worse on this probe. Above it the window pays for itself on storage while still costing
  answers here. Our configurations carry no residual window, so their cost is flat in length (4.274 to 4.253).
- **The predictor is validated against real data, not just against the encoder.** It predicted 3.809 bits per
  value for KIVI 2-bit at T = 2066; the measured mean at nominal 2048 is **3.810**. That is the closed-form
  budget agreeing with a full campaign, which is the point of having pinned it.
- **Scope, and it matters here more than usual.** KIVI was designed and evaluated for long-context workloads,
  and this is a 512–4096-token synthetic retrieval probe with one model. "KIVI 2-bit loses 3–7% of answers" is
  a statement about *this task at these lengths*, not about KIVI in general; its 4-bit configuration is lossless
  here, at 6.13 bits per original token. What generalises is narrower and sturdier: a fixed-size residual
  window has a cost that scales as 1/context, and a method that sets one must be evaluated at the context
  length it will actually see.

## D40 — The iso-budget frontier has an interior optimum at 3–4 bits, and it reconciles D38 with D16 (2026-09-21)

Two of this project's own findings pointed opposite ways at the same budget. D38: at ~1 bit per original token
more tokens at lower precision wins at every step. D16: 2-bit block-32 costs 12.8% of probe accuracy per token
where 4-bit block-128 is lossless. The allocator could not be written without resolving it, so the missing
points were measured rather than argued.

Five arms, all at ~1.05 bits per original token, query-aware span retention throughout, 117 samples:

| retention | bits | bits/orig | kept_target | task failure |
|---|---|---|---|---|
| 6.25% | bf16 | 0.967 | 0.915 | 0.1966 |
| 12.5% | 8 | 1.016 | 1.000 | 0.0684 |
| **25%** | **4** | **1.054** | 1.000 | **0.0085** |
| 31.5% | 3 | 1.059 | 1.000 | 0.0171 |
| 40.5% | 2 | 1.059 | 1.000 | 0.1368 |

| contrast | difference | p | discordant |
|---|---|---|---|
| 12.5% @ 8-bit vs 25% @ 4-bit (too few tokens) | +0.0598 [+0.0171, +0.1026] | **0.0082** | 7 |
| 40.5% @ 2-bit vs 25% @ 4-bit (too little precision) | +0.1282 [+0.0684, +0.1966] | **0.00027** | 17 |
| 40.5% @ 2-bit vs 31.5% @ 3-bit | +0.1197 [+0.0598, +0.1880] | **0.00047** | 16 |
| 31.5% @ 3-bit vs 25% @ 4-bit | +0.0085 [+0.0000, +0.0256] | 0.317 | 1 |

- **A clean U, significant on both arms.** The optimum is a **plateau at 3–4 bits** — those two are
  indistinguishable — with significant degradation from spending the budget on fewer tokens at higher
  precision (8-bit) *and* from spending it on more tokens at lower precision (2-bit).
- **Both earlier findings survive, in their own regimes.** D38's "more tokens wins" holds from bf16 down to
  4 bits, which is the whole range it measured. Below 4 bits D16's per-token damage takes over and the
  direction reverses. Neither was wrong; each was measured on one side of a turning point neither had crossed.
- **The mechanism is clean, and `kept_target` shows it.** The 2-bit and 3-bit arms retain **100%** of the
  answer tokens, so their damage is not lost content — it is per-token quantization error. The bf16 arm's
  damage *is* lost content (kept_target 0.915). The two failure modes sit on opposite sides of the optimum and
  are distinguishable in the diagnostic, not just in the outcome.
- **This is the allocator's switching rule, and it is measured rather than assumed.** Given a budget, spend it
  on tokens down to **4 bits and no further**; below that, precision is the binding constraint and extra tokens
  make things worse. That single number is what M5's coarse stage needs, and it could not have been derived
  from D16 or D38 alone.
- **Scope.** One model, one probe, one budget level (~1.05 bits per original token). Whether the turning point
  stays at 4 bits at other budgets is exactly the kind of thing that should be checked before the rule is
  stated generally, and the second model family is the right place to check it.

## D41 — H3: the joint policy beats every single-signal policy that can reach the budget, and one of them cannot reach it at all (2026-09-21)

H3 is the claim the project's title rests on: that allocating across three axes together beats varying one axis
and fixing the rest. Six arms at two budgets, **every config emitted by `scripts/emit_allocation_configs.py`
from the allocator's own `Allocation` object** rather than written by hand — so a gap between what the policy
decides and what the experiment runs cannot hide in a YAML file. 117 samples, paired.

| arm | retention | measured bits/orig | kept_target | task failure |
|---|---|---|---|---|
| **joint @ 1.05** (4-bit g128, query-aware spans) | 24.67% | 1.0401 | 1.000 | **0.0342** |
| retention-only @ 1.05 (bf16, query-aware spans) | 6.56% | 1.0116 | 0.949 | 0.1709 |
| recency-retention @ 1.05 (4-bit, recency tokens) | 24.67% | 1.0539 | 0.316 | 0.6923 |
| **joint @ 2.00** (4-bit g128, query-aware spans) | 46.98% | 1.9849 | 1.000 | **0.0000** |
| retention-only @ 2.00 (bf16, query-aware spans) | 12.50% | 1.9565 | 1.000 | 0.0684 |
| recency-retention @ 2.00 (4-bit, recency tokens) | 46.98% | 2.0011 | 0.371 | 0.6838 |
| precision-only, either budget | — | **infeasible** | — | not emitted |

| contrast (joint is the reference) | difference | p | Holm | discordant |
|---|---|---|---|---|
| 1.05 vs retention-only | −0.1368 [−0.2051, −0.0769] | 1.62e-04 | **3.2e-04** | 18 |
| 1.05 vs recency-retention | −0.6581 [−0.7436, −0.5726] | 1.71e-18 | **5.1e-18** | 77 |
| 2.00 vs retention-only | −0.0684 [−0.1197, −0.0256] | 4.68e-03 | **4.7e-03** | 8 |
| 2.00 vs recency-retention | −0.6838 [−0.7607, −0.5983] | 3.74e-19 | **1.5e-18** | 80 |

- **H3 is supported on this model family, and all four contrasts survive Holm correction.** Every single-signal
  policy *run* that can reach these budgets is beaten at both of them. *(Scoped by D42: the plan pre-registers
  **two model families** and a **magnitude-only** baseline that this campaign did not include, so "supported"
  here means supported on one family against three of four pre-registered baselines. The general claim is not
  yet made.)*
- **Precision-only cannot reach either budget, and that is the cleanest result here.** Quantization alone
  bottoms out at 2-bit, about 2.63 bits per original token once its fp16 scales are counted, so the policy the
  quantization literature uses does not exist at 1.05 or 2.00. The allocator returns `None` rather than
  overspending and calling the result a comparison (§44). H3's first answer is an absence, not a number.
- **The comparison against recency-retention is NOT independent evidence for joint allocation, and must not be
  reported as if it were.** Read `kept_target` first: recency keeps 31.6% of the answer tokens against the joint
  arm's 100%, and by depth it fails 1.0000 on shallow targets and 0.0270 on deep ones, which its window keeps
  for free. The 66-point margin is the query-awareness effect **D36 already measured**, re-measured at a fixed
  budget. The contrast that isolates what H3 is actually about is **joint vs retention-only**, where both arms
  use query-aware span retention and differ only in whether the budget buys bits or tokens — and that margin is
  13.7 points at 1.05 and 6.8 at 2.00, not 66.

### The iso-budget check fails by 1.5–2.8% against retention-only, in the winning arm's favour

| pair | joint | opponent | deviation |
|---|---|---|---|
| 1.05 vs retention-only | 1.0401 | 1.0116 | joint spends **+2.8%** |
| 1.05 vs recency-retention | 1.0401 | 1.0539 | joint spends −1.3% |
| 2.00 vs retention-only | 1.9849 | 1.9565 | joint spends **+1.45%** |
| 2.00 vs recency-retention | 1.9849 | 2.0011 | joint spends −0.8% |

- **Mechanism.** Span retention keeps whole spans, so the realised ratio undershoots the requested one, and it
  undershoots *more* at a small ratio because spans are coarse relative to a tiny budget: 0.0632 realised
  against 0.0656 requested (−3.7%) versus 0.2438 against 0.2467 (−1.2%). Nothing is mis-accounted; the
  allocator's `bits_per_original_token` is exact for the ratio it asked for, and the *experiment* lands below it.
- **This breaks the ±1% rule the project set for itself, and in the direction that flatters the conclusion.**
  Against recency-retention the deviation runs the other way and the comparison is conservative; against
  retention-only it does not. Named here rather than left for a reviewer.
- **Sensitivity estimate, labelled as an estimate and not a measurement.** D40's own bf16 frontier moved 2.6
  points of task failure for a 5% relative change in retention (0.1966 at 6.25% to 0.1709 at 6.56%), so closing
  a 2.8% budget gap is worth roughly **1.4 points** against a 13.7-point margin. The ordering is not at risk,
  but the number is.
- **Fix before this goes in a paper:** emit the single-signal arms by *realised* bits rather than requested
  ratio — a solve for the ratio whose retained-span count lands on the joint arm's measured
  `bits_per_original_token`. That closes the loop properly instead of arguing about the size of the gap.

### Reconciling D39's joint point (0.85%) with H3's joint arm (3.42%)

These read as the same configuration in two tables and are not: `m4y_q25_4bit` keeps **25.00%** of tokens,
`m5h3_1p05_joint` keeps **24.67%**, at identical 4-bit quantization, so the H3 arm has a 1.3% smaller realised
budget (1.0401 against 1.0543). Paired over the same 117 samples:

    m4y_q25_4bit 0.0085  vs  m5h3_1p05_joint 0.0342
    difference -0.0256 [-0.0598, +0.0000]   p = 0.083   3 discordant

Not significant, and the interval touches zero — 1 event against 4 out of 117. Measured rather than argued,
because a reviewer comparing the two tables would otherwise see 0.85% and 3.42% for what looks like one row.
It also doubles as a small replication: the joint point reproduces at a tighter budget with a difference below
what this design can resolve. Both numbers stay in the record, each labelled with the arm that produced it.

### The emitter found a real coupling between two axes the project had treated as separable

Both retention-only configs were **refused by the config schema** on first emission:

```
Value error, protection guards the quantized representation, so quantization must be configured
```

- **The allocator was wrong and the schema was right.** `_protection()` ran as a lookup on the error rate
  before anything knew whether the policy would quantize, so it attached the +0.075% metadata range guard
  (D30) to a bf16 cache that has no quantization metadata to guard.
- **The fix is a finding, not a patch.** Every protection mechanism M3 selected acts on **quantization
  metadata** (D25–D30). So a retention-only policy is *unprotectable by the cheapest mechanism this project
  has*, and its only remaining option is uniform data ECC at +14%, which D25 measured as never worth its bits.
  This is the **one place where the three axes are genuinely coupled rather than separable** — and it is an
  argument for joint allocation that the brief did not anticipate and that no experiment was designed to find.
- `_protection_for(bits, ber)` now returns no guard and no codec when `bits is None`, carrying that sentence as
  its rationale so it appears in the emitted config's header. Four tests pin it, including two that check the
  fix did not silently disarm protection on the policies that *do* quantize.
- **Process note.** This is the second time a schema validator has caught a scientific error rather than a
  typing error (D24's provenance check was the first). Both were rules written to stop a class of mistake and
  both fired on something the author believed was correct.

### Scope

One model, one probe, two budget levels. The joint-vs-retention-only margin is the load-bearing one and it is
13.7 and 6.8 points on 18 and 8 discordant samples respectively — real, Holm-surviving, and not large. H3 is
recorded as **supported with a stated iso-budget deviation**, not as a clean result.

## D42 — A pre-registration audit that scopes D41, an iso-budget deviation measured rather than estimated, and a latent provenance bug (2026-09-22)

D41 closed H3 a day earlier. Writing the external-review dossier meant reading `docs/m4_m5_plan.md` §M5 line by
line, and the plan says things D41 did not check itself against. Nothing below is new data about the model; it
is an audit of a claim against its own pre-registration, and it makes D41's "supported" narrower.

### 1. The iso-budget deviation is now measured, and the estimate was wrong in both directions

D41 reported the joint arm spending **+2.8%** and **+1.45%** more than retention-only — outside the project's
±1% rule, in the winning arm's favour — and estimated the correction at "roughly 1.4 points" from D40's bf16
frontier. `scripts/calibrate_iso_budget.py` re-emits a short arm at the ratio whose *realised* budget matches
its reference, and the arms were re-run:

| pair | before | after | residual | margin before | margin after | correction |
|---|---|---|---|---|---|---|
| 1.05 vs retention-only | 1.0116 (−2.75%) | 1.0409 | **+0.07%** | 0.1368 | **0.1282** | 0.86 pts |
| 2.00 vs retention-only | 1.9565 (−1.43%) | 1.9882 | **+0.17%** | 0.0684 | **0.0513** | 1.71 pts |

- **The single estimate was wrong in both directions**: it overstated the 1.05 correction (0.86 actual against
  1.4 estimated) and understated the 2.00 one (1.71). The span-coarseness mechanism predicts the *smaller*
  budget suffers more, and the correction came out larger at the larger budget — unexplained, stated, and not
  worth a story on two points.
- **Both results survive**: −0.1282 [−0.1966, −0.0684], p = 2.7e-04, 17 discordant; and −0.0513 [−0.0940,
  −0.0171], p = 1.4e-02, 6 discordant.
- **The correction is deliberately one-directional.** The recency arms sit at **+1.33%** and **+0.82%**, also
  outside a symmetric ±1% rule, and are left alone: pulling a baseline *down* to satisfy a tolerance is shaving
  its budget to make it lose by more, which is the manipulation the rule exists to prevent. `decide()` is one
  function with three outcomes and `tests/test_iso_calibration.py` pins the asymmetry, because an asymmetry
  nobody tests gets tidied into symmetry by the next person who reads it.

### 2. Two pre-registered conditions D41 did not check itself against

`docs/m4_m5_plan.md` §M5 lists four baselines and a refutation rule. The audit:

| pre-registered | status |
|---|---|
| uniform 4-bit, all tokens, metadata protected | ✅ `m4x_all4bit`: 4.2600 bits, **0.0000** failure. The joint point wins on bits (4.0×), not on failure — a Pareto comparison, and it must be reported as one |
| KIVI at its own optimum, residual window 128 | ✅ D39 |
| recency-only retention at the matched budget | ✅ D41 |
| **magnitude-only allocation, "the strongest single-signal policy"** | ❌ **never run.** Now run, below |
| **"on at least two model families", else H3 is refuted** | ❌ **one family.** Qwen3-1.7B only |

**So D41's "H3 is supported" was an overclaim** — not because the data was wrong, but because the claim was
never held against the rule that was frozen before it. The honest statement is: *supported on one model family
against the baselines actually run.* The general claim is not made, and by the plan's own wording H3 cannot be
called supported until a second family answers. This is the twelfth finding in the project withdrawn or
re-scoped after a check, and the first one caught by re-reading the pre-registration rather than by a control
arm.

### 3. Magnitude-only, the baseline the plan called strongest, is the weakest thing measured

| arm | bits/orig | kept_target | task failure | vs joint |
|---|---|---|---|---|
| magnitude-only @ 1.05 (key_norm, tokens, 4-bit) | 1.0539 (+1.33%) | 0.2128 | **0.9573** | −0.9231 [−0.9658, −0.8718], p = 2.7e-25, 108 discordant |
| magnitude-only @ 2.00 | 2.0011 (+0.82%) | 0.4984 | **0.8376** | −0.8376 [−0.8974, −0.7692], p = 4.2e-23, 98 discordant |

- **It is worse than recency** (0.9573 against 0.6923 at 1.05), which is the trivial baseline recency was
  supposed to be. The plan's expectation that magnitude is "the strongest single-signal policy" is refuted.
- **Key magnitude carries no usable signal about which tokens the answer needs.** Against the share a uniformly
  random selection of the same size would keep in expectation: **−0.0339, p = 0.077** at 1.05 and **+0.0286,
  p = 0.17** at 2.00. Indistinguishable from chance at both budgets, and the two trends point *opposite* ways,
  which supports "no signal" over "anti-signal". Tested rather than eyeballed: the 0.2128-against-0.2467 gap
  looked like a below-chance result and is not one.
- **Why it matters beyond H3.** Magnitude is computable at prefill, so it is the only H3 baseline a *shared
  prefix cache* could actually run (D36's open problem). That the cheap query-agnostic signal is at chance
  makes the 0.624-against-0.026 gap harder, not easier.

**All six contrasts survive Holm across the family** (1.6e-24, 2.1e-22, 1.5e-18, 5.1e-18, 5.5e-04, 1.4e-02).

### 4. A `protection_arm` config silently accepts a `compression` block it never applies

Found while designing the fault×retention experiment (`docs/m4_exit.md` row 10). `ProtectionArmTask.run_unit`
goes prefill → quantize → protect → decode with **no compression step**, but the schema accepts a
`compression` block on that task kind, the block survives validation into the config object, and
`_pruning_token` reads it unconditionally. A run would therefore be **named `…_p25qaS_…` and hashed as pruned
while executing unpruned** — a manifest that is internally consistent and wrong.

This is D24's failure mode arrived at from the opposite direction: D24 was a manifest recording code that did
not run, this is a manifest recording a *transform* that did not run. Nothing in the repository has hit it,
because no such config was ever written. **The schema must refuse the combination until the task implements
it**; deferred to the next commit because two campaigns were in flight and `config/` and `experiments/` are
inside the runner's import graph.

### 5. D24's narrowing, applied before it was verified

`raq_kv.allocation.policy` was edited while two arms were in flight, on the reasoning that the runner does not
import it. That reasoning is correct — verified afterwards, and each manifest records the tree state its own
process imported, so nothing is corrupted. But **it was acted on before it was checked**, which is how D24
happened in the first place. `tests/test_runner_import_surface.py` now pins the exemption: the modules declared
safe to edit mid-campaign must stay outside the runner's import graph, and a test fails if an import is added.
An exemption that lives only in an argument decays.

## D43 — The reliability tax is NOT flat in retention: pruning concentrates risk, and the allocator's first stage assumes otherwise (2026-09-23)

Row 10 of `docs/m4_exit.md`, and the first experiment in this project to vary **two axes at once**. Until now
protection was priced on unpruned caches (M3) and retention on fault-free ones (M4, M5, all 30 runs `nofault`),
so the allocator's staged design — `_protection()` is a lookup on the bit error rate with **no retention term** —
rested on an assumption nothing had tested.

Five arms, 4-bit keys `group_token` 128 / values `per_token`, query-aware span retention at 24.67%, BER 1e-6,
60 samples × 8 trials + 1 no-fault control (all controls exactly zero damage). Paired over the **56 of 60**
samples every arm answers correctly with no faults *and* that the full cache also answers, so a sample pruning
alone breaks is never charged to the fault model.

| arm | stored | flips/trial | task failure |
|---|---|---|---|
| unpruned, unprotected | 540.5 Mbit | 540 | 0.3125 |
| pruned 25%, unprotected (matched **rate**) | 132.6 Mbit | 132 | 0.1875 |
| pruned 25%, unprotected (matched **exposure**) | 132.6 Mbit | **540** | **0.4018** |
| unpruned, +0.075% range guard | 540.5 Mbit | 540 | **0.0000** |
| pruned 25%, +0.075% range guard | 132.6 Mbit | 132 | 0.0268 |

| contrast | difference | p | discordant | pre-registered rule |
|---|---|---|---|---|
| 1. matched **rate**: pruning helps | −0.1250 [−0.1942, −0.0558] | **6.1e-04** | 45 | **met** |
| 2. matched **exposure**: pruning hurts | **+0.0893 [+0.0223, +0.1562]** | **0.016** | 42 | **met** |
| 3. same pruned cache, 132 vs 540 flips | +0.2143 [+0.1652, +0.2656] | 5.1e-09 | 44 | met |
| 4. guard, pruned vs unpruned | +0.0268 [+0.0022, +0.0625] | 0.066 | **4** | **NOT met** |

**Reproduce with** `scripts/report_fault_retention.py`, which is the canonical path for these numbers. The first
version of this analysis was throwaway shell snippets, which is not a provenance for a structural claim; writing
the script caught that the first pass had used the `answer_changed` proxy rather than `task_failure` as the
endpoint.

**The script itself then had to be fixed twice, and both bugs are the kind that produce numbers nobody can
reproduce.** (1) Its composition control tested target names by substring, so `"scale"` also matched
`check:scale`: a metadata-SECDED arm looked as though 8.4% of its flips hit metadata against a guarded arm's
6.2%, and the control cried "composition moved" on a comparison where nothing *vulnerable* had moved — the extra
bits are SECDED's own check bits. Payload metadata and protection overhead are now counted separately. (2) All
contrasts shared one bootstrap generator, so a contrast's interval depended on **how many other contrasts were
requested in the same command**; the same comparison returned [+0.0223, +0.1562] alone and [+0.0246, +0.1607]
third in a batch. Each contrast now draws from a generator keyed by its own arm names, so its interval is a
function of the data and `--seed` only. The intervals in the table above are the fixed script's, and they match
the original shell values — an intermediate revision of this record briefly quoted the batch-order artefacts and
has been corrected.

### The finding, and the arithmetic it had to be separated from

**Contrast 1 is not a result, and the config header said so before the run.** A pruned cache stores a quarter of
the bits, so at a matched *rate* it takes a quarter of the flips — 132 against 540. "Pruning improved
reliability" is then arithmetic about exposure, not a statement about importance. This is exactly why
`ber_basis: unpruned` exists.

**Contrast 2 is the result.** Give the pruned cache the same **absolute** number of flips the unpruned one takes
on that same sample, and it fails **more**: 0.4018 against 0.3125, +0.0893 [+0.0223, +0.1562], p = 0.016 on 42
discordant samples. **A flip that lands in a pruned cache is more likely to cost an answer.** The mechanism is
the one D34/D36 make obvious in hindsight: query-aware span retention keeps *exactly* the answer-bearing spans
(`kept_fraction_target` = 1.000), so every scale that survives pruning governs a group the answer depends on,
where most groups in an unpruned cache govern filler.

**Contrast 3 explains why contrast 1 looks so good.** Damage is strongly **sublinear** in flip count: 4.1× the
flips buys only 2.1× the failure (0.1875 → 0.4018), because an answer already broken cannot break further. So
the exposure drop pays off more than proportionally, which is what lets the concentration effect hide inside a
matched-rate comparison.

### The compositional alternative, checked and ruled out

Contrast 2's arms differ in more than pruning: the pruned cache also holds ~4× fewer quantization groups, so a
reviewer can reasonably ask whether its flips land in **metadata** more often than the unpruned arm's. Metadata
is at least 1543× more damaging per bit (D25/D28), so even a small shift in where flips land would explain part
of +0.0893 without any appeal to importance. Both arms record `storage_composition` and per-trial `by_target`,
so this is measured rather than argued:

| | stored bits | metadata share of **stored bits** | metadata share of **flips** |
|---|---|---|---|
| unpruned | 540.5 Mbit | 6.008% | **6.1192%** |
| pruned, matched exposure | 132.6 Mbit | 6.002% | **6.1468%** |

Paired per sample, the difference in metadata share of flips is **+0.028 percentage points, p = 0.57** — no
effect. The reason is structural and worth stating: with keys on `group_token` at group size 128 and values
`per_token`, the group count scales with the token count exactly as the code count does, so **pruning leaves the
code/metadata composition invariant**. Both arms take 6.1% of their flips in metadata, 4324 flips per sample.

So the +0.0893 is not compositional. **The mechanism sentence stands as written:** at equal exposure and equal
composition, a flip in a pruned cache is more likely to cost an answer because the groups that survived pruning
are the ones the answer depends on. Note the scope this control also creates: the invariance is a property of
token-proportional axes. An arm quantizing on `per_channel` or `per_tensor`, where group count does **not** scale
with tokens, would shift composition under pruning and this control would have to be rerun.

### What this does to the allocator

**The staged design's first assumption is falsified as a mechanism.** Protection is *not* separable from
retention: at equal exposure, how aggressively the cache was pruned changes how much a fault costs. §6's
multiplicative form, which this project rejected on four grounds (D22, D25–D31, D18, D32–D34), was right about
*this* one thing — the axes do interact, just not through a product of per-cell scores.

**But the practical rule survives, and saying otherwise would overstate the finding.** At the matched BER a real
deployment actually faces, pruning still *reduces* absolute failure (0.3125 → 0.1875, contrast 1), because the
exposure drop more than covers the concentration. And the guard still does most of its job: 0.0268 pruned
against 0.3125 unprotected.

**What changed in the record, and what deliberately did not.** `_protection()` keeps its constants. The
measurement that would justify moving them has not been made: contrast 4 — the guard reaching exactly 0.0000
unpruned and 0.0268 pruned — has only **4 discordant samples** and p = 0.066, so it **does not meet the
pre-registered rule** (p < 0.05 *and* a CI excluding zero, D31). The direction is consistent with concentration
and is reported as a direction, not a result. Inventing a retention-dependent BER threshold from four events is
exactly the mistake D31 was written after.

**The honest statement of the allocator's status:** its protection lookup is **unconservative for aggressively
pruned caches** by an amount this experiment bounds but does not resolve. A deployment pruning to 25% should not
assume the +0.075% guard buys it the clean zero that M3 measured on a full cache.

### The experiment that would set the constant

Contrast 4 at a rate where the guard has room to fail. At 1e-6 the unpruned arm is at zero, so the comparison
has almost no discordant samples by construction — the same both-at-zero problem D30 hit. D29/D30 located the
guard's breakdown at **1e-5**, where unpruned reaches 5.93%. Repeating contrast 4 at 1e-5 is where a
retention-dependent threshold becomes measurable, and it is one arm pair.

### Scope

One model, one probe, one rate, one retention ratio, one quantization configuration. The concentration effect is
measured at 24.67% retention with a query-aware span scorer; whether it scales with the retention ratio, and
whether a query-*agnostic* scorer concentrates risk less (it keeps fewer answer tokens, so it should), is
unmeasured. **M4 closes with this row; M5's allocator carries a stated structural caveat rather than a fix.**

## D44 — At 1e-5 the guard survives pruning, escalation buys nothing, and D43's headline could not be replicated because both arms hit the ceiling (2026-09-23)

D43 left one measurement named as the next thing to do: repeat the guard contrast at 1e-5, where D29/D30 located
the guard's breakdown and the unpruned arm is not pinned at zero. Six arms were run at 1e-5. The answer to the
allocator's question is clean; **the attempt to replicate D43's own headline failed, for a diagnosable and
uninteresting reason, and that failure is the most important thing in this record.**

All arms: 4-bit keys `group_token` 128 / values `per_token`, query-aware span retention at 24.67% where pruned,
60 samples × 8 trials + 1 no-fault control (every control exactly zero damage), paired over 56 shared samples.
Canonical numbers from `scripts/report_fault_retention.py`.

| arm | flips/trial | task failure |
|---|---|---|
| unpruned, range guard | 5405 | **0.0603** |
| pruned 25%, range guard (matched rate) | 1326 | 0.0312 |
| pruned 25%, range guard (matched exposure) | 5405 | 0.0737 |
| pruned 25%, metadata SECDED (matched rate) | 1356 | 0.0290 |
| unpruned, unprotected | 5405 | **0.9420** |
| pruned 25%, unprotected (matched exposure) | 5405 | **0.9621** |

### 1. The guard replicates D29/D30 through a new code path

Unpruned range guard at 1e-5 reaches **0.0603**, against the **0.0593** D29/D30 measured. That is a clean
replication and it is load-bearing for everything below: these arms run through the pruning-capable
`ProtectionArm` added in D43, under **eager** attention rather than sdpa, generating their own golden. If the new
path had changed what the guard does, this is where it would show.

### 2. Under the guard, pruning has no measurable effect — and the direction is not even consistent

| rate | contrast | difference | p | discordant |
|---|---|---|---|---|
| 1e-6 | guard, pruned vs unpruned | +0.0268 [+0.0022, +0.0625] | 0.066 | 4 |
| 1e-5 | guard, pruned vs unpruned (matched rate) | **−0.0290** [−0.0714, +0.0134] | 0.059 | 25 |
| 1e-5 | guard, pruned vs unpruned (matched exposure) | +0.0134 [−0.0312, +0.0603] | **0.71** | 29 |

**The sign flips between rates and neither contrast meets the rule.** At 1e-6 the pruned guarded arm looked
slightly worse on 4 discordant samples; at 1e-5, on 25, it is better. Under the guard there is no consistent
retention effect to report in either direction.

### 3. Escalating to metadata SECDED buys nothing on a pruned cache at this rate

Pruned guard 0.0312 against pruned SECDED 0.0290: **−0.0022 [−0.0201, +0.0156], p = 0.91, 6 discordant.** On an
*unpruned* cache at the same rate the escalation was decisive (D30: +0.0593, p = 1.1e-5). The coherent reading is
that the guard's breakdown is driven by **flip count**, and a pruned cache at a matched rate takes 4× fewer
flips, so it sits below the breakdown. Pruning *delays* the guard's failure in rate terms rather than hastening it.

### 4. **D43's caveat in `_protection()` was wrong, and is corrected**

D43 wrote, on the strength of the 1e-6 guard contrast, that the protection lookup is *"unconservative for
aggressively pruned caches"*. Rows 2 and 3 do not support that: at 1e-5 the pruned guarded arm is **better** than
the unpruned one, and escalation adds nothing. The caveat took a direction resting on **4 discordant samples**
more seriously than the project's own rule allows — the error D31 exists to prevent, committed in the same
decision record that cited D31 for refusing to move the constants. The docstring and its test now say what is
actually established:

* the concentration effect is real on an **unprotected** cache at 1e-6 (D43 contrast 2);
* under the metadata guard it is **not measurable at either rate**, with inconsistent sign;
* so whether protection needs a retention term is **open in both directions**, not "unconservative".

### 5. The replication of D43's headline failed on saturation, which is not a null result

The direct test — unprotected, matched exposure, 1e-5 — gives +0.0201 [−0.0022, +0.0446], p = 0.097, 14
discordant. But look at the levels: **0.9420 and 0.9621.** Unprotected at 1e-5 destroys nearly every answer in
both arms, so there is no room to measure a difference. This is D30's both-arms-pinned problem arriving from the
**ceiling** instead of the floor, and the design error is mine: 1e-5 was chosen because it is where the *guard*
breaks, which is a different question from where the *effect* is measurable.

**It is not evidence against D43 and must not be reported as such.** The direction is positive, matching 1e-6,
and 14 discordant samples at a 94% base rate is an uninformative test. D43's headline therefore remains a
**single-rate result** until it is tested where the unprotected arm has room.

### 6. What is running, and why that rate

3e-7, on both unprotected arms at matched exposure. M3's calibration sweep — measured on a different set than the
one that suggested the effect, which is the D31 rule — puts the unprotected arm at **12.5%** there: a different
decade from 1e-6 and well clear of both floor and ceiling. Both outcomes are pre-committed in the config headers:
a positive difference replicates D43 across a decade; a null **with room to measure** narrows D43 to one rate and
is the thirteenth withdrawal.

### Scope

One model, one probe, one retention ratio, one quantization configuration. Rows 2 and 3 are matched-rate
comparisons except where stated, which is the deployment condition; row 5's failure is about the rate chosen, not
about the effect.

## D45 — The concentration effect replicates across a decade, and the 1e-5 null was saturation exactly as diagnosed (2026-09-23)

D44 named the measurement: repeat D43's headline contrast at a rate where the unprotected arm has room. 3e-7 was
chosen from M3's calibration sweep — measured on a different set than the one that suggested the effect, which is
the D31 rule — where the unprotected arm sits near 12.5%. Both outcomes were pre-committed in the config headers
before the run.

**It replicates, and more strongly than at 1e-6.** Unprotected arms, matched exposure, 56 paired samples:

| rate | flips/trial | unpruned | pruned 25% | absolute difference | risk ratio | p | discordant |
|---|---|---|---|---|---|---|---|
| **3e-7** | 162 | **0.1116** | **0.2232** | **+0.1116 [+0.0536, +0.1719]** | **2.000** | **4.6e-04** | 43 |
| **1e-6** | 541 | 0.3125 | 0.4018 | +0.0893 [+0.0223, +0.1562] | 1.286 | 0.016 | 42 |
| 1e-5 | 5405 | 0.9420 | 0.9621 | +0.0201 [−0.0022, +0.0446] | 1.021 | 0.097 | 14 |

- **Two rates, a decade apart, both meeting the pre-registered rule** (p < 0.05 *and* a CI excluding zero). The
  concentration effect is no longer a single-rate result.
- **The rate choice validated itself.** The unpruned arm landed at 0.1116 against the 12.5% the calibration sweep
  predicted, which is the check that makes the choice a prediction rather than a search for a friendly rate.
- **The 1e-5 null is explained, not excused.** The risk ratio falls monotonically — **2.00 → 1.29 → 1.02** — as
  the unpruned base rate rises from 11% to 31% to 94%. That is what saturation does: once nearly every answer is
  destroyed in both arms, no multiplicative difference can express itself as an absolute one. D44 diagnosed this
  from the levels before this run existed, and the diagnosis held.
- **The composition control passes at every rate.** Payload-metadata share of flips 6.1988% → 6.2654% at 3e-7,
  p = 0.69 (and 6.1192% → 6.1468%, p = 0.58 at 1e-6), so the effect is not a shift in *where* flips land. As at
   1e-6, this is structural: `group_token` group count scales with the token count exactly as the code count does.
- **The direction is the same at all three rates**, including the saturated one.

### What now stands, and what still does not

**Stands:** a fault landing in a pruned cache is about **twice** as likely to cost an answer at low exposure, and
the mechanism is that query-aware span retention keeps exactly the answer-bearing spans (`kept_target` = 1.000 in
every pruned arm here), so every surviving scale governs a group the answer depends on. Protection is **not
separable from retention** on an unprotected cache, replicated across a decade.

**Still does not:** whether the *allocator* needs a retention term. Under the metadata guard the effect remains
unmeasurable at both rates tested with an inconsistent sign (D44), and the constants are unchanged. The natural
next measurement is the guard contrast at **3e-7** — the rate that just proved to be where this effect is
visible — rather than at 1e-5, where D44 looked because that is where the guard breaks. Those are two different
questions and conflating them is the error D44 recorded.

### Scope

One model, one probe, one retention ratio (24.67%), one quantization configuration, one scorer. Whether the risk
ratio scales with the retention ratio is unmeasured, and a query-*agnostic* scorer should concentrate risk less
because it keeps fewer answer tokens — that is a prediction the project has not tested.

## D46 — RAQ-Probe's `tool_arg` fact type encodes one family's tool-calling dialect, and does not transfer (2026-09-23)

The cross-family replication H3 pre-registers needs Llama-3.2-3B to answer RAQ-Probe at all. It does, except for
one fact type, and the exception is a **probe limitation rather than a model one**.

Golden RAQ-Probe, Llama-3.2-3B-Instruct, 720 samples:

| fact type | accuracy |
|---|---|
| id | 0.9917 |
| date | 0.9750 |
| number | 0.9750 |
| constraint | 0.9667 |
| entity | 0.9333 |
| **tool_arg** | **0.0000** (0 of 120, every context length) |
| overall | 0.8069 |
| **excluding `tool_arg`** | **0.9683** (against Qwen3's 0.983 — comparable) |

**Llama retrieves the right value every time and is scored wrong for its format.** The probe's gold text is
`<tool_call>\n{"name": …, "arguments": {…}}\n</tool_call>` — Qwen's convention — and `parse_tool_calls` requires
the `<tool_call>` tags. Llama emits bare JSON using `parameters` instead of `arguments`:

```
ref  gold=93259021  tool_name=get_account_balance  arg_name=account_number
pred {"name": "get_account_balance", "parameters": {"account_number": "93259021"}}
```

The fact is retrieved. The dialect differs.

### Why the scorer was not broadened

The obvious fix — accept untagged JSON and treat `parameters` as an alias for `arguments` — was written and
**tested against every stored Qwen tool_arg prediction before being considered**: 492 predictions re-scored,
**2 changed**. Not a safe superset. Both flips are *truncated generations* in a degraded 2-bit arm, where the
closing `</tool_call>` was cut off by `max_new_tokens` and the bare-JSON fallback rescued it — arguably the
current scorer is right to count a malformed output as wrong when the measurement is about damage.

Adopting it anyway would mean Llama and Qwen were scored by **different code**, since stored `correct` values are
computed at run time. Making that consistent would require re-running every Qwen arm. Changing the measurement
protocol (S0) mid-project so that a second family passes is exactly the move that destroys cross-run
comparability, so it was not made.

### What is done instead

`tool_arg` is **excluded from the cross-family comparison on both families**. On Llama the exclusion is automatic
— every `tool_arg` sample has `golden_correct = False`, and every paired analysis in this project conditions on
that — but Qwen's comparisons must be restricted to the same five fact types or the two families would be
compared on different compositions. That restriction is an analysis-side filter; no dataset or config changes,
so no identity moves.

### The finding this leaves behind

**The probe's only agent-shaped fact type is family-specific.** `tool_arg` is the closest thing RAQ-Probe has to
an agent task, and it turns out to measure adherence to one model family's tool-call formatting as much as
retrieval. That is worth stating against **C4** — "token-level metrics under-estimate agent failures" — whose
agent evidence was already thin and now has one less usable vehicle. A probe intended to generalize across
families should score tool calls by *what was called with what*, not by the syntax used to say it; the current
scorer is kept because changing it now costs more than it buys, not because it is right.

### Scope

One alternative family. Whether other families (Mistral, Gemma) emit yet other dialects is unmeasured, but the
mechanism — a probe encoding the convention of the model it was developed against — predicts they would.

## D47 — H3 on a second model family: decisive against two baselines, partial against the third, and the verdict depends on a family size nobody pre-registered (2026-09-23)

D42 recorded that D41's "H3 is supported" was an overclaim, because the plan pre-registers **two model families**
and only one had been run. Llama-3.2-3B-Instruct was run. Its KV geometry is identical to Qwen3-1.7B's — verified
empirically, both 57344 values per token — so the allocator's cost table transfers unchanged and only the family
varies. Per D46, `tool_arg` is excluded from **both** families, so the two are compared on the same five fact
types; that restriction alone changes Qwen's numbers materially and is not cosmetic.

Twelve contrasts: 2 families × 2 budgets × 3 baselines. Iso-budget enforced by measurement; every baseline spends
at or above the joint arm it is compared to. `precision_only` emits nothing at either budget on either family.

| family | budget | baseline | joint | baseline | p | Holm/12 | discordant | verdict |
|---|---|---|---|---|---|---|---|---|
| Qwen | 1.05 | retention-only | 0.0101 | 0.0707 | 0.034 | **0.10** | 8 | **fails Holm at 12** |
| Qwen | 1.05 | magnitude-only | 0.0101 | 0.9495 | 5.2e-22 | 5.2e-21 | 93 | meets |
| Qwen | 1.05 | recency | 0.0101 | 0.6364 | 3.4e-15 | 2.4e-14 | 62 | meets |
| Qwen | 2.00 | retention-only | 0.0000 | 0.0000 | 1 | 1 | **0** | **bound: both at the floor** |
| Qwen | 2.00 | magnitude-only | 0.0000 | 0.8081 | 3.7e-19 | 3.4e-18 | 80 | meets |
| Qwen | 2.00 | recency | 0.0000 | 0.6263 | 3.4e-15 | 2.4e-14 | 62 | meets |
| **Llama** | 1.05 | retention-only | **0.0000** | 0.1386 | 1.8e-04 | **7.3e-04** | 14 | **meets** |
| Llama | 1.05 | magnitude-only | 0.0000 | 1.0000 | 1.5e-23 | 1.8e-22 | 100 | meets |
| Llama | 1.05 | recency | 0.0000 | 0.6400 | 1.2e-15 | 1.0e-14 | 64 | meets |
| Llama | 2.00 | retention-only | 0.0000 | 0.0099 | 0.32 | 0.63 | **1** | **bound: both at the floor** |
| Llama | 2.00 | magnitude-only | 0.0000 | 0.9900 | 2.5e-23 | 2.8e-22 | 99 | meets |
| Llama | 2.00 | recency | 0.0000 | 0.6100 | 5.7e-15 | 2.9e-14 | 61 | meets |

**9 of 12 survive Holm across the whole family of 12.**

### What stands

**Against magnitude-only and recency-retention: 8 of 8 contrasts, two families, two budgets, all surviving Holm
at 12.** That is as clean as this project has produced.

**On Llama the joint arm is perfect** — 0 of 101 samples fail at 1.05 bits per original token, against Qwen's 1 of
99. The cross-family replication is not marginal on the joint policy's own performance.

### What does not stand, and the part that is uncomfortable

**Against retention-only — the only baseline that isolates the precision axis — the rule is met on one family at
one budget.** Llama at 1.05 meets it decisively (Holm 7.3e-04). The other three do not:

* **Qwen at 1.05 fails Holm at a family of 12** (p = 0.034 → 0.10). Within a family of 3 — one model, one budget —
  it passes at 0.034. **The plan pre-registered "Holm-corrected p < 0.05" without defining the family**, so both
  readings are defensible and the conservative one is taken as the headline. Recording this rather than picking
  the favourable family size is the whole point.
* **Both 2.00 contrasts are bounds, not wins.** The joint and retention-only arms both reach the floor (0.0000 vs
  0.0000 on Qwen, 0 discordant; 0.0000 vs 0.0099 on Llama, 1 discordant). D30's rule applies: two arms at the
  floor are an upper bound on the difference, never a demonstrated equality *or* a win.

**So H3's pre-registered refutation rule — "beats every single-signal policy at equal bits on at least two model
families" — is NOT met as written.** It is met against two of three baselines on both families, and against the
third on one family at one budget. H3's status becomes **partially supported, with the unmet half named**, not
"supported".

### D46's restriction is why Qwen's numbers moved, and it moved them against us

On the full six fact types Qwen's 1.05 retention-only contrast was −0.1282 at p = 2.7e-04 (D42). On the five-type
composition it is −0.0606 at p = 0.034. Dropping `tool_arg` removed a large share of where the retention-only arm
was failing, so the contrast that carried H3 on Qwen is substantially weaker once the two families are compared
on the same footing. The weaker number is the honest one: the stronger one was measured on a composition the
second family cannot be scored on.

### The fourteenth withdrawal: key magnitude is an anti-signal, not "no signal"

D42 concluded that key magnitude selects answer-bearing tokens **at chance**, from −0.0339 (p = 0.077) and
+0.0286 (p = 0.17) on Qwen, and explicitly argued that the opposite signs supported "no signal" over
"anti-signal". On the five-type composition, with the second family:

| arm | kept_target | chance | delta | p |
|---|---|---|---|---|
| Llama 1.05 | 0.0709 | 0.2467 | **−0.1759** | **1.4e-08** |
| Llama 2.00 | 0.1645 | 0.4698 | **−0.3053** | **1.4e-12** |
| Qwen 1.05 | 0.1984 | 0.2467 | −0.0483 | **0.013** |
| Qwen 2.00 | 0.4388 | 0.4698 | −0.0310 | 0.73 |

**All four point the same way, three of four are significant, and the "opposite signs" that justified "no signal"
were an artefact of the composition.** Key magnitude actively selects *away* from answer-bearing tokens. On Llama
at 1.05 it keeps 7.1% of them while retaining 24.67% of the cache, and fails **100%** of samples.

**A mechanism worth naming as a hypothesis, not a result:** Qwen3 applies QK-Norm and Llama does not (D1, D15).
Normalised keys should carry less magnitude structure to be misled by, and Qwen's effect is indeed four times
smaller. That is consistent, untested, and exactly the kind of family difference this replication existed to
expose.

**It also closes a hope.** Magnitude is the one importance signal computable at prefill, so it was the only H3
baseline a shared prefix cache could run (D36's open problem). It is not merely useless there — it is worse than
random.

### Scope

Two families, one probe, five fact types, two budgets, one retention ratio per budget, no faults injected. The
joint arm's perfection on Llama (0 of 101) means the 1.05 comparisons are floor-limited on the joint side too: a
better baseline could not be distinguished from it here.

## D48 — H3's pre-registered rule is met at 0.70 bits on both families, and the joint policy's advantage grows as the budget tightens (2026-09-23)

D47 returned a **partial** verdict, and diagnosed why: against retention-only the joint arm was **floor-limited**
— 0 of 101 on Llama and 1 of 99 on Qwen at 1.05 bits, with both arms at the floor at 2.00. Two arms at the floor
are a bound, never a win (D30). The diagnosis implied its own fix: run a **tighter** budget, where the baselines
have room to fail and the comparison has something to discriminate.

0.70 bits per original token. The allocator picks 4-bit at **16.44%** retention (against 24.67% at 1.05);
retention-only drops to bf16 at 4.4%. `precision_only` again emits nothing — its floor is ~2.63, so the absence
result now holds at three budgets. Iso-budget enforced by measurement; every baseline spends at or above the joint
arm. `tool_arg` excluded from both families (D46).

| family | budget | baseline | joint | baseline | p | **Holm/18** | discordant |
|---|---|---|---|---|---|---|---|
| **Qwen** | **0.70** | **retention-only** | 0.0101 | 0.2222 | 4.6e-06 | **2.3e-05** | 21 |
| **Qwen** | **0.70** | **magnitude-only** | 0.0101 | 1.0000 | 4.2e-23 | **6.3e-22** | 98 |
| **Qwen** | **0.70** | **recency** | 0.0101 | 0.6465 | 2.1e-15 | **2.1e-14** | 63 |
| **Llama** | **0.70** | **retention-only** | 0.0000 | 0.2871 | 7.2e-08 | **4.3e-07** | 29 |
| **Llama** | **0.70** | **magnitude-only** | 0.0000 | 1.0000 | 1.5e-23 | **2.7e-22** | 100 |
| **Llama** | **0.70** | **recency** | 0.0000 | 0.6600 | 4.5e-16 | **5.4e-15** | 66 |

**All six meet the rule, on both families, corrected across the whole family of eighteen contrasts.** The
pre-registered refutation rule — "if the joint policy does not beat *every* single-signal policy at equal bits on
at least two model families, H3 is refuted" — is **met at 0.70 bits per original token.** H3 moves from
*partially supported* (D47) to **supported**.

**15 of 18 contrasts survive Holm overall.** The three that do not are all retention-only at the looser budgets,
and they are now explained rather than outstanding: Qwen 1.05 (p = 0.034, Holm 0.10, 8 discordant) and both 2.00
contrasts (0 and 1 discordant) fail because the **joint arm is essentially lossless there**, not because the
policy is weak. A bound at the floor is not a failure to win; it is a comparison with no room.

### The advantage grows monotonically as the budget tightens

| budget | Qwen: joint vs retention-only | Llama: joint vs retention-only |
|---|---|---|
| 2.00 | 0.0000 vs 0.0000 (bound) | 0.0000 vs 0.0099 (bound) |
| 1.05 | −0.0606 | −0.1386 |
| **0.70** | **−0.2121** | **−0.2871** |

Monotone in both families, and it is the pattern the project's own earlier measurements predict. D37/D38 found two
regimes: at ~4 bits per original token nothing discriminates, at ~1 bit every step matters. Joint allocation is a
way of spending scarce bits well, so its value should vanish where bits are plentiful and grow where they are not
— which is what these three budgets show, on two families, rather than being argued from one.

### What the joint policy achieves at 0.70

**22.9× compression** of the KV cache (0.70 against 16 bits per original token), with **0 of 101** samples failing
on Llama-3.2-3B and **1 of 99** on Qwen3-1.7B. The joint arm never came off the floor on Llama at any of the three
budgets tested.

### Scope, and the limitation this creates

The Llama comparisons are floor-limited **on the joint side at every budget**, so a policy better than this one
could not be distinguished from it here, and neither could a slightly worse one. Establishing where this policy
*does* break needs a budget below 0.70 — the allocator holds 4-bit down to its D40 floor and buys tokens, so
0.55 gives 12.9% retention and 0.45 about 10.6%. Two families, one probe, five fact types, three budgets, no
faults injected; the fault interaction is D43/D45's separate result and has not been crossed with this one.

## D49 — Precision × faults: lower precision costs more per flip, and about half of that is composition rather than precision (2026-09-23)

Row 8 of `docs/m5_exit.md`, and the last axis pair this project had never crossed. D43/D45 crossed retention ×
faults and refuted separability. The allocator's **precision** stage rests on the same untested independence: its
ladder picks a bit width with no reference to the fault rate.

Four arms, full retention, unprotected, **matched exposure of 558 flips per trial** via `n_flips_per_token`
= 0.24395 — the rate a 4-bit g128 cache takes at BER 1e-6. Matching per *original token* is what makes this
comparable across precisions and across the probe's eightfold context range; a flat `n_flips` would give a
512-token and a 4096-token sample the same exposure, the bug D43 nearly shipped one axis over.

### The confound is real this time, and the design brackets it instead of matching it

Unlike pruning, changing precision **does** change composition: fp16 scales cost the same at any bit width, so
metadata's share of storage rises as precision falls — and metadata is at least 1543× more damaging per bit
(D25/D28). No group size makes 2-bit's composition match 4-bit g128's, so matching was abandoned for
**bracketing**: two extra 4-bit arms vary composition at *fixed* precision, and the 2-bit arm's share falls
between them.

| arm | stored | metadata share of flips | task failure | clean accuracy |
|---|---|---|---|---|
| 4-bit g128 (reference, reused from D43) | 558.2 Mbit | 6.10% | 0.3277 | 0.9833 |
| 4-bit g64 | 574.5 Mbit | 8.88% | 0.4426 | 0.9667 |
| **2-bit g128** | 295.9 Mbit | **11.46%** | **0.6655** | **0.6333** |
| 4-bit g32 | 607.2 Mbit | 13.50% | 0.5574 | 0.9833 |

All three contrasts against the reference meet the pre-registered rule: 2-bit +0.3378 [+0.2770, +0.3986]
(p = 2.1e-07, 35 of 37 discordant); g64 +0.1149 [+0.0541, +0.1723] (p = 0.0019, 29); g32 +0.2297
[+0.1622, +0.2973] (p = 5.4e-06, 32).

### The decomposition, and the part of it that needs no model

**Model-free, and this is the solid half:** the 2-bit arm sits at **11.46%** metadata share and fails **0.6655**,
while 4-bit g32 sits at **13.50%** — a *worse* composition — and fails **0.5574**. More metadata exposure, less
damage. **Composition alone cannot explain the 2-bit arm's failure**, because a fixed-precision arm with more
metadata in the line of fire does better. That is an ordering violation, not an estimate.

**With a model, and this half is an estimate:** among the three 4-bit arms, failure rises with metadata share of
flips at 0.0304 per percentage point (r = 0.9897 — *on three points*, which is not evidence of linearity).
Interpolating to the 2-bit arm's 11.46% predicts 0.5024 against the observed 0.6655, so of the +0.3378 total gap
roughly **52% is compositional and 48% is precision proper**. The split is a three-point interpolation and is
reported as such; the ordering violation above is what carries the claim.

### The selection runs against the effect, which makes it conservative

The shared set is **37 of 60**, down from 56 in D43, because the 2-bit arm answers only **0.6333** of samples
correctly *before any faults* — it loses 22 of the reference arm's 59. That is D16 reproduced: 2-bit destroys the
task on its own. So the fault comparison is conditioned on the subset 2-bit can handle, i.e. its easiest samples,
and it still fails 0.6655 under matched exposure. The conditioning flatters the 2-bit arm and the effect survives
it.

### What this means for the allocator — and why it is less urgent than D43's

**The structural finding is the same shape as D43's: the reliability tax is not flat in precision either.** A
flip in a lower-precision cache costs more, partly because there is proportionally more metadata to hit and
partly because each surviving code carries more of the value.

**But unlike the retention case, the allocator is already fenced off from this regime.** `MIN_BITS = 4` (D40) was
set for a completely different reason — the iso-budget frontier has an interior optimum and 2-bit loses to 4-bit
at equal budget — and it happens to keep the allocator out of the precision range where this interaction bites.
The retention stage has no equivalent floor: it prunes to 16% and below, which is exactly where D43's
concentration effect lives. So the two structural gaps are not equally live, and saying otherwise would overstate
this one.

**What is not measured:** the matched-*rate* comparison. At BER 1e-6 over its own storage a 2-bit cache takes ~296
flips rather than 558, so the deployment-condition picture needs its own arm and does not follow from this one.
The `ber_basis`/`n_flips_per_token` machinery makes it one config, but it was not run here.

### Scope

One model, one probe, one rate, full retention only, three precision/group configurations. The 37-sample
denominator is small, and it is small for a reason that is itself a finding.

## D50 — M6: BFCL's own categories are too short to compress, and on the catalogue vehicle a shared prefix cache has no usable signal at all (2026-09-24)

M6 was to test C4 on an agent task. The plan named BFCL's AST categories. **Measured before building anything,
they cannot host a retention experiment:**

| category | median cache tokens | max |
|---|---|---|
| simple | 166 | 262 |
| multiple | 333 | 639 |
| parallel | 181 | 351 |
| **multi_turn_long_context** | **157** | 363 |

Against 512–4096 in this project and a policy pruning to 16–25%, a 166-token cache leaves ~27 tokens, of which
the attention sink pins 4. This is GSM8K's defect again (claims.md: "cached problem text averages ~62 tokens") and
the **third** time the project has hit *the task's cache is too short to compress*. BFCL's "long_context" category
earns its name from tool responses accumulated while **executing**, which needs the stateful harness the plan
flagged as its own decision point — not a static prompt.

### The vehicle: the scenario the benchmark leaves out

A real agent carries dozens of tool schemas in its prompt. BFCL's own files supply **769 distinct schemas** at a
median of 122 tokens, so embedding a question's gold schema in a catalogue of N real distractors gives 1442 cache
tokens at N = 10 and 3290 at N = 25 — the probe's range. The gold schema is then **4.0–8.8% of the cache**, so a
policy keeping a quarter of it must actually *find* the one schema the answer needs.

**Golden gate, Qwen3-1.7B, 120 samples:** AST accuracy **0.8667** at N = 10 and **0.8833** at N = 25, both inside
M0's 85–100% band, and a larger catalogue costs nothing on its own — so catalogue size is a clean cache-length
knob rather than a confound. That was checked before the retention arms, not after.

**Two pieces of machinery this needed, both consequences of earlier records.** The scorer is *new*, not adapted:
D46 froze `score_tool_arg` because broadening it changed 2 of 492 stored Qwen predictions, so `score_bfcl_ast`
is written the way the old one should have been — it reads what was called with which arguments, accepts
`arguments`/`parameters`/`args` and tagged or bare JSON, and reports `named_gold_function` separately from
`correct`, because "dropped the schema" and "kept it but got an argument wrong" are different failures. And the
retention unit is `schema`: a tool catalogue is JSON, so punctuation-delimited spans are meaningless in it, and
the block is one function schema — the agent-shaped form of D34, where the unit comes from the vehicle's own
structure.

### The vehicle turned out to BE a shared prefix cache, and that is the result

The two-phase protocol's guard refused the first version, where `cache_len` was the whole prompt. The fix is not a
workaround: a serving system caches the tool catalogue **once** and reuses it across requests, while the question
is per-request. So the catalogue is the cached prefix and the question is the query — which is exactly D36's
largest open problem, where query-aware scoring cannot be computed before the query arrives.

Four arms at 1.05 bits per original token (the H3 operating point), 105 paired samples:

| arm | scorer | deployable on a prefix cache? | kept_target | vs chance | task failure |
|---|---|---|---|---|---|
| joint | query_attention | **no** | 0.9906 | **+0.7577** (p = 6.0e-19) | **0.0286** |
| recency | recency | yes | 0.2805 | +0.0338 (p = 0.53) | 0.7714 |
| **prefix-cache** | window_attention | **yes** | 0.2026 | **−0.0296 (p = 0.0053)** | **0.8095** |
| magnitude | key_norm | yes | 0.2501 | +0.0034 (p = 0.43) | 1.0000 |

| contrast | difference | p | discordant |
|---|---|---|---|
| joint vs prefix-cache | **+0.7810 [+0.6952, +0.8571]** | **1.4e-19** | 82 |
| joint vs recency | +0.7429 [+0.6571, +0.8190] | 1.0e-18 | 78 |
| recency vs prefix-cache | +0.0381 [−0.0286, +0.1143] | 0.317 | 16 |

- **The gap is 78 points, larger than RAQ-Probe's 57 (D36).** Query-aware retention fails 2.86% of tool calls; the
  best arm a shared prefix cache could actually run fails **80.95%**.
- **Query-agnostic attention is significantly BELOW chance at finding the gold schema** (−0.0296, p = 0.0053) and
  is statistically indistinguishable from recency. It is not a weak signal here, it is no signal — a stronger
  version of what D47 found for key magnitude, and new for `window_attention`, which on RAQ-Probe was merely weak
  (0.624) rather than anti-correlated.
- **A mechanism worth naming as a hypothesis:** a tool catalogue has no recency structure (the gold schema sits at
  a random position) and no textual salience (every schema looks like every other schema). RAQ-Probe's fact
  sentence is distinctive against WikiText filler; a function schema among function schemas is not. So the
  query-agnostic scorers had something to latch onto there and have nothing here.

### **M6 does not close C4, and saying it did would be the overclaim**

C4 is that *token-level* metrics under-estimate agent failures. Its evidence remains D16's +29% perplexity
coexisting with −71% probe accuracy — measured on retrieval, under quantization. This campaign has **no
token-level metric on the same vehicle**: `cache_ppl` accepts WikiText only, and nothing here measures perplexity
over a tool catalogue. What M6 establishes is a different and arguably more useful thing — that the deployable
arm's failure is far worse on an agent task (0.8095) than on the probe (0.624) at the same budget — but that is
an *agent-vs-retrieval* comparison, not a *metric-vs-metric* one.

The one pointer the data does give runs against C4's direction in the joint arm: `answer_changed` 0.1143 against
task failure 0.0286, i.e. four times as many answers changed as became wrong. At high damage the two converge
(0.8286 against 0.8095). Closing C4 needs perplexity over the catalogue — one task-kind change — and is left open
rather than declared.

### Scope

One model, one probe of the agent task, one budget, two catalogue sizes pooled, no faults injected. The
`named_gold_function` split was built for this and is not yet analysed: whether the deployable arms fail by
dropping the schema or by keeping it and mis-filling arguments is answerable from the stored records.

## D51 — Second-family replication: the allocator's floor holds, the protection headline is stronger, and Llama tolerates 2-bit far better (2026-09-25)

Everything in the dossier except H3 was Qwen3-1.7B only. Seven arms on Llama-3.2-3B, chosen for stakes per GPU
hour: the five D40 frontier arms (the allocator's **one measured constant**) and the M3 protection pair (the
project's strongest **practical** recommendation). Identical KV geometry, so a difference is a family effect.

### 1. D40's interior optimum replicates in shape; the depth does not

All five arms at ~1.05 bits per original token, query-aware span retention, 101 paired samples:

| retention | bits | **Llama** task failure | Qwen (D40) | ratio |
|---|---|---|---|---|
| 6.25% | bf16 | **0.2277** | 0.1966 | 1.16× |
| 12.5% | 8 | **0.0099** | 0.0684 | 0.14× |
| **25%** | **4** | **0.0000** | 0.0085 | — |
| 31.5% | 3 | **0.0198** | 0.0171 | 1.16× |
| 40.5% | 2 | **0.0396** | 0.1368 | **0.29×** |

**The minimum is 4-bit at 25% on both families, so `MIN_BITS = 4` is not a Qwen artefact.** It is a clean U on
Llama too. Contrasts against the 4-bit arm:

| against 4-bit @ 25% | difference | p | discordant |
|---|---|---|---|
| bf16 @ 6.25% | −0.2277 [−0.3069, −0.1485] | **1.6e-06** | 23 |
| 2-bit @ 40.5% | −0.0396 [−0.0792, −0.0099] | **0.0455** | 4 |
| 3-bit @ 31.5% | −0.0198 [−0.0495, +0.0000] | 0.157 | 2 |
| 8-bit @ 12.5% | −0.0099 [−0.0297, +0.0000] | 0.317 | 1 |

- **Both ends of the U are significant**, which is what makes it an interior optimum rather than a monotone curve:
  too few tokens (bf16) and too little precision (2-bit) each cost significantly more than the 4-bit point.
- **The plateau is wider on Llama and less resolvable.** 3-bit and 8-bit are indistinguishable from 4-bit on 2 and
  1 discordant samples. The 4-bit arm fails **0 of 101**, so as in D48 the comparison is floor-limited on the
  reference side — a better operating point could not be distinguished from it here.
- **Llama tolerates 2-bit about 3.5× better than Qwen** (0.0396 against 0.1368) and 8-bit about 7× better, while
  being slightly *worse* at bf16 with too few tokens. So the frontier's *shape* is a property of the trade and its
  *depth* is a property of the family. A paper stating the 4-bit floor can now say it holds on two families; one
  quoting D40's magnitudes must say which model they came from.
- **Consistent with, and not evidence for, the QK-Norm story.** Qwen applies QK-Norm and Llama does not (D1, D15),
  and D47 already found key magnitude a 4× stronger anti-signal on Llama. Low-precision key quantization being
  *easier* without QK-Norm is the same axis, but the mechanism is untested and is named as a hypothesis.

### 2. The protection headline replicates, and what it removes is bigger

BER 1e-6, 4-bit, 50 paired samples, no-fault controls at exactly zero on both arms:

| | **Llama** | Qwen (D25/D30) |
|---|---|---|
| unprotected task failure | **0.5250** | 0.3211 |
| +0.075% metadata range guard | **0.0000** | 0.0000 |
| paired difference | **−0.5250 [−0.6075, −0.4450]** | −0.3211 [−0.2629, −0.3815] |
| p / discordant | **1.5e-09 / 48 of 50** | 6e-10 / 50 of 58 |

**A range check costing 0.075% of storage removes 100% of measured fault-induced failure on both families** — and
Llama's unprotected arm is **more** fragile (52.5 points against 32.1), so the guard has *more* to remove, not
less. That was the specific risk this pair was run to check, written into the config header beforehand: the
mechanism is arithmetic and should transfer (a float16 scale's bit 14 multiplies it by 65536 in any family) but the
base rate depends on what a rescaled group costs *that model's* answer, and it could have been much smaller.

The composition caveat D46 forced does not bite here: on Llama every `tool_arg` sample already fails the clean
answer, so restricting to the other five fact types changes nothing — 0.5250 → 0.0000 either way, identical to
four decimals.

### What is still single-model after this

Q48 (ρ = −0.377, the correlation that killed §6's multiplicative score), the full M2 fault map, H4's span finding,
D32's attention sink, D39's KIVI comparison, and the concentration effects of D43–D45 and D49. Q48 is the highest
value of those and the most expensive — it needs both a quantization-sensitivity campaign and a fault campaign on
Llama — and the QK-Norm hypothesis predicts precisely that it might differ, which is what makes it worth the hours.

## D52 — Q48 on the second family: the pre-registered branch replicates, the sign does not, and the reason is locatable (2026-09-25)

Q48 is the correlation that decided the project's architecture. Its threshold was frozen before measurement
(partial ρ ≥ 0.6 = one shared signal, ≤ 0.3 = separate signals), and on Qwen3-1.7B it came out **ρ = −0.377
[−0.588, −0.128], partial −0.282**: separate signals, and mildly *anti*-correlated. That is the number cited for
never building §6's multiplicative Joint Reliability Score. The risk this replication checks is specific: **if the
correlation is positive on Llama, the brief's design was right for some families and this project rejected it on
one model.**

Same dataset, strata, counts and seeds; identical KV geometry; 56 (layer, part) cells each.

| | **Llama-3.2-3B** | Qwen3-1.7B |
|---|---|---|
| Spearman ρ | **+0.0726 [−0.2372, +0.3809]** | −0.377 [−0.588, −0.128] |
| partial ρ given KV magnitude | **+0.0466** | −0.282 |
| pre-registered branch | **separate signals** | separate signals |

### What replicates

**The pre-registered verdict, which is the load-bearing part.** Both families land in the *separate signals*
branch (partial ρ well below 0.3). §6's premise — that semantic importance, quantization sensitivity and fault
vulnerability are facets of one signal that can be multiplied — is wrong on both families, and the frozen
threshold says so without interpretation.

### What does not replicate, and the mechanism behind it

**The sign.** And the maps say exactly why:

| | fault damage peak | early vs late | quant sensitivity peak | early vs late |
|---|---|---|---|---|
| Qwen | layer 1 | **0.2658 → 0.1764** | layer 26 (keys) | 0.0012 → **0.0023** |
| Llama | layer 1 | **0.1395 → 0.1135** | layer **2** (keys) | **0.0033** → 0.0024 |

Fault damage concentrates **early** in both families. Quantization sensitivity concentrates **late** in Qwen and
**early** in Llama. So on Qwen the two axes point at opposite ends of the network and the correlation is negative;
on Llama they point the same way and the opposition disappears. This is the sharpest evidence yet for the QK-Norm
hypothesis that D47 and D51 each touched from a different direction: **Qwen normalises keys and Llama does not, and
Qwen's sensitive keys are the late ones.** Still a hypothesis — nothing here ablates QK-Norm — but it is now a
hypothesis with a located effect rather than a hand-wave.

**Why the correlation is only +0.07 rather than strongly positive, given both peak early:** Llama's fault map is
much flatter. Coefficient of variation across cells is **14%** (sd 0.0179 on mean 0.1278) against Qwen's **27%**
(0.0604 on 0.2228). There is less heterogeneity to correlate with, which is a finding about H1 in its own right —
the "top 10% of cells carry 40.2% of damage" concentration is itself family-dependent.

### **The correction this forces on my own reasoning**

The decision records repeatedly justify rejecting the multiplicative score with "the two signals it multiplies are
**anti-correlated**, so their product is a worse ranking than either factor alone". **That argument is
Qwen-specific.** On Llama the signals are *orthogonal*, where the correct statement is weaker and different:
multiplying uncorrelated signals adds no information the factors do not already carry, and buys the cost of a
joint search for nothing. Both families reject the multiplicative form; only one supports the reason given for it.
Every place that reason appears should read "not one signal" rather than "anti-correlated", which is what the
frozen threshold actually licensed.

**What is NOT established:** that the sign flips between families. The intervals overlap on [−0.237, −0.128]. What
is established is that Qwen's correlation is significantly negative, Llama's is indistinguishable from zero, and
both are far below the 0.6 the "one shared signal" branch required.

### Scope

Two families, one dataset (WikiText-2 cache-conditioned), one quantization spec per axis, 56 cells. The QK-Norm
attribution is inferred from where the maps peak, not from an ablation, and a third family — or a Qwen variant
with QK-Norm disabled — is what would test it.

## D53 — H4 replicates on the second family and is stronger there; the locality control replicates too (2026-09-26)

H4 is the headline retention claim: **the unit of KV retention is the span, not the token.** It was one model.
Six arms on Llama-3.2-3B at 25% retention, bf16, 100 paired samples (`tool_arg` excluded per D46).

| arm | **Llama** | Qwen (D33/D34) |
|---|---|---|
| `oracle_drop_target` (99% kept, answer tokens removed) | **1.0000** | 0.977 |
| recency | 0.6400 | 0.6923 |
| **`oracle_keep_target`** (perfect token oracle) | **0.6200** | 0.5641 |
| `oracle_keep_random_span` (locality control) | **0.6100** | 0.5812 |
| `oracle_keep_sentence` (the fact's own span) | **0.0000** | 0.1538 |
| query-aware attention @ span (deployable) | **0.0000** | 0.0256 |

| contrast | difference | p | discordant |
|---|---|---|---|
| span oracle vs **perfect token oracle** | **−0.6200 [−0.7100, −0.5300]** | **3.4e-15** | 62 |
| span oracle vs locality control | −0.6100 [−0.7000, −0.5100] | 5.7e-15 | 61 |
| **token oracle vs locality control** | +0.0100 [+0.0000, +0.0300] | **0.317** | 1 |
| query-aware span vs recency | −0.6400 [−0.7300, −0.5500] | 1.2e-15 | 64 |
| query-aware span vs span oracle | 0.0000 | 1 | **0** |

### What replicates, and more strongly

**A perfect token oracle fails 62% of questions on Llama** — keeping *every* answer-bearing token and nothing else.
The fact's own sentence fails **0%**. The span-over-token gap is **62 points** against Qwen's 41. So H4 is not a
Qwen property, and the second family makes it larger rather than smaller.

**The locality control replicates, which is the control that gives H4 its meaning.** A same-sized contiguous window
placed elsewhere in the document scores 0.6100, and against the token oracle's 0.6200 that is **p = 0.317 on one
discordant sample** — indistinguishable. So a span is not valuable because it is contiguous; it is valuable because
it is *the fact's own* neighbourhood. Both families agree, and on Llama the two arms are within one sample of each
other.

**The precondition holds.** `oracle_drop_target` keeps 99% of the cache with the answer tokens removed and fails
**100%** of samples, so the fact is load-bearing and importance is measurable on this probe at all (D33's gate).

### What cannot be re-established on Llama, and why

**D36's retraction — that the sentence oracle is *not* a ceiling — is untestable here.** On Qwen the query-aware
span arm (0.0256) beat the sentence oracle (0.1538) by 13 points, which is what forced the word "ceiling" to be
withdrawn. On Llama **both reach exactly 0.0000** with **zero discordant samples**, so the comparison has no room.
By D30's rule that is a bound, not agreement: what can be said is that the deployable arm is no worse than the
oracle here, and the 13-point margin is a Qwen measurement that this family cannot confirm or deny.

This is the same floor-limitation D48 and D51 both ran into, arriving a third time: Llama is simply more tolerant
at 25% bf16 retention, so the good arms bottom out and the interesting comparisons among them stop resolving. It
also means the 62-point span-over-token gap is measured *against a floor on the span side* — the true gap could be
larger and this design could not show it.

### Scope

Two families, one synthetic probe whose facts are single sentences **by construction**, oracles that are not
deployable scorers, one retention ratio. The construction caveat is now the binding one: H4 replicating across
families says nothing about whether it survives a task whose facts are not sentence-shaped, which is what
RULER/LongBench would test.

## D54 — The concentration effect does NOT replicate on the second family, with adequate power, and the family difference itself is established (2026-09-26)

D43, D44 and D45 built up what those records call *the project's strongest structural result*: at matched fault
exposure an unprotected 25%-pruned cache fails about twice as often as an unpruned one, replicated across a decade
of error rate on Qwen3-1.7B. It is the evidence for **"protection is not separable from retention"** — the one thing
§6's multiplicative form was credited with getting right — and the reason the allocator's protection stage carries a
caveat about pruned caches.

**On Llama-3.2-3B it is absent.** Same rate (3e-7, the one D44 established as where the effect is visible), same
matched-exposure machinery (`ber_basis: unpruned`, 159 flips into both arms), 49 paired samples.

| | **Llama-3.2-3B** | Qwen3-1.7B (D45) |
|---|---|---|
| unpruned, unprotected | **0.2270** | 0.1116 |
| pruned 25%, matched exposure | **0.2168** | 0.2232 |
| difference | **−0.0102 [−0.0510, +0.0306]** | **+0.1116 [+0.0536, +0.1719]** |
| p / discordant | **0.694 / 37 of 49** | 4.6e-04 / 43 of 56 |
| risk ratio | **0.955** | **2.000** |

### This is a negative result with power, not a null from lack of room

Three checks, because the project has twice mistaken a pinned comparison for an absent effect (D30 at the floor,
D44 at the ceiling):

* **Not saturated.** Both arms sit at 0.22, mid-range in both directions. The 1e-5 failure was both arms above 0.94;
  this is nothing like it.
* **Powered.** **37 of 49 samples are discordant** — more discordance than Qwen's 43 of 56 as a fraction. The test
  had room to find a two-fold effect and found nothing.
* **Not compositional.** Payload-metadata share of flips 6.2516% → 6.0171%, p = 0.249. Controls exactly zero on
  both arms.

**And the family difference itself is established, not merely a failure to replicate.** Qwen's interval
[+0.0536, +0.1719] and Llama's [−0.0510, +0.0306] are **disjoint**, with a gap of 0.0230. That is a stronger
statement than Q48's family difference (D52), where the intervals overlapped and only the verdict replicated.

### What this withdraws

**"Protection is not separable from retention" is a Qwen result, not a general one.** The allocator's
retention-free protection lookup is *correct* for Llama and questionable for Qwen — the opposite of a structural
law. Every place the records call this "structural", including D43's own framing and the M4 exit checklist's row 10,
describes a single family. This is the **sixteenth** finding withdrawn or re-scoped in this project, and the largest
by what rested on it.

What survives unchanged: the *mechanism* is still coherent where it appears, D44's diagnosis of the 1e-5 null as
saturation still holds on Qwen, and the matched-exposure methodology is what made both the positive and the negative
measurable.

### A mechanism the earlier records already predicted

The config header for this run said a weaker effect would be *consistent* rather than contradictory, because D51
found Llama's fault map markedly flatter — coefficient of variation **14%** against Qwen's **27%**. A flatter map
means the cells pruning preserves are not systematically more fragile than the ones it discards, so there is nothing
for retention to concentrate. The effect came out not weaker but **absent**, which is the same prediction taken to
its limit.

D52 adds the other half: Qwen's quantization sensitivity peaks at **layer 26 keys** while Llama's peaks at **layer
2**. On Qwen, query-aware retention keeps the semantically important tokens *and* those happen to live where the
late-layer sensitive keys are; on Llama the two do not coincide. So three independent measurements — flatter fault
map, differently-located quantization sensitivity, absent concentration — all point at the same family difference,
and all three are consistent with Qwen applying QK-Norm where Llama does not. **Still a hypothesis: nothing here
ablates QK-Norm**, and a third family or a QK-Norm-disabled Qwen is what would test it.

### Scope, and what this does not say

It does not say pruning is safe in general, and it does not say Qwen's result was wrong. Llama is **2.03× more
fragile unprotected** at this rate (0.2270 against 0.1116), so the two families differ in base rate as well as in
the interaction, and a deployment on either should price protection from its own measurements. One rate, one
retention ratio, one scorer, one probe.

## D55 — D17 and H1c replicate on the second family from data already on disk, and metadata dominance is stronger there (2026-09-26)

The Llama fault campaign run for Q48 (D52) carries the bit-position and target-class strata as well, so two of M2's
findings replicate at **zero additional GPU cost** — a reminder that a campaign designed for one question often
already answers another, and that checking costs nothing.

Computed identically on both families (mean and max `delta_nll_max` per target class, 4000 code flips and ~790
metadata flips each):

| | Qwen3-1.7B | **Llama-3.2-3B** |
|---|---|---|
| code, mean damage | 0.2233 | **0.1273** |
| metadata, mean damage (scale + zero, count-weighted) | 0.6354 | **0.5659** |
| **metadata / code ratio** | **2.845×** | **4.446×** |
| worst-case metadata | 35.96 nats | 19.05 nats |
| worst-case code | 0.870 nats | **0.366 nats** |
| tail separation (worst metadata / worst code) | 41× | **52×** |
| **bit position, LSB → MSB damage** | **1.032×** | **1.023×** |

### D17's target-class finding replicates and is larger

**Metadata is 4.45× more damaging than codes on Llama against 2.85× on Qwen**, and the tail separation is cleaner:
Llama's worst code flip costs 0.366 nats where its worst metadata flip costs 19.0. This is the finding every
protection decision in the project descends from — M3's arm design, the +0.075% guard, the allocator's protection
stage — and it is now the **best-replicated result here**, holding on both families and stronger on the second.

That matters more than usual given D54. The concentration effect turned out to be a Qwen property; *this* one is
not. The two are easy to conflate because both concern faults and both were established in the same milestones, so:
**where the damage lands (metadata, both families) is general; whether pruning concentrates it (Qwen only) is not.**

### H1c replicates almost exactly

**Bit position moves damage 1.023× on Llama against 1.032× on Qwen**, while the stored value changes **128×** from
LSB to MSB. Two families, essentially the same number. MSB-only ECC is poor value per bit on both, and the project's
refusal to spend a bit on it was not a single-model decision.

### A discrepancy in my own citation, stated rather than smoothed

D17 records the ratio as "metadata 2.4× codes in mean damage". Computed as above it is **2.845×** on the same data.
The difference is an aggregation choice — D17 evidently pooled the two metadata classes differently — and the
number quoted in this record is the one computed identically on both families, which is what a cross-family
comparison requires. D17's figure is not withdrawn; it is a different statistic on the same measurement, and this
record is the like-for-like one.

### What is still single-model in M2

**Q49** (the role effect: OR 7.13 once a token's whole row is destroyed) needs the `fault_qa` probe runs, not this
WikiText campaign, so it is not free and has not been run on Llama. Given D54, it is now more interesting than it
was: Q49 is about *which tokens* matter under faults, which is the same axis the concentration effect turned out to
be family-specific on.

## D56 — H6 replicates and the margin grows; my pre-run prediction was wrong, and why it was wrong is the finding (2026-09-26)

KIVI at its own published configuration — keys `group_token` 32, values `group_channel` 32, residual window 128, the
window paid for in the budget — against the joint point, on Llama-3.2-3B. 100 paired samples.

| arm | bits / original token | **Llama** task failure | Qwen (D39) |
|---|---|---|---|
| **joint point** (4-bit @ 25%, query-aware spans) | **1.0539** | **0.0000** | 0.0085 |
| KIVI 2-bit, residual 128 | 4.3822 | **0.1000** | 0.0513 |
| KIVI 4-bit, residual 128 | 6.1707 | 0.0100 | 0.0000 |

| contrast | difference | p | discordant |
|---|---|---|---|
| joint vs KIVI 2-bit | **−0.1000 [−0.1600, −0.0500]** | **0.0016** | 10 |
| joint vs KIVI 4-bit | −0.0100 [−0.0300, +0.0000] | 0.317 | 1 |

**H6 replicates and the margin is larger.** The joint point is **4.2× cheaper** than KIVI at its published
configuration and beats it decisively (10 points against Qwen's 5.13), while KIVI 4-bit remains essentially lossless
on both families — a bound, not a win, at 5.9× the cost. The budget arithmetic transferred exactly as predicted,
since the geometries are identical: 4.3822 against Qwen's 4.331 and 6.1707 against 6.13, differences of 1.2% and
0.7% from context-length variation alone.

### The prediction I wrote before the run was wrong

The config header said: *"D51 found Llama tolerates 2-bit about 3.5× better than Qwen, and KIVI 2-bit IS a 2-bit
arm — so the most likely outcome is that KIVI looks BETTER on Llama and the margin over the joint point shrinks."*

It looks **worse**: 0.1000 against Qwen's 0.0513, roughly twice the failure rate. The margin grew instead of
shrinking.

### Why, and this is the part worth keeping

The two "2-bit" arms are not the same configuration:

| | keys | values | residual | retention |
|---|---|---|---|---|
| D51's 2-bit arm | 2b `group_token` 32 | 2b **`per_token`** | 0 | 40.5% |
| KIVI's 2-bit | 2b `group_token` 32 | 2b **`group_channel` 32** | 128 | 100% |

**D51's finding is axis-specific and I generalised it past its evidence.** "Llama tolerates 2-bit better" was
measured with values on `per_token`; KIVI quantizes values along `group_channel`, and there Llama does worse than
Qwen. Three things differ between the arms at once — values axis, residual window, retention ratio — so this pair
cannot attribute the reversal to any one of them. What *is* established is the negative: **there is no general "2-bit
is easier on Llama" property**, and D51's number belongs to its own configuration.

That is a correction to how I read my own result one record earlier, caught by a prediction failing rather than by a
control. It also sharpens what D51 says: a family's low-precision tolerance is a property of the *quantization
scheme*, not of the family alone — which is a more useful statement than the one I was about to carry forward.

### Scope

One probe, 512–4096 tokens. KIVI's own claims are made at longer context where its fixed residual window is
cheaper per value, and D39's finding that below ~1500 tokens the window is dominated by plain 4-bit with no window
still applies to both families — it is arithmetic about R/T, not about either model.

## D57 — Q49's scale-dependence replicates on both families; its key control does not, and D21 asserted an interval it never showed (2026-09-26)

Q49 asks whether a corrupted token's **role** predicts task failure. Its threshold was frozen before measurement:
**OR > 1 with a CI excluding 1.** On Qwen the answer came out scale-dependent across three records (D19 → D20 →
D21). Both scales re-run on Llama-3.2-3B.

### Single-bit scale: nothing, on both families

| role | Qwen | **Llama** |
|---|---|---|
| target_value | **0 / 960** | **0 / 800** |
| distractor_value | 0 / 720 | 0 / 600 |
| filler | 0 / 960 | 0 / 800 |
| **metadata** | **6 / 480 = 0.0125** | **11 / 400 = 0.0275** |
| no-fault control | 0 / 120 ✓ | 0 / 100 ✓ |
| OR target vs filler | 1.00 [0.02, 50.45] | 1.00 [0.02, 50.46] |

**D19 replicates exactly: a single bit flip in a code costs zero answers, in every role, on both families** — 0 of
960 and 0 of 800. And **metadata is the only thing that hurts at this scale on either model**, at 1.25% (Qwen) and
2.75% (Llama), so Llama is 2.2× worse there — consistent with D55's finding that metadata dominance is larger on
Llama.

### Row scale (2048 bits, one layer): the effect appears, on both

| role | Qwen | **Llama** |
|---|---|---|
| target_value | 14 / 654 = 0.0214 | 11 / 540 = **0.0204** |
| distractor_value | 2 / 436 = 0.0046 | **0 / 360 = 0.0000** |
| filler | 2 / 654 = 0.0031 | 1 / 540 = 0.0019 |
| metadata | 7 / 218 = 0.0321 | 6 / 180 = 0.0333 |

| contrast (Haldane-corrected OR, Woolf CI) | OR | 95% CI | frozen rule met? |
|---|---|---|---|
| Qwen, target vs filler | 5.91 | [1.54, 22.72] | **yes** |
| Qwen, target vs content-matched distractor | 3.93 | **[1.02, 15.14]** | yes, **barely** |
| **Llama, target vs filler** | **7.81** | [1.42, 43.00] | **yes** |
| **Llama, target vs content-matched distractor** | 15.66 | **[0.92, 266.58]** | **NO** |

**So the scale-dependence is general — single bits harmless, a destroyed row costly — on two families.** That is
Q49's headline and it replicates.

### But the control that gives Q49 its meaning is weaker than the records say

The content-matched distractor is what makes Q49 a claim about **role** rather than about **position**: without it,
"the answer token's row matters" could just mean "that region of the document matters". On Qwen that comparison
**barely** clears the frozen rule (lower bound **1.02**), and on Llama it **does not clear it at all** — Llama
recorded **zero** distractor failures, so by D30's rule the 15.66 is a bound, not an estimate, and zero events over
360 is consistent with a large effect without establishing one.

**D21 asserted this criterion was met without showing the interval it requires.** Its text reads *"The
pre-registered Q49 criterion (OR > 1, CI excluding 1) is met at this scale"* while reporting only Fisher p values
(0.004 and 0.036). Computed now, D21's verdict **holds** — both Qwen intervals do exclude 1 — but it was *asserted
rather than demonstrated*, and the interval reveals the key control to be marginal. This is the same class of
failure as the claims-matrix drift found on 2026-09-24: a frozen rule asks for one statistic and the record reports
another.

**What Q49 now supports, precisely:** destroying an answer-bearing token's row costs significantly more than
destroying filler, on both families. That it costs more than destroying a *content-matched distractor* is
established on Qwen at the edge of significance and is a bound on Llama.

### Q49 and the concentration effect come apart, which is the structural point

The config header named both branches before the run. The one that happened: **Q49 replicates while D54's
concentration effect does not.** They are not the same phenomenon and cannot share an explanation.

This **rules out** the most natural account of D54's absence: it is *not* that Llama's answer tokens fail to matter
under faults — they matter 7.8× more than filler there, slightly more than on Qwen. So D54's absence needs its own
cause, and the leading candidate remains D54's own: Llama's fault map is flatter (CV 14% against 27%, D51), so
uniform injection into a retention-selected subset does not land on systematically worse cells even though the
answer token itself is more fragile.

A hypothesis that would reconcile them, untested: on Llama the fragility may be concentrated tightly on the answer
token, while on Qwen it is spread across the token's neighbourhood — so pruning to *spans* (which keep the
neighbourhood) captures Qwen's concentration and dilutes Llama's. That is measurable with the existing machinery by
comparing row-destruction on the answer token against row-destruction on its neighbours.

### Scope

Two families, one probe, two corruption scales, one quantization per scale. The distractor comparison is
underpowered on both families and was always going to be: it needs the distractor to fail sometimes, and at these
rates it barely does.

## D58 — D49 replicates, its model-free half is stronger, its compositional half collapses, and my prediction was backwards (2026-09-26)

The last single-model claim. Precision × faults on Llama-3.2-3B at matched exposure (546 flips/trial via
`n_flips_per_token`), full retention, unprotected, 38 paired samples.

| arm | metadata share of flips | **Llama** | Qwen (D49) |
|---|---|---|---|
| 4-bit g128 (reference) | 6.14% | **0.5362** | 0.3277 |
| 4-bit g64 (composition control) | 8.68% | **0.4934** | 0.4426 |
| **2-bit g128** (precision arm) | 11.53% | **0.8092** | 0.6655 |
| 4-bit g32 (composition control) | 13.60% | **0.6053** | 0.5574 |

| contrast | difference | p | discordant | verdict |
|---|---|---|---|---|
| 4-bit g128 → 2-bit g128 | **+0.2730 [+0.2072, +0.3389]** | **8.3e-07** | 34 of 38 | **meets** |
| **4-bit g32 → 2-bit g128** | **+0.2039 [+0.1316, +0.2763]** | **2.8e-05** | 30 of 38 | **meets** |
| 4-bit g128 → 4-bit g32 | +0.0691 [+0.0033, +0.1349] | 0.054 | 29 of 38 | direction only |
| 4-bit g128 → 4-bit g64 | **−0.0428** [−0.1250, +0.0362] | 0.403 | 33 of 38 | direction only, **wrong sign** |

### The model-free result replicates and is stronger than on Qwen

**2-bit g128 at 11.53% metadata fails 0.8092 while 4-bit g32 at 13.60% — a *worse* composition — fails 0.6053.**
More metadata in the line of fire, less damage. On Qwen that ordering violation was reported as an ordering only; here
it is a **significant contrast: +0.2039, p = 2.8e-05 on 30 of 38 discordant samples.** So the claim that composition
alone cannot explain the low-precision penalty is better evidenced on the second family than on the first.

### The compositional half collapses, and my pre-run prediction was backwards

The config header predicted: *"D55 found metadata dominance LARGER on Llama (4.446× against 2.845×) … a larger
metadata penalty should make the COMPOSITIONAL half of the effect bigger here, not smaller."*

It is smaller, and not merely smaller — **non-monotone**. The 4-bit g64 arm carries *more* metadata than the
reference and fails *less* (−0.0428, wrong sign, p = 0.403), and the g32 arm only reaches p = 0.054. On Qwen these
two were +0.1149 (p = 0.0019) and +0.2297 (p = 5.4e-06), clean and monotone, which is what licensed the three-point
interpolation and the "≈52% compositional / 48% precision" split.

**That split is therefore a Qwen number and no equivalent should be quoted for Llama.** A linear interpolation over
three points was already reported as an estimate rather than a measurement; over three points one of which runs
backwards it is not supportable at all. What Llama supports is the model-free statement, which is the stronger one
anyway: the penalty is not compositional, because a worse composition at the same precision does better.

**A hypothesis for why, flagged as such.** Llama's reference arm already fails 0.5362 against Qwen's 0.3277, and the
2-bit arm reaches 0.8092. The composition controls therefore have much less room between the base rate and the
ceiling than they did on Qwen, which is exactly the compression D44 and D45 diagnosed at 1e-5 — a multiplicative
difference cannot express itself as an absolute one near the ceiling. So the compositional channel may be real here
and squeezed rather than absent. Testing that needs a lower exposure, which is one config away and was not run.

### Where this leaves D49's claim

Both families agree on the load-bearing statement: **the reliability tax is not flat in precision, and the penalty is
not explained by composition.** They disagree on the decomposition, and only Qwen supports one. The practical
consequence is unchanged on both — `MIN_BITS = 4` (D40, replicated in D51) already keeps the allocator out of this
regime, a floor set for an unrelated reason.

### Scope

One rate, full retention only, three precision/group configurations per family, 38 paired samples. The 2-bit arm's
0.8092 is close enough to the ceiling that this design's power to resolve *differences among the bad arms* is
limited, which is the same limitation on the opposite side from the floor-limits of D48, D51 and D53.

## D59 — Why the concentration effect does not replicate: Llama starts where Qwen ends, and D54's proposed mechanism looked at the wrong channel (2026-09-26)

D54 recorded that the concentration effect is absent on Llama and offered a mechanism: Llama's fault map is flatter
(CV 14% against 27%, D51), so there is less heterogeneity for retention to concentrate. **That explanation was
about the wrong channel**, and the arms already on disk say so.

### The failures are almost entirely metadata-driven, in every arm

Dose-response between the number of metadata flips in a trial and whether the answer broke:

| family | arm | metadata flips / trial | failure | corr(meta flips, failure) |
|---|---|---|---|---|
| Qwen | unpruned | 9.55 | 0.1059 | **+0.279 (p = 6.8e-10)** |
| Qwen | pruned | 9.87 | 0.2232 | **+0.199 (p = 2.2e-05)** |
| Llama | unpruned | 9.68 | 0.2270 | **+0.347 (p = 1.6e-12)** |
| Llama | pruned | 9.38 | 0.2150 | **+0.286 (p = 6.1e-09)** |

So D54's flatness argument was about the variance of **code**-flip damage, while the failures come through
**metadata**. D51's CV comparison is still a true statement about the fault map; it is just not the channel these
outcomes travel on.

### The per-metadata-flip hazard is where the answer is

Converting each arm's failure rate to a per-metadata-flip hazard (assuming flips act independently within a trial,
which the dose-response supports but does not prove — this is a derived quantity, not a measurement):

| family | unpruned | pruned | ratio |
|---|---|---|---|
| **Qwen** | **1.17%** | **2.53%** | **2.17×** |
| **Llama** | **2.63%** | **2.55%** | **0.97×** |

**Llama's unpruned hazard already equals Qwen's pruned hazard.** Pruning on Qwen raises 1.17% → 2.53%; Llama sits at
2.6% before any pruning. There is no headroom for retention to concentrate, because the thing concentration would
produce is already the starting condition.

**And both families converge to the same ~2.5% per metadata flip once pruned.** That is the shape of a ceiling
rather than of two different effects: a corrupted fp16 scale that lands somewhere consequential costs an answer
about 2.5% of the time in either model, and pruning moves Qwen toward that bound while Llama begins at it.

### This unifies findings that looked unrelated

* **D55** — metadata dominance larger on Llama (4.45× against 2.85×). Same fact from the other side: Llama's
  metadata flips are individually more damaging.
* **D51** — Llama more fragile unprotected at 1e-6 (0.5250 against 0.3211). Same cause, not a separate family quirk.
* **D57** — Q49's role effect replicates on both. Consistent, and no longer puzzling: Q49 destroys a *token's row*
  deliberately, which is a different channel from a uniformly-placed metadata flip. The two were never measuring
  the same thing, which is exactly what D57 concluded from the outside; this is the inside view of why.

### What this changes, and what it does not

**D54's verdict stands: the concentration effect is a Qwen result.** What changes is the account of why, and it is a
better one because it is measured: not "Llama has less heterogeneity to concentrate" but **"Llama's metadata is
already as damaging as concentration can make it."**

That also makes the Qwen effect less exotic than D43–D45 framed it. On Qwen, pruning does not create fragility; it
**raises the per-flip hazard toward a bound both families share.** So "protection is not separable from retention"
is best read as: *how far a model sits below the metadata hazard ceiling determines how much pruning can hurt it.*
A model already at the ceiling has nothing to lose, and one below it does.

### The open question this leaves, stated rather than answered

**Why does Llama start at the ceiling?** Nothing here explains that, and I checked the cheap candidate before
saying so. The stored quantization statistics, at the same 4-bit spec on the same probe:

| | key SQNR | value SQNR | key MSE | value MSE |
|---|---|---|---|---|
| Qwen | 27.05 dB | 18.61 dB | 0.0302 | **0.8336** |
| Llama | **24.84 dB** | 18.62 dB | 0.0112 | **0.0016** |

Two real differences, neither of which answers the question. Llama's keys quantize about **2.2 dB worse** at the
same spec, and its values are **two orders of magnitude smaller in absolute magnitude** (equal SQNR, 500× smaller
MSE). The magnitude difference cuts the wrong way for the obvious story: a corrupted fp16 scale multiplies its group
by up to 65536 either way, and from a *smaller* base that is a smaller absolute excursion, which would predict Llama
being **less** fragile per metadata flip, not more.

So the dynamic-range candidate is **ruled out as stated**, and what remains is the one thing these statistics cannot
see: **how many groups a model actually attends to enough that corrupting one changes the answer.** If Llama's
attention is spread more evenly across groups, a randomly-placed corrupted scale is more often consequential, and it
would sit at the ceiling without any pruning — which is exactly the observation. That is measurable in one forward
pass per sample (attention mass per quantization group, no fault injection, no new campaign) and has not been run.
Recorded as the next question rather than as an answer.

## D60 — Attention spread is ruled out as the reason Llama starts at the metadata hazard ceiling, and the numbers point the other way (2026-09-26)

D59 left one question and named its live candidate: **how many quantization groups a model attends to enough that
corrupting one would change the answer.** The prediction was explicit — *Llama's attention should be spread across
more groups than Qwen's*, because spread attention means more randomly-placed corruptions land somewhere that
matters, which would explain Llama's 2.63% per-metadata-flip hazard against Qwen's 1.17%.

Measured with `scripts/measure_group_attention.py`: one prefill and one `query_attention` read per sample, folded
into `group_token` groups of 128 — the size the fault arms used — over 40 samples per family spread across the
probe's whole 512–4096 length range, so the group count (15.9 on average) matches the arms'.

| model | groups | **effective groups** exp(H) | effective fraction | **top-1 share** | top-5 share | groups holding 90% |
|---|---|---|---|---|---|---|
| Qwen3-1.7B | 15.9 | **3.10** | 0.263 | **0.762** | 0.913 | **44.1%** |
| Llama-3.2-3B | 15.9 | **2.46** | 0.211 | **0.826** | 0.930 | **33.8%** |

**The prediction is refuted, and in the direction that makes it worse rather than merely null.** Llama's attention
is *more* concentrated on every measure — fewer effective groups, a larger top-1 share, and 90% of the mass in a
third of the groups against Qwen's nearly half. On the stated logic that should give Llama the *lower* hazard. Its
hazard is 2.2× higher.

**A methodological note that mattered.** The first run took the first 40 samples, which are all 512-token contexts,
giving 5 groups per sample — at which size `top5_share` is 1.0 by arithmetic and every concentration metric is
degenerate. The subset now spans the length distribution. A measurement whose metric is saturated by construction
would have looked like a clean null.

### Two candidates eliminated, and the arithmetic now points at a third

* **Dynamic range** — ruled out in D59: Llama's values carry 500× smaller MSE at equal SQNR, and a corrupted fp16
  scale multiplies by up to 65536 either way, so from a smaller base that is a smaller excursion.
* **Attention spread** — ruled out here, with the sign reversed.

What survives is the **opposite** of the spread story: concentration may raise the *conditional severity* of the
flips that do matter. If only the top-attention group is consequential, a uniform flip hits it with probability
1/15.9 = 6.3%, and the measured hazards then imply

| | P(flip hits the top group) | measured hazard | implied **P(break | hit top group)** |
|---|---|---|---|
| Qwen | 6.3% | 1.17% | **18.6%** |
| Llama | 6.3% | 2.63% | **41.8%** |

which is what more concentration would produce: when 82.6% of the mass sits on one group, corrupting *that* group
is more often fatal than corrupting the top group of a flatter distribution carrying 76.2%.

**This is arithmetic on an assumption, not a measurement.** It assumes only the top group matters, which the
top-5 shares (0.91 and 0.93) say is wrong in detail. It is recorded as the surviving hypothesis, not as the answer.

### The experiment that would settle it

Inject **one** metadata flip at a time into a *chosen* group and record the group's attention rank alongside the
outcome. That gives P(break | rank) directly per family, and distinguishes "more groups matter" from "the groups
that matter matter more" without any modelling. The fault machinery already addresses individual regions
(`faults/storage.py` enumerates them); what it does not yet do is let an arm target a group by rank, which is the
one piece of work this needs.

### Scope

One probe, one group size, 40 samples per family, query-aware attention only. A query-agnostic read would measure
a different distribution and is the right control if this is pursued — the fault arms are query-agnostic in the
sense that the flips are placed without reference to the question, so what should predict the hazard is arguably
attention as the *model* ends up using it, which is what is measured here.

## D61 — The group-targeted fault path, and D60's design frozen before the run (2026-09-26)

D60 ended with an experiment described and not built. This record is the build plus the **pre-registration**: the
rank cells, the trial counts, the primary contrast and which outcome supports which branch, all fixed before any
arm ran. GATE-1 got this treatment and it is why D18 is citable; eight cells inspected after the fact would not be.

### First, three corrections to what D60 said

**1. The granularity D60 implied is off by three orders of magnitude, and this *explains* its arithmetic rather
than breaking it.** D60 speaks of "corrupting a group" as though a group had one scale. Measured: for T = 2048 with
`group_token 128`, the key scale grid is `(1, H_kv = 8, 16, head_dim = 128)`, so one scale covers **128 tokens × 1
channel × 1 head × 1 layer** and there are **1024 scales per token group per layer-part**. A single flip therefore
corrupts 1/1024 of a group's metadata, not the group. P(a flip lands in the top *group*) is still 1/16 = 6.3%, so
D60's table stands as written — and the dilution is exactly why the implied conditional break probabilities are
18.6% and 41.8% rather than near 1. The arithmetic was right; the mental picture behind it was not.

**2. A corrupted scale does not damage every value in its group.** Reconstruction is `code × scale + zero`, so a
value whose code is **0** is *exactly* invariant under any corruption of its scale. Found by writing the
containment test, not by reasoning: in the fixture, one of the four tokens sharing the corrupted scale is
bit-for-bit unchanged and its code is 0. A second dilution on top of the 1024, and it applies to the real arms.
`tests/test_faults.py::test_group_targeted_faults_are_reversible_and_move_only_their_group` pins both the
containment (nothing outside the group moves) and this exception.

**3. `outputs/raw/` labels every fault campaign "nofault".** The slug's fault token is a hardcoded literal from M0,
when no config had a fault section; protection arms carry their fault model in `task`, which the token never reads.
All 14 existing arms and all 6 below are named `..._noecc_nofault_...` while injecting faults — D24's and D42's
failure mode, found while checking these configs. **Not repaired**, because the slug is part of `experiment_id` and
an honest token would change the id of every completed fault experiment and orphan its raw records. The record
fields (`exposure_mode`, `ber`, `n_flips_per_trial`, `fault_target`) are correct and are what the reports read;
the literal is pinned with its reason in `ids.py` and `tests/test_config_ids.py`.

### Why this is not the measurement D60 already refuted

D60 measured the **distribution** of attention mass over groups and refuted the spread prediction with the sign
reversed. This measures P(break | rank of the corrupted group), which is a different object and not implied by the
first: **a group can hold little mass and still be load-bearing.** The distribution says where attention *is*; the
conditional says what happens when a given group's metadata is destroyed. One is about mass, the other about
consequence, and the surviving hypothesis after D60 is precisely that the two diverge — that concentration raises
the *conditional severity* of a hit without changing how many groups get hit.

### What was built

`generate_group_targeted_plan` in `faults/storage.py` draws single-bit upsets confined to the key scale/zero grid
of one chosen token group, and `protection_arm` gained three fields — `fault_target`, `target_group_rank`,
`rank_scorer` — all defaulting to `None` so `exclude_none` leaves the 14 existing arm identities untouched (the
`ber_basis` lesson; `tests/experiment_identities.json` is unchanged). The rank fold (`analysis/groups.py`) is
shared with `scripts/measure_group_attention.py`, so "group 3" cannot mean two things.

Two guards worth naming, because the first version had the bug the second one describes:

* The helper checks `shape[2] == n_token_groups`, not `len(shape) == 4`. A `per_token` values grid is `(1, H, T, 1)`
  — also 4-D — and a rank check would have accepted it and silently reinterpreted `token_group` as a *token index*,
  returning sites for a different unit than the function's name. Caught in review before it ran.
* A group-targeted arm must state `n_flips`, never `ber`: `flips_for_rate` sizes a count against the arm's entire
  storage while the draw sees one group's scales, so the arm's stated exposure and its real one would differ.

### The frozen design

Six arms, `configs/experiments/d61_{qwen,llama}_{top,rank2,tail}.yaml`. Per arm: 40 probe samples × 8 trials =
**320 conditional observations**, `n_flips: 1` so each trial is one flip and each outcome one conditional, **no
protection** (a guard that corrected the flip would measure the guard) and **no pruning** (the unexplained hazard
difference is on unpruned caches). Ranks: **0** (most attended), **2**, and **999**, which wraps to the
least-attended group — the wrap is deliberate, because every fixed rank above 3 would be past the end for the
512-token samples, and the **realised** rank is recorded per sample so a wrapped cell is visible rather than
pooled with a rank it was not.

**Primary contrast:** rank 0 against the least-attended cell, within each family, paired at the sample level,
Wilcoxon signed-rank plus a 10,000-resample bootstrap CI over samples. D31 applies: a difference needs p < 0.05
**and** a CI excluding zero. **Secondary, descriptive only:** break rate against the corrupted group's recorded
attention share, which is length-independent where an absolute rank is not.

**Expect floor cells.** At a ~2% uniform hazard the tail cells may return 0/320, and D30/D44 makes that a **bound,
not a zero** — the rule of three gives ≤ 0.93% at 320 trials and the record will say so rather than reporting 0.

### Pre-registered predictions

| branch | prediction | what would confirm it |
|---|---|---|
| **A — concentration raises conditional severity** (survives D60) | P(break \| rank 0) is high in both families and **higher in Llama**, whose top group carries 0.826 of the mass against Qwen's 0.762; tail cells near the floor in both | rank-0 rate materially above the tail rate in both families, **and** Llama's rank-0 rate above Qwen's. D60's arithmetic, if only the top group mattered, predicts about 41.8% and about 18.6% |
| **B — more groups are load-bearing in Llama** | rank-0 rates **similar** across families, while Llama's rank-2 and tail rates exceed Qwen's | the family difference living in the non-top cells rather than the top one |
| **neither** | the rank gradient is flat in both families | tail and rank-0 intervals overlapping within each family, which would mean the hazard is not attention-mediated at all and would send the question back to the codec |

**A and B are not exclusive** and the honest outcome may be both; what the design buys is that the two are
separated by *which cells* carry the family difference, which no uniform plan can report.

### Scope, stated before the numbers exist

Eager attention is required for the ranking (D5) and D59's arms were not eager, so **D61's absolute rates are not
interchangeable with D59's 1.17% / 2.63%** — every D61 comparison is internal to this set with the implementation
held fixed. Keys only (the `group_token` grid is where a token group is a unit of stored metadata; values are
`per_token` and their scales are a different object). One probe, one group size, one query-aware scorer, 4 bits.

### AMENDMENT, 2026-09-26 — the primary contrast above is not identified, and here is the one that is

**No arm had produced an outcome when this was written.** The original design and its prediction table are left
exactly as committed in `11d1c12`, because that commit is the timestamped evidence that what follows was found
before any result existed. Editing the table in place would destroy the only thing that separates an amendment
from a post-hoc rationalisation.

**What the check found.** Before running anything, `scripts/measure_group_attention.py` was extended to report
*where* each rank cell sits, not only how much mass it holds. Over 40 samples per family:

| cell | attention share | is group 0 | relative position, mean (sd) |
|---|---|---|---|
| **rank 0** — Qwen / Llama | 0.762 / 0.826 | **1.000 / 1.000** | **0.000 (0.000) / 0.000 (0.000)** |
| rank 1 — Qwen / Llama | 0.054 / 0.038 | 0.000 / 0.000 | 0.473 (0.394) / 0.292 (0.279) |
| tail — Qwen / Llama | 0.013 / 0.008 | 0.000 / 0.000 | 0.521 (0.247) / 0.635 (0.281) |

**The most-attended group is group 0 in every one of the 80 samples, with zero variance in its position.** A rank-0
fault cell is therefore a measurement of *corrupting the first 128 tokens' key scales*, and on this probe
"most-attended" and "first" are the same set. Within a family they cannot be separated, so the pre-registered
primary — rank 0 against the tail, read as an attention gradient — is **confounded with position and is withdrawn
as the primary**. It is not a small caveat: the initial tokens are exactly what `protect_initial: 4` already
protects, so a positional explanation was live before this measurement and remains live after it.

**What survives, and why it is identified.** Two facts rescue the design:

1. **Group 0 is group 0 in both families.** A *cross-family* contrast at a fixed cell holds position constant by
   construction, so the family difference at that cell is not positional. This matters because the branches D60
   left open differ precisely in *which family breaks more at which cell*.
2. **Among non-first groups, share and position separate.** rank 1 carries more mass than the tail in **100%** of
   samples (median ratio **7.24×** Qwen, **5.51×** Llama, ranges non-overlapping), while their position
   distributions *do* overlap — every tail position falls inside the rank-1 range in both families.

### The amended primaries

| # | contrast | pairing | identified because | branch it tests |
|---|---|---|---|---|
| **1** | P(break \| group 0): **Llama against Qwen** | by sample | the cell is the same position in both families | **A** — concentration raises conditional severity. D60's arithmetic, if only the top group mattered, predicts about 41.8% against 18.6% |
| **2** | P(break \| least-attended group): **Llama against Qwen** | by sample | same cell, same selection rule, both families | **B** — more groups load-bearing in Llama puts the family difference here rather than at cell 1 |
| **3** | P(break \| rank 1) against P(break \| tail), **within** each family | by sample | shares separate completely (7.24× / 5.51×, no overlap) while positions overlap | whether attention mass predicts consequence **at all** among non-first groups |

All three keep D31's rule (p < 0.05 **and** a bootstrap CI excluding zero) and the sample-level pairing. Contrasts
1 and 2 pair across families on the same probe samples, which is legitimate because both arms draw the identical
subset — same `n_samples`, same seed — and each arm's break rate is measured against *its own* fault-free golden,
so no cross-model accuracy difference enters.

**Explicitly dropped:** every within-family claim involving rank 0. Position variance there is exactly zero and no
amount of analysis recovers an attention interpretation from a cell that is one fixed position.

**Residual confound, stated rather than fixed.** In contrast 3, rank 1 sits *earlier* than the tail in 60% of Qwen
samples and **82%** of Llama samples. So contrast 3 is share-separated but not position-balanced, and it is weaker
in Llama for that reason. The reversed subset — samples where rank 1 is later than the tail — is n = 16 (Qwen) and
n = 7 (Llama); the Qwen subset will be reported as a sensitivity check and the Llama one is too small to read,
which the record will say rather than reporting it anyway.

**A floor risk that follows from the same measurement.** Group 0 holds 76–83% of the mass, so every non-first cell
holds at most 5.4%. Contrast 3 may therefore be floor against floor, and D30/D44 makes that a **bound, not
equality** — at 320 trials the rule of three gives ≤ 0.93%. Contrasts 1 and 2 are not exposed to this, which is the
second reason they are the primaries.

**Cells changed accordingly:** the middle cell is now **rank 1** (`d61_{qwen,llama}_rank1.yaml`), replacing the
rank-2 cell, because rank 1 is the most-attended non-first group and is the strongest share contrast available
against the tail. Six arms, unchanged in every other respect.

### RESULT, 2026-09-26 — the third branch. Attention rank does not predict fault consequence, and D60's surviving hypothesis is refuted with power

Six arms, 06:52–07:17, 31 samples clean in all six, 248 trials per cell. `scripts/report_d61.py`,
`outputs/reports/d61_group_conditional.json`.

| cell | attention share | is group 0 | rel. position (sd) | k / n | break rate | Wilson 95% |
|---|---|---|---|---|---|---|
| qwen top | 0.759 | 1.000 | 0.000 (0.000) | 7/248 | **2.823%** | [1.374%, 5.711%] |
| qwen rank 1 | 0.060 | 0.000 | 0.651 (0.276) | 8/248 | 3.226% | [1.644%, 6.235%] |
| qwen tail | 0.011 | 0.000 | 0.365 (0.186) | 11/248 | **4.436%** | [2.495%, 7.767%] |
| llama top | 0.819 | 1.000 | 0.000 (0.000) | 9/248 | **3.629%** | [1.921%, 6.752%] |
| llama rank 1 | 0.047 | 0.000 | 0.663 (0.281) | 9/248 | 3.629% | [1.921%, 6.752%] |
| llama tail | 0.007 | 0.000 | 0.620 (0.290) | 14/248 | **5.645%** | [3.392%, 9.251%] |

The tail cells' realised rank is **17.0** against the 999 requested, so the wrap behaved and the records say so.

**Every one of the four pre-registered contrasts fails D31.**

| contrast | difference | 95% CI | p | samples moved | verdict |
|---|---|---|---|---|---|
| **1** — top, Llama against Qwen | +0.807% | [−2.016%, +4.032%] | 0.557 | 8/31 | fails |
| **2** — tail, Llama against Qwen | +1.210% | [−0.807%, +3.629%] | 0.317 | 6/31 | fails |
| **3** — Qwen, rank 1 against tail | −1.210% | [−2.823%, +0.403%] | 0.18 | 5/31 | fails |
| **3** — Llama, rank 1 against tail | −2.016% | [−5.645%, +1.613%] | 0.327 | 9/31 | fails |

This is the branch the pre-registration listed third: *"the rank gradient is flat in both families … the hazard is
not attention-mediated at all and would send the question back to the codec."* It is the outcome, and it was named
before the run.

**What is refuted, and it is refuted with power rather than by a null.** D60's surviving hypothesis put
P(break | top group) at about **18.6%** (Qwen) and **41.8%** (Llama) on the assumption that only the top group
matters. Measured: **2.823%** [1.374%, 5.711%] and **3.629%** [1.921%, 6.752%]. Both intervals **exclude** the
predicted values by a wide margin — factors of 6.6× and 11.5×. So this is not "we could not tell": the arithmetic
D60 recorded as its surviving hypothesis is wrong, and the assumption it rested on is the thing that fails.

**And the point estimates order the wrong way, consistently.** In *both* families the rate rises as the corrupted
group's attention falls — top < rank 1 < tail, over a **69×** range in attention share (0.759 → 0.011 on Qwen,
0.819 → 0.007 on Llama). Six cells, two independent families, the same monotone inversion. No single contrast
resolves it, so **this is a direction, not a result** (D31), and it is recorded as such. But it is the opposite of
every attention-mediated prediction anyone in this record has made, including mine.

**The mechanism this points at, stated as a hypothesis and not measured here.** Group 0 is the attention sink. A
sink absorbs attention mass without its *content* being read for information, so corrupting the sink's key scales
moves a large share of the attention weights and changes little about what the model retrieves. If that is right,
attention mass is simply the wrong covariate for fault consequence — the right one is something like how much the
answer depends on the values those weights multiply, which is not what any scorer in this project measures.

### The actionable consequence, which is the useful part

**A protection policy that spent its bits on the high-attention groups' metadata would be worse than uniform.** On
this probe the highest-attention group is the **least** damaging cell to corrupt in both families. That policy is
an obviously attractive refinement of the metadata guard — protect where attention is, save bits elsewhere — and it
is now closed off by measurement rather than by argument. The existing guard (H2d), which protects **all**
quantization metadata and costs +0.075% of storage, needs no change; what changes is that the cheaper
attention-weighted variant is not available.

This also closes the D40 allocator question from a new side. §6's multiplicative Joint Reliability Score is already
dead on GATE-1's grounds (D18: the two sensitivities are not one signal). D61 removes a *different* candidate
refinement — an importance-weighted protection term — on its own evidence.

### What this does NOT settle

**The original question is still open, and D61 narrows it rather than answering it.** Why Llama sits at the
metadata hazard ceiling (D59: 2.63% unpruned against Qwen's 1.17%) is not explained here. The cross-family
differences at matched cells run in the same direction — +0.81pp at the top, +1.21pp at the tail, ratios 1.29× and
1.27× — but neither is resolved at 31 samples, and both are far smaller than the 2.2× the uniform plan showed.
Three mechanisms are now eliminated (dynamic range, attention spread, attention-rank-conditioned severity) and
none of the three was the answer.

**Power.** 31 paired samples resolve differences of roughly 2–4 percentage points, so a residual gradient smaller
than that is invisible and is **not** shown to be absent — the same rule that makes a zero cell a bound. What the
design does have power for is the effect D60 predicted, which was an order of magnitude larger than its own
detection limit, and that one is excluded.

**Scope.** One probe, one group size (128), 4 bits, keys only, one query-aware scorer, single flips, no protection,
no pruning, eager attention throughout (so not interchangeable with D59's absolute rates). The position confound
on the rank-0 cell is unresolved *within* a family and always will be on this probe; the cross-family primaries
were chosen precisely because they do not depend on resolving it.

**One number worth flagging for anyone re-deriving this.** The position statistics here differ from D60's 40-sample
measurement (Qwen rank 1 at relative position 0.651 here against 0.473 there) because this is the 31-sample subset
all six arms answer cleanly, not the 40 the attention measurement used. The cells are the same construction; the
subsets are not.

### Three closing checks, one of which corrects a number reported above (2026-09-26)

Run in review, all on data already on disk, no GPU.

**1. The 31-sample intersection is not biased in context length, so the share range above is real.** If the samples
dropped from the intersection were disproportionately long, the 69× share range would be arithmetically true while
the *position* separation was compressed — a short cache has few groups and "tail" sits near "top". Measured: the
six-way shared set has mean `cache_len` **2204.1** against **2210.4** over the full 40, the 9 dropped samples mean
**2232.1**, Mann–Whitney **p = 0.897**, and mean groups per sample **18.00** against 18.05. The intersection is
representative.

**2. The inverse ordering does not meet a rule in aggregate either, and now it has numbers instead of an adjective.**
The pre-registration asked four per-cell questions and none asked whether the *ordering* is itself significant.
Added post-hoc and labelled so in `report_d61.py`:

* Spearman(attention share, break rate) over the six cells: **ρ = −0.754, p = 0.0835**
* Pooled top cells against non-top, 62 pairs with the families stacked: **3.226% → 4.234%**, difference
  **+1.008%** [−0.504%, +2.823%], **p = 0.349**
* Pooled counts: top **16/496 = 3.226%** [1.995, 5.175] against non-top **42/992 = 4.234%** [3.147, 5.673]

So the inversion stays a **direction**: suggestive at ρ = −0.75 on six points, not established at any conventional
threshold, and the intervals overlap. Recorded this way rather than as "unresolved", because a bound with a number
is worth more than an adjective.

**3. CORRECTION — primary 3 was computed on the wrong sample set, and on Qwen the magnitude differs 3.8×.**
`shared_samples` was called with all six arms, so a *within-family* contrast required the **other** family's arms to
answer a sample cleanly for it to count. That is stricter than any other contrast in this project and buys nothing.
On each family's own three-arm intersection:

| primary 3 | canonical (own family) | as first reported (six-way) |
|---|---|---|
| **Qwen**, n = 39 | tail 3.846% → rank 1 3.526%, **−0.321%** [−1.923%, +1.282%], p = **0.705** | −1.210% [−2.823%, +0.403%], p = 0.180 |
| **Llama**, n = 32 | tail 5.469% → rank 1 3.906%, **−1.563%** [−5.469%, +1.953%], p = **0.458** | −2.016% [−5.645%, +1.613%], p = 0.327 |

The verdict is unchanged — both fail D31 either way — but **the six-way set made the Qwen contrast look 3.8× larger
than that family's own data supports**, and that is the direction that would have overstated. The canonical numbers
are the own-family ones; both are printed by `report_d61.py` and stored in the JSON so the difference is visible
rather than buried. Primaries 1 and 2 are cross-family by construction and correctly use the six-way set.

The own-family cell rates, for completeness: Qwen top **2.564%** [1.305, 4.977] / rank 1 3.526% / tail **3.846%**
[2.214, 6.601] at 312 trials each; Llama top **3.906%** [2.135, 7.040] / rank 1 3.906% / tail **5.469%**
[3.285, 8.969] at 256 trials each. **The monotone inversion holds on these sets too** (weakly on Llama, where top
and rank 1 are equal), so nothing in the result above depends on the sample-set error.

**What none of the three changes.** D60's 18.6% / 41.8% prediction is excluded on either sample set — the top
cell's interval tops out at 5.7% (Qwen) and 7.0% (Llama). The actionable negative result stands: an
attention-weighted metadata protection policy would be worse than uniform.

## D62 — The graded measurement, pre-registered: is Llama's fragility per-flip severity or a thinner margin? (2026-09-26)

Frozen before either arm ran. D59 asked why Llama-3.2-3B's per-metadata-flip hazard is **2.63%** against Qwen3-1.7B's
**1.17%** on unpruned caches — a **2.25×** ratio. Three mechanisms are now eliminated and none was the answer:

* **dynamic range** (D59) — Llama's values carry 500× smaller MSE at equal SQNR, and a corrupted fp16 scale
  multiplies by up to 65536 either way, so from a smaller base that is a *smaller* excursion;
* **attention spread** (D60) — refuted with the sign reversed; Llama attends to **fewer** groups, not more;
* **attention-rank-conditioned severity** (D61) — no gradient at all, and the highest-attention group is the
  **least** damaging cell in both families.

### Why the previous design could not have answered it

D61's contrasts were underpowered for a structural reason rather than a design error: `answer_changed` runs at
**2–6%**, and 31 paired samples cannot resolve a 2× difference in a 3% rate whatever the pre-registration says. The
fix is not more samples — it is to stop thresholding. **The binary break event is a threshold crossing of a
continuous quantity**, and that quantity is measurable on the same trial for one extra forward pass on a cache
already built: the shift in the model's distribution over the **first answer token**.

`ProtectionArmTask.record_perturbation` now records, per trial, `kl_first_token` (KL(clean ‖ faulty) in nats, the
forward direction, so mass the clean model places where the faulty one abandons it is what counts — the failure
direction for greedy decoding), `margin_clean` and `margin_faulty` (the clean argmax token's lead over its best
rival under each cache), and `argmax_flipped`.

**The measurement has a zero baseline and it is exactly zero.** Control trials read the same fault-free cache the
reference came from, and their KL comes back at `0.00000000` on both smoke samples. Without that check, a KL of
1e-6 in a trial could be pipeline nondeterminism of unknown size rather than the effect of a flip; with it, every
nonzero value is signal. The project already demands exactly zero damage from controls on the binary outcome; this
holds the continuous one to the same standard.

### The two branches, and they are not the same kind of answer

The smoke run made the second one visible, which is why it is here: `margin_clean` is a property of the model's
**fault-free** answer, so the two are separable in one run.

| branch | claim | prediction | what it would mean |
|---|---|---|---|
| **A — per-flip severity** | the same corruption moves Llama's output distribution further | per-sample median `kl_first_token` higher on Llama, by a factor of roughly the **2.25×** hazard ratio if severity fully explains D59 | the mechanism is in the **codec**: a metadata flip does more damage there, and quantization design is the lever |
| **B — a thinner margin** | flips move both distributions comparably, but Llama holds its answers with less confidence, so the same shift crosses the threshold more often | `kl_first_token` distributions comparable, **`margin_clean` lower on Llama** | the mechanism is **not about quantization at all**: Llama is simply a less confident model on this probe, and the fault hazard difference is downstream of that. The right fix would be nothing in the allocator |

**A and B are not exclusive** and a partial split is the likely outcome; what the design buys is that they are
separated by *which of two recorded quantities* differs, not by argument.

### The frozen analysis

Two arms, `configs/experiments/d62_{qwen,llama}_meta.yaml`. Per arm: **40 samples × 16 trials = 640 flips**,
`n_flips: 1` (one flip per trial, so each outcome is one conditional), `fault_target: metadata` (uniform over the
arm's metadata **bits**, via the new `generate_metadata_plan` — deliberately not `generate_plan(n_metadata=…)`,
which is uniform over *regions* and induces a different distribution over bits whenever
`group_size != head_dim`), no protection, no pruning, eager attention so D62 sits in the same implementation as
D61 rather than straddling two.

1. **PRIMARY (branch A).** Per-sample median `kl_first_token`, Llama against Qwen, paired over shared samples,
   Wilcoxon plus a 10,000-resample bootstrap CI. D31's rule applies. Reported on the **log** scale as well as the
   raw, because KL over 640 flips is heavy-tailed and a mean would be one outlier's report.
2. **SECONDARY (branch B).** Per-sample `margin_clean`, which is fault-free and therefore one number per sample,
   Llama against Qwen, paired, same rule.
3. **ANCHOR, and it can void the whole design.** Within each family, `kl_first_token` on trials where
   `answer_changed` against trials where it did not. **If KL does not separate them, the graded measure is not
   measuring what breaks answers and primaries 1 and 2 are uninterpretable** — that is stated now so it cannot be
   quietly skipped later. Reported as an AUC with a bootstrap CI and as the median in each group.

`argmax_flipped` is recorded but is **not** the same event as `answer_changed`: a flip can leave the first token
intact and derail a later step. Both are kept so the gap is visible rather than assumed away.

### Scope, stated before the numbers exist

Logits come back in **bf16**, so `margin_clean` is quantized to about 0.25 and very small shifts are floor-limited
by the model's own output dtype. The KL keeps resolution because it sums over the whole vocabulary, and both
families are measured identically, so the cross-family contrast is unaffected — but no absolute margin here should
be read to more than two significant figures. One probe, one group size (128), 4 bits, single flips, first answer
token only. A flip that changes nothing at all is expected and observed (one smoke trial returned KL exactly zero
with a scale flip), consistent with D61's finding that a value whose code is 0 is invariant under scale corruption.

### RESULT, 2026-09-26 — branch B, and the answer is not in the codec at all

Two arms, 31 samples shared, 496 flips each. `scripts/report_d62.py`,
`outputs/reports/d62_flip_severity.json`; the scale control is `scripts/measure_answer_confidence.py` and
`outputs/reports/answer_confidence.json`.

**The zero baseline is exactly 0.0** on all 80 control trials in both families, so every nonzero KL below is the
effect of a flip and not pipeline nondeterminism.

**The anchor holds, decisively.** KL separates the flips that broke the answer from the flips that did not:
Qwen **AUC 0.999** [0.998, 1.000], p = 1.4e-08; Llama **AUC 0.885** [0.774, 0.978], p = 5.5e-09. On
`argmax_flipped` instead, 1.000 and 0.999. The graded measure is measuring what breaks answers, so the primaries
are interpretable — which is what the pre-registration said had to be true first.

#### A finding the pre-registration did not anticipate: about 1.3% of single metadata flips make the model output NaN

| family | non-finite | rate | Wilson 95% | share of all breaks | from a zero-point |
|---|---|---|---|---|---|
| Qwen3-1.7B | 6 / 496 | **1.210%** | [0.556%, 2.614%] | **35.3%** | 3 / 6 |
| Llama-3.2-3B | 7 / 496 | **1.411%** | [0.685%, 2.884%] | **25.9%** | **7 / 7** |

One corrupted fp16 metadata word can blow a key up until the attention softmax overflows. Every one of the 13
changed the answer, and in every case the prediction is a degenerate repetition (`""""""…`) rather than a wrong
answer. So **roughly a third of what "the answer changed" counts in every fault arm in this project is total
numerical collapse, not a subtly wrong answer** — a distinction no binary outcome could show, and one that matters
because a collapse is trivially detectable at runtime while a wrong answer is not. All 7 of Llama's came from a
**zero-point** flip against 3 of 6 on Qwen; with these counts that is a suggestion, not a result.

This class was invisible until the KL was computed, and it is what the first campaign's NaNs turned out to be.

#### PRIMARY 1 (branch A, per-flip severity) — significant, and **not** evidence for branch A

| | mean of per-sample medians | median over all finite flips | 10% | 90% | 99% | max | exactly zero |
|---|---|---|---|---|---|---|---|
| Qwen | 2.782e-03 | **3.085e-11** | — | 6.599e-06 | 6.301e-01 | 25.15 | **10.6%** |
| Llama | 1.740e-04 | **2.216e-05** | — | 1.770e-03 | 1.803e+00 | 13.59 | 2.7% |

The raw contrast and the log contrast point opposite ways, and **the log one is right**: raw
−2.608e-03 [−8.082e-03, +1.692e-04] fails D31 because Qwen's *mean* is carried by a handful of enormous flips
(its max, 25.15 nats, is nearly twice Llama's). On log10 the difference is **+4.503 dex [+3.480, +5.464],
p = 1.0e-07**, meeting D31 — the **typical** flip moves Llama's distribution about **3.2e4×** further than Qwen's.

**That is far past branch A's prediction of ~2.25×, and that is the reason to distrust it.** A ratio four and a
half orders of magnitude larger than the hazard ratio it is supposed to explain is not a stronger version of the
hypothesis; it is a sign the quantity is measuring something else. The scale control says what.

#### PRIMARY 2 (branch B, a thinner margin) — supported, and it survives the scale control

Fault-free margin: Qwen **17.07** against Llama **3.27**, difference **−13.80 [−16.13, −11.44]**, p = 1.2e-06,
**31 of 31** samples moved. Meets D31.

The scale control (`measure_answer_confidence.py`, 40 correctly-answered samples per family, medians):

| | margin | logprob spread | **margin / spread** | entropy | top-1 probability |
|---|---|---|---|---|---|
| Qwen3-1.7B | 21.000 | 4.515 | **4.657** | **0.0000** | **1.0000** |
| Llama-3.2-3B | 5.062 | 1.861 | **2.709** | 0.0970 | 0.9880 |
| ratio | 0.241× | 0.412× | **0.582×** | — | — |

About **40% of the raw margin gap is scale** — Llama's log-probabilities simply live on a narrower spread. But the
scale-free margin is still **0.582×**, so branch B survives: on this probe Llama holds its answers with a
materially thinner margin even after the units are removed.

#### What unifies the two, and it is the actual answer

**Qwen's first-token distribution is saturated: top-1 probability 1.0000 and entropy 0.0000 at float64.** Llama's
is not (0.9880, 0.0970). A distribution already at p = 1 has nowhere to move, so a small perturbation produces a
KL of ~1e-11 — until the perturbation is large enough to unseat the top token, at which point the KL jumps to tens
of nats. That is exactly the shape of Qwen's distribution: **10.6% of its flips produce a KL of exactly zero**
against Llama's 2.7%, its median is 1e-11, and its maximum is the largest in either family.

So **primary 1 does not establish branch A.** Qwen's small typical KL is a consequence of having no room to move,
not evidence that a metadata flip does less damage there, and the KL comparison cannot separate the two. Branch A
is recorded as **not established**, not as refuted.

One more piece of the same picture, from the descriptive table: on Qwen, **11 breaks and 11 first-token flips** —
every break is the first token being unseated. On Llama, **20 breaks but only 7 first-token flips** — 13 of 20
derail later, with the first token intact. A saturated first token cannot be nudged, only overturned; an
unsaturated answer can come apart anywhere.

#### The answer to D59's question, stated at the confidence it has earned

**The fragility difference is a property of how confidently each model holds these answers, not of what
quantization does to their caches.** Four mechanisms have now been tested: dynamic range (eliminated, D59),
attention spread (refuted with the sign reversed, D60), attention-rank-conditioned severity (no gradient, D61),
and per-flip severity (**not established** here, confounded by saturation). What is left standing and measured is
that Qwen answers this probe at p = 1.0000 and Llama at 0.9880, and a flip has to do far more work to change the
first than the second.

**This is a claim about this probe, and that is a real limit.** A saturated output distribution is a property of
model × task, not of the model; on a harder benchmark Qwen would not sit at p = 1 and the whole difference could
vanish or reverse. RULER or LongBench is the test, and it is not run. **What it does settle for the allocator is
negative and useful: there is no codec-side lever here.** Nothing in D62 suggests a quantization or protection
policy that would close the gap, because the gap is not made by the codec.

#### Scope and corrections

* The control's margins (Qwen 21.000, Llama 5.062) are medians over each family's own first 40 correctly-answered
  samples; the arms' (17.07, 3.27) are means over the 31 samples **both** families answer cleanly. Different sets,
  same direction, and neither is the other's replication.
* `entropy` and `top1_prob` **saturate** at float64 for Qwen, so its 0.0000 and 1.0000 are floor and ceiling
  rather than measurements. The margin does not saturate — it is a difference of two finite log-probabilities —
  so it stays the quantity to read.
* bf16 logits quantize the margin to about 0.25; two significant figures at most.
* The first campaign returned NaN for most trials and was superseded, not deleted. Its `margin_clean` and outcome
  flags were correct and unaffected — primary 2 already read the same result off them — but a record carrying a
  NaN in a field the report reads is not something to leave live.
* One probe, one group size, 4 bits, single flips, first answer token, 496 flips per family.

### Three closing controls, and the third is stronger than the result it checks (2026-09-26)

All on data already on disk.

**1. Seed stability.** Re-run at `--seed 1` and `--seed 7`: the AUCs are identical to three decimals (0.999 and
0.885), primary 2 is identical to four (−13.8044), and the log10 contrast is **+4.503 dex** in all three runs. Only
the bootstrap CI bounds move, in the third decimal (Llama's AUC lower bound 0.774 / 0.771 / 0.768). Nothing here
rests on a draw.

**2. The collapse originates in the cache, not in the model's arithmetic.** The trials with a non-finite KL are
**exactly** the trials with a non-finite `max_relative_change` — 6 of 6 on Qwen, 7 of 7 on Llama, the same trials in
both columns. `max_relative_change` is measured on the dequantized stored values before the model runs, so the
corrupted metadata word made the *cache* non-finite and the forward pass merely carried it. That is a cleaner
statement than "the softmax overflowed": nothing in the model overflowed, it was handed an infinity.

**3. THE EXPOSURE IS MATCHED AND THE RESPONSE IS NOT, which is the strongest form of the result.**

The fault model delivers almost exactly the same value-space perturbation to both families, as identical KV geometry
predicts: median `max_relative_change` **0.09661** (Qwen) against **0.09149** (Llama), p90 = **3** for both, maximum
**65536** for both — which is 2^16 exactly, the largest factor an fp16 exponent flip can produce, so the fault
model's arithmetic is confirmed at its own ceiling.

Binned on that exposure, with comparable counts in every bin:

| `max_relative_change` | Qwen n / median KL | Llama n / median KL |
|---|---|---|
| [0.001, 0.1) | 221 / **2.56e-11** | 214 / **1.57e-05** |
| [0.1, 1) | 141 / **9.44e-11** | 152 / **1.07e-05** |
| [1, 1000) | 85 / **2.88e-11** | 69 / **3.84e-05** |

**Qwen's KL does not respond to its own exposure.** Across a **million-fold** range of stored-value perturbation its
median KL stays at 2.6e-11, 9.4e-11, 2.9e-11 — flat, within a factor of four, with no trend. Llama's rises with
exposure and sits five to six orders of magnitude higher in every bin.

This is what settles the interpretation. Because the exposure is matched by construction, the difference cannot be
attributed to what quantization did to the cache — that quantity is held equal and measured. What differs is
entirely on the output side, and a response that is **flat against a million-fold change in its input** is the
signature of a saturated distribution, not of a robust one. It is the same conclusion as the entropy and top-1
measurements reached independently, and it is the version a reviewer should be shown, because it needs no appeal to
float64 saturation limits: the flatness is visible in the KL itself.

**So branch A is not merely "not established" — the control that would have supported it argues against it.** At
matched exposure the codec has been equalised and the families still differ, which locates the difference outside
the codec. Per-flip severity in the sense branch A meant it — *the same corruption doing more damage because of how
it is stored* — has no room left to be true here.

**A limit of control 3 worth stating.** The bins are wide (three orders of magnitude in the first two) and matched
on `max_relative_change`, which is the largest single value change in the plan, not the whole perturbation's shape.
Two flips with the same maximum can differ in how many values they moved. A tighter version would match on the
number of affected values as well; the counts per bin (221/214, 141/152, 85/69) show the *distributions* align, not
that each pair of flips is equivalent.

### Two more controls, one of which is a limitation of D62's own mechanism claim (2026-09-26)

Both on D62's existing records, no GPU.

**1. Context length is not the difficulty dial, so testing the saturation limit needs a harder TASK.** The obvious
cheap route to an unsaturated Qwen was the probe's own 4096-token condition. It does not get there. The margin does
fall with cache length in both families, in the same direction and resolvably in neither at n = 31 — Qwen
ρ = **−0.292** (p = 0.111), Llama ρ = **−0.250** (p = 0.174):

| cache_len | Qwen median margin (min) | share below 1 nat | Llama median margin (min) | share below 1 nat |
|---|---|---|---|---|
| 512–800 | 19.75 (9.88) | **0.00** | 5.72 (1.50) | 0.00 |
| 800–1600 | 19.50 (17.25) | **0.00** | 3.12 (0.12) | 0.22 |
| 1600–3000 | 16.25 (7.00) | **0.00** | 1.75 (0.38) | 0.20 |
| 3000–5000 | 13.25 (1.00) | **0.00** | 2.50 (0.00) | **0.36** |

**Qwen has no sample below 1 nat at any length.** Eight times the context costs it about 6 nats and leaves it at 13 —
still saturated by any standard that matters here. Llama is below 1 nat on 23% of samples overall and 36% at the
longest. So `raq_probe_v0_long` cannot serve as the desaturated condition, and the scope limit D62 stated has to be
tested against a harder **task**, not a longer one. That settles a question that would otherwise have been answered
by assumption.

**2. LIMITATION. The margin story is strong across families and weak within them.** If a thin margin is the
mechanism, then within a family the thin-margin samples should break more. They do, by about 2×, and it is not
resolvable at this size:

| family | break rate, thin-margin half | thick-margin half | Spearman(margin, break rate) |
|---|---|---|---|
| Qwen | **4.412%** | 2.232% | ρ = +0.000, p = 1 |
| Llama | **7.031%** | 3.750% | ρ = −0.109, p = 0.558 |

The ratio is in the predicted direction in both families (1.98× and 1.88×) and neither correlation approaches
significance. **So D62's cross-family result — 17.07 against 3.27 nats at p = 1.2e-06 — is solid, and the
within-family dose-response that would establish margin as the *mechanism* rather than a correlate is not.** The
break rates are 2–7% on 31 samples, which is the same power wall D61 hit and the reason the graded KL exists; the
difference is that the graded measure has no within-family analogue here, because `margin_clean` is one number per
sample and cannot be enriched by adding trials.

This is recorded as a limitation rather than folded into the result. The honest statement of D62's mechanism claim
is: **the two families differ enormously in how confidently they hold these answers, that difference is in the
right direction to explain the hazard gap, and the within-family dose-response is consistent with it at about 2×
but underpowered.** A proper test needs many more samples at fixed model, which is cheap — it needs no faults, only
`margin_clean` and a break rate — and is not run.

## D63 — Does difficulty desaturate Qwen? The limit D62 stated, pre-registered as a controlled ladder (2026-09-26)

Frozen before the run. D62 answered why Llama-3.2-3B's per-metadata-flip hazard is 2.25× Qwen3-1.7B's: **Qwen holds
these answers at top-1 probability 1.0000 with entropy 0.0000 and Llama at 0.9880**, so a fault must do far more
work to unseat the first. It stated its own limit plainly — *saturation is a property of model × task, so on a
harder task the whole difference could vanish or reverse* — and this tests it.

### Why not LongBench, and why not length

**Length is not the dial, and that is measured rather than assumed** (D62's controls): Qwen has **no sample below 1
nat at any context length**, and eight times the context costs it about 6 nats and leaves it at 13. So
`raq_probe_v0_long` cannot serve as the desaturated condition.

**LongBench was the plan and is deferred, for a reason that is about the measurement and not the cost.** It changes
difficulty, domain, answer format and scorer all at once, so a desaturation observed there could not be attributed
to difficulty. Two existing vehicles were considered and rejected on the same grounds, one of them decisively:
**GSM8K's first generated token is not an answer token at all** — it opens a reasoning chain, and the answer arrives
300 tokens later — so the quantity D62 measured does not transfer to it. BFCL's first token is a tool-call opener
(`<tool_call>`, `<|python_tag|>`), which both families emit with high confidence regardless of whether they get the
call right, so it would likely be *more* saturated for a reason unrelated to task difficulty.

What this vehicle can vary while holding everything else identical is the number of competing **same-type**
distractor facts. Generator seed, filler corpus and revision, fact types, depths, answer format and scorer are
unchanged across rungs; only the number of facts of the same type and template with different keys moves. That is
the manipulation, and it is the one a confidence claim needs: more plausible candidates for the same query.

### The ladder, and the ceiling it ran into

`configs/datasets/raq_probe_v0_d{2,10,20}.yaml`: **2, 10 and 20** distractors, context lengths pinned to
**[2048, 4096]** for both rungs so length is held fixed.

The ceiling was measured, not guessed. The generator refuses when the filler cannot host the facts at distinct
depths (`filler has too few sentences for distinct depths and distractors`): at 512 tokens even **6** fails; at
[1024, 2048, 4096] **10** works and **14** fails; at [2048, 4096] **20** works and **30** fails. Hence the pinned
lengths and the 20-distractor top rung — a tenfold range, which is what the ladder gets.

### The anchor, which can void the comparison

**Accuracy per rung.** A rung on which accuracy does not move did not make the task harder, and a flat margin there
would say nothing about saturation. It is printed beside every cell and stated here so it cannot be skipped later —
the same role the AUC plays in D62. If accuracy is unchanged from d2 to d20 on a family, that family's margins
across the ladder are uninterpretable and will be reported as such rather than as a null.

### Pre-registered predictions

| branch | prediction | what it would mean |
|---|---|---|
| **A — saturation is task-driven** (D62's own stated limit, so the expected outcome) | Qwen's margin falls monotonically with distractors and its top-1 probability drops below 1.0; some Qwen samples appear below 1 nat, where currently **none** do at any length | D62's answer is **probe-specific**. The fragility ordering could vanish or reverse on a harder task, and the honest scope of D62 narrows to "on tasks where this model saturates" |
| **B — saturation is model-driven** | Qwen stays at p = 1.0000 with no sample below 1 nat even at 20 competing facts, while accuracy falls (so the rung did bite) | D62's answer **generalises further than it claimed**. A model that answers at p = 1 whatever the difficulty is structurally more fault-tolerant, and that is a property worth naming |
| **void** | accuracy does not move from d2 to d20 | the manipulation failed; nothing about saturation is learned and a harder *vehicle* is then unavoidable |

**Both A and B are informative and neither is the result I want**, which is the point of writing them down first.
Branch A narrows a conclusion I committed to two hours ago; branch B widens it. The measurement is the same either
way.

### Scope

First answer token only, 40 samples per cell spread across both pinned lengths, the same 4-bit `group_token`/
`per_token` quantization the D62 arms used, no faults, no pruning. Accuracy is scored by the probe's own scorer, so
"correct" means what it means everywhere else in this project. `entropy` and `top1_prob` saturate at float64 for a
distribution at p = 1 — which is the condition being looked for, so they are ceiling readings there and `margin`,
a difference of two finite log-probabilities, stays the quantity to read.

### RESULT, 2026-09-26 — VOID by its own anchor: the knob did not make the task harder

Six cells, 18 minutes, 40 samples attempted per cell. `outputs/reports/d63_difficulty_ladder.json`.

**The anchor fails, and the pre-registration said this outcome exists.** A rung on which accuracy does not move did
not make the task harder, and D63 declared in advance that its margins would then be uninterpretable.

| rung | Qwen accuracy | Wilson 95% | Llama accuracy | Wilson 95% |
|---|---|---|---|---|
| d2 | 40/40 = 1.000 | [0.912, 1.000] | 33/40 = 0.825 | [0.681, 0.913] |
| d10 | 39/40 = 0.975 | [0.871, 0.996] | 34/40 = 0.850 | [0.709, 0.929] |
| d20 | 39/40 = 0.975 | [0.871, 0.996] | 31/40 = 0.775 | [0.625, 0.877] |

d2 against d20: Fisher **p = 1.000** on Qwen (a one-sample change) and **p = 0.781** on Llama, whose accuracy is not
even monotonic. **A tenfold increase in competing same-type facts did not measurably reduce accuracy for either
family.** The margins do not move either — Kruskal over the three rungs gives p = 0.478 (Qwen) and p = 0.556
(Llama) — which is exactly what a manipulation that did not bite should produce, and is therefore no evidence about
saturation in either direction.

| rung | Qwen margin | top-1 p | below 1 nat | Llama margin | top-1 p | below 1 nat |
|---|---|---|---|---|---|---|
| d2 | 16.500 | 1.0000 | 0.000 | 2.500 | 0.9140 | 0.303 |
| d10 | 17.250 | 1.0000 | 0.026 | 3.188 | 0.9503 | 0.088 |
| d20 | 19.500 | 1.0000 | 0.000 | 3.125 | 0.9477 | 0.161 |

**So branch B is NOT supported and branch A is NOT refuted.** Qwen staying at p = 1.0000 across the ladder would
have meant something only if the ladder had been a difficulty gradient, and it was not. D62's stated limit —
saturation is a property of model × task — **remains untested**, and recording it as tested would be the single
most tempting error available here: the numbers look exactly like branch B.

### Why the knob was wrong, which is the useful part

**Qwen answers 97.5–100% correctly at every rung.** The task is trivially easy for it whatever the distractor count,
so there was no difficulty to vary. Llama sits at **77.5–85.0%** — the task is already much harder for it — so the
difficulty axis this vehicle needs does exist, it just does not reach Qwen. Twenty facts of the same type and
template with different keys is apparently not a retrieval problem for a model that can key-match.

That converts the case for a harder benchmark from an assumption into a measurement: **the next vehicle has to be
one Qwen itself finds hard**, and the probe cannot be made into one by adding distractors. LongBench (or RULER at a
difficulty where Qwen's accuracy actually falls) is now justified by this result rather than by argument, and the
rejected-vehicle reasoning in the pre-registration stands — GSM8K's first token is not an answer token, BFCL's is a
tool-call opener both families emit confidently.

### What the ladder does deliver: D62's cross-family gap, three more times

Not what it was built for, and worth having. At pinned long lengths (2048 and 4096, mean cache ≈ 3089 against
D62's 2204) and at three distractor counts:

| rung | Qwen margin | Llama margin | Mann-Whitney | logprob spread ratio |
|---|---|---|---|---|
| d2 | 16.50 | 2.50 | **p = 2.7e-11** | 0.406× |
| d10 | 17.25 | 3.19 | **p = 3.3e-09** | 0.408× |
| d20 | 19.50 | 3.12 | **p = 3.9e-12** | 0.417× |

Three independent replications of the finding D62 rests on, at longer contexts than D62 measured, with the
margin/spread ratio at 0.391×, 0.463×, 0.401× — so the scale-free gap holds too. **The logprob spread ratio is
stable to within 3% across the whole ladder (0.406, 0.408, 0.417)**, which makes the scale component of the gap a
robust property of the two models rather than a feature of one measurement. And Qwen shows **0 of 40, 1 of 39, 0 of
40** samples below 1 nat, against Llama's 30.3%, 8.8%, 16.1%.

### Scope

First answer token, 4-bit `group_token`/`per_token` quantization, no faults, no pruning, one probe, one seed.
The ladder's own ceiling was measured before the run: the generator refuses more than 20 distractors even at these
lengths, so this is the full range the vehicle offers rather than a chosen subset of it.

### PILOT, 2026-09-26 — LongBench desaturates Qwen completely, so D62's answer is probe-specific

**Status of this measurement, stated first because it decides how much weight it can carry.** A throwaway pilot, not
a committed vehicle: 12 non-randomly chosen rows of LongBench `2wikimqa`, LongBench's own prompt template verbatim,
its own `qa_f1_score`, **Qwen3-1.7B only**, and **sdpa rather than eager** attention. The script lives in the
session scratchpad and is not in the repo, so this carries no exp_id and is **not** a result of the kind D62 or D63
report. It is evidence about whether to build the vehicle, and it is decisive enough to correct a scope claim.

| | RAQ-Probe ladder (d2–d20) | LongBench 2wikimqa |
|---|---|---|
| accuracy | **97.5–100%** exact match | F1 mean **0.071**, median **0.000**, **75%** score exactly 0, none above 0.5 |
| cache length | 2047–4179 (median ~3089) | median **7735** |
| margin, median | **16.50 / 17.25 / 19.50** | **2.875** (min 0.250) |
| top-1 probability | **1.0000** at every rung | **0.9284** (min **0.5445**) |
| share below 1 nat | 0.000 / 0.026 / 0.000 | **0.25** |

**Qwen desaturates completely, and the anchor D63 demanded is satisfied here where it failed there.** The task is
genuinely hard for it — three quarters of its answers share no token at all with any reference — and its top-1
probability leaves the ceiling for the first time in this project: 1.0000 on every probe rung against 0.9284 here,
with one sample at 0.5445.

**So D63's branch A holds and D62's answer is probe-specific.** Saturation is a property of model × task, exactly as
D62 said it might be, and it is now measured rather than conceded. The striking number is that **Qwen's LongBench
margin (2.875) is essentially Llama's RAQ-Probe margin (3.27)**: on a hard task Qwen looks like Llama did on an easy
one. That is what a shared difficulty–confidence curve would produce, with the two families sitting at different
points on it rather than having different shapes.

**What this does and does not do to D62.** It does not touch D62's internal findings — the matched-exposure control,
the flat KL response, the non-finite collapse rate, the eliminated mechanisms all stand on their own data. What it
narrows is the *generality* of the answer: "the fragility difference is output saturation" is a statement about a
regime where one model saturates, and that regime is the probe. Whether the fault-hazard **ordering** reverses where
the saturation ordering closes is a different question and is **not** measured here — it needs fault arms on this
vehicle, and Llama's half of even the confidence measurement is still missing.

### Three costs of this vehicle, measured rather than estimated

1. **Eager attention is impossible.** At these lengths a single eager softmax asked for **14.45 GiB** and the run
   died with CUDA OOM. So the attention-scorer path (D5) cannot be used on LongBench at all, which puts every
   D60/D61-style attention-ranking experiment out of reach there and leaves only the logit-based measurements.
2. **Roughly 2 minutes per sample** on Qwen3-1.7B at a median 7735-token cache — 12 samples took 25 minutes. A
   40-sample two-family confidence cell is therefore a couple of hours, and fault arms with 16 trials per sample
   would be a different order of cost than the probe's 3–6 minutes per arm.
3. **Contexts are 2–4× the project's range.** The 12 rows span 5.2k–15.6k tokens against the probe's 512–4096. So a
   LongBench vehicle extends the length regime as well as the difficulty, and the two cannot be separated within
   it — though D62's length control (Qwen stays saturated to 4096) argues the difficulty is doing the work.

### A trap worth recording

The pilot was launched under `timeout 1500` and **Qwen alone consumed the whole 25 minutes**, so Llama never ran —
and the shell reported **exit code 0**. A truncated run that looks successful is the same class of hazard as the
`--set task.n_samples=2` smoke run that shared a config name with its real arm: the output is not wrong, it is
*incomplete in a way nothing announces*. The missing half was caught by noticing the absent second block, not by
any check.

### PILOT, part two — both families desaturate, the ordering survives, and confidence comes apart from accuracy

Llama-3.2-3B on the same rows, **10 of 12** completed: the run died with CUDA OOM on row 11, a 15574-token context,
after 44 minutes. sdpa asked for **19.36 GiB** there, so the memory-efficient path is not memory-efficient enough at
that length on a 16 GB card — the same wall eager hit at 7k, moved out to 15k.

| | RAQ-Probe | LongBench 2wikimqa |
|---|---|---|
| **Qwen margin** (median) | 16.50 / 17.25 / 19.50 | **2.875** |
| **Llama margin** | 2.50 / 3.19 / 3.12 | **0.938** (min 0.125, **50%** below 1 nat) |
| **Qwen top-1 p** | **1.0000** at every rung | **0.9284** |
| **Llama top-1 p** | 0.9140 / 0.9503 / 0.9477 | **0.5350** (min 0.1755, **90%** below 0.9) |
| **Qwen accuracy** | 97.5–100% exact match | F1 mean **0.071**, 75% exactly 0 |
| **Llama accuracy** | 77.5–85.0% exact match | F1 mean **0.311**, 60% exactly 0, 30% above 0.5 |

**Both families desaturate, and the ordering survives.** Qwen is still the more confident of the two — margin 2.875
against 0.938, top-1 probability 0.928 against 0.535 — so the saturation ordering D62 rests on is not an artefact of
the probe. What changes is the *size* of the gap: the margin ratio goes from **0.152–0.185×** on the probe to
**0.326×** here, so the two families converge without crossing.

### The unexpected part, and it strengthens D62's mechanism rather than weakening it

**On the probe Qwen was both more accurate and more confident, so confidence could be read as "Qwen finds this
easier". On LongBench Qwen is LESS accurate and STILL more confident** — F1 0.071 against Llama's 0.311, while
holding its answers at four times the margin. Accuracy and confidence come apart across the two families, and they
come apart in the direction that rules out the competence reading.

So the honest mechanism statement gets sharper: **Qwen's fault tolerance comes from calibration, not competence.**
Its answers sit further from the decision boundary whether or not they are right, and a KV corruption has to travel
that distance before the answer changes. That is why D62's hazard difference exists, and it is not a codec property
in any regime measured.

**A caution this implies for the field, and for this project's own metric.** "Did the answer change under fault" —
the outcome every arm here uses — **rewards overconfidence**. A model that commits hard to a wrong answer scores as
more reliable than one that holds a right answer tentatively. On the probe that distinction was invisible because
the more confident model was also the more accurate one; here it separates, and Qwen wins the reliability metric on
the task it is worse at. Any reliability comparison across models that does not report calibration alongside is
measuring something other than what it claims.

### Status and limits of this pilot, stated plainly

* **A pilot, not a result.** Scratchpad script, no exp_id, no committed vehicle, sdpa rather than eager.
* **Unpaired.** Qwen's numbers are over 12 rows, Llama's over the first 10; both runs were truncated by their own
  failure mode (Qwen by a 25-minute shell timeout, Llama by OOM). The top-1 probability difference (0.928 against
  0.535) is far too large for the mismatch to explain, but no interval is computed and none should be quoted.
* **Neither accuracy figure is resolvable** at 10–12 samples. That Llama's F1 exceeds Qwen's is the direction, not a
  demonstrated difference, and the dissociation argument rests on it — so it is the first thing a committed vehicle
  should settle.
* **Nothing here measures fault hazard.** Whether the hazard ordering follows the confidence ordering on a hard task
  is still unmeasured, and it is the question that would close D59's line properly.

### What a committed vehicle would have to do

**Filter by length, and say so as a condition.** A 16 GB card cannot run 2wikimqa's longest rows on the 3B model at
all, so a vehicle must select rows whose context fits — roughly under 8k tokens on this hardware — and that
selection is a stated condition of the result, not a convenience. It also means the vehicle cannot be used for the
attention-scorer path at any length (eager OOMs at 7k), so retention experiments with query-aware scoring are out of
reach on it, and only logit-based and fault work is available.

### PILOT, part three — the predictions, read rather than scored, and one real caveat

The pilot scored 22 predictions and discarded every string, so both of its claims were made without a single answer
being looked at. Reading four of them (the shortest rows, both families) settles two questions the F1 number could
not and raises one caveat that was invisible in the aggregate.

| row | gold | Qwen prediction (first token) | Llama prediction (first token) |
|---|---|---|---|
| 11 | Marie of Hohenstaufen | *"Marie of Brabant, Queen of France's paternal grandmother was Margaret of Brabant."* (**'Mar'**, 19 tok) | *"Beatrice d'Avesnes"* (**'Beat'**, 7 tok) |
| 8 | St Patrick's College | *"Central School"* (**'Central'**, 3 tok) | *"St Patrick's College, Dublin"* (**'St'**, 7 tok) — **correct** |
| 2 | Cahiers du cinéma | *"The director of film Man At Bath is Christophe Honoré, and he works at the Opéra National de Lyon."* (**'The'**, 25 tok) | *"France"* (**'France'**, 2 tok) |
| 5 | Edward Watson | *"Lewis Watson, 1st Earl of Rockingham"* (**'Lewis'**, 11 tok) | — |

**1. The low F1 is genuine retrieval failure, not prompt-following.** Every prediction is short and terminated on
EOS, and Qwen's errors are *plausible-but-wrong entities lifted from the context* — Marie of **Brabant** for Marie of
**Hohenstaufen**, Lewis Watson for Edward Watson, Christophe Honoré for Cahiers du cinéma. That is what multi-hop
failure looks like. The worry that `qa_f1`'s `overlap / len(pred)` precision was punishing a correct answer wrapped
in preamble does not hold: in these four the answer is simply absent.

A consistency check passed on the way: Llama's *"St Patrick's College, Dublin"* against gold *"St Patrick's College"*
gives F1 = 0.857 by hand, exactly the value the pilot recorded for that sample.

**2. The margin is measured on an answer-bearing token, mostly.** Six of the seven first tokens are content words
('Mar', 'Central', 'Lewis', 'Beat', 'St', 'France'). So the LongBench margin is broadly the same object as the
probe's, which the cross-task comparison needs and which had not been checked — only the *gold* answers' first-token
diversity had been.

**3. THE CAVEAT, and it is real.** Qwen answers two of four rows as a **full sentence** rather than a bare value
(19 and 25 tokens, one of them opening with **'The'**), where all three of Llama's are bare (7, 7, 2 tokens). Two
consequences:

* **Qwen is the less compliant of the two with "only give me the answer",** and that depresses its F1 precision
  relative to Llama's on any row where it *does* find the right entity. From four rows this cannot be sized, so
  **the accuracy inversion (Llama F1 0.311 against Qwen 0.071) is weaker than the raw numbers suggest** and is the
  first thing a committed vehicle must settle — with a format-insensitive score, or a compliance rate reported
  beside it.
* **Roughly one in four of Qwen's LongBench margins may sit on a function word.** This cuts the *conservative* way
  for the desaturation finding: function words are highly predictable, so contamination would inflate Qwen's margin,
  and Qwen's measured collapse from 1.0000 to 0.9284 is therefore if anything understated.

**What survives unchanged and what does not.** The **confidence** half stands and is strengthened: Qwen desaturates
(top-1 1.0000 → 0.9284), Llama further (→ 0.5350), the ordering survives, and the function-word caveat works against
rather than for that reading. The **accuracy inversion** — and with it the "calibration, not competence" framing and
the "the fault metric rewards overconfidence" caution — is **downgraded to a hypothesis** pending a format-robust
accuracy measurement. It was recorded as a standing project constraint before this check; that entry now carries the
caveat.

**The lesson is the same one this project keeps relearning.** A number computed from text that nobody read is a
number about the scorer as much as about the model. Four predictions and ten minutes were enough to find a confound
that the aggregate hid completely, and the check happened only because the claim had already been written down and
was noticed to be resting on nothing.

### PILOT, part four — the accuracy inversion survives a format-insensitive score, and stays underpowered

Part three found Qwen answering two of four rows as a full sentence where Llama's were bare, and downgraded the
accuracy inversion on the grounds that `qa_f1`'s `overlap / len(pred)` precision would punish that. The test of
whether that confound explains the inversion is a score with no denominator in the prediction at all: **does any
gold answer appear verbatim in the output.** Eight shortest rows, both families.

| | containment (format-insensitive) | Wilson 95% | bare answers (≤ 8 tokens) | median prediction |
|---|---|---|---|---|
| Qwen3-1.7B | **0 / 8 = 0.000** | [0.000, 0.324] | 2 / 8 = **0.250** | 11 tokens |
| Llama-3.2-3B | **2 / 8 = 0.250** | [0.071, 0.591] | 6 / 8 = **0.750** | 7 tokens |

**The confound is confirmed and it does not explain the inversion.** Qwen is indeed the less compliant model —
bare on 25% of rows against Llama's 75%, Fisher p = 0.132 — but a containment score has no prediction-length
denominator, and Qwen still scores **zero**. Verbosity cannot turn a found answer into an absent one. Qwen's errors
are uniformly wrong entities drawn from the context: *West…* for Bob Jones University, *January…* for 16 September
1360, *Her…* for Ozalj, *2…* for 1510.

**And it stays underpowered, as it must at this size.** Fisher on 0/8 against 2/8 gives **p = 0.467**, and Qwen's
zero is a **bound** — rule of three ≤ 0.375, Wilson [0.000, 0.324] — not a demonstrated floor (D30/D44). So the
status is unchanged in kind and stronger in support: **the inversion is still a hypothesis, but the specific
mechanism that would have explained it away has been tested and does not.**

**What this settles and what it does not.** The "calibration, not competence" framing and the caution that "answer
changed under fault" rewards overconfidence both rest on Qwen being *less accurate while more confident* on a hard
task. That pattern now survives the one confound identified against it, at n = 8, with no interval separating the
families. It needs a committed vehicle with enough rows, and until then it should be cited as a direction with its
n attached — which is how the project memory entry now carries it.

The confidence half is untouched and remains the solid part: Qwen top-1 probability 1.0000 on the probe against
0.9284 here, Llama 0.914–0.950 against 0.5350, ordering preserved, and the function-word caveat from part three
cutting conservatively against the desaturation reading rather than for it.

## D64 — The agent arms fail by losing the tool, never by mis-transcribing it (2026-09-26)

D50 recorded that `named_gold_function` "was built for this and is not yet analysed": whether the deployable arms
fail by **picking the wrong function** or by **filling its arguments wrong**. The field has been in every M6 BFCL
record since the vehicle was built. Analysed now, with no GPU — `scripts/report_m6_failure_modes.py`,
`outputs/reports/m6_failure_modes.json`.

Conditioned on the calls each arm's own golden answers correctly, so the vehicle's floor is excluded rather than
charged to compression.

| arm | n | correct | failures | **wrong function** | wrong arguments | arguments' share, Wilson 95% |
|---|---|---|---|---|---|---|
| `m6_bfcl_golden` (no compression) | 120 | 105 | 15 | 7 | **8** | **0.533** [0.301, 0.752] |
| `m6_bfcl_joint` (query-aware, span) | 105 | **102** | 3 | 2 | 1 | 0.333 [0.061, 0.792] |
| `m6_bfcl_prefixcache` (query-agnostic) | 105 | 20 | 85 | **85** | **0** | **0.000** [0.000, 0.043] |
| `m6_bfcl_magnitude` (key magnitude) | 105 | **0** | 105 | **105** | **0** | **0.000** [0.000, 0.035] |
| `m6_bfcl_recency` | 105 | 24 | 81 | **81** | **0** | **0.000** [0.000, 0.045] |

**Across the three failing arms, 0 of 271 failures are argument errors.** The upper bounds are 3.5–4.5%, so this is
a measured near-absence rather than a small count. The mode is effectively binary: when retention drops the gold
schema the model names a *different* tool, and it never names the right tool and then botches its arguments.

**The contrast with the golden arm is the sharp part.** With no compression at all, the model's own failures split
roughly evenly — 8 of 15 with the right function named. So argument errors are a real failure mode of this vehicle;
they are simply **not** what compression causes. Compression damage is entirely tool *selection*.

### What this settles

**The compression axis owns all of the agent-level damage, and quantization fidelity owns none of it here.** Every
arm ran at the same 4-bit `group_token`/`per_token` quantization, and if that quantization were degrading surviving
schemas enough to corrupt an argument, argument errors would appear among the failures. They do not appear at all.
So on this vehicle the lever for agent reliability is *what gets kept*, not *how precisely it is stored* — which is
the opposite of where the fault-side work (D25–D30, metadata protection) points, and they are consistent because
they concern different perturbations: pruning removes a schema, a metadata flip rescales a group.

**`m6_bfcl_magnitude` fails 105 of 105.** The `magnitude_only` policy does not merely underperform, it never
selects the gold schema on any sample. That is the strongest statement yet of the key-magnitude result: D42 called
it "at chance", D47 corrected that to an **anti-signal**, and here the anti-signal is total. A policy ranking tool
schemas by key magnitude is worse than random at keeping the one the question needs.

**H3's span finding gains a mechanism.** `m6_bfcl_joint` answers 102 of 105 — 3 failures against prefixcache's 85 —
and the difference is entirely in whether the gold schema survives. Query-aware span retention works on this vehicle
because it keeps the right *schema*, not because it preserves argument detail better.

### What it does not settle

**This is one model, one BFCL category, one catalogue setting.** The Llama BFCL arms do not carry the field split
because the second-family M6 work used the probe, so the decomposition is Qwen3-1.7B only.

**The golden arm's 53.3% argument share has a wide interval** ([0.301, 0.752] on 15 failures) and is not the claim;
the claim is the arms' 0.000 with upper bounds under 5%.

**Nothing here is a fault result.** These are unfaulted retention arms. Whether a *fault* can produce an argument
error while leaving the function name intact is a different question and is unmeasured — and it is the interesting
one for C4, because that would be exactly a case where a token-level metric moves and the tool call still parses.

## D65 — Pass^k, the metric M6 asked for and nothing computed, and it restates H2d far more sharply (2026-09-26)

The plan's §M6 lists "**Pass^k**: bağımsız k fault gerçekleşmesinin hepsinde başarı (güvenilirlik odaklı tanım)" and
it existed nowhere. It needs no GPU: every protection arm already draws `n_trials` independent fault plans per
sample, which are exactly the k realisations the metric is about. `scripts/report_pass_k.py`,
`outputs/reports/pass_k.json`.

**Estimator.** For a sample with `s` successes of `n` trials, the chance that k trials drawn without replacement all
succeed is `C(s, k) / C(n, k)`, averaged over samples. The combinatorial form uses all n trials rather than asking
"were the first k clean" and discarding the rest — the code-generation pass@k device applied to a conjunction.

| arm | outcome | Pass^1 | Pass^2 | Pass^4 | **Pass^8** | geometric Pass^1⁸ | excess at k = 4 |
|---|---|---|---|---|---|---|---|
| `m3_arm_a_unprot` | task failure | 0.6822 | 0.4933 | 0.2956 | **0.1356** | 0.0469 | **+0.0790** |
| `m3_arm_a_unprot` | answer changed | 0.6377 | 0.4340 | 0.2317 | **0.0847** | 0.0274 | **+0.0663** |
| `m3_arm_g_meta_only` | task failure | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 1.0000 | n/a — no failures |
| `m3_arm_g_meta_only` | answer changed | 0.9831 | 0.9752 | 0.9661 | **0.9492** | 0.8722 | **+0.0322** |
| `m3_arm_h_range_guard` | task failure | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 1.0000 | n/a — no failures |

### The headline, and it is the best statement of H2d in the project

**Over eight independent faults, an unprotected 4-bit cache returns a correct answer 13.6% of the time; the metadata
guard returns one 100% of the time.** Same bit budget to within +0.075% of storage. Per-trial rates said the guard
removes "100% of measured task failure at BER 1e-6", which is true and sounds like a rounding artefact of small
counts; Pass^8 says the same thing as a survival probability and the gap is 0.1356 against 1.0000. That is the
number a reliability budget is actually about, and it is why the plan asked for this metric rather than another rate.

The guard is not perfect on the stricter proxy — `answer_changed` Pass^8 is 0.9492, so about 1 deployment in 20
sees *some* change across eight faults even guarded — and reporting both is the point. The answer surviving and the
answer being right are different guarantees.

### Failures cluster, in every arm that has any

The excess of the measured curve over `Pass^1` to the k is positive everywhere: **+0.0790** and **+0.0663** on the
unprotected arm, **+0.0322** on the guarded one's proxy. If each fault struck independently across samples the curve
would follow the geometric prediction; it sits above it, so **failures repeat on the same samples**.

That changes how the headline should be read. 13.6% is not "every deployment survives 13.6% of the time" — it is
closer to "13.6% of inputs are immune and most of the rest are not". **Risk is concentrated in which inputs a
deployment sees, not spread over how many faults it takes.** And `always_fails` is **0.0000** in every arm, so no
input is certainly doomed either; the concentration is in the middle of the distribution, not at its ends.

This is the same clustering that bit D26, which withdrew a 1.73× ratio after it turned out to be a pooled-trial
artefact. There it was a hazard to a comparison; here it is the measured quantity.

### A bug in the first version of this analysis, recorded because it was mine

The script's first clustering diagnostic compared `Pass^n_max` against the share of samples that never fail — and
those are **the same number by construction**, since `C(s, n) / C(n, n)` is 1 exactly when `s = n`. So it printed
"CLUSTERED" for every arm including `m3_arm_g_meta_only`, which has **zero failures** and for which clustering is
undefined. Caught by reading the output rather than by any test: an arm with no failures being told its failures live
on a subset of samples is the kind of sentence that has to be noticed. The diagnostic is now the excess over the
geometric curve, and the no-failure case says so explicitly.

### Scope

Qwen3-1.7B, BER 1e-6, the M3 probe arms, 8 trials per sample so Pass^k is measurable only to k = 8, and 59 of 60
samples (one excluded by its own golden). Pass^k over a *larger* k needs more trials per sample, not more samples,
and 8 was chosen for M3's purposes rather than for this metric.

## D66 — C4 closed: token-level metrics do under-estimate agent damage, and by less than the cross-vehicle version said (2026-09-26)

C4 has been `PROPOSED` since M0 and open after M6 — the last unresolved claim in the matrix. Its evidence was always
D16's **"+29% perplexity alongside −71% probe accuracy"**, and the weakness was never the numbers: that perplexity
was measured on **WikiText** and that accuracy on the **probe**, two corpora and two vehicles, so the gap could have
been a domain difference rather than a metric one. `docs/claims.md` named the fix — perplexity over the catalogue —
and this is it.

### What was built

`cache_ppl` now accepts the BFCL vehicle: `bfcl_corpus_ids` serialises all 769 distinct schemas as JSON in the pool's
deterministic name order, tokenizes them as **raw text with no chat template** (applying it would score control
tokens and the result would stop being a perplexity over tool schemas), and hands them to WikiText's own
`make_chunks`. The geometry is WikiText's unchanged — prefix 1536, continuation 512 — because that is what makes the
two perplexities comparable rather than merely both being perplexities. The corpus is 48,487 tokens, which is 23
chunks of 2049; measured, not chosen.

Two details that are provenance rather than plumbing. The new `BFCLConfig` fields default to `None` so
`exclude_none` leaves every existing BFCL retention identity untouched. And `data_fingerprint` keeps **exactly** the
one key it always had for WikiText: the runner refuses any experiment whose stored fingerprint differs
(`_check_existing`), so the `n_chunks` key that was briefly added there would have made every one of the 27 completed
WikiText perplexity experiments raise `ExperimentConflictError` on any re-run. Caught by checking the runner before
committing, and verified against a stored manifest.

### The result, on one vehicle and one axis

The perturbation is quantization precision, because it is the axis D16 used and the only one both metrics can read.
The M6 retention arms vary the *scorer* at a fixed 4 bits, which is a different axis and would have reintroduced the
confound C4 needs removed.

| precision | catalogue perplexity | Δ | tool calls correct | accuracy | Δ | Wilson 95% | Fisher vs bf16 |
|---|---|---|---|---|---|---|---|
| bf16 | 2.3630 | — | **105/120** | 0.8750 | — | [0.804, 0.923] | — |
| 4-bit | 2.3624 | **−0.03%** | 104/120 | 0.8667 | **−0.95%** | [0.794, 0.916] | p = 1 |
| 2-bit | 2.6799 | **+13.41%** | 80/120 | 0.6667 | **−23.81%** | [0.578, 0.745] | **p = 1.9e-04** |

**The control reproduced exactly:** the bf16 variant returns 105/120, the same as `m6_bfcl_golden`, which is what
makes the other two rows trustworthy rather than merely plotted.

**C4 is supported.** At 2 bits the token-level metric moves **+13.41%** and the task metric **−23.81%** — task damage
is **1.8×** the perplexity damage in relative terms, and the task effect is significant where the perplexity rise is
the sort of number a quantization paper reports as acceptable. A reader who budgeted by catalogue perplexity would
have accepted a 13% rise and lost a quarter of their tool calls.

**And the cross-vehicle version overstated it.** D16's pairing implies a ratio of about **2.4×** (+29% against −71%);
within one vehicle it is **1.8×**. So part of D16's gap was the domain difference, exactly as the objection to it
supposed — and the effect survives removing that part. This is the honest shape of the closure: the claim holds, and
the number that made it striking was inflated by the comparison it was drawn from.

**At 4 bits the two metrics agree that nothing happened** (−0.03% and −0.95%, p = 1). That is the control behaviour a
mismatch claim needs: the metrics diverge under real damage, not everywhere.

### The perplexity axis now has a second family, and D16's magnitude replicates

The audit found `cache_ppl` had 27 live runs and every one was Qwen3-1.7B — the whole token-level axis was
single-family while every other axis had a second. Fixed in the same campaign:

| corpus / family | bf16 | 4-bit | 2-bit | 4-bit Δ | 2-bit Δ |
|---|---|---|---|---|---|
| WikiText-2, **Llama-3.2-3B** | 9.4557 | 9.4749 | 12.0617 | +0.20% | **+27.56%** |
| BFCL catalogue, Qwen3-1.7B | 2.3630 | 2.3624 | 2.6799 | −0.03% | +13.41% |
| BFCL catalogue, Llama-3.2-3B | 2.1022 | 2.1042 | 2.3706 | +0.09% | +12.77% |

**D16's "+29%" replicates on the second family at +27.56%** — measured on the same corpus with the same geometry, so
this is a replication rather than an echo. **4-bit is free on perplexity in every corpus and both families** (−0.03%
to +0.20%), which replicates M1's granularity conclusion outside the model it was found on.

**And the agent corpus is roughly twice as robust to precision loss as general text**: +13% against +28% at 2 bits.
The catalogue's perplexity is 2.1–2.4 against WikiText's 9.5, so JSON schemas are far more predictable and there is
less for quantization to destroy. That is worth stating beside the C4 result, because it means the *perplexity* side
of the mismatch is domain-dependent in a way the task side need not be — and it is the mechanism by which a
cross-vehicle comparison like D16's inflates the gap.

### A reporting bug of mine, in the table of the claim this exists to settle

The first version formatted the deltas as `{value:+.2f}%` on a **fraction**, so the headline table printed
**"+0.13%"** for a 13.41% perplexity rise while the contrast lines below it printed the same number correctly. A
hundredfold error, in the one table a reader of C4 would look at. Caught by reading the output against the lines
underneath it; nothing else would have.

### Scope

One model on the task side (Qwen3-1.7B; the second family has the perplexity half but no BFCL task arms), the
`simple`+`multiple`+`parallel`+`parallel_multiple` schema pool for the corpus against `simple` questions for the
task, 120 tool calls, unpruned throughout. The C4 comparison is precision-only: whether *compression* shows the same
mismatch is unmeasured, and it cannot be read off the M6 retention arms because `cache_ppl` does not prune.

## D67 — H2b's burst form: SECDED is decisively defeated and the answer still never moves (2026-09-27)

H2b asks whether the fault-sensitivity map predicts *where* protection belongs. Its single-bit resolution (D31) was
**0 of 1416 task failures in both** arms D and E — a floor-to-floor comparison, which D30/D44 makes a bound rather
than an equality — and the claim row has said since then that *"the live version of the question belongs to the burst
regime"*. The arms' own headers pre-registered the crossover: **D18** (single-bit faults hurt early layers) predicts
E beats D under uniform SEUs, which it does on the proxy (0.64% against 1.77%); **D21** (whole-row bursts hurt late
layers) predicts the opposite sign under bursts. Both holding would mean no single static map suffices.

`ProtectionArmTask.burst_bits` delivers the arm's `n_flips` as `n_flips / burst_bits` bursts, each confined to one
(layer, part, token, head) slot — one corrupted memory word.

**The burst arms are deliberately not matched on bits to the single-bit arms, and that is an argument.** 64-bit
SECDED corrects one error per word. 11,457 single flips spread over ~1.79 × 10⁷ words leave almost every word with at
most one error, so they are nearly all corrected. A 16-bit burst inside one slot spans ~8 of those words and puts 2+
errors in several, which SECDED detects and cannot correct. The same bit count is a different exposure once grouped,
so matching on it would saturate the comparison rather than make it fair — hence a calibration, the way
`m3_rate_calibration` sized the rate.

### The calibration answered the question it was only meant to scope

`d67_burst_calibration`, arm D's protection map, 4096-token probe, 28 usable samples × 8 trials = 224 trials per rung.

| bits | bursts | trials | answers changed | Wilson upper bound | task failures | ECC **corrected** / trial | ECC **DUE** / trial |
|---|---|---|---|---|---|---|---|
| 16 | 1 | 224 | **0** | ≤ 0.0169 | **0** | 2.4 | **2.4** |
| 160 | 10 | 224 | **0** | ≤ 0.0169 | **0** | 24.7 | **24.4** |
| 1600 | 100 | 224 | **0** | ≤ 0.0169 | **0** | 124.9 | **127.0** |

**The machinery works and is doing exactly what it was built to do.** Corrected and uncorrectable word counts scale
linearly with the burst count (2.4 → 24.7 → 124.9 and 2.4 → 24.4 → 127.0), so at every rung about **half the ECC
words a burst touches are beyond SECDED's reach**. This is not a case of protection quietly absorbing the damage.

**And the answer never changes.** At the top rung the codec reports **127 uncorrectable 64-bit data words per
trial** — roughly **8,100 bits of unrecoverable data corruption** — and 0 of 224 trials produced a different answer,
let alone a wrong one. The bound is ≤ 1.69% at every rung.

### So H2b closes, and for a stronger reason than before

The single-bit answer was "placement does not matter for task outcomes", and a fair objection was that SECDED had
simply corrected everything, leaving nothing for placement to matter *to*. **That objection is now answered by
measurement: SECDED is decisively defeated and placement still has nothing to matter to.** Data corruption at this
operating point is task-irrelevant whether or not the codec recovers it.

**The pre-registered crossover is not observed**, and the reason is not that the sign came out the other way — there
is no signal in either direction to have a sign. D21's prediction is not refuted either; it is untested by this
design, for the reason below.

### Why two burst results in this project point different ways

`m2_fault_qa_burst` **did** produce task failures. It differs in two ways that matter more than the burst unit:
it aims at the **answer token's** KV, and it uses `layer_token` scope — the token's whole row in one layer. These
bursts are `head` scope at **uniformly drawn** tokens.

So the consistent reading across both: **a burst matters when it is aimed at the tokens the answer depends on, and
does not matter when it lands where a memory error would actually land.** The first is Q49's finding (role, D19–D21,
replicated on Llama in D57); the second is H2b's. They are not in tension — they are the same fact seen from the
targeting axis rather than the placement axis, and it is consistent with everything else: D19 (single code flips
harmless), D25/D30 (uniform data ECC buys nothing), D64 (agent failures are tool *selection*, never argument
transcription).

### What this does not establish

**Placement is untested at an exposure where random data corruption does hurt, because no such exposure was found
below 100 bursts.** Whether one exists is open; the linear DUE scaling suggests it would need orders of magnitude
more, at which point the arm is no longer modelling a memory error rate anyone deploys against.

**One model, one protection map, 28 samples, `head` scope only.** A `layer_token`-scope placement comparison is the
obvious next design and would test D21's prediction properly — it is the scope that produced damage — but it
targets whole rows, so it is closer to Q49's vehicle than to H2b's and would need its own pre-registration.

### A trap found by being caught in it

The calibration was launched as `--set task.n_samples=2`, believing it a smoke run. **It ran all 30 samples.**
`apply_overrides` runs on the raw config and `expand_sweep` then rewrites each swept key with `set_dotted`, so an
override under a swept key is applied and silently overwritten — and this config sweeps whole `task` dicts. Fifteen
minutes of GPU, and the output gives no sign: the run that happens is a perfectly valid one nobody asked for, which
is worse than an error. `load_experiment_configs` now refuses such an override by name, with the reason.

## D68 — A third model, and the first falsifiable test of the QK-Norm account (2026-09-27, pre-registered)

Frozen before either campaign finished. The plan's §M6 lists "Genelleme: Qwen3-4B", and it is the last unbuilt
generalisation. Unlike Llama-3.2-3B it varies **scale, not family**: Qwen3-4B shares Qwen3-1.7B's architecture
choices, QK-Norm included. That is the whole reason it is worth running.

### Why this is not just a third data point

Three findings differ between Qwen3-1.7B and Llama-3.2-3B, and one explanation has been offered for all three:
**Qwen applies QK-Norm to keys and Llama does not** (D1, D15), so key statistics diverge.

* **Q48's sign.** Qwen3-1.7B: ρ = **−0.377** [−0.588, −0.128], partial −0.282 (D18). Llama-3.2-3B: **+0.0726**
  [−0.2372, +0.3809], partial +0.0466 (D52). Both land in the frozen *separate signals* branch (partial ≤ 0.3) — the
  finding that killed §6's multiplicative Joint Reliability Score — but the sign differs.
* **Key-magnitude as a signal.** A 4× stronger anti-signal on Llama (D47).
* **The concentration effect.** Present on Qwen, absent on Llama (D54), with the mechanism later relocated (D59).

Every record says the same thing about the QK-Norm account: *consistent with it, and nothing ablates it.* D52 wrote
exactly that. **Qwen3-4B is the first configuration where it makes a prediction that can fail**, because it holds
QK-Norm fixed while changing everything else about the model's size.

### Pre-registered predictions

| outcome | what it means |
|---|---|
| **Q48's ρ is negative on Qwen3-4B**, like Qwen3-1.7B and unlike Llama | the QK-Norm account gains its first real support: the sign tracks the architecture feature, not the model |
| **Q48's ρ is positive or near zero** | **QK-Norm is not doing the work.** D52's account is wrong, and the Qwen/Llama sign difference needs a different explanation — scale, training data, or something unmeasured |
| the frozen *separate signals* branch (partial ≤ 0.3) holds | expected, since it has on both models; the branch is not what is at stake here |
| the frozen branch **fails** (partial ≥ 0.6) on a third model | the multiplicative score was rejected on evidence that does not generalise, and D18's consequence for the allocator would have to be reopened |

The sign is the live question. The branch has replicated twice and a third replication would be confirmation rather
than news; a *failure* of the branch would be the most consequential result in the project, which is why it is
written down even though it is unlikely.

**H2d comes free.** The same campaign yields the metadata-vs-code damage ratio, the way D55 recomputed it from the
Llama campaign at zero extra cost: **2.845×** on Qwen3-1.7B, **4.446×** on Llama-3.2-3B. It is the project's
best-replicated finding and the one every protection decision descends from, so a third model is worth having on it
whatever Q48 does.

### What the geometry change forces, and what it does not

Qwen3-4B has **36 layers** against 28, so `kv_values_per_token` is **73,728** against the 57,344 that Qwen3-1.7B and
Llama-3.2-3B happen to share. Verified in the smoke run: a 1536-token prefix at 4.25 bits/value records 60,162,048
analytic bytes, which is exactly 1536 × 73,728 × 4.25 / 8.

Consequences, honoured in the configs:

* The same BER delivers **~1.29× more flips per token**, so absolute flip counts are not comparable across models.
  Every quantity compared here is either **per value** or a **within-model ratio**, which is what makes it admissible.
* **Layer-indexed protection maps do not transfer.** `data_layers: [9..27]` was written for a 28-layer network; a
  Qwen3-4B protection arm would need its own map, and none is run here.
* Q48's correlation runs over each model's own layers. Nothing assumes layer index *k* is the same depth in a
  36-layer network as in a 28-layer one.

### One process note

`d68_quant_sensitivity_4b` was first produced by search-and-replacing the include path in
`m9l_quant_sensitivity_blocks.yaml`, which left a header reading "ON THE SECOND FAMILY" and "Identical KV geometry
(verified: both 57344 values per token)" — **both false for this model, and neither touched by the replacement.** A
config whose comment describes a different experiment than it runs is the D24/D42 failure mode, and the header was
rewritten from scratch. Copying a config copies its claims.

### Scope

WikiText-2, 40 chunks, the same strata and counts and seed as both earlier campaigns so the cells are comparable one
by one. This is a scale axis within one family; it says nothing new about Llama, and a third *family* remains unrun.

### RESULT, 2026-09-27 — the frozen branch holds on a third model; the QK-Norm prediction is directionally right and does not meet the project's own bar

Two campaigns on Qwen3-4B, 26 minutes of compute after an 11-minute weight download.
`d68_fault_campaign_4b`, `d68_quant_sensitivity_4b`, `outputs/reports/target_class_damage.json`.

#### Q48 across three models

| model | layers | ρ(quant sens, fault damage) | 95% CI | **partial ρ** on mean \|KV\| | frozen branch |
|---|---|---|---|---|---|
| Qwen3-1.7B | 28 | **−0.377** | [−0.588, −0.128] | **−0.282** | separate (≤ 0.3) ✓ |
| **Qwen3-4B** | 36 | **−0.229** | [−0.448, **+0.004**] | **−0.156** | separate ✓ |
| Llama-3.2-3B | 28 | +0.073 | [−0.237, +0.381] | +0.047 | separate ✓ |

**The frozen branch holds on all three.** Partial |ρ| is 0.282, 0.156 and 0.047, every one inside the *separate
signals* branch fixed before any measurement. So §6's multiplicative Joint Reliability Score is now rejected on three
models rather than one, and D18's consequence for the allocator — a staged rule instead of a product — does not rest
on a single configuration. That was the outcome flagged as unlikely and most consequential if it failed; it did not
fail.

**The QK-Norm prediction is directionally right and does not clear the project's own rule.** The sign is negative on
both Qwen models and positive on Llama, which is what the account predicts. But **Qwen3-4B's interval includes zero**
(upper bound +0.004), so by D31 — a result needs its interval to exclude zero — the negative sign is **not
established** on the third model. The pre-registration said a negative ρ would give the account "its first real
support"; the honest verdict is weaker than that sentence, and the sentence was mine.

**And the effect attenuates with scale, which QK-Norm alone does not predict.** ρ goes −0.377 → −0.229 and the
partial −0.282 → −0.156 between 1.7B and 4B, both Qwen, both applying QK-Norm. If the architecture feature were the
whole story the two Qwens should agree; instead the larger one sits roughly halfway to Llama. So scale is doing
something too, and the clean two-way story (Qwen negative, Llama positive, QK-Norm the cause) is not what three
models show.

#### H2d on a third model, and the ratio now comes from one definition

Extracted into `scripts/report_target_class_damage.py` after being computed by hand in D17 and D55 — which
**disagreed on the same data** (2.4× against 2.845×) purely through a different pooling of scale and zero. The
extracted definition reproduces D55 exactly, which is the check that matters.

| model | code mean | metadata mean | **ratio** | worst metadata | worst code | tail | LSB→MSB |
|---|---|---|---|---|---|---|---|
| Qwen3-1.7B | 0.2233 | 0.6353 | **2.845×** | 35.96 | 0.870 | 41× | 1.032 |
| **Qwen3-4B** | **0.2232** | **0.7886** | **3.534×** | 25.00 | **2.187** | **11×** | **1.057** |
| Llama-3.2-3B | 0.1273 | 0.5659 | **4.446×** | 19.05 | 0.366 | 52× | 1.023 |

**Metadata dominates on all three**, at 2.8×, 3.5× and 4.4×. This is the finding every protection decision descends
from and it is now three-for-three. **H1c likewise**: LSB→MSB moves damage 1.032×, 1.057× and 1.023× while the stored
value changes **128×**, so MSB-only ECC is poor value per bit on every model measured.

Two things the third model adds. **Code damage is scale-invariant in the mean** — 0.2232 against the 1.7B's 0.2233,
which is a coincidence to three decimals and worth stating as such rather than as a law. But **the 4B's worst code
flip is 2.187 nats against 0.870**, so its tail separation collapses to 11× from 41×: the larger model has code flips
that hurt much more than any the 1.7B produced, even though the average is identical. A tail that grows with scale
while the mean does not is the sort of thing a mean-based protection argument would miss.

**The fault map's layer structure replicates.** Layer × part cells are statistically separated on all three (spread
0.9219 / 0.8909 / 0.7656, disjoint intervals in each), while bit position is nearly flat (0.0408 / 0.0205 / 0.0244)
and on the 4B is **not** separated at all. M2's criterion-2 finding — layers matter, bit positions barely — is
general.

#### A new three-model finding: the zero-point is the most dangerous word in the cache

Counted explicitly because the first version of the extracted script reported `nan` for every metadata mean, and
tracking down why turned out to be the finding:

| model | code non-finite | scale non-finite | **zero non-finite** |
|---|---|---|---|
| Qwen3-1.7B | **0 / 4000** | 2 / 387 = 0.52% | **5 / 413 = 1.21%** |
| Qwen3-4B | **0 / 4000** | 2 / 423 = 0.47% | **4 / 377 = 1.06%** |
| Llama-3.2-3B | **0 / 4000** | 0 / 387 = 0.00% | **7 / 413 = 1.69%** |

**Not one of 12,000 code flips drove the NLL non-finite, and about 1% of metadata flips did** — matching D62's
independently measured ~1.3% on a different vehicle. And the zero-point is the worse half in **all three models**,
which generalises the observation D62 could only make on Llama (7 of 7 collapses from zero-point flips).

**This also makes every metadata/code ratio above a lower bound.** The non-finite sites are excluded from the means
by necessity, and they are not missing data — they are the worst outcomes there are. D55's numbers were computed the
same way without saying so; the script now prints the count beside the mean.

#### Two provenance defects fixed on the way

**`kv_magnitude.json` was one fixed path for model-specific data.** Running `kv_stats.py` for a second model
overwrote the first, after which `report_m2`'s partial correlation for the original campaign would merge the *wrong*
magnitudes onto layers 0–27 and report a number with nothing announcing it. The reverse direction fails safely
(missing layers merge as NaN and the partial is skipped), so exactly one direction was dangerous. Now written per
model and resolved by matching the file's own recorded `model` against the campaign's manifest. **D18 reproduces
exactly through the new path** — ρ = −0.3773 [−0.5883, −0.1279], partial −0.2817 against the recorded −0.282 — which
is the check that the change moved nothing.

**A copied config carried a false header.** `d68_quant_sensitivity_4b` was first produced by search-and-replacing an
include path, leaving a header that read "ON THE SECOND FAMILY" and "Identical KV geometry (verified: both 57344
values per token)" — both false for a 36-layer model with 73,728, and neither touched by the replacement. Rewritten
from scratch. Copying a config copies its claims.

#### Scope

WikiText-2, 40 chunks, same strata and seed as both earlier campaigns. This is a **scale** axis inside one family:
it says nothing new about Llama, and a third *family* remains unrun. No protection arm was run on Qwen3-4B, because
the layer-indexed maps (`data_layers: [9..27]`) were written for 28 layers and do not transfer.

## D69 — The multi-turn state probe, and a baseline that is a finding in its own right (2026-09-27)

The plan's §M6 asks for "deterministik cok turlu durum tasiyan gorevler" and it was the last unbuilt deliverable.
BFCL cannot serve it: `multi_turn_long_context` caches **157 tokens at the median** (measured in `data/bfcl.py`)
against this project's 512–4096, and its "long_context" comes from tool responses accumulated during execution rather
than a static prompt — which needs the stateful harness the plan flagged as its own decision point.

**What a multi-turn vehicle adds that no single-turn one can: a wrong answer with a known provenance.** Three user
turns — a document establishing ``<key> is <old_value>`` among same-type distractors, a correction saying it changed
to ``<new_value>``, then the question. Gold is the new value and the **superseded** one is registered as a scorer
distractor, so `mentions_distractor` measures **state regression** directly. Every other vehicle here can say damage
broke an answer; none can say what the broken answer came from.

Both assistant turns are fixed strings, because a vehicle whose intermediate turns were model-generated would not
rebuild the same cache and none of this project's provenance rules would apply to it. The document, filler and depth
placement are `RAQProbeBuilder`'s unchanged, so a state sample's document is the same object a probe sample's is.

### The baseline, on a bf16 cache with no faults and no compression

360 samples, `d69_state_golden`.

| | correct | **state regression** | other wrong |
|---|---|---|---|
| all types | **56.67%** [51.50, 61.69] | **43.06%** [38.04, 48.22] | **0.28%** |

**The outcome is almost perfectly binary: 1 sample of 360 answered anything other than the new value or the old
one.** That is what makes the vehicle work — a wrong answer here is a *regression*, not noise.

And the rate is wildly fact-type dependent:

| fact type | correct | regressed |
|---|---|---|
| tool_arg | **100.0%** | **0.0%** |
| constraint | 90.0% | 10.0% |
| number | 85.0% | 15.0% |
| date | 53.3% | 46.7% |
| **entity** | **5.0%** | **95.0%** |
| **id** | **6.7%** | **91.7%** |

**A 1.7B model carries a correction reliably for some value types and almost never for others, with a perfect
cache.** Nine times in ten it answers `entity` and `id` questions from the turn that was explicitly superseded. The
mechanism is not measured here and I will not guess at it beyond noting that the two failing types are the ones whose
values are invented multi-word or alphanumeric strings while the three reliable ones are numeric or short — which is
a correlation across six cells, not an explanation.

**Consequence for the damage measurement, and it is a design decision rather than an observation:** a cell already
regressing 95% of the time cannot show a *fault-induced* regression, so the fault arm runs on `tool_arg`,
`constraint` and `number` only (floors 0%, 10%, 15%). The exclusion comes from this unfaulted baseline and nothing
else, which is the same reasoning that gives every arm here its own golden, applied to the choice of cells.

### The bug that would have invalidated all of it, and how it surfaced

The first baseline run reported **44.4% regression and 38.3% correct** with a plausible fact-type breakdown, and it
was **a malformed prompt**. The probe's `query` span runs to the *end* of its prompt, so it contains the question and
the trailing chat-template scaffolding; embedding that as a message's content made the rendered conversation end with
**two assistant headers**, one closed and empty, and the model answered inside that.

It surfaced through `tool_arg` scoring **0% correct and 0% regressed** — a third failure mode no model explanation
fits. Reading the predictions showed the model emitting the *right* account number with the opening `<tool_call>` tag
missing, which traced to the duplicated header rather than to the scorer. **The symptom appeared on one fact type and
the cause was on all six.** After the fix `tool_arg` is 100% correct.

`_question` now cuts at the earliest special token, taken from the tokenizer's own `all_special_tokens` rather than a
hardcoded `<|im_end|>`, and raises if only scaffolding remains. Two tests pin it: exactly three assistant headers per
sample and no template marker inside the question turn. The malformed run is superseded with the reason in its
manifest.

**Worth stating plainly: I was one step from interpreting that breakdown as a model finding.** The golden arm is what
caught it, and only because one cell failed in a way that could not be a model behaviour. A vehicle whose bug had
degraded all six types uniformly would have passed.

### A second dispatcher

`build_probe_samples` was wired for the new dataset and `QAGenerate.__init__` has its own dispatch, which was not —
so the first run died on `NotImplementedError: state_probe`. It died **loudly**, which is why that branch is a raise
and not a fallback: a silent default would have run the probe and recorded it under the state probe's name.

### Scope

One model, one seed, 6 fact types × 2 lengths × 10 groups × 3 depths, cache 1091–2186 tokens. The correction always
targets the document's own target fact and always supersedes exactly one value; multi-update conversations, contested
updates and longer histories are all unbuilt. Nothing here is a fault result — that is `d69_state_fault_meta`.
