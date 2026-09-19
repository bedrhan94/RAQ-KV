# Claim ↔ evidence matrix

Status values: `planned` → `running` → `supported` / `refuted` / `inconclusive`.
Thresholds marked **PROPOSED** must be frozen (status `FROZEN`, with date) before M2 starts; they are never
changed after looking at the data. Full rationale: `00_research_plan.md` §I.

| ID | Claim | Experiments | Figures / tables | Decision rule | Threshold status | Status |
|---|---|---|---|---|---|---|
| S0 | Measurement protocol is valid (reference for all Δ) | M0a/M0b golden, protocol tests | Setup table | Exit criteria M0 (plan §G) | n/a | **supported** — all 8 criteria met 2026-09-19 (docs/m0_exit.md) |
| H1 | Sensitivity is heterogeneous across layer / KV-head / token type at fixed bit position | M1, M2 | Layer and bit heatmaps | Top 10% cells carry ≥ 50% damage (support); ≤ 20% (refute) | PROPOSED | **quantization side (D13): 40.2% — intermediate**, ranking stable, keys 11× values. **Fault pilot (D17): the dominant axis is not layer or bit position but the target class — metadata 2.4× codes in mean damage and the entire tail; MSB vs LSB nearly flat at 4 bits** |
| H1b | Quantization metadata is the highest-value protection target per bit | M2 pilot, M3 | Damage by target class | Metadata damage ≫ code damage while metadata is a small share of the bit budget | **added after D17** | pilot evidence: metadata is ~6% of stored bits and carries the damage tail (worst case 23.6 nats vs 0.81 for codes) |
| H2 | Adaptive allocation beats uniform INT4 + SECDED and magnitude-only at equal total bits (±1%, metadata included) | M3 gate-2, M5 | Pareto fronts, paired table | Paired bootstrap 95% CI excludes 0; Holm-corrected McNemar p < 0.05; ≥ 2 model families | PROPOSED | planned |
| H3 | Joint score beats the best single-signal policy | M5 ablations | Ablation table | Significant at equal budget; every ablation degrades | PROPOSED | planned |
| Q48 | Corr(QuantSens, FaultSens), raw and partial on KV magnitude | M1 + M2, gate-1 | Scatter + CI | Partial ρ ≥ 0.6 shared signal; ≤ 0.3 separate signals | FROZEN before measurement | **answered (D18): ρ = −0.377 [−0.588, −0.128], partial −0.282, replicated at 8 bits.** Falls in the "separate signals" branch and is mildly negative: faults hurt most in early layers, quantization in late-layer keys |
| H1c | Bit position is a weak protection signal for KV codes | M2 campaigns | Damage vs bit position | Damage scales far sublinearly with the stored-value change | **added after D17** | LSB→MSB: value change 128×, damage 1.30×, changed tokens 1.52% → MSB-only ECC is poor value; metadata protection dominates |
| Q49 | Critical token types have higher task-level SDC than filler at matched position / layer / bit | M2 (RAQ-Probe), M6 | Odds-ratio plot | OR > 1 with CI excluding 1 | FROZEN before measurement | **scale-dependent (D19-D21): no effect for single bits or a 16-bit word (OR 1.00); supported when the token's whole row in a layer is destroyed (OR 7.13 vs filler, 4.75 vs content-matched distractor), and then only in late layers** |
| H1d | Per protected bit, quantization metadata is the highest-value target | M2 | Damage per bit | Metadata damage per bit ≫ data damage per bit | **added after D21** | one scale bit fails 3.21% of answers; 2048 bits inside the answer token's row fail 2.14% — ~10³ per-bit ratio |
| C4 | Token-level metrics under-estimate agent failures | M6 | Metric-disagreement plot | Weak rank correlation between KL damage and task failure | PROPOSED | **first evidence (D16): +29% perplexity ↔ −71% probe accuracy; +7.5% ↔ −13%.** Measured under quantization, not faults, and on retrieval rather than agent tasks — the agent-level claim still needs M6 |

## Vehicle notes

- **RAQ-Probe** is the fault-sensitivity vehicle for H1/Q49 (known token roles, facts inside the cache). Analyses of
  margin changes use samples that are correct at golden; golden accuracy is reported separately.
- **GSM8K is a Level-1 reasoning reference only, not a fault-sensitivity vehicle in M0–M3.** Its cached problem text
  averages ~62 tokens, while most of its KV comes from generated reasoning, which is not transformed until decode-time
  policies exist (M5). Weak GSM8K sensitivity in M1–M3 is expected by construction and must not be read as evidence
  against H1.
- **WikiText cache-conditioned perplexity** measures general LM degradation from the cached prefix.
- **Noise floor:** in bf16, cache-geometry changes alone move logits by up to ~0.3 (KL ~1e-4) without changing the
  argmax (D10). Effect sizes below that floor are not evidence of anything.
