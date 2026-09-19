# M1 exit checklist — KV quantization

**Status: M1 CLOSED on 2026-09-19.** Reports: `outputs/reports/m1_quantization_report.md`,
`outputs/reports/m1_sensitivity_report.md`; figure: `outputs/figures/m1_quality_vs_bits.png`.

Headline: the *granularity* of key quantization matters more than the bit width (per-token keys: PPL 3227 at
4 bits; per-channel keys: 14.53). With keys per-channel and values per-token, 8-bit is lossless, 4-bit costs
+0.14 PPL at 3.9× less KV memory, and 2-bit stays usable (PPL 23.0). Per-region sensitivity at 2-bit deployment
precision is heterogeneous and stably ranked, but only moderately concentrated (top 10% of regions carry 40%).

Criteria from `00_research_plan.md` §G (M1). M1 closes when every row is ✅ with evidence.

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | Passthrough reproduces the M0 golden records bit-exactly | ✅ | `tests/slow/test_golden_regression.py` (per-token NLL, predictions, margins); sweep passthrough corpus PPL 14.391535 = M0 golden 14.391535 |
| 2 | Quantizer verified independently of the model | ✅ | `tests/test_quantization.py`: grid-representable values round-trip exactly, error decreases monotonically with bits, finer granularity never worse, bit accounting equals the stored objects, region patching touches only its region |
| 3 | Quality vs precision/granularity measured with paired statistics | ✅ | `outputs/reports/m1_quantization_report.md`: 13 variants × 40 chunks, paired per-token ΔNLL with bootstrap CIs (D12) |
| 4 | Effective bits include quantization metadata | ✅ | fp16 scales/zeros counted in `BitBudget.quant_metadata_bits`; e.g. K4/chan+V4/token = 4.14 bits/value, not 4.00 |
| 5 | Per-region sensitivity profile with a measured noise floor | ✅ | `m1_quant_sensitivity` (40 chunks × 56 regions + 320 nulls): noise band [−0.0048, +0.0038] nats; **24/56 regions differ from the null (Welch, Holm)**, 18 clear the conservative band; top 10% carry 40.2%; split-half stability 0.901; keys 11× values (D13). Head-level map (448 regions): only 37 differ from the null (D14) |
| 6 | Positive control: the profiler detects heterogeneity when it exists | ✅ | `m1_quant_sensitivity_mismatched` (per-token key stress): top 10% of regions carry 76.6% of damage, split-half Spearman 0.93 (Spearman-Brown 0.96), ANOVA F = 572, keys 0.458 vs values 0.0007 nats |
| 7 | Defaults for later milestones stated and justified | ✅ | D12: keys per-channel, values per-token; operating points 8-bit (lossless), 4-bit (+0.14 PPL), 2-bit (PPL 23.0) |

## Interpretation rules fixed before reading criterion 5

- A region's mean ΔNLL counts as sensitivity only if its lower 95% bound clears the null band's upper edge.
- Negative region means are physically meaningless and are reported as a lower bound on noise.
- Ranking claims require split-half stability, not a single-chunk comparison.
- The mismatched-granularity profile localizes **outliers**, not deployment sensitivity; it must not be quoted as
  "layer X matters" for allocation.

## Handover to M2

- **Shared granularity for Q48 is layer×part** (D14): at head level only 8% of the 448 regions are distinguishable
  from the null, versus 43% of the 56 layer×part regions, and rank stability drops from 0.90 to 0.72. The fault
  side may still be analysed at head and bit granularity on its own; the correlation uses layer×part.
- **Sensitivity is declared by a Welch test against the null distribution with Holm correction** (D14), with the
  conservative band criterion reported alongside.
- **The fault campaign needs its own null condition**, and it is not either of the existing ones: D10 measures
  cache-geometry change in bf16 and D13's band measures lossless re-quantization. Neither says what a bit flip
  that lands somewhere inert does. A fraction of injections must therefore target bits expected to be harmless
  (e.g. the LSB of a small-magnitude code) and be reported next to the real injections.
- Fault injection targets the same regions the profiler scores, so the sensitivity map and the fault map are
  directly comparable (Q48) — same granularity, same chunks, same golden.
- Effects below the bf16 noise floor (D10) or the profiler's null band cannot be attributed to a fault.
- The 2-bit deployment setting (K per-channel, V per-token) is the precision at which faults will be injected
  first, because it is the operating point where allocation decisions matter.
