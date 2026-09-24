# M2 exit checklist — fault injection (no ECC)

**Status: M2 CLOSED on 2026-09-24**, four days after the work and one criterion later than assumed.

Four criteria were met on 2026-09-19 and M3 was built on them the next day. **Criterion 2 was not**: the plan asks
for the fault map "reported with Wilson CI", and the report carried mean nats with no intervals and no figure.
Writing this checklist is what found that — which is the argument for writing one. M0, M1, M3, M4 and M5 all had
exit documents; this one did not, and an unchecked criterion sat unchecked for four days while five later
milestones were built on top of it.

The gap is now closed rather than waived: `raq_kv.analysis.stats.wilson` (shared, so the estimator cannot drift
the way this project's budget predictor and span selector both once did) and a `changed_rate_map` section in
`scripts/report_m2.py`. **Adding the intervals immediately showed something the means had hidden** — see
criterion 2 below.

Headline: **the vulnerable part of a quantized KV cache is not the data, it is the quantization metadata**, and
that redirected the rest of the project. M2 was designed to produce a fault map over layer × bit position × K/V,
on the assumption that some layers and some bit positions would matter more. **Bit position barely does** — 128×
the stored-value change buys 1.30× the damage. **Layer does**, but only at the very top of the network, and that
part stayed hidden until criterion 2 was actually met four days later. And the axis that dominates both is the one
the plan never listed: the **target class** — a scale or zero-point, against a code.

Criteria, from `docs/00_research_plan.md` §M2.

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | The injector flips only the intended bit, proven by unit test | ✅ | `tests/test_faults.py`: every site addressable and **exactly reversible** — the cache is restored bit-for-bit after injection, so a plan cannot leak state into the next trial. Bit order fixed as LSB = bit 0 and pinned, because the whole bit-position analysis is meaningless if that convention drifts |
| 2 | Fault map reported with Wilson confidence intervals | ✅ **as of 2026-09-24; it was NOT met before** | The original report gave mean `delta_nll` per (layer, part) — a damage magnitude with no denominator, so no interval applied and none was shown. `changed_rate_map` now reports the *rate* the criterion is about: k code flips that changed a decoded token out of n attempted, per cell, with a Wilson interval. **What the intervals revealed:** by **bit position** the effect is statistically resolvable and practically negligible — 0.9181 [0.9025, 0.9314] at bit 0 against 0.9589 [0.9471, 0.9681] at bit 3, four points of rate against a **128×** change in the stored value. By **layer** it is large and was invisible in the means: layer 27 values sit at 0.0781 [0.0338, 0.1702] against layer 0 keys at 1.0000 [0.9547, 1.0000], **disjoint intervals**, and pooled by depth 0.974 (layers 0–9) / 0.980 (10–19) / **0.828** (20–27) |
| 3 | The same fault plan replayed twice gives the same result | ✅ | Plans are seeded from `derive_seed(seed, unit_key, stratum)` and serialised to `outputs/fault_plans/`; replay is byte-identical. **Later qualified by D23:** a plan is replayable only *within one layout*. Protected and unprotected arms at the same quantization are a paired comparison on identical faults; arms with different group sizes or protection maps have different bit layouts and are matched on *rate* instead. The unqualified replay rule M2 shipped with was wrong and M3 fixed it |
| 4 | **GATE-1:** Corr(QuantSens, FaultSens), raw and magnitude-controlled | ✅ | **D18:** ρ = **−0.377** [−0.588, −0.128], partial ρ on KV magnitude −0.282, replicated at 8 bits. The threshold was frozen before measurement (≥0.6 = one shared signal, ≤0.3 = separate signals) and the answer landed in the *separate signals* branch — and mildly **negative**: faults hurt most in early layers, quantization most in late-layer keys |
| 5 | H1's refutation condition checked: is there heterogeneity at fixed bit position? | ✅ | **Not refuted, and not cleanly supported either.** Quantization side (D13): the top 10% of cells carry **40.2%** of the damage — above the 20% refutation line, below the 50% support line, so H1 is recorded as *intermediate*. Fault side (D17): heterogeneity exists, but on the **layer** axis and the **target class**, not on bit position — and the layer part of that only became visible when criterion 2 was actually met |

## What M2 found that the plan did not anticipate

**The dominant axis is the target class, not the layer or the bit position (D17).** Metadata carries **2.4×** the
mean damage of codes and the entire damage tail — worst case **23.6 nats against 0.81** — while being about **6%**
of the stored bits. Every protection decision in M3, M4 and M5 follows from this one measurement.

**Bit position is a weak signal (D17, H1c).** LSB→MSB changes the stored value **128×** and the damage only
**1.30×**, with 1.52% of tokens changed. So MSB-only ECC — an obvious, cheap-sounding policy — is poor value per
bit, and the project never spent a bit on it. The Wilson map added in 2026-09-24 sharpens rather than softens
this: the bit-position difference is real (disjoint intervals) and four points wide, which is exactly "resolvable
and not worth protecting".

**A layer effect that the original report's means hid.** Code flips in the last two layers change a decoded token
far less often than anywhere else — layer 27 values at 0.078, against 0.97–1.00 through layers 0–19, with
non-overlapping intervals. Mechanically unsurprising in hindsight: a flip in layer 0's cache propagates through
27 more layers, one in layer 27 reaches only the final attention read. It is the same direction as D18's finding
that fault damage concentrates **early**, reached from a different endpoint, and it is the one piece of layer
structure M2's own criteria were designed to surface and did not.

**Single-bit faults on codes are harmless at the task level (D19).** This is the result that made M3's design
possible: if a single code flip almost never costs an answer, then uniform data ECC is paying for something that
does not happen, and the budget belongs on metadata. D28 later turned the accompanying *ratio* into a bound
(at least 1543×) because the per-code-flip rate rested on three events in one non-shared sample.

**Q49 is scale-dependent, and the first two answers were wrong (D19 → D20 → D21).** Frozen before measurement,
the question was whether a token's *role* (answer-bearing against filler) predicts task-level SDC. Single bits:
**no effect, OR 1.00**. A whole 16-bit word: **still no effect**. Destroying the token's entire row in one layer:
**OR 7.13** against filler and **4.75** against a content-matched distractor, and only in late layers. The
content-matched distractor is what makes that a claim about role rather than about position, and the progression
across three decision records is the honest shape of the answer: the effect exists, but only above a corruption
scale that single faults do not reach.

## What M2 got wrong, and where it was fixed

1. **The replay rule was stated without its qualification** (criterion 3). M3's D23 restricted it to a single bit
   layout after it became clear that arms with different group sizes cannot share a fault plan.
2. **The ratios it produced did not survive.** "Metadata is 762× more damaging per bit" and its successor 315×
   were both retired by D28 in favour of a bound. M2's *ordering* held; its *magnitudes* did not.
3. **H1d carried a retired ratio until 2026-09-24.** The "3072× per bit" figure derived here sat in a live claim
   row for four days after D28 retired its siblings. Caught by auditing `docs/claims.md` against the decision log,
   not by any control — the reason that audit is now a standing rule.
4. **Criterion 2 was assumed met and was not.** The plan asked for Wilson intervals on the fault map; the report
   shipped means. Nothing downstream depended on it, which is exactly why nobody noticed for four days — and when
   it was finally met, it surfaced a layer effect the means had hidden. The lesson is not about intervals: it is
   that a milestone with no exit document has criteria nobody is checking.

## What M2 changed for everything after it

GATE-1's negative correlation is why **§6's multiplicative Joint Reliability Score was never built**. A score
multiplying quantization sensitivity by fault vulnerability multiplies two signals that point at opposite ends of
the network, and the product is a worse ranking than either factor alone. The allocator that exists instead is a
staged rule (D22, D30, D40), and this is the measurement that started it.

## Carried forward from M2, still open

1. **Burst and correlated faults.** M2 built the interface and measured only single and independent-random flips.
   D18 (single-bit, early layers) and D21 (whole-row, late layers) predict **opposite** signs for where protection
   belongs, and the burst regime is where that disagreement becomes testable. It is where H2b's live form lives
   (D31) and it has still not been run.
2. **A second model family for M2's own findings.** D47 replicated H3 on Llama-3.2-3B, but the fault map is
   Qwen3-1.7B only. Qwen applies QK-Norm and Llama does not (D1, D15), so key statistics differ — and key
   quantization is what the whole metadata story is built around.
