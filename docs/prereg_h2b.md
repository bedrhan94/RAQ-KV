# Pre-registration — the H2b placement test

Written before the data existed. Committed as the record of what was predicted, so the predictions cannot be
reshaped after the fact. Two parts: the direction (written while arms A/B were running, before D or E existed)
and the endpoint (written while the long-context run was in flight and unread).

---

# Pre-registration: the direction of the D vs E prediction
Written 2026-09-20, while the campaign was running arm A/B — before arm D or E existed.
To be committed into docs/decisions.md once the tree can be touched again (D24 process rule).

## The conflict in our own evidence

Arms D and E are bit-identical and carry the same amount of SECDED on different layers:
* **D** protects layers 9–27, leaving **0–8 (early)** exposed.
* **E** protects layers 0–18, leaving **19–27 (late)** exposed.

Two M2 findings predict opposite winners, and they were measured in different fault regimes:

| source | regime | metric | which layers are sensitive |
|---|---|---|---|
| D18 (Q48) | single-bit faults | token-level NLL/margin | **early** layers |
| D21 (Q49) | whole-row 2048-bit bursts | task-level answer correctness | **late** layers only (14/338 vs 0/316) |

The M3 campaign's main rate (1e-6) is the **uniform SEU** regime: ~0 words carry more than one flip, so every
fault in a protected layer is corrected and the only damage comes from the ~9/28 of code flips that land in an
arm's unprotected layers. That is D18's regime, not D21's.

## Pre-registered predictions

1. **At BER 1e-6 (uniform SEU): E < D** in task failure — protecting the early layers should win, because D18 says
   single-bit faults hurt most there. Note this is the *opposite* of the direction written in `docs/m3_plan.md`
   ("support needs D < E"), which was set from D21 without separating the regimes. The plan's direction is hereby
   corrected before the data exists; what is pre-registered is that **the two arms differ**, with the sign
   predicted per regime as in this table.
2. **At whole-row burst scale: D < E** — D21's regime, opposite sign.
3. **If both hold**, the finding is a crossover: *where* protection belongs depends on the fault's spatial scale,
   so no single static sensitivity map is sufficient, and the M5 allocator must take the fault model as an input
   rather than a constant. This would be a stronger result than either arm simply winning.
4. **If D ≈ E at every scale**, the layer map has no predictive value for protection placement and H2b is refuted.

## Power caveat, also pre-registered

If B, D and E all report **zero** task failures at 1e-6, the rate has no power for H2b regardless of trial count
(a 0-vs-0 comparison is uninformative, not a refutation). In that case the arms are rerun at the next rate up
(3e-6, then 1e-5), where the occupancy table puts the first multi-bit-per-word coincidences (~8192 flips), and the
1e-6 table is reported as a ceiling effect rather than as the answer.

## Dose and expected uncorrected load per arm (computed 2026-09-20, before any arm result existed)

At BER 1e-6 and T=2066, from the closed-form budget:

| arm | flips/trial | vs A | data-ECC layers | est. *uncorrected* code flips |
|---|---|---|---|---|
| A unprotected | 504 | — | 0/28 | 474 |
| B uniform | 575 | +14.1% | 28/28 | **0** |
| C precision | 576 | +14.3% | 0/28 | 474 |
| D late19 | 576 | +14.3% | 19/28 | 152 |
| E early19 | 576 | +14.3% | 19/28 | 152 |

Two things follow, both pre-registered:

1. **B/C/D/E receive an equal dose (575–576); arm A receives 14% fewer flips** because the rate is applied to each
   arm's declared storage and A declares 12.3% less. That is the correct definition (D23) and the reason arm A is a
   cost reference, never an equal-budget competitor. The report must print measured `flips_per_trial` beside every
   comparison and mark the A-vs-B row as unequal dose. Renormalising A's dose to match B is explicitly rejected:
   it would contradict the frozen definition.
2. **The D-vs-E test has power**: both arms carry ~152 uncorrected code flips, so neither is pinned at zero. Only
   arm B risks a floor effect (expected ~0), which makes B-vs-C a ceiling/floor comparison whose informative part
   is *C's* absolute level, not the significance of the gap.

**The decisive number is arm C.** C leaves the same ~474 code flips uncorrected as arm A, but unlike A it protects
metadata. So:
* **C ≈ A (~35% task failure)** → damage is carried by the code bits; uniform data ECC earns its 0.5 bits/value and
  H2a is refuted.
* **C ≪ A** → damage is carried by quantization metadata, which costs 0.05–0.24 bits/value to protect instead of
  0.5; uniform data ECC is largely wasted and H2a is supported.

The retired pilot's 0/9 code-only trials versus 4/71 metadata-containing trials at 1e-7 hints at the second, but
n=9 gives p=1 and it is not evidence. Arm C is the test.

## Sharpened prediction for arm C, from the completed calibration curve (still before any arm result)

The unprotected dose-response is **linear in flip count over two decades**: 13.4 / 14.8 / 13.3 / 16.4 flips per 1%
of answers broken at BER 3e-8 / 1e-7 / 3e-7 / 1e-6, rising to 24.7 at 3e-6 (saturation). Linearity means failures
are independent single-flip events, not accumulation: at this operating point one unlucky bit anywhere in 554 Mbit
breaks the answer with probability ~7.5e-4.

Arm C corrects only the ~6.1% of flips that land in quantization metadata. Under linearity that gives two sharply
different predictions, and they are the two competing accounts of M2's evidence:

| account | damage carried by metadata | predicted arm C task failure at 1e-6 |
|---|---|---|
| every stored bit is equally damaging | 6.1% | ~0.94 x arm A = **~32%** |
| metadata is ~3072x more damaging per bit (D17, D21) | 99.5% | ~0.005 x arm A = **~0.2%** |

Both accounts fit the *unprotected* curve equally well (both are linear in flip count), so the curve cannot
separate them. **Arm C separates them by a factor of 150.** This is the measurement, and either outcome is a
result:

* C ~ 32% -> D17/D21's per-bit metadata dominance does not survive at a realistic storage-wide rate; the 3072x
  figure came from tiny event counts and must be withdrawn or re-scoped. Uniform data ECC then earns its bits.
* C ~ 0.2% -> metadata protection at 0.05-0.24 bits/value buys almost all of what uniform data ECC buys at 0.5,
  and H2a is supported with a mechanism.

Anything in between is informative too: the ratio C/A directly estimates metadata's share of the damage, which is
the single number the M5 allocator needs in order to price protection.

---

# The D vs E test is answerable at 1e-6 after all — on a higher-resolution endpoint

Written 2026-09-20 with arms A–D in hand and **arm E still running (48/60)**, so before the comparison exists.
Companion to `prereg_DE_direction.md`.

Task failure is at the floor for both protected arms (B 0/472, D 0/464), which is why the pre-registered primary
endpoint cannot separate D from E. But `answer_changed` is not at the floor: arm D recorded **2.50%** while arm B,
which protects every layer, recorded **0.00%**. That 2.50% is produced entirely by D's nine *unprotected* layers, so
it is exactly the quantity the placement question is about, measured on a more sensitive endpoint than correctness.

D and E leave the same number of layers exposed (9 each), so exposure count is matched and only the depth band
differs: **D exposes layers 0–8 (early), E exposes 19–27 (late).**

**Pre-registered prediction, from D18** (single-bit faults hurt early layers most): early exposure should be worse,
so **E's `answer_changed` should be below D's 2.50%**. The reverse, or equality, contradicts D18 in this regime.

Stated limits, so this is not read as more than it is:

* `answer_changed` counts any change to the generated text, including changes that leave the answer correct. It is a
  damage *proxy*, not the task outcome, and will be reported as such.
* This endpoint was chosen after seeing that the primary one was at the floor, but before seeing arm E. That is a
  legitimate secondary analysis and is labelled secondary; it cannot promote H2b from "not answerable" to
  "supported" on its own. It can only indicate which way a properly powered burst-regime test should come out.
* n = 12/480 for arm D. A Fisher test on ~12 events has wide bounds, so a confidence interval will be reported
  rather than a bare p-value.

---

## Endpoint for the long-context run, fixed before the data exists (2026-09-20 03:05)

The 180-sample 4096-token placement run (`m3_arm_{d,e}_*_long`, BER 1e-5) started at 03:01:53 and no record of it
has been read. Declaring the endpoint now, so it is not chosen after seeing which one moved:

**Primary endpoint: `answer_changed`.** Task failure is at the floor for both arms *by construction of the
protection* — D and E each protect 19 of 28 layers, and every run so far (1e-6 and 1e-5, mixed and long contexts)
has recorded 0 task failures for both. An endpoint that cannot vary cannot answer the question. `answer_changed`
is a damage proxy, not the task outcome, and will be labelled that way in the result exactly as it has been
throughout.

**Secondary: task failure**, reported anyway. If it is non-zero for either arm at this rate and length, it becomes
the primary endpoint after all, because a real outcome always beats a proxy — but that is a rule fixed now, not a
choice made later.

**Direction, unchanged from the top of this file:** D18 predicts E below D (exposing early layers is worse). The
direction has now replicated at 1e-6 (2.59% vs 1.51%) and 1e-5 (3.02% vs 1.29%; 9.21% vs 2.63% on long contexts)
without reaching significance at either. This run is sized for ~130 events per arm against the 14 and 12 that
those runs produced.

**What counts as an answer.** Support: E below D, paired over samples, Wilcoxon p < 0.05 with a bootstrap CI
excluding zero. Refutation: a CI that contains zero and is narrow enough to exclude the ~2x ratio seen so far.
Anything else is a third underpowered null and will be reported as one rather than dressed up.
