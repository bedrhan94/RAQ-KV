# M3 exit checklist — error protection

**Status: M3 CLOSED on 2026-09-20.** Reports: `scripts/report_m3.py`; figure:
`outputs/figures/m3_protection.png`.

Headline: **the whole measured benefit of error protection on a 4-bit KV cache
is obtained by a plausibility check on the quantization metadata.** At a bit error rate of 1e-6 over stored
representation, an unprotected cache loses **31.8%** of its answers; a range check costing **+0.075%** of storage,
metadata SECDED costing **+2.3%**, and uniform SECDED over everything costing **+14.0%** all reach **0.0%**. The
protection axis of the joint allocation problem therefore turns out to be nearly free, which is not the answer the
project set out to find and is worth more than the answer it expected.

Criteria (M3, from `docs/m3_plan.md` as revised by D22/D23). M3 closes when every row is ✅ with evidence.

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | Codecs verified independently of the model | ✅ | `tests/test_ecc.py`: every 1-bit error corrected and every 2-bit error detected exhaustively on small words; triple-error behaviour recorded rather than assumed; encode/decode bit-identical on CPU and GPU (the parity kernel runs in float32 because CUDA has no integer matmul) |
| 2 | Protection expressed as config, not code | ✅ | `ProtectionConfig` (data codec/word/layers, metadata codec/word, metadata guard, DUE policy); six arms differ only by config; the slug encodes the policy (`dSEC64L9-27-mSEC`, `mRNG`) so two arms can never share a directory |
| 3 | A detected-uncorrectable error does something explicit | ✅ | `due_policy`, with `erasure_mid` branching on what the region stores: mid-code for codes, float16 zero for metadata (D27). SDC, DUE and masked counted separately in every record |
| 4 | ECC bits enter the bit budget, and equal budget is enforced before running | ✅ | `BitBudget.ecc_bits`; `budget/iso.py` pinned bit-for-bit against the real encoders (`tests/test_iso_budget.py`); `scripts/check_iso_budget.py` exits non-zero on a non-comparable set. Arms B/C/D/E within **0.18%** at every probe length, and re-checked from each run's own recorded budget |
| 5 | The fault model reaches everything an arm declares it stores | ✅ | `faults/storage.py` draws uniformly over codes + metadata + **check bits** + the range guard's own **stored bounds** (D23 and its addendum), shares verified against storage shares to four standard errors. Arm H was rerun with the bounds exposed: **4 bound flips occurred over the campaign** and the result did not move (0/480 task failures, unchanged). The guard also validates its own limits — ordered and finite, two free comparisons — and steps aside when they fail rather than clamping against a wrong limit |
| 6 | The operating rate is chosen by measurement, not assumption | ✅ | Calibration sweep on the unprotected arm, 5 rates: task failure 1.25% / 3.75% / 12.5% / 33.75% / 67.5% at 3e-8 / 1e-7 / 3e-7 / 1e-6 / 3e-6, controls exactly zero at every rate. Damage is linear in flip count over two decades (13–16 flips per 1% of answers), i.e. failures are independent single-flip events |
| 7 | Claims use a statistic that respects the data's clustering | ✅ | D26: trials within a sample are strongly dependent, so every claim is a per-sample rate paired over the samples **every** arm answers correctly without faults (58–59 of 60), Wilcoxon + bootstrap CI; pooled-trial Fisher is printed as descriptive only |
| 8 | The mechanism is measured, not inferred | ✅ | D25/D27: metadata carries **98%** of the damage at **6%** of the flips; per bit it is **315×** more damaging than codes (clean one-variable contrast A vs G). The reason is magnitude explosion, not NaN: a float16 scale's bit 14 multiplies it by 65536, bit 13 divides it by 250, bit 15 flips its sign, and the guard catches **19%** of metadata flips — the 3 of 16 bits the mechanism predicts |
| 9 | H2b (placement) answered or honestly declared unanswerable | ✅ | **Answered for task outcomes and declared unresolved on the proxy (D31).** The sized test (180 samples of the 4096-token condition at 1e-5, 177 paired) records **0 task failures of 1416 trials in both arms**: protecting 19 of 28 layers is sufficient regardless of which 19, so placement does not affect task outcomes. The proxy gives +0.0113 [+0.0007, +0.0247], p = 0.074, 7 discordant — the pre-registered rule needed p < 0.05 *and* a CI excluding zero, and is not met. The direction has replicated 4/4 (1.7×–3.5×) and in 6 of 7 discordant samples, reported as a consistent direction below significance. The live form of the question belongs to the burst regime (D21's opposite sign) and moves to M5 |
| 10 | Negative and null results reported as such | ✅ | Uniform data ECC buys nothing measurable at the task level for 0.5 bits/value (B ≡ C on all 58 samples); the plan's original decisive pair was arithmetically impossible (D22); the D-vs-E direction was withdrawn; the unverified "1e-3 tolerance" comparison with prior work was struck rather than used |

## What M3 changes for M5

The protection dimension nearly collapses: protect the metadata, by whatever mechanism turns out to be cheapest,
and stop. That frees the joint allocator to spend its budget on the two axes that actually trade against each
other at scale — retention and precision — under a metadata-protection floor, rather than treating protection as a
third competing axis as the brief's §6 assumed.

## Carried forward, not blocking closure

1. **ECC golden vectors for later RTL** (plan §39) — an artefact, not a measurement.
2. **A second model family before any claim is called general.** Llama-3.2-3B has identical KV geometry (D1), so
   a family effect separates from a geometry effect. Every number here is Qwen3-1.7B and says so.
3. **H2b in the burst regime.** D18 (single-bit, early layers) and D21 (whole-row, late layers) predict opposite
   placements; if both hold it is a crossover, and the allocator must take the fault model as an input. Needs the
   burst fault model, not more samples of uniform SEU.
4. **The absolute error rate is a condition, not a claim.** Everything here is stated at a rate over declared
   storage; mapping those rates onto a real memory technology is outside what this project measured.

## What M3 hands to M5

The protection axis is not a dimension to search. Read the deployment's error rate, take the range check below
~1e-6 and metadata SECDED above it, never buy uniform data ECC, and spend the entire remaining budget on
retention and precision — the two axes that actually trade against each other (D22, `docs/m4_m5_plan.md`).
