# M4 exit checklist — token retention

**Status: M4 NOT CLOSED as of 2026-09-22.** Nine of ten criteria are met. The one that is not is the
experiment the milestone was designed around, and it is named in row 10 rather than quietly dropped.

Headline: **the unit of KV retention is the span, not the token.** At 25% retention a *perfect token oracle* —
one that keeps every answer-bearing token — still fails 56.4% of questions, while the fact's own sentence fails
15.4% and a deployable query-aware span scorer fails **2.6%**. A same-sized contiguous window elsewhere scores
58.1%, indistinguishable from keeping no context at all, so what the span buys is the fact's own
neighbourhood and not contiguity.

Criteria, from `docs/m4_m5_plan.md` §M4. M4 closes when every row is ✅ with evidence.

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | Scorers behind one interface, each declaring what it may see | ✅ | `importance/scorers.py`: `recency`, `key_norm`, `value_norm`, `window_attention`, `query_attention`, plus four oracles. Every scorer declares `needs_model` and `is_oracle`, so an information leak is visible in the interface rather than buried in a config (D36: the query-aware/query-agnostic distinction is worth 57 points, so which scorer saw the query is not a detail) |
| 2 | The informative oracle answered the redundancy question | ✅ | D33: `oracle_drop_target` keeps 99% of the cache while removing the answer-bearing tokens and fails **97.7% shallow / 97.2% middle**. The fact is load-bearing, so importance scoring is measurable on this probe at all — the precondition M4 had to establish before producing any table |
| 3 | Positions stay absolute under pruning | ✅ | `tests/test_protocol_equivalence.py`: middle-pruning + a 5-token chunk + 3 greedy steps equals the masked full cache to max \|Δ\| = 4.2e-7 under both `sdpa` and `eager`. Retained tokens keep their original RoPE positions; pruning never renumbers |
| 4 | Budget is per **original** token, never per retained token | ✅ | `BitBudget` keeps `retained_tokens` a separate field; `report_m4.py` reports `bf16_bits_per_original_token`; `allocation/policy.py` takes its budget in the same unit and says why in its docstring. Bits per *retained* token improves for free under pruning and would flatter every retention policy |
| 5 | The grid was sized from a pilot, not from a power calculation on a handful of events | ✅ | D31's rule, applied: one arm at the most aggressive retention first, rate read, then the grid. The probe sits at 0.983 clean accuracy, so a 25% arm could have landed at ceiling or floor and the pilot is what told us which |
| 6 | `kept_fraction_target` printed beside accuracy, always | ✅ | Enforced in `report_m4.py` and stated in its module docstring. It is the diagnostic that separates a scorer finding importance from a scorer finding redundancy (D32), and it is what shows recency's 66-point loss to be a query-awareness effect rather than an allocation effect (D41) |
| 7 | Every number broken out by target depth | ✅ | `DEPTH_BANDS` in `report_m4.py`. Recency keeps deep targets for free — 0.027 failure deep against 1.000 shallow — so an aggregate would hide the entire comparison. Adopted as a standing reporting rule in D33 |
| 8 | Claims use sample-level pairing | ✅ | D26's rule carried forward from M3: paired over the samples **every** arm answers correctly without pruning (117 of 120), Wilcoxon signed-rank, bootstrap CI over samples, Holm across pre-registered families |
| 9 | Negative, withdrawn and re-scoped results reported as such | ✅ | D32 (both gate arms failed 100%; the attention sink, found by a control arm). D36 retraction 1: `oracle_keep_sentence` is **not** a ceiling — it spends a fifth of its budget by recency and a real method beats it by 13 points; every use of the word "ceiling" for it is withdrawn. D36 retraction 2: "token-level scoring is actively harmful" holds only for query-**agnostic** scoring. D37: at ~4 bits nothing is significant, reported as a floor rather than a replication |
| 10 | **The retention frontier re-measured under faults** | ❌ **NOT DONE** | All **30** retention runs to date are `nofault`. The plan's sharpest prediction — "the reliability tax is set by precision alone and is flat in retention, so the *precision* half of the published optimum should move once reliability is priced and the *retention* half should not" — has never been tested. See below |

## Why row 10 is the one that matters

It is the only row that tests the word "joint". The three axes have each been measured, but **never two at
once**: M3 priced protection on unpruned caches, M4 and M5 priced retention and precision with no faults
injected. The allocator's staged design *assumes* row 10's answer — `_protection()` is a lookup on the error
rate alone, with no retention term — and that assumption is currently untested.

It is also not obviously true. Two effects push opposite ways:

* Pruning removes tokens, and with them their quantization groups, so at a fixed bit error rate a pruned cache
  presents **fewer metadata bits to hit**. Exposure falls roughly with the retention ratio.
* But query-aware span retention keeps *exactly* the answer-bearing spans (`kept_target` = 1.000 at 25%). Every
  surviving scale now governs a group the answer depends on, so each metadata flip that does land should be
  **more damaging** than it was in an unpruned cache where most groups were filler.

If the first dominates, the allocator is right and retention is a free reliability win — compression and
robustness pulling the same way, which would be a result worth stating on its own. If the second dominates,
protection is **not** separable from retention, the staged design is wrong at its first stage, and the
multiplicative form the brief proposed looks better than the measurements have so far made it look.

**The experiment:** the M5 joint point (25% of tokens at 4-bit, query-aware spans) and uniform 4-bit at full
retention, both at the same BER, unprotected and with the metadata range guard — four arms, one rate, paired
over samples. It is cheap and it is the one measurement that can falsify the allocator's structure.

**It cannot be run yet, and the reason is a bug the design surfaced (D42).** `ProtectionArmTask` is the only
task that injects faults, and its `run_unit` goes prefill → quantize → protect → decode with **no compression
step**. Worse, a `protection_arm` config carrying a `compression` block is *accepted*: the block validates,
survives into the config object, and reaches `_pruning_token`, so the run would be named `…_p25qaS_…` and
hashed as pruned while executing unpruned. That is a manifest that is internally consistent and wrong — the
failure mode D24's provenance check exists to prevent, arrived at from the other direction. The schema must
refuse the combination until the task implements it.

**Two things the experiment must report or it proves nothing.** `flips_for_rate` and `storage_regions` both
derive exposure from the quantized cache, so a pruned arm automatically receives **fewer flips at the same
BER**. That is simultaneously the confound and the finding, and it is only separable if flips-per-trial is
reported beside the failure rate — otherwise "pruning improves reliability" is unfalsifiable from the output.
The arms must also be compared at matched *flip count* as well as matched rate, because those are different
questions: "is a pruned cache safer to store" and "is a surviving scale more precious". **Matched flip count is
a required arm, not an optional one** — matched BER alone cannot separate "pruning is a reliability win" from
"pruning concentrates risk and the exposure drop happens to cover it", and only the first of those is a result.

**The path exists.** `generate_storage_plan(qkv, guard, seed, n_flips)` already takes an absolute flip count;
`flips_for_rate` is the convenience wrapper that derives one from a BER. So a matched-count arm needs a config
knob on `ProtectionArmTask` (`n_flips` as an alternative to `ber`, exactly one of the two required) rather than
any change to the fault machinery, and no arm ever has to fake a rate to get a count.

**One thing to check when relaxing the D42 schema guard.** `_pruning_token` reads `cfg.compression`
unconditionally, so once `protection_arm` may carry the block its slug changes shape. Every existing
`protection_arm` config has `compression is None` and must keep rendering `p0` exactly as before — otherwise
the relaxation silently orphans the M3 run directories.

## Carried forward, not blocking closure

1. **The shared-prefix-cache gap.** Query-agnostic scoring sits at 0.624 against query-aware 0.026 (D36). A
   prefix cache is compressed before the query exists, so the best arm in this milestone cannot be used there.
   `magnitude_only` is the only H3 baseline that could be (D42).
2. **A second model family.** Every number here is Qwen3-1.7B. Llama-3.2-3B has identical KV geometry (D1), so
   a family effect separates from a geometry effect.
3. **The probe's construction.** RAQ-Probe's facts are single sentences *by construction*, which is exactly the
   structure the span finding exploits. The result must be treated as probe-dependent until it replicates on a
   natural long-context task (RULER, LongBench).
