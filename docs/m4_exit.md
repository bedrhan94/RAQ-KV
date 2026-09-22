# M4 exit checklist — token retention

**Status: M4 CLOSED on 2026-09-23**, when row 10 was measured (D43). It was written a day earlier with row 10
open and nine of ten criteria met; that version is preserved in git rather than rewritten, because the reason
the milestone stayed open for a day is part of its record.

**Row 10's answer changed a structural assumption, not a constant.** At equal exposure a pruned cache is
significantly *more* fragile (+0.0893, p = 0.016): query-aware span retention keeps exactly the answer-bearing
spans, so every surviving scale governs a group the answer needs. Protection is therefore **not separable from
retention**, and the allocator's first stage assumes it is. The allocator's constants are unchanged — the
measurement that would move them has four discordant samples — so M5 carries a stated caveat instead of a fix.
See D43.

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
| 10 | **The retention frontier re-measured under faults** | ✅ **done, and it refuted the prediction (D43)** | Five arms at BER 1e-6, 56 paired samples. The plan predicted the reliability tax is "flat in retention". It is not: at **matched exposure** a pruned cache fails 0.4018 against the unpruned 0.3125, +0.0893 [+0.0246, +0.1607], p = 0.016, 42 discordant. At a matched *rate* pruning appears to win (−0.1250, p = 6.1e-04) but takes a quarter of the flips, which is arithmetic about exposure and was flagged in the config header before the run. Damage is sublinear in flip count (4.1× flips → 2.1× failure), which is what lets the concentration effect hide inside a matched-rate comparison |

## Why row 10 was the one that mattered

Row 10 was the only row that tested the word "joint". The three axes had each been measured, but **never two
at once**: M3 priced protection on unpruned caches, M4 and M5 priced retention and precision with no faults
injected. Two effects opposed each other, and which dominated was the whole question:

* Pruning removes tokens, and with them their quantization groups, so at a fixed bit error rate a pruned cache
  presents **fewer metadata bits to hit**. Exposure falls roughly with the retention ratio.
* But query-aware span retention keeps *exactly* the answer-bearing spans (`kept_target` = 1.000 at 25%). Every
  surviving scale governs a group the answer depends on, so each metadata flip that does land should be **more
  damaging** than in an unpruned cache where most groups are filler.

**Both are real, and they act at different places.** The exposure drop dominates the absolute outcome — at a
matched rate pruning cuts failure from 0.3125 to 0.1875 — while the concentration effect dominates the
*mechanism*: at matched exposure the pruned cache fails more. So compression and robustness do pull the same way
in practice, but not for the reason the staged design assumed, and the allocator's protection lookup is
**unconservative for aggressively pruned caches** by an amount D43 bounds without resolving.

**What was not done, deliberately.** Contrast 4 — the guard reaching 0.0000 unpruned and 0.0268 pruned — has
only 4 discordant samples and p = 0.066, so it does not meet the pre-registered rule and no retention-dependent
BER threshold was invented from it. The experiment that would set one is contrast 4 repeated at **1e-5**, where
D29/D30 located the guard's breakdown and the unpruned arm is no longer pinned at zero.

## Carried forward, not blocking closure

1. **The shared-prefix-cache gap.** Query-agnostic scoring sits at 0.624 against query-aware 0.026 (D36). A
   prefix cache is compressed before the query exists, so the best arm in this milestone cannot be used there.
   `magnitude_only` is the only H3 baseline that could be (D42).
2. **A second model family.** Every number here is Qwen3-1.7B. Llama-3.2-3B has identical KV geometry (D1), so
   a family effect separates from a geometry effect.
3. **The probe's construction.** RAQ-Probe's facts are single sentences *by construction*, which is exactly the
   structure the span finding exploits. The result must be treated as probe-dependent until it replicates on a
   natural long-context task (RULER, LongBench).
