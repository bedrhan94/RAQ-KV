# M5 exit checklist — joint allocation

**Status: M5 NOT CLOSED as of 2026-09-23 — seven of eight.** Row 7 closed (D48): at 0.70 bits per original
token the joint policy beats every single-signal policy on both model families, all six contrasts surviving Holm
across a family of eighteen. D47's partial verdict was diagnosed as a floor artefact and the diagnosis implied
the fix — a tighter budget, where the baselines have room to fail.

**Row 8 is the only one left: precision × faults at fixed retention has never been crossed.** Retention × faults
was crossed in D43/D45 and refuted separability, so the allocator's precision stage rests on the same kind of
untested independence assumption its protection stage turned out to violate. That needs runs, not analysis.

Headline: **the joint point costs 1.054 bits per original token — 15.2× compression — and fails 0.85% of the
probe's questions.** A faithful KIVI at its own published configuration costs 4.331 and fails 5.13%. Every
single-signal policy that can reach 1.05 or 2.00 bits is beaten at both, all six contrasts surviving Holm; the
policy the quantization literature uses cannot reach either budget at all.

Criteria, from `docs/m4_m5_plan.md` §M5. M5 closes when every row is ✅ with evidence.

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | Search is **staged**, not a flat sweep | ✅ | D22: retention and bit width move a budget by tens of percent, group size and protection by a few or a fraction of one, so a flat search spends itself on infeasible corners. `allocation/policy.py` is coarse (highest precision whose full-retention cost fits) → fine (spend the remainder on smaller groups) → protection by lookup. Every rule cites the decision that set it |
| 2 | The one rule that could not be derived was **measured** | ✅ | D40: the iso-budget frontier has an interior optimum. Spending budget on tokens pays down to 4 bits and **reverses** below it — 2-bit at 40.5% loses to 4-bit at 25% by +0.1282 (p = 0.00027) with 100% of answer tokens kept in both, so the damage is per-token quantization error, not lost content. 3-bit ≈ 4-bit (p = 0.32), so `MIN_BITS = 4` is the conservative end of a plateau |
| 3 | Baselines at **their own** optimum, none weakened | ✅ | D39: KIVI built at its published configuration — keys `group_token` 32, values `group_channel` 32, **residual window 128** — with the window paid for in the budget. One finding runs against us and is reported: the window costs R/T per value, so below ~1500 tokens KIVI is dominated by plain 4-bit with no window, which is a result about the window and not about KIVI's own claim, made at long context |
| 4 | Equal total bits, **enforced by measurement** rather than asserted | ✅ | D41 found the joint arm spending +2.8% and +1.45% more than retention-only, outside the project's own ±1% rule and in the winning arm's favour. D42 fixed it by re-running: `scripts/calibrate_iso_budget.py` re-emits a short arm at the ratio whose *realised* budget matches its reference, and every H3 comparison is now within **+0.17%**. The correction is one-directional by design — raising a short baseline, never cutting a generous one — and that asymmetry is pinned by tests |
| 5 | Configs **emitted by the allocator**, not written by hand | ✅ | `scripts/emit_allocation_configs.py` generates each arm from the allocator's own `Allocation` object, carrying its rationale into the config header. A gap between what the policy decides and what the experiment runs therefore cannot hide in a YAML file — which is the thing H3 is actually testing |
| 6 | Infeasibility reported, never papered over (plan §44) | ✅ | `single_signal("precision_only", …)` returns `None` below ~2.63 bits per original token rather than overspending and calling the result a comparison. H3's first answer is an **absence**: the policy the quantization literature uses does not exist at either budget. Separately, D42: the allocator attached its metadata guard to an unquantized policy and the schema refused it — an unquantized cache has no metadata to guard, so a retention-only policy is unprotectable by the cheapest mechanism this project has |
| 7 | **H3 on at least two model families**, else refuted | ✅ **MET at 0.70 bits (D48)** | Eighteen contrasts, 2 families × 3 budgets × 3 baselines; 15 survive Holm across 18. **At 0.70 bits per original token all six contrasts meet the rule on both families** — retention-only at Holm 2.3e-05 (Qwen) and 4.3e-07 (Llama), magnitude-only and recency at 1e-14 or below. 22.9× compression, 0 of 101 failures on Llama and 1 of 99 on Qwen. D47's partial verdict was a floor artefact: the joint arm is essentially lossless at 1.05 and 2.00, so those retention-only contrasts are bounds with no room, not failures to win. The advantage grows monotonically as the budget tightens, which is what D37/D38's two-regime result predicts |
| 8 | Every axis **pair** crossed, not just priced independently | ❌ **NOT MET** | D43 crossed retention × faults and refuted separability (+0.0893 at matched exposure, p = 0.016). **Precision × faults at fixed retention has never been run**, and neither has the three-way interaction. So the allocator's *precision* stage rests on the same kind of untested independence assumption that its protection stage turned out to violate |

## What M5 learned that changed the allocator's shape

**§6's multiplicative Joint Reliability Score is not what got built, and four measurements are why.** The axes are
not commensurate (H2c: 4.844 against 2.344 bits/value for the brief's own "equal budget" pair). Protection is a
lookup on the deployment's error rate, not a search dimension (D25–D31). The two sensitivity signals the score
would multiply are **anti-correlated** (Q48: ρ = −0.377), so their product is worse than either. And the
retention decision is over **spans, all-or-nothing across layers** (D32, D34), so half the cells a per-(layer,
token) score would rank do not exist as choices.

**But D43 showed the brief was right about one thing the staged design denies: the axes interact — and D45
replicated it across a decade.** At equal fault exposure a cache pruned to 25% fails about **twice** as often at
low exposure (0.2232 against 0.1116 at 3e-7, p = 4.6e-04; 0.4018 against 0.3125 at 1e-6, p = 0.016), because
query-aware span retention keeps exactly the answer-bearing spans and every surviving scale governs a group the
answer needs. So protection is not separable from retention — just not through a product of per-cell scores.

**The allocator's constants were deliberately left alone, and D44 showed that was the right call for a reason
D43 had wrong.** D43 justified it with a 4-discordant-sample contrast and then overstated the conclusion,
calling the lookup "unconservative for aggressively pruned caches". At 1e-5 that contrast reverses: the pruned
guarded arm beats the unpruned one and escalating to SECDED buys nothing. Under the guard the retention effect
is not measurable at either rate, so a retention term is **open in both directions** -- which is what
`_protection()` now says, pinned by a test. Row 8's precision × faults pair is the same question one axis over.

## Carried forward, not blocking closure

1. **Latency and bandwidth are not claimed anywhere.** Quantization is fake — codes are reconstructed to bf16,
   so nothing shrinks in memory and no kernel exists. Every memory number is analytic, validated against the
   CUDA allocator at full precision (ratio 1.0000). A reviewer should treat any speed claim as absent rather
   than implied.
2. **The probe's construction.** RAQ-Probe's facts are single sentences *by construction*, which is exactly the
   structure the span finding exploits, and D43's concentration effect is measured through that same span
   scorer. Both need a natural long-context task (RULER, LongBench) before they are called general.
3. **The shared-prefix-cache gap.** Query-agnostic scoring sits at 0.624 against query-aware 0.026 (D36), and
   D42 closed the cheapest hope: magnitude, the one signal computable at prefill, selects answer-bearing tokens
   **at chance** (−0.034, p = 0.077; +0.029, p = 0.17).
4. **M6 agent benchmarks.** C4 — that token-level metrics understate agent failures — has evidence from
   retrieval only (+29% perplexity with −71% probe accuracy). The agent-level claim needs BFCL / tool-call.
