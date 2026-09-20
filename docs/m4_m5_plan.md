# M4 + M5 plan — token retention, and the joint allocation the project exists for

The brief's third axis has no code: `importance/`, `compression/` and `allocation/` do not exist. This is the
design, written after M3 and after the literature review, both of which change what the joint problem *is*.

## What M3 and the literature already settled

**The protection axis is nearly free and rate-dependent, not a competing dimension.** At BER 1e-6 a plausibility
check on the quantization metadata costing 0.075% of storage removes every measured fault-induced task failure;
at 1e-5 metadata SECDED at 2.3% is needed; uniform data ECC at 14% is not worth its bits in either regime
(D25–D30). So M5 does not search a protection dimension — it reads the deployment's error rate and picks one of
two cheap settings. That frees the whole budget for the two axes that genuinely trade against each other.

**The token-versus-precision trade is already answered, and we must not re-derive it.** "More Tokens, Lower
Precision" (arXiv 2412.12706) compares 1× tokens @16-bit, 2× @8-bit and 4× @4-bit at equal memory on LongBench,
Needle-in-a-Haystack and RULER, and finds more tokens at lower precision wins, most strongly on retrieval tasks.
An earlier draft of this plan called that pairing untested; it was wrong and the claim is withdrawn (see
`docs/literature.md`). **Their optimum is M4's starting point, not M4's question.**

**The question they leave open is ours, stated precisely.** Their recommendation drives systems toward lower
precision, and lower precision is what raises the metadata share — the regime where, by our measurements,
essentially all fault damage lives. Nobody has priced the reliability axis of that frontier. That is the
contribution.

> **Correction, computed before any M4 run (2026-09-20).** An earlier draft said "many tokens at low precision"
> pushes systems into the metadata-dominant regime. **Retention has no effect on the metadata share at all** —
> it is 5.9% at 100%, 75%, 50% and 25% retention, exactly invariant, because codes and groups both scale
> linearly with token count. Only the precision knobs move it:
>
> | knob | metadata share of stored bits (T = 4096) |
> |---|---|
> | 8-bit, g128 | 3.03% |
> | 4-bit, g128 | 5.88% |
> | 2-bit, g128 | 11.11% |
> | 4-bit, g64 | 8.57% |
> | 4-bit, g32 | 13.51% |
> | 4-bit, g16 | 21.95% |
>
> This *sharpens* the framing rather than weakening it: the reliability tax is a pure function of (bits, group
> size) and is **independent of how many tokens are kept**, which means M5 can set the protection floor from the
> precision choice alone, before retention is decided. The half of the earlier sentence that named token count
> was wrong and is deleted.

## The structural fact M5 is built on

Measured exchange rates against a 4-bit, group-128, unprotected baseline (4.26 bits per cached value):

| knob | one step | cost in bits/value | relative to data bits |
|---|---|---|---|
| token retention | drop half the tokens | −2.00 | **−50%** |
| bit width | 4 → 2 or 4 → 8 | ∓2.00 | **∓50% / +100%** |
| quantization group size | 128 → 64 → 32 | +0.25 / +0.75 | +6% / +18% |
| metadata range check | off → on | **+0.003** | **+0.075%** |
| metadata SECDED | off → on | +0.10 | +2.3% |
| uniform data SECDED | off → on | +0.50 | +12.5% |

The axes are **not commensurate**. Retention and bit width are coarse knobs moving the budget by tens of percent;
group size and protection are fine knobs moving it by a few or by a fraction of one. A joint allocator therefore
has a two-level structure — the coarse pair selects an operating region, the fine pair trims inside it — and
treating all of them as one flat search, as §6's multiplicative score implies, spends the search on infeasible
combinations. That non-commensurability is itself a paper-worthy correction of the brief's premise (D22).

## M4 — token retention, built to be falsifiable

**`importance/`** — one interface, several scorers, and an oracle that is not a method:

* `recency` — keep the last k tokens. The trivial baseline most published methods must beat.
* `attention_mass` — attention received during prefill, summed over heads and layers. Needs attention weights, so
  it needs a second pass under eager attention (D5: `sdpa` returns none). That cost is measured and reported.
* `key_norm` / `value_norm` — cheap magnitude proxies.
* `oracle_span` — the probe's own span labels. **An upper bound, not a method.** If the oracle does not beat
  recency, the task cannot support a retention claim and M4 stops rather than producing a table.

**`compression/`** — apply a retention mask to a `KVSnapshot`, with two properties that must hold:

1. **Positions stay absolute.** Retained tokens keep their original RoPE positions; pruning must not renumber.
   `tests/test_protocol_equivalence.py` already proves prune-equals-mask for middle pruning, so M4 extends an
   established fact rather than assuming a new one.
2. **Budget is per original token.** After pruning, bits per *retained* token flatters pruning automatically.
   Every comparison uses total bits and bits per *original* token; `BitBudget` already keeps `retained_tokens`
   separate for exactly this reason.

**The M4 experiment.** Start at the published optimum (many tokens, 4-bit) rather than re-deriving it. Retention
∈ {100, 75, 50, 25}% × scorer ∈ {recency, attention_mass, key_norm, oracle_span}, on the probe, reporting task
accuracy, answer margin, and the fraction of *answer-bearing* tokens each scorer actually kept. That last column
is the diagnostic: a scorer that holds accuracy while discarding the answer tokens has found redundancy, which
would echo D19–D21 and is a finding in its own right.

**Then the question that is ours:** repeat the frontier under faults. If the token-precision optimum shifts once
reliability is priced — and it should, because more tokens at lower precision means more metadata — that is the
result the compression literature is missing.

## M5 — joint allocation

Decision variables per region: `(retain, bits, group_size, protection)`. Search is staged, per D22:

1. **Coarse:** pick (retention, bits) on the iso-bit frontier, informed by M4.
2. **Fine:** spend the remainder on group size, with protection set from the deployment's error rate rather than
   searched — M3 showed there is nothing to search there.

Baselines that must be beaten at equal total bits, none weakened:

* uniform 4-bit, all tokens, metadata protected (M3 arm G/H);
* **KIVI at its own optimum** — per-channel keys group 32, per-token values, **`residual_window: 128`**. Our
  configs currently set 0, which is not faithful (see `docs/literature.md`); a comparison against KIVI without
  its residual window would be a weakened baseline, and D15 must be re-measured in that configuration before it
  is used to argue against KIVI's axis choice;
* magnitude-only allocation, the strongest single-signal policy;
* recency-only retention at the matched budget.

**Pre-registered refutation:** if the joint policy does not beat *every* single-signal policy at equal bits on at
least two model families, H3 is refuted and reported as such. Given Q48's anti-correlation (D18), a
multiplicative joint score is *expected* to underperform an additive or staged one, so §6's form is one candidate
among several rather than the definition.

## Order of work

M3 closure → M4 (importance + compression, starting from the published optimum, then the same frontier under
faults) → M5 staged allocator with a faithful KIVI baseline → second model family (Llama-3.2-3B, identical KV
geometry, so family separates from geometry) → M6 agent benchmark.
