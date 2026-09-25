"""The joint allocator: a bit budget in, a cache configuration out.

Every rule here is a measurement from M1-M4, cited by decision number, and the module exists in this shape
because the measurements refused the shape the brief assumed. §6 proposed one multiplicative score over
semantic importance, quantization sensitivity and fault vulnerability. Four things make that the wrong object:

* the axes are **not commensurate** (D22) - retention and bit width move a budget by tens of percent, group
  size and protection by a few or a fraction of one, so a flat search spends itself on infeasible corners;
* protection is **not a search dimension** (D25-D31) - it is a lookup on the deployment's error rate, and
  uniform data ECC is never worth its bits;
* the two sensitivity signals the score would multiply are **not one signal** on either model family measured
  (D18, D52) -- Qwen3-1.7B rho = -0.377, Llama-3.2-3B rho = +0.073, both far inside the frozen "separate
  signals" branch -- so their product carries no information the factors do not already carry. *(D52 corrected
  the wording here: this used to read "anti-correlated, so their product is not a ranking anyone should want",
  which is true of Qwen and not of Llama, where the two are simply orthogonal. Both reject the multiplicative
  form; only one supports that reason for it.)*
* the retention decision is over **spans, all-or-nothing across layers** (D32, D34), not over (layer, token)
  cells, so half the cells a per-cell score would rank do not exist as choices.

What replaces it is a staged rule with one measured switching point, which is the part that could not be
derived and had to be run (D40).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

# Ladder of admissible precisions, cheapest first, with the measured cost of each at its best group size.
# Costs are bits per *stored* value including fp16 scales and zero points, from the pinned predictor at the
# probe's mean context; they are the x-axis of every iso-budget comparison in M3 and M4.
_PRECISIONS: tuple[tuple[int, int, float], ...] = (
    (2, 32, 2.6284),
    (3, 64, 3.3806),
    (4, 128, 4.2567),
    (8, 128, 8.2567),
)
_BF16_COST = 16.0

# The fine knob (D22). Group size moves a budget by a few percent where retention and bit width move it by
# tens, so it is what a remainder is spent on once the coarse pair is fixed. Costs are from the pinned
# predictor for keys group_token at each size with values per_token, at the probe's mean context.
_GROUPS: dict[int, tuple[tuple[int, float], ...]] = {
    2: ((128, 2.2567), (64, 2.3750), (32, 2.6284), (16, 3.1250)),
    3: ((128, 3.2567), (64, 3.3806), (32, 3.6250), (16, 4.1250)),
    4: ((128, 4.2567), (64, 4.3750), (32, 4.6250), (16, 5.1250)),
    8: ((128, 8.2567), (64, 8.3750), (32, 8.6250), (16, 9.1250)),
}

# D40: the frontier has an interior optimum. Spending a budget on more tokens keeps paying down to 4 bits and
# reverses below it - 2-bit at 40.5% retention loses to 4-bit at 25% by +0.1282 (p = 0.00027) with 100% of the
# answer tokens kept in both, so the damage is per-token quantization error, not lost content. 3-bit is
# statistically indistinguishable from 4-bit (p = 0.32), so the optimum is a plateau and the floor is the
# conservative end of it.
MIN_BITS = 4

# D32/D34/D36: retention operates on punctuation-delimited spans, keeps an initial block because position 0 is
# an attention sink, and uses the query when the query is known at compression time.
_SINK = 4


@dataclass(frozen=True)
class Allocation:
    """A complete cache configuration, plus why each field has the value it has."""

    retention_ratio: float
    bits: int | None  # None means the model dtype
    group_size: int
    scorer: str
    unit: Literal["token", "span"]
    protect_initial: int
    metadata_guard: str | None
    metadata_codec: str
    bits_per_original_token: float
    rationale: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "retention_ratio": self.retention_ratio, "bits": self.bits, "group_size": self.group_size,
            "scorer": self.scorer, "unit": self.unit, "protect_initial": self.protect_initial,
            "metadata_guard": self.metadata_guard, "metadata_codec": self.metadata_codec,
            "bits_per_original_token": self.bits_per_original_token, "rationale": list(self.rationale),
        }


def _protection(ber: float) -> tuple[str | None, str, str]:
    """Protection is a lookup, not a search (D30).

    Below ~1e-6 a range check on the quantization metadata costs 0.075% of storage and removes every measured
    fault-induced failure; above it the check loses decisively to metadata SECDED at 2.3% (+0.0593, p = 1.1e-5)
    because the thirteen of sixteen float16 bits it cannot see accumulate. Uniform data ECC at 14% is never
    chosen: arm B matched arm G at 1e-6 and nothing has separated them since.

    **This lookup takes no retention argument, and D43/D45 showed that is wrong as a mechanism.** At equal fault
    *exposure* a cache pruned to 25% fails about twice as often at low exposure -- 0.2232 against 0.1116 at 3e-7
    (+0.1116 [+0.0536, +0.1719], p = 4.6e-04) and 0.4018 against 0.3125 at 1e-6 (+0.0893 [+0.0223, +0.1562],
    p = 0.016), a decade apart and both meeting the pre-registered rule. Query-aware span retention keeps exactly
    the answer-bearing spans, so every scale that survives pruning governs a group the answer depends on, where
    most groups in a full cache govern filler. Protection is therefore **not separable from retention**, which is
    the one thing the brief's multiplicative form got right. (The risk ratio falls 2.00 -> 1.29 -> 1.02 as the
    unprotected base rate rises 11% -> 31% -> 94%, so the 1e-5 null is saturation rather than absence.)

    **The constants below are unchanged, and whether they should be is open in both directions.** D43 asserted
    here that the lookup is "unconservative for aggressively pruned caches", on the strength of a 1e-6 contrast
    with **4 discordant samples** at p = 0.066. D44 measured the same contrast at 1e-5 and it went the other way:
    the pruned guarded arm reaches 0.0312 against the unpruned arm's 0.0603, and escalating from the guard to
    metadata SECDED on a pruned cache buys nothing (-0.0022, p = 0.91) where on an unpruned cache at that rate it
    was decisive (D30: +0.0593, p = 1.1e-5). The guard's breakdown tracks **flip count**, and a pruned cache at a
    matched rate takes about four times fewer flips, so it sits below the breakdown.

    So what is established is narrower than D43 claimed:

    * the concentration effect is real on an **unprotected** cache at 1e-6;
    * under this guard it is **not measurable at either rate**, and the sign is inconsistent (+0.0268 at 1e-6 on
      4 discordant samples, -0.0290 at 1e-5 on 25);
    * therefore a retention term is neither justified nor ruled out, and the earlier "unconservative" wording
      overstated a direction the project's own rule (D31) says to report as a direction.
    """
    if ber <= 1e-6:
        return "range", "none", "range check on metadata (D30): +0.075% storage, sufficient at this rate"
    return None, "secded", "metadata SECDED (D30): +2.3% storage, needed above ~1e-6 where the guard breaks"


def _protection_for(bits: int | None, ber: float) -> tuple[str | None, str, str]:
    """The same lookup, except that an unquantized cache has nothing for it to protect.

    Every protection mechanism this project chose acts on the *quantization metadata* (D25-D30), so a policy
    that does not quantize cannot use any of them, and the config schema rightly refuses the combination. This
    is not a missing feature: it is the one place where the axes are coupled rather than separable, and it means
    a retention-only policy is unprotectable by the mechanism that M3 measured as the best value per bit. The
    only protection left to it is uniform data ECC at +14%, which D25 measured as never worth its bits.
    """
    if bits is None:
        return None, "none", ("no quantization, so no quantization metadata to protect (D25-D30): this policy "
                              "cannot use the +0.075% guard at all, and its only option is uniform data ECC at "
                              "+14%, which is never worth its bits")
    return _protection(ber)


def allocate(budget_bits_per_original_token: float, ber: float = 1e-6, query_known: bool = True) -> Allocation:
    """Choose retention, precision and protection for a budget, by the measured rules.

    ``budget_bits_per_original_token`` is per *original* token, not per retained one: bits per retained token
    improves for free under pruning and would flatter every policy (the convention `BitBudget` keeps
    ``retained_tokens`` separate to enforce).
    """
    if budget_bits_per_original_token <= 0:
        raise ValueError("budget must be positive")
    guard, codec, protection_why = _protection(ber)
    scorer = "query_attention" if query_known else "window_attention"
    scorer_why = (
        "query-aware attention (D36): worth 57 points over the query-agnostic scorer, and the query is known"
        if query_known else
        "query-agnostic attention (D36): the query is not known at compression time, as on a shared prefix "
        "cache, so the best available arm sits at 0.624 and the gap to 0.026 is an open problem"
    )

    # Coarse stage. Walking the ladder the wrong way is a real hazard and an earlier version did it: at a
    # budget of 4.30 it escalated to 8-bit at 52% of tokens rather than taking 4-bit at every token, which is
    # precisely the choice D38 and D40 measured as worse. The rule that matches the measurements is to take the
    # HIGHEST precision whose full-retention cost fits, and only trade tokens when even the floor does not fit.
    ladder = [(b, g, c) for b, g, c in _PRECISIONS if b >= MIN_BITS]
    floor_bits, floor_group, floor_cost = ladder[0]
    if budget_bits_per_original_token >= _BF16_COST:
        # Unquantized, so the metadata guard has nothing to guard; say so rather than emitting a config the
        # schema will reject.
        bf16_guard, bf16_codec, bf16_why = _protection_for(None, ber)
        return Allocation(1.0, None, 0, scorer, "span", _SINK, bf16_guard, bf16_codec, _BF16_COST,
                          ("budget covers the uncompressed cache; nothing to trade", bf16_why))
    if budget_bits_per_original_token < floor_cost:
        # Scarce: hold the precision floor and spend everything on tokens (D38, down to the D40 turning point).
        chosen = (floor_bits, floor_group, floor_cost, budget_bits_per_original_token / floor_cost)
    else:
        affordable = [(b, g, c) for b, g, c in ladder if c <= budget_bits_per_original_token]
        bits, group, cost = affordable[-1]
        chosen = (bits, group, cost, 1.0)

    bits, group, cost, ratio = chosen
    # Fine stage (D22): with retention capped at every token, a remainder can still buy finer quantization
    # groups. Without this the allocator strands budget - at 9.0 bits it would sit at 8-bit block-128 and
    # leave 0.74 unspent, which is exactly the waste a "joint" policy exists to avoid.
    fine_note = ""
    if ratio >= 1.0:
        ratio = 1.0
        for candidate_group, candidate_cost in _GROUPS[bits]:
            if candidate_cost <= budget_bits_per_original_token and candidate_cost >= cost:
                group, cost = candidate_group, candidate_cost
        if group != _GROUPS[bits][0][0]:
            fine_note = (f"group size {group} rather than 128 (D22): retention is already at every token, so "
                         f"the remainder buys finer quantization groups instead of being stranded")
    rationale = [
        f"precision floor {MIN_BITS} bits (D40): below it extra tokens make things worse, not better",
        f"retention {ratio:.3f} of original tokens, spending the budget on tokens down to that floor (D38)",
        "spans, not tokens (D34): the unit is worth +18 to +26 points on every scorer tried",
        f"first {_SINK} tokens pinned (D32): dropping position 0 alone fails every answer",
        scorer_why,
        protection_why,
    ]
    if fine_note:
        rationale.insert(1, fine_note)
    return Allocation(round(ratio, 4), bits, group, scorer, "span", _SINK, guard, codec,
                      ratio * cost, tuple(rationale))


def single_signal(kind: str, budget_bits_per_original_token: float,
                  ber: float = 1e-6) -> Allocation | None:
    """The policies H3 requires the joint allocator to beat, each varying one axis and fixing the rest.

    These are not strawmen: ``precision_only`` is what the quantization literature does, ``retention_only`` is
    what the eviction literature does, and ``recency_retention`` is the trivial baseline that published methods
    are expected to beat and sometimes do not (D35).

    **Returns ``None`` when the policy cannot meet the budget at all**, which is the honest outcome rather than
    letting it overspend and calling the result a comparison. Quantization alone bottoms out at 2-bit, i.e.
    about 2.63 bits per original token with its scales, so a precision-only policy simply does not exist below
    that - and that is a result about the axis, not a gap in the code.
    """
    b = budget_bits_per_original_token
    if kind == "precision_only":
        for bits, group, cost in ((16, 0, _BF16_COST),) + tuple(reversed(_PRECISIONS)):
            if cost <= b:
                stored = None if bits == 16 else bits
                guard, codec, protection_why = _protection_for(stored, ber)
                return Allocation(1.0, stored, group, "recency", "token", _SINK, guard, codec, cost,
                                  ("all tokens kept; the budget buys precision only", protection_why))
        return None  # below ~2.63 bits per original token this policy has nothing left to give up
    if kind == "retention_only":
        guard, codec, protection_why = _protection_for(None, ber)
        return Allocation(round(min(1.0, b / _BF16_COST), 4), None, 0, "query_attention", "span", _SINK,
                          guard, codec, min(_BF16_COST, b),
                          ("no quantization; the budget buys tokens only", protection_why))
    if kind == "recency_retention":
        bits, group, cost = _PRECISIONS[2]  # 4-bit, the measured safe point
        guard, codec, protection_why = _protection_for(bits, ber)
        return Allocation(round(min(1.0, b / cost), 4), bits, group, "recency", "token", _SINK, guard, codec,
                          min(cost, b), ("recency at the safe operating point: the trivial baseline",
                                         protection_why))
    if kind == "magnitude_only":
        # Pre-registered in `docs/m4_m5_plan.md` as "the strongest single-signal policy", and it is the one the
        # joint arm shares least with: key magnitude is computable at prefill, so unlike query-aware attention
        # it works on a shared prefix cache (D36's open problem). Held at 4 bits and token granularity so the
        # only thing varying against the joint arm is the *signal*, not the unit or the precision.
        bits, group, cost = _PRECISIONS[2]
        guard, codec, protection_why = _protection_for(bits, ber)
        return Allocation(round(min(1.0, b / cost), 4), bits, group, "key_norm", "token", _SINK, guard, codec,
                          min(cost, b),
                          ("key magnitude at the safe operating point: the strongest single-signal policy the "
                           "plan pre-registered, and the only baseline here that a shared prefix cache could "
                           "actually run (D36)", protection_why))
    raise ValueError(f"unknown single-signal policy {kind!r}")
