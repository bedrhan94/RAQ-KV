"""The allocator encodes measurements, so the tests check it against those measurements, not against taste."""

from __future__ import annotations

import pytest

from raq_kv.allocation.policy import MIN_BITS, allocate, single_signal


def test_it_never_goes_below_the_measured_precision_floor():
    """D40: at ~1.05 bits per original token, 2-bit at 40.5% retention loses to 4-bit at 25% by +0.1282
    (p = 0.00027) with 100% of the answer tokens kept in both. Below 4 bits extra tokens make things worse."""
    for budget in (0.3, 0.5, 1.05, 2.0, 4.0):
        plan = allocate(budget)
        assert plan.bits is None or plan.bits >= MIN_BITS


def test_it_reproduces_the_measured_joint_point():
    """The best configuration M4 measured is 25% of tokens at 4 bits, 1.054 bits per original token."""
    plan = allocate(1.0543)
    assert plan.bits == 4 and plan.group_size == 128
    assert plan.retention_ratio == pytest.approx(0.25, abs=0.005)
    assert plan.unit == "span" and plan.protect_initial == 4


def test_a_bigger_budget_buys_tokens_before_precision():
    """D38 measured more tokens beating higher precision at every step from bf16 down to the floor."""
    small, large = allocate(1.0), allocate(3.0)
    assert large.retention_ratio > small.retention_ratio
    assert large.bits == small.bits == MIN_BITS


def test_a_budget_that_covers_everything_stops_compressing():
    plan = allocate(16.0)
    assert plan.bits is None and plan.retention_ratio == 1.0


def test_protection_is_a_lookup_on_the_error_rate_not_a_search():
    """D30: the free range check is sufficient below ~1e-6 and loses decisively at 1e-5."""
    cheap, dear = allocate(1.0, ber=1e-7), allocate(1.0, ber=1e-5)
    assert cheap.metadata_guard == "range" and cheap.metadata_codec == "none"
    assert dear.metadata_codec == "secded" and dear.metadata_guard is None
    # Protection never changes the precision or retention decision: the axes are separable (D22, D25-D31).
    assert (cheap.bits, cheap.retention_ratio) == (dear.bits, dear.retention_ratio)


def test_the_scorer_degrades_honestly_when_the_query_is_unknown():
    """D36: query-awareness is worth 57 points, and it is unavailable on a shared prefix cache."""
    known, shared = allocate(1.0, query_known=True), allocate(1.0, query_known=False)
    assert known.scorer == "query_attention" and shared.scorer == "window_attention"
    assert any("shared prefix cache" in r for r in shared.rationale)


def test_every_decision_carries_its_evidence():
    plan = allocate(1.0)
    assert plan.rationale, "an allocation with no stated reasons is not auditable"
    assert any("D40" in r for r in plan.rationale) and any("D32" in r for r in plan.rationale)


@pytest.mark.parametrize("kind", ["precision_only", "retention_only", "recency_retention"])
def test_single_signal_policies_never_overspend(kind):
    """A baseline that exceeds the budget is not a comparison. Infeasible is reported, not fudged."""
    for budget in (0.5, 1.05, 4.0, 9.0):
        plan = single_signal(kind, budget)
        if plan is None:
            continue
        assert plan.bits_per_original_token <= budget + 1e-9, f"{kind} overspent at {budget}"


def test_precision_only_does_not_exist_at_the_budgets_the_joint_point_operates_at():
    """Quantization alone bottoms out at 2-bit, about 2.63 bits per original token with its scales.

    So at the budget where M4's joint point lives (1.054), the precision-only policy is not weak - it is
    unavailable. That is a result about the axis, and reporting it as a number would require letting the
    baseline overspend, which is the failure this returns None to avoid.
    """
    assert single_signal("precision_only", 1.054) is None
    assert single_signal("precision_only", 0.5) is None
    feasible = single_signal("precision_only", 4.5)
    assert feasible is not None and feasible.bits == 4 and feasible.retention_ratio == 1.0


def test_single_signal_policies_vary_the_axis_they_name():
    b = 4.5  # precision_only exists only where a ladder rung fits; 1.05 is below every rung
    assert single_signal("precision_only", b).retention_ratio == 1.0
    assert single_signal("retention_only", b).bits is None
    assert single_signal("recency_retention", b).scorer == "recency"
    with pytest.raises(ValueError, match="unknown single-signal policy"):
        single_signal("wishful", b)


def test_the_allocator_spends_a_remainder_that_a_laddered_policy_strands():
    """At 9.0 bits precision_only must sit on the 8-bit rung and strand 0.74.

    The joint policy caps retention at every token and then spends the remainder on finer quantization groups,
    which is the fine stage D22 reserved for exactly this. Without it the allocator would strand the same
    budget the baseline does, and "joint" would be a label rather than a behaviour.
    """
    b = 9.0
    joint, po = allocate(b), single_signal("precision_only", b)
    assert po is not None and po.bits_per_original_token == pytest.approx(8.2567)
    assert joint.retention_ratio == 1.0
    assert joint.bits_per_original_token > po.bits_per_original_token
    assert joint.bits_per_original_token <= b + 1e-9
    assert joint.group_size < 128, "the remainder should buy finer groups"
    assert any("group size" in r for r in joint.rationale)


def test_it_does_not_escalate_precision_when_full_retention_already_fits():
    """An earlier version walked the ladder cheapest-first and, at 4.30 bits, chose 8-bit at 52% of tokens
    over 4-bit at every token — exactly the choice D38 and D40 measured as worse. The rule is to take the
    highest precision whose FULL-retention cost fits, and trade tokens only when even the floor does not."""
    plan = allocate(4.30)
    assert plan.bits == 4 and plan.retention_ratio == 1.0
    assert plan.bits_per_original_token == pytest.approx(4.2567)


@pytest.mark.parametrize("budget", [0.2, 0.5, 1.05, 2.0, 3.0, 4.2566, 4.2567, 4.3, 6.0, 8.3, 9.0, 15.9, 16.0])
def test_the_allocation_never_exceeds_its_budget_at_any_rung_boundary(budget):
    """Rung boundaries are where an off-by-one in the ladder shows up, so they are enumerated."""
    plan = allocate(budget)
    assert plan.bits_per_original_token <= budget + 1e-9, f"overspent at {budget}"
    assert 0.0 < plan.retention_ratio <= 1.0


@pytest.mark.parametrize("kind,budget", [("retention_only", 1.05), ("retention_only", 2.0),
                                         ("precision_only", 16.0)])
def test_an_unquantized_policy_claims_no_metadata_protection(kind, budget):
    """Protection acts on quantization metadata, so a policy that does not quantize cannot claim any.

    The emitter wrote `metadata_guard: range` into both retention-only arms of H3 and the config schema
    refused them -- correctly, since there is no metadata in an unquantized cache. The bug was the allocator's:
    it ran the protection lookup before it knew whether anything would be quantized. This is the one place the
    axes are genuinely coupled, so the coupling is asserted rather than left to the schema to catch later.
    """
    plan = single_signal(kind, budget)
    assert plan is not None and plan.bits is None
    assert plan.metadata_guard is None and plan.metadata_codec == "none"
    assert any("no quantization metadata to protect" in r for r in plan.rationale)


def test_the_uncompressed_allocation_also_claims_no_metadata_protection():
    plan = allocate(16.0)
    assert plan.bits is None and plan.metadata_guard is None and plan.metadata_codec == "none"


@pytest.mark.parametrize("kind", ["recency_retention", "precision_only"])
def test_a_quantized_policy_still_gets_the_guard(kind):
    """The fix must not silently disarm protection on the policies that do quantize."""
    plan = single_signal(kind, 4.3)
    assert plan is not None and plan.bits is not None
    assert plan.metadata_guard == "range"
