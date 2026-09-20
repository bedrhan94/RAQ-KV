"""Retention: does the plan keep what it says, cost what it says, and survive being handed scores out of order?

The equivalence that makes pruning meaningful at all — pruned cache equals masked full cache — is already pinned
in ``tests/test_protocol_equivalence.py`` and is not repeated. What is pinned here is everything between a score
vector and that call: ordering, the recent-tail guarantee, the budget convention, and the two oracles.
"""

from __future__ import annotations

import pytest
import torch

from raq_kv.compression.retain import apply_retention, select_keep
from raq_kv.importance.scorers import ScoreContext, build_scorer
from raq_kv.kv.snapshot import KVSnapshot


def _snapshot(tokens: int = 16, layers: int = 3, heads: int = 2, dim: int = 8) -> KVSnapshot:
    g = torch.Generator().manual_seed(4)
    shape = (1, heads, tokens, dim)
    return KVSnapshot(keys=[torch.randn(shape, generator=g) for _ in range(layers)],
                      values=[torch.randn(shape, generator=g) for _ in range(layers)],
                      positions=torch.arange(tokens), next_position=tokens)


def test_keep_indices_are_sorted_whatever_order_the_scores_imply():
    """`select_tokens` needs strictly increasing indices; top-k returns score order.

    Getting this wrong would not raise in a variant that skipped the check — the shapes stay valid and the cache
    is silently permuted — so the property is asserted directly rather than trusted to the callee.
    """
    scores = torch.tensor([0.1, 9.0, 0.2, 8.0, 0.3, 7.0, 0.4, 5.0])
    plan = select_keep(scores, 0.5, protect_initial=0)  # the sink rule is tested separately
    assert plan.keep.tolist() == sorted(plan.keep.tolist())
    assert bool((plan.keep[1:] > plan.keep[:-1]).all())
    assert plan.keep.tolist() == [1, 3, 5, 7]


def test_reordering_equal_scores_does_not_change_the_retained_cache():
    """The same set of winners must give the same cache regardless of the order they were discovered in."""
    snap = _snapshot()
    scores = torch.zeros(16)
    scores[[2, 7, 11]] = 5.0
    a = apply_retention(snap, select_keep(scores, 3 / 16, protect_initial=0))
    reordered = torch.zeros(16)
    reordered[[11, 7, 2]] = 5.0  # same set, written in the opposite order
    b = apply_retention(snap, select_keep(reordered, 3 / 16, protect_initial=0))
    assert torch.equal(a.positions, b.positions)
    for x, y in zip(a.keys + a.values, b.keys + b.values, strict=True):
        assert torch.equal(x, y)


def test_positions_stay_absolute_after_pruning():
    """A retained token keeps its original RoPE index; pruning must not renumber (conventions §3, D4)."""
    snap = _snapshot(tokens=20)
    plan = select_keep(torch.arange(20, dtype=torch.float32), 0.5, protect_initial=0)
    pruned = apply_retention(snap, plan)
    assert pruned.positions.tolist() == plan.keep.tolist()
    assert pruned.next_position == snap.next_position == 20
    assert pruned.seq_len == 10


@pytest.mark.parametrize("ratio", [1.0, 0.75, 0.5, 0.25, 0.1])
def test_budget_is_counted_per_original_token(ratio):
    """Bits per *retained* token improves for free under pruning, which flatters every policy. The plan therefore
    reports the original count alongside, and the ratio is what a comparison holds fixed."""
    plan = select_keep(torch.randn(80), ratio, protect_initial=0)
    assert plan.original_tokens == 80
    assert plan.retained == max(1, round(80 * ratio))
    assert plan.ratio == pytest.approx(plan.retained / 80)
    assert plan.to_dict()["tokens_original"] == 80


def test_recent_tail_is_protected_even_when_it_scores_worst():
    """Every published method keeps a recent block; without it a scorer is judged on rediscovering recency."""
    scores = torch.cat([torch.ones(12) * 9.0, torch.zeros(4)])  # the last four are the worst possible
    plan = select_keep(scores, 0.5, protect_recent=4, protect_initial=0)
    assert set(range(12, 16)) <= set(plan.keep.tolist())
    assert plan.protected_recent == 4 and plan.retained == 8


def test_protection_never_exceeds_the_budget():
    plan = select_keep(torch.randn(10), 0.2, protect_recent=8, protect_initial=0)
    assert plan.retained == 2 and plan.protected_recent == 2
    assert plan.keep.tolist() == [8, 9]


def test_the_two_oracles_are_opposites_and_both_are_labelled_oracles():
    snap = _snapshot(tokens=12)
    ctx = ScoreContext(snapshot=snap, target_tokens=(3, 4, 5))
    keep_scorer, drop_scorer = build_scorer("oracle_keep_target"), build_scorer("oracle_drop_target")
    assert keep_scorer.is_oracle and drop_scorer.is_oracle
    kept = select_keep(keep_scorer.score(ctx), 0.5, protect_initial=0).keep.tolist()
    dropped = select_keep(drop_scorer.score(ctx), 0.5, protect_initial=0).keep.tolist()
    assert {3, 4, 5} <= set(kept), "the keeping oracle must retain the answer tokens"
    assert not ({3, 4, 5} & set(dropped)), "the dropping oracle must discard exactly those"


def test_norm_scorers_rank_a_planted_large_token_first():
    snap = _snapshot(tokens=10)
    for layer in snap.keys:
        layer[0, :, 7, :] *= 50.0
    scores = build_scorer("key_norm").score(ScoreContext(snapshot=snap))
    assert int(scores.argmax()) == 7
    assert scores.shape == (10,) and scores.dtype == torch.float32


def test_scorers_declare_what_they_need_so_a_leak_is_visible():
    """A scorer that reads ground truth is an oracle and says so; one that needs a forward pass says that too."""
    assert not build_scorer("recency").needs_model and not build_scorer("recency").is_oracle
    assert build_scorer("window_attention").needs_model
    assert not build_scorer("window_attention").is_oracle
    assert build_scorer("oracle_drop_target").is_oracle
    with pytest.raises(ValueError, match="unknown scorer"):
        build_scorer("wishful_thinking")


def test_window_attention_refuses_to_guess_without_a_model():
    with pytest.raises(ValueError, match="needs the model"):
        build_scorer("window_attention").score(ScoreContext(snapshot=_snapshot()))


def test_a_full_ratio_is_a_no_op_that_returns_the_same_object():
    snap = _snapshot()
    plan = select_keep(torch.randn(16), 1.0)
    assert apply_retention(snap, plan) is snap, "retaining everything must not copy the cache"


def test_kept_fraction_reports_the_diagnostic_column():
    plan = select_keep(torch.tensor([9.0, 0.0, 9.0, 0.0]), 0.5, protect_initial=0)
    assert plan.kept_fraction_of((0, 2)) == 1.0
    assert plan.kept_fraction_of((1, 3)) == 0.0
    assert plan.kept_fraction_of((0, 1)) == 0.5


def test_the_attention_sink_is_protected_by_default():
    """Position 0 is an attention sink: dropping it alone fails 12 of 12 probe answers, while dropping ten
    middle filler tokens fails 1 of 12 (D32). A retention study that does not pin it measures the sink instead
    of its scorer, which is what the first M4 gate did. So the default protects it, and the default is tested."""
    scores = torch.zeros(40)
    scores[:4] = -99.0  # the sink scores worst; it must survive anyway
    plan = select_keep(scores, 0.5)
    assert set(range(4)) <= set(plan.keep.tolist())
    assert plan.protected_initial == 4
    assert select_keep(scores, 0.5, protect_initial=0).keep.tolist()[:4] != [0, 1, 2, 3]


def test_head_and_tail_protection_compose_without_double_counting():
    scores = torch.zeros(20)
    plan = select_keep(scores, 0.5, protect_recent=3, protect_initial=2)
    keep = plan.keep.tolist()
    assert keep == sorted(set(keep)), "an index protected by both ends must not be counted twice"
    assert {0, 1, 17, 18, 19} <= set(keep)
    assert plan.retained == 10 and plan.protected_initial == 2 and plan.protected_recent == 3


def test_protection_shrinks_to_fit_a_tiny_budget():
    plan = select_keep(torch.randn(20), 0.1, protect_recent=5, protect_initial=5)
    assert plan.retained == 2, "the budget wins; protection cannot inflate the cache"
    assert plan.protected_initial == 2 and plan.protected_recent == 0
