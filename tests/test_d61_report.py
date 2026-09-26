"""The D61 report's own arithmetic, tested away from the GPU.

The report is the thing a reviewer reads, and three of its rules are easy to get wrong in a direction that
overstates: a zero cell reported as a zero rather than a bound, a targeting summary that trusts the requested rank
instead of the realised one, and a contrast pooled over trials instead of paired over samples.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "report_d61", Path(__file__).resolve().parents[1] / "scripts" / "report_d61.py")
rep = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rep)


def _record(index: int, n_groups: int, requested: int, realised: int, share: float, changed: list[bool]):
    return {
        "golden_correct": True, "target_group_index": index, "n_token_groups": n_groups,
        "target_group_rank_requested": requested, "target_group_rank": realised, "target_group_share": share,
        "faults": ([{"trial": -1, "answer_changed": False}]
                   + [{"trial": i, "answer_changed": c} for i, c in enumerate(changed)]),
    }


def test_a_zero_cell_is_reported_as_a_bound_not_a_zero():
    """D30/D44. `0 of 320` is not "no risk"; the rule of three puts it at <= 0.94%."""
    assert "BOUND" in rep.rule_of_three(0, 320)
    assert "0.9375%" in rep.rule_of_three(0, 320)
    assert rep.rule_of_three(1, 320) == "", "a cell with events gets its Wilson interval, not a bound"
    assert rep.rule_of_three(0, 0) == "", "no trials is not a bound either"


def test_targeting_reports_the_realised_rank_and_the_real_position():
    """A tail cell asks for rank 999 and gets whatever the sample's length allows; the summary must say which."""
    data = {"a": _record(3, 9, 999, 8, 0.016, [False]),
            "b": _record(4, 9, 999, 8, 0.018, [False])}
    t = rep.targeting(data, ["a", "b"])
    assert t["requested_rank"] == 999.0 and t["realised_rank"] == 8.0
    assert t["is_first_group"] == 0.0 and t["is_last_group"] == 0.0, "the tail is not pinned to either end"
    assert t["position"] == pytest.approx((3 / 8 + 4 / 8) / 2)
    assert t["share_min"] == pytest.approx(0.016) and t["share_max"] == pytest.approx(0.018)


def test_targeting_detects_the_position_confound_it_exists_to_expose():
    """A rank-0 cell must come back as group 0 in every sample with zero position spread, or the amendment's
    premise is wrong for this run and the report says so instead of assuming it."""
    data = {"a": _record(0, 9, 0, 0, 0.77, [True]), "b": _record(0, 16, 0, 0, 0.68, [False])}
    t = rep.targeting(data, ["a", "b"])
    assert t["is_first_group"] == 1.0 and t["position"] == 0.0 and t["position_sd"] == 0.0
    # And a run where it is NOT always group 0 must not silently read as if it were.
    mixed = {"a": _record(0, 9, 0, 0, 0.77, [True]), "b": _record(5, 16, 0, 0, 0.68, [False])}
    assert rep.targeting(mixed, ["a", "b"])["is_first_group"] == 0.5


def test_pooled_counts_trials_and_excludes_the_no_fault_controls():
    data = {"a": _record(0, 9, 0, 0, 0.7, [True, False, True]),
            "b": _record(0, 9, 0, 0, 0.7, [False, False, False])}
    assert rep.pooled(data, ["a", "b"]) == (2, 6), "the control trial must not enter either count"


def test_the_contrast_is_paired_over_samples_not_pooled_over_trials():
    """Eight draws on one sample are not eight independent observations.

    Pooling would make a sample with more trials count more, and the two arms here have identical pooled rates
    with opposite per-sample patterns -- the case where pooling and pairing disagree.
    """
    from raq_kv.analysis.contrasts import contrast

    x = np.array([1.0, 0.0, 1.0, 0.0])
    y = np.array([0.0, 1.0, 0.0, 1.0])
    c = contrast("x", "y", x, y, seed=0)
    assert c["x_mean"] == c["y_mean"] == 0.5, "pooled, these arms are identical"
    assert c["difference"] == 0.0 and c["discordant"] == 4, "paired, every sample moved"
    assert not c["meets_rule"], "a mean difference of zero must never meet D31"
