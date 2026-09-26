"""D62's report arithmetic, away from the GPU.

Three of its rules decide whether the experiment means anything, so each is tested: the zero baseline must demand
*exactly* zero rather than approximately, the anchor must be able to report "no discrimination" rather than
producing a number regardless, and the per-sample summary must be a median over a heavy-tailed quantity.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "report_d62", Path(__file__).resolve().parents[1] / "scripts" / "report_d62.py")
rep = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rep)


def _rec(control_kl, trial_kls, changed=None, margin=8.0):
    changed = changed or [False] * len(trial_kls)
    return {
        "golden_correct": True,
        "faults": ([{"trial": -1, "kl_first_token": control_kl, "answer_changed": False,
                     "argmax_flipped": False, "margin_clean": margin}]
                   + [{"trial": i, "kl_first_token": k, "answer_changed": c, "argmax_flipped": c,
                       "margin_clean": margin}
                      for i, (k, c) in enumerate(zip(trial_kls, changed, strict=True))]),
    }


def test_the_zero_baseline_demands_exactly_zero_not_approximately():
    """A KL of 1e-12 in a control is not "close enough": it means the floor is nondeterminism of unknown size,
    and then a trial's 1e-6 cannot be attributed to the flip."""
    clean = {"a": _rec(0.0, [1e-6, 1e-3]), "b": _rec(0.0, [0.0, 2e-4])}
    assert rep.controls_are_exactly_zero(clean) == (0, 2, 0.0)

    bad, n, worst = rep.controls_are_exactly_zero({"a": _rec(1e-12, [1e-6]), "b": _rec(0.0, [1e-6])})
    assert (bad, n) == (1, 2) and worst == 1e-12, "a nonzero control must be counted, not rounded away"


def test_the_anchor_reports_no_discrimination_rather_than_inventing_a_number():
    """If KL does not separate breaking from non-breaking flips, the design is void and the report must say so."""
    rng = np.random.default_rng(0)
    kl = rng.random(400)
    # Outcome independent of KL: the AUC must sit at 0.5 and `discriminates` must be False.
    independent = rng.random(400) < 0.1
    a = rep.auc_with_ci(kl, independent, seed=0, n_boot=200)
    assert 0.35 < a["auc"] < 0.65 and a["discriminates"] is False

    # Outcome driven by KL: the AUC must be high and the CI must clear 0.5.
    driven = kl > 0.9
    b = rep.auc_with_ci(kl, driven, seed=0, n_boot=200)
    assert b["auc"] > 0.95 and b["ci"][0] > 0.5 and b["discriminates"] is True
    assert b["median_pos"] > b["median_neg"]


def test_the_anchor_survives_a_family_where_nothing_broke():
    """At a 2-6% break rate a 640-flip arm can return zero events, and the report must not divide by zero."""
    a = rep.auc_with_ci(np.array([1e-6, 1e-5, 1e-4]), np.array([False, False, False]), seed=0, n_boot=10)
    assert a["auc"] is None and a["n_pos"] == 0 and "one class is empty" in a["note"]


def test_the_per_sample_summary_is_a_median_so_one_outlier_cannot_carry_it():
    """KL over single-bit flips is heavy-tailed: most flips move nothing and a few move everything.

    With a mean, one flip of KL 1.0 among fifteen of 1e-6 reports 0.067 -- two orders above every observation but
    one -- and the cross-family contrast would be a comparison of whose outlier landed harder.
    """
    data = {"s": _rec(0.0, [1e-6] * 15 + [1.0])}
    med = rep.per_sample_median_kl(data, ["s"])
    assert med[0] == pytest.approx(1e-6), "the median must ignore the single large flip"
    assert float(np.mean([1e-6] * 15 + [1.0])) > 0.06, "the mean would not, which is why it is not used"


def test_flip_kls_flattens_trials_and_drops_the_controls():
    data = {"a": _rec(0.0, [1e-6, 1e-3], changed=[False, True]),
            "b": _rec(0.0, [2e-6], changed=[False])}
    kl, changed, flipped = rep.flip_kls(data, ["a", "b"])
    assert len(kl) == 3, "a control is not a flip and must not enter the distribution"
    assert changed.sum() == 1 and flipped.sum() == 1
    assert rep.per_sample_clean_margin(data, ["a", "b"]).tolist() == [8.0, 8.0]
