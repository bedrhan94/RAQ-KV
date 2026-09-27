"""Pins the softmax-curvature identity and, above all, that its null slope is not -1 in a thin-margin regime.

D75 found that a negative margin-KL rank correlation is expected under no mechanism at all, and then nearly published
a second error on top of the first: it tested the slope against **-1**, which is only the deep-margin limit of
``d ln(p1 p2) / d margin = 1 - 2 p1``. At D74's median margin of 1.312 nats the null is -0.576, and a measured -0.571
was briefly read as a violation. ``test_fixed_delta_excludes_deep_limit_at_thin_margins`` is the regression test for
exactly that mistake: data generated with a *constant* perturbation must produce a slope that excludes -1 whenever
margins are thin, so a future reader cannot repeat the reading.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import special, stats

from raq_kv.analysis.curvature import (
    calibrated_slope_null,
    effective_perturbation,
    identity_slope,
    log_curvature,
    log_curvature_observed,
    two_state_probs,
)


def synth(margins: np.ndarray, delta: float | np.ndarray) -> np.ndarray:
    """The KL a two-state softmax produces for a given margin and logit perturbation -- the identity, forward."""
    p1, p2 = two_state_probs(margins)
    return 0.5 * p1 * p2 * np.asarray(delta, dtype=float) ** 2


def synth_noisy(margins: np.ndarray, delta: float, sd: float = 0.4, seed: int = 11) -> np.ndarray:
    """The same, with the perturbation varying trial to trial but **not** with the margin.

    Every CI-containment assertion below uses this rather than the exact form, because noiseless data gives a
    zero-width confidence interval and ``lo <= null <= hi`` then turns on the last bit of the slope -- a test that
    passes or fails on floating-point luck rather than on the statistics it claims to check.
    """
    rng = np.random.default_rng(seed)
    return synth(margins, delta * np.exp(rng.normal(0.0, sd, size=len(margins))))


def test_log_curvature_survives_the_probe_margins_where_a_naive_log_underflows():
    # The probe sits at 17-19 nats and p2 there is ~1e-8; but the campaign also reaches 25+ nats, where expit(-m)
    # underflows to exactly 0.0 in float64 and log(expit(-m)) is -inf. log_expit keeps it finite, which is the whole
    # reason the module does not spell this as log(expit(...)).
    deep = np.array([17.0, 19.0, 25.0, 40.0, 80.0, 745.0])
    got = log_curvature(deep)
    assert np.all(np.isfinite(got)), got
    # ln(p1 p2) -> -margin as p1 -> 1.
    assert got[-1] == pytest.approx(-745.0, abs=1e-6)


def test_effective_perturbation_inverts_the_identity():
    margins = np.array([0.5, 1.3, 3.0, 8.0, 17.0])
    for delta in (0.01, 0.25, 2.0):
        assert effective_perturbation(margins, synth(margins, delta)) == pytest.approx(delta, rel=1e-9)


def test_identity_slope_against_curvature_is_one_for_any_fixed_perturbation():
    # The version that does not depend on where the margins sit: regressing ln KL on ln(p1 p2) must give exactly 1 on
    # the exact identity, and must still recover 1 within its CI once the perturbation varies trial to trial.
    for margins in (np.linspace(0.25, 6.5, 80), np.linspace(15.0, 20.0, 80)):
        assert identity_slope(margins, synth(margins, 0.1))["slope_vs_curvature"] == pytest.approx(1.0, abs=1e-9)
        assert identity_slope(margins, synth_noisy(margins, 0.1))["identity_holds"]


def test_fixed_delta_excludes_deep_limit_at_thin_margins():
    """The regression test for D75's own near-miss: constant delta, thin margins, and -1 is *correctly* excluded.

    A slope CI that excludes -1 therefore proves nothing about the perturbation varying with the margin -- which is
    what the first pass concluded from d74's -0.571 [-0.767, -0.375].
    """
    margins = np.linspace(0.25, 6.5, 80)
    got = identity_slope(margins, synth_noisy(margins, 0.1))
    assert got["excludes_deep_limit"], "a fixed perturbation at thin margins must still exclude the -1 asymptote"
    assert got["contains_two_state"], got
    # And the null it does match is nowhere near -1.
    assert got["null_two_state"] > -0.95
    # The honest residual is flat, because delta here does not vary *with the margin*.
    assert got["residual_p"] > 0.05, got
    # On the exact identity the recovered delta is constant to float precision. Asserting that directly rather
    # than through residual_rho: a Spearman over a vector that is constant up to 1e-16 ranks pure rounding noise
    # and returns an arbitrary value (0.093 here), so it cannot stand in for "flat".
    assert np.ptp(effective_perturbation(margins, synth(margins, 0.1))) < 1e-12


def test_deep_margins_do_reproduce_the_minus_one_asymptote():
    margins = np.linspace(15.0, 20.0, 80)
    got = identity_slope(margins, synth_noisy(margins, 0.1))
    assert got["null_two_state"] == pytest.approx(-1.0, abs=1e-6)
    assert not got["excludes_deep_limit"]


def test_residual_detects_a_perturbation_that_really_does_vary():
    # delta rising with the margin: the detrended residual must see it, and see it positive.
    margins = np.linspace(0.25, 6.5, 200)
    got = identity_slope(margins, synth(margins, 0.05 * np.exp(0.2 * margins)))
    assert got["residual_rho"] > 0.99
    assert got["residual_p"] < 1e-6


def test_subtracting_the_margin_instead_of_the_curvature_invents_a_trend():
    """Why the module detrends with ln(p1 p2) and not with the margin -- the first pass did the latter.

    With a constant delta, ``ln KL + margin`` is not flat wherever p1 < 1: it carries ``ln(p1 p2) + margin``, which
    rises. That artefact was measured as rho = +0.482, p < 1e-4 and read as a finding.
    """
    margins = np.linspace(0.25, 6.5, 80)
    kl = synth(margins, 0.1)
    # Spearman, not Pearson: ln(p1 p2) + margin is strictly increasing in the margin (its derivative is 2(1 - p1) > 0)
    # but curved, so the rank correlation is exactly 1 while Pearson is only 0.84.
    assert stats.spearmanr(margins, np.log(kl) + margins).statistic == pytest.approx(1.0, abs=1e-12)
    # The right detrending recovers the constant on the same data: a spread of 1e-16 against the wrong
    # detrending's spread of ln(p1 p2) + margin over this range, which is about 1.1 nats.
    assert np.ptp(np.log(kl) - log_curvature(margins)) < 1e-12
    assert np.ptp(np.log(kl) + margins) > 1.0


def test_calibrated_null_is_shallower_than_two_state_in_the_thin_regime():
    # Llama's recorded p1 gives a shallower tail than two states imply (the non-top mass is 2.30x the runner-up), so
    # the calibrated null must sit between the two-state value and zero at a thin margin.
    two_state = float(1.0 - 2.0 * special.expit(1.312))
    calibrated = calibrated_slope_null(1.312, "llama-3.2-3b")
    assert calibrated is not None
    assert two_state < calibrated < 0.0, (two_state, calibrated)
    # Deep margins collapse the distinction; and an unknown family declines rather than guessing.
    assert calibrated_slope_null(18.0, "qwen3-1.7b") == pytest.approx(-1.0, abs=0.02)
    assert calibrated_slope_null(1.312, "mistral-7b") is None


def test_observed_curvature_survives_a_top1_probability_of_one():
    """Why the arms record ``nontop_mass`` with expm1 instead of letting an analysis recompute ``1 - p1``.

    Past about 36.7 nats of margin the runner-up mass falls below the gap between 1.0 and its neighbour, so ``1 - p1``
    is exactly 0.0 and the curvature term is -inf; ``-expm1(logp)`` recovers the tail from the logprob instead.
    """
    # Stated as what it is: NOT a regime any recorded arm has reached. Every one of the 80 rows in
    # answer_confidence.json still has top1_prob < 1.0, the deepest at 25.25 nats, and the rounding needs about
    # **36.7** nats. D62's "top-1 probability 1.0000 at float64" is a report rounded to four places, not an
    # underflow. The field is written with expm1 anyway because the cost is nil and the margin ceiling here is set by
    # what has been run, not by anything in the method.
    logp_top = -1.11e-17
    p1 = float(np.exp(logp_top))
    assert p1 == 1.0 and 1.0 - p1 == 0.0, "the failure mode this field exists to avoid"
    tail = float(-np.expm1(logp_top))
    assert tail == pytest.approx(1.11e-17, rel=1e-2)
    got = log_curvature_observed(np.array([p1]), np.array([tail]))
    assert np.all(np.isfinite(got))
    # And with a one-hot-to-float64 p1 the observed form agrees with the two-state form fed the same tail as a margin.
    assert got[0] == pytest.approx(np.log(tail), abs=1e-9)


def test_observed_curvature_uses_the_whole_tail_not_the_runner_up():
    # A distribution whose non-top mass is spread over many tokens has a larger curvature than the margin alone
    # implies, so delta recovered from the margin is an OVERestimate -- which is why D75's family gap is a lower bound.
    margin = 1.312
    p1, p2 = two_state_probs(margin)
    inflation = 2.30  # Llama's measured non-top mass against its runner-up alone
    observed = log_curvature_observed(np.array([p1]), np.array([p2 * inflation]))
    assert observed[0] > log_curvature(np.array([margin]))[0]
    kl = 1e-4
    delta_assumed = effective_perturbation(np.array([margin]), np.array([kl]))[0]
    delta_observed = float(np.sqrt(2.0 * kl / np.exp(observed[0])))
    assert delta_observed < delta_assumed
    assert delta_assumed / delta_observed == pytest.approx(np.sqrt(inflation), rel=1e-9)
