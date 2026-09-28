"""Pins the binning defect that silently dropped D76's coarsest rung, and the nan verdict that inverted its meaning.

``binned_fit`` built octave edges up to ``2**ceil(log2(max_span))`` and its bins are
half-open ``[lo, hi)``. The spans it bins are **group sizes**, so they are always powers
of two, so the largest span fell outside every bin at every single invocation. D76's
published exponent was fitted on 2,920 trials with **no 512 bin at all**, and D77's
channel sweep lost every span-32 flip. Nothing in the output said so: the bin list simply
ended one rung early, which reads as a formatting choice rather than a dropped population.
"""

from __future__ import annotations

import importlib.util
import math
import pathlib

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "report_d76", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "report_d76.py")
report_d76 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(report_d76)


def test_every_flip_lands_in_a_bin_when_spans_are_powers_of_two():
    """The defect, stated as the property it broke: binning is a partition, not a filter."""
    for top in (32, 128, 512):
        spans = np.repeat([2.0 ** k for k in range(3, int(math.log2(top)) + 1)], 200)
        delta = np.full(spans.shape, 0.25)
        slope, _, bins = report_d76.binned_fit(spans, delta)
        binned = sum(b["n"] for b in bins)
        assert binned == spans.size, (
            f"top span {top}: {spans.size - binned} of {spans.size} flips fell outside every bin")
        # The coarsest rung must have its own bin, which is exactly what was missing.
        assert any(b["lo"] <= top < b["hi"] for b in bins), [(b["lo"], b["hi"]) for b in bins]
        assert math.isfinite(slope)


def test_binning_refuses_to_fit_when_it_would_drop_a_population():
    """The guard, not just the fix: a future edge change must fail loudly rather than quietly re-lose a rung."""
    spans = np.repeat([8.0, 16.0, 32.0], 200)
    delta = np.full(spans.shape, 0.25)
    original = report_d76.math.ceil

    def truncating_ceil(x: float) -> int:
        # Stands in for the old `+ 1`: pushes the top edge down onto the largest span.
        return original(x) - 1

    report_d76.math.ceil = truncating_ceil
    try:
        with pytest.raises(AssertionError, match="binning dropped"):
            report_d76.binned_fit(spans, delta)
    finally:
        report_d76.math.ceil = original


def test_a_constant_delta_still_fits_a_zero_slope_after_the_fix():
    spans = np.repeat([8.0, 16.0, 32.0, 64.0, 128.0], 300)
    slope, r2, bins = report_d76.binned_fit(spans, np.full(spans.shape, 0.3))
    assert len(bins) == 5
    assert slope == pytest.approx(0.0, abs=1e-12)
    assert sum(b["n"] for b in bins) == spans.size


def test_a_known_exponent_is_recovered():
    # delta = c * span**alpha, so the binned log-log slope must return alpha.
    spans = np.repeat([8.0, 16.0, 32.0, 64.0, 128.0], 300)
    for alpha in (0.5, 1.0, -0.25):
        slope, r2, _ = report_d76.binned_fit(spans, 0.1 * spans ** alpha)
        assert slope == pytest.approx(alpha, abs=1e-9), alpha
        assert r2 == pytest.approx(1.0, abs=1e-9)


def test_too_few_bins_returns_nan_rather_than_a_slope():
    """And the caller must not read that nan as "every null excluded" -- see the report's own guard.

    With a nan interval, ``not lo <= v <= hi`` is vacuously true for every candidate null, so the first version of the
    report printed EXCLUDED for span-independence, quadrature and coherent all at once, on data that could not support
    a slope at all.
    """
    spans = np.repeat([8.0, 16.0], 200)
    slope, r2, bins = report_d76.binned_fit(spans, np.full(spans.shape, 0.3))
    assert math.isnan(slope) and math.isnan(r2)
    assert len(bins) == 2
    lo = hi = float("nan")
    for null in (0.0, 0.5, 1.0):
        assert not (lo <= null <= hi), "the vacuous truth this documents"
    assert not (math.isfinite(slope) and math.isfinite(lo) and math.isfinite(hi)), (
        "so the report must branch on finiteness before printing any verdict")
