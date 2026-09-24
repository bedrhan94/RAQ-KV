"""Interval estimators shared by the report scripts, so one definition cannot drift into two.

The Wilson score interval lived only in `scripts/plot_m3.py` until 2026-09-24, when writing M2's exit checklist
found that M2's own criterion -- "heatmaps reported with Wilson CI" -- had never been met: the fault map carried
bare means with no intervals, and there was no M2 figure at all. Copying the estimator into a second script would
have been the obvious fix and the wrong one; this project has already had to fix two cases of one quantity
computed twice and drifting (the budget predictor against the real encoders, the span selector across two tasks).
"""

from __future__ import annotations

Z_95 = 1.959963985


def wilson(k: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """95% Wilson score interval for ``k`` successes of ``n``.

    Chosen over the normal approximation for the reason most cells in a fault map are small and several are
    **zero**: at k = 0 a normal interval is degenerate (width zero, centred on zero), which reads as certainty
    where there is none. The Wilson interval stays finite and asymmetric there, which is the whole point -- a
    cell with 0 of 8 flips changing a token is not the same evidence as 0 of 800.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1.0 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def wilson_row(k: int, n: int) -> dict[str, float | int]:
    """The same interval as a flat record, for tables that want it beside the point estimate."""
    low, high = wilson(k, n)
    return {"k": k, "n": n, "rate": (k / n) if n else 0.0, "ci_low": low, "ci_high": high}
