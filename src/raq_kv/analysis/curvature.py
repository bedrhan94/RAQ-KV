"""Separating how far a fault moves the logits from how much the softmax amplifies that at a given margin.

**Why this module exists.** D62 and D73 both used the first-token KL as a measure of how much damage a fault does, and
both compared it across models and vehicles whose answer margins differ by many nats. That comparison does not measure
what it appears to. For a small logit perturbation the KL is a quadratic form in the softmax Fisher information
``diag(p) - p p^T``; for two effective states that is

    KL  ~=  0.5 * p1 * p2 * delta^2         so      ln KL  =  ln(p1 p2)  +  2 ln delta  +  const

where ``delta`` is the perturbation in logit units and ``p1``, ``p2`` are the top two probabilities. Since
``p2 / p1 = exp(-margin)``, **a fault of fixed size produces a KL that falls by roughly a factor of e per nat of
margin, through nothing but the shape of the softmax.** So a cross-vehicle KL ratio between distributions of different
peakedness is mostly a restatement of their margins, and a negative correlation between margin and KL is expected
under no mechanism at all.

``delta`` is the quantity those records wanted: curvature-free, in logit units, and comparable across models and
tasks. ``effective_perturbation`` inverts the identity to recover it.

**The slope null is not -1.** ``d ln(p1 p2) / d margin = 1 - 2 p1``, which tends to -1 only as ``p1 -> 1``. At a
margin of 1.312 nats (D74's median) the two-state value of ``p1`` is 0.788 and the null slope is **-0.576**, not -1.
Testing against -1 in a thin-margin regime manufactures a violation, so ``identity_slope`` reports the null evaluated
at the data's own margins rather than its asymptote.

**Where the two-state assumption is validated and where it is not.** Against the recorded ``top1_prob`` of the D63
ladder, ``d ln(1 - p1) / d margin`` is **-0.994** [-1.029, -0.958] for Qwen at margins 15.25-25.25 against a two-state
prediction of -1.000, and the true non-top mass is 1.34x the runner-up alone: the form is accurate where margins are
deep. For Llama at margins 0.38-10.38 it is **-0.774** [-0.823, -0.725] against a two-state prediction of -0.925, and
the non-top mass is **2.30x** the runner-up: in the thin-margin regime the tail matters and the two-state form is
measurably optimistic. So a *calibrated* null belongs beside the idealised one whenever margins are thin, which is
what ``calibrated_slope_null`` is for, and a per-sample ``p1`` recorded by the arm itself would remove the assumption
entirely. No arm records it yet.
"""

from __future__ import annotations

import numpy as np
from scipy import special, stats

# d ln(1 - p1) / d margin fitted on the recorded top1_prob of `outputs/reports/answer_confidence.json`, per family.
# These calibrate the null slope in the thin-margin regime where the two-state form is measurably optimistic; the
# intercept is carried so p1 itself can be reconstructed. R^2 is 0.962 (Llama, n=40) and 0.988 (Qwen, n=40).
TAIL_FIT = {"llama": (-0.283, -0.774), "qwen": (-0.017, -0.994)}


def two_state_probs(margin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(p1, p2)`` under the two-effective-state reading of a margin in nats, ``p2 / p1 = exp(-margin)``."""
    margin = np.asarray(margin, dtype=float)
    return special.expit(margin), special.expit(-margin)


def log_curvature(margin: np.ndarray) -> np.ndarray:
    """``ln(p1 p2)``, the softmax's own amplification at this margin -- the term a raw KL comparison confounds.

    Computed through ``log_expit`` rather than ``log(expit(...))`` because at the margins this project measures on the
    probe (17-19 nats) ``expit(-margin)`` underflows the float64 subnormal range and the log would be ``-inf``.
    """
    margin = np.asarray(margin, dtype=float)
    return special.log_expit(margin) + special.log_expit(-margin)


def log_curvature_observed(top1_prob: np.ndarray, nontop_mass: np.ndarray) -> np.ndarray:
    """``ln(p1 (1 - p1))`` from an arm's recorded fields -- the assumption-free form, and the more correct one.

    Two improvements over ``log_curvature`` at once. It needs no two-state reading of the margin, and the scale it uses
    is the **total** non-top mass rather than the runner-up alone, which is what a perturbation spread over the whole
    vocabulary actually sees: ``Var_p(delta)`` for a peaked ``p`` goes as ``p1 * (1 - p1)``, and the runner-up is only
    part of ``1 - p1`` (measured at 1/1.34 of it for Qwen and 1/2.30 for Llama).

    ``nontop_mass`` is taken as recorded rather than recomputed as ``1 - top1_prob``, because the arms compute it with
    ``-expm1`` on the logprob and that is the only way it survives a top-1 probability of 1.0 in float64.
    """
    p1 = np.asarray(top1_prob, dtype=float)
    tail = np.asarray(nontop_mass, dtype=float)
    return np.log(p1) + np.log(tail)


def effective_perturbation(margin: np.ndarray, kl: np.ndarray) -> np.ndarray:
    """``delta`` in logit units: how far the fault moved the scores, with the softmax's amplification divided out.

    This is the comparable quantity. A KL of 5.4e-10 at 17.1 nats of margin and one of 1.1e-05 at 6.5 nats differ by
    four orders of magnitude and correspond to **almost the same** ``delta`` (0.25 against 0.19), because nearly all of
    that gap is the curvature term.
    """
    return np.sqrt(2.0 * np.asarray(kl, dtype=float) / np.exp(log_curvature(margin)))


def calibrated_slope_null(margin: float, family: str) -> float | None:
    """The null slope of ``ln KL`` against margin using a recorded-``p1`` fit instead of the two-state idealisation.

    With ``1 - p1 = exp(a + b m)``, ``d ln(p1 p2) / dm = b (2 p1 - 1) / p1``. Returns None for a family with no fit.
    """
    fit = next((v for k, v in TAIL_FIT.items() if family.lower().startswith(k)), None)
    if fit is None:
        return None
    a, b = fit
    p1 = 1.0 - float(np.exp(a + b * margin))
    if p1 <= 0.5:
        return None  # below the two-state crossover the fit is extrapolating past where p1 is even the top token
    return float(b * (2.0 * p1 - 1.0) / p1)


def _ci(fit) -> tuple[float, float]:
    """A 95% interval on a regression slope."""
    return float(fit.slope - 1.96 * fit.stderr), float(fit.slope + 1.96 * fit.stderr)


def identity_slope(margin: np.ndarray, kl: np.ndarray, family: str | None = None) -> dict:
    """Test the identity two ways and report the nulls it must be judged against.

    * ``slope_vs_curvature`` regresses ``ln KL`` on ``ln(p1 p2)``; the identity predicts **1.0** and this is the
      version that does not depend on where the margins sit.
    * ``slope_vs_margin`` is the same fit against the raw margin, reported because it is what a reader computes first.
      Its null is ``mean(1 - 2 p1)`` over the data's own margins -- ``null_two_state`` -- and **not** -1, which is only
      the deep-margin limit and is carried as ``null_deep_limit`` so the difference is visible.
    * ``residual_rho`` is the honest detrended residual: ``ln KL - ln(p1 p2)`` is ``2 ln delta``, so its correlation
      with the margin asks whether the fault's actual size varies with the margin. Subtracting the margin instead of
      ``ln(p1 p2)`` under-corrects wherever ``p1 < 1`` and leaves a positive trend by construction.
    """
    margin = np.asarray(margin, dtype=float)
    kl = np.asarray(kl, dtype=float)
    keep = np.isfinite(margin) & np.isfinite(kl) & (kl > 0)
    margin, kl = margin[keep], kl[keep]
    lc, lk = log_curvature(margin), np.log(kl)
    p1, _ = two_state_probs(margin)

    fc = stats.linregress(lc, lk)
    fm = stats.linregress(margin, lk)
    two_state = float(np.mean(1.0 - 2.0 * p1))
    calibrated = calibrated_slope_null(float(np.median(margin)), family) if family else None
    resid = stats.spearmanr(margin, lk - lc)
    cc, cm = _ci(fc), _ci(fm)
    return {
        "n": int(keep.sum()), "margin_median": float(np.median(margin)),
        "slope_vs_curvature": float(fc.slope), "slope_vs_curvature_ci": cc, "r2_vs_curvature": float(fc.rvalue ** 2),
        "identity_holds": bool(cc[0] <= 1.0 <= cc[1]),
        "slope_vs_margin": float(fm.slope), "slope_vs_margin_ci": cm, "r2_vs_margin": float(fm.rvalue ** 2),
        "null_deep_limit": -1.0, "null_two_state": two_state, "null_calibrated": calibrated,
        "excludes_deep_limit": bool(not cm[0] <= -1.0 <= cm[1]),
        "contains_two_state": bool(cm[0] <= two_state <= cm[1]),
        "contains_calibrated": bool(calibrated is not None and cm[0] <= calibrated <= cm[1]),
        "residual_rho": float(resid.statistic), "residual_p": float(resid.pvalue),
        "delta_median": float(np.median(effective_perturbation(margin, kl))),
    }
