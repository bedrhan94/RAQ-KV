"""The escape D71 and D74 both left open: the graded KL as the outcome instead of the coarse break rate.

D71 and D74 bounded the margin's sample-level effect and both ended by naming the same limit. D71: *"a design
measuring fragility per sample more finely -- more trials, or the graded KL as the outcome instead of the rate -- could
resolve a smaller effect. The KL route is the obvious one and is not run here, because D62 already established that
Qwen's KL is saturation-limited on this probe and would be measuring the floor."* D74 repeated it and noted the
objection no longer applies on its vehicle.

**Why the outcome change is the whole point.** A per-sample break rate over 16 trials can take 17 values and in
practice takes six: D74's 80 samples are 51 at exactly zero, then 0.0625, 0.125, 0.1875, 0.25, 0.5625 -- **75 of 80
tied**. A rank correlation against an outcome that is 64% one tied value is blunt, and both records said so. The
per-sample median KL is continuous, unties every sample, and uses all 16 trials rather than only those that crossed
the answer boundary.

    python scripts/report_margin_kl.py --arm d74_margin_curve_llama_lb --condition stable

**Pre-registered in D75 before the run, direction included.** PRIMARY: Spearman(`margin_clean`, per-sample median
`kl_first_token`), D31 applies. **Predicted NEGATIVE** -- D62's saturation account stated per sample.
SECONDARY: the mean and max in place of the median, reported whatever the primary does.

**And the primary was the wrong test, which is D75's actual finding.** The rank correlation came back negative and
met D31 on all three arms, and it means nothing: for a small logit perturbation `ln KL = ln(p1 p2) + 2 ln delta`, so
a fault of *fixed* size already produces a strong negative correlation with the margin through nothing but the shape
of the softmax. The two CONTROL blocks below are what decide the question, and the null they test against is
`slope = 1.0` against `ln(p1 p2)` -- **not** `slope = -1` against the margin, which holds only as `p1 -> 1` and
manufactures a violation wherever margins are thin. See `raq_kv.analysis.curvature`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np
from scipy import stats

from raq_kv.analysis.arms import _trials, load_arm
from raq_kv.analysis.contrasts import contrast_rng
from raq_kv.analysis.curvature import identity_slope
from raq_kv.utils.paths import resolve_output_root


def per_sample(data: dict[str, dict], condition: str, stat: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """``(margin, KL statistic, diagnostics)`` per sample, one row per sample rather than per trial.

    **Non-finite trials are kept as ``+inf`` for the median and the max, and dropped for the mean.** A corrupted fp16
    metadata word makes the dequantized cache infinite and the forward pass collapses (D59), which is the *largest*
    perturbation the measure can report, not a missing one. For a rank statistic ``inf`` is simply the top rank, so a
    median over 16 values of which one or two are infinite is still a finite number with the ordering intact --
    dropping those trials would pull the median of the most-damaged samples *down* and bias the test toward exactly
    the null it is looking for. The mean cannot represent them at all, so it is computed on the finite trials and the
    count is reported beside it rather than being silently absorbed.
    """
    margins, values = [], []
    nonfinite_trials = nonfinite_samples = 0
    for record in data.values():
        if condition == "correct" and not record.get("golden_correct"):
            continue
        if condition == "stable" and any(t["answer_changed"] for t in record["faults"] if t["trial"] < 0):
            continue
        trials = _trials(record)
        if not trials or "margin_clean" not in trials[0]:
            continue
        kl = np.array([float(t["kl_first_token"]) for t in trials])
        bad = ~np.isfinite(kl)
        if bad.any():
            nonfinite_trials += int(bad.sum())
            nonfinite_samples += 1
        kl = np.where(bad, np.inf, kl)
        if stat == "median":
            value = float(np.median(kl))
        elif stat == "max":
            value = float(kl.max())
        else:
            finite = kl[np.isfinite(kl)]
            if not finite.size:
                continue
            value = float(finite.mean())
        # A median that lands on an infinite trial means over half this sample's trials collapsed; it has no finite
        # value and cannot be ranked against the others, so it is reported and excluded rather than clipped.
        if not math.isfinite(value) and stat != "max":
            continue
        margins.append(float(trials[0]["margin_clean"]))
        values.append(value)
    diag = {"nonfinite_trials": nonfinite_trials, "nonfinite_samples": nonfinite_samples}
    return np.array(margins), np.array(values), diag


def bootstrap_spearman(x: np.ndarray, y: np.ndarray, seed: int, n: int = 10_000) -> tuple[float, float]:
    rng = contrast_rng("margin", "kl", seed)
    out = []
    for _ in range(n):
        pick = rng.integers(0, len(x), len(x))
        if len(set(x[pick].tolist())) < 2 or len(set(y[pick].tolist())) < 2:
            continue
        out.append(stats.spearmanr(x[pick], y[pick]).statistic)
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", default="d74_margin_curve_llama_lb")
    parser.add_argument("--arms", help="comma-separated; overrides --arm")
    parser.add_argument("--condition", default="stable", choices=("correct", "stable"))
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--family", help="model family, to pick the recorded-p1 calibration for the "
                                         "slope null in a thin-margin regime (llama / qwen)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    raw_root = resolve_output_root(args.output_root) / "raw"
    names = tuple(a.strip() for a in args.arms.split(",")) if args.arms else (args.arm,)

    out: dict = {}
    print("margin against the per-sample KL response -- the outcome D71 and D74 both named and did not run\n")
    print("  PRIMARY is the median; the mean and max are pre-registered secondaries, not alternatives to pick from.")
    print("  Predicted sign is NEGATIVE: under D62's saturation account a thinner margin leaves more mass to move.\n")
    print(f"{'arm':28s}{'stat':>7s}{'n':>5s}{'median KL':>12s}{'Spearman':>10s}{'95% CI':>20s}{'p':>9s}  verdict")
    for name in names:
        try:
            data = load_arm(raw_root, name)
        except SystemExit:
            print(f"{name:28s}  not found or incomplete")
            continue
        for stat in ("median", "mean", "max"):
            x, y, diag = per_sample(data, args.condition, stat)
            if len(x) < 10:
                print(f"{name:28s}{stat:>7s}  only {len(x)} usable samples")
                continue
            res = stats.spearmanr(x, y)
            rho, pval = float(res.statistic), float(res.pvalue)
            lo, hi = bootstrap_spearman(x, y, args.seed)
            meets = pval < 0.05 and lo * hi > 0
            finite_y = y[np.isfinite(y)]
            print(f"{name:28s}{stat:>7s}{len(x):5d}{np.median(finite_y):12.3e}{rho:+10.3f}"
                  f"   [{lo:+.3f}, {hi:+.3f}]{pval:9.4f}  "
                  f"{'MEETS D31' if meets else 'fails D31'}{'' if meets else ' (null)'}")
            out[f"{name}:{stat}"] = {"n": len(x), "spearman": rho, "ci": (lo, hi), "p": pval, "meets_d31": meets,
                                     "kl_median": float(np.median(finite_y)), **diag}
        # The rate is what D71 and D74 actually tested; printed beside the KL so the two outcomes are comparable on
        # the same samples rather than across records.
        rates = []
        for record in data.values():
            if args.condition == "correct" and not record.get("golden_correct"):
                continue
            if args.condition == "stable" and any(t["answer_changed"] for t in record["faults"] if t["trial"] < 0):
                continue
            trials = _trials(record)
            if trials and "margin_clean" in trials[0]:
                rates.append(sum(1 for t in trials if t["answer_changed"]) / len(trials))
        if rates:
            vals, counts = np.unique(np.array(rates), return_counts=True)
            print(f"{'':28s}  the outcome this replaces: {len(vals)} distinct break rates over {len(rates)} samples, "
                  f"{counts.max()} tied at {vals[counts.argmax()]:.4f}")
            print(f"{'':28s}  non-finite KL trials: {diag['nonfinite_trials']} across "
                  f"{diag['nonfinite_samples']} samples, kept as +inf in the median and max")
        # THE CONTROL, and it is what decides the primary. Printed for every arm rather than on request, because a
        # negative rho read without it is not interpretable.
        mx, my, _ = per_sample(data, args.condition, "median")
        if len(mx) >= 10:
            c = identity_slope(mx, my, family=args.family)
            print(f"{'':28s}  CONTROL 1, ln KL against ln(p1 p2) -- identity predicts slope 1.0, and this form does "
                  f"not depend on where the margins sit:")
            print(f"{'':28s}    slope {c['slope_vs_curvature']:+.3f} "
                  f"[{c['slope_vs_curvature_ci'][0]:+.3f}, {c['slope_vs_curvature_ci'][1]:+.3f}], "
                  f"R^2 {c['r2_vs_curvature']:.3f}  -> "
                  f"{'IDENTITY HOLDS' if c['identity_holds'] else 'identity violated'}")
            nulls = (f"deep-limit {c['null_deep_limit']:+.3f}, two-state {c['null_two_state']:+.3f}"
                     + (f", calibrated {c['null_calibrated']:+.3f}" if c["null_calibrated"] is not None else ""))
            print(f"{'':28s}  CONTROL 2, ln KL against margin: slope {c['slope_vs_margin']:+.3f} per nat "
                  f"[{c['slope_vs_margin_ci'][0]:+.3f}, {c['slope_vs_margin_ci'][1]:+.3f}], "
                  f"R^2 {c['r2_vs_margin']:.3f}")
            print(f"{'':28s}    nulls at these margins: {nulls}")
            print(f"{'':28s}    -> excludes the deep limit: {c['excludes_deep_limit']}; "
                  f"contains two-state: {c['contains_two_state']}; contains calibrated: {c['contains_calibrated']}")
            print(f"{'':28s}  residual, curvature divided out properly (ln KL - ln(p1 p2) = 2 ln delta): "
                  f"rho = {c['residual_rho']:+.3f}, p = {c['residual_p']:.4f}")
            print(f"{'':28s}  effective logit perturbation delta: median {c['delta_median']:.4f}")
            out[f"{name}:curvature"] = c

    dest = resolve_output_root(args.output_root) / "reports" / f"margin_kl_{args.condition}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
