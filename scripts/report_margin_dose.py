"""Is the answer margin the mechanism of fault fragility, or a correlate? D62's own stated limitation.

D62 established the cross-family half decisively — Qwen3-1.7B holds its answers at **17.07 nats** against
Llama-3.2-3B's **3.27**, p = 1.2e-06 on 31 of 31 samples — and then said plainly what it could not establish. If a
thin margin is the *mechanism*, then **within** a family the thin-margin samples should break more. They do, by 1.98×
(Qwen) and 1.88× (Llama), and neither correlation approaches significance at n = 31.

The obstacle is arithmetic, not design. Break rates run 2–6%, so a per-sample rate over 16 trials is a coarse number,
and unlike the KL, `margin_clean` is **one number per sample** and cannot be enriched by adding trials. So the fix is
samples and only samples: D71 runs 200 against D62's 40, everything else identical.

    python scripts/report_margin_dose.py --arm d71_margin_dose_qwen

**Pre-registered in D71 before the run**, because a dose-response invites choosing a threshold after looking:

* **PRIMARY** — Spearman(`margin_clean`, per-sample break rate) over every sample, with a bootstrap CI. D31 applies:
  the result needs p < 0.05 **and** an interval excluding zero.
* **SECONDARY** — the thin/thick median split D62 reported, kept so the two records can be read against each other
  and not because a median split is the better test of a monotone relationship.

Both outcomes are worth having. A resolved negative correlation promotes margin from correlate to mechanism; a null at
this size bounds how large the within-family effect can be, which is what D62 could not do.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from scipy import stats

from raq_kv.analysis.arms import _trials, load_arm
from raq_kv.analysis.contrasts import contrast_rng
from raq_kv.utils.paths import resolve_output_root


def per_sample(data: dict[str, dict], field: str) -> tuple[np.ndarray, np.ndarray]:
    """``(margin, break rate)`` per sample, over the samples the arm's own golden answers correctly.

    The margin is read from the first trial because it is a property of the *fault-free* distribution and is
    therefore identical across a sample's trials; the break rate is the fraction of that sample's trials whose
    answer changed.
    """
    margins, rates = [], []
    for record in data.values():
        if not record.get("golden_correct"):
            continue
        trials = _trials(record)
        if not trials or "margin_clean" not in trials[0]:
            continue
        margins.append(float(trials[0]["margin_clean"]))
        rates.append(sum(1 for t in trials if t["answer_changed"]) / len(trials))
    return np.array(margins), np.array(rates)


def bootstrap_spearman(x: np.ndarray, y: np.ndarray, seed: int, n: int = 10_000) -> tuple[float, float]:
    rng = contrast_rng("margin", "break_rate", seed)
    out = []
    for _ in range(n):
        pick = rng.integers(0, len(x), len(x))
        # A resample can be constant in either variable, which makes rho undefined rather than zero.
        if len(set(x[pick].tolist())) < 2 or len(set(y[pick].tolist())) < 2:
            continue
        out.append(stats.spearmanr(x[pick], y[pick]).statistic)
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", default="d71_margin_dose_qwen")
    parser.add_argument("--compare", default="d62_qwen_meta", help="the smaller arm this one supersedes in power")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    raw_root = resolve_output_root(args.output_root) / "raw"

    out: dict = {}
    print("margin against per-sample break rate -- is the margin the mechanism?\n")
    print(f"{'arm':24s}{'n':>5s}{'margin median':>15s}{'break rate':>12s}{'Spearman':>11s}{'95% CI':>22s}{'p':>10s}")
    for name in (args.arm, args.compare):
        try:
            data = load_arm(raw_root, name)
        except SystemExit:
            print(f"{name:24s}  not found or incomplete")
            continue
        x, y = per_sample(data, "answer_changed")
        if len(x) < 10:
            print(f"{name:24s}  only {len(x)} usable samples")
            continue
        rho = float(stats.spearmanr(x, y).statistic)
        pval = float(stats.spearmanr(x, y).pvalue)
        lo, hi = bootstrap_spearman(x, y, args.seed)
        meets = pval < 0.05 and lo * hi > 0
        print(f"{name:24s}{len(x):5d}{np.median(x):15.3f}{y.mean():12.4f}{rho:+11.3f}"
              f"   [{lo:+.3f}, {hi:+.3f}]{pval:10.4f}  {'MEETS D31' if meets else 'fails D31'}")
        # SECONDARY: D62's median split, reported so the two records are comparable.
        thin = x <= np.median(x)
        split = {"thin_rate": float(y[thin].mean()), "thick_rate": float(y[~thin].mean()),
                 "ratio": float(y[thin].mean() / y[~thin].mean()) if y[~thin].mean() else None}
        print(f"{'':24s}  secondary, median split: thin {split['thin_rate']:.4f} against thick "
              f"{split['thick_rate']:.4f}"
              + (f", ratio {split['ratio']:.2f}x" if split["ratio"] else ""))
        out[name] = {"n": len(x), "spearman": rho, "ci": (lo, hi), "p": pval, "meets_d31": meets,
                     "margin_median": float(np.median(x)), "break_rate": float(y.mean()), "median_split": split}

    if args.arm in out and args.compare in out:
        a, b = out[args.arm], out[args.compare]
        print(f"\n  {args.arm} carries {a['n'] / b['n']:.1f}x the samples of {args.compare} at the same 16 trials "
              f"each,\n  so it is a strict superset in power rather than a different experiment. The interval is what "
              f"to read:\n  a resolved negative rho promotes margin from correlate to mechanism, and a null bounds "
              f"how large the\n  within-family effect can be -- which is the thing D62 could not do at n = 31.")

    dest = resolve_output_root(args.output_root) / "reports" / "margin_dose.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
