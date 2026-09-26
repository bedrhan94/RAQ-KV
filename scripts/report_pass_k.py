"""Pass^k: does the arm succeed on **all** k independent fault realisations? M6's reliability metric.

The research plan (§M6) asks for "**Pass^k**: bağımsız k fault gerçekleşmesinin hepsinde başarı (güvenilirlik odaklı
tanım)" and it exists nowhere in the repo. It needs no GPU: every protection arm already draws `n_trials`
independent fault plans per sample, which is exactly the k realisations the metric is about.

**Why this is worth having rather than a restatement of Pass@1.** Pass@1 is a rate over trials and says how often a
fault costs an answer. Pass^k asks whether a *deployment* gets through k faults untouched, which is the question a
reliability budget actually poses, and it is sensitive to something Pass@1 is blind to: **whether failures cluster**.
If every sample is equally fragile, Pass^k decays geometrically as Pass^1 to the k. If a few samples carry all the
damage, Pass^k drops fast and then plateaus at the share of samples that never fail. The gap between the measured
curve and the geometric prediction is therefore a direct measure of clustering, and this project has been bitten by
clustering before -- D26 withdrew a 1.73x ratio that was a pooled-trial artefact.

**Estimator.** For a sample with `s` successes out of `n` trials, the probability that k trials drawn without
replacement all succeed is `C(s, k) / C(n, k)`; Pass^k is that averaged over samples. Using the combinatorial form
rather than "were the first k trials clean" uses all n trials instead of discarding n - k of them, and it is the same
device the code-generation pass@k estimator uses, applied to the conjunction instead of the disjunction.

    python scripts/report_pass_k.py --arms m3_arm_a_unprot,m3_arm_g_meta_only

Two outcomes are reported per arm because this project distinguishes them everywhere: `task_failure` (the answer is
wrong) and `answer_changed` (the answer moved at all, a strictly more sensitive proxy).
"""

from __future__ import annotations

import argparse
import json
import sys
from math import comb

import numpy as np

from raq_kv.analysis.arms import _controls_clean, _trials, load_arm
from raq_kv.analysis.stats import wilson
from raq_kv.utils.paths import resolve_output_root


def per_sample_successes(data: dict[str, dict], field: str) -> list[tuple[int, int]]:
    """``(successes, trials)`` per sample, over the samples the arm's own golden answers correctly.

    An arm cannot be charged for a sample its unfaulted cache already gets wrong, so those are excluded here for
    the same reason every other report in this project excludes them.
    """
    out = []
    for record in data.values():
        if not record.get("golden_correct"):
            continue
        trials = _trials(record)
        if not trials:
            continue
        if field == "task_failure":
            good = sum(1 for t in trials if t.get("faulty_correct"))
        else:
            good = sum(1 for t in trials if not t["answer_changed"])
        out.append((good, len(trials)))
    return out


def pass_hat_k(counts: list[tuple[int, int]], k: int) -> float | None:
    """Unbiased Pass^k over the samples with at least k trials; None when no sample has that many."""
    vals = [comb(s, k) / comb(n, k) for s, n in counts if n >= k]
    return float(np.mean(vals)) if vals else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", required=True, help="Comma-separated config names")
    parser.add_argument("--field", default="both", choices=("task_failure", "answer_changed", "both"))
    parser.add_argument("--output-root", default="outputs")
    args = parser.parse_args()

    raw_root = resolve_output_root(args.output_root) / "raw"
    fields = ("task_failure", "answer_changed") if args.field == "both" else (args.field,)
    out: dict = {}

    for name in (a.strip() for a in args.arms.split(",") if a.strip()):
        data = load_arm(raw_root, name)
        changed, total = _controls_clean(data)
        print(f"\n{'=' * 96}\n{name}   ({len(data)} samples; {changed}/{total} no-fault controls changed)\n{'=' * 96}")
        for field in fields:
            counts = per_sample_successes(data, field)
            if not counts:
                print(f"  {field}: no usable sample")
                continue
            n_max = max(n for _, n in counts)
            p1 = pass_hat_k(counts, 1) or 0.0
            never = float(np.mean([s == n for s, n in counts]))
            always = float(np.mean([s == 0 for s, n in counts]))
            print(f"\n  {field}  ({len(counts)} samples, up to {n_max} trials each)")
            print(f"    {'k':>3s}{'Pass^k':>10s}{'geometric Pass^1^k':>21s}{'excess':>10s}")
            curve = {}
            for k in sorted({1, 2, 4, 8, n_max} & set(range(1, n_max + 1))):
                pk = pass_hat_k(counts, k)
                if pk is None:
                    continue
                geo = p1 ** k
                curve[k] = pk
                print(f"    {k:3d}{pk:10.4f}{geo:21.4f}{pk - geo:+10.4f}")
            lo, hi = wilson(int(round(never * len(counts))), len(counts))
            print(f"    samples that NEVER fail: {never:.4f} [{lo:.4f}, {hi:.4f}]  |  "
                  f"samples that ALWAYS fail: {always:.4f}")
            # The clustering diagnostic is the EXCESS above the geometric curve, not Pass^n against `never`.
            # Those two are the same number by construction -- C(s, n) / C(n, n) is 1 exactly when s = n, so
            # Pass^n IS the never-fails share -- and the first version of this script compared them and therefore
            # always printed "clustered", including for an arm with zero failures where clustering is undefined.
            if p1 >= 1.0:
                print("    no failures at all: Pass^k = 1 at every k, and clustering is undefined here")
            else:
                mid = max((k for k in curve if 1 < k < n_max), default=None)
                excess = curve[mid] - p1 ** mid if mid else curve[n_max] - p1 ** n_max
                at = mid or n_max
                verdict = ("CLUSTERED -- failures repeat on the same samples" if excess > 0.02 else
                           "roughly INDEPENDENT -- the curve tracks the geometric prediction")
                print(f"    excess over geometric at k = {at}: {excess:+.4f}  ->  {verdict}")
            out.setdefault(name, {})[field] = {
                "n_samples": len(counts), "pass_1": p1, "curve": curve,
                "never_fails": never, "always_fails": always,
                "geometric_excess": {k: v - p1 ** k for k, v in curve.items()}}

    print("\n  Excess above the geometric curve is clustering: a positive excess means failures repeat on the same"
          "\n  samples rather than striking independently, so a deployment's risk is concentrated in which inputs it"
          "\n  sees rather than spread over how many faults it takes.")
    dest = resolve_output_root(args.output_root) / "reports" / "pass_k.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
