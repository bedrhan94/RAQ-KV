"""P(break | which quantization group was corrupted), per model family -- D61's canonical report.

Answers what D60 left open. D59 measured a per-metadata-flip hazard of 1.17% (Qwen3-1.7B) against 2.63%
(Llama-3.2-3B) on unpruned caches and could not say why. D60 ruled out dynamic range, then ruled out attention
spread with the sign reversed -- Llama's attention is *more* concentrated. What survived is that concentration
raises the **conditional severity** of the flips that land somewhere consequential, and separating that from "more
groups are load-bearing" needs the corrupted group chosen rather than drawn.

**The contrasts printed here are the ones D61's amendment declared, not the ones its first draft did.** The
amendment exists because the most-attended group is group 0 in 1.000 of 80 measured samples with zero position
variance: a rank-0 cell is a *position* cell, so no within-family reading of it can be called attention-mediated.
What is identified:

1. P(break | group 0), **Llama against Qwen** -- the same position in both families, so the family difference is
   not positional. Branch A's prediction, and D60's arithmetic put it at about 41.8% against 18.6%.
2. P(break | least-attended group), **Llama against Qwen** -- where branch B puts the family difference instead.
3. P(break | rank 1) against P(break | tail), **within** each family -- shares separate completely (median 7.24x
   Qwen, 5.51x Llama) while position distributions overlap. Tests whether attention mass predicts consequence at
   all among non-first groups. Weaker by construction, and the residual asymmetry is printed with it.

Every contrast is paired at the **sample** level over the samples all six arms answer correctly with no faults,
one set for all three so they are comparable, and carries D31's rule: p < 0.05 **and** a bootstrap CI excluding
zero. Cells at zero are reported as a **bound** (D30/D44), never as equality.

    python scripts/report_d61.py

The identifiability evidence is re-derived from the arms themselves rather than trusted from the separate
measurement, because the arms are what the claim rests on.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from scipy import stats

from raq_kv.analysis.arms import _controls_clean, _trials, load_arm, rates, shared_samples
from raq_kv.analysis.contrasts import contrast
from raq_kv.analysis.stats import wilson
from raq_kv.utils.paths import resolve_output_root

CELLS = ("top", "rank1", "tail")
FAMILIES = ("qwen", "llama")


def pooled(data: dict[str, dict], samples: list[str]) -> tuple[int, int]:
    """Trials that changed the answer, out of trials attempted, over the shared samples.

    Pooled only for the interval on the cell itself -- every *contrast* uses the per-sample rates. A Wilson
    interval needs a k and an n, and the per-sample rates have no denominator; reporting both is how the cell's
    precision and the contrast's pairing coexist without one being used for the other.
    """
    k = n = 0
    for s in samples:
        trials = _trials(data[s])
        n += len(trials)
        k += sum(1 for t in trials if t["answer_changed"])
    return k, n


def targeting(data: dict[str, dict], samples: list[str]) -> dict[str, float]:
    """What was actually corrupted, re-derived from the arm's own records.

    The point is that the report does not have to be believed about which group it hit: `target_group_index`,
    `target_group_rank` and `target_group_share` are per-sample fields written by the arm, and a cell whose
    requested rank wrapped shows up here as a realised rank below the request.
    """
    idx = np.array([data[s]["target_group_index"] for s in samples], dtype=float)
    n_groups = np.array([data[s]["n_token_groups"] for s in samples], dtype=float)
    last = np.maximum(n_groups - 1.0, 1.0)
    return {
        "share": float(np.mean([data[s]["target_group_share"] for s in samples])),
        "share_min": float(np.min([data[s]["target_group_share"] for s in samples])),
        "share_max": float(np.max([data[s]["target_group_share"] for s in samples])),
        "is_first_group": float(np.mean(idx == 0)),
        "is_last_group": float(np.mean(idx == n_groups - 1)),
        "position": float(np.mean(idx / last)),
        "position_sd": float(np.std(idx / last)),
        "realised_rank": float(np.mean([data[s]["target_group_rank"] for s in samples])),
        "requested_rank": float(np.mean([data[s]["target_group_rank_requested"] for s in samples])),
        "n_groups": float(n_groups.mean()),
    }


def rule_of_three(k: int, n: int) -> str:
    """A zero cell is a bound, not a zero (D30/D44). At n trials with no events, the 95% bound is 3/n."""
    return f"0 of {n}: rate <= {3.0 / n:.4%} (rule of three), a BOUND not a zero" if k == 0 and n else ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    raw_root = resolve_output_root(args.output_root) / "raw"
    arms = {f"{fam}_{cell}": load_arm(raw_root, f"d61_{fam}_{cell}")
            for fam in FAMILIES for cell in CELLS}
    samples = shared_samples(arms)
    if not samples:
        raise SystemExit("no sample is answered correctly with no faults by all six arms")

    print(f"D61: P(break | corrupted group), {len(samples)} samples shared by all six arms\n")
    for name, data in arms.items():
        changed, total = _controls_clean(data)
        if changed:
            print(f"  WARNING {name}: {changed} of {total} no-fault controls changed the answer -- the arm's own "
                  f"reference is unstable and every rate below is measured against a moving baseline")

    # What was corrupted, before what it cost. A reader who does not accept the targeting has no reason to read on.
    print("\nwhat each cell actually corrupted (re-derived from the arms' own records)\n")
    print(f"{'arm':14s}{'rank req':>9s}{'realised':>9s}{'share':>8s}{'[min':>8s}{'max]':>8s}"
          f"{'is grp 0':>10s}{'is last':>9s}{'rel.pos':>9s}{'sd':>7s}")
    tgt = {}
    for name, data in arms.items():
        t = tgt[name] = targeting(data, samples)
        print(f"{name:14s}{t['requested_rank']:9.0f}{t['realised_rank']:9.1f}{t['share']:8.3f}"
              f"{t['share_min']:8.3f}{t['share_max']:8.3f}{t['is_first_group']:10.3f}"
              f"{t['is_last_group']:9.3f}{t['position']:9.3f}{t['position_sd']:7.3f}")
    for fam in FAMILIES:
        t = tgt[f"{fam}_top"]
        if t["is_first_group"] < 1.0 or t["position_sd"] > 0.0:
            print(f"  NOTE {fam}_top: the most-attended group was NOT always group 0 here "
                  f"({t['is_first_group']:.3f}), which weakens the amendment's premise rather than the result")

    print("\nbreak rate per cell (pooled trials, Wilson 95%; contrasts use per-sample rates)\n")
    print(f"{'arm':14s}{'k':>6s}{'n':>7s}{'rate':>9s}{'95% CI':>20s}")
    per_sample = {}
    for name, data in arms.items():
        k, n = pooled(data, samples)
        lo, hi = wilson(k, n)
        per_sample[name] = rates(data, samples, "answer_changed")
        print(f"{name:14s}{k:6d}{n:7d}{k / n:9.4%}   [{lo:.4%}, {hi:.4%}]")
        bound = rule_of_three(k, n)
        if bound:
            print(f"{'':14s}  {bound}")

    def show(title: str, x: str, y: str, note: str = "") -> dict:
        c = contrast(x, y, per_sample[x], per_sample[y], args.seed)
        verdict = "MEETS D31" if c["meets_rule"] else "does NOT meet D31"
        print(f"\n{title}\n  {x} {c['x_mean']:.4%} -> {y} {c['y_mean']:.4%}, difference {c['difference']:+.4%} "
              f"[{c['ci'][0]:+.4%}, {c['ci'][1]:+.4%}], p = {c['p']:.3g}, {c['discordant']}/{c['n_pairs']} "
              f"samples moved\n  {verdict} (p < 0.05 AND a CI excluding zero)")
        if note:
            print(f"  {note}")
        return c

    print("\n" + "=" * 100)
    out = {"n_samples": len(samples), "samples": samples, "targeting": tgt,
           "cells": {n: dict(zip(("k", "n"), pooled(d, samples), strict=True)) for n, d in arms.items()},
           "contrasts": {}}

    out["contrasts"]["primary1_top_across_families"] = show(
        "PRIMARY 1 -- P(break | group 0), across families. Identified: the same position in both.",
        "qwen_top", "llama_top",
        "branch A predicts Llama above Qwen; D60's arithmetic put it at about 18.6% -> 41.8%")
    out["contrasts"]["primary2_tail_across_families"] = show(
        "PRIMARY 2 -- P(break | least-attended group), across families. Same cell, same rule, both families.",
        "qwen_tail", "llama_tail",
        "branch B predicts Llama above Qwen HERE rather than at primary 1")
    # Primary 3 is WITHIN a family, so it belongs on that family's own three-arm intersection. Restricting it to
    # the six-way set requires the OTHER family's arms to answer a sample cleanly for it to count -- a stricter
    # condition than any other contrast in this project imposes, buying nothing. Both are printed because D61's
    # record first reported the six-way numbers and on Qwen the magnitude differs 3.8x: the six-way set made the
    # contrast look stronger than that family's own data supports.
    for fam in FAMILIES:
        own = shared_samples({k: v for k, v in arms.items() if k.startswith(fam)})
        c_own = contrast(f"{fam}_tail", f"{fam}_rank1",
                         rates(arms[f"{fam}_tail"], own, "answer_changed"),
                         rates(arms[f"{fam}_rank1"], own, "answer_changed"), args.seed)
        print(f"\nPRIMARY 3 ({fam}) -- P(break | rank 1) against P(break | tail), within the family."
              f"\n  CANONICAL, on this family's own {len(own)}-sample intersection: "
              f"tail {c_own['x_mean']:.4%} -> rank 1 {c_own['y_mean']:.4%}, difference "
              f"{c_own['difference']:+.4%} [{c_own['ci'][0]:+.4%}, {c_own['ci'][1]:+.4%}], p = {c_own['p']:.3g}, "
              f"{c_own['discordant']}/{c_own['n_pairs']} samples moved"
              f"\n  {'MEETS D31' if c_own['meets_rule'] else 'does NOT meet D31'}")
        out["contrasts"][f"primary3_{fam}_rank1_vs_tail"] = c_own
        out["contrasts"][f"primary3_{fam}_rank1_vs_tail_sixway"] = show(
            f"  the same contrast on the six-way {len(samples)}-sample set, for comparability:",
            f"{fam}_tail", f"{fam}_rank1",
            "share-separated (no overlap) but NOT position-balanced: rank 1 sits earlier than the tail in 60% of "
            "Qwen and 82% of Llama samples, so a positional reading is not excluded here")

    # Added in review AFTER the four pre-registered contrasts and labelled as such. Each per-cell contrast fails,
    # but the six cells order inversely to attention in both families, and "does that ordering itself meet a rule"
    # is a question the pre-registration did not ask. Reported so the direction carries a number rather than an
    # adjective -- not as a replacement for anything above.
    print("\n" + "=" * 100)
    print("\nPOST-HOC (not pre-registered) -- does the inverse ordering itself meet a rule?\n")
    rho, p_rho = stats.spearmanr(np.array([tgt[n]["share"] for n in arms]),
                                 np.array([per_sample[n].mean() for n in arms]))
    print(f"  Spearman(attention share, break rate) over the 6 cells: rho = {rho:+.3f}, p = {p_rho:.4f}")
    c_pool = contrast("top_cells", "non_top_cells",
                      np.concatenate([per_sample[f"{f}_top"] for f in FAMILIES]),
                      np.concatenate([(per_sample[f"{f}_rank1"] + per_sample[f"{f}_tail"]) / 2
                                      for f in FAMILIES]), args.seed)
    print(f"  pooled top against non-top, {c_pool['n_pairs']} pairs with the families stacked: "
          f"{c_pool['x_mean']:.4%} -> {c_pool['y_mean']:.4%}, difference {c_pool['difference']:+.4%} "
          f"[{c_pool['ci'][0]:+.4%}, {c_pool['ci'][1]:+.4%}], p = {c_pool['p']:.4f}\n  "
          f"{'MEETS D31' if c_pool['meets_rule'] else 'does NOT meet D31'} -- the inversion stays a direction")
    out["post_hoc"] = {"spearman_rho": float(rho), "spearman_p": float(p_rho), "pooled_top_vs_non_top": c_pool}

    # Descriptive only, and labelled so. Position is confounded with share across cells, which is the whole reason
    # primaries 1 and 2 are cross-family; this curve is for shape, not for a verdict.
    print("\n" + "=" * 100)
    print("\nDESCRIPTIVE -- break rate against the corrupted group's attention share (no verdict attaches)\n")
    print(f"{'arm':14s}{'mean share':>12s}{'break rate':>12s}")
    for name in arms:
        print(f"{name:14s}{tgt[name]['share']:12.4f}{per_sample[name].mean():12.4%}")

    dest = resolve_output_root(args.output_root) / "reports" / "d61_group_conditional.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
