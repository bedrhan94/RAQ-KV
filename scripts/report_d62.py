"""Is Llama's higher per-metadata-flip hazard per-flip severity, or a thinner margin? -- D62's canonical report.

D59 measured a per-metadata-flip hazard of 2.63% (Llama-3.2-3B) against 1.17% (Qwen3-1.7B) on unpruned caches, a
**2.25x** ratio, and three mechanisms have since been eliminated without answering it: dynamic range (D59),
attention spread (D60, refuted with the sign reversed) and attention-rank-conditioned severity (D61, no gradient at
all). D61's contrasts were also underpowered for a structural reason -- `answer_changed` runs at 2-6% and no
realistic sample count resolves a 2x difference in a 3% rate.

So this measures the continuous quantity the binary event thresholds: the shift in the distribution over the first
answer token, per flip. Two branches, pre-registered in D62 and separated by *which recorded quantity* differs:

* **A, per-flip severity** -- the same corruption moves Llama's output distribution further. The mechanism is in
  the codec and quantization design is the lever.
* **B, a thinner margin** -- flips move both comparably, but Llama holds its answers with less confidence, so the
  same shift crosses the threshold more often. Then the mechanism has nothing to do with quantization.

    python scripts/report_d62.py

**The anchor runs first and can void the rest.** If KL does not separate trials whose answer changed from trials
whose answer did not, the graded measure is not measuring what breaks answers and neither primary means anything.
That order is deliberate: it is checked before the contrasts it would invalidate are printed.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from scipy import stats

from raq_kv.analysis.arms import _controls_clean, _trials, load_arm, shared_samples
from raq_kv.analysis.contrasts import contrast, contrast_rng
from raq_kv.analysis.stats import wilson
from raq_kv.utils.paths import resolve_output_root

# The probe pair by default; D73 runs the same comparison on LongBench, where Qwen is not saturated. Parameterised
# rather than fixed, because a hardcoded pair is how a report ends up able to describe only the vehicle it was written
# for -- the same repair D70 needed in the failure-mode report.
VEHICLES = {"probe": {"qwen": "d62_qwen_meta", "llama": "d62_llama_meta"},
            "longbench": {"qwen": "d73_lb_fault_qwen", "llama": "d73_lb_fault_llama"}}


def controls_are_exactly_zero(data: dict[str, dict]) -> tuple[int, int, float]:
    """Control trials read the fault-free cache the reference came from, so their KL must be exactly 0.0.

    Not approximately. A nonzero value here means the measurement floor is pipeline nondeterminism of unknown
    size, and every small KL in the trials is noise rather than the effect of a flip. Returned rather than
    asserted so the report can state it as evidence instead of dying on it.
    """
    worst, n, bad = 0.0, 0, 0
    for record in data.values():
        for trial in record["faults"]:
            if trial["trial"] >= 0 or "kl_first_token" not in trial:
                continue
            n += 1
            worst = max(worst, abs(float(trial["kl_first_token"])))
            bad += int(float(trial["kl_first_token"]) != 0.0)
    return bad, n, worst


def flip_kls(data: dict[str, dict], samples: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every flip's KL, flattened, with its trial-level outcome flags beside it. May contain NaN -- see below."""
    kl, changed, flipped = [], [], []
    for s in samples:
        for t in _trials(data[s]):
            kl.append(float(t["kl_first_token"]))
            changed.append(bool(t["answer_changed"]))
            flipped.append(bool(t["argmax_flipped"]))
    return np.array(kl), np.array(changed, dtype=bool), np.array(flipped, dtype=bool)


def nonfinite_mask(kl: np.ndarray) -> np.ndarray:
    """Trials where the model's own forward pass went non-finite -- a result, not missing data.

    One corrupted fp16 metadata word can blow a key up until the attention softmax overflows; the model then emits
    a degenerate repetition instead of a wrong answer. `faulty_nonfinite` records this directly for runs made after
    it was added, and a NaN KL identifies it exactly for the runs made before, which is why this derives the mask
    rather than reading the field: re-running two campaigns to add a boolean that is already recoverable would be
    GPU time spent on nothing.

    These trials are excluded from the KL statistics because a NaN propagates through every median that touches the
    column, and reported as **their own rate** because they are the most severe failure class there is. Excluding
    them makes the anchor conservative: all of them changed the answer and all carry effectively infinite KL, so
    leaving them out can only weaken KL's apparent discrimination, never inflate it.
    """
    return ~np.isfinite(kl)


def per_sample_median_kl(data: dict[str, dict], samples: list[str]) -> np.ndarray:
    """One number per sample, so the contrast is paired at the sample level like every other in this project.

    The median over a sample's 16 flips, not the mean: KL over hundreds of single-bit flips is heavy-tailed, most
    flips move the distribution by ~1e-6 and a few by ~1e-1, and a mean would be one outlier's report.
    """
    return np.array([float(np.nanmedian([float(t["kl_first_token"]) for t in _trials(data[s])]))
                     for s in samples])


def per_sample_clean_margin(data: dict[str, dict], samples: list[str]) -> np.ndarray:
    """The fault-free answer's margin -- branch B's quantity. Identical across a sample's trials by construction."""
    return np.array([float(_trials(data[s])[0]["margin_clean"]) for s in samples])


def auc_with_ci(kl: np.ndarray, positive: np.ndarray, seed: int, n_boot: int = 10_000) -> dict:
    """AUC of KL as a discriminator of the outcome, with a bootstrap CI over flips.

    Mann-Whitney U over n_pos * n_neg gives the AUC exactly, which is the probability that a randomly chosen
    breaking flip carries a higher KL than a randomly chosen non-breaking one. 0.5 is no discrimination and is
    exactly the value that would void the primaries.
    """
    pos, neg = kl[positive], kl[~positive]
    if len(pos) == 0 or len(neg) == 0:
        return {"auc": None, "n_pos": int(len(pos)), "n_neg": int(len(neg)),
                "note": "one class is empty, so KL cannot be scored as a discriminator here"}
    auc = float(stats.mannwhitneyu(pos, neg, alternative="two-sided").statistic / (len(pos) * len(neg)))
    p = float(stats.mannwhitneyu(pos, neg, alternative="two-sided").pvalue)
    rng = contrast_rng("kl", "changed", seed)
    boot = []
    for _ in range(n_boot):
        ip = rng.integers(0, len(pos), len(pos))
        ineg = rng.integers(0, len(neg), len(neg))
        boot.append(stats.mannwhitneyu(pos[ip], neg[ineg]).statistic / (len(pos) * len(neg)))
    lo, hi = (float(v) for v in np.percentile(boot, [2.5, 97.5]))
    return {"auc": auc, "ci": (lo, hi), "p": p, "n_pos": int(len(pos)), "n_neg": int(len(neg)),
            "median_pos": float(np.median(pos)), "median_neg": float(np.median(neg)),
            "discriminates": bool(lo > 0.5)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vehicle", default="probe", choices=sorted(VEHICLES))
    parser.add_argument("--condition", default="correct", choices=("correct", "stable"),
                        help="'correct' keeps samples both arms answer correctly unfaulted (the probe convention); "
                             "'stable' keeps samples whose no-fault controls did not move")
    args = parser.parse_args()

    raw_root = resolve_output_root(args.output_root) / "raw"
    arms = {fam: load_arm(raw_root, name) for fam, name in VEHICLES[args.vehicle].items()}
    if args.condition == "correct":
        samples = shared_samples(arms)
    else:
        # LongBench accuracy is low -- the D63 pilot put Qwen's F1 mean at 0.071 -- so conditioning on a CORRECT
        # golden leaves almost nothing: it left 4 samples here, which is why this option exists. `d73`'s config
        # pre-registered the looser condition and the first version of this report did not implement it, so the two
        # disagreed. The quantity needs the golden STABLE, not right: a KL is measured against the arm's own
        # unfaulted distribution, and `answer_changed` is measured against its own unfaulted answer, so both are
        # well defined on a sample the model gets wrong as long as it gets it wrong *consistently*.
        keep: set[str] | None = None
        for data in arms.values():
            stable = {s for s, r in data.items()
                      if all(not t["answer_changed"] for t in r["faults"] if t["trial"] < 0)}
            keep = stable if keep is None else (keep & stable)
        samples = sorted(keep or ())
    if not samples:
        raise SystemExit(f"no sample satisfies --condition {args.condition} in both arms")
    print(f"[condition: {args.condition}]")
    out: dict = {"n_samples": len(samples), "samples": samples}

    print(f"D62: one metadata flip at a time, graded. {len(samples)} samples shared by both arms\n")

    print("zero baseline -- the control trials read the fault-free cache and their KL must be EXACTLY 0.0\n")
    for fam, data in arms.items():
        bad, n, worst = controls_are_exactly_zero(data)
        changed, total = _controls_clean(data)
        verdict = "exactly zero" if bad == 0 else f"*** {bad} of {n} NONZERO ***"
        print(f"  {fam:6s} {n} controls, KL {verdict} (worst |KL| = {worst:.3e}); "
              f"{changed}/{total} changed the answer")
        out.setdefault("zero_baseline", {})[fam] = {"n": n, "nonzero": bad, "worst_abs_kl": worst}
        if bad:
            print("         every small KL below is noise of unknown size rather than the effect of a flip")

    # The catastrophic class, before anything that excludes it. A third of all break events are this, not a subtle
    # answer change, and a report that dropped them into a NaN would have said nothing about the worst thing a
    # single metadata flip can do.
    print("\n" + "=" * 100)
    print("\nNON-FINITE COLLAPSE -- flips after which the model's own forward pass produced NaN/inf\n")
    print(f"{'family':8s}{'k':>5s}{'n':>7s}{'rate':>9s}{'95% CI':>22s}{'all changed':>13s}{'zero-point':>12s}")
    for fam, data in arms.items():
        kl, changed, _ = flip_kls(data, samples)
        bad = nonfinite_mask(kl)
        # Which metadata word did it: a scale or a zero-point. Recorded per trial by `plan_composition`.
        targets = [t.get("by_target", {}) for s in samples for t in _trials(data[s])]
        zeros = sum(1 for t, b in zip(targets, bad, strict=True) if b and t.get("zero"))
        lo, hi = wilson(int(bad.sum()), int(len(kl)))
        print(f"{fam:8s}{int(bad.sum()):5d}{len(kl):7d}{bad.mean():9.3%}   [{lo:.3%}, {hi:.3%}]"
              f"{str(bool(changed[bad].all()) if bad.any() else '-'):>13s}{f'{zeros}/{int(bad.sum())}':>12s}")
        out.setdefault("nonfinite", {})[fam] = {
            "k": int(bad.sum()), "n": int(len(kl)), "rate": float(bad.mean()), "ci": (lo, hi),
            "all_answer_changed": bool(changed[bad].all()) if bad.any() else None,
            "from_zero_point": int(zeros), "share_of_all_breaks":
                float(bad.sum() / changed.sum()) if changed.sum() else None}
        if bad.any():
            print(f"{'':8s}  that is {bad.sum() / max(1, changed.sum()):.1%} of this family's break events, and "
                  f"the prediction is a degenerate repetition rather than a wrong answer")

    # The anchor, next, because it can void the two primaries and printing them above it would invite reading
    # them before knowing whether they mean anything.
    print("\n" + "=" * 100)
    print("\nANCHOR -- does KL separate the flips that broke the answer from the flips that did not?")
    print("  (AUC 0.5 = no discrimination = the graded measure is not measuring what breaks answers,")
    print("   and both primaries below would be uninterpretable)\n")
    anchor_ok = True
    for fam, data in arms.items():
        kl, changed, flipped = flip_kls(data, samples)
        ok = ~nonfinite_mask(kl)
        kl, changed, flipped = kl[ok], changed[ok], flipped[ok]
        a = auc_with_ci(kl, changed, args.seed)
        out.setdefault("anchor", {})[fam] = a
        if a["auc"] is None:
            print(f"  {fam:6s} {a['note']} (changed {a['n_pos']}, unchanged {a['n_neg']})")
            anchor_ok = False
            continue
        print(f"  {fam:6s} AUC = {a['auc']:.3f} [{a['ci'][0]:.3f}, {a['ci'][1]:.3f}], p = {a['p']:.3g}; "
              f"median KL {a['median_pos']:.3e} when the answer changed ({a['n_pos']} flips) against "
              f"{a['median_neg']:.3e} when it did not ({a['n_neg']})")
        anchor_ok &= bool(a["discriminates"])
        # argmax_flipped is a different event from answer_changed and is reported so the gap is visible.
        af = auc_with_ci(kl, flipped, args.seed)
        if af["auc"] is not None:
            print(f"  {'':6s} on argmax_flipped instead: AUC = {af['auc']:.3f} "
                  f"[{af['ci'][0]:.3f}, {af['ci'][1]:.3f}] ({af['n_pos']} flips flipped the first token)")
    print(f"\n  anchor {'HOLDS -- the primaries below are interpretable' if anchor_ok else 'FAILS'}")

    print("\n" + "=" * 100)
    kl_q, kl_l = (per_sample_median_kl(arms[f], samples) for f in ("qwen", "llama"))
    c1 = contrast("qwen_median_kl", "llama_median_kl", kl_q, kl_l, args.seed)
    # Also on the log scale: a ratio is the natural form for a heavy-tailed quantity, and branch A predicts a
    # factor near 2.25x rather than an additive shift in nats.
    floor = min(v for v in np.concatenate([kl_q, kl_l]) if v > 0) if (np.concatenate([kl_q, kl_l]) > 0).any() else 1.0
    c1log = contrast("qwen_log_kl", "llama_log_kl",
                     np.log10(np.maximum(kl_q, floor)), np.log10(np.maximum(kl_l, floor)), args.seed)
    print("\nPRIMARY 1 (branch A, per-flip severity) -- per-sample MEDIAN KL, Llama against Qwen\n"
          f"  qwen {c1['x_mean']:.4e} -> llama {c1['y_mean']:.4e}, difference {c1['difference']:+.4e} "
          f"[{c1['ci'][0]:+.4e}, {c1['ci'][1]:+.4e}], p = {c1['p']:.3g}, "
          f"{c1['discordant']}/{c1['n_pairs']} samples moved\n"
          f"  {'MEETS D31' if c1['meets_rule'] else 'does NOT meet D31'}\n"
          f"  on log10: {c1log['difference']:+.3f} dex [{c1log['ci'][0]:+.3f}, {c1log['ci'][1]:+.3f}] "
          f"= {10 ** c1log['difference']:.2f}x, p = {c1log['p']:.3g}  "
          f"({'MEETS D31' if c1log['meets_rule'] else 'does NOT meet D31'})\n"
          f"  branch A predicts Llama higher by roughly the 2.25x hazard ratio if severity fully explains D59")
    out["primary1_median_kl"] = c1
    out["primary1_median_kl_log10"] = c1log

    m_q, m_l = (per_sample_clean_margin(arms[f], samples) for f in ("qwen", "llama"))
    c2 = contrast("qwen_margin", "llama_margin", m_q, m_l, args.seed)
    print("\nPRIMARY 2 (branch B, a thinner margin) -- per-sample FAULT-FREE margin, Llama against Qwen\n"
          f"  qwen {c2['x_mean']:.4f} -> llama {c2['y_mean']:.4f}, difference {c2['difference']:+.4f} "
          f"[{c2['ci'][0]:+.4f}, {c2['ci'][1]:+.4f}], p = {c2['p']:.3g}, "
          f"{c2['discordant']}/{c2['n_pairs']} samples moved\n"
          f"  {'MEETS D31' if c2['meets_rule'] else 'does NOT meet D31'}\n"
          f"  branch B predicts Llama LOWER here; bf16 logits quantize this to about 0.25, so read two "
          f"significant figures at most")
    out["primary2_clean_margin"] = c2

    print("\n" + "=" * 100)
    print("\nDESCRIPTIVE -- the KL distribution per family, over the FINITE flips (no verdict attaches)\n")
    print(f"{'family':8s}{'flips':>7s}{'zero KL':>9s}{'p50':>11s}{'p90':>11s}{'p99':>11s}{'max':>11s}"
          f"{'broke':>7s}{'flip1st':>8s}")
    for fam, data in arms.items():
        kl, changed, flipped = flip_kls(data, samples)
        ok = ~nonfinite_mask(kl)
        kl, changed, flipped = kl[ok], changed[ok], flipped[ok]
        print(f"{fam:8s}{len(kl):7d}{np.mean(kl == 0):9.3f}{np.median(kl):11.3e}"
              f"{np.percentile(kl, 90):11.3e}{np.percentile(kl, 99):11.3e}{kl.max():11.3e}"
              f"{changed.sum():7d}{flipped.sum():8d}")
        out.setdefault("distribution", {})[fam] = {
            "n_flips": int(len(kl)), "frac_zero_kl": float(np.mean(kl == 0)),
            "p50": float(np.median(kl)), "p90": float(np.percentile(kl, 90)),
            "p99": float(np.percentile(kl, 99)), "max": float(kl.max()),
            "n_answer_changed": int(changed.sum()), "n_argmax_flipped": int(flipped.sum())}

    dest = resolve_output_root(args.output_root) / "reports" / f"flip_severity_{args.vehicle}_{args.condition}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
