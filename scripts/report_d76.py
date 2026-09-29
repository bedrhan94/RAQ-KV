"""Does the codec move delta? The exponent relating a fault's damage to the span its corrupted scale reaches.

D62 tested four mechanisms for the cross-family fragility gap and none was a codec-side lever, which left the allocator
nothing to choose between on reliability grounds. D75 then produced a codec-side quantity where there was none:
``delta``, the effective logit perturbation from one metadata flip, comparable across configurations in a way a KL is
not. This measures whether a **codec setting** moves it.

    python scripts/report_d76.py

**PRIMARY, pre-registered** (`docs/preregistrations/d76_prereg.md`): ``alpha``, the slope of ln delta against ln of the
**realised per-flip span** -- how many stored values the corrupted scale actually governs, recorded per trial as
``governed_values``. Three answers named in advance: ``alpha ~ 1`` means the span's values contribute additively,
``alpha ~ 0.5`` means their errors add in quadrature, ``alpha ~ 0`` means damage is set by something other than span.

**THE ALLOCATOR CONSEQUENCE, committed before the run.** Key metadata bits scale as ``1/g``, so at a fixed bit error
rate the *number* of key metadata flips also scales as ``1/g`` while damage per flip scales as ``g^alpha``. Total
expected damage therefore goes as ``g^(alpha-1)``:

* ``alpha = 1`` -- the two effects cancel and group size is **not a reliability lever**; choose it on storage alone.
* ``alpha < 1`` -- larger groups are **strictly better** for reliability and also cheaper in bits. No trade-off.
* ``alpha > 1`` -- smaller groups are better and cost more bits; the exponent sets the exchange rate.

**Why the realised span and not the nominal group size.** The trailing group of a sequence is short and at coarse
granularity it is not rare: at the probe's median 2052-token cache and ``group_size`` 512 there are 5 groups and the
last holds 4 values. Regressing on the nominal size compresses the realised 32-512 range from 16x to 13x and biases
alpha **downward by 0.925**, toward the most consequential of the three answers.
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np
from scipy import stats

from raq_kv.analysis.arms import sweep_runs
from raq_kv.analysis.contrasts import contrast_rng
from raq_kv.analysis.curvature import effective_perturbation, effective_perturbation_observed
from raq_kv.utils.paths import resolve_output_root

NULLS = {"span-independent": 0.0, "quadrature": 0.5, "coherent": 1.0}


def collect(records: list[dict]) -> dict[str, np.ndarray]:
    """Per-trial arrays, one row per fault trial, with the sample index kept for the cluster bootstrap."""
    cols: dict[str, list] = {k: [] for k in
                             ("sample", "span", "kl", "margin", "p1", "tail", "changed", "nonfinite", "zero")}
    for s, record in enumerate(records):
        for trial in record["faults"]:
            if trial["trial"] < 0:
                continue  # controls carry no fault and no span
            cols["sample"].append(s)
            cols["span"].append(float(trial.get("governed_values", np.nan)))
            cols["kl"].append(float(trial.get("kl_first_token", np.nan)))
            cols["margin"].append(float(trial.get("margin_clean", np.nan)))
            cols["p1"].append(float(trial.get("top1_prob_clean", np.nan)))
            cols["tail"].append(float(trial.get("nontop_mass_clean", np.nan)))
            cols["changed"].append(bool(trial["answer_changed"]))
            cols["nonfinite"].append(bool(trial.get("faulty_nonfinite", False)))
            cols["zero"].append(int((trial.get("by_target") or {}).get("zero", 0)) > 0)
    return {k: np.array(v) for k, v in cols.items()}


def binned_fit(span: np.ndarray, delta: np.ndarray) -> tuple[float, float, list[dict]]:
    """``(slope, r2, bins)`` from the **median** delta in each octave of span.

    Octave bins rather than one bin per distinct span: the distinct values are the nominal group size plus one trailing
    remainder per cache length, so binning by value would give a handful of huge bins and a long tail of singletons.
    The median within each bin is what makes this describe the typical flip rather than the tail, which the
    pre-registration commits to.
    """
    keep = np.isfinite(span) & np.isfinite(delta) & (span > 0) & (delta > 0)
    span, delta = span[keep], delta[keep]
    if span.size < 10:
        return float("nan"), float("nan"), []
    # `+ 2`, and it is load-bearing. The bins are half-open `[lo, hi)`, so the top edge has to sit STRICTLY ABOVE the
    # largest span or every flip at exactly that span falls outside every bin. With `+ 1` the edges stopped at
    # 2**ceil(log2(max)) == max whenever the max is a power of two -- which it always is here, because these spans are
    # group sizes. That silently dropped the coarsest rung's entire full-span population: D76's first pass fitted alpha
    # on 2,920 of 3,200 trials with no 512 bin at all, and D77's channel sweep lost every span-32 flip (913 of 1,432).
    # The count is asserted against the bins below so the failure cannot recur quietly.
    edges = [2.0 ** k for k in range(0, 2 + int(math.ceil(math.log2(max(span.max(), 2.0)))))]
    bins = []
    for lo, hi in zip(edges, edges[1:], strict=False):
        m = (span >= lo) & (span < hi)
        if m.sum() >= 5:
            bins.append({"lo": lo, "hi": hi, "n": int(m.sum()),
                         "span": float(np.exp(np.log(span[m]).mean())), "delta": float(np.median(delta[m]))})
    binned = sum(b["n"] for b in bins)
    if binned < 0.95 * span.size:
        raise AssertionError(f"binning dropped {span.size - binned} of {span.size} flips (spans "
                             f"{span.min():.0f}-{span.max():.0f}, edges up to {edges[-1]:.0f}); the exponent would be "
                             f"fitted on a population the caller did not choose")
    if len(bins) < 3:
        return float("nan"), float("nan"), bins
    fit = stats.linregress([math.log(b["span"]) for b in bins], [math.log(b["delta"]) for b in bins])
    return float(fit.slope), float(fit.rvalue ** 2), bins


def cluster_bootstrap(sample: np.ndarray, span: np.ndarray, delta: np.ndarray,
                      seed: int, n: int = 2_000) -> tuple[float, float]:
    """Bootstrap the binned slope by resampling **samples**, not trials.

    Trials within a sample share its prompt, its clean distribution and its margin, so resampling trials would treat
    16 correlated draws as 16 independent ones and report an interval several times too tight.
    """
    rng = contrast_rng("span", "delta", seed)
    ids = np.unique(sample)
    index = {i: np.flatnonzero(sample == i) for i in ids}
    out = []
    for _ in range(n):
        pick = np.concatenate([index[i] for i in rng.choice(ids, size=len(ids), replace=True)])
        slope, _, _ = binned_fit(span[pick], delta[pick])
        if math.isfinite(slope):
            out.append(slope)
    if len(out) < 100:
        return float("nan"), float("nan")
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(lo), float(hi)


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    z, p = 1.96, k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="d76_group_span_qwen")
    parser.add_argument("--groups", help="comma-separated group sizes the exponent may use; the rest are still listed "
                                         "in PART 1 and excluded only from the fit")
    parser.add_argument("--min-golden", type=float, default=None,
                        help="exclude rungs whose CLEAN accuracy is below this fraction, naming each and why. D77's "
                             "channel-axis sweep needs it: at group_channel 64 and 128 the codec alone drives "
                             "golden_correct to 0/40 and the model emits degenerate repetition, so those rungs' delta "
                             "is measured on a different model and does not belong in a slope describing one")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    raw_root = resolve_output_root(args.output_root) / "raw"

    rungs: dict[int, dict] = {}
    for exp, manifest, records in sweep_runs(raw_root, args.name):
        quant = manifest["config"]["quantization"]
        # The rung label is the group size of the part the FAULT DRAW is confined to, not always the keys'. D82 holds
        # keys at group_token 128 and sweeps VALUES, so keying on the keys collapsed all five rungs onto 128 and four
        # silently overwrote each other -- the report then showed one rung and an unfittable slope. `metadata_part` is
        # what the draw was confined to and is therefore what the span varies with.
        part = (manifest["config"]["task"].get("metadata_part") or "keys")
        g = int(quant[part]["group_size"])
        if g in rungs:
            raise SystemExit(
                f"two rungs of {args.name!r} both label as {part} group_size {g} ({rungs[g]['exp']} and {exp.name}); "
                f"one would overwrite the other and the exponent would be fitted on a population the caller did not "
                f"choose. Key the rungs on whatever this sweep actually varies")
        rungs[g] = {"exp": exp.name, "records": records, "cols": collect(records), "part": part}
    if not rungs:
        raise SystemExit(f"no rungs for {args.name!r}")

    out: dict = {"rungs": {}}
    print(f"D76 -- does the codec move delta?  {len(rungs)} rungs of {args.name}\n")

    # PART 1, the per-rung picture, including every pre-registered secondary that is per-rung.
    part_label = f"{next(iter(rungs.values()))['part'][:3]} SQNR"
    print(f"{'group':>6s}{'n':>4s}{'trials':>7s}{'span med':>10s}{'span mean':>11s}{part_label:>10s}"
          f"{'bits/val':>10s}{'delta':>9s}{'break':>8s}{'95% CI':>18s}{'nonfin':>8s}")
    for g in sorted(rungs):
        c = rungs[g]["cols"]
        rec0 = rungs[g]["records"][0]
        part = rungs[g]["part"]
        sqnr = ((rec0.get("quantization") or {}).get(part) or {}).get("sqnr_db_mean")
        bits = rec0.get("avg_kv_bits")
        fin = np.isfinite(c["kl"]) & (c["kl"] > 0) & np.isfinite(c["margin"])
        delta = effective_perturbation(c["margin"][fin], c["kl"][fin])
        k, n = int(c["changed"].sum()), len(c["changed"])
        lo, hi = wilson(k, n)
        nf = int(c["nonfinite"].sum())
        print(f"{g:6d}{len(rungs[g]['records']):4d}{n:7d}{np.median(c['span']):10.1f}{c['span'].mean():11.1f}"
              f"{(sqnr if sqnr is not None else float('nan')):10.2f}"
              f"{(bits if bits is not None else float('nan')):10.4f}{np.median(delta):9.4f}"
              f"{k / n:8.4f}   [{lo:.4f}, {hi:.4f}]{nf:8d}")
        out["rungs"][g] = {"exp": rungs[g]["exp"], "n_samples": len(rungs[g]["records"]), "n_trials": n,
                           "span_median": float(np.median(c["span"])), "span_mean": float(c["span"].mean()),
                           "key_sqnr_db": sqnr, "avg_kv_bits": bits, "delta_median": float(np.median(delta)),
                           "break_rate": k / n, "break_ci": (lo, hi), "nonfinite": nf,
                           "nonfinite_zero": int((c["nonfinite"] & c["zero"]).sum())}

    # PART 2, the PRIMARY: the exponent from the realised per-flip span, pooled across rungs.
    # Which rungs may enter it is a scientific choice, not a convenience. A rung whose clean answers are ALL wrong is a
    # different model -- D77's `group_channel` 64 and 128 reach golden_correct 0/40 on the codec alone, emitting
    # degenerate repetition -- and its delta cannot sit in a slope that claims to describe one model.
    keep = sorted(rungs)
    if args.groups:
        wanted = {int(x) for x in args.groups.split(",")}
        keep = [g for g in keep if g in wanted]
    if args.min_golden is not None:
        for g in sorted(rungs):
            if g not in keep:
                continue
            recs = rungs[g]["records"]
            acc = sum(1 for r in recs if r["golden_correct"]) / len(recs)
            if acc < args.min_golden:
                keep.remove(g)
                print(f"  EXCLUDED from the exponent: group {g}, clean accuracy {acc:.3f} < --min-golden "
                      f"{args.min_golden:.3f}; its delta is measured on a model the codec already broke")
    if not keep:
        raise SystemExit("every rung was excluded from the exponent; --min-golden or --groups is too strict")
    if keep != sorted(rungs):
        print(f"  exponent fitted on groups {keep} of {sorted(rungs)}")
    allc = {k: np.concatenate([rungs[g]["cols"][k] for g in keep]) for k in rungs[next(iter(rungs))]["cols"]}
    # Sample ids repeat across rungs, so offset them per rung or the bootstrap would tie unrelated samples together.
    offset, sample_ids = 0, []
    for g in keep:
        s = rungs[g]["cols"]["sample"]
        sample_ids.append(s + offset)
        offset += int(s.max()) + 1 if s.size else 0
    allc["sample"] = np.concatenate(sample_ids)

    print("\nPRIMARY -- alpha, the slope of ln delta against ln(realised span)\n")
    for label, (margin, kl, p1, tail) in {
            "assumed (two-state from the margin)": (allc["margin"], allc["kl"], None, None),
            "observed (recorded p1, no assumption)": (None, allc["kl"], allc["p1"], allc["tail"])}.items():
        if p1 is None:
            fin = np.isfinite(margin) & np.isfinite(kl) & (kl > 0)
            delta = effective_perturbation(margin[fin], kl[fin])
        else:
            fin = np.isfinite(p1) & np.isfinite(tail) & (tail > 0) & np.isfinite(kl) & (kl > 0)
            if not fin.any():
                print(f"  {label:40s} not recorded on these rungs")
                continue
            delta = effective_perturbation_observed(p1[fin], tail[fin], kl[fin])
        span, samp = allc["span"][fin], allc["sample"][fin]
        alpha, r2, bins = binned_fit(span, delta)
        lo, hi = cluster_bootstrap(samp, span, delta, args.seed)
        # A nan alpha makes every comparison False, so `not lo <= v <= hi` would print "EXCLUDED" for all three
        # nulls -- the exact opposite of what an unfittable slope means. Say so instead.
        if not (math.isfinite(alpha) and math.isfinite(lo) and math.isfinite(hi)):
            verdicts = ("NOT FITTABLE on these rungs -- too few span bins for a slope, so no null is excluded "
                        "and none is supported")
        else:
            verdicts = ", ".join(f"{name} ({v:.1f}): {'EXCLUDED' if not lo <= v <= hi else 'in interval'}"
                                 for name, v in NULLS.items())
        print(f"  {label:40s} alpha {alpha:+.3f}  [{lo:+.3f}, {hi:+.3f}]  R^2 {r2:.3f}  n={int(fin.sum())}")
        print(f"  {'':40s} {verdicts}")
        if bins:
            print(f"  {'':40s} bins: " + "  ".join(f"{b['span']:.0f}->{b['delta']:.4f}(n={b['n']})" for b in bins))
        out[f"primary:{label}"] = {"alpha": alpha, "ci": (lo, hi), "r2": r2, "n": int(fin.sum()), "bins": bins}
        if not math.isfinite(alpha):
            continue
        print(f"  {'':40s} ALLOCATOR: total damage goes as g^({alpha:+.3f} - 1) = g^{alpha - 1:+.3f}  ->  "
              f"{'larger groups better' if alpha < 1 else 'smaller groups better' if alpha > 1 else 'no lever'}")

    # PART 3, SECONDARY 1: the original per-rung reading, kept so both are on the record.
    gs = keep
    med = []
    for g in gs:
        c = rungs[g]["cols"]
        fin = np.isfinite(c["kl"]) & (c["kl"] > 0) & np.isfinite(c["margin"])
        med.append(float(np.median(effective_perturbation(c["margin"][fin], c["kl"][fin]))))
    if len(gs) >= 3:
        f = stats.linregress(np.log(gs), np.log(med))
        print(f"\nSECONDARY 1 -- the original per-rung statistic: alpha {f.slope:+.3f} "
              f"[{f.slope - 1.96 * f.stderr:+.3f}, {f.slope + 1.96 * f.stderr:+.3f}], R^2 {f.rvalue ** 2:.3f} "
              f"over {len(gs)} rungs")
        out["secondary:per_rung"] = {"alpha": float(f.slope), "r2": float(f.rvalue ** 2),
                                     "group_sizes": gs, "delta_medians": med}

    # PART 4, SECONDARY 5: how wrong was D75's two-state assumption, measured on the same trials?
    fin = (np.isfinite(allc["p1"]) & np.isfinite(allc["tail"]) & (allc["tail"] > 0)
           & np.isfinite(allc["margin"]) & np.isfinite(allc["kl"]) & (allc["kl"] > 0))
    if fin.any():
        a = effective_perturbation(allc["margin"][fin], allc["kl"][fin])
        o = effective_perturbation_observed(allc["p1"][fin], allc["tail"][fin], allc["kl"][fin])
        ratio = float(np.median(a / o))
        print(f"\nSECONDARY 5 -- D75's assumption, measured: assumed delta / observed delta = {ratio:.3f}x "
              f"(median over {int(fin.sum())} trials)")
        print(f"  implied tail inflation, true non-top mass against the runner-up alone: {ratio ** 2:.2f}x")
        out["secondary:assumption"] = {"assumed_over_observed": ratio, "implied_tail_inflation": ratio ** 2,
                                       "n": int(fin.sum())}
    else:
        print("\nSECONDARY 5 -- top1_prob_clean not present on these rungs, so the assumption cannot be checked here")

    # PART 5, SECONDARY 4: the zero-point split, D68's finding on a fourth axis.
    nf_tot = int(allc["nonfinite"].sum())
    if nf_tot:
        nz = int((allc["nonfinite"] & allc["zero"]).sum())
        print(f"\nSECONDARY 4 -- non-finite collapse: {nf_tot} of {len(allc['nonfinite'])} trials, "
              f"{nz} of them ({nz / nf_tot:.0%}) from a zero-point flip")
        out["secondary:nonfinite"] = {"n": nf_tot, "from_zero": nz, "trials": len(allc["nonfinite"])}
    else:
        print(f"\nSECONDARY 4 -- no non-finite collapse in {len(allc['nonfinite'])} trials")

    dest = resolve_output_root(args.output_root) / "reports" / "d76_group_span.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
