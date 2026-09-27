"""How much of a cross-vehicle KL change is the margin change restated, and what is the fault's own size?

D73's headline is that Qwen's median per-flip KL rises **39,005x** between the probe and LongBench `2wikimqa`, read as
"as direct a confirmation as this design can give" of D62's saturation account. D75's control undermines the reading.
On the probe, `ln KL` against `ln(p1 p2)` has slope **+0.985** [+0.943, +1.027] with R^2 = 0.914 -- the softmax
curvature identity

    ln KL  =  ln(p1 p2)  +  2 ln delta  +  const

holds there to within its confidence interval. So a margin change predicts a KL change **with no change whatever in
how much damage a flip does**, and the only question that bears on D73's inference is what is left after dividing that
out.

    python scripts/report_kl_curvature_accounting.py

**Two things are reported and the second is the one to keep.** The residual ratio answers D73 directly. `delta`, the
effective logit perturbation, answers the question D62 and D73 were really asking -- how hard does one metadata flip
hit this model -- in units that are comparable across models and tasks, which a KL is not.

**The two-state assumption and which way it cuts.** `p1 p2` here is the two-effective-state reading of the margin.
Against the recorded `top1_prob` of the D63 ladder that is accurate where margins are deep (Qwen: the non-top mass is
1.34x the runner-up alone) and optimistic where they are thin (Llama: **2.30x**). Since `delta` scales as
`1 / sqrt(p1 p2)`, correcting it *shrinks* Llama's delta more than Qwen's -- so the family gap reported below is a
**lower bound**, and the sensitivity row makes that explicit rather than leaving it as a caveat.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from raq_kv.analysis.arms import _trials, load_arm
from raq_kv.analysis.contrasts import contrast_rng
from raq_kv.analysis.curvature import effective_perturbation, log_curvature
from raq_kv.utils.paths import resolve_output_root

# (family, vehicle, arm, condition). The two vehicles use different datasets, so a cross-vehicle comparison is
# necessarily unpaired -- D73 paired within a vehicle across families, which is a different contrast.
CELLS = (("Qwen3-1.7B", "probe", "d62_qwen_meta", "correct"),
         ("Qwen3-1.7B", "LongBench", "d73_lb_fault_qwen", "stable"),
         ("Llama-3.2-3B", "probe", "d62_llama_meta", "correct"),
         ("Llama-3.2-3B", "LongBench", "d73_lb_fault_llama", "stable"),
         ("Llama-3.2-3B", "LongBench-d74", "d74_margin_curve_llama_lb", "stable"))
# D73's own quoted medians, verbatim, so the residual's sensitivity to the sample set is visible rather than argued.
D73_QUOTED = {"Qwen3-1.7B": (17.07, 7.500, 39_005.0), "Llama-3.2-3B": (3.27, 1.906, 12.0)}
# Measured ratio of the true non-top mass to the runner-up alone, from the recorded top1_prob of the D63 ladder.
TAIL_INFLATION = {"Qwen3-1.7B": 1.34, "Llama-3.2-3B": 2.30}


def cell(raw_root, arm: str, condition: str) -> dict | None:
    """Per-sample margins, median KLs and deltas for one arm, on that arm's own samples (D71's rule)."""
    try:
        data = load_arm(raw_root, arm)
    except SystemExit:
        return None
    margins, kls = [], []
    for record in data.values():
        if condition == "correct" and not record.get("golden_correct"):
            continue
        if condition == "stable" and any(t["answer_changed"] for t in record["faults"] if t["trial"] < 0):
            continue
        trials = _trials(record)
        if not trials or "margin_clean" not in trials[0]:
            continue
        kl = np.array([float(t["kl_first_token"]) for t in trials])
        kl = np.where(np.isfinite(kl), kl, np.inf)
        value = float(np.median(kl))
        if not np.isfinite(value) or value <= 0:
            continue
        margins.append(float(trials[0]["margin_clean"]))
        kls.append(value)
    if len(margins) < 5:
        return None
    m, k = np.array(margins), np.array(kls)
    return {"margin": m, "kl": k, "delta": effective_perturbation(m, k), "n": len(m)}


def boot_ratio(a: np.ndarray, b: np.ndarray, seed: int = 0, n: int = 10_000) -> tuple[float, tuple[float, float]]:
    rng = contrast_rng("delta", "ratio", seed)
    out = [np.median(a[rng.integers(0, len(a), len(a))]) / np.median(b[rng.integers(0, len(b), len(b))])
           for _ in range(n)]
    return float(np.median(out)), tuple(float(v) for v in np.percentile(out, [2.5, 97.5]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    raw_root = resolve_output_root(args.output_root) / "raw"
    cells = {(f, v): cell(raw_root, a, c) for f, v, a, c in CELLS}
    out: dict = {}

    print("PART 1 -- the effective logit perturbation, which is what a KL was standing in for\n")
    print(f"{'family':14s}{'vehicle':16s}{'n':>4s}{'median margin':>15s}{'median KL':>12s}{'delta':>9s}{'IQR':>18s}")
    for family, vehicle, _, _ in CELLS:
        c = cells[(family, vehicle)]
        if c is None:
            print(f"{family:14s}{vehicle:16s}  missing")
            continue
        print(f"{family:14s}{vehicle:16s}{c['n']:4d}{np.median(c['margin']):15.3f}{np.median(c['kl']):12.3e}"
              f"{np.median(c['delta']):9.4f}   {np.percentile(c['delta'], 25):.4f}-"
              f"{np.percentile(c['delta'], 75):.4f}")
        out[f"{family}:{vehicle}"] = {"n": c["n"], "margin_median": float(np.median(c["margin"])),
                                      "kl_median": float(np.median(c["kl"])),
                                      "delta_median": float(np.median(c["delta"]))}

    print("\nPART 2 -- D73's cross-vehicle KL rise against what the margin change alone predicts\n")
    print(f"{'family':14s}{'sample set':>16s}{'curvature gain':>16s}{'predicted':>12s}{'measured':>11s}"
          f"{'residual':>10s}")
    for family in ("Qwen3-1.7B", "Llama-3.2-3B"):
        a, b = cells[(family, "probe")], cells[(family, "LongBench")]
        rows = []
        if a and b:
            # The gain going probe -> LongBench: ln(p1 p2) is LESS negative at the thinner LongBench margin, so this
            # is positive and exp() of it is the predicted KL RISE. Subtracting the other way round inverts the
            # prediction and the residual explodes, which is how this line read on its first pass.
            gain = float(log_curvature(np.median(b["margin"])) - log_curvature(np.median(a["margin"])))
            rows.append(("each arm's own", gain, float(np.median(b["kl"]) / np.median(a["kl"]))))
        if family in D73_QUOTED:
            pm, lm, rise = D73_QUOTED[family]
            rows.append(("D73's quoted", float(log_curvature(lm) - log_curvature(pm)), rise))
        for label, gain, measured in rows:
            predicted = float(np.exp(gain))
            print(f"{family:14s}{label:>16s}{gain:+16.3f}{predicted:11,.0f}x{measured:10,.0f}x"
                  f"{measured / predicted:9.1f}x")
            out[f"{family}:residual:{label}"] = {"curvature_gain_nats": gain, "predicted": predicted,
                                                 "measured": measured, "residual": measured / predicted}
        print()

    print("PART 3 -- is the family difference in delta real, and does it run with the hazard or against it?\n")
    print(f"{'contrast':46s}{'ratio':>9s}{'95% CI':>18s}  D31")
    contrasts = (("Qwen/Llama on the probe", ("Qwen3-1.7B", "probe"), ("Llama-3.2-3B", "probe")),
                 ("Qwen/Llama on LongBench", ("Qwen3-1.7B", "LongBench"), ("Llama-3.2-3B", "LongBench")),
                 ("Qwen/Llama on LongBench, d74's 80 samples",
                  ("Qwen3-1.7B", "LongBench"), ("Llama-3.2-3B", "LongBench-d74")),
                 ("Qwen across vehicles (LongBench/probe)", ("Qwen3-1.7B", "LongBench"), ("Qwen3-1.7B", "probe")),
                 ("Llama across vehicles (LongBench/probe)", ("Llama-3.2-3B", "LongBench"),
                  ("Llama-3.2-3B", "probe")))
    for label, ka, kb in contrasts:
        ca, cb = cells[ka], cells[kb]
        if ca is None or cb is None:
            continue
        ratio, (lo, hi) = boot_ratio(ca["delta"], cb["delta"], args.seed)
        # D31 for a ratio: the interval must exclude 1, not 0 -- a ratio of 1 is the no-difference point.
        meets = not lo <= 1.0 <= hi
        print(f"{label:46s}{ratio:8.2f}x   [{lo:.2f}, {hi:.2f}]  {'yes' if meets else 'no'}")
        out[f"delta:{label}"] = {"ratio": ratio, "ci": (lo, hi), "excludes_one": bool(not lo <= 1.0 <= hi)}

    # Sensitivity, stated as a number rather than a caveat: correcting the two-state p1p2 by each family's measured
    # tail inflation shrinks delta by sqrt(inflation), and Llama's tail is the fatter one, so the gap widens.
    qa, la = cells[("Qwen3-1.7B", "probe")], cells[("Llama-3.2-3B", "probe")]
    if qa and la:
        naive = float(np.median(qa["delta"]) / np.median(la["delta"]))
        corrected = naive * float(np.sqrt(TAIL_INFLATION["Llama-3.2-3B"] / TAIL_INFLATION["Qwen3-1.7B"]))
        print(f"\n  sensitivity: correcting p1 p2 by each family's measured tail inflation "
              f"({TAIL_INFLATION['Qwen3-1.7B']}x against {TAIL_INFLATION['Llama-3.2-3B']}x) moves the probe gap from "
              f"{naive:.2f}x to {corrected:.2f}x -- the two-state form UNDERSTATES it, so the gap is a lower bound.")
        out["delta:sensitivity"] = {"naive": naive, "tail_corrected": corrected}

    print("\n  Read PART 2's residual column and PART 3's direction. A residual near 1x means the KL rise is the")
    print("  margin drop restated, not independent corroboration of it -- which is how D62 and D73 both used it. And")
    print("  delta runs OPPOSITE to the hazard: Llama breaks more (2.63% against 1.17% on the probe, 9.38% against")
    print("  0.78% on LongBench) while one flip moves its logits LESS. The family hazard gap is the margin, not the")
    print("  size of the perturbation.")
    dest = resolve_output_root(args.output_root) / "reports" / "kl_curvature_accounting.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
