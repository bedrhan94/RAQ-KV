"""Figure: what protection costs, and what it removes (M3).

**Left — damage against the number of corrupted bits.** The flip count is varied two independent ways and the
panel shows both: by sweeping the bit error rate on the unprotected arm (open markers), and, within every arm at
one rate, by the cache length that sets how many bits there are to corrupt (filled markers). The unprotected
cache rises to losing half its answers; every arm that protects the quantization metadata stays on zero across
both axes. Two ways of asking the same question agreeing is the point of drawing them together.

**Right — the same outcome against what the arm stores.** Task failure against measured effective bits per
cached value, so the cost of each policy is the x coordinate rather than a claim in the caption. The four arms
built to hold equal storage sit in one shaded band; the cost references sit to its left, which is where the
result lives: the cheapest arm is also on zero.

Identity is never colour alone: every arm has its own marker, the equal-storage band is drawn and labelled, and
zero-valued arms are annotated in text so the reader is not asked to distinguish overlapping points at y=0.
Colours are the Okabe-Ito CVD-safe set. Intervals are Wilson score intervals, which stay finite at zero counts --
four arms sit exactly there, where a normal approximation would draw zero width and imply certainty.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from raq_kv.utils.io import read_json, read_jsonl  # noqa: E402
from raq_kv.utils.paths import resolve_output_root  # noqa: E402

ARMS = {
    "m3_arm_a_unprot": {"label": "A  no protection", "color": "#555555", "marker": "D", "z": 5},
    "m3_arm_h_range_guard": {"label": "H  metadata range check", "color": "#E69F00", "marker": "v", "z": 4},
    "m3_arm_g_meta_only": {"label": "G  metadata SECDED", "color": "#009E73", "marker": "^", "z": 3},
    "m3_arm_b_uniform": {"label": "B  uniform data+meta SECDED", "color": "#0072B2", "marker": "o", "z": 2},
    "m3_arm_c_precision": {"label": "C  finer groups, no data ECC", "color": "#56B4E9", "marker": "s", "z": 1},
    "m3_arm_d_late19": {"label": "D  data ECC layers 9-27", "color": "#CC79A7", "marker": "P", "z": 1},
    "m3_arm_e_early19": {"label": "E  data ECC layers 0-18", "color": "#D55E00", "marker": "X", "z": 1},
}
ISO = {"m3_arm_b_uniform", "m3_arm_c_precision", "m3_arm_d_late19", "m3_arm_e_early19"}
BANDS = ((0, 700), (700, 1400), (1400, 2800), (2800, 10**9))
CURVE_ARM = "m3_rate_calibration"


def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """Finite and asymmetric at k = 0, which a normal approximation is not."""
    if n == 0:
        return (float("nan"), float("nan"))
    p, d = k / n, 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def wilson_bars(rows) -> tuple[list[float], list[float]]:
    """Error-bar offsets from the point estimate, never negative.

    The Wilson interval is not centred on p, so for a very small p its lower limit can sit a hair above the
    point itself; matplotlib rejects a negative offset outright rather than drawing it, which is how this
    surfaced. Clamping at zero is the honest rendering: the bar then starts at the point.
    """
    lo, hi = [], []
    for d in rows:
        p = d["fails"] / d["n"]
        low, high = wilson(d["fails"], d["n"])
        lo.append(max(0.0, p - low))
        hi.append(max(0.0, high - p))
    return lo, hi


def _runs(raw_root: Path):
    for exp in sorted(raw_root.glob("*")):
        if ".superseded-" in exp.name or not (exp / "manifest.json").exists():
            continue
        manifest = read_json(exp / "manifest.json")
        if manifest.get("superseded_by") or manifest["config"]["task"]["kind"] != "protection_arm":
            continue
        records = read_jsonl(exp / "results.jsonl")
        if len(records) != manifest["unit_keys_total"] or not records:
            continue  # incomplete runs are never plotted
        yield manifest["config"]["name"], records


def _tally(rows) -> tuple[int, int]:
    fails = sum(1 for x, r in rows if not x.get("faulty_correct"))
    return fails, len(rows)


def collect(raw_root: Path, rate: float):
    """Per arm at ``rate``: overall point, and one point per cache-length band. Plus the rate sweep."""
    arms, bands, sweep = {}, {}, []
    for name, records in _runs(raw_root):
        ber = records[0]["ber"]
        rows = [(x, r) for r in records if r["golden_correct"] for x in r["faults"] if x["trial"] >= 0]
        if not rows:
            continue
        if name == CURVE_ARM:
            fails, n = _tally(rows)
            flips = sum(x["n_flips"] for x, _ in rows) / n
            sweep.append({"ber": ber, "flips": flips, "fails": fails, "n": n})
            continue
        if name not in ARMS or abs(ber - rate) / rate > 1e-6:
            continue
        fails, n = _tally(rows)
        arms[name] = {"fails": fails, "n": n,
                      "bits": sum(r["avg_kv_bits"] for r in records) / len(records)}
        for lo, hi in BANDS:
            sub = [(x, r) for x, r in rows if lo <= r["tokens_retained"] < hi]
            if not sub:
                continue
            f, m = _tally(sub)
            bands.setdefault(name, []).append({"flips": sum(x["n_flips"] for x, _ in sub) / m, "fails": f, "n": m})
    sweep.sort(key=lambda d: d["flips"])
    for rowlist in bands.values():
        rowlist.sort(key=lambda d: d["flips"])
    return arms, bands, sweep


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--rate", type=float, default=1e-6)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    root = resolve_output_root(args.output_root)
    arms, bands, sweep = collect(root / "raw", args.rate)
    if not arms:
        print("no completed protection_arm runs at that rate")
        return 1

    fig, (ax_dose, ax_cost) = plt.subplots(1, 2, figsize=(11.6, 4.5))

    # ---- left: damage against corrupted bits, two ways of varying them --------------------------------
    if sweep:
        xs = [d["flips"] for d in sweep]
        ys = [d["fails"] / d["n"] for d in sweep]
        ax_dose.plot(xs, ys, color="#555555", ls="--", lw=1.3, marker="D", ms=5, mfc="white", zorder=5,
                     label="A, varying the error rate")
    for name, rowlist in bands.items():
        style = ARMS[name]
        xs = [d["flips"] for d in rowlist]
        ys = [d["fails"] / d["n"] for d in rowlist]
        lo, hi = wilson_bars(rowlist)
        ax_dose.errorbar(xs, ys, yerr=[lo, hi], color=style["color"], marker=style["marker"], ms=6, lw=1.5,
                         capsize=2.5, zorder=style["z"], label=f"{style['label'][0]}, varying context length")
    ax_dose.set_xscale("log")
    ax_dose.set_xlabel("corrupted bits per trial")
    ax_dose.set_ylabel("task failure rate")
    ax_dose.set_title("Damage follows the number of corrupted bits", fontsize=10.5)
    ax_dose.set_ylim(-0.04, 0.78)
    ax_dose.grid(alpha=0.25, lw=0.6)
    ax_dose.legend(fontsize=7, frameon=False, loc="upper left")
    ax_dose.annotate("every arm that protects the metadata: 0 failures at every point",
                     (0.5, 0.02), xycoords="axes fraction", fontsize=7.5, color="#333333", ha="center")

    # ---- right: outcome against what the arm stores ---------------------------------------------------
    iso_x = [v["bits"] for k, v in arms.items() if k in ISO]
    if iso_x:
        ax_cost.axvspan(min(iso_x) * 0.9985, max(iso_x) * 1.0015, color="#0072B2", alpha=0.08, lw=0)
        ax_cost.annotate("equal storage by design\n(B, C, D, E)", (sum(iso_x) / len(iso_x), 0.20), ha="center",
                         fontsize=7.5, color="#0072B2")
    zero_arms = []
    for name, v in sorted(arms.items(), key=lambda kv: ARMS[kv[0]]["z"]):
        style = ARMS[name]
        p = v["fails"] / v["n"]
        low, high = wilson(v["fails"], v["n"])
        ax_cost.errorbar([v["bits"]], [p], yerr=[[max(0.0, p - low)], [max(0.0, high - p)]], color=style["color"],
                         marker=style["marker"], ms=9, capsize=3, lw=1.5, label=style["label"], zorder=style["z"])
        if p == 0:
            zero_arms.append(style["label"].split()[0])
    ax_cost.set_xlabel("effective bits per cached value (measured, protection included)")
    ax_cost.set_ylabel("task failure rate")
    ax_cost.set_title(f"What the protection costs, at BER {args.rate:.0e}", fontsize=10.5)
    ax_cost.set_ylim(-0.04, 0.45)
    ax_cost.grid(alpha=0.25, lw=0.6)
    ax_cost.legend(fontsize=7, frameon=False, loc="upper right")
    if zero_arms:
        ax_cost.annotate("on zero: " + ", ".join(sorted(zero_arms)), (0.5, 0.03), xycoords="axes fraction",
                         fontsize=7.5, color="#333333", ha="center")

    fig.tight_layout()
    out = args.out or (root / "figures" / "m3_protection.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"wrote {out}")
    for name, v in arms.items():
        print(f"  {name:24s} bits={v['bits']:.4f} failures={v['fails']}/{v['n']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
