"""Figure: what protection bits buy, and where they should go (M3).

Two panels, one story.

**Left — the unprotected cache's fault tolerance.** Task failure against bit error rate on the unprotected arm,
log x, with Wilson intervals. This panel exists because the operating rate of every other measurement is read off
it, and because the curve is nearly linear in flip count over two decades, which is the evidence that failures are
independent single-flip events rather than accumulation.

**Right — where the same bits go.** Task failure per arm at the campaign rate, against each arm's *measured*
effective bits per cached value. Arms that hold equal storage sit in one vertical band, so the reader sees the
equal-budget comparison as geometry rather than as a claim: the unprotected arm sits left of the band (it is
cheaper, and labelled a cost reference), the metadata-only arm sits just right of it.

Identity is never carried by colour alone: every arm has its own marker, and the equal-budget band is drawn as a
shaded region with a text label. Colours are the Okabe-Ito CVD-safe qualitative set. Error bars are Wilson score
intervals, which behave at zero counts — three arms are expected to sit exactly at zero, where a normal
approximation would draw an interval of width zero and imply a precision the data does not have.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from raq_kv.utils.io import read_json, read_jsonl  # noqa: E402
from raq_kv.utils.paths import resolve_output_root  # noqa: E402

# Okabe-Ito, plus grey for the cost-reference arm.
ARMS = {
    "m3_arm_a_unprot": {"label": "A  no protection", "color": "#555555", "marker": "D"},
    "m3_arm_g_meta_only": {"label": "G  metadata ECC only", "color": "#E69F00", "marker": "v"},
    "m3_arm_b_uniform": {"label": "B  uniform data+meta ECC", "color": "#0072B2", "marker": "o"},
    "m3_arm_c_precision": {"label": "C  finer groups, no data ECC", "color": "#009E73", "marker": "^"},
    "m3_arm_d_late19": {"label": "D  data ECC on layers 9-27", "color": "#CC79A7", "marker": "s"},
    "m3_arm_e_early19": {"label": "E  data ECC on layers 0-18", "color": "#D55E00", "marker": "P"},
}


def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score interval: finite and asymmetric at k = 0, which a normal approximation is not."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def collect(raw_root: Path) -> dict[str, list[dict]]:
    """Per config name, one row per (bit error rate) with counts, bits/value and flip count."""
    out: dict[str, list[dict]] = {}
    for exp in sorted(raw_root.glob("*")):
        if ".superseded-" in exp.name or not (exp / "manifest.json").exists():
            continue
        manifest = read_json(exp / "manifest.json")
        if manifest.get("superseded_by") or manifest["config"]["task"]["kind"] != "protection_arm":
            continue
        records = read_jsonl(exp / "results.jsonl")
        if len(records) != manifest["unit_keys_total"]:
            continue  # incomplete runs are never plotted
        fails = trials = 0
        bits: list[float] = []
        flips: list[float] = []
        for r in records:
            if not r["golden_correct"]:
                continue
            bits.append(r["avg_kv_bits"])
            for x in r["faults"]:
                if x["trial"] < 0:
                    continue
                trials += 1
                flips.append(x.get("n_flips", 0))
                fails += 0 if x.get("faulty_correct") else 1
        if not trials:
            continue
        name = manifest["config"]["name"]
        out.setdefault(name, []).append({
            "ber": records[0]["ber"], "fails": fails, "trials": trials,
            "bits_per_value": sum(bits) / len(bits), "flips": sum(flips) / len(flips),
        })
    for rows in out.values():
        rows.sort(key=lambda r: r["ber"])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--curve-arm", default="m3_rate_calibration",
                        help="Config whose rate sweep is the left panel")
    parser.add_argument("--rate", type=float, default=1e-6, help="Campaign rate shown in the right panel")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    root = resolve_output_root(args.output_root)
    data = collect(root / "raw")
    if not data:
        print("no completed protection_arm runs to plot")
        return 1

    fig, (ax_curve, ax_arms) = plt.subplots(1, 2, figsize=(11.0, 4.3))

    # ---- left: dose-response of the unprotected cache -------------------------------------------------
    rows = data.get(args.curve_arm, [])
    if rows:
        xs = [r["ber"] for r in rows]
        ys = [r["fails"] / r["trials"] for r in rows]
        lo = [y - wilson(r["fails"], r["trials"])[0] for y, r in zip(ys, rows, strict=True)]
        hi = [wilson(r["fails"], r["trials"])[1] - y for y, r in zip(ys, rows, strict=True)]
        ax_curve.errorbar(xs, ys, yerr=[lo, hi], color="#555555", marker="D", ms=5, lw=1.6, capsize=3,
                          label="no protection")
        for r in rows:
            ax_curve.annotate(f"{r['flips']:.0f}", (r["ber"], r["fails"] / r["trials"]),
                              textcoords="offset points", xytext=(6, -11), fontsize=7.5, color="#333333")
        ax_curve.axvline(args.rate, color="#0072B2", ls=":", lw=1.2)
        ax_curve.annotate("campaign rate", (args.rate, 0.93), rotation=90, fontsize=7.5, color="#0072B2",
                          textcoords="offset points", xytext=(-11, 0), va="top")
    ax_curve.set_xscale("log")
    ax_curve.set_xlabel("bit error rate over stored representation")
    ax_curve.set_ylabel("task failure rate")
    ax_curve.set_title("Unprotected cache: what a stored bit costs", fontsize=10.5)
    ax_curve.set_ylim(-0.03, 1.0)
    ax_curve.grid(alpha=0.25, lw=0.6)
    ax_curve.legend(fontsize=8, frameon=False)
    ax_curve.text(0.02, 0.96, "labels: mean flips per trial", transform=ax_curve.transAxes, fontsize=7.5,
                  color="#333333", va="top")

    # ---- right: arms at the campaign rate ------------------------------------------------------------
    points = []
    for name, rows in data.items():
        if name not in ARMS:
            continue
        for r in rows:
            if abs(r["ber"] - args.rate) / args.rate < 1e-6:
                points.append((name, r))
    iso = [r["bits_per_value"] for n, r in points if n not in ("m3_arm_a_unprot", "m3_arm_g_meta_only")]
    if iso:
        ax_arms.axvspan(min(iso) * 0.999, max(iso) * 1.001, color="#0072B2", alpha=0.07, lw=0)
        ax_arms.annotate("equal storage\n(spread < 0.2%)", (sum(iso) / len(iso), 0.30), ha="center",
                         fontsize=7.5, color="#0072B2")
    for name, r in points:
        style = ARMS[name]
        p = r["fails"] / r["trials"]
        low, high = wilson(r["fails"], r["trials"])
        ax_arms.errorbar([r["bits_per_value"]], [p], yerr=[[p - low], [high - p]], color=style["color"],
                         marker=style["marker"], ms=8, capsize=3, lw=1.4, label=style["label"])
    ax_arms.set_xlabel("effective bits per cached value (measured, ECC included)")
    ax_arms.set_ylabel("task failure rate")
    ax_arms.set_title(f"Where the bits go, at BER {args.rate:.0e}", fontsize=10.5)
    ax_arms.set_ylim(-0.03, 0.45)
    ax_arms.grid(alpha=0.25, lw=0.6)
    ax_arms.legend(fontsize=7.5, frameon=False, loc="upper right")

    fig.tight_layout()
    out = args.out or (root / "figures" / "m3_protection.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"wrote {out} and {out.with_suffix('.pdf')}")
    print(json.dumps({n: {"ber": r["ber"], "fails": r["fails"], "trials": r["trials"],
                          "bits": round(r["bits_per_value"], 4)} for n, r in points}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
