"""Figure: cache-conditioned quality versus KV precision (M1 sweep).

Two panels sharing one x axis (effective bits per cached value, metadata included):
left = corpus perplexity, right = paired per-token NLL increase with bootstrap CIs. Two panels rather than one
plot with two y scales, and a log y axis because the catastrophic variants are four orders of magnitude away.

Identity is never carried by colour alone: each granularity family also has its own marker and line style, which
keeps the figure readable in greyscale print and under colour-vision deficiency. Colours are the Okabe-Ito
qualitative set (published as CVD-safe); the skill's validator could not be run here because node is unavailable,
so the secondary encoding is doing the accessibility work.
"""

from __future__ import annotations

import argparse
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from raq_kv.utils.paths import resolve_output_root  # noqa: E402

# Okabe-Ito: blue, vermillion, bluish green, reddish purple.
FAMILIES = {
    "per-channel keys": {"color": "#0072B2", "marker": "o", "ls": "-"},
    "per-token keys": {"color": "#D55E00", "marker": "s", "ls": "--"},
    "grouped channels": {"color": "#009E73", "marker": "^", "ls": "-."},
    "reference": {"color": "#555555", "marker": "D", "ls": ":"},
}


def _family(variant: str) -> str:
    if variant.startswith("bf16"):
        return "reference"
    if "chan" in variant.split("·")[0]:
        return "per-channel keys"
    if "g" in variant.split("·")[0].replace("bf16", ""):
        return "grouped channels"
    return "per-token keys"


def _short(variant: str) -> str:
    return variant.replace("K ", "K").replace(" · V ", "/V").replace("b/", "b ").replace(" · resid", "+r")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", default="outputs")
    args = parser.parse_args()
    root = resolve_output_root(args.output_root)
    df = pd.read_csv(root / "csv" / "m1_kv_quant_ppl.csv")
    df["family"] = df["variant"].map(_family)
    baseline = df[df["family"] == "reference"]
    base_ppl = float(baseline["corpus_ppl"].iloc[0]) if not baseline.empty else None

    fig, (ax_ppl, ax_nll) = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True)
    for family, style in FAMILIES.items():
        sub = df[df["family"] == family].sort_values("effective_bits_per_value")
        if sub.empty:
            continue
        ax_ppl.plot(sub["effective_bits_per_value"], sub["corpus_ppl"], style["ls"], color=style["color"],
                    marker=style["marker"], markersize=7, linewidth=2, label=family, zorder=3)
        positive = sub[sub["delta_nll_per_token"] > 0]
        if not positive.empty:
            err = [positive["delta_nll_per_token"] - positive["delta_nll_ci95_low"],
                   positive["delta_nll_ci95_high"] - positive["delta_nll_per_token"]]
            ax_nll.errorbar(positive["effective_bits_per_value"], positive["delta_nll_per_token"], yerr=err,
                            fmt=style["marker"], color=style["color"], markersize=7, linewidth=2, capsize=3,
                            linestyle=style["ls"], label=family, zorder=3)

    if base_ppl is not None:
        ax_ppl.axhline(base_ppl, color="#555555", linestyle=":", linewidth=1.5, zorder=2)
        ax_ppl.annotate(f"bf16 baseline {base_ppl:.2f}", xy=(12.5, base_ppl), xytext=(0, 7),
                        textcoords="offset points", fontsize=9, color="#555555", ha="center")

    # Direct labels only on the points that carry the argument, with hand-placed offsets (no collisions).
    labels = {
        "K 2b/token · V 2b/token": ("2-bit per-token keys", (8, -4)),
        "K 4b/token · V 4b/token": ("4-bit per-token keys", (8, -4)),
        "K 2b/g32 · V 2b/g32": ("2-bit grouped", (-10, 10)),
        "K 2b/chan · V 2b/token": ("2-bit per-channel keys", (2, 12)),
        "K 4b/chan · V 4b/token": ("4-bit per-channel keys", (12, -14)),
        "K 4b/token/sym · V 4b/token/sym": ("4-bit per-token, symmetric", (10, 6)),
    }
    for _, r in df.iterrows():
        entry = labels.get(r["variant"])
        if entry:
            text, offset = entry
            ax_ppl.annotate(text, xy=(r["effective_bits_per_value"], r["corpus_ppl"]), xytext=offset,
                            textcoords="offset points", fontsize=8.5, color="#333333")

    ax_ppl.set_yscale("log")
    ax_ppl.set_ylabel("cache-conditioned perplexity (log)")
    ax_ppl.set_title("Quality", loc="left", fontsize=11)
    ax_nll.set_yscale("log")
    ax_nll.set_ylabel("paired ΔNLL per token (nats, log)")
    ax_nll.set_title("Paired degradation vs. bf16 (95% CI)", loc="left", fontsize=11)
    for ax in (ax_ppl, ax_nll):
        ax.set_xlabel("effective bits per cached value (codes + fp16 scales)")
        ax.grid(True, which="major", color="#DDDDDD", linewidth=0.8, zorder=0)
        ax.grid(True, which="minor", color="#EEEEEE", linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    ax_ppl.set_xlim(1.5, 17.0)
    ax_ppl.legend(frameon=False, fontsize=9, loc="upper right", bbox_to_anchor=(1.0, 1.02))
    fig.suptitle("KV-cache precision: the granularity of key quantization dominates the bit width",
                 fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(figures / f"m1_quality_vs_bits.{ext}", dpi=200, bbox_inches="tight")
    print(f"wrote {figures / 'm1_quality_vs_bits.png'} and .pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
