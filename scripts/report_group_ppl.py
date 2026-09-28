"""The quality side of the key-granularity choice: cache-conditioned perplexity paired over chunks.

D76 established that **reliability does not constrain the key group size** anywhere in 32-512 -- delta is
span-independent (alpha ~ 0, both quadrature and coherent excluded) and the break rate is flat. That leaves
quantization quality as the only constraint, and H2f's recommendation conditional on one number: the perplexity cost of
moving from 128 to 256.

    python scripts/report_group_ppl.py --name d77a_group_ppl_qwen

**This is a bound, by design, and the power calculation was done before the run.** The paired difference has
sd = 0.010638 on the existing 40-chunk arms, so a 95% half-width smaller than the observed +0.00133 effect needs
**245 chunks**; WikiText-2 raw test provides **145** (299,078 tokens at 2,049 per chunk). At n = 145 the half-width is
0.00173, still larger than the effect. So the sign cannot be resolved on this split and this run is not for resolving
it -- it tightens the bound, and the allocator decision is made against the bound, which is what D30/D44 requires of a
non-significant difference anyway.

Pairing is over chunk ids, because the same chunk under two group sizes is the same text and the between-chunk variance
(perplexity 2 to 40 across WikiText chunks) dwarfs the effect. Unpaired means of these arms would not resolve a 0.13%
difference at any n this corpus can supply.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from scipy import stats

from raq_kv.analysis.arms import sweep_runs
from raq_kv.analysis.contrasts import contrast_rng
from raq_kv.utils.paths import resolve_output_root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="d77a_group_ppl_qwen")
    parser.add_argument("--baseline", type=int, default=128, help="the group size every contrast is measured against")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    raw_root = resolve_output_root(args.output_root) / "raw"

    rungs: dict[int, dict] = {}
    for exp, manifest, records in sweep_runs(raw_root, args.name):
        g = int(manifest["config"]["quantization"]["keys"]["group_size"])
        rungs[g] = {"nll": {r["sample_id"]: float(r["nll_mean"]) for r in records},
                    "bits": records[0]["avg_kv_bits"], "exp": exp.name,
                    "sqnr": ((records[0].get("quantization") or {}).get("keys") or {}).get("sqnr_db_mean")}
    if args.baseline not in rungs:
        raise SystemExit(f"baseline group {args.baseline} not among the rungs {sorted(rungs)}")

    out: dict = {"rungs": {}, "contrasts": {}}
    print(f"key group size against cache-conditioned perplexity -- {args.name}\n")
    print(f"{'group':>6s}{'chunks':>8s}{'perplexity':>12s}{'key SQNR':>10s}{'bits/val':>10s}")
    for g in sorted(rungs):
        r = rungs[g]
        ppl = float(np.exp(np.mean(list(r["nll"].values()))))
        print(f"{g:6d}{len(r['nll']):8d}{ppl:12.4f}"
              f"{(r['sqnr'] if r['sqnr'] is not None else float('nan')):10.2f}{r['bits']:10.4f}")
        out["rungs"][g] = {"n": len(r["nll"]), "ppl": ppl, "key_sqnr_db": r["sqnr"], "avg_kv_bits": r["bits"],
                           "exp": r["exp"]}

    base = rungs[args.baseline]
    print(f"\npaired against g={args.baseline}, over the chunks both arms scored\n")
    print(f"{'contrast':>18s}{'n':>5s}{'d NLL':>11s}{'95% CI':>24s}{'Wilcoxon p':>12s}{'d ppl %':>10s}  D31")
    for g in sorted(rungs):
        if g == args.baseline:
            continue
        shared = sorted(set(base["nll"]) & set(rungs[g]["nll"]))
        d = np.array([rungs[g]["nll"][k] - base["nll"][k] for k in shared])
        rng = contrast_rng(f"g{args.baseline}", f"g{g}", args.seed)
        boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10_000)]
        lo, hi = (float(v) for v in np.percentile(boot, [2.5, 97.5]))
        w = stats.wilcoxon(d)
        pa = float(np.exp(np.mean([base["nll"][k] for k in shared])))
        pb = float(np.exp(np.mean([rungs[g]["nll"][k] for k in shared])))
        # bool(): numpy's float64 comparison yields numpy.bool_, which json.dumps refuses.
        meets = bool(w.pvalue < 0.05 and lo * hi > 0)
        print(f"{f'g{args.baseline} -> g{g}':>18s}{len(d):5d}{d.mean():+11.5f}   [{lo:+.5f}, {hi:+.5f}]"
              f"{w.pvalue:12.2e}{100 * (pb / pa - 1):+10.3f}  {'meets' if meets else 'FAILS'}")
        # The bound is the deliverable when the contrast fails: the cost is at most the interval's upper end.
        note = (f"cost is at most {hi:+.5f} nats = {100 * (np.exp(hi) - 1):+.3f}% perplexity"
                if not meets else f"resolved at {100 * (pb / pa - 1):+.3f}% perplexity")
        print(f"{'':18s}  -> {note}")
        out["contrasts"][f"{args.baseline}->{g}"] = {
            "n": len(d), "d_nll": float(d.mean()), "ci": (lo, hi), "p": float(w.pvalue),
            "d_ppl_pct": 100 * (pb / pa - 1), "meets_d31": meets,
            "bound_ppl_pct": None if meets else 100 * (float(np.exp(hi)) - 1)}

    # The pre-registered power statement, recomputed from THIS run's own scatter so a reader can check it.
    g = min((x for x in rungs if x > args.baseline), default=None)
    if g is not None:
        shared = sorted(set(base["nll"]) & set(rungs[g]["nll"]))
        d = np.array([rungs[g]["nll"][k] - base["nll"][k] for k in shared])
        sd, eff = d.std(ddof=1), abs(d.mean())
        if eff > 0:
            need = int(np.ceil((1.96 * sd / eff) ** 2))
            print(f"\n  power, from this run's own scatter: sd {sd:.6f} on the g{args.baseline}->g{g} contrast, so a "
                  f"half-width\n  below the observed effect ({eff:.6f}) needs {need} chunks against the "
                  f"{len(d)} this split supplies.")
            out["power"] = {"sd": float(sd), "effect": float(eff), "chunks_needed": need, "chunks_available": len(d)}

    dest = resolve_output_root(args.output_root) / "reports" / f"{args.name}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
