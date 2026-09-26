"""C4: do token-level metrics under-estimate agent failures? Both metrics, one vehicle, one axis.

C4 has been `PROPOSED` since M0 and open after M6. Its evidence was always D16's **"+29% perplexity alongside −71%
probe accuracy"** — but that perplexity was measured on **WikiText** and that accuracy on the **probe**: two corpora
and two vehicles, so the gap could have been a domain difference rather than a metric one. `docs/claims.md` records
what closing it needs, and it is exactly this: perplexity over the tool catalogue, beside task success on the tool
calls, under the same perturbation.

The perturbation is **quantization precision** (bf16 / 4-bit / 2-bit), because it is the axis D16 used and the only
one both metrics can read. The M6 retention arms vary the *scorer* at a fixed 4 bits, which is a different axis and
cannot be set against a precision sweep — mixing them would reproduce the very confound C4 needs removed.

    python scripts/report_c4.py

**The decision rule, from the matrix: support is a weak rank correlation between token-level damage and task
failure.** Made concrete here as the ratio of relative damages: if a precision step costs far more task success than
perplexity, the token-level metric under-estimates. The bf16 row doubles as a control — it must reproduce
`m6_bfcl_golden`'s 105/120, and a mismatch there invalidates the comparison rather than being a detail.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

from raq_kv.analysis.stats import wilson
from raq_kv.utils.io import read_json, read_jsonl
from raq_kv.utils.paths import resolve_output_root

PRECISIONS = ("bf16", "4b", "2b")


def _bits(config: dict) -> str:
    q = config.get("quantization")
    return "bf16" if q is None else f"{q['keys']['bits']}b"


def collect(raw_root: Path, name: str) -> dict[str, list[dict]]:
    """Every live complete variant of a sweep config, keyed by its precision."""
    out: dict[str, list[dict]] = {}
    for exp in sorted(raw_root.glob("*")):
        if ".superseded-" in exp.name or not (exp / "manifest.json").exists():
            continue
        manifest = read_json(exp / "manifest.json")
        if manifest.get("superseded_by") or manifest["config"]["name"] != name:
            continue
        records = read_jsonl(exp / "results.jsonl")
        if len(records) == manifest["unit_keys_total"]:
            out[_bits(manifest["config"])] = records
    return out


def corpus_ppl(records: list[dict]) -> tuple[float, int]:
    """Corpus-level perplexity: exp(total NLL / total tokens).

    Not the mean of per-chunk perplexities, which weights a short chunk like a long one and is not a perplexity of
    anything. Every chunk here is the same length, so the two coincide -- stated because they usually do not.
    """
    nll = sum(r["nll_sum"] for r in records)
    tok = sum(r["n_scored_tokens"] for r in records)
    return float(np.exp(nll / tok)), tok


def task_rate(records: list[dict]) -> tuple[int, int]:
    """Correct tool calls out of attempted, by the vehicle's own AST scorer."""
    return sum(1 for r in records if r.get("correct")), len(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ppl-arm", default="d66_ppl_bfcl_qwen")
    parser.add_argument("--task-arm", default="d66_task_bfcl_qwen")
    parser.add_argument("--control-arm", default="m6_bfcl_golden")
    parser.add_argument("--output-root", default="outputs")
    args = parser.parse_args()
    raw_root = resolve_output_root(args.output_root) / "raw"

    ppl = collect(raw_root, args.ppl_arm)
    task = collect(raw_root, args.task_arm)
    missing = [p for p in PRECISIONS if p not in ppl or p not in task]
    if missing:
        raise SystemExit(f"incomplete: {missing} absent from {args.ppl_arm} or {args.task_arm}")

    print(f"C4 on one vehicle and one axis: {args.ppl_arm} (perplexity over the catalogue) against "
          f"{args.task_arm} (tool calls)\n")

    # The control first: if the bf16 task variant does not reproduce the golden arm, nothing below is trustworthy.
    control = collect(raw_root, args.control_arm)
    if "bf16" in control:
        ck, cn = task_rate(control["bf16"])
        bk, bn = task_rate(task["bf16"])
        agree = (ck, cn) == (bk, bn)
        print(f"control: bf16 variant {bk}/{bn} against {args.control_arm}'s {ck}/{cn} -- "
              f"{'reproduced' if agree else '*** MISMATCH, the comparison below is not trustworthy ***'}\n")
    else:
        print(f"control: {args.control_arm} not found, so the bf16 point is unverified\n")

    base_ppl, tok = corpus_ppl(ppl["bf16"])
    base_k, base_n = task_rate(task["bf16"])
    base_acc = base_k / base_n

    print(f"{'precision':11s}{'perplexity':>12s}{'ppl delta':>11s}{'task correct':>14s}{'accuracy':>10s}"
          f"{'acc delta':>11s}{'Wilson 95%':>20s}")
    rows: dict[str, dict] = {}
    for p in PRECISIONS:
        pv, _ = corpus_ppl(ppl[p])
        k, n = task_rate(task[p])
        acc = k / n
        lo, hi = wilson(k, n)
        d_ppl = pv / base_ppl - 1.0
        d_acc = acc / base_acc - 1.0
        rows[p] = {"ppl": pv, "ppl_delta": d_ppl, "correct": k, "n": n, "accuracy": acc,
                   "acc_delta": d_acc, "ci": (lo, hi)}
        # `:+.2%` scales the fraction; `:+.2f` followed by a literal % does not, and the first version of this table
        # printed "+0.13%" for a 13.41% perplexity rise while the contrast lines below printed it correctly. A
        # hundredfold error in the headline table of the claim this script exists to settle.
        print(f"{p:11s}{pv:12.4f}{d_ppl:+11.2%}{f'{k}/{n}':>14s}{acc:10.4f}{d_acc:+11.2%}"
              f"   [{lo:.4f}, {hi:.4f}]")
    print(f"\n  perplexity over {tok} scored catalogue tokens per variant; accuracy over {base_n} tool calls")

    print("\nthe C4 comparison -- relative task damage against relative token-level damage\n")
    out: dict = {"rows": rows, "contrasts": {}}
    for p in PRECISIONS[1:]:
        r = rows[p]
        # Fisher against the bf16 row: the task side needs a test, not just a delta.
        pvalue = float(stats.fisher_exact([[r["correct"], r["n"] - r["correct"]],
                                           [base_k, base_n - base_k]]).pvalue)
        ratio = abs(r["acc_delta"]) / abs(r["ppl_delta"]) if r["ppl_delta"] else float("inf")
        out["contrasts"][p] = {"fisher_p": pvalue, "damage_ratio": ratio,
                               "acc_delta": r["acc_delta"], "ppl_delta": r["ppl_delta"]}
        print(f"  {p:5s} perplexity {r['ppl_delta']:+.2%}   accuracy {r['acc_delta']:+.2%}   "
              f"Fisher p = {pvalue:.3g}")
        if abs(r["ppl_delta"]) < 0.005:
            print(f"        the token-level metric barely moved ({r['ppl_delta']:+.2%}), so a ratio is not "
                  f"meaningful here; what matters is whether the task moved while it did not")
        else:
            print(f"        task damage is {ratio:.1f}x the perplexity damage in relative terms")

    print("\n  Reading: C4 is supported where the task moves and perplexity does not, or moves far less. It is"
          "\n  undercut where the two track each other. Both directions are reportable and the sign is what the"
          "\n  claim is about, not the magnitude -- D16's original pairing was +29% perplexity against -71%"
          "\n  accuracy, and that was across two vehicles; this is the same question inside one.")

    dest = resolve_output_root(args.output_root) / "reports" / "c4_metric_mismatch.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
