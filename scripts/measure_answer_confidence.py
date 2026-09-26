"""How confidently does each model hold its answer, and does making the task harder change that?

Two jobs, one measurement.

**D62's scale control.** Branch B measures the fault-free margin on the first answer token and finds
**Qwen3-1.7B at 17.07 nats against Llama-3.2-3B at 3.27**, which explains D59's hazard difference without any appeal
to quantization. But a margin in nats is not scale-free: if Qwen's output head simply produces log-probabilities on
a wider spread, every one of its margins is larger for reasons that have nothing to do with confidence, a
perturbation of its cache would move them further too, and the *ratio* that decides whether an answer flips would be
unchanged. So the spread is measured beside the margin, and `margin / spread` is the scale-free form.

**D63's difficulty ladder.** D62 stated its limit: saturation is a property of model x task, so on a harder task the
whole difference could vanish or reverse. Context length was measured and is **not** the dial -- Qwen has no sample
below 1 nat at any length, and eight times the context costs it about 6 nats. Difficulty is therefore varied the one
way this vehicle can vary it while holding everything else fixed: the number of competing **same-type** distractor
facts, across `raq_probe_v0_d{2,10,20}`. Generator seed, filler corpus and revision, fact types, depths, answer
format and scorer are identical across rungs; only the number of competing facts moves.

    python scripts/measure_answer_confidence.py --datasets raq_probe_v0_d2,raq_probe_v0_d10,raq_probe_v0_d20

**Accuracy is the anchor and it can void the comparison.** A rung on which accuracy does not move is a rung that did
not make the task harder, and a flat margin there says nothing about saturation. It is printed beside every cell,
the same role the AUC plays in D62.

A structural measurement of the models rather than an experiment arm, so it carries no exp_id; it is deterministic
given the configs and the seed, and the script is committed.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import numpy as np
import torch

from raq_kv.config.loader import load_yaml
from raq_kv.config.schema import ModelConfig, QuantizationConfig, QuantSpec, RAQProbeConfig
from raq_kv.data.raq_probe import build_raq_probe
from raq_kv.evaluation.protocol import greedy_decode, prefill, score_through_cache
from raq_kv.evaluation.scoring import SCORERS
from raq_kv.models.loader import load_model, load_tokenizer
from raq_kv.quantization.apply import quantize_to_codes
from raq_kv.utils.paths import configs_dir, resolve_output_root

# The D62 arms' quantization, restated here rather than imported from a config so this script cannot silently
# measure a different cache than the arms did if a config is edited.
QUANT = QuantizationConfig(keys=QuantSpec(bits=4, axis="group_token", group_size=128),
                           values=QuantSpec(bits=4, axis="per_token"))

SHAPE_KEYS = ("margin", "logprob_spread", "margin_over_spread", "entropy", "top1_prob")


def shape_of(logprobs: torch.Tensor) -> dict[str, float]:
    """Summaries of one next-token distribution, in float64 log-space.

    `entropy` and `top1_prob` saturate at float64 for a distribution sitting at p = 1, which is exactly the
    condition being looked for -- so they are floor and ceiling readings there, not measurements. `margin` is a
    difference of two finite log-probabilities and does not saturate, which is why it stays the quantity to read.
    """
    lp = logprobs.double()
    order = torch.topk(lp, 2).values
    p = lp.exp()
    live = p > 0
    entropy = float(-(p[live] * lp[live]).sum())
    spread = float(lp.std())
    margin = float(order[0] - order[1])
    return {"margin": margin, "logprob_spread": spread, "margin_over_spread": margin / spread if spread else 0.0,
            "entropy": entropy, "top1_prob": float(p.max()), "top1_logprob": float(order[0]),
            "vocab": int(lp.numel())}


def measure(stem: str, probe: RAQProbeConfig, n_samples: int, dataset: str) -> list[dict[str, Any]]:
    """One row per CORRECTLY answered sample, each carrying the number attempted so accuracy is recoverable."""
    cfg = ModelConfig.model_validate(load_yaml(configs_dir() / "models" / f"{stem}.yaml")["model"])
    tokenizer = load_tokenizer(cfg)
    # Spread across the length range rather than taking the head: the rungs pin lengths to [2048, 4096] and the
    # first samples are all the shorter one, so a head slice would compare 2048-token cells against a mix.
    everything = build_raq_probe(probe, tokenizer, dict(cfg.chat_template_kwargs))
    if n_samples < len(everything):
        step = len(everything) / n_samples
        samples = [everything[int(i * step)] for i in range(n_samples)]
    else:
        samples = everything
    lm = load_model(cfg)
    eos = set(lm.eos_token_ids)
    rows: list[dict[str, Any]] = []
    for sample in samples:
        ids = sample.prompt_ids.to(lm.device).unsqueeze(0)
        context_ids, query_ids = ids[:, : sample.cache_len], ids[:, sample.cache_len :]
        base = quantize_to_codes(prefill(lm.model, context_ids), QUANT).to_snapshot()
        # Only samples the arm would have counted: D62 pairs over samples both families answer correctly with no
        # faults, so a margin measured on a sample the model gets wrong is not the quantity branch B is about.
        text = lm.tokenizer.decode(greedy_decode(lm.model, query_ids, sample.max_new_tokens, snapshot=base,
                                                 eos_token_ids=eos).tokens, skip_special_tokens=True)
        if SCORERS[sample.scorer](text, sample.reference)["correct"]:
            lp = score_through_cache(lm.model, base, query_ids, query_ids, return_logprobs=True,
                                     logprob_dtype=torch.float64).logprobs[-1]
            rows.append({"model": cfg.short_name, "sample_id": sample.sample_id, "dataset": dataset,
                         "n_distractors": sample.meta.get("n_distractors"),
                         "cache_len": sample.meta["cache_len"], "n_attempted": len(samples), **shape_of(lp)})
        del base
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default="qwen3_1p7b_eager,llama3p2_3b_eager")
    parser.add_argument("--datasets", default="raq_probe_v0",
                        help="Comma-separated dataset config stems; D63's ladder is "
                             "raq_probe_v0_d2,raq_probe_v0_d10,raq_probe_v0_d20")
    parser.add_argument("--n-samples", type=int, default=40)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--out-name", default="answer_confidence")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for ds in (s.strip() for s in args.datasets.split(",") if s.strip()):
        probe = RAQProbeConfig.model_validate(load_yaml(configs_dir() / "datasets" / f"{ds}.yaml")["dataset"])
        for stem in (s.strip() for s in args.models.split(",") if s.strip()):
            rows += measure(stem, probe, args.n_samples, ds)
    if not rows:
        raise SystemExit("no sample was answered correctly anywhere; nothing to summarise")

    dest = resolve_output_root(args.output_root) / "reports" / f"{args.out_name}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rows, indent=1), encoding="utf-8")

    print(f"fault-free first-token distribution, {len(rows)} correctly answered samples\n")
    print(f"{'dataset':22s}{'model':22s}{'acc':>7s}{'n':>4s}{'margin':>9s}{'spread':>9s}"
          f"{'marg/spr':>10s}{'entropy':>9s}{'top1 p':>9s}{'<1 nat':>8s}")
    summary: dict[tuple[str, str], dict[str, float]] = {}
    for ds in dict.fromkeys(r["dataset"] for r in rows):
        for model in dict.fromkeys(r["model"] for r in rows if r["dataset"] == ds):
            sub = [r for r in rows if r["model"] == model and r["dataset"] == ds]
            m = {k: float(np.median([r[k] for r in sub])) for k in SHAPE_KEYS}
            # Accuracy is the anchor: a rung where it does not move did not make the task harder, so a flat margin
            # there says nothing about saturation.
            acc = len(sub) / sub[0]["n_attempted"]
            thin = float(np.mean([r["margin"] < 1.0 for r in sub]))
            summary[ds, model] = m | {"accuracy": acc, "frac_below_1_nat": thin, "n": float(len(sub))}
            print(f"{ds:22s}{model:22s}{acc:7.3f}{len(sub):4d}{m['margin']:9.3f}{m['logprob_spread']:9.3f}"
                  f"{m['margin_over_spread']:10.4f}{m['entropy']:9.4f}{m['top1_prob']:9.4f}{thin:8.3f}")
    print("\n(medians; margin and spread in nats of log-probability. acc = share of attempted samples answered"
          "\n correctly, and it is the anchor: a rung where accuracy does not move did not make the task harder.)")

    models = list(dict.fromkeys(r["model"] for r in rows))
    if len(models) == 2:
        print(f"\nratios {models[1]} / {models[0]}, per rung:")
        for ds in dict.fromkeys(r["dataset"] for r in rows):
            if (ds, models[0]) not in summary or (ds, models[1]) not in summary:
                continue
            ma, mb = summary[ds, models[0]], summary[ds, models[1]]
            parts = "  ".join(
                f"{k.split('_')[0]:6s} {mb[k] / ma[k]:6.3f}x" if ma[k] else f"{k.split('_')[0]:6s}    n/a"
                for k in ("margin", "logprob_spread", "margin_over_spread"))
            print(f"  {ds:22s} {parts}")
        print("\n  If `logprob_spread` moves by about the same factor as `margin`, the gap is a scale artefact and"
              "\n  `margin_over_spread` is the number to read; if the spreads match while the margins do not, the"
              "\n  gap is real sharpness.")

    if len(summary) > len(models):
        print("\nthe ladder, per model -- does difficulty desaturate the saturated one?\n")
        for model in models:
            rungs = [(ds, v) for (ds, m), v in summary.items() if m == model]
            for ds, v in rungs:
                print(f"  {model:22s}{ds:22s}acc {v['accuracy']:.3f}  margin {v['margin']:7.3f}  "
                      f"top1 p {v['top1_prob']:.4f}  below 1 nat {v['frac_below_1_nat']:.3f}")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
