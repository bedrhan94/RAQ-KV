"""Is Llama's thinner answer margin real sharpness, or just a smaller logit scale? -- D62's scale control.

D62's branch B measures the fault-free margin on the first answer token and finds **Qwen3-1.7B at 17.07 nats
against Llama-3.2-3B at 3.27**, a 13.8-nat gap that meets D31 on 31 of 31 samples. Read as "Llama holds its
answers with less confidence, so the same perturbation crosses the threshold more often", that explains D59's
hazard difference without any appeal to quantization.

**But a margin in nats is not scale-free.** If Qwen's output head simply produces logits on a wider scale, every
one of its margins is larger for a reason that has nothing to do with confidence -- and a perturbation of its cache
would move its log-probabilities further too, leaving the *ratio* unchanged and branch B empty. The margin gap is
only evidence for branch B if the two families' distributions are on comparable scales.

So this measures the shape of the fault-free first-token distribution and nothing else -- no faults, no pruning,
the same quantization the D62 arms used:

* **margin** -- top-1 minus top-2 log-probability, reproducing the arms' own number as a cross-check;
* **logprob spread** -- the standard deviation of the log-probabilities over the vocabulary, which is the scale
  factor the margin would be riding on if branch B were a units artefact;
* **margin / spread** -- the scale-free version of the margin, which is what branch B actually needs;
* **entropy** and **top-1 probability** -- interpretable summaries a reader can sanity-check the other two against.

    python scripts/measure_answer_confidence.py

A structural measurement of the two models rather than an experiment arm, so it carries no exp_id; it is
deterministic given the configs and the seed, and the script is committed.
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


def shape_of(logprobs: torch.Tensor) -> dict[str, float]:
    """Summaries of one next-token distribution, in float64 log-space."""
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


def measure(stem: str, probe: RAQProbeConfig, n_samples: int) -> list[dict[str, Any]]:
    cfg = ModelConfig.model_validate(load_yaml(configs_dir() / "models" / f"{stem}.yaml")["model"])
    tokenizer = load_tokenizer(cfg)
    samples = build_raq_probe(probe, tokenizer, dict(cfg.chat_template_kwargs))[:n_samples]
    lm = load_model(cfg)
    eos = set(lm.eos_token_ids)
    rows = []
    for sample in samples:
        ids = sample.prompt_ids.to(lm.device).unsqueeze(0)
        context_ids, query_ids = ids[:, : sample.cache_len], ids[:, sample.cache_len :]
        base = quantize_to_codes(prefill(lm.model, context_ids), QUANT).to_snapshot()
        # Only samples the arm would have counted: D62 pairs over samples both families answer correctly with no
        # faults, so a margin measured on a sample the model gets wrong is not the quantity branch B is about.
        text = lm.tokenizer.decode(greedy_decode(lm.model, query_ids, sample.max_new_tokens, snapshot=base,
                                                 eos_token_ids=eos).tokens, skip_special_tokens=True)
        if not SCORERS[sample.scorer](text, sample.reference)["correct"]:
            continue
        lp = score_through_cache(lm.model, base, query_ids, query_ids, return_logprobs=True,
                                 logprob_dtype=torch.float64).logprobs[-1]
        rows.append({"model": cfg.short_name, "sample_id": sample.sample_id,
                     "cache_len": sample.meta["cache_len"], **shape_of(lp)})
        del base
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default="qwen3_1p7b_eager,llama3p2_3b_eager")
    parser.add_argument("--n-samples", type=int, default=40)
    parser.add_argument("--output-root", default="outputs")
    args = parser.parse_args()

    probe = RAQProbeConfig.model_validate(load_yaml(configs_dir() / "datasets" / "raq_probe_v0.yaml")["dataset"])
    rows: list[dict[str, Any]] = []
    for stem in (s.strip() for s in args.models.split(",") if s.strip()):
        rows += measure(stem, probe, args.n_samples)

    dest = resolve_output_root(args.output_root) / "reports" / "answer_confidence.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rows, indent=1), encoding="utf-8")

    print(f"fault-free first-token distribution, {len(rows)} correctly answered samples\n")
    print(f"{'model':22s}{'n':>4s}{'vocab':>8s}{'margin':>9s}{'spread':>9s}{'margin/spread':>15s}"
          f"{'entropy':>9s}{'top1 p':>9s}")
    summary = {}
    for model in dict.fromkeys(r["model"] for r in rows):
        sub = [r for r in rows if r["model"] == model]
        m = {k: float(np.median([r[k] for r in sub]))
             for k in ("margin", "logprob_spread", "margin_over_spread", "entropy", "top1_prob")}
        summary[model] = m
        print(f"{model:22s}{len(sub):4d}{sub[0]['vocab']:8d}{m['margin']:9.3f}{m['logprob_spread']:9.3f}"
              f"{m['margin_over_spread']:15.4f}{m['entropy']:9.4f}{m['top1_prob']:9.4f}")
    print("\n(medians; margin and spread in nats of log-probability)")

    if len(summary) == 2:
        (a, ma), (b, mb) = summary.items()
        print(f"\nratios {b} / {a}:")
        for k in ("margin", "logprob_spread", "margin_over_spread", "entropy"):
            r = mb[k] / ma[k] if ma[k] else float("nan")
            print(f"  {k:20s} {r:6.3f}x")
        print("\n  If `logprob_spread` differs by about the same factor as `margin`, branch B's gap is a scale\n"
              "  artefact and `margin_over_spread` is the number to read. If the spreads are comparable while the\n"
              "  margins are not, the gap is real sharpness and branch B stands.")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
