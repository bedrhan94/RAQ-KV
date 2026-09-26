"""How evenly does a model spread attention across its quantization groups?

D59 left one question open. The failures in every fault arm are metadata-driven, and the per-metadata-flip hazard
is **1.17% on an unpruned Qwen cache against 2.63% on an unpruned Llama one** -- Llama already sits at the ~2.5%
both families converge to once pruned, which is why pruning cannot concentrate anything further there. What D59
could not say is *why* Llama starts at the ceiling.

Dynamic range was the cheap candidate and D59 ruled it out: Llama's values carry 500x smaller MSE at equal SQNR,
and a corrupted fp16 scale multiplies by up to 65536 either way, so from a smaller base that is a *smaller*
excursion -- which predicts less fragility, not more. What remains is the thing the stored statistics cannot see:
**how many quantization groups a model attends to enough that corrupting one would change the answer.**

A corrupted scale rescales one whole group. If attention is concentrated on a few groups, a randomly placed
corruption usually lands somewhere inconsequential; if it is spread evenly, most corruptions matter. So the
prediction is specific and falsifiable: **Llama's attention should be spread across more groups than Qwen's.**

This measures exactly that and nothing else -- no faults, no pruning, no quantization error. One prefill and one
attention read per sample, the same `query_attention` scorer M4 uses, aggregated into `group_token` groups of the
size the arms actually used. It is a structural measurement of the two models rather than an experiment arm, so it
carries no exp_id; it is deterministic given the config and the seed, and the script is committed.

    python scripts/measure_group_attention.py --models qwen3_1p7b_eager,llama3p2_3b_eager --group-size 128

Reported per sample and then aggregated: the **effective number of groups** exp(H) of the attention distribution
over groups, the share of mass in the top group and top 5, and the share of groups holding 90% of the mass. A
hazard proportional to "fraction of groups that matter" predicts the last of those to track the measured hazards.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any

import numpy as np
import torch

from raq_kv.config.loader import load_yaml
from raq_kv.config.schema import ModelConfig, RAQProbeConfig
from raq_kv.data.raq_probe import build_raq_probe
from raq_kv.evaluation.protocol import prefill
from raq_kv.importance.scorers import ScoreContext, build_scorer
from raq_kv.models.loader import load_model, load_tokenizer
from raq_kv.utils.paths import configs_dir, resolve_output_root


def group_shares(scores: torch.Tensor, group_size: int) -> np.ndarray:
    """Attention mass per `group_token` group, normalised to sum to 1.

    Scores are non-negative attention weights summed over heads and layers, so they are already a mass; the only
    work here is folding them into the groups a scale actually covers and renormalising so two samples of
    different length are comparable.
    """
    v = scores.detach().float().reshape(-1).clamp_min(0.0).cpu().numpy()
    if v.sum() <= 0:
        return np.array([])
    n_groups = -(-len(v) // group_size)
    padded = np.zeros(n_groups * group_size, dtype=np.float64)
    padded[: len(v)] = v
    per_group = padded.reshape(n_groups, group_size).sum(axis=1)
    return per_group / per_group.sum()


def concentration(shares: np.ndarray) -> dict[str, float]:
    """Four views of the same distribution, because no single number is obviously the right one.

    `effective_groups` is exp(Shannon entropy) -- the number of equally-weighted groups that would carry the same
    entropy, which is the quantity a "fraction of groups that matter" story is about. The top-k shares and the
    90%-mass fraction are reported beside it so a reader can see whether they agree rather than trusting the
    entropy alone.
    """
    if shares.size == 0:
        return {}
    p = shares[shares > 0]
    entropy = float(-(p * np.log(p)).sum())
    order = np.sort(shares)[::-1]
    cumulative = np.cumsum(order)
    n90 = int(np.searchsorted(cumulative, 0.90) + 1)
    return {
        "n_groups": float(shares.size),
        "effective_groups": float(math.exp(entropy)),
        "effective_fraction": float(math.exp(entropy) / shares.size),
        "top1_share": float(order[0]),
        "top5_share": float(order[:5].sum()),
        "groups_for_90pc": float(n90),
        "fraction_for_90pc": float(n90 / shares.size),
    }


def measure(model_stem: str, probe: RAQProbeConfig, group_size: int, n_samples: int) -> list[dict[str, Any]]:
    cfg = ModelConfig.model_validate(load_yaml(configs_dir() / "models" / f"{model_stem}.yaml")["model"])
    if cfg.attn_implementation != "eager":
        raise SystemExit(f"{model_stem}: attention weights need eager attention (D5), got "
                         f"{cfg.attn_implementation!r}")
    tokenizer = load_tokenizer(cfg)
    # Spread the subset across the probe's whole length range instead of taking the head. The first samples are
    # all 512-token contexts, which give only 4-5 groups -- at that size `top5_share` is 1.0 by arithmetic and the
    # concentration metrics are degenerate. The fault arms ran over 512-4096 with a mean near 2066 tokens, so the
    # measurement has to cover the same range to say anything about them.
    everything = build_raq_probe(probe, tokenizer, dict(cfg.chat_template_kwargs))
    if n_samples < len(everything):
        step = len(everything) / n_samples
        samples = [everything[int(i * step)] for i in range(n_samples)]
    else:
        samples = everything
    lm = load_model(cfg)
    scorer = build_scorer("query_attention", 64)
    out = []
    for sample in samples:
        ids = sample.prompt_ids.to(lm.device).unsqueeze(0)
        context_ids, query_ids = ids[:, : sample.cache_len], ids[:, sample.cache_len :]
        snapshot = prefill(lm.model, context_ids)
        scores = scorer.score(ScoreContext(
            snapshot=snapshot, context_ids=context_ids, model=lm.model, query_ids=query_ids,
            target_tokens=(), distractor_tokens=(), sentence_tokens=(), control_span_tokens=()))
        row = concentration(group_shares(scores, group_size))
        if row:
            out.append({"model": cfg.short_name, "sample_id": sample.sample_id,
                        "cache_len": sample.cache_len, **row})
        del snapshot
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default="qwen3_1p7b_eager,llama3p2_3b_eager")
    parser.add_argument("--group-size", type=int, default=128, help="group_token size the fault arms used")
    parser.add_argument("--n-samples", type=int, default=40)
    parser.add_argument("--output-root", default="outputs")
    args = parser.parse_args()

    probe = RAQProbeConfig.model_validate(
        load_yaml(configs_dir() / "datasets" / "raq_probe_v0.yaml")["dataset"])
    rows: list[dict[str, Any]] = []
    for stem in (s.strip() for s in args.models.split(",") if s.strip()):
        rows += measure(stem, probe, args.group_size, args.n_samples)

    out = resolve_output_root(args.output_root) / "reports" / f"group_attention_g{args.group_size}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")

    print(f"attention spread over group_token groups of {args.group_size}, {len(rows)} samples\n")
    print(f"{'model':22s}{'groups':>8s}{'eff.groups':>12s}{'eff.frac':>10s}"
          f"{'top1':>8s}{'top5':>8s}{'frac90':>9s}")
    keys = ("n_groups", "effective_groups", "effective_fraction", "top1_share", "top5_share",
            "fraction_for_90pc")
    for model in dict.fromkeys(r["model"] for r in rows):
        sub = [r for r in rows if r["model"] == model]
        m = {k: float(np.mean([r[k] for r in sub])) for k in keys}
        print(f"{model:22s}{m['n_groups']:8.1f}{m['effective_groups']:12.2f}{m['effective_fraction']:10.3f}"
              f"{m['top1_share']:8.3f}{m['top5_share']:8.3f}{m['fraction_for_90pc']:9.3f}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
