"""Per-region KV magnitude statistics — the control variable for Q48.

Quantization sensitivity and fault sensitivity could both be proxies for "this region holds large values", which
would make any correlation between them vacuous. This dumps mean |value|, spread and an outlier ratio per
(layer, part) so the correlation can be computed with magnitude partialled out.
"""

from __future__ import annotations

import argparse
import sys

import torch

from raq_kv.config.loader import load_experiment_configs
from raq_kv.data.wikitext import make_chunks, tokenize_wikitext2
from raq_kv.evaluation.protocol import prefill
from raq_kv.models.loader import load_model
from raq_kv.utils.io import write_json_atomic
from raq_kv.utils.paths import configs_dir, resolve_output_root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="m2_fault_campaign", help="experiment config providing model/dataset")
    parser.add_argument("--n-chunks", type=int, default=8)
    parser.add_argument("--output-root", default="outputs")
    args = parser.parse_args()

    (cfg,) = load_experiment_configs(configs_dir() / "experiments" / f"{args.config}.yaml")
    lm = load_model(cfg.model)
    chunks = make_chunks(tokenize_wikitext2(cfg.dataset, lm.tokenizer), cfg.dataset)[: args.n_chunks]

    sums: dict[tuple[int, str], list[torch.Tensor]] = {}
    for chunk in chunks:
        snapshot = prefill(lm.model, chunk.context_ids.to(lm.device).unsqueeze(0))
        for layer in range(snapshot.num_layers):
            for part, tensors in (("keys", snapshot.keys), ("values", snapshot.values)):
                sums.setdefault((layer, part), []).append(tensors[layer].float().flatten().cpu())
        del snapshot

    rows = []
    for (layer, part), pieces in sorted(sums.items()):
        values = torch.cat(pieces)
        absolute = values.abs()
        rows.append({
            "layer": layer, "part": part,
            "mean_abs": float(absolute.mean()), "std": float(values.std()),
            "p999_abs": float(absolute.quantile(0.999)), "max_abs": float(absolute.max()),
            # How heavy the tail is: the ratio that motivates per-channel scales in the first place.
            "outlier_ratio": float(absolute.max() / absolute.mean()),
        })

    out = resolve_output_root(args.output_root) / "reports" / "kv_magnitude.json"
    write_json_atomic(out, {"model": cfg.model.hf_id, "revision": cfg.model.revision,
                            "n_chunks": len(chunks), "context_tokens": cfg.dataset.prefix_len, "per_region": rows})
    worst = max(rows, key=lambda r: r["outlier_ratio"])
    print(f"wrote {out}: {len(rows)} regions; heaviest tail = layer {worst['layer']} {worst['part']} "
          f"(max/mean = {worst['outlier_ratio']:.1f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
