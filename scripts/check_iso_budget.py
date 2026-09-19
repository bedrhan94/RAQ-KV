"""Check that a set of arm configs really is iso-budget, before any of them is run.

An equal-budget claim that is verified after the campaign is a claim verified too late: if the arms turn out to
differ by 9%, the GPU time is already spent and the comparison is worthless. So this runs on the configs alone,
costs milliseconds, and exits non-zero when the set is not comparable.

    python scripts/check_iso_budget.py configs/experiments/m3_arm_{b,c,d,e}*.yaml
    python scripts/check_iso_budget.py --tolerance 0.01 --lengths 533,1044,2066,4110 <configs...>

The budget comes from :mod:`raq_kv.budget.iso`, whose arithmetic is pinned against the real quantizer and codec
in ``tests/test_iso_budget.py``. Cache length matters (metadata and pad words do not scale perfectly), so the
check is repeated at every probe length and the worst case decides.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from raq_kv.budget.iso import iso_budget_report, predict_budget
from raq_kv.config.loader import load_experiment_configs
from raq_kv.models.arch import KVGeometry

# Qwen3-1.7B (D1). Declared here rather than loaded from the hub so the check needs no model and no network;
# tests/test_kv_snapshot.py pins these numbers against the real config.
QWEN3_1P7B = KVGeometry(num_layers=28, num_q_heads=16, num_kv_heads=8, head_dim=128, dtype_bits=16)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+", type=Path)
    parser.add_argument("--tolerance", type=float, default=0.01, help="Max relative spread in bits per value")
    parser.add_argument("--lengths", default="533,1044,2066,4110",
                        help="Cache lengths to check (probe means for nominal 512/1024/2048/4096)")
    args = parser.parse_args()

    lengths = [int(x) for x in args.lengths.split(",")]
    configs = {}
    for path in args.configs:
        for cfg in load_experiment_configs(path):
            if cfg.quantization is None:
                print(f"SKIP {path.name}: no quantization section, nothing to cost")
                continue
            configs[cfg.name] = cfg
    if len(configs) < 2:
        print("need at least two costable configs to compare")
        return 2

    worst = 0.0
    for tokens in lengths:
        arms = {name: predict_budget(cfg.quantization, cfg.protection, QWEN3_1P7B, tokens)
                for name, cfg in configs.items()}
        report = iso_budget_report(arms, args.tolerance)
        worst = max(worst, report["max_abs_deviation"])
        flag = "ok " if report["iso_budget"] else "FAIL"
        print(f"[{flag}] tokens={tokens:5d}  spread={report['max_abs_deviation'] * 100:6.3f}%  "
              f"mean={report['mean_bits_per_value']:.4f} bits/value")
        for name, arm in sorted(arms.items()):
            print(f"         {name:28s} {arm.bits_per_value:.4f}  "
                  f"(data {arm.data_bits / arm.kv_values:.3f} + meta {arm.quant_metadata_bits / arm.kv_values:.3f}"
                  f" + eccD {arm.ecc_data_bits / arm.kv_values:.3f} + eccM {arm.ecc_metadata_bits / arm.kv_values:.3f})"
                  f"  {report['relative_deviation'][name] * 100:+6.3f}%")

    if worst > args.tolerance:
        print(f"\nNOT iso-budget: worst spread {worst * 100:.3f}% exceeds the {args.tolerance * 100:.1f}% tolerance.")
        print("Fix the arms (group size and protected layer count are the fine knobs) before running them.")
        return 1
    print(f"\niso-budget confirmed: worst spread {worst * 100:.3f}% <= {args.tolerance * 100:.1f}% at every length.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
