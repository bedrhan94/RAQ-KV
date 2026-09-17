"""Run (or resume) every experiment defined by a YAML config.

Examples:
    python scripts/run_experiment.py --config configs/experiments/m0a_golden_ppl.yaml
    python scripts/run_experiment.py --config configs/experiments/m0a_golden_ppl.yaml --set dataset.n_chunks=2
    python scripts/run_experiment.py --config configs/experiments/m0a_golden_ppl.yaml --dry-run
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # before torch initializes cuBLAS
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

from raq_kv.config.loader import load_experiment_configs  # noqa: E402
from raq_kv.experiments.ids import experiment_id  # noqa: E402
from raq_kv.experiments.runner import experiment_dir, run_experiment  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY.PATH=VALUE")
    parser.add_argument("--dry-run", action="store_true", help="print resolved experiment ids and exit")
    parser.add_argument("--allow-code-change", action="store_true")
    args = parser.parse_args()

    configs = load_experiment_configs(args.config, args.overrides)
    print(f"{len(configs)} experiment(s) resolved from {args.config}")
    for cfg in configs:
        print(f"  {experiment_id(cfg)}  ->  {experiment_dir(cfg)}")
    if args.dry_run:
        return 0
    for cfg in configs:
        summary = run_experiment(cfg, allow_code_change=args.allow_code_change)
        print(f"[{summary.exp_id}] finished: {summary.units_run} run, {summary.units_skipped} skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
