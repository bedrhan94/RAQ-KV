"""Regenerate summary CSVs from raw JSONL results.

Example:
    python scripts/aggregate.py --name m0a --filter qwen3-1p7b_wikitext2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from raq_kv.analysis.aggregate import across_seeds, summarize_all
from raq_kv.utils.paths import resolve_output_root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--name", default="summary", help="prefix of the CSV files written to outputs/csv/")
    parser.add_argument("--filter", default=None, help="only experiment ids containing this substring")
    args = parser.parse_args()

    root = resolve_output_root(args.output_root)
    per_exp = summarize_all(root / "raw", args.filter)
    if per_exp.empty:
        print("no experiments with results found")
        return 1
    csv_dir = root / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    per_exp_path: Path = csv_dir / f"{args.name}_per_experiment.csv"
    seeds_path: Path = csv_dir / f"{args.name}_across_seeds.csv"
    per_exp.to_csv(per_exp_path, index=False, encoding="utf-8")
    across_seeds(per_exp).to_csv(seeds_path, index=False, encoding="utf-8")
    print(f"wrote {per_exp_path} ({len(per_exp)} rows) and {seeds_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
