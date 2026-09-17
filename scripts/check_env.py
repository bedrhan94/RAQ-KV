"""Print and save the environment provenance (does not create a CUDA context)."""

from __future__ import annotations

import json
import sys

from raq_kv.experiments.provenance import collect_provenance
from raq_kv.utils.io import write_json_atomic
from raq_kv.utils.paths import resolve_output_root


def main() -> int:
    prov = collect_provenance()
    print(json.dumps(prov, indent=2, ensure_ascii=False))
    out = resolve_output_root("outputs") / "reports" / "env.json"
    write_json_atomic(out, prov)
    print(f"saved to {out}")
    for gpu in prov["gpus_at_start"]:
        used, total = float(gpu["memory_used_mib"]), float(gpu["memory_total_mib"])
        if used / total > 0.1 or float(gpu["utilization_pct"]) > 10:
            print(f"WARNING: GPU {gpu['index']} busy ({used:.0f}/{total:.0f} MiB, {gpu['utilization_pct']}% util); "
                  "latency measurements would be invalid and memory may be insufficient")
    return 0


if __name__ == "__main__":
    sys.exit(main())
