"""The two-axis report: does pruning change what a fault costs?

D43's numbers were first produced by throwaway snippets in a shell, which is not a reproducible provenance for a
structural claim. This is that analysis as a script, and it enforces the three things the ad-hoc version had to
be talked into:

* **Pairing over the intersection.** A pruned arm's clean-correct set is smaller than an unpruned arm's, because
  pruning itself costs accuracy. Arms are paired over the samples *every* arm answers correctly unfaulted **and**
  that the full cache also answers, so a sample pruning alone broke is never charged to the fault model.
* **`task_failure`, not `answer_changed`.** The first version of this analysis used the proxy and got different
  numbers; the endpoint is whether the answer became wrong, and the proxy is reported beside it, never instead.
* **The composition control, always.** A pruned cache holds fewer quantization groups, so before any claim about
  *importance* concentrating, the share of flips landing in metadata has to be shown not to have moved --
  metadata is at least 1543x more damaging per bit (D25/D28), so a compositional shift would explain the effect
  with no appeal to importance at all.

    python scripts/report_fault_retention.py --arms m6_fr_unpruned_unprot,m6_fr_pruned_unprot_matched \
        --contrast m6_fr_unpruned_unprot:m6_fr_pruned_unprot_matched

Every contrast prints the pre-registered verdict (D31: p < 0.05 **and** a bootstrap CI excluding zero), because
a difference that meets one and not the other is a direction, not a result -- and this project has withdrawn
findings for treating the second kind as the first.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
from scipy import stats

from raq_kv.utils.io import read_json, read_jsonl
from raq_kv.utils.paths import resolve_output_root

BOOTSTRAP = 10_000

# Region names as `faults/storage.py` emits them: `code`, `scale`, `zero`, plus `check:<region>` for ECC check
# bits and `bound:<region>` for the range guard's own stored limits.
#
# **Payload metadata and protection overhead are counted separately, and conflating them was a real mistake in
# the first version of this script.** A substring test for "scale" matches `check:scale` too, so a metadata-SECDED
# arm looked as though 8.4% of its flips hit metadata against a range-guarded arm's 6.2% -- and the control
# reported "composition moved" on a comparison where nothing vulnerable had moved at all. The extra bits are
# SECDED's own check bits, which exist to protect the metadata rather than to be damaged. The payload share is
# what a concentration claim is about; the overhead share is reported beside it because D23 requires those bits
# to be exposed to faults like any others, not because a shift in them means the same thing.
def _classify(target: str) -> str:
    if target.startswith(("check:", "bound:")):
        return "overhead"
    return "meta" if target in ("scale", "zero") else "code"


def load_arm(raw_root: Path, name: str) -> dict[str, dict]:
    for exp in sorted(raw_root.glob("*")):
        if ".superseded-" in exp.name or not (exp / "manifest.json").exists():
            continue
        manifest = read_json(exp / "manifest.json")
        if manifest.get("superseded_by") or manifest["config"]["name"] != name:
            continue
        records = read_jsonl(exp / "results.jsonl")
        if len(records) != manifest["unit_keys_total"]:
            raise SystemExit(f"{name}: incomplete ({len(records)}/{manifest['unit_keys_total']})")
        return {r["sample_id"]: r for r in records}
    raise SystemExit(f"no complete run found for arm {name!r}")


def shared_samples(arms: dict[str, dict[str, dict]]) -> list[str]:
    """Samples every arm answers correctly with no faults, and that the full cache answers too."""
    keep: set[str] | None = None
    for data in arms.values():
        good = {s for s, r in data.items()
                if r["golden_correct"] and r.get("unpruned_correct") is not False}
        keep = good if keep is None else (keep & good)
    return sorted(keep or ())


def _trials(record: dict) -> list[dict]:
    return [t for t in record["faults"] if t["trial"] >= 0]


def _controls_clean(data: dict[str, dict]) -> tuple[int, int]:
    """No-fault scorings must show exactly zero damage, or the arm's own reference is unstable."""
    total = changed = 0
    for r in data.values():
        for t in r["faults"]:
            if t["trial"] < 0:
                total += 1
                changed += int(bool(t["answer_changed"]))
    return changed, total


def rates(data: dict[str, dict], samples: list[str], field: str) -> np.ndarray:
    out = []
    for s in samples:
        trials = _trials(data[s])
        if field == "task_failure":
            out.append(sum(1 for t in trials if not t["faulty_correct"]) / len(trials))
        else:
            out.append(sum(1 for t in trials if t["answer_changed"]) / len(trials))
    return np.array(out)


def flip_shares(data: dict[str, dict], samples: list[str], kind: str) -> np.ndarray:
    """Per sample, the fraction of injected flips that landed in ``kind`` -- 'meta', 'overhead' or 'code'."""
    out = []
    for s in samples:
        agg: dict[str, int] = {}
        for t in _trials(data[s]):
            for target, count in (t.get("by_target") or {}).items():
                agg[_classify(target)] = agg.get(_classify(target), 0) + count
        total = sum(agg.values())
        out.append(agg.get(kind, 0) / total if total else 0.0)
    return np.array(out)


def stored_shares(data: dict[str, dict], samples: list[str]) -> dict[str, float]:
    agg: dict[str, int] = {}
    for s in samples:
        for target, bits in data[s]["storage_composition"].items():
            agg[_classify(target)] = agg.get(_classify(target), 0) + bits
    total = sum(agg.values()) or 1
    return {k: v / total for k, v in agg.items()}


def _contrast_rng(x_name: str, y_name: str, seed: int) -> np.random.Generator:
    """A generator keyed by the pair, so a contrast's interval does not depend on what was asked alongside it.

    With one shared generator, requesting three contrasts in one invocation gave the third a different bootstrap
    draw than requesting it alone -- so the same comparison had two intervals depending on the command line, which
    is not a reproducible number. Keying on the arm names makes each contrast's interval a function of the data
    and the ``--seed`` alone.
    """
    digest = hashlib.sha256(f"{seed}|{x_name}|{y_name}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def contrast(x_name: str, y_name: str, xv: np.ndarray, yv: np.ndarray, seed: int) -> dict:
    d = yv - xv
    discordant = int((d != 0).sum())
    p = 1.0 if discordant == 0 else float(stats.wilcoxon(xv, yv).pvalue)
    rng = _contrast_rng(x_name, y_name, seed)
    boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(BOOTSTRAP)])
    lo, hi = (float(v) for v in np.percentile(boot, [2.5, 97.5]))
    return {"x": x_name, "y": y_name, "x_mean": float(xv.mean()), "y_mean": float(yv.mean()),
            "difference": float(d.mean()), "ci": (lo, hi), "p": p, "discordant": discordant,
            # D31's rule, both halves. A difference meeting one and not the other is a direction, not a result.
            "meets_rule": bool(p < 0.05 and lo * hi > 0)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", required=True, help="Comma-separated config names, in the order to print")
    parser.add_argument("--contrast", action="append", default=[],
                        help="'x:y' pair to test, repeatable; y is compared against x")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    raw = resolve_output_root(args.output_root) / "raw"
    names = [a.strip() for a in args.arms.split(",") if a.strip()]
    arms = {n: load_arm(raw, n) for n in names}
    samples = shared_samples(arms)
    if not samples:
        raise SystemExit("no shared samples: every arm must answer some sample correctly without faults")
    print(f"Paired over {len(samples)} samples, of {min(len(d) for d in arms.values())} run "
          f"(clean-correct in every arm; unpruned-correct where an arm prunes)\n")

    print("No-fault controls (must be exactly zero damage)")
    for n, data in arms.items():
        changed, total = _controls_clean(data)
        flag = "ok " if changed == 0 else "FAIL"
        print(f"  [{flag}] {n:34s} {changed} of {total} control scorings changed")

    print(f"\n{'arm':34s} {'stored Mbit':>12s} {'flips':>7s} {'meta%':>7s} {'ovhd%':>7s} "
          f"{'kept_tgt':>9s} {'task_fail':>10s} {'ans_chg':>8s}")
    for n, data in arms.items():
        stored = np.mean([data[s]["stored_bits"] for s in samples]) / 1e6
        flips = np.mean([data[s]["n_flips_per_trial"] for s in samples])
        shares = stored_shares(data, samples)
        kept = data[samples[0]].get("kept_fraction_target")
        kept_txt = "-" if kept is None else f"{np.mean([data[s]['kept_fraction_target'] for s in samples]):.3f}"
        print(f"{n:34s} {stored:12.1f} {flips:7.0f} {shares.get('meta', 0):6.3%} "
              f"{shares.get('overhead', 0):6.3%} {kept_txt:>9s} "
              f"{rates(data, samples, 'task_failure').mean():10.4f} "
              f"{rates(data, samples, 'answer_changed').mean():8.4f}")

    for pair in args.contrast:
        if ":" not in pair:
            raise SystemExit(f"--contrast expects 'x:y', got {pair!r}")
        x, y = (p.strip() for p in pair.split(":", 1))
        for missing in (n for n in (x, y) if n not in arms):
            raise SystemExit(f"{missing!r} is not in --arms")
        r = contrast(x, y, rates(arms[x], samples, "task_failure"), rates(arms[y], samples, "task_failure"), args.seed)
        verdict = "MEETS the rule" if r["meets_rule"] else "direction only (needs p<0.05 AND a CI excluding 0)"
        print(f"\n{x}  ->  {y}   [task_failure]"
              f"\n  {r['x_mean']:.4f} -> {r['y_mean']:.4f}   difference {r['difference']:+.4f} "
              f"[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]"
              f"\n  p = {r['p']:.4g}   discordant = {r['discordant']} of {len(samples)}   => {verdict}")

        # The composition control, printed with every contrast rather than on request: a shift here explains a
        # difference without any appeal to importance, so it is not optional context.
        mx, my = flip_shares(arms[x], samples, "meta"), flip_shares(arms[y], samples, "meta")
        dm = my - mx
        pm = 1.0 if not (dm != 0).any() else float(stats.wilcoxon(mx, my).pvalue)
        note = ("PAYLOAD composition moved - part of the difference may be compositional, not importance"
                if pm < 0.05 else "payload composition unchanged - an importance explanation is not undercut")
        ox, oy = flip_shares(arms[x], samples, "overhead"), flip_shares(arms[y], samples, "overhead")
        print(f"  composition control: payload-metadata share of flips {mx.mean():.4%} -> {my.mean():.4%}, "
              f"difference {dm.mean():+.4%}, p = {pm:.3g}\n    {note}"
              f"\n    protection overhead share {ox.mean():.4%} -> {oy.mean():.4%} (check and bound bits are "
              f"exposed per D23, but they are the protection, not the payload)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
