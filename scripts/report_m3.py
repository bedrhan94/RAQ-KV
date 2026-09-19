"""M3 report: at equal stored bits, does *where* the bits go change what faults cost?

Three pre-registered comparisons (docs/m3_plan.md, docs/claims.md):

1. **B vs C** — uniform data ECC versus no data ECC at all, the spare bits spent on finer quantization instead.
   If C matches B, uniform data ECC is dominated by simply quantizing better.
2. **D vs E** — bit-identical arms with the same *amount* of ECC on different layers. This is the falsification
   test for the D18/D21 fault-sensitivity map: if D and E agree, the map does not predict where protection helps.
3. **A vs the protected arms** — the cost-benefit framing. Arm A is 12.3% cheaper and is never presented as an
   equal-budget competitor.

Three guards, each of which can void the report rather than decorate it:

* **Completeness** — an experiment with fewer records than its manifest promises is skipped, loudly (an earlier
  M1 report presented a 75/720 run as a result; that must not recur).
* **No-fault controls** — every arm's controls must be exactly zero damage. They are the campaign's floor.
* **Iso-budget, measured** — the budget equality is re-checked from each run's own recorded ``avg_kv_bits``, not
  from the prediction that designed the arms. A set outside tolerance is reported as not comparable.

Outcomes are split into silent corruption (answer changed, codec noticed nothing) and detected failure (answer
changed, codec raised a DUE), because a system can recompute after a DUE and cannot after an SDC.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from raq_kv.utils.io import read_json, read_jsonl
from raq_kv.utils.paths import resolve_output_root

ARM_ORDER = ["m3_arm_a_unprot", "m3_arm_g_meta_only", "m3_arm_b_uniform", "m3_arm_c_precision",
             "m3_arm_d_late19", "m3_arm_e_early19"]
# The first three pairs were pre-registered before any arm ran. The last two involve arm G, which was added
# after arms A-E landed, and are labelled as such wherever they are reported: they are a cost-benefit follow-up,
# not part of the frozen equal-budget test.
PREREGISTERED = [("m3_arm_b_uniform", "m3_arm_c_precision"), ("m3_arm_d_late19", "m3_arm_e_early19"),
                 ("m3_arm_a_unprot", "m3_arm_b_uniform"),
                 ("m3_arm_a_unprot", "m3_arm_g_meta_only"), ("m3_arm_g_meta_only", "m3_arm_b_uniform")]
POST_HOC = {("m3_arm_a_unprot", "m3_arm_g_meta_only"), ("m3_arm_g_meta_only", "m3_arm_b_uniform")}


def load_arms(raw_root: Path, names: list[str]) -> pd.DataFrame:
    """Flatten every completed ``protection_arm`` run whose config name is in ``names``."""
    frames = []
    for exp in sorted(raw_root.glob("*")):
        manifest_path = exp / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = read_json(manifest_path)
        cfg = manifest["config"]
        if cfg["name"] not in names or manifest.get("superseded_by"):
            continue
        if cfg["task"]["kind"] != "protection_arm":
            continue
        records = read_jsonl(exp / "results.jsonl")
        if len(records) != manifest["unit_keys_total"]:
            print(f"  [skipped incomplete] {manifest['exp_id']} ({len(records)}/{manifest['unit_keys_total']})")
            continue
        rows = []
        for r in records:
            base = {k: v for k, v in r.items() if k != "faults"}
            base["arm"] = cfg["name"]
            base["exp_id"] = manifest["exp_id"]
            base["depth"] = (r.get("meta") or {}).get("depth_actual")
            base["context_len"] = (r.get("meta") or {}).get("context_length")
            base.pop("meta", None)
            base.pop("kv_budget", None)
            base.pop("storage_composition", None)
            base.pop("quantization", None)
            for row in r["faults"]:
                rows.append({**base, **row})
        frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def check_controls(df: pd.DataFrame) -> bool:
    """No-fault controls must be exactly zero. If they are not, the pipeline is not deterministic and nothing
    downstream is attributable."""
    controls = df[df["trial"] < 0]
    ok = True
    print("\nNo-fault controls (must be exactly zero damage)")
    for (arm, ber), sub in controls.groupby(["arm", "ber"]):
        changed = int(sub["answer_changed"].sum())
        flag = "ok " if changed == 0 else "FAIL"
        print(f"  [{flag}] {arm:22s} ber={ber:.1e}  n={len(sub):4d}  changed={changed}")
        ok = ok and changed == 0
    if not controls.empty and not ok:
        print("  Controls are non-zero: the report below is not interpretable.")
    return ok


def iso_budget_measured(df: pd.DataFrame, tolerance: float) -> tuple[bool, pd.DataFrame]:
    """Re-derive the budget equality from what the runs recorded, per bit error rate."""
    per_arm = df.groupby("arm").agg(bits_per_value=("avg_kv_bits", "mean"),
                                    bits_sd=("avg_kv_bits", "std"),
                                    stored_bits=("stored_bits", "mean"),
                                    flips=("n_flips_per_trial", "mean")).reset_index()
    iso_arms = [a for a in per_arm["arm"] if a != "m3_arm_a_unprot"]
    subset = per_arm[per_arm["arm"].isin(iso_arms)]
    if subset.empty:
        return True, per_arm
    mean = subset["bits_per_value"].mean()
    per_arm["rel_dev"] = (per_arm["bits_per_value"] - mean) / mean
    worst = subset["bits_per_value"].apply(lambda v: abs(v - mean) / mean).max()
    return bool(worst <= tolerance), per_arm


def arm_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per arm and rate: clean accuracy, damage, and the SDC/DUE split."""
    faults = df[df["trial"] >= 0]
    rows = []
    for (arm, ber), sub in faults.groupby(["arm", "ber"]):
        clean = sub.drop_duplicates("sample_id")["golden_correct"]
        n = len(sub)
        changed = int(sub["answer_changed"].sum())
        sdc = int(sub.get("silent_corruption", pd.Series(dtype=bool)).fillna(False).sum())
        due_fail = int(sub.get("detected_failure", pd.Series(dtype=bool)).fillna(False).sum())
        correct = int(sub["faulty_correct"].sum()) if "faulty_correct" in sub else np.nan
        rows.append({
            "arm": arm, "ber": ber, "trials": n, "samples": sub["sample_id"].nunique(),
            "clean_accuracy": float(clean.mean()),
            "answer_changed": changed / n,
            "silent_corruption": sdc / n,
            "detected_failure": due_fail / n,
            "faulty_accuracy": correct / n if n else np.nan,
            "task_failure": float(sub["task_failure"].mean()) if "task_failure" in sub else np.nan,
            "corrected_words": float(sub["corrected_words"].mean()),
            "due_words": float(sub["due_words"].mean()),
            "flips_per_trial": float(sub["n_flips_per_trial"].mean()),
        })
    out = pd.DataFrame(rows)
    # Known arms in their comparison order, then anything else (pilots, calibration runs) after them, so an
    # unrecognised config name is still shown rather than silently turned into a blank row.
    seen = list(dict.fromkeys(out["arm"]))
    order = [a for a in ARM_ORDER if a in seen] + [a for a in seen if a not in ARM_ORDER]
    out["arm"] = pd.Categorical(out["arm"], order, ordered=True)
    out["ber"] = out["ber"].map(lambda v: f"{v:.1e}")
    return out.sort_values(["ber", "arm"]).reset_index(drop=True)


def add_task_failure(df: pd.DataFrame) -> pd.DataFrame:
    """The primary outcome: a fault broke an answer the arm got right without faults.

    ``answer_changed`` is measured against each arm's *own* fault-free answer, and the arms deliberately differ in
    clean quality (arm C quantizes keys 4x finer than arm B), so it is not comparable across arms on its own: an
    arm whose golden answer is already wrong cannot have it "changed" into being wrong. Task failure conditions on
    the arm having been right, which is the denominator the claim is about.
    """
    out = df.copy()
    out["task_failure"] = out["golden_correct"].astype(bool) & ~out["faulty_correct"].fillna(False).astype(bool)
    return out


def common_samples(df: pd.DataFrame, arms: list[str]) -> set[str]:
    """Samples every listed arm answers correctly without faults: the strictest shared denominator."""
    sets = []
    for arm in arms:
        sub = df[(df["arm"] == arm) & df["golden_correct"].astype(bool)]
        sets.append(set(sub["sample_id"]))
    return set.intersection(*sets) if sets else set()


def compare_samples(df: pd.DataFrame, metric: str, alpha: float = 0.05) -> pd.DataFrame:
    """Pre-registered arm pairs, **paired over samples** — the only statistic that respects the clustering.

    Trials within a sample are strongly dependent: in the 1e-6 campaign arm D's 12 changed answers came from 3
    samples and arm E's 7 from 2 (D26). A test on pooled trials would treat those as 464 independent observations
    and report a difference that is really "one arm drew one more fragile sample". So each arm gets a per-sample
    rate over its draws, arms are compared on the samples **every** arm answers correctly without faults, and the
    p-value comes from a Wilcoxon signed-rank test with a paired bootstrap CI over samples.
    """
    faults = df[df["trial"] >= 0]
    rows = []
    for ber, sub in faults.groupby("ber"):
        arms = sorted(set(sub["arm"]))
        shared = common_samples(sub, arms)
        if not shared:
            continue
        rates = {a: sub[(sub["arm"] == a) & sub["sample_id"].isin(shared)]
                 .groupby("sample_id")[metric].mean() for a in arms}
        for a, b in PREREGISTERED:
            if a not in rates or b not in rates:
                continue
            x = rates[a].reindex(sorted(shared)).fillna(0.0).to_numpy(dtype=float)
            y = rates[b].reindex(sorted(shared)).fillna(0.0).to_numpy(dtype=float)
            diff = x - y
            discordant = int((diff != 0).sum())
            rng = np.random.default_rng(0)
            boot = [float(np.mean(rng.choice(diff, diff.size, replace=True))) for _ in range(10000)]
            p = 1.0 if discordant == 0 else float(stats.wilcoxon(x, y).pvalue)
            rows.append({"ber": f"{ber:.1e}", "metric": metric, "arm_a": a, "arm_b": b,
                         "prereg": (a, b) not in POST_HOC,
                         "samples": len(shared), "rate_a": float(x.mean()), "rate_b": float(y.mean()),
                         "diff": float(diff.mean()), "ci_lo": float(np.quantile(boot, 0.025)),
                         "ci_hi": float(np.quantile(boot, 0.975)), "discordant": discordant, "p_raw": p})
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values("p_raw").reset_index(drop=True)
    holm, running = [], 0.0
    for i, p in enumerate(out["p_raw"]):
        running = max(running, min(1.0, (len(out) - i) * p))
        holm.append(running)
    out["p_holm"] = holm
    out["significant"] = out["p_holm"] < alpha
    return out


def compare(df: pd.DataFrame, metric: str, alpha: float = 0.05, paired: bool = True) -> pd.DataFrame:
    """Pre-registered arm pairs on a binary outcome, per rate, Fisher exact with Holm correction.

    With ``paired`` the pair is restricted to the samples *both* arms answer correctly at golden, so the two arms
    are compared on the same questions rather than on their own easiest ones.
    """
    faults = df[df["trial"] >= 0]
    rows = []
    for ber, sub in faults.groupby("ber"):
        present = set(sub["arm"])
        for a, b in PREREGISTERED:
            if a not in present or b not in present:
                continue
            scope = sub
            shared = common_samples(sub, [a, b]) if paired else None
            if shared is not None:
                scope = sub[sub["sample_id"].isin(shared)]
            xa = scope[scope["arm"] == a][metric].fillna(False)
            xb = scope[scope["arm"] == b][metric].fillna(False)
            ka, kb = int(xa.sum()), int(xb.sum())
            na, nb = len(xa), len(xb)
            if not na or not nb:
                continue
            table = [[ka, na - ka], [kb, nb - kb]]
            odds, p = stats.fisher_exact(table)
            rows.append({"ber": f"{ber:.1e}", "metric": metric, "arm_a": a, "arm_b": b,
                         "shared_samples": len(shared) if shared is not None else scope["sample_id"].nunique(),
                         "rate_a": ka / na, "rate_b": kb / nb, "n_a": na, "n_b": nb,
                         "odds_ratio": odds, "p_raw": p})
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values("p_raw").reset_index(drop=True)
    m = len(out)
    holm, running = [], 0.0
    for i, p in enumerate(out["p_raw"]):
        running = max(running, min(1.0, (m - i) * p))
        holm.append(running)
    out["p_holm"] = holm
    out["significant"] = out["p_holm"] < alpha
    return out


def _fmt(df: pd.DataFrame) -> str:
    return df.to_string(index=False, float_format=lambda v: f"{v:.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--arms", default=",".join(ARM_ORDER))
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--out", type=Path, default=None, help="Optional CSV directory for the tables")
    args = parser.parse_args()

    raw_root = resolve_output_root(args.output_root) / "raw"
    names = args.arms.split(",")
    print(f"M3 report — reading {raw_root}")
    df = load_arms(raw_root, names)
    if df.empty:
        print("no completed protection_arm runs found for: " + ", ".join(names))
        return 1
    print(f"loaded {len(df)} rows from {df['exp_id'].nunique()} experiment(s), arms: "
          f"{', '.join(sorted(set(df['arm'])))}")
    df = add_task_failure(df)

    controls_ok = check_controls(df)

    iso_ok, budgets = iso_budget_measured(df, args.tolerance)
    print("\nMeasured storage per arm (from each run's own bit budget)")
    print(_fmt(budgets))
    print(f"  iso-budget (excluding the cost-reference arm A): {'confirmed' if iso_ok else 'NOT CONFIRMED'} "
          f"at tolerance {args.tolerance * 100:.1f}%")

    summary = arm_summary(df)
    print("\nOutcome per arm and bit error rate")
    print(_fmt(summary))

    tables = {"arm_summary": summary, "budgets": budgets}
    # task_failure first: it is the primary outcome, and the only one comparable across arms whose clean
    # accuracy differs by design.
    for metric in ("task_failure", "answer_changed", "silent_corruption"):
        if metric not in df:
            continue
        primary = compare_samples(df, metric)
        if not primary.empty:
            label = "PRIMARY" if metric == "task_failure" else "secondary"
            print(f"\n[{label}] {metric}, paired over the samples every arm answers correctly without faults "
                  f"(Wilcoxon signed-rank, bootstrap CI over samples, Holm-corrected)")
            print(_fmt(primary))
            tables[f"compare_{metric}"] = primary
        pooled = compare(df, metric, paired=True)
        if not pooled.empty and metric == "task_failure":
            # Descriptive only: trials within a sample are not independent, so this overstates significance.
            print(f"\n[descriptive, NOT a basis for any claim] pooled-trial Fisher on {metric}")
            print(_fmt(pooled))
            tables[f"pooled_{metric}"] = pooled

    if not iso_ok or not controls_ok:
        print("\nREPORT VOID for equal-budget claims: "
              + ("controls non-zero. " if not controls_ok else "")
              + ("arms are not within budget tolerance." if not iso_ok else ""))

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for name, table in tables.items():
            table.to_csv(args.out / f"m3_{name}.csv", index=False)
        print(f"\nwrote {len(tables)} table(s) to {args.out}")
    return 0 if (iso_ok and controls_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
