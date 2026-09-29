"""Resumable experiment runner.

Layout per experiment (never deleted or overwritten by the runner):

    outputs/raw/<exp_id>/manifest.json   identity, full config, data fingerprint, one provenance entry per run
    outputs/raw/<exp_id>/results.jsonl   one record per finished unit (append-only)
    outputs/raw/<exp_id>/errors.jsonl    one record per failed unit (the run stops at the first failure)

Resume semantics: units whose ``unit_id`` already appears in ``results.jsonl`` are skipped. Re-running a
finished experiment is a no-op and does not load the model. A different full config hash under the same
``exp_id`` or a changed dataset fingerprint aborts. A changed git commit aborts unless explicitly allowed,
because mixing records from different code versions in one experiment would be silent contamination.
"""

from __future__ import annotations

import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from raq_kv.config.schema import ExperimentConfig
from raq_kv.experiments.ids import config_hash, experiment_id, group_hash, unit_id
from raq_kv.experiments.provenance import collect_provenance
from raq_kv.experiments.tasks import Task, build_task
from raq_kv.utils.io import JsonlAppender, read_json, read_jsonl, write_json_atomic
from raq_kv.utils.paths import resolve_output_root
from raq_kv.utils.seed import derive_seed, seed_everything


class ExperimentConflictError(RuntimeError):
    """Existing experiment directory is incompatible with the requested run."""


@dataclass
class RunSummary:
    exp_id: str
    exp_dir: Path
    units_total: int
    units_skipped: int
    units_run: int


def experiment_dir(cfg: ExperimentConfig) -> Path:
    return resolve_output_root(cfg.output.root) / "raw" / experiment_id(cfg)


def _code_state(git: dict[str, Any]) -> tuple[str | None, str | None]:
    return git.get("commit"), git.get("diff_sha256")


# The code that runs is the code Python imported at process start; the working tree can be edited afterwards. A
# manifest would then record a state that never executed, which is how a sweep ends up with three different diff
# hashes for one campaign. So the first state observed in this process is remembered and later ones are refused.
#
# What is compared is `executed_code_sha256` -- the contents of every `src/raq_kv` module a campaign actually imports --
# and NOT the repository-wide `diff_sha256` this used to use. D24's rule is about `src/`, but the old comparison aborted
# a campaign for editing a decision record, a test or a report script, which cost a rung of D76 plus a supersede. It
# also aborted on `git commit`, because a diff is relative to HEAD while the executing files are unchanged -- and
# committing a record between runs is the normal workflow here. `_check_existing` below still compares the old tuple, so
# the 48 recorded runs carrying a `diff_sha256` continue to match on resume.
_PROCESS_CODE_STATE: tuple[str | None, dict[str, Any]] | None = None


def _executed_state(git: dict[str, Any]) -> str | None:
    return git.get("executed_code_sha256")


def _check_process_code_state(git: dict[str, Any]) -> None:
    """Refuse to start an experiment if the code this campaign EXECUTES changed since the process imported it."""
    global _PROCESS_CODE_STATE
    state = _executed_state(git)
    if _PROCESS_CODE_STATE is None:
        _PROCESS_CODE_STATE = (state, git)
        return
    if state != _PROCESS_CODE_STATE[0]:
        raise ExperimentConflictError(
            f"the executed source changed while this process was running: started at {_PROCESS_CODE_STATE[0]}, now "
            f"{state}. The already-imported code is what executes, so a manifest written now would record a state "
            "that never ran. Restart the run instead of editing src/raq_kv during a campaign -- the modules in "
            "`provenance.EXECUTED_CODE_EXEMPT` are the ones you may still edit."
        )


def _check_existing(manifest: dict[str, Any], cfg: ExperimentConfig, fingerprint: dict[str, Any],
                    git: dict[str, Any], allow_code_change: bool) -> None:
    if manifest["config_hash"] != config_hash(cfg):
        raise ExperimentConflictError(f"{manifest['exp_id']}: stored config hash differs from requested config")
    if manifest["data_fingerprint"] != fingerprint:
        raise ExperimentConflictError(f"{manifest['exp_id']}: dataset/tokenizer fingerprint changed since first run")
    previous = {_code_state(r["provenance"]["git"]) for r in manifest["runs"]}
    if _code_state(git) not in previous and not allow_code_change:
        raise ExperimentConflictError(
            f"{manifest['exp_id']}: code state (commit, local diff) {_code_state(git)} differs from previous runs "
            f"{sorted(map(str, previous))}; pass allow_code_change=True (--allow-code-change) only if the change "
            "cannot affect results"
        )


def run_experiment(
    cfg: ExperimentConfig,
    *,
    allow_code_change: bool = False,
    task_factory: Callable[[ExperimentConfig], Task] = build_task,
    log: Callable[[str], None] = print,
) -> RunSummary:
    exp_id = experiment_id(cfg)
    exp_dir = experiment_dir(cfg)
    manifest_path, results_path, errors_path = (exp_dir / n for n in ("manifest.json", "results.jsonl", "errors.jsonl"))

    task = task_factory(cfg)
    keys = task.unit_keys()
    if len(set(keys)) != len(keys):
        raise ValueError("task produced duplicate unit keys")
    fingerprint = task.data_fingerprint()
    provenance = collect_provenance()
    _check_process_code_state(provenance["git"])

    if manifest_path.exists():
        manifest = read_json(manifest_path)
        _check_existing(manifest, cfg, fingerprint, provenance["git"], allow_code_change)
    else:
        manifest = {
            "exp_id": exp_id,
            "config_hash": config_hash(cfg),
            "group_hash": group_hash(cfg),
            "config": cfg.model_dump(mode="json"),
            "data_fingerprint": fingerprint,
            "unit_keys_total": len(keys),
            "runs": [],
        }

    done = {r["unit_id"] for r in read_jsonl(results_path)}
    pending = [k for k in keys if unit_id(exp_id, k) not in done]
    log(f"[{exp_id}] units: {len(keys)} total, {len(keys) - len(pending)} done, {len(pending)} pending")
    if not pending:
        return RunSummary(exp_id, exp_dir, len(keys), len(keys), 0)

    run_id = uuid.uuid4().hex[:12]
    run_entry: dict[str, Any] = {"run_id": run_id, "provenance": provenance, "status": "running"}
    manifest["runs"].append(run_entry)
    write_json_atomic(manifest_path, manifest)

    seed_everything(cfg.seed)
    run_entry["setup"] = task.setup()
    write_json_atomic(manifest_path, manifest)

    n_run = 0
    with JsonlAppender(results_path) as results:
        for key in pending:
            uid = unit_id(exp_id, key)
            unit_seed = derive_seed(cfg.seed, key)
            seed_everything(unit_seed)
            t0 = time.perf_counter()
            try:
                record = task.run_unit(key)
            except Exception as exc:
                with JsonlAppender(errors_path) as errors:
                    errors.append(
                        {"unit_id": uid, "run_id": run_id, "error": repr(exc), "traceback": traceback.format_exc()}
                    )
                run_entry["status"] = f"failed at {key}"
                write_json_atomic(manifest_path, manifest)
                raise
            results.append({
                "unit_id": uid,
                "exp_id": exp_id,
                "run_id": run_id,
                "unit_key": key,
                "seed": cfg.seed,
                "unit_seed": unit_seed,
                **record,
                "unit_wall_s": time.perf_counter() - t0,
            })
            n_run += 1
            log(f"[{exp_id}] {key} done ({n_run}/{len(pending)})")

    run_entry["status"] = "completed"
    run_entry["units_run"] = n_run
    write_json_atomic(manifest_path, manifest)
    return RunSummary(exp_id, exp_dir, len(keys), len(keys) - len(pending), n_run)
