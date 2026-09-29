"""Environment provenance recorded with every run (plan §28).

GPU information is read through ``nvidia-smi`` rather than ``torch.cuda`` so that collecting provenance never
creates a CUDA context on a GPU that another job is using.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from raq_kv.utils.paths import repo_root

_PACKAGES = ("torch", "transformers", "datasets", "tokenizers", "huggingface-hub", "numpy", "pandas", "pydantic")


def _run_bytes(cmd: list[str], cwd: Path | None = None) -> bytes | None:
    # Bytes, not text: on Windows ``text=True`` decodes with the ANSI code page and crashes on UTF-8 diffs.
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, check=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def _run(cmd: list[str], cwd: Path | None = None) -> str | None:
    out = _run_bytes(cmd, cwd)
    return out.decode("utf-8", errors="replace").strip() if out is not None else None


# D24's rule -- "do not edit `src/` while a campaign runs" -- was always narrower than the guard that enforced it. The
# guard hashed `git diff HEAD` over the WHOLE repository, so editing a decision record, a test or a report script
# aborted a run that D24 permits. That cost a rung of D76 plus a supersede on 2026-09-28.
#
# These are the modules a campaign never executes, measured rather than assumed: importing the campaign entry point's
# own graph (`config.loader`, `experiments.ids`, `experiments.runner`, and `experiments.tasks` through it) pulls in 53
# `raq_kv` modules and leaves exactly these eight out. `config.loader` is deliberately NOT here -- the entry point
# imports it to read the configs, so it participates in a campaign even though `runner.py` does not import it.
# `tests/test_runner_import_surface.py` re-measures this list against the entry point, so the exemption is a property
# the suite checks rather than a claim someone remembers.
EXECUTED_CODE_EXEMPT: tuple[str, ...] = (
    "raq_kv/allocation",
    "raq_kv/analysis/aggregate.py",
    "raq_kv/analysis/arms.py",
    "raq_kv/analysis/contrasts.py",
    "raq_kv/analysis/curvature.py",
    "raq_kv/analysis/stats.py",
    "raq_kv/budget/iso.py",
)


def executed_code_sha256(root: Path | None = None) -> str | None:
    """sha256 over the source a campaign actually executes: every ``src/raq_kv`` module except the exempt ones.

    Hashed from FILE CONTENTS rather than from a git diff, for two reasons. A diff is relative to HEAD, so committing
    between runs of one campaign changes it while the executing code is byte-identical -- and committing a record
    between runs is the normal workflow here. And a diff cannot see an untracked new module, which a campaign would
    happily import.

    Returned as None only if the tree is missing, which keeps the caller's "unknown" path intact.
    """
    root = root or repo_root()
    base = root / "src"
    if not (base / "raq_kv").is_dir():
        return None
    exempt = tuple((base / p).resolve() for p in EXECUTED_CODE_EXEMPT)
    h = hashlib.sha256()
    for f in sorted((base / "raq_kv").rglob("*.py"), key=lambda x: x.as_posix()):
        r = f.resolve()
        if any(r == e or e in r.parents for e in exempt):
            continue
        h.update(f.relative_to(base).as_posix().encode("utf-8"))
        h.update(bytes(1))  # separator; written without a backslash escape on purpose
        h.update(f.read_bytes())
        h.update(bytes(1))  # separator; written without a backslash escape on purpose
    return h.hexdigest()


def git_state(root: Path | None = None) -> dict[str, Any]:
    root = root or repo_root()
    commit = _run(["git", "rev-parse", "HEAD"], root)
    status = _run(["git", "status", "--porcelain", "--untracked-files=no"], root)
    dirty = bool(status) if status is not None else None
    diff = _run_bytes(["git", "diff", "HEAD"], root) if commit else None
    return {
        "commit": commit,
        "dirty": dirty,
        "describe": f"{commit}-dirty" if commit and dirty else commit,
        # Identifies the exact uncommitted tracked-file changes, so "same commit, different local edits" is detectable.
        "diff_sha256": hashlib.sha256(diff).hexdigest() if diff else None,
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], root),
        # What the runner will actually execute, independent of HEAD and of every file it never imports. This is what
        # the in-process guard compares; `diff_sha256` above is left exactly as it was so the 48 recorded runs that
        # carry one still match on resume.
        "executed_code_sha256": executed_code_sha256(root),
    }


def gpu_state() -> list[dict[str, str]]:
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    if not out:
        return []
    fields = ("index", "name", "driver_version", "memory_total_mib", "memory_used_mib", "utilization_pct")
    return [dict(zip(fields, (c.strip() for c in line.split(",")), strict=False)) for line in out.splitlines()]


def collect_provenance() -> dict[str, Any]:
    versions = {}
    for pkg in _PACKAGES:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    return {
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
        "packages": versions,
        "torch_cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cuda_available": torch.cuda.is_available(),
        "gpus_at_start": gpu_state(),
        "git": git_state(),
        "env": {k: os.environ.get(k) for k in ("CUBLAS_WORKSPACE_CONFIG", "HF_HOME", "CUDA_VISIBLE_DEVICES")},
    }
