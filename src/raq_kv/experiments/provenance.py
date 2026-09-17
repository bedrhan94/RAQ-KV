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
