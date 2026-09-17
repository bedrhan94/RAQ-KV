"""Seeding and determinism controls.

Every experiment unit gets its own seed derived from (experiment seed, unit key), so a resumed run
reproduces exactly the same random draws as an uninterrupted one.
"""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np
import torch


def set_determinism_env() -> None:
    """Must run before the first cuBLAS call (i.e. before any CUDA work) to take effect."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def derive_seed(base_seed: int, *keys: object) -> int:
    """Stable 31-bit seed from a base seed and arbitrary keys (independent of PYTHONHASHSEED)."""
    material = "|".join([str(base_seed), *map(str, keys)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "little") & 0x7FFFFFFF


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # lazy: does not create a CUDA context on a busy GPU
    if deterministic:
        set_determinism_env()
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
