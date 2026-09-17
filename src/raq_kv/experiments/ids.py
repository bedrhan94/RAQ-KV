"""Deterministic experiment and unit identifiers.

``exp_id = <readable slug>-<first 8 hex of sha256(canonical identity JSON)>``. The slug makes directories
human-browsable (model, dataset, task, precision, ECC, fault model, pruning ratio, device, seed); the hash
guarantees that two different configurations never share a directory. The full hash is stored in the
manifest and re-checked on resume.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from raq_kv.config.schema import ExperimentConfig

_DTYPE_SHORT = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def config_hash(cfg: ExperimentConfig) -> str:
    return sha256_hex(cfg.identity_dict())


def group_hash(cfg: ExperimentConfig) -> str:
    """Hash of everything except the seed: runs sharing it are replicates of the same condition."""
    identity = cfg.identity_dict()
    identity.pop("seed")
    return sha256_hex(identity)


def slug(cfg: ExperimentConfig) -> str:
    # M0 configs have no quantization / ECC / fault / pruning sections: they are full precision, unprotected,
    # fault-free and unpruned by construction. Later schema versions derive these tokens from their sections.
    device = cfg.model.device.split(":")[0]
    version = getattr(cfg.dataset, "version", "")
    parts = [
        cfg.model.short_name,
        f"{cfg.dataset.name}{version}",
        cfg.task.kind,
        _DTYPE_SHORT[cfg.model.dtype],
        "noecc",
        "nofault",
        "p0",
        device,
        f"s{cfg.seed}",
    ]
    return "_".join(parts)


def experiment_id(cfg: ExperimentConfig) -> str:
    return f"{slug(cfg)}-{config_hash(cfg)[:8]}"


def unit_id(exp_id: str, unit_key: str) -> str:
    return f"{exp_id}::{unit_key}"
