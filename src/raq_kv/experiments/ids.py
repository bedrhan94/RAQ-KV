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
_AXIS_SHORT = {"per_tensor": "T", "per_head": "H", "per_channel": "C", "per_token": "t", "group_channel": "g",
               "group_token": "G"}


def _quant_token(cfg: ExperimentConfig) -> str:
    """Compact description of the KV precision, e.g. ``k4C-v4t`` or ``k2g32-v2t-r32``; dtype when unquantized."""
    q = cfg.quantization
    if q is None:
        return _DTYPE_SHORT[cfg.model.dtype]
    parts = []
    for name, spec in (("k", q.keys), ("v", q.values)):
        if spec is None:
            parts.append(f"{name}{_DTYPE_SHORT[cfg.model.dtype]}")
        else:
            grouped = spec.axis in ("group_channel", "group_token")
            axis = _AXIS_SHORT[spec.axis] + (str(spec.group_size) if grouped else "")
            parts.append(f"{name}{spec.bits}{axis}{'s' if spec.symmetric else ''}")
    if q.residual_window:
        parts.append(f"r{q.residual_window}")
    return "-".join(parts)


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


def _protection_token(cfg: ExperimentConfig) -> str:
    """e.g. ``noecc``, ``mSEC`` (metadata only), ``dSEC64-mSEC``, ``dSEC64L0-6`` (layer-selective)."""
    p = cfg.protection
    if p is None:
        return "noecc"
    short = {"none": "", "parity": "PAR", "secded": "SEC"}
    parts = []
    if p.data_codec != "none":
        scope = "" if p.data_layers is None else f"L{min(p.data_layers)}-{max(p.data_layers)}"
        parts.append(f"d{short[p.data_codec]}{p.data_word_bits}{scope}")
    if p.metadata_codec != "none":
        parts.append(f"m{short[p.metadata_codec]}")
    if p.metadata_guard is not None:
        parts.append("mRNG")  # range-checked metadata: a plausibility guard, not a code
    return "-".join(parts) or "noecc"


_SCORER_SHORT = {"recency": "rec", "key_norm": "kn", "value_norm": "vn", "window_attention": "wa",
                 "oracle_keep_target": "oK", "oracle_drop_target": "oD"}


def _pruning_token(cfg: ExperimentConfig) -> str:
    """e.g. ``p0`` (nothing pruned), ``p50rec`` (half kept by recency), ``p25wa+r32`` (with a protected tail)."""
    c = cfg.compression
    if c is None or c.ratio >= 1.0:
        return "p0"
    tail = f"+r{c.protect_recent}" if c.protect_recent else ""
    return f"p{round(c.ratio * 100)}{_SCORER_SHORT[c.scorer]}{tail}"


def slug(cfg: ExperimentConfig) -> str:
    # M0 configs have no quantization / ECC / fault / compression sections: they are full precision,
    # unprotected, fault-free and unpruned by construction, and each token below reproduces the literal the
    # slug used before its section existed, so earlier experiment ids stay valid.
    device = cfg.model.device.split(":")[0]
    version = getattr(cfg.dataset, "version", "")
    parts = [
        cfg.model.short_name,
        f"{cfg.dataset.name}{version}",
        cfg.task.kind,
        _quant_token(cfg),
        _protection_token(cfg),
        "nofault",
        _pruning_token(cfg),
        device,
        f"s{cfg.seed}",
    ]
    return "_".join(parts)


def experiment_id(cfg: ExperimentConfig) -> str:
    return f"{slug(cfg)}-{config_hash(cfg)[:8]}"


def unit_id(exp_id: str, unit_key: str) -> str:
    return f"{exp_id}::{unit_key}"
