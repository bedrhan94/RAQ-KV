"""Apply quantization to a KV snapshot between the two measurement phases.

M1 uses fake quantization: keys/values are quantized and immediately reconstructed, so the model consumes the
reconstruction while memory is reported from the exact bit budget of the codes and scales. Keys and values get
their own specification, because the literature's best settings differ (KIVI: per-channel keys, per-token values).
A ``residual_window`` keeps the most recent tokens in full precision, as streaming implementations do.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from raq_kv.budget.bit_accounting import BitBudget
from raq_kv.config.schema import QuantizationConfig, QuantSpec  # noqa: TC001  (runtime use in signatures)
from raq_kv.kv.snapshot import KVSnapshot
from raq_kv.quantization.base import QuantizedTensor, error_stats
from raq_kv.quantization.uniform import dequantize, quantize, quantize_dequantize


@dataclass
class QuantReport:
    budget: BitBudget
    stats: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"kv_budget": self.budget.to_dict(), "quantization": self.stats}


@dataclass
class QuantizedKV:
    """A snapshot held in its stored form: integer codes plus scales, per layer and part.

    This is what a real system keeps in memory and therefore what M2 corrupts; ``to_snapshot`` performs the
    dequantization the model consumes. An entry is a plain tensor when that part is left in full precision.
    """

    keys: list[QuantizedTensor | torch.Tensor]
    values: list[QuantizedTensor | torch.Tensor]
    positions: torch.Tensor
    next_position: int
    report: QuantReport

    @property
    def num_layers(self) -> int:
        return len(self.keys)

    def part(self, layer: int, part: str) -> QuantizedTensor | torch.Tensor:
        if part not in ("keys", "values"):
            raise ValueError(f"part must be 'keys' or 'values', got {part!r}")
        return (self.keys if part == "keys" else self.values)[layer]

    def to_snapshot(self) -> KVSnapshot:
        return KVSnapshot(
            keys=[t if isinstance(t, torch.Tensor) else dequantize(t) for t in self.keys],
            values=[t if isinstance(t, torch.Tensor) else dequantize(t) for t in self.values],
            positions=self.positions,
            next_position=self.next_position,
        )


def _quantize_part(x: torch.Tensor, spec: QuantSpec | None, keep_last: int) -> tuple[torch.Tensor, int, int, dict]:
    """Quantize all but the last ``keep_last`` tokens. Returns tensor, data bits, metadata bits and error stats."""
    full_bits = x.numel() * torch.finfo(x.dtype).bits
    if spec is None:
        return x, full_bits, 0, {"bits": torch.finfo(x.dtype).bits, "quantized_tokens": 0}
    n_tokens = x.shape[-2]
    keep = min(keep_last, n_tokens)
    head, tail = x[..., : n_tokens - keep, :], x[..., n_tokens - keep :, :]
    if head.shape[-2] == 0:
        return x, full_bits, 0, {"bits": torch.finfo(x.dtype).bits, "quantized_tokens": 0}
    deq, qt = quantize_dequantize(head, spec.bits, spec.axis, spec.group_size, spec.symmetric)
    out = torch.cat([deq, tail], dim=-2) if keep else deq
    tail_bits = tail.numel() * torch.finfo(x.dtype).bits
    stats = {**qt.describe(), **error_stats(head, deq), "quantized_tokens": head.shape[-2],
             "residual_tokens": keep, "residual_bits": tail_bits}
    return out, qt.data_bits + tail_bits, qt.metadata_bits, stats


def quantize_region(
    snapshot: KVSnapshot, layer: int, part: str, spec: QuantSpec, head: int | None = None
) -> tuple[KVSnapshot, dict[str, Any]]:
    """Quantize exactly one region (one layer's keys or values, optionally a single KV head) and leave the rest
    untouched. Used by the sensitivity profiler; tensors of other layers are shared, not copied."""
    if part not in ("keys", "values"):
        raise ValueError(f"part must be 'keys' or 'values', got {part!r}")
    keys, values = list(snapshot.keys), list(snapshot.values)
    tensors = keys if part == "keys" else values
    original = tensors[layer]
    target = original if head is None else original[:, head : head + 1]
    deq, qt = quantize_dequantize(target, spec.bits, spec.axis, spec.group_size, spec.symmetric)
    if head is None:
        tensors[layer] = deq
    else:
        patched = original.clone()
        patched[:, head : head + 1] = deq
        tensors[layer] = patched
    stats = {"layer": layer, "part": part, "head": head, **qt.describe(), **error_stats(target, deq)}
    return KVSnapshot(keys=keys, values=values, positions=snapshot.positions,
                      next_position=snapshot.next_position), stats


def _quantize_part_codes(
    x: torch.Tensor, spec: QuantSpec | None
) -> tuple[QuantizedTensor | torch.Tensor, int, int, dict[str, Any]]:
    if spec is None:
        bits = torch.finfo(x.dtype).bits
        return x, x.numel() * bits, 0, {"bits": bits, "quantized_tokens": 0}
    qt = quantize(x, spec.bits, spec.axis, spec.group_size, spec.symmetric)
    stats = {**qt.describe(), **error_stats(x, dequantize(qt)), "quantized_tokens": x.shape[-2]}
    return qt, qt.data_bits, qt.metadata_bits, stats


def quantize_to_codes(snapshot: KVSnapshot, cfg: QuantizationConfig) -> QuantizedKV:
    """Quantize a snapshot and keep the codes — the stored form that fault injection corrupts.

    Not compatible with ``residual_window``: a residual window mixes two storage formats in one tensor, which the
    fault address space does not model yet.
    """
    if cfg.residual_window:
        raise NotImplementedError("code-level storage does not support a residual window yet")
    keys: list[QuantizedTensor | torch.Tensor] = []
    values: list[QuantizedTensor | torch.Tensor] = []
    data_bits = meta_bits = 0
    key_stats: list[dict[str, Any]] = []
    value_stats: list[dict[str, Any]] = []
    for k, v in zip(snapshot.keys, snapshot.values, strict=True):
        kq, kd, km, ks = _quantize_part_codes(k, cfg.keys)
        vq, vd, vm, vs = _quantize_part_codes(v, cfg.values)
        keys.append(kq)
        values.append(vq)
        data_bits += kd + vd
        meta_bits += km + vm
        key_stats.append(ks)
        value_stats.append(vs)
    report = _build_report(cfg, snapshot.seq_len, data_bits, meta_bits, key_stats, value_stats)
    return QuantizedKV(keys=keys, values=values, positions=snapshot.positions,
                       next_position=snapshot.next_position, report=report)


def quantize_snapshot(snapshot: KVSnapshot, cfg: QuantizationConfig) -> tuple[KVSnapshot, QuantReport]:
    """Return a fake-quantized copy of the snapshot plus its exact bit budget and per-layer error statistics."""
    if not cfg.residual_window:
        qkv = quantize_to_codes(snapshot, cfg)
        return qkv.to_snapshot(), qkv.report
    keys, values = [], []
    data_bits = meta_bits = 0
    key_stats: list[dict[str, Any]] = []
    value_stats: list[dict[str, Any]] = []
    for k, v in zip(snapshot.keys, snapshot.values, strict=True):
        kq, kd, km, ks = _quantize_part(k, cfg.keys, cfg.residual_window)
        vq, vd, vm, vs = _quantize_part(v, cfg.values, cfg.residual_window)
        keys.append(kq)
        values.append(vq)
        data_bits += kd + vd
        meta_bits += km + vm
        key_stats.append(ks)
        value_stats.append(vs)
    report = _build_report(cfg, snapshot.seq_len, data_bits, meta_bits, key_stats, value_stats)
    return KVSnapshot(keys=keys, values=values, positions=snapshot.positions,
                      next_position=snapshot.next_position), report


def _build_report(
    cfg: QuantizationConfig, retained_tokens: int, data_bits: int, meta_bits: int,
    key_stats: list[dict[str, Any]], value_stats: list[dict[str, Any]],
) -> QuantReport:
    def _mean(rows: list[dict[str, Any]], field: str) -> float | None:
        vals = [r[field] for r in rows if field in r]
        return sum(vals) / len(vals) if vals else None

    return QuantReport(
        budget=BitBudget(kv_data_bits=data_bits, quant_metadata_bits=meta_bits, retained_tokens=retained_tokens),
        stats={
            "keys": {"spec": None if cfg.keys is None else cfg.keys.model_dump(),
                     "mse_mean": _mean(key_stats, "mse"), "sqnr_db_mean": _mean(key_stats, "sqnr_db"),
                     "max_abs_err": max((r.get("max_abs_err", 0.0) for r in key_stats), default=0.0)},
            "values": {"spec": None if cfg.values is None else cfg.values.model_dump(),
                       "mse_mean": _mean(value_stats, "mse"), "sqnr_db_mean": _mean(value_stats, "sqnr_db"),
                       "max_abs_err": max((r.get("max_abs_err", 0.0) for r in value_stats), default=0.0)},
            "per_layer_key_sqnr_db": [r.get("sqnr_db") for r in key_stats],
            "per_layer_value_sqnr_db": [r.get("sqnr_db") for r in value_stats],
            "residual_window": cfg.residual_window,
        },
    )
