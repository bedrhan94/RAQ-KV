"""Applying and undoing single-bit faults on a quantized KV snapshot.

Injection is in-place on the stored tensors and is exactly reversible (flipping the same bit twice restores the
original), so one prefill can be reused for many injections without re-quantizing. The context manager makes the
restore automatic, which is what keeps a campaign's golden reference valid throughout.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch

from raq_kv.faults.bit_utils import flip_code_bit, flip_float16_bit
from raq_kv.faults.plan import FaultSite
from raq_kv.kv.snapshot import KVSnapshot
from raq_kv.quantization.apply import QuantizedKV
from raq_kv.quantization.base import QuantizedTensor
from raq_kv.quantization.uniform import dequantize


def _target_tensor(qkv: QuantizedKV, site: FaultSite) -> tuple[QuantizedTensor, torch.Tensor]:
    part = qkv.part(site.layer, site.part)
    if not isinstance(part, QuantizedTensor):
        raise ValueError(f"layer {site.layer} {site.part} is not quantized; nothing to corrupt")
    if site.target == "code":
        return part, part.codes
    if site.target == "scale":
        return part, part.scales
    if part.zeros is None:
        raise ValueError("site targets a zero point but this tensor is symmetric (no zero points stored)")
    return part, part.zeros


def inject(qkv: QuantizedKV, site: FaultSite) -> dict[str, Any]:
    """Flip one bit in place and report what changed (old/new stored value)."""
    qt, tensor = _target_tensor(qkv, site)
    if site.target == "code":
        before = int(tensor[site.index])
        after = flip_code_bit(tensor, site.index, site.bit, qt.bits)
    else:
        before = float(tensor[site.index])
        after = flip_float16_bit(tensor, site.index, site.bit)
    return {"stored_before": before, "stored_after": after,
            "relative_change": _relative_change(before, after, qt, site)}


def _relative_change(before: float, after: float, qt: QuantizedTensor, site: FaultSite) -> float:
    if site.target == "code":
        return abs(after - before) / max(1, (1 << qt.bits) - 1)
    return float("inf") if before == 0 else abs(after - before) / abs(before)


@contextmanager
def injected(qkv: QuantizedKV, site: FaultSite):
    """Apply a fault for the duration of the block, then restore the original bit."""
    info = inject(qkv, site)
    try:
        yield info
    finally:
        inject(qkv, site)  # flipping the same bit again restores it


@contextmanager
def injected_snapshot(qkv: QuantizedKV, site: FaultSite, base: KVSnapshot):
    """Yield (snapshot, info) where only the corrupted layer/part is re-dequantized.

    A campaign runs hundreds of injections on one prefill; dequantizing all 56 layer-parts each time would
    dominate the runtime. Every other tensor is shared with ``base``, so the comparison stays exactly paired.
    """
    tensors = base.keys if site.part == "keys" else base.values
    original = tensors[site.layer]
    with injected(qkv, site) as info:
        part = qkv.part(site.layer, site.part)
        tensors[site.layer] = part if isinstance(part, torch.Tensor) else dequantize(part)
        try:
            yield base, info
        finally:
            tensors[site.layer] = original
