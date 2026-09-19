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
def injected_burst(qkv: QuantizedKV, sites: list[FaultSite], base: KVSnapshot):
    """Inject several faults at once (a burst) and re-dequantize only the affected layer-parts.

    ``base`` is modified in place and yielded back (that is what avoids re-dequantizing 56 layer-parts per
    injection), so a caller that wants the fault-free tensors must copy them beforehand; everything is restored
    on exit.
    """
    if not sites:
        raise ValueError("empty burst")
    infos = [inject(qkv, site) for site in sites]
    touched = {(s.layer, s.part) for s in sites}
    saved = {}
    try:
        for layer, part in touched:
            tensors = base.keys if part == "keys" else base.values
            saved[(layer, part)] = tensors[layer]
            stored = qkv.part(layer, part)
            tensors[layer] = stored if isinstance(stored, torch.Tensor) else dequantize(stored)
        yield base, {"n_flips": len(sites),
                     "max_relative_change": max(i["relative_change"] for i in infos),
                     "mean_relative_change": sum(i["relative_change"] for i in infos) / len(infos)}
    finally:
        for (layer, part), tensor in saved.items():
            (base.keys if part == "keys" else base.values)[layer] = tensor
        for site in sites:
            inject(qkv, site)  # restore each flipped bit


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
