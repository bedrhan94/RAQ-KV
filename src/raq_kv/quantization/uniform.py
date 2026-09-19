"""Uniform (affine) integer quantization for KV tensors shaped ``[batch, kv_heads, tokens, head_dim]``.

One parametric implementation covers every bit width and granularity; the bit width is configuration, not code
(decisions D7). Granularities follow the KV-quantization literature:

* ``per_tensor``     - one scale for the whole layer tensor;
* ``per_head``       - one scale per KV head;
* ``per_channel``    - one scale per (head, channel), shared across tokens  -> KIVI's choice for keys;
* ``per_token``      - one scale per (head, token), shared across channels  -> KIVI's choice for values;
* ``group_channel``  - one scale per ``group_size`` consecutive channels within a token.

Scales/zeros are computed in float32 and stored in float16 (what a real implementation would hold), so the
reconstruction error includes the scale's own rounding.
"""

from __future__ import annotations

import torch

from raq_kv.quantization.base import QuantizedTensor

AXES = ("per_tensor", "per_head", "per_channel", "per_token", "group_channel")
_REDUCE_DIMS = {"per_tensor": (1, 2, 3), "per_head": (2, 3), "per_channel": (2,), "per_token": (3,)}


def _grouped(x: torch.Tensor, group_size: int) -> torch.Tensor:
    b, h, t, d = x.shape
    if d % group_size:
        raise ValueError(f"head_dim {d} is not divisible by group_size {group_size}")
    return x.reshape(b, h, t, d // group_size, group_size)


def quantize(
    x: torch.Tensor, bits: int, axis: str = "per_token", group_size: int = 64, symmetric: bool = False
) -> QuantizedTensor:
    """Quantize a 4-D KV tensor. Returns codes plus float16 scales (and zero points when asymmetric)."""
    if x.ndim != 4:
        raise ValueError(f"expected [batch, kv_heads, tokens, head_dim], got {tuple(x.shape)}")
    if axis not in AXES:
        raise ValueError(f"unknown axis '{axis}' (expected one of {AXES})")
    qmax = (1 << bits) - 1
    work = _grouped(x, group_size) if axis == "group_channel" else x
    dims = (-1,) if axis == "group_channel" else _REDUCE_DIMS[axis]
    f32 = work.to(torch.float32)

    if symmetric:
        amax = f32.abs().amax(dim=dims, keepdim=True)
        levels = qmax // 2  # codes are offset by `levels` so that 0 maps to the middle of the range
        scale = (amax / levels).clamp_min(torch.finfo(torch.float16).tiny).to(torch.float16)
        codes = torch.round(f32 / scale.to(torch.float32)) + levels
        zeros_store = None
    else:
        lo = f32.amin(dim=dims, keepdim=True)
        hi = f32.amax(dim=dims, keepdim=True)
        scale = ((hi - lo) / qmax).clamp_min(torch.finfo(torch.float16).tiny).to(torch.float16)
        zeros_store = lo.to(torch.float16)
        codes = torch.round((f32 - zeros_store.to(torch.float32)) / scale.to(torch.float32))

    codes = codes.clamp_(0, qmax).to(torch.uint8)
    if axis == "group_channel":
        codes = codes.reshape(x.shape)
        scale = scale.squeeze(-1)
        zeros_store = None if zeros_store is None else zeros_store.squeeze(-1)

    if symmetric:
        # Symmetric storage keeps only the scale; the fixed offset is reconstructed from `bits`.
        return _SymmetricQuantizedTensor(codes, scale, None, bits, axis,
                                         group_size if axis == "group_channel" else None, x.dtype)
    return QuantizedTensor(codes, scale, zeros_store, bits, axis,
                           group_size if axis == "group_channel" else None, x.dtype)


class _SymmetricQuantizedTensor(QuantizedTensor):
    """Symmetric variant: codes are offset by half the range, so no zero point is stored."""

    def dequantize(self) -> torch.Tensor:
        levels = ((1 << self.bits) - 1) // 2
        scales = self.scales.to(torch.float32)
        if self.axis == "group_channel":
            b, h, t, d = self.codes.shape
            codes = self.codes.reshape(b, h, t, scales.shape[-1], -1).to(torch.float32)
            out = (codes - levels) * scales.unsqueeze(-1)
            return out.reshape(b, h, t, d).to(self.orig_dtype)
        return ((self.codes.to(torch.float32) - levels) * scales).to(self.orig_dtype)


def _dequantize_grouped(qt: QuantizedTensor) -> torch.Tensor:
    b, h, t, d = qt.codes.shape
    codes = qt.codes.reshape(b, h, t, qt.scales.shape[-1], -1).to(torch.float32)
    out = codes * qt.scales.to(torch.float32).unsqueeze(-1)
    if qt.zeros is not None:
        out = out + qt.zeros.to(torch.float32).unsqueeze(-1)
    return out.reshape(b, h, t, d).to(qt.orig_dtype)


def dequantize(qt: QuantizedTensor) -> torch.Tensor:
    """Reconstruct a tensor, handling the grouped layout where scales carry one fewer dimension."""
    if qt.axis == "group_channel" and not isinstance(qt, _SymmetricQuantizedTensor):
        return _dequantize_grouped(qt)
    return qt.dequantize()


def quantize_dequantize(
    x: torch.Tensor, bits: int, axis: str = "per_token", group_size: int = 64, symmetric: bool = False
) -> tuple[torch.Tensor, QuantizedTensor]:
    qt = quantize(x, bits, axis, group_size, symmetric)
    return dequantize(qt), qt
