"""Uniform (affine) integer quantization for KV tensors shaped ``[batch, kv_heads, tokens, head_dim]``.

One parametric implementation covers every bit width and granularity; the bit width is configuration, not code
(decisions D7). Granularities follow the KV-quantization literature:

* ``per_tensor``     - one scale for the whole layer tensor;
* ``per_head``       - one scale per KV head;
* ``per_channel``    - one scale per (head, channel), shared across tokens  -> KIVI's choice for keys;
* ``per_token``      - one scale per (head, token), shared across channels  -> KIVI's choice for values;
* ``group_channel``  - one scale per ``group_size`` consecutive channels within a token;
* ``group_token``    - one scale per (head, channel) within a block of ``group_size`` consecutive tokens.

``per_channel`` needs statistics over every token in the context, which a streaming decoder cannot have; it is
therefore an *offline / prefill-time* scheme. ``group_token`` is its causality-safe counterpart: a block is
quantized once it is complete, so no scale depends on tokens that have not arrived yet.

Scales/zeros are computed in float32 and stored in float16 (what a real implementation would hold), so the
reconstruction error includes the scale's own rounding.
"""

from __future__ import annotations

import torch

from raq_kv.quantization.base import QuantizedTensor

AXES = ("per_tensor", "per_head", "per_channel", "per_token", "group_channel", "group_token")
_REDUCE_DIMS = {"per_tensor": (1, 2, 3), "per_head": (2, 3), "per_channel": (2,), "per_token": (3,)}
_GROUPED_AXES = ("group_channel", "group_token")


def _grouped(x: torch.Tensor, group_size: int, axis: str) -> torch.Tensor:
    """Split the channel axis (``group_channel``) or the token axis (``group_token``) into groups.

    The token axis is padded up to a whole number of blocks so that a context length that is not a multiple of
    ``group_size`` still quantizes; padding is replicated from the last real token and never read back.
    """
    b, h, t, d = x.shape
    if axis == "group_channel":
        if d % group_size:
            raise ValueError(f"head_dim {d} is not divisible by group_size {group_size}")
        return x.reshape(b, h, t, d // group_size, group_size)
    pad = (-t) % group_size
    if pad:
        x = torch.cat([x, x[:, :, -1:, :].expand(b, h, pad, d)], dim=2)
    # [b, h, blocks, group, d] -> group over the tokens inside a block, per channel
    return x.reshape(b, h, (t + pad) // group_size, group_size, d).transpose(-1, -2)


def quantize(
    x: torch.Tensor, bits: int, axis: str = "per_token", group_size: int = 64, symmetric: bool = False
) -> QuantizedTensor:
    """Quantize a 4-D KV tensor. Returns codes plus float16 scales (and zero points when asymmetric)."""
    if x.ndim != 4:
        raise ValueError(f"expected [batch, kv_heads, tokens, head_dim], got {tuple(x.shape)}")
    if axis not in AXES:
        raise ValueError(f"unknown axis '{axis}' (expected one of {AXES})")
    qmax = (1 << bits) - 1
    grouped_axis = axis in _GROUPED_AXES
    work = _grouped(x, group_size, axis) if grouped_axis else x
    dims = (-1,) if grouped_axis else _REDUCE_DIMS[axis]
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
    if grouped_axis:
        codes = _ungroup(codes, x.shape, group_size, axis)
        scale = scale.squeeze(-1)
        zeros_store = None if zeros_store is None else zeros_store.squeeze(-1)

    stored_group = group_size if grouped_axis else None
    if symmetric:
        # Symmetric storage keeps only the scale; the fixed offset is reconstructed from `bits`.
        return _SymmetricQuantizedTensor(codes, scale, None, bits, axis, stored_group, x.dtype)
    return QuantizedTensor(codes, scale, zeros_store, bits, axis, stored_group, x.dtype)


def _ungroup(codes: torch.Tensor, shape: torch.Size, group_size: int, axis: str) -> torch.Tensor:
    """Inverse of :func:`_grouped` on the code tensor, dropping any token padding."""
    b, h, t, d = shape
    if axis == "group_channel":
        return codes.reshape(shape)
    return codes.transpose(-1, -2).reshape(b, h, -1, d)[:, :, :t, :]


class _SymmetricQuantizedTensor(QuantizedTensor):
    """Symmetric variant: codes are offset by half the range, so no zero point is stored."""

    def dequantize(self) -> torch.Tensor:
        return _reconstruct(self, offset=((1 << self.bits) - 1) // 2)


def _reconstruct(qt: QuantizedTensor, offset: int = 0) -> torch.Tensor:
    """code -> value, in the layout the scales were computed in (grouped axes carry one extra dimension)."""
    scales = qt.scales.to(torch.float32)
    zeros = None if qt.zeros is None else qt.zeros.to(torch.float32)
    if qt.axis in _GROUPED_AXES:
        assert qt.group_size is not None
        codes = _grouped(qt.codes, qt.group_size, qt.axis).to(torch.float32)
        out = (codes - offset) * scales.unsqueeze(-1)
        if zeros is not None:
            out = out + zeros.unsqueeze(-1)
        return _ungroup(out, qt.codes.shape, qt.group_size, qt.axis).to(qt.orig_dtype)
    out = (qt.codes.to(torch.float32) - offset) * scales
    if zeros is not None:
        out = out + zeros
    return out.to(qt.orig_dtype)


def dequantize(qt: QuantizedTensor) -> torch.Tensor:
    """Reconstruct a tensor, handling the grouped layouts where scales carry one fewer dimension."""
    if isinstance(qt, _SymmetricQuantizedTensor):
        return qt.dequantize()
    return _reconstruct(qt)


def quantize_dequantize(
    x: torch.Tensor, bits: int, axis: str = "per_token", group_size: int = 64, symmetric: bool = False
) -> tuple[torch.Tensor, QuantizedTensor]:
    qt = quantize(x, bits, axis, group_size, symmetric)
    return dequantize(qt), qt
