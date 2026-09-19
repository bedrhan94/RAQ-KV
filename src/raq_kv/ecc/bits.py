"""Conversions between stored values and bit matrices.

Bit order follows docs/conventions.md §5: **LSB = bit 0**, so ``bits[..., 0]`` is the least significant bit of a
code. Every conversion here is exactly invertible; the tests assert round-trips rather than trusting the shapes.
"""

from __future__ import annotations

import torch


def codes_to_bits(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """uint8 codes -> uint8 bit matrix of shape ``[*codes.shape, bits]`` (LSB first)."""
    if codes.dtype != torch.uint8:
        raise TypeError(f"codes must be uint8, got {codes.dtype}")
    if not 1 <= bits <= 8:
        raise ValueError(f"bits must be in 1..8, got {bits}")
    shifts = torch.arange(bits, device=codes.device, dtype=torch.uint8)
    return (codes.unsqueeze(-1) >> shifts) & 1


def bits_to_codes(bit_matrix: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`codes_to_bits`."""
    bits = bit_matrix.shape[-1]
    weights = (1 << torch.arange(bits, device=bit_matrix.device, dtype=torch.int32))
    return (bit_matrix.to(torch.int32) * weights).sum(dim=-1).to(torch.uint8)


def float16_to_bits(tensor: torch.Tensor) -> torch.Tensor:
    """float16 -> uint8 bit matrix ``[*tensor.shape, 16]`` (LSB first; bit 15 is the sign)."""
    if tensor.dtype != torch.float16:
        raise TypeError(f"expected float16, got {tensor.dtype}")
    as_int = tensor.view(torch.int16).to(torch.int32) & 0xFFFF
    shifts = torch.arange(16, device=tensor.device, dtype=torch.int32)
    return ((as_int.unsqueeze(-1) >> shifts) & 1).to(torch.uint8)


def bits_to_float16(bit_matrix: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`float16_to_bits`."""
    weights = (1 << torch.arange(16, device=bit_matrix.device, dtype=torch.int32))
    as_int = (bit_matrix.to(torch.int32) * weights).sum(dim=-1)
    # Values above 0x7FFF are negative in two's complement; wrap before viewing as float16.
    return torch.where(as_int >= 0x8000, as_int - 0x10000, as_int).to(torch.int16).view(torch.float16)


def pack_words(bit_stream: torch.Tensor, word_bits: int) -> tuple[torch.Tensor, int]:
    """Reshape a flat bit vector into ``[n_words, word_bits]``, zero-padding the tail.

    Returns the matrix and the number of padding bits, which the caller needs to strip on the way back (and
    which the bit budget counts, because a real implementation stores them too).
    """
    if bit_stream.ndim != 1:
        raise ValueError("expected a flat bit vector")
    pad = (-bit_stream.numel()) % word_bits
    if pad:
        bit_stream = torch.cat([bit_stream, torch.zeros(pad, dtype=bit_stream.dtype, device=bit_stream.device)])
    return bit_stream.reshape(-1, word_bits), pad


def unpack_words(words: torch.Tensor, pad: int) -> torch.Tensor:
    flat = words.reshape(-1)
    return flat[: flat.numel() - pad] if pad else flat
