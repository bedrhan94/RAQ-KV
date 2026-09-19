"""Bit manipulation with one fixed convention (docs/conventions.md §5): **LSB = bit 0**.

Two storage formats carry faults:
* integer codes — uint8 holding ``bits`` significant bits, so valid positions are ``0 .. bits-1``;
* quantization metadata — IEEE-754 float16 scales and zero points, positions ``0 .. 15`` with bit 15 the sign
  and bits 10-14 the exponent. A flip in an exponent bit of a scale rescales an entire group, which is why
  metadata is a fault site class of its own.
"""

from __future__ import annotations

import torch


def flip_code_bit(codes: torch.Tensor, index: tuple[int, ...], bit: int, bits: int) -> int:
    """Flip one bit of one integer code in place. Returns the new code value."""
    if not 0 <= bit < bits:
        raise ValueError(f"bit {bit} outside the {bits}-bit code")
    if codes.dtype != torch.uint8:
        raise TypeError(f"codes must be uint8, got {codes.dtype}")
    old = int(codes[index])
    new = old ^ (1 << bit)
    codes[index] = new
    return new


def flip_float16_bit(tensor: torch.Tensor, index: tuple[int, ...], bit: int) -> float:
    """Flip one bit of one float16 value in place (bit 15 = sign, 10-14 = exponent). Returns the new value."""
    if not 0 <= bit < 16:
        raise ValueError(f"bit {bit} outside float16")
    if tensor.dtype != torch.float16:
        raise TypeError(f"expected float16 metadata, got {tensor.dtype}")
    as_int = tensor[index].view(torch.int16)
    flipped = (as_int.to(torch.int32) ^ (1 << bit)).to(torch.int16)
    tensor[index] = flipped.view(torch.float16)
    return float(tensor[index])


def bit_significance(bit: int, bits: int) -> str:
    """Coarse label used when stratifying and reporting: MSB, upper, lower or LSB of a code."""
    if bit == bits - 1:
        return "msb"
    if bit == 0:
        return "lsb"
    return "upper" if bit >= bits / 2 else "lower"


def float16_field(bit: int) -> str:
    return "sign" if bit == 15 else ("exponent" if bit >= 10 else "mantissa")
