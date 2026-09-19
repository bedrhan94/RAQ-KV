"""Quantized tensor container.

Storage model (what a real system would hold, and what M2 corrupts):
* ``codes``: unsigned integers in ``[0, 2**bits - 1]``, one per original value, held in uint8 (bits <= 8).
  Bit order inside a code follows docs/conventions.md: LSB = bit 0.
* ``scales`` and ``zeros``: float16, one per quantization group, kept as broadcastable tensors.

Dequantization is ``code * scale + zero`` computed in the original dtype's accumulation precision. M1-M3 use
fake quantization: the dequantized tensor feeds the model, while memory is accounted analytically from
``data_bits`` + ``metadata_bits`` (decisions D6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from raq_kv.budget.bit_accounting import BitBudget

SCALE_BITS = 16  # float16 storage for scales and zero points


@dataclass(frozen=True)
class QuantizedTensor:
    codes: torch.Tensor  # uint8, same shape as the original tensor
    scales: torch.Tensor  # float16, broadcastable to the original shape
    zeros: torch.Tensor | None  # float16, broadcastable; None for symmetric quantization
    bits: int
    axis: str
    group_size: int | None
    orig_dtype: torch.dtype

    def __post_init__(self) -> None:
        if self.codes.dtype != torch.uint8:
            raise TypeError(f"codes must be uint8, got {self.codes.dtype}")
        if not 1 <= self.bits <= 8:
            raise ValueError(f"bits must be in 1..8, got {self.bits}")
        if int(self.codes.max()) > (1 << self.bits) - 1:
            raise ValueError("code out of range for the declared bit width")

    @property
    def shape(self) -> torch.Size:
        return self.codes.shape

    @property
    def n_values(self) -> int:
        return self.codes.numel()

    @property
    def n_groups(self) -> int:
        return self.scales.numel()

    def dequantize(self) -> torch.Tensor:
        out = self.codes.to(torch.float32) * self.scales.to(torch.float32)
        if self.zeros is not None:
            out = out + self.zeros.to(torch.float32)
        return out.to(self.orig_dtype)

    @property
    def data_bits(self) -> int:
        return self.n_values * self.bits

    @property
    def metadata_bits(self) -> int:
        """Scales (and zero points) actually stored, at ``SCALE_BITS`` each."""
        return self.n_groups * SCALE_BITS * (2 if self.zeros is not None else 1)

    def budget(self, retained_tokens: int) -> BitBudget:
        return BitBudget(
            kv_data_bits=self.data_bits, quant_metadata_bits=self.metadata_bits, retained_tokens=retained_tokens
        )

    def describe(self) -> dict[str, Any]:
        return {
            "bits": self.bits, "axis": self.axis, "group_size": self.group_size,
            "n_values": self.n_values, "n_groups": self.n_groups,
            "data_bits": self.data_bits, "metadata_bits": self.metadata_bits,
            "symmetric": self.zeros is None,
        }


def error_stats(original: torch.Tensor, reconstructed: torch.Tensor) -> dict[str, float]:
    """Quantization error of one tensor (float64 accumulation; SQNR in dB)."""
    o = original.to(torch.float64)
    err = reconstructed.to(torch.float64) - o
    mse = float((err**2).mean())
    signal = float((o**2).mean())
    return {
        "mse": mse,
        "rmse": mse**0.5,
        "max_abs_err": float(err.abs().max()),
        "sqnr_db": float("inf") if mse == 0 else 10.0 * torch.log10(torch.tensor(signal / mse)).item(),
    }
