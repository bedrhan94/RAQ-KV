"""Protected-bit budget of a KV cache.

Every memory number in the project (policy constraints, iso-budget comparisons, reported KV memory)
must come from this module, so that metadata and ECC bits can never be silently left out of one side
of a comparison. M0 implements the full-precision path; quantization metadata (M1), ECC check bits (M3)
and policy maps (M5) are added as explicit terms, never as implicit zeros.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from raq_kv.models.arch import KVGeometry


@dataclass(frozen=True)
class BitBudget:
    kv_data_bits: int
    retained_tokens: int
    quant_metadata_bits: int = 0
    ecc_bits: int = 0
    policy_metadata_bits: int = 0

    @property
    def total_bits(self) -> int:
        return self.kv_data_bits + self.quant_metadata_bits + self.ecc_bits + self.policy_metadata_bits

    @property
    def total_bytes(self) -> int:
        return -(-self.total_bits // 8)

    @property
    def ecc_overhead_ratio(self) -> float:
        return self.ecc_bits / self.kv_data_bits if self.kv_data_bits else 0.0

    @property
    def protected_bits_per_retained_token(self) -> float:
        """PBRT = (KV data bits + ECC bits) / retained tokens (exploratory metric, see plan §21)."""
        if self.retained_tokens == 0:
            return float("nan")
        return (self.kv_data_bits + self.ecc_bits) / self.retained_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "kv_data_bits": self.kv_data_bits,
            "quant_metadata_bits": self.quant_metadata_bits,
            "ecc_bits": self.ecc_bits,
            "policy_metadata_bits": self.policy_metadata_bits,
            "total_bits": self.total_bits,
            "total_bytes": self.total_bytes,
            "retained_tokens": self.retained_tokens,
            "ecc_overhead_ratio": self.ecc_overhead_ratio,
            "pbrt": self.protected_bits_per_retained_token,
        }


def full_precision_kv_budget(geometry: KVGeometry, retained_tokens: int, batch_size: int = 1) -> BitBudget:
    """Unquantized, unprotected cache: every cached scalar costs ``geometry.dtype_bits`` bits, no metadata."""
    if retained_tokens < 0 or batch_size < 1:
        raise ValueError("retained_tokens must be >= 0 and batch_size >= 1")
    return BitBudget(
        kv_data_bits=batch_size * retained_tokens * geometry.kv_bits_per_token,
        retained_tokens=retained_tokens,
    )
