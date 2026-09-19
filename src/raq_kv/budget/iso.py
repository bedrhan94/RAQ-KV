"""Predicting an arm's bit budget before it runs, so iso-budget comparisons are enforced, not asserted.

The M3 question is whether *where* the bits go matters at a fixed total. That only means something if the
totals really are fixed, and discovering a 9% gap after a GPU campaign is too late. The functions here
reproduce the accounting of :mod:`raq_kv.quantization.base` and :class:`raq_kv.ecc.protect.ProtectedKV` in
closed form -- no tensors -- so a config pair can be checked in milliseconds. ``tests/test_iso_budget.py``
pins the prediction against the real encoders, which is what keeps the two paths from drifting apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from raq_kv.config.schema import ProtectionConfig, QuantizationConfig, QuantSpec
from raq_kv.ecc.codecs import make_codec
from raq_kv.models.arch import KVGeometry
from raq_kv.quantization.base import SCALE_BITS


def _n_groups(spec: QuantSpec, geometry: KVGeometry, tokens: int) -> int:
    """Number of quantization groups for one layer-part of shape ``[1, kv_heads, tokens, head_dim]``."""
    heads, dim = geometry.num_kv_heads, geometry.head_dim
    if spec.axis == "per_tensor":
        return 1
    if spec.axis == "per_head":
        return heads
    if spec.axis == "per_channel":
        return heads * dim
    if spec.axis == "per_token":
        return heads * tokens
    size = spec.group_size
    if spec.axis == "group_channel":
        return heads * tokens * -(-dim // size)
    if spec.axis == "group_token":
        return heads * dim * -(-tokens // size)
    raise ValueError(f"unknown axis {spec.axis!r}")


def _check_bits(stream_bits: int, codec_name: str, word_bits: int) -> int:
    """Check bits over one region's bit stream; the tail word is padded, and a real system pays for it too."""
    if codec_name == "none" or stream_bits == 0:
        return 0
    n_words = -(-stream_bits // word_bits)
    return n_words * make_codec(codec_name, word_bits).check_bits


@dataclass(frozen=True)
class ArmBudget:
    """Predicted per-arm storage, split so a reviewer can see which term pays for which claim."""

    data_bits: int
    quant_metadata_bits: int
    ecc_data_bits: int
    ecc_metadata_bits: int
    tokens: int
    kv_values: int

    @property
    def ecc_bits(self) -> int:
        return self.ecc_data_bits + self.ecc_metadata_bits

    @property
    def total_bits(self) -> int:
        return self.data_bits + self.quant_metadata_bits + self.ecc_bits

    @property
    def bits_per_value(self) -> float:
        """Effective bits per cached scalar: the axis M1 plots and the quantity M3 holds fixed."""
        return self.total_bits / self.kv_values

    def to_dict(self) -> dict[str, Any]:
        return {"data_bits": self.data_bits, "quant_metadata_bits": self.quant_metadata_bits,
                "ecc_data_bits": self.ecc_data_bits, "ecc_metadata_bits": self.ecc_metadata_bits,
                "ecc_bits": self.ecc_bits, "total_bits": self.total_bits,
                "bits_per_value": self.bits_per_value, "tokens": self.tokens}


def predict_budget(
    quantization: QuantizationConfig,
    protection: ProtectionConfig | None,
    geometry: KVGeometry,
    tokens: int,
) -> ArmBudget:
    """Closed-form budget of a quantized (and optionally protected) cache of ``tokens`` cached tokens."""
    if quantization.residual_window:
        # A residual window leaves recent tokens in the model dtype; predicting that needs the tail split, and
        # guessing it would understate an arm's budget. Refused rather than approximated (plan §44).
        raise NotImplementedError("residual_window is not covered by the closed-form predictor")
    per_part = geometry.num_kv_heads * tokens * geometry.head_dim
    layers = geometry.num_layers
    data = meta = ecc_data = ecc_meta = 0
    for spec in (quantization.keys, quantization.values):
        if spec is None:  # the part stays in the model dtype: unquantized, and nothing to protect
            data += per_part * geometry.dtype_bits * layers
            continue
        groups = _n_groups(spec, geometry, tokens)
        meta_per_layer = groups * SCALE_BITS * (1 if spec.symmetric else 2)
        data += per_part * spec.bits * layers
        meta += meta_per_layer * layers
        if protection is None:
            continue
        protected_layers = layers if protection.data_layers is None else len(set(protection.data_layers))
        ecc_data += _check_bits(per_part * spec.bits, protection.data_codec,
                                protection.data_word_bits) * protected_layers
        # Scales and zero points are encoded as separate regions, exactly as ProtectedKV does.
        n_meta_regions = 1 if spec.symmetric else 2
        ecc_meta += _check_bits(groups * SCALE_BITS, protection.metadata_codec,
                                protection.metadata_word_bits) * layers * n_meta_regions
    return ArmBudget(data, meta, ecc_data, ecc_meta, tokens, per_part * layers * 2)


def iso_budget_report(arms: dict[str, ArmBudget], tolerance: float = 0.01) -> dict[str, Any]:
    """Are these arms comparable? Spread is measured against the mean, and the verdict is explicit."""
    values = {name: arm.bits_per_value for name, arm in arms.items()}
    mean = sum(values.values()) / len(values)
    deviations = {name: (v - mean) / mean for name, v in values.items()}
    worst = max(deviations, key=lambda n: abs(deviations[n]))
    return {
        "bits_per_value": values,
        "mean_bits_per_value": mean,
        "relative_deviation": deviations,
        "max_abs_deviation": abs(deviations[worst]),
        "worst_arm": worst,
        "tolerance": tolerance,
        "iso_budget": abs(deviations[worst]) <= tolerance,
    }
