"""The closed-form budget predictor must agree with the encoders it predicts, to the bit.

If these drift apart, an "equal budget" comparison silently stops being equal, which is the single failure
mode that would invalidate the M3 claim. So the prediction is checked against a real quantize+protect run on
a small geometry, for every axis the project uses.
"""

from __future__ import annotations

import pytest
import torch

from raq_kv.budget.iso import iso_budget_report, predict_budget
from raq_kv.config.schema import ProtectionConfig, QuantizationConfig, QuantSpec
from raq_kv.ecc.protect import ProtectedKV
from raq_kv.kv.snapshot import KVSnapshot
from raq_kv.models.arch import KVGeometry
from raq_kv.quantization.apply import quantize_to_codes

GEOM = KVGeometry(num_layers=3, num_q_heads=4, num_kv_heads=2, head_dim=8, dtype_bits=16)
TOKENS = 10  # deliberately not a multiple of the group sizes, so tail groups and pad words are exercised


def _snapshot() -> KVSnapshot:
    g = torch.Generator().manual_seed(7)
    shape = (1, GEOM.num_kv_heads, TOKENS, GEOM.head_dim)
    return KVSnapshot(keys=[torch.randn(shape, generator=g) for _ in range(GEOM.num_layers)],
                      values=[torch.randn(shape, generator=g) for _ in range(GEOM.num_layers)],
                      positions=torch.arange(TOKENS), next_position=TOKENS)


AXES = [
    ("per_tensor", 4, False), ("per_head", 4, False), ("per_channel", 4, True),
    ("per_token", 4, False), ("group_channel", 4, False), ("group_token", 4, True),
    ("group_token", 3, False),  # group size that does not divide the token count
]


@pytest.mark.parametrize("axis,group,symmetric", AXES)
@pytest.mark.parametrize("bits", [2, 4])
def test_prediction_matches_the_real_quantizer_and_codec(axis, group, symmetric, bits):
    spec = QuantSpec(bits=bits, axis=axis, group_size=group, symmetric=symmetric)
    quant = QuantizationConfig(keys=spec, values=spec)
    prot = ProtectionConfig(data_codec="secded", data_word_bits=64, metadata_codec="secded", metadata_word_bits=16)

    qkv = quantize_to_codes(_snapshot(), quant)
    guard = ProtectedKV(qkv, data_codec="secded", data_word_bits=64, metadata_codec="secded",
                        metadata_word_bits=16)
    predicted = predict_budget(quant, prot, GEOM, TOKENS)

    assert predicted.data_bits == qkv.report.budget.kv_data_bits
    assert predicted.quant_metadata_bits == qkv.report.budget.quant_metadata_bits
    assert predicted.ecc_bits == guard.ecc_bits
    assert predicted.kv_values == GEOM.kv_values_per_token * TOKENS


def test_prediction_matches_metadata_only_and_layer_selective_policies():
    spec = QuantSpec(bits=4, axis="group_token", group_size=4, symmetric=False)
    quant = QuantizationConfig(keys=spec, values=spec)
    qkv = quantize_to_codes(_snapshot(), quant)

    meta_only = ProtectionConfig(metadata_codec="secded", metadata_word_bits=16)
    guard = ProtectedKV(qkv, metadata_codec="secded", metadata_word_bits=16)
    assert predict_budget(quant, meta_only, GEOM, TOKENS).ecc_bits == guard.ecc_bits
    assert predict_budget(quant, meta_only, GEOM, TOKENS).ecc_data_bits == 0

    selective = ProtectionConfig(data_codec="secded", data_word_bits=64, data_layers=(0, 2))
    guard_sel = ProtectedKV(qkv, data_codec="secded", data_word_bits=64, data_layers=(0, 2))
    assert predict_budget(quant, selective, GEOM, TOKENS).ecc_bits == guard_sel.ecc_bits


def test_unprotected_arm_costs_nothing_extra():
    spec = QuantSpec(bits=4, axis="group_token", group_size=4, symmetric=False)
    quant = QuantizationConfig(keys=spec, values=spec)
    arm = predict_budget(quant, None, GEOM, TOKENS)
    assert arm.ecc_bits == 0 and arm.total_bits == arm.data_bits + arm.quant_metadata_bits


def test_iso_budget_report_flags_a_mismatched_pair():
    # per_tensor keeps metadata negligible, so the gap here is the bit width itself.
    spec4 = QuantSpec(bits=4, axis="per_tensor", symmetric=False)
    spec2 = QuantSpec(bits=2, axis="per_tensor", symmetric=False)
    a = predict_budget(QuantizationConfig(keys=spec4, values=spec4), None, GEOM, TOKENS)
    b = predict_budget(QuantizationConfig(keys=spec2, values=spec2), None, GEOM, TOKENS)
    report = iso_budget_report({"A": a, "B": b})
    assert not report["iso_budget"] and report["max_abs_deviation"] > 0.1
    assert iso_budget_report({"A": a, "A2": a})["max_abs_deviation"] == 0.0
