from __future__ import annotations

import pytest
import torch
from transformers import Qwen3Config

from raq_kv.budget.bit_accounting import BitBudget, full_precision_kv_budget
from raq_kv.evaluation.protocol import prefill
from raq_kv.models.arch import KVGeometry


@pytest.mark.parametrize("n_tokens", [1, 7, 48])
def test_analytic_bytes_equal_actual_tensor_bytes(tiny_model, token_ids, n_tokens):
    geometry = KVGeometry.from_hf_config(tiny_model.config, torch.float32)
    snap = prefill(tiny_model, token_ids[:, :n_tokens])
    assert full_precision_kv_budget(geometry, snap.seq_len).total_bytes == snap.nbytes


def test_qwen3_1p7b_geometry_matches_verified_config():
    # Numbers read from the pinned Hugging Face config on 2026-09-17 (plan §D).
    cfg = Qwen3Config(
        num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8, head_dim=128, hidden_size=2048
    )
    geometry = KVGeometry.from_hf_config(cfg, torch.bfloat16)
    assert geometry.gqa_group_size == 2
    assert full_precision_kv_budget(geometry, 1).total_bytes == 114_688  # 112 KiB per token
    assert full_precision_kv_budget(geometry, 4096).total_bytes == 448 * 2**20


def test_hybrid_attention_models_are_rejected():
    cfg = Qwen3Config(num_hidden_layers=2, layer_types=["full_attention", "sliding_attention"], use_sliding_window=True)
    with pytest.raises(NotImplementedError):
        KVGeometry.from_hf_config(cfg, torch.bfloat16)


def test_budget_terms_and_pbrt():
    b = BitBudget(kv_data_bits=800, retained_tokens=10, quant_metadata_bits=16, ecc_bits=100, policy_metadata_bits=4)
    assert b.total_bits == 920 and b.total_bytes == 115
    assert b.ecc_overhead_ratio == pytest.approx(0.125)
    assert b.protected_bits_per_retained_token == pytest.approx(90.0)
    assert BitBudget(kv_data_bits=9, retained_tokens=1).total_bytes == 2  # rounds up
