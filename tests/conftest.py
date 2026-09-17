"""Shared fixtures. Unit tests run on CPU with a tiny randomly initialized Qwen3 (no download, float32)."""

from __future__ import annotations

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM


def tiny_qwen3_config(attn_implementation: str = "sdpa") -> Qwen3Config:
    cfg = Qwen3Config(
        vocab_size=512,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=1024,
    )
    cfg._attn_implementation = attn_implementation
    return cfg


@pytest.fixture(params=["sdpa", "eager"])
def tiny_model(request: pytest.FixtureRequest) -> Qwen3ForCausalLM:
    torch.manual_seed(1234)
    return Qwen3ForCausalLM(tiny_qwen3_config(request.param)).float().eval()


@pytest.fixture
def token_ids() -> torch.Tensor:
    g = torch.Generator().manual_seed(7)
    return torch.randint(0, 512, (1, 48), generator=g)
