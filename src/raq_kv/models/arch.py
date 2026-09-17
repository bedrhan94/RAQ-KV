"""KV-cache geometry read from a Hugging Face config (never hard-coded per model).

The allocation axis ``h`` in R(l, h, t, b) is the *KV head*, not the query head: with grouped-query
attention several query heads share one cached key/value head.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class KVGeometry:
    num_layers: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    dtype_bits: int

    @property
    def gqa_group_size(self) -> int:
        """Number of query heads that read each KV head."""
        return self.num_q_heads // self.num_kv_heads

    @property
    def kv_values_per_token(self) -> int:
        """Scalars cached per token across all layers, heads and both K and V."""
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim

    @property
    def kv_bits_per_token(self) -> int:
        return self.kv_values_per_token * self.dtype_bits

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "gqa_group_size": self.gqa_group_size, "kv_values_per_token": self.kv_values_per_token}

    @classmethod
    def from_hf_config(cls, config: Any, dtype: torch.dtype) -> KVGeometry:
        text_cfg = config.get_text_config(decoder=True) if hasattr(config, "get_text_config") else config
        layer_types = getattr(text_cfg, "layer_types", None)
        if layer_types is not None and any(t != "full_attention" for t in layer_types):
            raise NotImplementedError(
                f"Only uniform full-attention models are supported; got layer types {sorted(set(layer_types))}"
            )
        num_q_heads = text_cfg.num_attention_heads
        num_kv_heads = getattr(text_cfg, "num_key_value_heads", None) or num_q_heads
        head_dim = getattr(text_cfg, "head_dim", None) or text_cfg.hidden_size // num_q_heads
        if num_q_heads % num_kv_heads != 0:
            raise ValueError(f"num_attention_heads={num_q_heads} not divisible by num_key_value_heads={num_kv_heads}")
        return cls(
            num_layers=text_cfg.num_hidden_layers,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype_bits=torch.finfo(dtype).bits,
        )
