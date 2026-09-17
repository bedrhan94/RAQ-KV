"""Load a causal LM + tokenizer exactly as specified by :class:`ModelConfig` (pinned revision)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from raq_kv.config.schema import ModelConfig
from raq_kv.evaluation.memory import module_bytes
from raq_kv.models.arch import KVGeometry

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def resolve_dtype(name: str) -> torch.dtype:
    return _DTYPES[name]


@dataclass
class LoadedModel:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    geometry: KVGeometry
    device: torch.device
    dtype: torch.dtype
    memory: dict[str, int]


def load_tokenizer(cfg: ModelConfig) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(cfg.hf_id, revision=cfg.revision)


def load_model(cfg: ModelConfig) -> LoadedModel:
    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("config requests CUDA but torch.cuda.is_available() is False")
    dtype = resolve_dtype(cfg.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.hf_id,
        revision=cfg.revision,
        dtype=dtype,
        attn_implementation=cfg.attn_implementation,
    )
    model.to(device).eval()
    actual_impl = model.config._attn_implementation
    if actual_impl != cfg.attn_implementation:
        raise RuntimeError(f"requested attention '{cfg.attn_implementation}' but model uses '{actual_impl}'")
    return LoadedModel(
        model=model,
        tokenizer=load_tokenizer(cfg),
        geometry=KVGeometry.from_hf_config(model.config, dtype),
        device=device,
        dtype=dtype,
        memory=module_bytes(model),
    )
