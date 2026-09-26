"""WikiText-2 chunks for cache-conditioned perplexity.

Each chunk has ``prefix_len + continuation_len + 1`` tokens:
* ``tokens[:prefix_len]``                        -> phase 1, stored in the KV cache;
* ``tokens[prefix_len : prefix_len + C]``        -> phase 2 inputs (fed on top of the cache);
* ``tokens[prefix_len + 1 : prefix_len + C + 1]`` -> targets, so every scored logit reads the cache.

This is not the standard sliding-window WikiText perplexity protocol; absolute values are therefore only
comparable across our own conditions, not number-for-number with published tables.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

from raq_kv.config.schema import WikiText2Config


@dataclass(frozen=True)
class PPLChunk:
    sample_id: str
    token_ids: torch.Tensor  # int64 [prefix_len + continuation_len + 1]
    prefix_len: int
    continuation_len: int

    @property
    def context_ids(self) -> torch.Tensor:
        return self.token_ids[: self.prefix_len]

    @property
    def read_input_ids(self) -> torch.Tensor:
        return self.token_ids[self.prefix_len : self.prefix_len + self.continuation_len]

    @property
    def read_target_ids(self) -> torch.Tensor:
        return self.token_ids[self.prefix_len + 1 : self.prefix_len + self.continuation_len + 1]


def tokenize_wikitext2(cfg: WikiText2Config, tokenizer: PreTrainedTokenizerBase) -> torch.Tensor:
    ds = load_dataset(cfg.hf_id, cfg.subset, split=cfg.split, revision=cfg.revision)
    text = "\n\n".join(ds["text"])
    ids = tokenizer(text, add_special_tokens=False, return_attention_mask=False, verbose=False)["input_ids"]
    return torch.tensor(ids, dtype=torch.long)


def make_chunks(all_ids: torch.Tensor, cfg: Any, prefix: str = "wt2") -> list[PPLChunk]:
    """Non-overlapping chunks of ``prefix_len + continuation_len + 1`` tokens.

    ``cfg`` is read structurally rather than by type: `BFCLConfig` carries the same three field names so the tool
    catalogue is chunked by this exact function (D66). That is deliberate -- a second chunker would be a second
    protocol, and the whole point of putting agent-domain text through perplexity is that the number is comparable
    to WikiText's, which requires the geometry to come from one place.
    """
    size = cfg.prefix_len + cfg.continuation_len + 1
    available = all_ids.numel() // size
    if available < cfg.n_chunks:
        raise ValueError(f"requested {cfg.n_chunks} chunks of {size} tokens but corpus provides only {available}")
    return [
        PPLChunk(
            sample_id=f"{prefix}-{i:04d}",
            token_ids=all_ids[i * size : (i + 1) * size].clone(),
            prefix_len=cfg.prefix_len,
            continuation_len=cfg.continuation_len,
        )
        for i in range(cfg.n_chunks)
    ]


def chunks_fingerprint(chunks: list[PPLChunk]) -> str:
    """sha256 over all chunk token ids; detects tokenizer or dataset drift between runs."""
    h = hashlib.sha256()
    for chunk in chunks:
        h.update(chunk.sample_id.encode("utf-8"))
        h.update(chunk.token_ids.numpy().astype("<i8").tobytes())
    return h.hexdigest()
