"""Detached snapshots of a Hugging Face ``DynamicCache`` (validated against transformers 5.17).

Why snapshots instead of a live cache subclass (M0-M2): golden and perturbed runs start from the *same*
prefill tensors, which makes paired comparisons exact by construction, and the code only depends on the
public ``DynamicCache.layers[i].keys / .values`` tensors, shaped ``[batch, kv_heads, tokens, head_dim]``.

Position convention (see docs/conventions.md): ``positions[i]`` is the ORIGINAL sequence position of cached
entry ``i``. After pruning, the physical cache length shrinks but RoPE positions of new tokens must continue
from ``next_position``. transformers derives the attention mask size and query offset from the physical cache
length automatically; ``position_ids`` (RoPE) must always be passed explicitly by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import DynamicCache
from transformers.cache_utils import DynamicLayer


@dataclass
class KVSnapshot:
    keys: list[torch.Tensor]
    values: list[torch.Tensor]
    positions: torch.Tensor  # int64 [tokens], strictly increasing original positions
    next_position: int

    def __post_init__(self) -> None:
        if len(self.keys) != len(self.values) or not self.keys:
            raise ValueError("keys and values must be non-empty lists of equal length")
        seq = self.keys[0].shape[-2]
        for k, v in zip(self.keys, self.values, strict=True):
            if k.shape != v.shape or k.shape[-2] != seq or k.ndim != 4:
                raise ValueError(f"inconsistent KV shapes: {tuple(k.shape)} vs {tuple(v.shape)}")
        if self.positions.shape != (seq,):
            raise ValueError(f"positions shape {tuple(self.positions.shape)} != ({seq},)")
        if seq and int(self.positions[-1]) >= self.next_position:
            raise ValueError("next_position must be greater than every cached position")

    @property
    def num_layers(self) -> int:
        return len(self.keys)

    @property
    def seq_len(self) -> int:
        return self.keys[0].shape[-2]

    @property
    def nbytes(self) -> int:
        return sum(k.nbytes + v.nbytes for k, v in zip(self.keys, self.values, strict=True))

    def clone(self) -> KVSnapshot:
        return KVSnapshot(
            keys=[k.clone() for k in self.keys],
            values=[v.clone() for v in self.values],
            positions=self.positions.clone(),
            next_position=self.next_position,
        )

    def select_tokens(self, keep: torch.Tensor) -> KVSnapshot:
        """Keep cache entries at physical indices ``keep`` (order preserved). Used by token pruning."""
        keep = torch.as_tensor(keep, dtype=torch.long)
        if keep.ndim != 1 or (keep.numel() > 1 and not bool((keep[1:] > keep[:-1]).all())):
            raise ValueError("keep must be a strictly increasing 1-D index tensor")
        dev_keep = keep.to(self.keys[0].device)
        return KVSnapshot(
            keys=[k.index_select(-2, dev_keep).contiguous() for k in self.keys],
            values=[v.index_select(-2, dev_keep).contiguous() for v in self.values],
            positions=self.positions.index_select(0, keep.to(self.positions.device)),
            next_position=self.next_position,
        )

    def to_cache(self) -> DynamicCache:
        """Build a fresh ``DynamicCache``; its tensors never alias the snapshot (``DynamicLayer.update`` copies)."""
        cache = DynamicCache(ddp_cache_data=list(zip(self.keys, self.values, strict=True)))
        if cache.layers[0].keys.data_ptr() == self.keys[0].data_ptr():  # guard against a future no-copy update
            raise RuntimeError("DynamicCache aliases snapshot storage; snapshot would be mutated by decoding")
        return cache


def snapshot_from_cache(cache: DynamicCache, positions: torch.Tensor, next_position: int) -> KVSnapshot:
    """Take ownership of a prefill cache's tensors (no copy; the cache object should be discarded)."""
    if not isinstance(cache, DynamicCache):
        raise TypeError(f"expected DynamicCache, got {type(cache).__name__}")
    layers = cache.layers
    if not layers or any(type(layer) is not DynamicLayer for layer in layers):
        kinds = sorted({type(layer).__name__ for layer in layers})
        raise TypeError(f"only full-attention DynamicLayer caches are supported, got {kinds}")
    return KVSnapshot(
        keys=[layer.keys for layer in layers],
        values=[layer.values for layer in layers],
        positions=positions.detach().to("cpu", torch.long),
        next_position=int(next_position),
    )
