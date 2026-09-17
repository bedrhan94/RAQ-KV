from __future__ import annotations

import pytest
import torch
from transformers import DynamicCache

from raq_kv.evaluation.protocol import prefill, score_through_cache
from raq_kv.kv.snapshot import KVSnapshot, snapshot_from_cache


def test_snapshot_shapes_and_positions(tiny_model, token_ids):
    snap = prefill(tiny_model, token_ids[:, :20])
    cfg = tiny_model.config
    assert snap.num_layers == cfg.num_hidden_layers
    assert tuple(snap.keys[0].shape) == (1, cfg.num_key_value_heads, 20, cfg.head_dim)
    assert torch.equal(snap.positions, torch.arange(20)) and snap.next_position == 20


def test_to_cache_roundtrip_is_exact_and_not_aliased(tiny_model, token_ids):
    snap = prefill(tiny_model, token_ids[:, :20])
    cache = snap.to_cache()
    for layer, k, v in zip(cache.layers, snap.keys, snap.values, strict=True):
        assert torch.equal(layer.keys, k) and torch.equal(layer.values, v)
        assert layer.keys.data_ptr() != k.data_ptr()
    cache.layers[0].keys.zero_()
    assert not torch.equal(snap.keys[0], torch.zeros_like(snap.keys[0]))


def test_mutating_snapshot_changes_read_logits(tiny_model, token_ids):
    snap = prefill(tiny_model, token_ids[:, :32])
    inp, tgt = token_ids[:, 32:40], token_ids[:, 33:41]
    clean = score_through_cache(tiny_model, snap, inp, tgt).token_nll
    corrupted = snap.clone()
    corrupted.keys[1][0, 0, 4, :] += 5.0
    dirty = score_through_cache(tiny_model, corrupted, inp, tgt).token_nll
    assert (clean - dirty).abs().max() > 1e-4


def test_select_tokens_rejects_unsorted_indices(tiny_model, token_ids):
    snap = prefill(tiny_model, token_ids[:, :10])
    with pytest.raises(ValueError):
        snap.select_tokens(torch.tensor([3, 1]))


def test_snapshot_validation():
    k = torch.zeros(1, 2, 5, 4)
    with pytest.raises(ValueError):
        KVSnapshot(keys=[k], values=[k], positions=torch.arange(4), next_position=5)
    with pytest.raises(ValueError):
        KVSnapshot(keys=[k], values=[k], positions=torch.arange(5), next_position=4)


def test_snapshot_from_cache_rejects_other_types():
    with pytest.raises(TypeError):
        snapshot_from_cache(object(), torch.arange(1), 1)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        snapshot_from_cache(DynamicCache(), torch.arange(0), 0)
