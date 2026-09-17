"""The measurement protocol must be numerically equivalent to the plain model where no transform is applied,
and pruning must be equivalent to masking. These tests double as the transformers cache-API probe: rerun them
after any transformers upgrade.
"""

from __future__ import annotations

import torch

from raq_kv.evaluation.protocol import greedy_decode, prefill, score_through_cache

ATOL = 1e-5


def test_two_phase_scoring_matches_single_pass(tiny_model, token_ids):
    prefix, cont = 32, 12
    with torch.no_grad():
        single = torch.log_softmax(tiny_model(token_ids[:, : prefix + cont]).logits[0].float(), -1)
    snap = prefill(tiny_model, token_ids[:, :prefix])
    res = score_through_cache(
        tiny_model, snap, token_ids[:, prefix : prefix + cont], token_ids[:, prefix + 1 : prefix + cont + 1],
        return_logprobs=True,
    )
    torch.testing.assert_close(res.logprobs, single[prefix : prefix + cont], atol=ATOL, rtol=0)
    expected_nll = -single[prefix : prefix + cont].gather(-1, token_ids[0, prefix + 1 : prefix + cont + 1, None])
    torch.testing.assert_close(res.token_nll, expected_nll.squeeze(-1), atol=ATOL, rtol=0)


def test_scoring_does_not_mutate_snapshot(tiny_model, token_ids):
    snap = prefill(tiny_model, token_ids[:, :32])
    before = snap.clone()
    score_through_cache(tiny_model, snap, token_ids[:, 32:40], token_ids[:, 33:41])
    for a, b in zip(before.keys + before.values, snap.keys + snap.values, strict=True):
        assert torch.equal(a, b)
    assert snap.seq_len == 32


def test_greedy_decode_matches_generate(tiny_model, token_ids):
    prompt = token_ids[:, :20]
    with torch.no_grad():
        ref = tiny_model.generate(prompt, max_new_tokens=8, do_sample=False, min_new_tokens=8, pad_token_id=0)
    ref_tokens = ref[0, prompt.shape[1] :].tolist()
    assert greedy_decode(tiny_model, prompt, max_new_tokens=8).tokens == ref_tokens
    # Same result when the first part of the prompt comes from a snapshot.
    snap = prefill(tiny_model, prompt[:, :15])
    assert greedy_decode(tiny_model, prompt[:, 15:], max_new_tokens=8, snapshot=snap).tokens == ref_tokens


def test_greedy_decode_stops_on_eos(tiny_model, token_ids):
    first = greedy_decode(tiny_model, token_ids[:, :20], max_new_tokens=1).tokens[0]
    out = greedy_decode(tiny_model, token_ids[:, :20], max_new_tokens=8, eos_token_ids={first})
    assert out.tokens == [first] and out.stopped_on_eos


def _masked_full_cache_logits(model, snap, chunk, pruned, n_steps):
    """Reference: full cache, pruned positions hidden by a 2D attention mask, then greedy steps."""
    cache = snap.to_cache()
    total = snap.seq_len + chunk.shape[1]
    mask = torch.ones(1, total, dtype=torch.long)
    mask[0, pruned] = 0
    pos = torch.arange(snap.next_position, snap.next_position + chunk.shape[1]).unsqueeze(0)
    with torch.no_grad():
        logits = model(chunk, past_key_values=cache, position_ids=pos, attention_mask=mask, use_cache=True).logits
        steps = [logits]
        next_pos = snap.next_position + chunk.shape[1]
        for _ in range(n_steps):
            tok = steps[-1][:, -1].argmax(-1, keepdim=True)
            mask = torch.cat([mask, torch.ones(1, 1, dtype=torch.long)], dim=1)
            steps.append(model(tok, past_key_values=cache, position_ids=torch.tensor([[next_pos]]),
                               attention_mask=mask, use_cache=True).logits)
            next_pos += 1
    return steps


def _pruned_cache_logits(model, snap, chunk, n_steps, explicit_positions=True):
    cache = snap.to_cache()
    pos = torch.arange(snap.next_position, snap.next_position + chunk.shape[1]).unsqueeze(0)
    with torch.no_grad():
        logits = model(chunk, past_key_values=cache, position_ids=pos if explicit_positions else None,
                       use_cache=True).logits
        steps = [logits]
        next_pos = snap.next_position + chunk.shape[1]
        for _ in range(n_steps):
            tok = steps[-1][:, -1].argmax(-1, keepdim=True)
            steps.append(model(tok, past_key_values=cache,
                               position_ids=torch.tensor([[next_pos]]) if explicit_positions else None,
                               use_cache=True).logits)
            next_pos += 1
    return steps


def test_middle_pruning_equals_masking(tiny_model, token_ids):
    prefix = 32
    pruned = [0, 5, 6, 13, 20, 31]
    keep = torch.tensor([i for i in range(prefix) if i not in pruned])
    full = prefill(tiny_model, token_ids[:, :prefix])
    pruned_snap = full.select_tokens(keep)
    assert pruned_snap.seq_len == prefix - len(pruned) and pruned_snap.next_position == prefix

    chunk = token_ids[:, prefix : prefix + 6]
    ref = _masked_full_cache_logits(tiny_model, full, chunk, pruned, n_steps=3)
    got = _pruned_cache_logits(tiny_model, pruned_snap, chunk, n_steps=3)
    for r, g in zip(ref, got, strict=True):
        torch.testing.assert_close(g, r, atol=ATOL, rtol=0)


def test_pruning_without_explicit_positions_is_detected(tiny_model, token_ids):
    """Guards the test above: if RoPE positions were derived from the compacted length, results must differ."""
    prefix = 32
    keep = torch.tensor([i for i in range(prefix) if i not in (3, 9, 10, 27)])
    pruned_snap = prefill(tiny_model, token_ids[:, :prefix]).select_tokens(keep)
    chunk = token_ids[:, prefix : prefix + 6]
    good = _pruned_cache_logits(tiny_model, pruned_snap, chunk, n_steps=0)[0]
    bad = _pruned_cache_logits(tiny_model, pruned_snap, chunk, n_steps=0, explicit_positions=False)[0]
    assert (good - bad).abs().max() > 1e-3
