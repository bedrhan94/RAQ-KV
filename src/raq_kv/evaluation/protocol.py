"""Two-phase measurement protocol.

Phase 1 (prefill): the context is run once and its KV cache is captured as a :class:`KVSnapshot`.
Phase 2 (read):   later tokens are scored or generated *through* that snapshot.

Why: in a single forward pass the keys/values are produced and consumed in the same call, so quantizing,
pruning or corrupting the stored cache has no effect on the measured logits. Every metric that is meant to
reflect KV-cache degradation must therefore read the cache in phase 2. Transformations (quantization, ECC,
faults, pruning) are applied to the snapshot between the two phases; golden and perturbed runs share the
identical phase-1 tensors.

Rules enforced here (see docs/conventions.md):
* ``position_ids`` (RoPE) are always passed explicitly and continue from ``snapshot.next_position``;
* ``model.generate()`` is never used on a snapshot-derived cache (it re-derives positions from cache length);
* batch size is 1 on snapshot paths.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from raq_kv.kv.snapshot import KVSnapshot, snapshot_from_cache


def _check_batch(input_ids: torch.Tensor) -> None:
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError(f"expected input_ids of shape [1, n>0], got {tuple(input_ids.shape)}")


def _positions(start: int, n: int, device: torch.device) -> torch.Tensor:
    return torch.arange(start, start + n, device=device, dtype=torch.long).unsqueeze(0)


@torch.no_grad()
def prefill(model: nn.Module, context_ids: torch.Tensor) -> KVSnapshot:
    """Run the context and return its KV snapshot. Context logits are intentionally discarded."""
    _check_batch(context_ids)
    n = context_ids.shape[1]
    out = model(
        input_ids=context_ids,
        position_ids=_positions(0, n, context_ids.device),
        use_cache=True,
        logits_to_keep=1,
    )
    return snapshot_from_cache(out.past_key_values, torch.arange(n), next_position=n)


@dataclass
class ScoreResult:
    token_nll: torch.Tensor  # float32 [n], natural log
    logprobs: torch.Tensor | None  # float32 [n, vocab] when requested (for KL against golden)


@torch.no_grad()
def score_through_cache(
    model: nn.Module,
    snapshot: KVSnapshot,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    return_logprobs: bool = False,
) -> ScoreResult:
    """Teacher-forced scoring of ``target_ids`` given ``input_ids`` fed on top of the snapshot cache."""
    _check_batch(input_ids)
    if target_ids.shape != input_ids.shape:
        raise ValueError("target_ids must have the same shape as input_ids")
    cache = snapshot.to_cache()
    logits = model(
        input_ids=input_ids,
        past_key_values=cache,
        position_ids=_positions(snapshot.next_position, input_ids.shape[1], input_ids.device),
        use_cache=True,
    ).logits
    logprobs = torch.log_softmax(logits[0].float(), dim=-1)
    token_nll = -logprobs.gather(-1, target_ids[0].unsqueeze(-1)).squeeze(-1)
    return ScoreResult(token_nll=token_nll, logprobs=logprobs if return_logprobs else None)


@dataclass
class GenerationResult:
    tokens: list[int]
    stopped_on_eos: bool
    final_next_position: int


@torch.no_grad()
def greedy_decode(
    model: nn.Module,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    snapshot: KVSnapshot | None = None,
    eos_token_ids: set[int] | None = None,
    on_token: Callable[[int], None] | None = None,
) -> GenerationResult:
    """Greedy decoding with explicit RoPE positions.

    ``snapshot=None`` starts from an empty cache (used for latency measurement of the unmodified serving path).
    ``eos_token_ids=None`` never stops early (fixed-length decoding). ``on_token`` is called right after each
    token has been materialized on the host (``.item()`` synchronizes CUDA, so timestamps taken there are valid).
    """
    _check_batch(prompt_ids)
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")
    device = prompt_ids.device
    cache = snapshot.to_cache() if snapshot is not None else None
    next_pos = snapshot.next_position if snapshot is not None else 0

    out = model(
        input_ids=prompt_ids,
        past_key_values=cache,
        position_ids=_positions(next_pos, prompt_ids.shape[1], device),
        use_cache=True,
        logits_to_keep=1,
    )
    cache = out.past_key_values
    next_pos += prompt_ids.shape[1]
    logits = out.logits

    tokens: list[int] = []
    stopped = False
    for step in range(max_new_tokens):
        token = int(logits[0, -1].argmax().item())
        tokens.append(token)
        if on_token is not None:
            on_token(token)
        if eos_token_ids and token in eos_token_ids:
            stopped = True
            break
        if step == max_new_tokens - 1:
            break
        logits = model(
            input_ids=torch.tensor([[token]], device=device),
            past_key_values=cache,
            position_ids=_positions(next_pos, 1, device),
            use_cache=True,
        ).logits
        next_pos += 1
    return GenerationResult(tokens=tokens, stopped_on_eos=stopped, final_next_position=next_pos)
