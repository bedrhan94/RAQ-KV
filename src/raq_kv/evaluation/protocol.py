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
from typing import Any

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
    token_nll: torch.Tensor  # [n], natural log, in the requested log-prob dtype
    token_margin: torch.Tensor  # float32 [n]: logit(target) - max logit over all other tokens (< 0 => argmax flips)
    logprobs: torch.Tensor | None  # [n, vocab] when requested (for KL against golden)

    def slice(self, start: int) -> ScoreResult:
        return ScoreResult(
            self.token_nll[start:], self.token_margin[start:], None if self.logprobs is None else self.logprobs[start:]
        )


def target_margins(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """logit[target] - max_{v != target} logit[v] per position. Unlike NLL it does not saturate near p = 1."""
    logits = logits.float()
    target_logit = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    others = logits.scatter(-1, targets.unsqueeze(-1), float("-inf"))
    return target_logit - others.max(dim=-1).values


@torch.no_grad()
def score_through_cache(
    model: nn.Module,
    snapshot: KVSnapshot,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    return_logprobs: bool = False,
    logprob_dtype: torch.dtype = torch.float32,
) -> ScoreResult:
    """Teacher-forced scoring of ``target_ids`` given ``input_ids`` fed on top of the snapshot cache.

    ``logprob_dtype=torch.float64`` is needed when NLL values are tiny: in float32, log-probabilities of confident
    tokens round to exactly 0 (observed on RAQ-Probe answers), hiding small degradations.
    """
    _check_batch(input_ids)
    if target_ids.shape != input_ids.shape:
        raise ValueError("target_ids must have the same shape as input_ids")
    cache = snapshot.to_cache()
    logits = model(
        input_ids=input_ids,
        past_key_values=cache,
        position_ids=_positions(snapshot.next_position, input_ids.shape[1], input_ids.device),
        use_cache=True,
    ).logits[0]
    targets = target_ids[0]
    logprobs = torch.log_softmax(logits.to(logprob_dtype), dim=-1)
    token_nll = -logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return ScoreResult(token_nll, target_margins(logits, targets), logprobs if return_logprobs else None)


@torch.no_grad()
def score_answer(
    model: nn.Module,
    snapshot: KVSnapshot,
    query_ids: torch.Tensor,
    answer_ids: torch.Tensor,
    logprob_dtype: torch.dtype = torch.float64,
) -> ScoreResult:
    """Scores of ``answer_ids`` when the query is fed on top of the snapshot and the answer is teacher-forced.

    Entry j of the result refers to answer token j: -log p(answer_j | cache, query, answer_<j) and its logit margin.
    """
    _check_batch(query_ids)
    _check_batch(answer_ids)
    seq = torch.cat([query_ids, answer_ids], dim=1)
    res = score_through_cache(model, snapshot, seq[:, :-1], seq[:, 1:], logprob_dtype=logprob_dtype)
    return res.slice(query_ids.shape[1] - 1)


@dataclass
class GenerationResult:
    tokens: list[int]
    margins: list[float]  # per step: logit(chosen) - logit(runner-up); the decode-path distance to a different token
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
    margins: list[float] = []
    stopped = False
    for step in range(max_new_tokens):
        top2 = torch.topk(logits[0, -1].float(), 2)
        token = int(top2.indices[0].item())
        tokens.append(token)
        margins.append(float(top2.values[0] - top2.values[1]))
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
    return GenerationResult(tokens=tokens, margins=margins, stopped_on_eos=stopped, final_next_position=next_pos)


def token_char_spans(tokenizer: Any, tokens: list[int]) -> tuple[list[tuple[int, int]], str]:
    """Character span of each generated token inside the incrementally decoded text.

    Byte-level BPE makes per-token decoding unreliable, so spans come from the growth of the decoded prefix.
    """
    spans: list[tuple[int, int]] = []
    previous = ""
    for i in range(len(tokens)):
        current = tokenizer.decode(tokens[: i + 1], skip_special_tokens=True)
        spans.append((len(previous), len(current)))
        previous = current
    return spans, previous
