"""Segmenting a cached context into spans, and retaining whole spans instead of scattered tokens.

D34 measured that the unit matters far more than the score: at a fixed 25% budget, a *perfect* token-level
oracle fails 56% of questions while keeping the fact's own sentence fails 15%, and a same-sized contiguous
window elsewhere fails 58% — so it is locality around the informative tokens, not contiguity, and not the
quality of the token ranking. That result used the probe's ground-truth sentence labels, which no deployed
system has. This module is the deployable version: segment by punctuation, score spans from whatever token
scorer is in use, and keep whole spans.

Segmentation is deliberately the cheapest thing that could work. A tokenizer's own pieces are inspected for
sentence-ending punctuation, which costs one pass over the ids and no model call. It is a modelling choice and
it is wrong sometimes — abbreviations, decimals, lists — and the honest framing is that a *better* segmenter
can only improve the result, so the numbers here are a lower bound on what span-level retention achieves.
"""

from __future__ import annotations

from typing import Any

import torch

from raq_kv.compression.retain import RetentionPlan

# A span ends after a token containing one of these. Newline is included because the probe's documents (and
# most real ones) separate items by line as often as by full stop.
_TERMINATORS = (".", "!", "?", "\n", "。", "؟")


def sentence_spans(tokenizer: Any, context_ids: torch.Tensor, min_len: int = 1) -> list[tuple[int, int]]:
    """Split ``[0, n)`` into ``[start, end)`` spans at sentence-ending tokens.

    Deployable by construction: one pass over the token pieces, no model call and no ground truth. Every token
    belongs to exactly one span and the spans cover the whole context, so a retention plan built from them can
    still spend the entire budget.
    """
    ids = context_ids.reshape(-1).tolist()
    pieces = tokenizer.convert_ids_to_tokens(ids)
    spans: list[tuple[int, int]] = []
    start = 0
    for i, piece in enumerate(pieces):
        text = piece.replace("Ġ", " ").replace("Ċ", "\n") if isinstance(piece, str) else ""
        if any(t in text for t in _TERMINATORS) and (i + 1 - start) >= min_len:
            spans.append((start, i + 1))
            start = i + 1
    if start < len(ids):
        spans.append((start, len(ids)))
    return spans


def select_keep_spans(
    scores: torch.Tensor,
    spans: list[tuple[int, int]],
    ratio: float,
    protect_recent: int = 0,
    protect_initial: int = 4,
    aggregate: str = "mean",
) -> RetentionPlan:
    """Keep whole spans, highest-scoring first, until the token budget is spent.

    ``aggregate="mean"`` ranks a span by its average token score, which is the neutral choice: ``sum`` would
    rank by length as much as by content, and ``max`` would let one outlier token carry a long span. The
    protected head and tail are kept as tokens, not spans, because the attention sink (D32) is a property of
    position 0 and not of whatever sentence happens to contain it.

    A span that does not fit the remaining budget is skipped rather than truncated: half a sentence is exactly
    the scattered-token condition this is meant to avoid.
    """
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"ratio must be in (0, 1], got {ratio}")
    if scores.ndim != 1:
        raise ValueError("scores must be 1-D, one per cached token")
    n = int(scores.numel())
    budget = max(1, min(n, round(n * ratio)))
    head = min(protect_initial, n, budget)
    tail = min(protect_recent, max(0, n - head), max(0, budget - head))

    keep: set[int] = set(range(head)) | set(range(n - tail, n))
    remaining = budget - len(keep)
    if remaining > 0 and spans:
        values = scores.to(torch.float32)
        ranked = []
        for start, end in spans:
            chunk = values[start:end]
            if chunk.numel() == 0:
                continue
            key = {"mean": chunk.mean(), "sum": chunk.sum(), "max": chunk.max()}[aggregate]
            ranked.append((float(key), start, end))
        ranked.sort(reverse=True)
        for _, start, end in ranked:
            new = [i for i in range(start, end) if i not in keep]
            if not new:
                continue
            if len(new) > remaining:
                continue  # skipped, not truncated: half a span is the condition this exists to avoid
            keep.update(new)
            remaining -= len(new)
            if remaining <= 0:
                break

    kept = torch.tensor(sorted(keep), dtype=torch.long)
    return RetentionPlan(keep=kept, original_tokens=n, protected_recent=tail, protected_initial=head)
