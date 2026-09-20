"""Turning a score vector into a retained cache, with the budget counted honestly.

Two things here are easy to get wrong and are therefore enforced rather than assumed.

**Index order.** ``KVSnapshot.select_tokens`` requires strictly increasing indices, while a top-k by score
returns them in score order. Sorting is not a detail: passing score-ordered indices would either raise or, worse
in a variant that did not check, silently permute the cache while keeping every shape valid.

**What "budget" means.** After pruning there are fewer cached tokens, so *bits per retained token* improves for
free and flatters every pruning policy. Comparisons use total bits and bits per **original** token, which is why
:class:`raq_kv.budget.bit_accounting.BitBudget` keeps ``retained_tokens`` as its own field.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from raq_kv.kv.snapshot import KVSnapshot


@dataclass(frozen=True)
class RetentionPlan:
    """Which cached tokens survive, and the accounting a report needs to describe the decision."""

    keep: torch.Tensor  # int64, strictly increasing
    original_tokens: int
    protected_recent: int
    protected_initial: int = 0

    @property
    def retained(self) -> int:
        return int(self.keep.numel())

    @property
    def ratio(self) -> float:
        return self.retained / self.original_tokens if self.original_tokens else 1.0

    def kept_fraction_of(self, tokens: tuple[int, ...]) -> float:
        """Share of a labelled token group that survived — the diagnostic column of the M4 table.

        A scorer that holds accuracy while discarding the answer-bearing tokens has not found importance; it has
        found that the model does not need them, which is a different and more interesting result.
        """
        if not tokens:
            return float("nan")
        kept = set(int(i) for i in self.keep.tolist())
        return sum(1 for t in tokens if t in kept) / len(tokens)

    def to_dict(self) -> dict[str, Any]:
        return {"tokens_original": self.original_tokens, "tokens_retained": self.retained,
                "retention_ratio": self.ratio, "protected_recent": self.protected_recent,
                "protected_initial": self.protected_initial}


def select_keep(scores: torch.Tensor, ratio: float, protect_recent: int = 0,
                protect_initial: int = 4) -> RetentionPlan:
    """Keep the highest-scoring ``ratio`` of tokens, always keeping the first ``protect_initial`` and the last
    ``protect_recent``.

    **The initial tokens are protected by default, and the default is not conservatism.** Position 0 is an
    attention sink: measured on this probe, dropping token 0 *alone* — one token of about a thousand — fails
    12 of 12 answers, while dropping ten middle filler tokens fails 1 of 12. A retention study that does not
    pin the sink measures the sink instead of its scorer, which is exactly what the first M4 gate did before
    this argument existed. StreamingLLM reports the same phenomenon and every practical method keeps an initial
    block for it.

    The recent tail is protected for a different reason: every published method keeps one (KIVI a 128-token
    residual window, H2O and SnapKV a recent block), so without it a scorer is partly judged on whether it
    happens to rediscover recency, which is not the question being asked.
    """
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"ratio must be in (0, 1], got {ratio}")
    n = int(scores.numel())
    if scores.ndim != 1:
        raise ValueError("scores must be 1-D, one per cached token")
    budget = max(1, min(n, round(n * ratio)))
    head = min(protect_initial, n, budget)
    tail = min(protect_recent, max(0, n - head), max(0, budget - head))
    mask = torch.ones(n, dtype=torch.bool)
    if head:
        mask[:head] = False
    if tail:
        mask[n - tail :] = False
    forced = torch.cat([torch.arange(head), torch.arange(n - tail, n) if tail else torch.empty(0, dtype=torch.long)])

    picked = torch.empty(0, dtype=torch.long)
    remaining = min(budget - head - tail, int(mask.sum()))
    if remaining > 0:
        candidates = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        picked = candidates[torch.topk(scores[candidates].to(torch.float32), remaining).indices]

    # Sorted, because select_tokens requires strictly increasing indices and top-k returns score order.
    keep = torch.cat([picked, forced.to(torch.long)]).to(torch.long).unique(sorted=True)
    return RetentionPlan(keep=keep, original_tokens=n, protected_recent=tail, protected_initial=head)


def apply_retention(snapshot: KVSnapshot, plan: RetentionPlan) -> KVSnapshot:
    """Prune the cache to the plan. Positions stay ABSOLUTE: a retained token keeps its original RoPE index."""
    if plan.retained == snapshot.seq_len:
        return snapshot
    return snapshot.select_tokens(plan.keep)
