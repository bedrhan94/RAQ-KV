"""Quality metrics computed from per-token log-probabilities."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


def nll_stats(token_nll: torch.Tensor) -> dict[str, float]:
    """Sum/mean negative log-likelihood (nats) and perplexity of one scored segment."""
    n = token_nll.numel()
    total = float(token_nll.double().sum())
    mean = total / n if n else float("nan")
    return {"nll_sum": total, "n_scored_tokens": n, "nll_mean": mean, "ppl": math.exp(mean) if n else float("nan")}


def corpus_ppl(nll_sums: Sequence[float], counts: Sequence[int]) -> float:
    """Token-weighted perplexity over segments: exp(sum NLL / sum tokens)."""
    tokens = sum(counts)
    return math.exp(sum(nll_sums) / tokens) if tokens else float("nan")


def kl_divergence(logp_ref: torch.Tensor, logp_other: torch.Tensor) -> torch.Tensor:
    """Per-position KL(P_ref || P_other) in nats, from log-probabilities over the last dimension."""
    if logp_ref.shape != logp_other.shape:
        raise ValueError("log-prob tensors must have identical shapes")
    p_ref = logp_ref.exp()
    return (p_ref * (logp_ref - logp_other)).sum(dim=-1)
