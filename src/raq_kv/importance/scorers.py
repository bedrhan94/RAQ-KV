"""Per-token importance scores over a cached context.

A scorer maps a :class:`KVSnapshot` (plus whatever side information it legitimately has) to one float per cached
token: higher means more worth keeping. Nothing here decides *how many* to keep — that is
:mod:`raq_kv.compression.retain`, so a scorer can be swapped without touching the budget arithmetic.

What a scorer is allowed to see matters as much as what it computes, so it is part of the interface:

* ``recency`` and the norm scorers see only the cache. Deployable as-is.
* ``window_attention`` additionally runs the last few context tokens through the model. Still deployable — it
  needs no knowledge of the query — but it costs a forward pass and eager attention (D5).
* the **oracles** see the probe's ground-truth spans. They are not methods and are never reported as one; they
  exist to bound what any scorer could achieve, and the more useful of the two is the *inverse* oracle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn

from raq_kv.kv.snapshot import KVSnapshot


@dataclass
class ScoreContext:
    """Everything a scorer may be given. A scorer that reads a field it should not have is a leak, so the fields
    are separated by what a deployed system would actually know."""

    snapshot: KVSnapshot
    context_ids: torch.Tensor | None = None  # [1, cache_len]: the cached tokens themselves
    model: nn.Module | None = None  # for scorers that need a forward pass
    target_tokens: tuple[int, ...] = ()  # ORACLE ONLY: cached indices holding the answer value
    distractor_tokens: tuple[int, ...] = ()  # ORACLE ONLY
    sentence_tokens: tuple[int, ...] = ()  # ORACLE ONLY: the whole sentence the answer sits in
    control_span_tokens: tuple[int, ...] = ()  # ORACLE ONLY: a same-sized contiguous window somewhere else


class Scorer(Protocol):
    name: str
    needs_model: bool
    is_oracle: bool

    def score(self, ctx: ScoreContext) -> torch.Tensor: ...


class Recency:
    """Position itself. The baseline every published retention method has to beat, and often does not."""

    name = "recency"
    needs_model = False
    is_oracle = False

    def score(self, ctx: ScoreContext) -> torch.Tensor:
        n = ctx.snapshot.seq_len
        return torch.arange(n, dtype=torch.float32)


class KeyNorm:
    """Mean L2 norm of a token's key vectors across layers and heads.

    A magnitude proxy, and the cheapest thing that could work. D13 found keys 11x more quantization-sensitive
    than values, which is the reason the key variant is the default of the two.
    """

    name = "key_norm"
    needs_model = False
    is_oracle = False
    part = "keys"

    def score(self, ctx: ScoreContext) -> torch.Tensor:
        tensors = ctx.snapshot.keys if self.part == "keys" else ctx.snapshot.values
        # [B, H, T, D] -> norm over D, mean over B, H and layers, all in float32 so bf16 does not round the tail.
        total = None
        for t in tensors:
            norms = t.to(torch.float32).norm(dim=-1).mean(dim=(0, 1))  # [T]
            total = norms if total is None else total + norms
        assert total is not None
        return (total / len(tensors)).cpu()


class ValueNorm(KeyNorm):
    name = "value_norm"
    part = "values"


class WindowAttention:
    """Attention paid to each cached token by the last ``window`` tokens of the context.

    Full prefill attention is not affordable: 28 layers x 16 heads x 4096^2 in bf16 is about 15 GB. An
    observation window is, and it is also what SnapKV and H2O use, so this matches the literature rather than
    inventing a variant. Requires eager attention, because ``sdpa`` returns no weights (D5).

    The window tokens are re-run against the *preceding* cache, so the score is "how much did the end of the
    context look at this token" — query-agnostic, and available before any query arrives.
    """

    name = "window_attention"
    needs_model = True
    is_oracle = False

    def __init__(self, window: int = 64) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window

    def score(self, ctx: ScoreContext) -> torch.Tensor:
        if ctx.model is None or ctx.context_ids is None:
            raise ValueError("window_attention needs the model and the context ids")
        n = ctx.snapshot.seq_len
        w = min(self.window, n - 1) if n > 1 else 0
        if w <= 0:  # nothing to observe from
            return torch.zeros(n, dtype=torch.float32)
        prefix = ctx.snapshot.select_tokens(torch.arange(n - w))
        device = ctx.context_ids.device
        window_ids = ctx.context_ids[:, n - w : n]
        out = ctx.model(
            input_ids=window_ids,
            past_key_values=prefix.to_cache(),
            position_ids=torch.arange(n - w, n, device=device, dtype=torch.long).unsqueeze(0),
            use_cache=True,
            output_attentions=True,
            logits_to_keep=1,
        )
        if not out.attentions or out.attentions[0] is None:
            raise RuntimeError("no attention weights returned; the model must use eager attention (D5)")
        # Each layer: [B, heads, w, n]. Sum the mass every window query sends to each cached position.
        scores = torch.zeros(n, dtype=torch.float32, device=device)
        for layer_attn in out.attentions:
            scores += layer_attn.to(torch.float32).sum(dim=(0, 1, 2))[:n]
        # The window tokens attend to themselves, which would simply re-rank the most recent tokens to the top
        # by construction. They are scored on the same footing as everything else instead: their self-attention
        # is left in, but `retain` protects the recent tail explicitly rather than relying on the scorer to.
        return (scores / len(out.attentions)).cpu()


class OracleKeepTarget:
    """Ground truth: the answer-bearing tokens rank highest. An upper bound, and a weak one.

    The probe is answerable from its target tokens by construction, so a ceiling here demonstrates the
    construction rather than any scorer's headroom. Kept because a ceiling that is *not* reached would mean
    something is wrong with the pruning path itself.
    """

    name = "oracle_keep_target"
    needs_model = False
    is_oracle = True

    def score(self, ctx: ScoreContext) -> torch.Tensor:
        n = ctx.snapshot.seq_len
        s = torch.arange(n, dtype=torch.float32) / max(n, 1)  # recency as the tie-break among unlabelled tokens
        idx = torch.tensor([i for i in ctx.target_tokens if i < n], dtype=torch.long)
        if idx.numel():
            s[idx] += 10.0
        return s


class OracleKeepSentence:
    """Keep the whole sentence the answer sits in, not just its value tokens.

    Measured motivation, not a hunch: an oracle that retained **100%** of the answer-value tokens still failed
    77% of shallow and 86% of middle-depth questions at 25% retention, while the same tokens dropped failed 98%.
    The value tokens are necessary and not sufficient, so the unit of retention is under test here — if keeping
    the sentence rescues those questions, importance is a property of spans rather than of tokens, and every
    token-level scorer inherits a ceiling it cannot cross.
    """

    name = "oracle_keep_sentence"
    needs_model = False
    is_oracle = True

    def score(self, ctx: ScoreContext) -> torch.Tensor:
        n = ctx.snapshot.seq_len
        s = torch.arange(n, dtype=torch.float32) / max(n, 1)
        idx = torch.tensor([i for i in ctx.sentence_tokens if i < n], dtype=torch.long)
        if idx.numel():
            s[idx] += 10.0
        return s


class OracleKeepRandomSpan:
    """The control for the span result: the answer tokens plus a same-sized contiguous window **elsewhere**.

    Keeping the fact's sentence beat keeping only its value tokens by 41 points (D34), but two readings survive
    that comparison. Either the value tokens need *their own* local context, or a contiguous block of context
    anywhere helps and the fact's neighbourhood is not special. This arm separates them: it matches the
    sentence arm on total budget, on value tokens retained, on contiguity and on window size, and differs only
    in **where** the window sits. If it recovers accuracy the gain was contiguity; if it does not, the gain was
    the fact's local context, which is the claim D34 makes.
    """

    name = "oracle_keep_random_span"
    needs_model = False
    is_oracle = True

    def score(self, ctx: ScoreContext) -> torch.Tensor:
        n = ctx.snapshot.seq_len
        s = torch.arange(n, dtype=torch.float32) / max(n, 1)
        for group in (ctx.target_tokens, ctx.control_span_tokens):
            idx = torch.tensor([i for i in group if i < n], dtype=torch.long)
            if idx.numel():
                s[idx] += 10.0
        return s


class OracleDropTarget:
    """The informative oracle: keep everything *except* the answer-bearing tokens.

    This asks whether the model recovers the fact from elsewhere in the context — the D19–D21 redundancy
    question, whose answer is genuinely unknown. If accuracy survives here, the target tokens are not
    load-bearing and no importance scorer can be evaluated on this probe at all.
    """

    name = "oracle_drop_target"
    needs_model = False
    is_oracle = True

    def score(self, ctx: ScoreContext) -> torch.Tensor:
        n = ctx.snapshot.seq_len
        s = torch.arange(n, dtype=torch.float32) / max(n, 1)
        idx = torch.tensor([i for i in ctx.target_tokens if i < n], dtype=torch.long)
        if idx.numel():
            s[idx] -= 10.0
        return s


def build_scorer(name: str, window: int = 64) -> Scorer:
    if name == "window_attention":
        return WindowAttention(window)
    table: dict[str, Scorer] = {
        s.name: s for s in (Recency(), KeyNorm(), ValueNorm(), OracleKeepTarget(),
                            OracleKeepSentence(), OracleKeepRandomSpan(), OracleDropTarget())
    }
    if name not in table:
        raise ValueError(f"unknown scorer {name!r}; have {sorted(table) + ['window_attention']}")
    return table[name]
