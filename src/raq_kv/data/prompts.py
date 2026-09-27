"""Chat-formatted prompts split into a cached part (phase 1) and a query part (phase 2), with token-exact spans.

The full prompt is rendered with the model's chat template and tokenized once, so the phase-1/phase-2 split
never changes tokenization (verified: ``tokenize(rendered) == apply_chat_template(tokenize=True)`` for Qwen3).
The split is placed at a character offset and moved to the nearest token boundary on the left side, so a token
that straddles the offset belongs to phase 2.

Every span is mapped from characters to tokens via the fast tokenizer's offset mapping and checked at build
time: the decoded tokens must contain the span's surface text. An off-by-one here would silently corrupt every
token-level analysis (e.g. Q49), so the check raises instead of warning.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True)
class Span:
    label: str  # e.g. template_prefix | document | query | fact_sentence | fact_key | fact_value
    role: str  # context | query | target | distractor
    char_start: int
    char_end: int
    token_start: int  # [token_start, token_end) into prompt_ids; may include adjacent characters of edge tokens
    token_end: int
    text: str
    fact_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label, "role": self.role, "fact_type": self.fact_type,
            "token_start": self.token_start, "token_end": self.token_end,
            "char_start": self.char_start, "char_end": self.char_end, "text": self.text,
        }


@dataclass(frozen=True)
class PromptSample:
    sample_id: str
    prompt_text: str
    prompt_ids: torch.Tensor  # int64 [n]
    cache_len: int  # tokens [0, cache_len) are prefilled into the snapshot
    scorer: str  # key into evaluation.scoring.SCORERS
    reference: dict[str, Any]  # scorer inputs (gold value, distractor values, ...)
    gold_text: str | None  # gold continuation for teacher-forced answer NLL (None: not scored)
    gold_scored_chars: tuple[int, int] | None  # sub-span of gold_text whose tokens enter the NLL
    max_new_tokens: int
    spans: tuple[Span, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)


def render_chat(
    tokenizer: PreTrainedTokenizerBase,
    user_content: str,
    template_kwargs: dict[str, Any],
    tools: list[dict[str, Any]] | None = None,
) -> str:
    kwargs = dict(template_kwargs)
    if tools:
        kwargs["tools"] = tools
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}], tokenize=False, add_generation_prompt=True, **kwargs
    )


def render_conversation(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
    template_kwargs: dict[str, Any],
    tools: list[dict[str, Any]] | None = None,
) -> str:
    """Multi-turn analogue of :func:`render_chat`, for vehicles whose state lives in earlier turns.

    Separate from `render_chat` rather than a widened signature, because every existing sample's identity depends on
    that function producing byte-identical output: the probe, GSM8K and BFCL all hash their rendered prompts into a
    data fingerprint, and a refactor that changed one whitespace would orphan every stored run.

    The assistant turns are supplied by the caller and must be fixed strings. A vehicle whose intermediate turns
    were model-generated would not be deterministic, and the plan asks for "deterministik cok turlu" tasks for
    exactly that reason -- two runs of the same config have to build the same cache.
    """
    if not messages or messages[-1]["role"] != "user":
        raise ValueError("a conversation must end with a user turn, since the answer is what gets generated")
    kwargs = dict(template_kwargs)
    if tools:
        kwargs["tools"] = tools
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)


def tokenize_with_offsets(tokenizer: PreTrainedTokenizerBase, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    if not tokenizer.is_fast:
        raise TypeError("a fast tokenizer (offset mapping) is required")
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)
    return enc["input_ids"], [tuple(o) for o in enc["offset_mapping"]]


def split_index(offsets: list[tuple[int, int]], char_offset: int) -> int:
    """Number of leading tokens that end at or before ``char_offset`` (straddling tokens go to phase 2)."""
    n = 0
    for _start, end in offsets:
        if end > char_offset:
            break
        n += 1
    return n


def char_span_to_tokens(offsets: list[tuple[int, int]], char_start: int, char_end: int) -> tuple[int, int]:
    """Smallest token range covering every token that overlaps [char_start, char_end)."""
    if char_end <= char_start:
        raise ValueError("empty character span")
    covered = [i for i, (s, e) in enumerate(offsets) if s < char_end and e > char_start]
    if not covered:
        raise ValueError(f"no token overlaps characters [{char_start}, {char_end})")
    return covered[0], covered[-1] + 1


def make_span(
    tokenizer: PreTrainedTokenizerBase,
    ids: list[int],
    offsets: list[tuple[int, int]],
    text: str,
    label: str,
    role: str,
    char_start: int,
    char_end: int,
    fact_type: str | None = None,
    verify: bool = True,
) -> Span:
    t0, t1 = char_span_to_tokens(offsets, char_start, char_end)
    surface = text[char_start:char_end]
    if verify:
        decoded = tokenizer.decode(ids[t0:t1])
        if surface.strip() not in decoded:
            raise AssertionError(f"span '{label}' tokens [{t0},{t1}) decode to {decoded!r}, missing {surface!r}")
    return Span(label, role, char_start, char_end, t0, t1, surface, fact_type)


def samples_fingerprint(samples: list[PromptSample]) -> str:
    h = hashlib.sha256()
    for s in samples:
        h.update(s.sample_id.encode("utf-8"))
        h.update(s.prompt_ids.numpy().astype("<i8").tobytes())
        h.update(str(s.cache_len).encode())
        h.update(repr((s.reference, s.gold_text, s.gold_scored_chars)).encode("utf-8"))
        for span in s.spans:
            h.update(repr((span.label, span.role, span.token_start, span.token_end)).encode("utf-8"))
    return h.hexdigest()
