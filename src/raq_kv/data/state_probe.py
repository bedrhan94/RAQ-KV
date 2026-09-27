"""Deterministic multi-turn state carrying: does cache damage make the model answer from a stale value?

The plan's §M6 asks for "deterministik cok turlu durum tasiyan gorevler" and it was the last unbuilt deliverable.
BFCL cannot serve: its `multi_turn_long_context` category caches **157 tokens at the median** (measured in
`data/bfcl.py`), an order of magnitude short of this project's range, and its "long_context" comes from tool
responses accumulated during execution rather than from a static prompt -- which needs the stateful harness the plan
flagged as its own decision point.

**What a multi-turn vehicle adds that no single-turn one can: a wrong answer with a known provenance.** Three user
turns:

1. a document establishes ``<key> is <old_value>`` among same-type distractors;
2. a correction turn says the value has changed to ``<new_value>``;
3. the question asks for the value.

The gold answer is ``new_value``, and the **stale** value goes into the scorer's `distractors`, so
``mentions_distractor`` becomes a direct measure of **state regression** -- the model answering from turn 1 after
turn 2 superseded it. On a single-turn probe a damaged cache produces a wrong answer and nothing says where it came
from; here one specific wrong answer is diagnostic.

Both assistant turns are fixed strings. A vehicle whose intermediate turns were model-generated would not be
deterministic, and two runs of one config have to build the same cache for any of this project's provenance rules to
mean anything.

The document is built by `RAQProbeBuilder`, unchanged, so a state sample's document is the same object a probe
sample's is, its depth means the same thing, and its `fact_value` spans are comparable across the two vehicles.
"""

from __future__ import annotations

import random
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

from raq_kv.config.schema import StateProbeConfig
from raq_kv.data.prompts import (
    PromptSample,
    Span,
    make_span,
    render_conversation,
    split_index,
    tokenize_with_offsets,
)
from raq_kv.data.raq_probe import (
    INTRO,
    VALUE_SUFFIX,
    RAQProbeBuilder,
    _make_value,
    fill_template,
    load_filler_paragraphs,
)

# Fixed on both sides, so the conversation is reproducible from the config alone.
ACK_DOCUMENT = "Understood. I have read the document."
ACK_UPDATE = "Understood. I have recorded the change."
READ_REQUEST = "\n\nConfirm that you have read it."
UPDATE_PREFIX = "Correction to the record: "


class StateProbeBuilder:
    """Three-turn samples whose answer depends on a correction the cache has to have carried."""

    def __init__(self, cfg: StateProbeConfig, tokenizer: PreTrainedTokenizerBase,
                 template_kwargs: dict[str, Any]) -> None:
        self.cfg = cfg
        self.tok = tokenizer
        self.template_kwargs = template_kwargs
        # `StateProbeConfig` carries the field names `RAQProbeBuilder` reads, so this is a structural reuse rather
        # than a conversion: the document, the filler and the depth placement are the probe's own.
        self.inner = RAQProbeBuilder(cfg, tokenizer, template_kwargs, load_filler_paragraphs(cfg))

    def _new_value(self, rng: random.Random, fact_type: str, taken: list[str]) -> str:
        """A replacement value distinct from every value already in the document.

        Distinctness is not cosmetic. The scorer detects a stale answer by substring, so a new value overlapping the
        old one -- or any distractor -- would make `mentions_distractor` fire on a *correct* answer and the vehicle
        would report state regression that never happened.
        """
        for _ in range(1000):
            value = _make_value(rng, fact_type)
            if all(value not in t and t not in value for t in taken):
                return value
        raise RuntimeError(f"could not invent a value distinct from {taken}")

    def build_group(self, fact_type: str, nominal_len: int, group: int) -> list[PromptSample]:
        """One sample per depth, each reusing the probe's sample of the same (type, length, group, depth)."""
        inner = self.inner.build_group(fact_type, nominal_len, group)
        return [self._assemble(fact_type, nominal_len, group, depth, inner[i])
                for i, depth in enumerate(self.cfg.depths)]

    def _assemble(self, fact_type: str, nominal_len: int, group: int, depth: float,
                  inner: PromptSample) -> PromptSample:
        rng = random.Random(f"{self.cfg.generator_seed}|state|{fact_type}|{nominal_len}|{group}|{depth}")
        # Keyed lookup only for the roles that are unique. A dict keyed by (label, role) COLLAPSES the distractors,
        # which are several spans sharing one key -- the first version did that and silently reported one distractor
        # where the document had two, so the scorer's distractor list was short and a stale-answer measurement would
        # have missed regressions onto the dropped value.
        spans_in = {(s.label, s.role): s for s in inner.spans if s.role in ("context", "target", "query")}
        document = spans_in[("document", "context")].text
        target_key = spans_in[("fact_key", "target")].text
        old_value = spans_in[("fact_value", "target")].text
        distractors = [s.text for s in inner.spans if s.label == "fact_value" and s.role == "distractor"]

        new_value = self._new_value(rng, fact_type, [old_value, *distractors])
        # The correction reuses the document's own sentence template, so turn 2 states the fact the same way turn 1
        # did and the only difference between them is the value.
        target_sentence = spans_in[("fact_sentence", "target")].text
        tpl = target_sentence.replace(target_key, "{key}", 1).replace(old_value, "{value}", 1)
        correction, _, _ = fill_template(tpl, target_key, new_value)

        is_tool = fact_type == "tool_arg"
        ask = self._question(inner, is_tool)
        messages = [
            {"role": "user", "content": INTRO + document + READ_REQUEST},
            {"role": "assistant", "content": ACK_DOCUMENT},
            {"role": "user", "content": UPDATE_PREFIX + correction},
            {"role": "assistant", "content": ACK_UPDATE},
            {"role": "user", "content": ask},
        ]
        prompt = render_conversation(self.tok, messages, self.template_kwargs, self.inner._tools(fact_type))
        ids, offsets = tokenize_with_offsets(self.tok, prompt)

        # The cache holds turns 1-2 and turn 3's template scaffolding; the question itself is the query chunk, the
        # same split the single-turn probe uses.
        q_start = prompt.rindex(ask)
        cache_len = split_index(offsets, q_start)

        for value, what in ((old_value, "stale value"), (new_value, "updated value")):
            if prompt.count(value) != 1:
                raise AssertionError(f"{what} {value!r} occurs {prompt.count(value)} times; it must occur once, or "
                                     f"the spans and the substring scorer stop referring to one place")

        doc_start = prompt.index(document)
        spans: list[Span] = [
            Span("template_prefix", "context", 0, doc_start, 0, split_index(offsets, doc_start), prompt[:doc_start]),
            make_span(self.tok, ids, offsets, prompt, "document", "context", doc_start,
                      doc_start + len(document), verify=False),
            Span("query", "query", q_start, len(prompt), cache_len, len(ids), prompt[q_start:]),
        ]
        # The two values that make this vehicle what it is: superseded in turn 1, live in turn 2.
        for value, role in ((old_value, "stale"), (new_value, "target"), *[(d, "distractor") for d in distractors]):
            if prompt.count(value) != 1:
                continue  # a distractor value that the filler happens to repeat is left unlabelled, not asserted on
            at = prompt.index(value)
            spans.append(make_span(self.tok, ids, offsets, prompt, "fact_value", role, at, at + len(value),
                                   fact_type))
        for span in spans:
            if span.role in ("target", "stale", "distractor") and not span.token_end <= cache_len:
                raise AssertionError(f"{span.label}/{span.role} is not inside the cache")

        # The stale value is a DISTRACTOR for scoring, which is what turns `mentions_distractor` into a measure of
        # state regression rather than a generic wrong-answer flag.
        reference: dict[str, Any] = {"gold": new_value, "distractors": [old_value, *distractors]}
        scorer, gold_text, scored = "exact_value", new_value, (0, len(new_value))
        if is_tool:
            prefix = '<tool_call>\n{"name": "get_account_balance", "arguments": {"account_number": "'
            gold_text = prefix + new_value + '"}}\n</tool_call>'
            scored = (len(prefix), len(prefix) + len(new_value))
            reference |= {"tool_name": "get_account_balance", "arg_name": "account_number"}
            scorer = "tool_arg"

        group_id = f"state-{fact_type}-L{nominal_len}-g{group:02d}"
        return PromptSample(
            sample_id=f"{group_id}-d{round(depth * 100):03d}",
            prompt_text=prompt,
            prompt_ids=torch.tensor(ids, dtype=torch.long),
            cache_len=cache_len,
            scorer=scorer,
            reference=reference,
            gold_text=gold_text,
            gold_scored_chars=scored,
            max_new_tokens=self.cfg.max_new_tokens,
            spans=tuple(spans),
            meta={"group_id": group_id, "fact_type": fact_type, "nominal_len": nominal_len,
                  "depth_nominal": depth, "cache_len": cache_len, "n_distractors": len(distractors),
                  "stale_value": old_value, "updated_value": new_value, "n_turns": len(messages)},
        )

    def _question(self, inner: PromptSample, is_tool: bool) -> str:
        """The probe sample's own question, recovered from its query span and **stripped of template scaffolding**.

        Taken from the reused sample rather than regenerated, so turn 3 asks exactly what the single-turn probe asks
        and any difference in outcome between the two vehicles is the turns, not the wording.

        The stripping is the part that matters. The probe's query span runs to the **end of the prompt**, so it
        contains the question *and* the trailing chat-template scaffolding -- for Qwen,
        ``<|im_end|>\\n<|im_start|>assistant\\n<think>...``. Embedding that as a message's content made the rendered
        conversation end with two assistant headers, one of them closed and empty, and the model was being asked its
        question inside a malformed prompt. It affected every fact type, not only the tool ones where it was first
        noticed, and it would have invalidated the vehicle silently: the numbers looked plausible.

        Cut at the earliest special token rather than at a hardcoded `<|im_end|>`, so the vehicle does not acquire a
        dependency on one tokenizer's markers.
        """
        query = next(s.text for s in inner.spans if s.label == "query")
        cuts = [query.index(tokenspec) for tokenspec in self.tok.all_special_tokens if tokenspec in query]
        question = query[: min(cuts)].rstrip() if cuts else query.rstrip()
        if not question:
            raise AssertionError("the query span held nothing but template scaffolding")
        return question if is_tool or question.endswith(VALUE_SUFFIX) else question + VALUE_SUFFIX

    def build_all(self) -> list[PromptSample]:
        out: list[PromptSample] = []
        for fact_type in self.cfg.fact_types:
            for nominal_len in self.cfg.context_lengths:
                for g in range(self.cfg.groups_per_cell):
                    out.extend(self.build_group(fact_type, nominal_len, g))
        return out


def build_state_probe(cfg: StateProbeConfig, tokenizer: PreTrainedTokenizerBase,
                      template_kwargs: dict[str, Any]) -> list[PromptSample]:
    return StateProbeBuilder(cfg, tokenizer, template_kwargs).build_all()
