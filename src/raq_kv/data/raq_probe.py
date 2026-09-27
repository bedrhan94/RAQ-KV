"""RAQ-Probe v0: typed-fact retrieval with known token roles and paired depth variants.

Design (plan §E, decisions D9):
* Six fact types. ``number`` and ``constraint`` share the surface type (integers) and differ only in semantic role
  (plain fact vs. user rule), which separates role from surface form. ``tool_arg`` asks for a tool call whose
  argument must be copied from the context.
* Keys and names are invented from syllables, so answers cannot come from pretraining knowledge.
* Each sample has one target fact and ``n_distractors`` facts of the same type and template with different keys and
  values: critical-looking content that is irrelevant to the query (a control for Q49).
* Filler comes from WikiText-2 *train* (disjoint from the test split used for perplexity). Keys/values are checked
  not to occur in the filler, and every value occurs exactly once in the prompt.
* Paired depth variants: within a group, filler text and distractor slots are identical; only the target sentence
  moves. Depth effects can therefore be compared pairwise.
* Phase split: the cache holds the chat-template prefix, the intro and the document; the question and the assistant
  header are phase 2. Known confound: template/system tokens always occupy the first cache positions (attention
  sinks), and ``tool_arg`` prompts carry a longer tool-schema prefix.
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

from raq_kv.config.schema import RAQProbeConfig
from raq_kv.data.prompts import PromptSample, Span, make_span, render_chat, split_index, tokenize_with_offsets
from raq_kv.utils.seed import derive_seed

INTRO = "Read the following document carefully.\n\n"
DOC_QUERY_SEP = "\n\n"
VALUE_SUFFIX = " Reply with only the answer, copied exactly from the document."

BALANCE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_account_balance",
        "description": "Return the current balance of a customer account.",
        "parameters": {
            "type": "object",
            "properties": {"account_number": {"type": "string", "description": "The customer's account number."}},
            "required": ["account_number"],
        },
    },
}

# (fact sentence, question); {key} and {value} are filled per fact. Two phrasings per type.
TEMPLATES: dict[str, tuple[tuple[str, str], ...]] = {
    "number": (
        ("Storage room {key} currently holds exactly {value} sealed crates.",
         "How many sealed crates does storage room {key} currently hold?"),
        ("The {key} survey recorded a total of {value} nesting birds.",
         "How many nesting birds did the {key} survey record in total?"),
    ),
    "id": (
        ("The tracking code assigned to the {key} shipment is {value}.",
         "What is the tracking code assigned to the {key} shipment?"),
        ("Laboratory sample {key} was archived under reference {value}.",
         "Under which reference was laboratory sample {key} archived?"),
    ),
    "date": (
        ("The {key} contract takes effect on {value}.", "On what date does the {key} contract take effect?"),
        ("The inspection of the {key} bridge is scheduled for {value}.",
         "For what date is the inspection of the {key} bridge scheduled?"),
    ),
    "entity": (
        ("The project lead of the {key} initiative is {value}.", "Who is the project lead of the {key} initiative?"),
        ("The {key} expedition was organized by {value}.", "Who organized the {key} expedition?"),
    ),
    "constraint": (
        ("Instruction from the client: the total budget for the {key} trip must not exceed {value} euros.",
         "What is the maximum total budget in euros allowed for the {key} trip?"),
        ("Operating rule: the {key} reactor must never run above {value} kelvin.",
         "What is the highest temperature in kelvin at which the {key} reactor may run?"),
    ),
    "tool_arg": (
        ("The customer account number of {key} is {value}.",
         "Use the available tool to look up the current balance of {key}."),
        ("Customer {key} is registered under account number {value}.",
         "Look up the current account balance of customer {key} with the tool."),
    ),
}

_ONSETS = ("b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z", "br", "dr", "gr", "kr", "st", "th")
_VOWELS = ("a", "e", "i", "o", "u", "ae", "ai", "ou", "ei")
_CODAS = ("", "", "n", "r", "l", "s", "th", "m")
_ID_LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"  # no I / O


def _invented_word(rng: random.Random, syllables: int) -> str:
    parts = (rng.choice(_ONSETS) + rng.choice(_VOWELS) + rng.choice(_CODAS) for _ in range(syllables))
    return "".join(parts).capitalize()


def _make_key(rng: random.Random, fact_type: str) -> str:
    if fact_type == "tool_arg":
        return f"{_invented_word(rng, 2)} {_invented_word(rng, 3)}"
    return _invented_word(rng, rng.choice((2, 3)))


def _make_value(rng: random.Random, fact_type: str) -> str:
    # Integer ranges start above 2100 so values are not year-like (years are frequent in WikiText filler).
    if fact_type == "number":
        return str(rng.randint(2100, 9999))
    if fact_type == "constraint":
        return str(rng.randint(2100, 9800))
    if fact_type == "id":
        letters = lambda n: "".join(rng.choice(_ID_LETTERS) for _ in range(n))  # noqa: E731
        return f"{letters(3)}-{rng.randint(1000, 9999)}-{letters(2)}"
    if fact_type == "date":
        return f"{rng.randint(2027, 2031)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    if fact_type == "entity":
        return f"{_invented_word(rng, 2)} {_invented_word(rng, 3)}"
    if fact_type == "tool_arg":
        return str(rng.randint(10_000_000, 99_999_999))
    raise ValueError(fact_type)


@dataclass(frozen=True)
class Fact:
    fact_type: str
    role: str  # target | distractor
    key: str
    value: str
    sentence: str
    question: str
    key_offset: int  # character offset of the key inside ``sentence`` (from the template, not a search)
    value_offset: int


def fill_template(template: str, key: str, value: str) -> tuple[str, int, int]:
    """Substitute {key} and {value} and return the exact character offsets of both in the result."""
    k, v = template.index("{key}"), template.index("{value}")
    text = template.format(key=key, value=value)
    if k < v:
        return text, k, v - len("{key}") + len(key)
    return text, k - len("{value}") + len(value), v


# ---------------------------------------------------------------------------------------------------------- filler

_WIKI_FIXES = ((" @-@ ", "-"), (" @,@ ", ","), (" @.@ ", "."))
_WIKI_REGEX_FIXES = (
    (re.compile(r" ([.,;:!?%)\]])"), r"\1"),  # "poem ." -> "poem."
    (re.compile(r"([(\[$]) "), r"\1"),  # "( 1990" -> "(1990"
    (re.compile(r" (n't|'s|'re|'ve|'ll|'d|'m)\b"), r"\1"),  # "Tennyson 's" -> "Tennyson's"
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def clean_wikitext_paragraphs(lines: Sequence[str], min_chars: int = 200) -> list[str]:
    """Drop headings/short lines and undo the most visible WikiText tokenization artefacts."""
    paragraphs = []
    for line in lines:
        text = line.strip()
        if len(text) < min_chars or text.startswith("="):
            continue
        for old, new in _WIKI_FIXES:
            text = text.replace(old, new)
        for pattern, repl in _WIKI_REGEX_FIXES:
            text = pattern.sub(repl, text)
        paragraphs.append(text)
    return paragraphs


def load_filler_paragraphs(cfg: RAQProbeConfig) -> list[str]:
    ds = load_dataset(cfg.filler_hf_id, cfg.filler_subset, split=cfg.filler_split, revision=cfg.filler_revision)
    return clean_wikitext_paragraphs(ds["text"])


# --------------------------------------------------------------------------------------------------------- builder


class RAQProbeBuilder:
    def __init__(
        self,
        cfg: RAQProbeConfig,
        tokenizer: PreTrainedTokenizerBase,
        template_kwargs: dict[str, Any],
        filler_paragraphs: Sequence[str],
    ) -> None:
        if len(filler_paragraphs) < 10:
            raise ValueError("need at least 10 filler paragraphs")
        self.cfg = cfg
        self.tok = tokenizer
        self.template_kwargs = template_kwargs
        self.paragraphs = list(filler_paragraphs)
        self._sentences: dict[int, list[tuple[str, int]]] = {}

    def _paragraph_sentences(self, idx: int) -> list[tuple[str, int]]:
        if idx not in self._sentences:
            parts = [s.strip() for s in _SENTENCE_SPLIT.split(self.paragraphs[idx]) if s.strip()]
            self._sentences[idx] = [(s, len(self.tok(" " + s, add_special_tokens=False)["input_ids"])) for s in parts]
        return self._sentences[idx]

    def _facts(self, rng: random.Random, fact_type: str) -> list[Fact]:
        variant = rng.randrange(len(TEMPLATES[fact_type]))
        sentence_tpl, question_tpl = TEMPLATES[fact_type][variant]
        keys: list[str] = []
        values: list[str] = []
        while len(keys) < 1 + self.cfg.n_distractors:
            key, value = _make_key(rng, fact_type), _make_value(rng, fact_type)
            clashes = any(key in k or k in key for k in keys) or any(value in v or v in value for v in values)
            if not clashes:
                keys.append(key)
                values.append(value)
        facts = []
        for i, (k, v) in enumerate(zip(keys, values, strict=True)):
            sentence, k_off, v_off = fill_template(sentence_tpl, k, v)
            facts.append(Fact(fact_type, "target" if i == 0 else "distractor", k, v, sentence,
                              question_tpl.format(key=k), k_off, v_off))
        return facts

    def _prefix_tokens(self, fact_type: str) -> int:
        marker = "⁣DOC⁣"
        rendered = render_chat(self.tok, INTRO + marker, self.template_kwargs, self._tools(fact_type))
        return len(self.tok(rendered[: rendered.index(marker)], add_special_tokens=False)["input_ids"])

    @staticmethod
    def _tools(fact_type: str) -> list[dict[str, Any]] | None:
        return [BALANCE_TOOL] if fact_type == "tool_arg" else None

    def _filler(self, rng: random.Random, budget: int) -> tuple[list[str], list[int]]:
        """Consecutive sentences from a random paragraph until ``budget`` tokens; returns sentences and token counts."""
        start = rng.randrange(len(self.paragraphs))
        sentences: list[str] = []
        counts: list[int] = []
        p = start
        while sum(counts) < budget and p < start + len(self.paragraphs):
            for sent, n_tok in self._paragraph_sentences(p % len(self.paragraphs)):
                if sum(counts) >= budget:
                    break
                sentences.append(sent)
                counts.append(n_tok)
            p += 1
        return sentences, counts

    def build_group(self, fact_type: str, nominal_len: int, group_idx: int) -> list[PromptSample]:
        group_id = f"{fact_type}-L{nominal_len}-g{group_idx:02d}"
        rng = random.Random(derive_seed(self.cfg.generator_seed, "raq_probe", self.cfg.version, group_id))
        prefix_tokens = self._prefix_tokens(fact_type)
        for _attempt in range(100):  # redraw facts and filler until no key/value occurs in the filler
            facts = self._facts(rng, fact_type)
            fact_tokens = sum(len(self.tok(" " + f.sentence, add_special_tokens=False)["input_ids"]) for f in facts)
            filler, counts = self._filler(rng, max(16, nominal_len - prefix_tokens - fact_tokens - 1))
            filler_text = " ".join(filler).lower()
            if not any(s.lower() in filler_text for f in facts for s in (f.key, f.value)):
                break
        else:
            raise RuntimeError(f"{group_id}: could not find filler free of the probe's keys/values")

        # Gap g = insert before filler sentence g. Target gaps are chosen by filler-token fraction (not sentence
        # count), then distractors take random gaps that no depth variant uses, so they are identical across variants.
        tokens_before = [0]
        for n in counts:
            tokens_before.append(tokens_before[-1] + n)
        total = tokens_before[-1]
        target_gaps = [min(range(len(tokens_before)), key=lambda g, d=d: (abs(tokens_before[g] / total - d), g))
                       for d in self.cfg.depths]
        free = [g for g in range(len(tokens_before)) if g not in target_gaps]
        if len(set(target_gaps)) != len(target_gaps) or len(free) < self.cfg.n_distractors:
            raise RuntimeError(f"{group_id}: filler has too few sentences for distinct depths and distractors")
        distractor_gaps = sorted(rng.sample(free, k=self.cfg.n_distractors))
        return [
            self._render(group_id, fact_type, nominal_len, depth, facts, filler, gap, distractor_gaps)
            for depth, gap in zip(self.cfg.depths, target_gaps, strict=True)
        ]

    def _render(
        self,
        group_id: str,
        fact_type: str,
        nominal_len: int,
        depth: float,
        facts: list[Fact],
        filler: list[str],
        target_gap: int,
        distractor_gaps: list[int],
    ) -> PromptSample:
        inserts = {target_gap: facts[0], **{g: f for g, f in zip(distractor_gaps, facts[1:], strict=False)}}
        units: list[tuple[str, Fact | None]] = []
        for g in range(len(filler) + 1):
            if g in inserts:
                units.append((inserts[g].sentence, inserts[g]))
            if g < len(filler):
                units.append((filler[g], None))
        document = " ".join(u for u, _ in units)
        target = facts[0]
        is_tool = fact_type == "tool_arg"
        question = target.question if is_tool else target.question + VALUE_SUFFIX
        user_content = INTRO + document + DOC_QUERY_SEP + question
        tools = self._tools(fact_type)
        prompt = render_chat(self.tok, user_content, self.template_kwargs, tools)

        doc_start = prompt.index(INTRO + document) + len(INTRO)
        doc_end = doc_start + len(document)
        cache_end_char = doc_end + len(DOC_QUERY_SEP)
        ids, offsets = tokenize_with_offsets(self.tok, prompt)
        cache_len = split_index(offsets, cache_end_char)
        if prompt[cache_end_char:].find(question) != 0:
            raise AssertionError("query must start exactly at the cache boundary")

        spans: list[Span] = [
            Span("template_prefix", "context", 0, doc_start, 0, split_index(offsets, doc_start),
                 prompt[:doc_start]),
            make_span(self.tok, ids, offsets, prompt, "document", "context", doc_start, doc_end, verify=False),
            Span("query", "query", cache_end_char, len(prompt), cache_len, len(ids), prompt[cache_end_char:]),
        ]
        cursor = doc_start
        for text, fact in units:
            if fact is not None:
                s0 = cursor
                if (prompt[s0 + fact.key_offset : s0 + fact.key_offset + len(fact.key)] != fact.key
                        or prompt[s0 + fact.value_offset : s0 + fact.value_offset + len(fact.value)] != fact.value):
                    raise AssertionError(f"character offsets of fact {fact.key!r} do not match the rendered prompt")
                spans.append(make_span(self.tok, ids, offsets, prompt, "fact_sentence", fact.role, s0, s0 + len(text),
                                       fact_type))
                k0 = s0 + fact.key_offset
                spans.append(make_span(self.tok, ids, offsets, prompt, "fact_key", fact.role, k0, k0 + len(fact.key),
                                       fact_type))
                v0 = s0 + fact.value_offset
                spans.append(make_span(self.tok, ids, offsets, prompt, "fact_value", fact.role, v0,
                                       v0 + len(fact.value), fact_type))
            cursor += len(text) + 1

        for fact in facts:
            if prompt.count(fact.value) != 1:
                raise AssertionError(f"value {fact.value!r} must occur exactly once in the prompt")
        for span in spans:
            if span.role in ("target", "distractor") and not span.token_end <= cache_len:
                raise AssertionError(f"fact span {span.label} is not inside the cache")

        doc_tok0, doc_tok1 = spans[1].token_start, spans[1].token_end
        target_value_span = next(s for s in spans if s.label == "fact_value" and s.role == "target")
        depth_actual = (target_value_span.token_start - doc_tok0) / max(1, doc_tok1 - doc_tok0)

        if is_tool:
            prefix = '<tool_call>\n{"name": "get_account_balance", "arguments": {"account_number": "'
            gold_text = prefix + target.value + '"}}\n</tool_call>'
            scored = (len(prefix), len(prefix) + len(target.value))
            reference = {"gold": target.value, "distractors": [f.value for f in facts[1:]],
                         "tool_name": "get_account_balance", "arg_name": "account_number"}
            scorer = "tool_arg"
        else:
            gold_text = target.value
            scored = (0, len(target.value))
            reference = {"gold": target.value, "distractors": [f.value for f in facts[1:]]}
            scorer = "exact_value"

        depth_tag = f"d{round(depth * 100):03d}"
        return PromptSample(
            sample_id=f"{group_id}-{depth_tag}",
            prompt_text=prompt,
            prompt_ids=torch.tensor(ids, dtype=torch.long),
            cache_len=cache_len,
            scorer=scorer,
            reference=reference,
            gold_text=gold_text,
            gold_scored_chars=scored,
            max_new_tokens=self.cfg.max_new_tokens,
            spans=tuple(spans),
            meta={
                "group_id": group_id, "fact_type": fact_type, "nominal_len": nominal_len, "depth_nominal": depth,
                "depth_actual": round(depth_actual, 4), "cache_len": cache_len, "n_distractors": len(facts) - 1,
                "target_gap": target_gap, "distractor_gaps": distractor_gaps, "n_filler_sentences": len(filler),
            },
        )

    def build_all(self) -> list[PromptSample]:
        samples: list[PromptSample] = []
        for fact_type in self.cfg.fact_types:
            for nominal_len in self.cfg.context_lengths:
                for g in range(self.cfg.groups_per_cell):
                    samples.extend(self.build_group(fact_type, nominal_len, g))
        return samples


def build_raq_probe(
    cfg: RAQProbeConfig, tokenizer: PreTrainedTokenizerBase, template_kwargs: dict[str, Any]
) -> list[PromptSample]:
    """Single-turn probe samples, and **only** from a `RAQProbeConfig`.

    The type check is not defensive programming, it is a repair. `StateProbeConfig` is deliberately a superset of
    this config's field names so the two vehicles can share one document generator -- which means this function
    accepts one and **silently builds single-turn samples from it**. `ProtectionArm` and `FaultQA` call this
    directly rather than through `build_probe_samples`, so D69's first fault arm ran the single-turn probe while
    its manifest said `state_probe`, and the zero state regressions it reported were measured on a vehicle with no
    correction turn at all. Nothing crashed and nothing in the output looked wrong.

    The same shape of bug has happened here before: a scripted edit left `Retention.__init__` calling this function,
    and BFCL only revealed it by crashing on `filler_hf_id`. It crashed because BFCL's config has different fields.
    A config designed to share field names removes even that accident, so the refusal has to be explicit.
    """
    if not isinstance(cfg, RAQProbeConfig):
        raise TypeError(f"build_raq_probe was given a {type(cfg).__name__}; its field names overlap RAQProbeConfig's "
                        f"so this would silently build single-turn samples. Use build_probe_samples, which "
                        f"dispatches on the dataset.")
    return RAQProbeBuilder(cfg, tokenizer, template_kwargs, load_filler_paragraphs(cfg)).build_all()
