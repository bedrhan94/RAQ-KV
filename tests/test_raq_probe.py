"""RAQ-Probe construction. Uses the pinned Qwen3 tokenizer from the local HF cache (skipped if unavailable) and
synthetic filler, so no dataset download is needed."""

from __future__ import annotations

import pytest
import torch

from raq_kv.config.schema import RAQProbeConfig
from raq_kv.data.raq_probe import RAQProbeBuilder, fill_template

REV = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
TEMPLATE_KWARGS = {"enable_thinking": False}


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", revision=REV, local_files_only=True)
    except OSError:
        pytest.skip("Qwen3 tokenizer not in local HF cache")


FILLER = [
    " ".join(
        f"Paragraph {chr(65 + p % 26)} discusses the history of the river valley and its old mills in detail. "
        f"Local farmers described the seasons, the markets and the roads that crossed the northern hills. "
        f"Several travellers wrote letters about the weather and the long journeys between the villages."
        for _ in range(2)
    )
    for p in range(40)
]


def make_builder(tokenizer, **overrides) -> RAQProbeBuilder:
    cfg = RAQProbeConfig.model_validate({
        "filler_revision": "b08601e04326c79dfdd32d625aee71d232d685c3", "context_lengths": [512],
        "groups_per_cell": 1, **overrides,
    })
    return RAQProbeBuilder(cfg, tokenizer, TEMPLATE_KWARGS, FILLER)


def test_fill_template_offsets():
    text, k, v = fill_template("Room {key} holds {value} crates.", "Velmora", "4821")
    assert text[k : k + 7] == "Velmora" and text[v : v + 4] == "4821"
    text, k, v = fill_template("{value} is the code of {key}.", "Velmora", "AB-1")
    assert text[k : k + 7] == "Velmora" and text[v : v + 4] == "AB-1"


@pytest.mark.parametrize("fact_type", ["number", "id", "date", "entity", "constraint", "tool_arg"])
def test_group_structure_and_spans(tokenizer, fact_type):
    samples = make_builder(tokenizer).build_group(fact_type, 512, 0)
    assert [s.meta["depth_nominal"] for s in samples] == [0.1, 0.5, 0.9]
    for s in samples:
        ids = s.prompt_ids.tolist()
        # tokenization of the rendered text equals the chat template's own tokenization
        assert ids == list(tokenizer(s.prompt_text, add_special_tokens=False)["input_ids"])
        assert 0.8 * 512 <= s.cache_len <= 1.2 * 512
        by_label = {}
        for span in s.spans:
            by_label.setdefault((span.label, span.role), []).append(span)
            if span.label.startswith("fact_"):
                assert span.token_end <= s.cache_len
                assert span.text.strip() in tokenizer.decode(ids[span.token_start : span.token_end])
                assert s.prompt_text[span.char_start : span.char_end] == span.text
        assert len(by_label[("fact_value", "target")]) == 1
        assert len(by_label.get(("fact_value", "distractor"), [])) == 2
        query = by_label[("query", "query")][0]
        assert query.token_start == s.cache_len and query.token_end == len(ids)
        target_value = by_label[("fact_value", "target")][0].text
        assert target_value == s.reference["gold"] and s.prompt_text.count(target_value) == 1
        assert s.prompt_text[s.spans[2].char_start :].split("\n")[0] not in s.prompt_text[: s.spans[2].char_start]
        if fact_type == "tool_arg":
            assert s.scorer == "tool_arg" and "get_account_balance" in s.prompt_text[: s.spans[1].char_start]
            g0, g1 = s.gold_scored_chars
            assert s.gold_text[g0:g1] == target_value


def test_depth_variants_share_filler_and_distractors(tokenizer):
    samples = make_builder(tokenizer).build_group("id", 512, 3)
    target_sentences = []
    for s in samples:
        target = next(sp for sp in s.spans if sp.label == "fact_sentence" and sp.role == "target")
        target_sentences.append(target.text)
        without_target = s.prompt_text.replace(" " + target.text, "").replace(target.text + " ", "")
        if s is samples[0]:
            reference_text = without_target
        assert without_target == reference_text
    assert len(set(target_sentences)) == 1
    depths = [s.meta["depth_actual"] for s in samples]
    assert depths == sorted(depths) and depths[0] < 0.3 and depths[-1] > 0.7


def test_token_char_spans_cover_generated_text(tokenizer):
    from raq_kv.evaluation.protocol import token_char_spans

    text = '<tool_call>\n{"name": "get_account_balance", "arguments": {"account_number": "25768669"}}\n</tool_call>'
    tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
    spans, decoded = token_char_spans(tokenizer, tokens)
    assert decoded == text
    assert spans[0][0] == 0 and spans[-1][1] == len(text)
    assert all(a <= b for a, b in spans) and all(x[1] == y[0] for x, y in zip(spans, spans[1:], strict=False))
    v0 = text.index("25768669")
    value_tokens = [i for i, (a, b) in enumerate(spans) if a < v0 + 8 and b > v0]
    assert value_tokens and "25768669" in tokenizer.decode([tokens[i] for i in value_tokens])


def test_builder_is_deterministic(tokenizer):
    a = make_builder(tokenizer).build_group("entity", 512, 1)
    b = make_builder(tokenizer).build_group("entity", 512, 1)
    for x, y in zip(a, b, strict=True):
        assert torch.equal(x.prompt_ids, y.prompt_ids) and x.spans == y.spans and x.reference == y.reference
    c = make_builder(tokenizer, generator_seed=1).build_group("entity", 512, 1)
    assert not torch.equal(a[0].prompt_ids, c[0].prompt_ids)
