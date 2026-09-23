"""BFCL as a long-cache agent vehicle, and an AST scorer that does not encode one family's dialect.

The vehicle exists because BFCL's own categories were measured first and are too short: 157-333 cache tokens at
the median against the 512-4096 this project works at. The catalogue construction is the fix. The scorer exists
because D46 measured what a dialect-encoding tool-call scorer costs -- Llama-3.2 scored 0 of 120 on RAQ-Probe's
`tool_arg` type for writing `parameters` instead of `arguments`, while retrieving the right value every time.

These tests do not touch the network: the vehicle's builder is exercised in `tests/test_bfcl_vehicle_gpu.py`
territory only, and what is pinned here is the scoring and span logic, which is where the mistakes are.
"""

from __future__ import annotations

import pytest

from raq_kv.config.schema import BFCLConfig
from raq_kv.evaluation.scoring import SCORERS, parse_any_calls, score_bfcl_ast, score_tool_arg

GT = [{"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}]
REF = {"ground_truth": GT, "gold_names": ["calculate_triangle_area"]}


@pytest.mark.parametrize("label,prediction,expected", [
    # The two dialects D46 found. Both must score the same, because both called the same function the same way.
    ("qwen tagged + arguments",
     '<tool_call>{"name": "calculate_triangle_area", "arguments": {"base": 10, "height": 5, "unit": "units"}}'
     "</tool_call>", True),
    ("llama bare + parameters",
     '{"name": "calculate_triangle_area", "parameters": {"base": 10, "height": 5}}', True),
    ("args alias", '{"name": "calculate_triangle_area", "args": {"base": 10, "height": 5}}', True),
    # `unit` accepts "" so it is optional; `base` and `height` are not.
    ("optional parameter omitted",
     '{"name": "calculate_triangle_area", "arguments": {"base": 10, "height": 5}}', True),
    ("numbers as strings",
     '{"name": "calculate_triangle_area", "arguments": {"base": "10", "height": "5"}}', True),
    ("wrong argument value",
     '{"name": "calculate_triangle_area", "arguments": {"base": 99, "height": 5}}', False),
    ("required argument missing",
     '{"name": "calculate_triangle_area", "arguments": {"base": 10}}', False),
    ("wrong function", '{"name": "calculate_circle_area", "arguments": {"radius": 10}}', False),
    ("prose with the right number", "The area is 25 square units, computed from base 10 and height 5.", False),
    ("empty", "", False),
])
def test_the_ast_scorer_reads_what_was_called_not_how_it_was_written(label, prediction, expected):
    assert score_bfcl_ast(prediction, REF)["correct"] is expected, label


def test_naming_the_gold_function_is_reported_separately_from_calling_it_right():
    """A wrong argument and a wrong function are different failures, and the retention diagnostic needs both.

    If the compressor drops the gold schema, the model cannot name the function at all. If it keeps the schema
    but the cache is degraded, it can name it and still get an argument wrong. Collapsing those into one bit
    would hide which of the two a compression policy caused.
    """
    wrong_value = score_bfcl_ast('{"name": "calculate_triangle_area", "arguments": {"base": 99, "height": 5}}', REF)
    wrong_fn = score_bfcl_ast('{"name": "calculate_circle_area", "arguments": {"radius": 10}}', REF)
    assert wrong_value["named_gold_function"] and not wrong_value["correct"]
    assert not wrong_fn["named_gold_function"] and not wrong_fn["correct"]


def test_a_parallel_answer_needs_every_call():
    gt = [{"f": {"x": [1]}}, {"g": {"y": [2]}}]
    ref = {"ground_truth": gt, "gold_names": ["f", "g"]}
    both = '{"name": "f", "arguments": {"x": 1}} {"name": "g", "arguments": {"y": 2}}'
    assert score_bfcl_ast(both, ref)["correct"]
    assert not score_bfcl_ast('{"name": "f", "arguments": {"x": 1}}', ref)["correct"]


def test_the_new_scorer_does_not_change_the_old_one():
    """D46's decision: `score_tool_arg` and `parse_tool_calls` stay exactly as they were.

    492 stored Qwen predictions were scored by the old code, and broadening it changed 2 of them, which would
    have left RAQ-Probe's two families scored differently. A new vehicle gets a new scorer; the old one is frozen.
    """
    probe_ref = {"tool_name": "get_account_balance", "arg_name": "account_number", "gold": "93259021",
                 "distractors": []}
    bare = '{"name": "get_account_balance", "parameters": {"account_number": "93259021"}}'
    # Still wrong under the frozen scorer -- that is the point, not a bug.
    assert not score_tool_arg(bare, probe_ref)["correct"]
    tagged = '<tool_call>{"name": "get_account_balance", "arguments": {"account_number": "93259021"}}</tool_call>'
    assert score_tool_arg(tagged, probe_ref)["correct"]


def test_parse_any_calls_finds_nested_objects_and_ignores_junk():
    text = 'thinking... {"name": "f", "arguments": {"a": {"b": 1}}} and then some prose {not json}'
    calls = parse_any_calls(text)
    assert len(calls) == 1 and calls[0]["name"] == "f" and calls[0]["arguments"] == {"a": {"b": 1}}


def test_bfcl_ast_is_registered():
    assert SCORERS["bfcl_ast"] is score_bfcl_ast


def test_a_catalogue_larger_than_the_pool_is_refused():
    """769 distinct schemas exist; asking for a catalogue near that size makes every prompt nearly identical."""
    BFCLConfig(catalogue_sizes=(10, 25, 50))
    with pytest.raises(Exception, match="769 distinct"):
        BFCLConfig(catalogue_sizes=(10, 900))


def test_the_retention_task_accepts_bfcl_and_the_fault_tasks_do_not():
    """M6's question is a retention question. The fault tasks were never given this vehicle, so they must refuse
    it rather than silently run on a dataset whose spans they do not understand."""
    from raq_kv.config.schema import _TASK_DATASETS

    assert "bfcl" in _TASK_DATASETS["retention"]
    assert "bfcl" in _TASK_DATASETS["qa_generate"]
    assert "bfcl" not in _TASK_DATASETS["protection_arm"]


def test_tool_schema_spans_fill_both_the_target_and_the_span_role():
    """In the catalogue the schema IS the span, so `kept_fraction_target` means what it means in M4."""
    from raq_kv.data.prompts import PromptSample, Span
    from raq_kv.experiments.tasks import role_tokens

    spans = (Span("tool_schema", "target", 0, 10, 0, 5, "gold", "tool_schema"),
             Span("tool_schema", "distractor", 10, 20, 5, 9, "other", "tool_schema"))
    sample = PromptSample("s", "text", None, 9, "bfcl_ast", {}, None, None, 8, spans, {})
    roles = role_tokens(sample, 9)
    assert roles["target"] == (0, 1, 2, 3, 4)
    assert roles["distractor"] == (5, 6, 7, 8)
    assert roles["sentence"] == roles["target"], "the gold schema is its own span"


def test_schema_spans_partition_the_cache_without_gaps():
    """`select_keep_spans` must be able to spend the whole budget, so the blocks have to cover [0, cache_len).

    The chat template contributes tokens before, between and after the schemas. If those were dropped rather than
    emitted as their own blocks they would be unprunable, and the realised retention ratio would drift from the
    requested one -- the same class of error D42 had to fix by re-running every short arm.
    """
    from raq_kv.data.prompts import PromptSample, Span
    from raq_kv.experiments.tasks import schema_spans

    spans = (Span("tool_schema", "distractor", 0, 1, 10, 20, "a", "tool_schema"),
             Span("tool_schema", "target", 0, 1, 30, 45, "gold", "tool_schema"))
    sample = PromptSample("s", "t", None, 60, "bfcl_ast", {}, None, None, 8, spans, {})
    blocks = schema_spans(sample, 60)

    assert blocks[0][0] == 0 and blocks[-1][1] == 60, "the blocks must cover the whole cache"
    for (_, end), (start, _) in zip(blocks, blocks[1:], strict=False):
        assert end == start, "the blocks must not leave a gap"
    assert (10, 20) in blocks and (30, 45) in blocks, "each schema must be its own block"
    assert all(hi > lo for lo, hi in blocks), "no empty block"


def test_a_schema_span_clips_to_the_cache_boundary():
    """A schema straddling the cache/query split must not claim tokens the cache does not hold."""
    from raq_kv.data.prompts import PromptSample, Span
    from raq_kv.experiments.tasks import schema_spans

    spans = (Span("tool_schema", "target", 0, 1, 40, 80, "gold", "tool_schema"),)
    sample = PromptSample("s", "t", None, 50, "bfcl_ast", {}, None, None, 8, spans, {})
    blocks = schema_spans(sample, 50)
    assert blocks[-1][1] == 50 and all(hi <= 50 for _, hi in blocks)


def test_the_schema_unit_is_a_distinct_choice_from_the_sentence_unit():
    """A tool catalogue is JSON: punctuation-delimited spans are meaningless in it, so the units must not alias."""
    from raq_kv.config.schema import CompressionConfig

    assert CompressionConfig(scorer="recency", ratio=0.25, unit="schema").unit == "schema"
    assert CompressionConfig(scorer="recency", ratio=0.25, unit="span").unit == "span"
    with pytest.raises(Exception):
        CompressionConfig(scorer="recency", ratio=0.25, unit="sentence")
