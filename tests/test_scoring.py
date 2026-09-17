from __future__ import annotations

import pytest

from raq_kv.evaluation.scoring import (
    extract_gsm8k_prediction,
    gsm8k_gold,
    parse_tool_calls,
    score_exact_value,
    score_gsm8k,
    score_tool_arg,
)

REF = {"gold": "TRK-4829-QX", "distractors": ["LMN-1111-AB", "PQR-2222-CD"]}


@pytest.mark.parametrize("pred", ["TRK-4829-QX", "TRK-4829-QX.", "`trk-4829-qx`", " TRK-4829-QX\n"])
def test_exact_value_strict(pred):
    out = score_exact_value(pred, REF)
    assert out["correct"] and out["em_strict"]


def test_exact_value_contained_but_not_strict():
    out = score_exact_value("The tracking code is TRK-4829-QX.", REF)
    assert out["correct"] and not out["em_strict"]


def test_listing_distractors_is_not_retrieval():
    out = score_exact_value("TRK-4829-QX or LMN-1111-AB", REF)
    assert not out["correct"] and out["mentions_distractor"]


def test_numbers_are_token_bounded():
    ref = {"gold": "4821", "distractors": []}
    assert not score_exact_value("48210", ref)["correct"]
    assert not score_exact_value("14821", ref)["correct"]
    assert score_exact_value("4821 crates", ref)["correct"]


def test_tool_call_parsing_and_scoring():
    ref = {"gold": "12345678", "distractors": ["87654321"], "tool_name": "get_account_balance",
           "arg_name": "account_number"}
    good = '<tool_call>\n{"name": "get_account_balance", "arguments": {"account_number": "12345678"}}\n</tool_call>'
    assert score_tool_arg(good, ref) == {"correct": True, "em_strict": True, "n_tool_calls": 1,
                                         "mentions_distractor": False}
    as_int = '<tool_call>{"name": "get_account_balance", "arguments": {"account_number": 12345678}}</tool_call>'
    assert score_tool_arg(as_int, ref)["correct"]
    truncated = '<tool_call>\n{"name": "get_account_balance", "arguments": {"account_number": "12345678"}}'
    assert score_tool_arg(truncated, ref)["correct"]
    wrong = '<tool_call>{"name": "get_account_balance", "arguments": {"account_number": "87654321"}}</tool_call>'
    out = score_tool_arg(wrong, ref)
    assert not out["correct"] and out["mentions_distractor"]
    assert parse_tool_calls("no call here") == []
    other_tool = '<tool_call>{"name": "other", "arguments": {"account_number": "12345678"}}'
    assert not score_tool_arg(other_tool, ref)["correct"]


def test_gsm8k_gold_and_extraction():
    assert gsm8k_gold("Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 eggs.\n#### 18") == "18"
    assert gsm8k_gold("...\n#### 1,000") == "1000"
    with pytest.raises(ValueError):
        gsm8k_gold("no marker 5")
    assert extract_gsm8k_prediction("so 3+4=7\n#### 7") == ("7", "marker")
    assert extract_gsm8k_prediction("#### $1,250") == ("1250", "marker")
    assert extract_gsm8k_prediction("The answer is 42.") == ("42", "last_number")
    # Real Qwen3 pilot outputs (2026-09-17)
    boxed_only = "A = 498\n$$\n\n### **Final Answer:**\n\n$$\n\\boxed{623}\n$$"
    assert extract_gsm8k_prediction(boxed_only) == ("623", "boxed")
    boxed_marker = "= 48 hours\n$$\n\n### Final Answer:\n\n$$\n\\boxed{#### 48}\n$$"
    assert extract_gsm8k_prediction(boxed_marker) == ("48", "marker")
    assert extract_gsm8k_prediction("step 1 gives 12\n\\boxed{1,250} dollars") == ("1250", "boxed")
    assert extract_gsm8k_prediction("no idea") == (None, "none")
    assert score_gsm8k("#### 18.0", {"gold": "18"})["correct"]
    out = score_gsm8k("it is 18", {"gold": "18"})
    assert out["correct"] and not out["em_strict"]
