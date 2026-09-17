"""Answer scorers. Each returns a flat dict with at least ``correct`` (bool).

* ``exact_value``: ``correct`` = the gold value appears as a whole token-bounded string and no distractor value
  does (listing every candidate is not retrieval); ``em_strict`` = the normalized prediction equals the gold value.
* ``tool_arg``: ``correct`` = some parsed ``<tool_call>`` names the expected tool with the gold argument;
  ``em_strict`` additionally requires exactly one call.
* ``gsm8k``: numeric equality with the gold ``####`` answer; the extraction method is recorded.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

_EDGE_PUNCT = " \t\r\n.,;:!?\"'`*()[]{}"


def normalize_answer(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().strip(_EDGE_PUNCT)).casefold()


def _contains_token_bounded(haystack: str, needle: str) -> bool:
    pattern = r"(?<![0-9a-z])" + re.escape(needle) + r"(?![0-9a-z])"
    return re.search(pattern, haystack) is not None


def score_exact_value(prediction: str, reference: dict[str, Any]) -> dict[str, Any]:
    pred = normalize_answer(prediction)
    gold = normalize_answer(reference["gold"])
    has_gold = _contains_token_bounded(pred, gold)
    has_distractor = any(_contains_token_bounded(pred, normalize_answer(d)) for d in reference.get("distractors", []))
    return {"correct": has_gold and not has_distractor, "em_strict": pred == gold,
            "mentions_distractor": has_distractor}


_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*(?:</tool_call>|$)", re.DOTALL)


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    calls = []
    for match in _TOOL_CALL.finditer(text):
        try:
            obj = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            calls.append(obj)
    return calls


def score_tool_arg(prediction: str, reference: dict[str, Any]) -> dict[str, Any]:
    calls = parse_tool_calls(prediction)

    def matches(call: dict[str, Any]) -> bool:
        args = call.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return False
        return (
            call.get("name") == reference["tool_name"]
            and isinstance(args, dict)
            and str(args.get(reference["arg_name"])).strip() == reference["gold"]
        )

    correct = any(matches(c) for c in calls)
    wrong_value_from_distractor = any(
        isinstance(c.get("arguments"), dict)
        and str(c["arguments"].get(reference["arg_name"])).strip() in reference.get("distractors", [])
        for c in calls
    )
    return {"correct": correct, "em_strict": correct and len(calls) == 1, "n_tool_calls": len(calls),
            "mentions_distractor": wrong_value_from_distractor}


_NUMBER = r"-?\d[\d,]*(?:\.\d+)?"


def _to_number(text: str) -> float | None:
    try:
        return float(text.replace(",", "").replace("$", "").strip())
    except ValueError:
        return None


def gsm8k_gold(answer_field: str) -> str:
    if "####" not in answer_field:
        raise ValueError("GSM8K answer without '####' marker")
    value = answer_field.split("####")[-1].strip().replace(",", "")
    if _to_number(value) is None:
        raise ValueError(f"unparseable GSM8K gold answer {value!r}")
    return value


def extract_gsm8k_prediction(text: str) -> tuple[str | None, str]:
    """Priority: requested '#### N' marker, then LaTeX \\boxed{N} (Qwen3's habit), then the last number."""
    marked = re.findall(r"####\s*\$?\s*(" + _NUMBER + ")", text)
    if marked:
        return marked[-1].replace(",", ""), "marker"
    boxed = re.findall(r"\\boxed\{\s*\\?\$?\s*(" + _NUMBER + r")\s*\}", text)
    if boxed:
        return boxed[-1].replace(",", ""), "boxed"
    numbers = re.findall(_NUMBER, text)
    if numbers:
        return numbers[-1].replace(",", ""), "last_number"
    return None, "none"


def score_gsm8k(prediction: str, reference: dict[str, Any]) -> dict[str, Any]:
    value, method = extract_gsm8k_prediction(prediction)
    pred_num = _to_number(value) if value is not None else None
    gold_num = _to_number(reference["gold"])
    correct = pred_num is not None and gold_num is not None and abs(pred_num - gold_num) < 1e-6
    return {"correct": correct, "em_strict": correct and method == "marker", "extracted": value,
            "extraction_method": method}


SCORERS: dict[str, Callable[[str, dict[str, Any]], dict[str, Any]]] = {
    "exact_value": score_exact_value,
    "tool_arg": score_tool_arg,
    "gsm8k": score_gsm8k,
}
