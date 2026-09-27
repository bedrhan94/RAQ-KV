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
import string
from collections import Counter
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


_JSON_OBJECT = re.compile(r"\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}", re.DOTALL)


def parse_any_calls(text: str) -> list[dict[str, Any]]:
    """Every JSON object in the output, with the common argument keys normalised to ``arguments``.

    Deliberately separate from :func:`parse_tool_calls`, which requires ``<tool_call>`` tags and is left exactly
    as it was. D46 measured what happens when a tool-call scorer encodes one family's dialect: Llama-3.2 scored
    0 of 120 on RAQ-Probe's `tool_arg` type for writing ``parameters`` instead of ``arguments``, while retrieving
    the right value every time. Broadening the *existing* scorer was rejected there because it changed 2 of 492
    stored Qwen predictions, which would have left the two families scored by different code. A new scorer for a
    new vehicle has no stored results to invalidate, so it can be written the way the old one should have been:
    scoring what was called with which arguments, not the syntax used to say it.
    """
    calls: list[dict[str, Any]] = []
    for match in _JSON_OBJECT.finditer(text):
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        for alias in ("parameters", "args", "arguments"):
            if isinstance(obj.get(alias), dict):
                obj = {**obj, "arguments": obj[alias]}
                break
        calls.append(obj)
    return calls


def _value_matches(got: Any, allowed: Any) -> bool:
    """BFCL ground truth gives each parameter a LIST of acceptable values; "" means the parameter is optional."""
    options = allowed if isinstance(allowed, list) else [allowed]
    for option in options:
        if got == option:
            return True
        if str(got).strip().lower() == str(option).strip().lower():
            return True
        try:
            if float(got) == float(option):
                return True
        except (TypeError, ValueError):
            pass
    return False


def score_bfcl_ast(prediction: str, reference: dict[str, Any]) -> dict[str, Any]:
    """BFCL AST match: did the model call a gold function with acceptable arguments?

    ``ground_truth`` is a list of ``{func_name: {param: [acceptable, values]}}`` -- one entry per call the answer
    requires, so a parallel question needs every entry matched. A parameter whose acceptable list contains ``""``
    is optional and may be absent.
    """
    calls = parse_any_calls(prediction)
    truths = reference.get("ground_truth") or []
    matched_names: list[str] = []
    for truth in truths:
        if not isinstance(truth, dict):
            continue
        satisfied = False
        for want_name, want_args in truth.items():
            for call in calls:
                if call.get("name") != want_name:
                    continue
                got_args = call.get("arguments")
                if not isinstance(got_args, dict):
                    continue
                if all(_value_matches(got_args[param], allowed) if param in got_args
                       else any(o == "" for o in (allowed if isinstance(allowed, list) else [allowed]))
                       for param, allowed in want_args.items()):
                    satisfied = True
                    matched_names.append(want_name)
                    break
            if satisfied:
                break
        if not satisfied:
            return {"correct": False, "em_strict": False, "n_calls": len(calls),
                    "named_gold_function": bool(set(matched_names) or
                                                {c.get("name") for c in calls} & set(reference.get("gold_names", []))),
                    "calls_matched": len(matched_names)}
    correct = bool(truths)
    return {"correct": correct, "em_strict": correct and len(calls) == len(truths), "n_calls": len(calls),
            "named_gold_function": True, "calls_matched": len(matched_names)}


_ARTICLES = re.compile(r"(a|an|the)")


def _f1_normalise(text: str) -> list[str]:
    """LongBench's own normalisation: casefold, drop punctuation, drop articles, split on whitespace.

    Reimplemented rather than imported because LongBench ships it inside a dataset script this project does not
    execute (D66 reads the corpus from the pinned `data.zip` instead). It is the standard SQuAD recipe and is stated
    here so a reader can check it against the benchmark's own `metrics.py`.
    """
    text = text.casefold()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return _ARTICLES.sub(" ", text).split()


def score_longbench_qa_f1(prediction: str, reference: dict[str, Any]) -> dict[str, Any]:
    """Token-level F1 against the best of several references -- LongBench's metric for its QA subsets.

    **Why F1 and not containment.** A free-form answer to a multi-hop question is a phrase, not a value, so exact
    match scores a correct answer wrong for a trailing qualifier ("St Patrick's College, Dublin" against "St
    Patrick's College"). `correct` is therefore a *thresholded* F1, and the threshold is stated rather than implied:
    0.5, which is LongBench's own convention for calling an answer right.

    `contains_gold` is reported beside it because the D63 pilot showed the two can disagree, and a format-insensitive
    signal is what distinguishes a retrieval failure from a transcription one -- the distinction D64 and D70 turn on.
    """
    pred = _f1_normalise(prediction)
    best, best_ref = 0.0, ""
    for ref in reference.get("answers", []):
        gold = _f1_normalise(ref)
        common = Counter(pred) & Counter(gold)
        overlap = sum(common.values())
        if not overlap:
            continue
        precision, recall = overlap / len(pred), overlap / len(gold)
        f1 = 2 * precision * recall / (precision + recall)
        if f1 > best:
            best, best_ref = f1, ref
    contains = any(normalize_answer(r) in normalize_answer(prediction)
                   for r in reference.get("answers", []) if r)
    return {"correct": best >= 0.5, "f1": best, "em_strict": best == 1.0,
            "contains_gold": contains, "best_reference": best_ref}


SCORERS: dict[str, Callable[[str, dict[str, Any]], dict[str, Any]]] = {
    "exact_value": score_exact_value,
    "tool_arg": score_tool_arg,
    "bfcl_ast": score_bfcl_ast,
    "gsm8k": score_gsm8k,
    "longbench_qa_f1": score_longbench_qa_f1,
}
