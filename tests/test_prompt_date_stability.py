"""Every model's rendered prompt must be the same on any two days. Llama's is not, and D79 records why.

Llama-3.x chat templates call ``strftime_now`` and inject ``"Today Date: <today>"`` into the system prompt. So every
Llama arm in this project embeds its **run date** in the cache it then quantizes, faults and scores. The experiment
identity hash cannot see it -- the date is not a config field -- so the runner will happily treat a re-run on another
day as the same experiment, while ``data_fingerprint`` correctly reports different data.

Found by D78's pre-registered reproduction gate: its bf16 rung scored 68/120 against ``m9l_bfcl_golden``'s 69/120, and
chasing the one-sample difference produced this. Measured impact on that pair: **118 of 120 predictions identical**, the
two that differ both on the same tie-prone BFCL item, one of which recorded an exact bf16 margin of 0.0 (D11's tie
case). So the effect is small but it is not nothing, and it makes no Llama arm byte-reproducible on a different day.

``test_rendered_prompts_are_date_stable`` asserts the invariant the project wants. It is **expected to fail** until the
date is pinned, because pinning changes ``chat_template_kwargs``, which IS part of the identity hash -- so it would
re-key all 61 Llama arms (6,529 units, 9.17 recorded GPU-hours) and change published numbers. That is a decision with a
cost, recorded in D79 and not taken unilaterally. When it is taken, deleting the ``xfail`` marker is the whole change.
"""

from __future__ import annotations

import datetime as _dt
import pathlib
import re
import types

import pytest
import yaml

DATE = re.compile(r"\b\d{1,2} [A-Z][a-z]{2} \d{4}\b")
CONFIGS = sorted(pathlib.Path("configs/models").glob("*.yaml"))


def _model(path: pathlib.Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["model"]


def test_the_model_configs_are_discoverable():
    """Guards the guard: a glob that silently matches nothing would make every test below vacuous."""
    assert len(CONFIGS) >= 3, [p.name for p in CONFIGS]
    assert any("llama" in p.name for p in CONFIGS)
    assert any("qwen" in p.name for p in CONFIGS)


@pytest.mark.gpu
@pytest.mark.xfail(strict=True, reason="D79: Llama-3.x injects the run date and pinning it re-keys 61 arms; the "
                                      "decision and its 9.17 GPU-hour cost are recorded, not taken silently")
def test_rendered_prompts_are_date_stable():
    """The invariant: a rendered prompt must carry no date that moves, on any model this project uses."""
    from raq_kv.data.prompts import render_chat
    from raq_kv.models.loader import load_tokenizer

    offenders = []
    for path in CONFIGS:
        model = _model(path)
        tok = load_tokenizer(types.SimpleNamespace(**model))
        rendered = render_chat(tok, "PING", model.get("chat_template_kwargs") or {})
        found = DATE.findall(rendered)
        today = _dt.date.today().strftime("%d %b %Y").lstrip("0")
        if any(d.lstrip("0") == today for d in found):
            offenders.append((path.name, found))
    assert not offenders, f"these configs embed today's date in every prompt: {offenders}"


@pytest.mark.gpu
def test_a_pinned_date_string_fixes_it_without_changing_the_token_count():
    """The fix, verified, so the record's claim that it is cheap is checked rather than asserted.

    Token count identical means pinning changes only the date VALUE, not any prompt's length -- so cache_len, span
    bounds and the bit accounting are untouched and only the fingerprint moves.
    """
    from raq_kv.models.loader import load_tokenizer

    llama = next((p for p in CONFIGS if "llama" in p.name), None)
    assert llama is not None
    tok = load_tokenizer(types.SimpleNamespace(**_model(llama)))
    messages = [{"role": "user", "content": "PING"}]
    free = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    pinned = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                     date_string="26 Jul 2024")
    assert DATE.findall(pinned) == ["26 Jul 2024"], DATE.findall(pinned)
    assert DATE.findall(free) != DATE.findall(pinned), "the free render should carry a different (today's) date"
    assert len(tok(free).input_ids) == len(tok(pinned).input_ids), (
        "pinning must not change the token count, or it would move every cache_len and span bound")


@pytest.mark.gpu
def test_qwen_needs_no_pin_which_is_why_its_arms_agree_across_days():
    """The control, and it explains the evidence: Qwen's m6 (09-23) and d66 (09-26) arms share one fingerprint."""
    from raq_kv.models.loader import load_tokenizer

    qwen = next((p for p in CONFIGS if "qwen" in p.name), None)
    assert qwen is not None
    tok = load_tokenizer(types.SimpleNamespace(**_model(qwen)))
    rendered = tok.apply_chat_template([{"role": "user", "content": "PING"}],
                                       tokenize=False, add_generation_prompt=True)
    assert not DATE.findall(rendered), DATE.findall(rendered)
    assert "strftime_now" not in (tok.chat_template or "")
