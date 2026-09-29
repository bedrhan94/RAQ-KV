"""The multi-turn state probe: the vehicle's whole value is that one wrong answer is diagnostic.

If the stale value is not registered as a distractor, or the two values are not distinct, or either appears twice in
the prompt, then `mentions_distractor` stops meaning "answered from the superseded turn" and becomes a generic
wrong-answer flag -- at which point the vehicle measures nothing a single-turn probe could not.
"""

from __future__ import annotations

import pytest

from raq_kv.config.loader import load_yaml
from raq_kv.config.schema import ModelConfig, StateProbeConfig
from raq_kv.data.state_probe import ACK_DOCUMENT, ACK_UPDATE, build_state_probe
from raq_kv.evaluation.scoring import SCORERS
from raq_kv.models.loader import load_tokenizer
from raq_kv.utils.paths import configs_dir


@pytest.fixture(scope="module")
def samples():
    cfg = StateProbeConfig.model_validate(
        load_yaml(configs_dir() / "datasets" / "state_probe_v0.yaml")["dataset"])
    cfg = cfg.model_copy(update={"fact_types": ("number", "id", "tool_arg"), "context_lengths": (1024,),
                                 "groups_per_cell": 2})
    model = ModelConfig.model_validate(load_yaml(configs_dir() / "models" / "qwen3_1p7b.yaml")["model"])
    return build_state_probe(cfg, load_tokenizer(model), dict(model.chat_template_kwargs))


def test_the_stale_value_is_scored_as_a_distractor_so_regression_is_measurable(samples):
    """The one property the vehicle exists for: answering from turn 1 is distinguishable from any other error."""
    for s in samples:
        stale, gold = s.meta["stale_value"], s.reference["gold"]
        assert stale != gold
        assert stale in s.reference["distractors"], "the superseded value must be a scored distractor"
        good = SCORERS[s.scorer](s.gold_text, s.reference)
        assert good["correct"] and not good["mentions_distractor"]
        regressed = SCORERS[s.scorer](s.gold_text.replace(gold, stale), s.reference)
        assert not regressed["correct"] and regressed["mentions_distractor"], (
            f"{s.sample_id}: answering with the stale value must flag mentions_distractor")


def test_both_values_are_distinct_from_every_other_value_in_the_prompt(samples):
    """A new value overlapping the old one, or a distractor, would make `mentions_distractor` fire on a CORRECT
    answer and the vehicle would report regressions that never happened."""
    for s in samples:
        vals = [s.meta["stale_value"], s.meta["updated_value"], *s.reference["distractors"]]
        for i, a in enumerate(vals):
            for b in vals[i + 1:]:
                if a == b:
                    continue
                assert a not in b and b not in a, f"{s.sample_id}: {a!r} and {b!r} overlap"


def test_each_value_occurs_exactly_once_in_the_prompt(samples):
    for s in samples:
        for value in (s.meta["stale_value"], s.meta["updated_value"]):
            assert s.prompt_text.count(value) == 1, f"{s.sample_id}: {value!r} is not unique in the prompt"


def test_the_conversation_is_five_messages_with_fixed_assistant_turns(samples):
    """Deterministic is the plan's word. Model-generated intermediate turns would not rebuild the same cache."""
    for s in samples:
        assert s.meta["n_turns"] == 5
        assert s.prompt_text.count(ACK_DOCUMENT) == 1 and s.prompt_text.count(ACK_UPDATE) == 1


def test_every_labelled_value_is_inside_the_cache_and_the_question_is_not(samples):
    """The split this project's whole protocol rests on: transforms act on the cache, the query is read through it."""
    for s in samples:
        for span in s.spans:
            if span.role in ("target", "stale", "distractor"):
                assert span.token_end <= s.cache_len, f"{s.sample_id}: {span.label}/{span.role} escaped the cache"
        query = next(sp for sp in s.spans if sp.label == "query")
        assert query.token_start == s.cache_len
        assert s.meta["updated_value"] not in query.text, "the answer must not be visible in the query"


def test_generation_is_deterministic_given_the_config(samples):
    cfg = StateProbeConfig.model_validate(
        load_yaml(configs_dir() / "datasets" / "state_probe_v0.yaml")["dataset"])
    cfg = cfg.model_copy(update={"fact_types": ("number", "id", "tool_arg"), "context_lengths": (1024,),
                                 "groups_per_cell": 2})
    model = ModelConfig.model_validate(load_yaml(configs_dir() / "models" / "qwen3_1p7b.yaml")["model"])
    again = build_state_probe(cfg, load_tokenizer(model), dict(model.chat_template_kwargs))
    assert [s.sample_id for s in again] == [s.sample_id for s in samples]
    assert [s.prompt_text for s in again] == [s.prompt_text for s in samples]


def test_the_document_is_the_single_turn_probe_s_own(samples):
    """Reuse, not reimplementation: a state sample's document has to be comparable to a probe sample's."""
    for s in samples:
        doc = next(sp for sp in s.spans if sp.label == "document")
        assert doc.text and doc.text in s.prompt_text
        # The correction restates the fact, so the target KEY appears in both turn 1 and turn 2.
        assert s.prompt_text.count(s.meta["updated_value"]) == 1


def test_the_rendered_conversation_has_exactly_one_generation_prompt(samples):
    """The bug that would have invalidated the vehicle silently.

    The probe's `query` span runs to the END of its prompt, so it contains the question *and* the trailing chat
    template scaffolding. Embedding that as a message's content made the conversation end with two assistant
    headers, one closed and empty, and the model answered inside a malformed prompt -- on every fact type, with
    numbers that looked plausible. Three assistant headers is correct here: two acknowledgements and one generation
    prompt.
    """
    for s in samples:
        assert s.prompt_text.count("<|im_start|>assistant") == 3, (
            f"{s.sample_id}: {s.prompt_text.count('<|im_start|>assistant')} assistant headers")
        assert s.prompt_text.count("<|im_start|>user") == 3
        # And nothing after the final generation prompt.
        assert s.prompt_text.rstrip().endswith("</think>")


def test_the_question_carries_no_template_markers(samples):
    """What the stripping in `_question` exists for, asserted on the content rather than on the rendered result."""
    for s in samples:
        turns = s.prompt_text.split("<|im_start|>user\n")
        question_turn = turns[-1].split("<|im_end|>")[0]
        for marker in ("<|im_start|>", "<|im_end|>", "<think>"):
            assert marker not in question_turn, f"{s.sample_id}: question turn contains {marker!r}"


def test_build_raq_probe_refuses_a_state_probe_config():
    """The bug that voided D69's first fault arm, and the reason it could not have crashed.

    `StateProbeConfig` is deliberately a superset of `RAQProbeConfig`'s field names so the two vehicles share one
    document generator. That makes `build_raq_probe(state_cfg)` **succeed** and build single-turn samples, which is
    exactly what happened: `ProtectionArm` called it directly, the manifest said `state_probe`, and the arm reported
    zero state regressions measured on a vehicle that had no correction turn.

    The project has hit this shape before -- a scripted edit left `Retention.__init__` calling `build_raq_probe`, and
    BFCL revealed it only by crashing on a field its config lacks. A config designed to share field names removes
    even that accident.
    """
    from raq_kv.data.raq_probe import build_raq_probe

    cfg = StateProbeConfig.model_validate(
        load_yaml(configs_dir() / "datasets" / "state_probe_v0.yaml")["dataset"])
    model = ModelConfig.model_validate(load_yaml(configs_dir() / "models" / "qwen3_1p7b.yaml")["model"])
    with pytest.raises(TypeError, match="silently build single-turn samples"):
        build_raq_probe(cfg, load_tokenizer(model), dict(model.chat_template_kwargs))


def test_the_fault_and_qa_tasks_reach_the_state_probe_through_the_dispatcher():
    """Three task classes built their samples directly; only `Retention` used the dispatcher. Pinned so a fourth
    vehicle does not have to rediscover which ones were wired."""
    import inspect

    from raq_kv.experiments import tasks

    for name in ("FaultQA", "ProtectionArm", "Retention"):
        src = inspect.getsource(getattr(tasks, name))
        assert "build_probe_samples(" in src, f"{name} does not use the dispatcher"
        assert "build_raq_probe(" not in src, f"{name} still calls build_raq_probe directly"


@pytest.fixture(scope="module")
def multi_samples():
    """The same cells with TWO corrections (D83), so a reversion becomes attributable to a position."""
    cfg = StateProbeConfig.model_validate(
        load_yaml(configs_dir() / "datasets" / "state_probe_v0.yaml")["dataset"])
    cfg = cfg.model_copy(update={"fact_types": ("number", "id", "tool_arg"), "context_lengths": (1024,),
                                 "groups_per_cell": 2, "n_updates": 2})
    model = ModelConfig.model_validate(load_yaml(configs_dir() / "models" / "qwen3_1p7b.yaml")["model"])
    return build_state_probe(cfg, load_tokenizer(model), dict(model.chat_template_kwargs))


def test_n_updates_unset_is_byte_identical_to_one(samples):
    """The field had to default to None, not 1, or every existing state-probe identity would have moved.

    This checks the stronger property the default exists for: the samples themselves are unchanged, not merely the
    identity hash. Verified against the store too -- all four live state-probe arms reproduce their recorded
    fingerprints with this code (D69 golden, D69 fault, D72, D81).
    """
    from raq_kv.data.prompts import samples_fingerprint

    cfg = StateProbeConfig.model_validate(
        load_yaml(configs_dir() / "datasets" / "state_probe_v0.yaml")["dataset"])
    assert cfg.n_updates is None, "the default must stay None so `exclude_none` leaves existing identities alone"
    cfg = cfg.model_copy(update={"fact_types": ("number", "id", "tool_arg"), "context_lengths": (1024,),
                                 "groups_per_cell": 2})
    model = ModelConfig.model_validate(load_yaml(configs_dir() / "models" / "qwen3_1p7b.yaml")["model"])
    explicit = build_state_probe(cfg.model_copy(update={"n_updates": 1}),
                                 load_tokenizer(model), dict(model.chat_template_kwargs))
    assert samples_fingerprint(explicit) == samples_fingerprint(samples)


def test_two_corrections_leave_two_stale_values_in_order(multi_samples, samples):
    for s in multi_samples:
        stale = s.meta["stale_values"]
        assert s.meta["n_updates"] == 2
        assert len(stale) == 2, stale
        # Oldest first: the document's own value, then the superseded correction.
        assert stale[0] == s.meta["stale_value"], "the first stale value must be the document's, as it was at n=1"
        assert s.meta["updated_value"] not in stale
        # Both stale values, and the live one, must be scorable distinctly.
        assert len({*stale, s.meta["updated_value"]}) == 3
        for value in (*stale, s.meta["updated_value"]):
            assert s.prompt_text.count(value) == 1, f"{value!r} must occur once"
        # Every superseded value is a scorer distractor, so `mentions_distractor` keeps its old meaning.
        assert set(stale) <= set(s.reference["distractors"])
        assert s.reference["gold"] == s.meta["updated_value"]
    # Two more messages than the single-correction shape, and the cache is correspondingly longer.
    assert {s.meta["n_turns"] for s in multi_samples} == {7}
    assert {s.meta["n_turns"] for s in samples} == {5}


def test_both_stale_values_are_inside_the_cache(multi_samples):
    """A stale value outside the cache could not be reverted to by a cache fault, so the measurement would be void."""
    for s in multi_samples:
        stale_spans = [sp for sp in s.spans if sp.label == "fact_value" and sp.role == "stale"]
        assert len(stale_spans) == 2, [(sp.role, sp.text) for sp in s.spans if sp.label == "fact_value"]
        for sp in stale_spans:
            assert sp.token_end <= s.cache_len
        assert {sp.text for sp in stale_spans} == set(s.meta["stale_values"])


def test_each_correction_states_the_fact_the_same_way(multi_samples):
    """Both corrections reuse the document's own sentence template, so the only difference between them is the value."""
    for s in multi_samples:
        corrections = [line for line in s.prompt_text.split("\n") if line.startswith("Correction to the record: ")]
        assert len(corrections) == 2, corrections
        live, prior = s.meta["updated_value"], s.meta["stale_values"][1]
        assert prior in corrections[0] and live in corrections[1], corrections
        # Same sentence, different value: strip the values and the remainder must match.
        assert corrections[0].replace(prior, "") == corrections[1].replace(live, "")


def test_the_multi_update_scorer_still_separates_a_correct_answer_from_each_reversion(multi_samples):
    """The property the whole vehicle rests on, now with two ways to be stale."""
    s = multi_samples[0]
    score = SCORERS[s.scorer]
    good = score(s.gold_text, s.reference)
    assert good["correct"] and not good["mentions_distractor"]
    for stale in s.meta["stale_values"]:
        bad = score(s.gold_text.replace(s.meta["updated_value"], stale), s.reference)
        assert not bad["correct"] and bad["mentions_distractor"], stale
