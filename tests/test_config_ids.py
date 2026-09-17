from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from raq_kv.config.loader import apply_overrides, expand_sweep, load_experiment_configs
from raq_kv.config.schema import ExperimentConfig
from raq_kv.experiments.ids import config_hash, experiment_id, group_hash
from raq_kv.utils.paths import configs_dir

MODEL = {"hf_id": "org/m", "revision": "abcdef123", "short_name": "m"}
DATASET = {"name": "wikitext2", "revision": "1234567abc", "n_chunks": 2, "prefix_len": 8, "continuation_len": 4}


def make_cfg(**kw) -> ExperimentConfig:
    data = {"name": "t", "seed": 0, "model": MODEL, "dataset": DATASET, "task": {"kind": "cache_ppl"}}
    data.update(kw)
    return ExperimentConfig.model_validate(data)


def test_unknown_keys_are_rejected():
    with pytest.raises(ValidationError):
        make_cfg(modle=MODEL)
    with pytest.raises(ValidationError):
        make_cfg(task={"kind": "cache_ppl", "store_token_nl": True})


def test_ids_are_deterministic_and_sensitive():
    a, b = make_cfg(), make_cfg()
    assert experiment_id(a) == experiment_id(b)
    assert experiment_id(make_cfg(seed=1)) != experiment_id(a)
    assert group_hash(make_cfg(seed=1)) == group_hash(a)
    assert config_hash(make_cfg(dataset={**DATASET, "prefix_len": 9})) != config_hash(a)


def test_output_root_and_notes_do_not_change_identity():
    a = make_cfg()
    assert experiment_id(make_cfg(output={"root": "elsewhere"}, notes="x")) == experiment_id(a)


def test_slug_contains_required_metadata():
    exp = experiment_id(make_cfg(seed=3))
    for token in ("m_", "wikitext2", "cache_ppl", "bf16", "noecc", "nofault", "p0", "cuda", "s3"):
        assert token in exp


def test_overrides_and_sweep():
    raw = {"a": {"b": 1}, "sweep": {"seed": [0, 1], "a.b": [5, 6]}}
    raw = apply_overrides(raw, ["a.c=[1, 2]", "x=hello"])
    variants = expand_sweep(raw)
    assert len(variants) == 4
    assert {(v["seed"], v["a"]["b"]) for v in variants} == {(0, 5), (0, 6), (1, 5), (1, 6)}
    assert all(v["a"]["c"] == [1, 2] and v["x"] == "hello" and "sweep" not in v for v in variants)


def test_repository_experiment_configs_are_valid():
    for path in sorted((configs_dir() / "experiments").glob("*.yaml")):
        configs = load_experiment_configs(path)
        assert configs, path
        assert len({experiment_id(c) for c in configs}) == len(configs)


def test_existing_experiment_identity_is_stable():
    """Regression guard: schema changes must not silently re-key experiments that already have raw results."""
    (cfg,) = load_experiment_configs(configs_dir() / "experiments" / "m0a_cpu_smoke.yaml")
    assert experiment_id(cfg) == "qwen3-1p7b_wikitext2_cache_ppl_fp32_noecc_nofault_p0_cpu_s0-f96bb32e"


def test_task_dataset_compatibility():
    with pytest.raises(ValidationError, match="does not support"):
        make_cfg(task={"kind": "qa_generate"})
    probe = {"name": "raq_probe", "filler_revision": "b08601e04326c79dfdd32d625aee71d232d685c3"}
    with pytest.raises(ValidationError, match="does not support"):
        make_cfg(dataset=probe)
    assert "raq_probev0" in experiment_id(make_cfg(dataset=probe, task={"kind": "qa_generate"}))


def test_include_cycle_detected(tmp_path: Path):
    (tmp_path / "a.yaml").write_text("include: [b.yaml]\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("include: [a.yaml]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cycle"):
        load_experiment_configs(tmp_path / "a.yaml", root=tmp_path)
