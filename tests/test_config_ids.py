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


def test_protection_token_describes_the_arm():
    """An arm must be identifiable from its directory name: the M3 arms differ only in protection."""
    from raq_kv.config.schema import ProtectionConfig, QuantizationConfig, QuantSpec
    from raq_kv.experiments.ids import _protection_token

    base = make_cfg(quantization=QuantizationConfig(keys=QuantSpec(bits=4, axis="group_token", group_size=64)))
    assert _protection_token(base) == "noecc"
    assert _protection_token(base.model_copy(update={"protection": ProtectionConfig(metadata_codec="secded")})) \
        == "mSEC"
    uniform = ProtectionConfig(data_codec="secded", data_word_bits=64, metadata_codec="secded")
    assert _protection_token(base.model_copy(update={"protection": uniform})) == "dSEC64-mSEC"
    late = ProtectionConfig(data_codec="secded", data_word_bits=64, data_layers=tuple(range(9, 28)),
                            metadata_codec="secded")
    early = ProtectionConfig(data_codec="secded", data_word_bits=64, data_layers=tuple(range(0, 19)),
                             metadata_codec="secded")
    assert _protection_token(base.model_copy(update={"protection": late})) == "dSEC64L9-27-mSEC"
    assert _protection_token(base.model_copy(update={"protection": early})) == "dSEC64L0-18-mSEC"
    # D and E are bit-identical but must never collide in storage.
    assert _protection_token(base.model_copy(update={"protection": late})) \
        != _protection_token(base.model_copy(update={"protection": early}))


def test_the_nofault_slug_token_is_known_wrong_and_pinned_on_purpose():
    """A fault campaign whose directory name says "nofault" -- documented, not repaired, and pinned so it stays
    that way.

    The literal dates from M0, when no config had a fault section. Protection arms put their fault model in
    `task`, which this token never reads, so all 14 of them are labelled "nofault" while injecting faults. That is
    exactly the defect D24 and D42 were about, and the reason it is not fixed is that the slug is part of
    `experiment_id`: an honest token would change the id of every completed fault experiment and orphan its raw
    records. This test exists so the next reader who spots the defect finds the reason attached to it rather than
    "fixing" it into a history-breaking rename.
    """
    import pathlib

    from raq_kv.config.loader import load_experiment_configs
    from raq_kv.experiments.ids import slug

    root = pathlib.Path(__file__).resolve().parents[1] / "configs" / "experiments"
    cfg = load_experiment_configs(root / "d61_qwen_top.yaml")[0]
    assert cfg.task.n_flips == 1, "this arm injects one fault per trial"
    assert "nofault" in slug(cfg), (
        "the token is wrong and pinned; see the comment in ids.py before changing it, because making it honest "
        "renames every completed fault experiment")
    # The record's own fields are the honest description, and they are what the reports read.
    assert cfg.task.fault_target == "group_rank" and cfg.task.target_group_rank == 0


def test_a_group_targeted_arm_needs_its_rank_and_its_scorer():
    """Three fields that only mean something together, so the schema refuses any two of them."""
    import pytest

    from raq_kv.config.schema import ProtectionArmTask

    ok = ProtectionArmTask(n_flips=1, fault_target="group_rank", target_group_rank=0,
                           rank_scorer="query_attention")
    assert ok.fault_target == "group_rank"

    with pytest.raises(Exception, match="needs both"):
        ProtectionArmTask(n_flips=1, fault_target="group_rank", target_group_rank=0)
    with pytest.raises(Exception, match="needs both"):
        ProtectionArmTask(n_flips=1, fault_target="group_rank", rank_scorer="query_attention")
    with pytest.raises(Exception, match="only mean something"):
        ProtectionArmTask(n_flips=1, target_group_rank=0, rank_scorer="query_attention")


def test_a_group_targeted_arm_is_refused_a_rate():
    """`flips_for_rate` sizes a count against the arm's whole storage while the draw sees one group's scales.

    An arm setting both would state an exposure it does not have -- the same class of defect as D43's matched-rate
    confound, reached from the configuration side instead of the analysis side.
    """
    import pytest

    from raq_kv.config.schema import ProtectionArmTask

    with pytest.raises(Exception, match="stated and real exposure differ"):
        ProtectionArmTask(ber=1e-6, fault_target="group_rank", target_group_rank=0,
                          rank_scorer="query_attention")
    # Same rule for the metadata draw (D62), which narrows the population a different way and is refused a rate
    # for the identical reason: `flips_for_rate` sizes its count against bits the draw never visits.
    with pytest.raises(Exception, match="stated and real exposure differ"):
        ProtectionArmTask(ber=1e-6, fault_target="metadata")
    ok = ProtectionArmTask(n_flips=1, fault_target="metadata", record_perturbation=True)
    assert ok.target_group_rank is None and ok.rank_scorer is None, (
        "a metadata arm needs no ranking, so it must not be required to carry one")
    with pytest.raises(Exception, match="only mean something"):
        ProtectionArmTask(n_flips=1, fault_target="metadata", target_group_rank=0)


def test_the_existing_protection_arms_keep_their_identities():
    """The `ber_basis` lesson, as a standing check: three new fields defaulting to None must move nothing.

    Adding `ber_basis` with `default="stored"` shifted all 14 protection-arm identities at once, because
    `identity_dict` uses `exclude_none` and a non-None default serializes into every config that never mentions
    the field. `tests/experiment_identities.json` is the pinned baseline; this asserts the mechanism directly so a
    failure says *why* rather than only *that* the hashes moved.
    """
    from raq_kv.config.schema import ProtectionArmTask

    plain = ProtectionArmTask(ber=1e-6)
    assert plain.fault_target is None and plain.target_group_rank is None and plain.rank_scorer is None
    dumped = plain.model_dump(exclude_none=True)
    assert plain.record_perturbation is None
    for field in ("fault_target", "target_group_rank", "rank_scorer", "ber_basis", "n_flips",
                  "record_perturbation"):
        assert field not in dumped, f"{field} would enter the identity hash of every arm that never sets it"


def test_a_burst_arm_states_its_exposure_in_bits_and_delivers_them_grouped():
    """H2b's live form (D67). The point of a burst arm is that it is matched on BITS to a single-bit arm.

    So the exposure stays `n_flips` -- the total delivered -- and `burst_bits` only says how they are grouped. A
    rate is refused because it would have to be converted per sample, and the two arms of a placement comparison
    would stop being matched on the quantity the comparison holds fixed.
    """
    import pytest

    from raq_kv.config.schema import ProtectionArmTask

    ok = ProtectionArmTask(n_flips=64, burst_bits=16)
    assert ok.n_flips // ok.burst_bits == 4, "64 bits arrive as 4 words of 16"

    with pytest.raises(Exception, match="stops? being matched|states its exposure as"):
        ProtectionArmTask(ber=1e-6, burst_bits=16)
    # A count that is not a multiple would deliver fewer bits than the arm claims.
    with pytest.raises(Exception, match="not a multiple"):
        ProtectionArmTask(n_flips=30, burst_bits=16)
    # Bursts hit codes; fault_target narrows to metadata or to one token group. Combining them would silently
    # pick one, and an arm whose manifest says both is the D24/D42 failure mode again.
    with pytest.raises(Exception, match="not implemented"):
        ProtectionArmTask(n_flips=64, burst_bits=16, fault_target="metadata")
    with pytest.raises(Exception, match="not implemented"):
        ProtectionArmTask(n_flips=64, burst_bits=16, fault_target="group_rank",
                          target_group_rank=0, rank_scorer="query_attention")


def test_burst_bits_defaults_to_none_so_no_existing_arm_identity_moves():
    from raq_kv.config.schema import ProtectionArmTask

    plain = ProtectionArmTask(ber=1e-6)
    assert plain.burst_bits is None
    assert "burst_bits" not in plain.model_dump(exclude_none=True)


def test_an_override_the_sweep_would_erase_is_refused(tmp_path):
    """The trap that cost 15 minutes: `--set task.n_samples=2` on a config sweeping `task` does nothing.

    `apply_overrides` runs first, then `expand_sweep` rewrites each swept key with `set_dotted`, so the override is
    applied and then overwritten. Nothing in the output says so -- the run that happens is a valid one nobody asked
    for, which is worse than an error because it looks like a smoke test and is the full campaign.
    """
    import pathlib

    import pytest

    from raq_kv.config.loader import load_experiment_configs

    src = pathlib.Path(__file__).resolve().parents[1] / "configs" / "experiments" / "d67_burst_calibration.yaml"
    with pytest.raises(ValueError, match="silently discarded"):
        load_experiment_configs(src, overrides=["task.n_samples=2"])
    # An override on a key the config does NOT sweep is still honoured.
    cfgs = load_experiment_configs(src, overrides=["seed=7"])
    assert cfgs and all(c.seed == 7 for c in cfgs)


def test_overrides_still_work_on_a_config_without_a_sweep():
    import pathlib

    from raq_kv.config.loader import load_experiment_configs

    src = pathlib.Path(__file__).resolve().parents[1] / "configs" / "experiments" / "d61_qwen_top.yaml"
    cfgs = load_experiment_configs(src, overrides=["task.n_samples=3"])
    assert len(cfgs) == 1 and cfgs[0].task.n_samples == 3
