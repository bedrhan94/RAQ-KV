"""The iso-budget correction is deliberately asymmetric, and an asymmetry nobody tests will be 'tidied' away.

D41 recorded the joint arm spending 2.8% more than the baseline it beat. The fix raises the baseline. What the
fix must never do is *lower* a baseline that already spends more, because a symmetric tolerance would license
exactly that -- shaving a baseline's budget until it loses by more, and calling it iso-budget compliance.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "calibrate_iso_budget", Path(__file__).resolve().parents[1] / "scripts" / "calibrate_iso_budget.py")
cal = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cal)


@pytest.mark.parametrize("deviation", [0.0, 0.005, 0.0132, 0.05, 0.5])
def test_an_arm_at_or_above_the_reference_is_never_cut(deviation):
    """The recency arms sit at +1.32% and +0.82%. Both are outside a symmetric 1% rule and both must stand."""
    assert cal.decide(deviation, tolerance=0.01) == "conservative"


@pytest.mark.parametrize("deviation", [-0.0275, -0.0143, -0.011, -0.9])
def test_a_short_arm_outside_the_rule_is_corrected(deviation):
    assert cal.decide(deviation, tolerance=0.01) == "correct"


@pytest.mark.parametrize("deviation", [-0.01, -0.009, -1e-9])
def test_a_short_arm_inside_the_rule_is_left_alone(deviation):
    """Re-emitting an arm that already complies would spend GPU time creating a second run of one comparison."""
    assert cal.decide(deviation, tolerance=0.01) == "within"


def test_the_correction_only_rewrites_the_two_fields_it_claims_to(tmp_path):
    source = tmp_path / "arm.yaml"
    source.write_text(
        "# a comment the corrected file replaces with its own header\n"
        "include:\n  - models/qwen3_1p7b_eager.yaml\n"
        "name: arm\n"
        "compression:\n  scorer: query_attention\n  ratio: 0.06560\n  unit: span\n  protect_initial: 4\n"
        "task:\n  kind: retention\n  n_samples: 120\nseed: 0\n", encoding="utf-8")

    out = cal.corrected_config(source, 0.06759, "arm_isocal", ["measured 1.0116", "reference 1.0401"])

    assert "ratio: 0.06759" in out and "0.06560" not in out
    assert "name: arm_isocal" in out and "\nname: arm\n" not in out
    # Everything else is the allocator's decision and must survive untouched.
    for kept in ("scorer: query_attention", "unit: span", "protect_initial: 4", "n_samples: 120", "seed: 0",
                 "models/qwen3_1p7b_eager.yaml"):
        assert kept in out
    # The measurement that justified the change travels with the config, so a reader never has to trust the
    # filename (D41: the emitted config carries its own rationale).
    assert "measured 1.0116" in out and "reference 1.0401" in out
    assert "a comment the corrected file replaces" not in out


def test_a_config_without_a_ratio_is_refused(tmp_path):
    source = tmp_path / "arm.yaml"
    source.write_text("name: arm\ntask:\n  kind: retention\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        cal._ratio_of(source)


def test_a_task_that_never_prunes_refuses_a_compression_block():
    """D42: the block would reach `_pruning_token` and name the run as pruned while it executed unpruned.

    Found by trying to design the fault x retention experiment: `ProtectionArmTask.run_unit` goes prefill ->
    quantize -> protect -> decode with no compression step, yet the schema accepted the block, it survived into
    the config object, and the slug read it. That is a manifest that is internally consistent and wrong -- D24's
    failure mode reached from the other direction, since D24 was a manifest recording code that did not run and
    this is one recording a transform that did not run.
    """
    import pathlib

    from raq_kv.config.loader import load_experiment_configs

    src = pathlib.Path(__file__).resolve().parents[1] / "configs" / "experiments" / "m3_arm_h_range_guard.yaml"
    text = src.read_text(encoding="utf-8").replace("name: m3_arm_h_range_guard", "name: _tmp_compression_leak")
    text += "compression:\n  scorer: query_attention\n  ratio: 0.25\n  unit: span\n  protect_initial: 4\n"
    tmp = src.parent / "_tmp_compression_leak.yaml"
    tmp.write_text(text, encoding="utf-8")
    try:
        with pytest.raises(Exception, match="does not apply a compression block"):
            load_experiment_configs(tmp)
    finally:
        tmp.unlink()


def test_the_retention_task_still_accepts_its_compression_block():
    """The guard must not break the one task kind that does prune."""
    import pathlib

    from raq_kv.config.loader import load_experiment_configs

    src = pathlib.Path(__file__).resolve().parents[1] / "configs" / "experiments" / "m5h3_1p05_joint.yaml"
    cfg = load_experiment_configs(src)[0]
    assert cfg.task.kind == "retention" and cfg.compression is not None
