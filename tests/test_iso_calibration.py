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

    Found by trying to design the fault x retention experiment: `ProtectionArmTask.run_unit` went prefill ->
    quantize -> protect -> decode with no compression step, yet the schema accepted the block, it survived into
    the config object, and the slug read it. That is a manifest that is internally consistent and wrong -- D24's
    failure mode reached from the other direction, since D24 was a manifest recording code that did not run and
    this is one recording a transform that did not run.

    `protection_arm` now prunes (D43) and is allowed. `fault_qa` is not, and stands in here: the guard must keep
    refusing every task kind that has not actually implemented the step.
    """
    import pathlib

    from raq_kv.config.loader import load_experiment_configs

    src = pathlib.Path(__file__).resolve().parents[1] / "configs" / "experiments" / "m2_fault_qa_probe.yaml"
    text = src.read_text(encoding="utf-8").replace("name: m2_fault_qa_probe", "name: _tmp_compression_leak")
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


@pytest.mark.parametrize("k,n", [(0, 8), (0, 800), (5, 64), (81, 81), (1278, 1392)])
def test_the_wilson_interval_contains_the_point_estimate_and_stays_in_range(k, n):
    """Shared so one definition cannot drift into two, which is why it has tests at all.

    It was duplicated in `scripts/plot_m3.py` until M2's checklist found that M2's own criterion -- the fault map
    reported with Wilson CI -- had never been met. Copying the estimator into a second script would have been the
    obvious fix; this project has already had to repair two cases of one quantity computed twice and drifting.
    """
    from raq_kv.analysis.stats import wilson

    low, high = wilson(k, n)
    assert 0.0 <= low <= k / n <= high <= 1.0


def test_wilson_stays_finite_at_zero_and_narrows_with_n():
    """The whole reason for choosing it: at k = 0 a normal interval has zero width and claims certainty."""
    from raq_kv.analysis.stats import wilson

    low_small, high_small = wilson(0, 8)
    low_big, high_big = wilson(0, 800)
    assert low_small == 0.0 and high_small > 0.0, "an interval at zero events must not be a point"
    assert high_big < high_small, "0 of 800 is stronger evidence than 0 of 8 and the interval must say so"
    # And a full count is not certainty either.
    assert wilson(81, 81)[0] < 1.0


def test_wilson_on_an_empty_cell_is_maximally_uninformative():
    from raq_kv.analysis.stats import wilson

    assert wilson(0, 0) == (0.0, 1.0)
