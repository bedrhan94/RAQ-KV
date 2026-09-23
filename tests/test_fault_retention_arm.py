"""`protection_arm` can now prune, which is the first time this project varies two axes in one experiment.

Row 10 of `docs/m4_exit.md`: all 30 retention runs were fault-free and all M3 arms were unpruned, so the
allocator's assumption that the reliability tax is flat in retention had never been tested. These pin the parts
of that change that are easy to get wrong and impossible to see in a result.
"""

from __future__ import annotations

import glob
import json
import pathlib

import pytest
from pydantic import ValidationError

from raq_kv.config.loader import load_experiment_configs
from raq_kv.config.schema import ProtectionArmTask
from raq_kv.experiments.ids import experiment_id

CONFIGS = sorted(glob.glob(str(pathlib.Path(__file__).resolve().parents[1] / "configs" / "experiments" / "*.yaml")))


def test_an_arm_needs_exactly_one_of_a_rate_and_a_count():
    """Both is an exposure nobody can state; neither is an exposure that does not exist."""
    assert ProtectionArmTask(ber=1e-6).ber == 1e-6
    assert ProtectionArmTask(n_flips=575).n_flips == 575
    with pytest.raises(ValidationError, match="exactly one"):
        ProtectionArmTask(ber=1e-6, n_flips=575)
    with pytest.raises(ValidationError, match="exactly one"):
        ProtectionArmTask()


def test_every_existing_experiment_identity_is_unchanged():
    """Relaxing the D42 guard let `compression` reach `_pruning_token` on a second task kind.

    `_pruning_token` reads `cfg.compression` unconditionally, so if a config with no block stopped rendering
    `p0` the change would silently orphan every M3 run directory. The snapshot was taken before the schema was
    touched; this is the assertion that it never moved.
    """
    before = json.loads((pathlib.Path(__file__).parent / "experiment_identities.json").read_text(encoding="utf-8"))
    after = {}
    for path in CONFIGS:
        for cfg in load_experiment_configs(pathlib.Path(path)):
            after[cfg.name] = experiment_id(cfg)
    moved = {k: (before[k], after[k]) for k in before if k in after and before[k] != after[k]}
    assert not moved, (
        f"experiment identities moved: {moved}. Raw results already live under the old ids, so this orphans "
        "them. If the change is deliberate, update tests/experiment_identities.json and say so in a decision "
        "record -- never just to make the test pass."
    )
    assert not set(before) - set(after), "a config stopped loading entirely"


def test_a_plain_protection_arm_still_declares_no_pruning():
    """An M3 arm has no compression block, so it must keep rendering `p0` and build no scorer."""
    src = pathlib.Path(__file__).resolve().parents[1] / "configs" / "experiments" / "m3_arm_h_range_guard.yaml"
    cfg = load_experiment_configs(src)[0]
    assert cfg.compression is None
    assert "_p0_" in experiment_id(cfg)


def test_a_fault_arm_that_prunes_is_accepted_and_names_itself_so(tmp_path):
    """The combination D42 refused is now legal for this one task kind, and the slug says what it does."""
    src = pathlib.Path(__file__).resolve().parents[1] / "configs" / "experiments" / "m3_arm_h_range_guard.yaml"
    text = src.read_text(encoding="utf-8").replace("name: m3_arm_h_range_guard", "name: _tmp_fault_retention")
    text = text.replace("  - models/qwen3_1p7b.yaml", "  - models/qwen3_1p7b_eager.yaml")
    text += "compression:\n  scorer: query_attention\n  ratio: 0.25\n  unit: span\n  protect_initial: 4\n"
    tmp = src.parent / "_tmp_fault_retention.yaml"
    tmp.write_text(text, encoding="utf-8")
    try:
        cfg = load_experiment_configs(tmp)[0]
        assert cfg.task.kind == "protection_arm" and cfg.compression is not None
        assert "_p25qaS_" in experiment_id(cfg), "a pruning arm must be named as one, or two arms collide"
    finally:
        tmp.unlink()


def test_the_shared_plan_builder_is_the_one_both_tasks_use():
    """Two copies of the span/token branch would let the fault and retention milestones prune differently.

    They are meant to be compared in exactly that place, so the import is asserted rather than trusted.
    """
    import inspect

    from raq_kv.experiments import tasks

    source = inspect.getsource(tasks.ProtectionArm.run_unit)
    assert "build_retention_plan(" in source
    assert "select_keep_spans(" not in source, "ProtectionArm re-implemented span selection"
    assert "select_keep(" not in source, "ProtectionArm re-implemented token selection"


def test_the_golden_runs_through_the_pruned_cache():
    """`task_failure` must isolate fault damage, so the arm's no-fault reference is its OWN cache.

    A golden taken from the full cache would fold pruning damage into every trial. `unpruned_correct` is the
    separate field that keeps the pruning cost visible instead of losing it.
    """
    import inspect

    from raq_kv.experiments import tasks

    source = inspect.getsource(tasks.ProtectionArm.run_unit)
    prune_at = source.index("apply_retention(")
    golden_at = source.index("golden_gen = greedy_decode(")
    assert prune_at < golden_at, "the golden is generated before pruning, so it is not this arm's reference"
    assert "unpruned_correct" in source, "the full-cache reference is not recorded, so pruning cost is invisible"


def test_the_matched_exposure_basis_needs_a_rate_to_apply():
    """`ber_basis` says what a rate is applied to, so it is meaningless beside an absolute count."""
    assert ProtectionArmTask(ber=1e-6, ber_basis="unpruned").ber_basis == "unpruned"
    assert ProtectionArmTask(ber=1e-6).ber_basis is None, "None means stored; a real default would shift every identity"
    with pytest.raises(ValidationError, match="needs `ber`"):
        ProtectionArmTask(n_flips=504, ber_basis="unpruned")


def test_the_matched_exposure_arm_derives_its_count_per_sample():
    """A fixed count would give a 512-token and a 4096-token sample the same exposure.

    The probe spans an eightfold range of context, so storage varies eightfold with it. The attribution arm
    must therefore derive the unpruned count from *that sample's* cache, not from a global constant -- which is
    the bug this project would have shipped had the matched-count arm been a single `n_flips` in a YAML file.
    """
    import inspect

    from raq_kv.experiments import tasks

    source = inspect.getsource(tasks.ProtectionArm.run_unit)
    assert 'ber_basis == "unpruned"' in source
    assert "flips_for_rate(full_qkv, full_guard" in source, "the count is not derived from the unpruned cache"
    # The unpruned reference must be captured BEFORE pruning overwrites the snapshot, or it is the pruned one.
    assert source.index("unpruned_snapshot = snapshot") < source.index("apply_retention(")


def test_the_protection_lookup_declares_what_is_and_is_not_established():
    """D43 falsified separability as a mechanism, and the caveat must travel with the code.

    D43 also overstated it: it wrote that the lookup is "unconservative for aggressively pruned caches" on a
    contrast with 4 discordant samples at p = 0.066, and D44 measured the same contrast at 1e-5 going the other
    way. The docstring now has to say both halves -- the effect is real unprotected, and not measurable under the
    guard -- because the only thing keeping a caller from over- or under-trusting the guard is that text.
    """
    from raq_kv.allocation.policy import _protection

    doc = _protection.__doc__ or ""
    assert "not separable from retention" in doc, "D43's mechanism finding must stay stated"
    assert "open in both directions" in doc, "D44 left the retention term unresolved, not settled"
    assert "not measurable at either rate" in doc
    # The withdrawn overclaim may appear only as a withdrawal, never as an assertion. Checking for the phrase
    # alone would fail on the sentence that retracts it, so what is required is that the retraction is there.
    if "unconservative for aggressively pruned caches" in doc:
        assert "overstated" in doc, "the withdrawn caveat is quoted without being withdrawn"
    # And the lookup really does still take only a rate, so the docstring is not describing a fixed bug.
    import inspect

    assert list(inspect.signature(_protection).parameters) == ["ber"]
