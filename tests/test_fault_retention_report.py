"""The two-axis report is the canonical path for D43's structural claim, and it shipped with two bugs.

Both produced numbers nobody could reproduce, and neither was visible in the output: the composition control
said "composition moved" where nothing vulnerable had, and a contrast's confidence interval depended on how many
*other* contrasts were requested in the same command. A script that a claim rests on needs tests for exactly
that class of defect.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "report_fault_retention", Path(__file__).resolve().parents[1] / "scripts" / "report_fault_retention.py")
rep = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rep)


@pytest.mark.parametrize("target,expected", [
    ("code", "code"),
    ("scale", "meta"),
    ("zero", "meta"),
    # The bug: a substring test for "scale" matches these too, so a metadata-SECDED arm read as though 8.4% of
    # its flips hit metadata against a guarded arm's 6.2%. The extra bits are the protection, not the payload.
    ("check:scale", "overhead"),
    ("check:zero", "overhead"),
    ("bound:scale", "overhead"),
    ("bound:zero", "overhead"),
])
def test_protection_overhead_is_not_counted_as_payload_metadata(target, expected):
    assert rep._classify(target) == expected


def test_a_secded_arm_does_not_look_like_it_has_more_vulnerable_metadata():
    """The real region names from a pruned SECDED arm and a pruned range-guard arm, side by side."""
    secded = {"code": 58032128, "scale": 1824256, "zero": 1824256, "check:scale": 684096, "check:zero": 684096}
    guarded = {"code": 58032128, "scale": 1824256, "zero": 1824256, "bound:scale": 1792, "bound:zero": 1792}

    def payload_share(comp):
        agg = {}
        for k, v in comp.items():
            agg[rep._classify(k)] = agg.get(rep._classify(k), 0) + v
        return agg["meta"] / sum(agg.values())

    # Payload metadata is nearly identical; only the overhead differs, and that is what SECDED buys.
    assert payload_share(secded) == pytest.approx(payload_share(guarded), abs=0.003)
    assert rep._classify("check:scale") != rep._classify("scale")


def test_a_contrast_interval_does_not_depend_on_what_else_was_requested():
    """The bug: one shared generator, so the same comparison returned two intervals depending on call order.

    Here `contrast` is called in two different orders and the pair's interval must be identical both times.
    """
    rng = np.random.default_rng(12345)
    a, b, c = (rng.random(56) for _ in range(3))

    alone = rep.contrast("x", "y", a, b, 0)
    # Same pair, but with two other contrasts drawn first -- which used to consume the shared generator.
    rep.contrast("p", "q", a, c, 0)
    rep.contrast("r", "s", b, c, 0)
    in_batch = rep.contrast("x", "y", a, b, 0)

    assert alone["ci"] == in_batch["ci"], "the interval moved because other contrasts were requested"
    assert alone["difference"] == in_batch["difference"]


def test_the_interval_still_responds_to_the_seed_and_to_the_arm_names():
    """Determinism must not have been bought by ignoring the seed, or by keying every pair the same."""
    rng = np.random.default_rng(7)
    a, b = rng.random(56), rng.random(56)
    assert rep.contrast("x", "y", a, b, 0)["ci"] != rep.contrast("x", "y", a, b, 1)["ci"]
    assert rep.contrast("x", "y", a, b, 0)["ci"] != rep.contrast("y", "x", a, b, 0)["ci"]


def test_the_pre_registered_rule_needs_both_halves():
    """D31: p < 0.05 AND a CI excluding zero. Meeting one is a direction, and this project has withdrawn
    findings for reporting the second kind as the first."""
    rng = np.random.default_rng(3)
    # Identical arms: no difference, no discordance, p forced to 1 rather than left to the test.
    same = rng.random(56)
    r = rep.contrast("x", "y", same, same, 0)
    assert r["discordant"] == 0 and r["p"] == 1.0 and not r["meets_rule"]

    # A large, consistent shift meets both halves.
    r2 = rep.contrast("x", "y", same, np.clip(same + 0.5, 0, 1), 0)
    assert r2["p"] < 0.05 and r2["ci"][0] > 0 and r2["meets_rule"]


def test_task_failure_and_the_proxy_are_computed_from_different_fields():
    """The first version of this analysis used `answer_changed` and got different numbers than the endpoint."""
    record = {"faults": [
        {"trial": -1, "answer_changed": False, "faulty_correct": True},
        {"trial": 0, "answer_changed": True, "faulty_correct": True},    # changed but still correct
        {"trial": 1, "answer_changed": True, "faulty_correct": False},   # changed and wrong
        {"trial": 2, "answer_changed": False, "faulty_correct": True},
    ]}
    data = {"s": record}
    assert rep.rates(data, ["s"], "task_failure")[0] == pytest.approx(1 / 3)
    assert rep.rates(data, ["s"], "answer_changed")[0] == pytest.approx(2 / 3)
    # The control (trial -1) is excluded from both; it is checked separately and must show zero damage.
    assert rep._controls_clean(data) == (0, 1)


def test_pairing_excludes_a_sample_that_pruning_alone_broke():
    """`unpruned_correct is False` means the full cache could not answer either, so the faults are not to blame."""
    arms = {
        "unpruned": {"a": {"golden_correct": True}, "b": {"golden_correct": True}},
        "pruned": {"a": {"golden_correct": True, "unpruned_correct": True},
                   "b": {"golden_correct": True, "unpruned_correct": False}},
    }
    assert rep.shared_samples(arms) == ["a"]
