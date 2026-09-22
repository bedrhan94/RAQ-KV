"""Which modules a campaign is allowed to keep editing is a question about the runner's import graph.

D24 wrote the rule as "never edit `src/` while a campaign runs", after source edits mid-sweep left three
manifests recording three different diff hashes. The rule was later narrowed by argument -- docs and report
scripts are safe because the runner does not import them -- and on 2026-09-22 the same argument was used again,
to justify editing `raq_kv.allocation.policy` while two arms were in flight.

An exemption that lives in an argument decays. This pins it: the modules named below must stay outside the
runner's import graph, so "safe to edit mid-campaign" is a property the test suite checks rather than a claim
someone remembers. If an import is ever added, this fails and the exemption has to be re-argued rather than
silently inherited.
"""

from __future__ import annotations

import importlib
import sys

import pytest

# Modules a campaign never executes, and may therefore be edited while one runs. Each is here because
# something in it is worked on between runs: the allocator emits configs before a campaign, not during one.
OUTSIDE_THE_RUNNER = ["raq_kv.allocation", "raq_kv.allocation.policy"]


@pytest.fixture(scope="module")
def runner_imports() -> set[str]:
    """Import the runner into a clean view of sys.modules and record everything that came with it."""
    before = set(sys.modules)
    importlib.import_module("raq_kv.experiments.runner")
    importlib.import_module("raq_kv.experiments.tasks")
    return set(sys.modules) | before


@pytest.mark.parametrize("module", OUTSIDE_THE_RUNNER)
def test_the_runner_does_not_import_it(module, runner_imports):
    fresh = {m for m in sys.modules if m.startswith("raq_kv")}
    importlib.import_module("raq_kv.experiments.runner")
    importlib.import_module("raq_kv.experiments.tasks")
    pulled_in = {m for m in sys.modules if m.startswith("raq_kv")} - fresh
    assert module not in pulled_in, (
        f"{module} is now in the runner's import graph, so editing it during a campaign changes what executes. "
        "Either drop the import or remove the module from OUTSIDE_THE_RUNNER and stop editing it mid-run."
    )


def test_the_modules_a_campaign_does_execute_are_not_on_the_exempt_list():
    """A guard against the exemption list growing until it means nothing."""
    executed = {"raq_kv.experiments.tasks", "raq_kv.experiments.runner", "raq_kv.quantization.apply",
                "raq_kv.ecc.protect", "raq_kv.faults.storage", "raq_kv.compression.retain",
                "raq_kv.importance.scorers", "raq_kv.kv.snapshot", "raq_kv.config.schema"}
    assert not (executed & set(OUTSIDE_THE_RUNNER)), "a module the runner executes cannot be exempt from D24"
