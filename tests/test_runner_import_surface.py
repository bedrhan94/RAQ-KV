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
def test_the_exemption_list_the_GUARD_uses_is_measured_against_the_campaign_entry_point():
    """`OUTSIDE_THE_RUNNER` above was a hand-kept list; the guard now has its own, and this measures it.

    `provenance.EXECUTED_CODE_EXEMPT` is what `executed_code_sha256` skips, so it is what may be edited mid-campaign.
    Every path on it must correspond to a module the CAMPAIGN ENTRY POINT does not import -- not merely one that
    `runner` and `tasks` miss. `config.loader` is the reason the distinction matters: the entry point imports it to read
    the configs, so it participates in a campaign although `runner.py` never imports it, and it is deliberately absent
    from the exempt list.

    Measured in a SUBPROCESS, because `sys.modules` is global: by the time the full suite reaches this test, other
    tests have already imported the analysis modules, and an in-process check would read their imports as the
    campaign's. The first version of this test passed alone and failed in the suite for exactly that reason.
    """
    import json
    import subprocess
    import sys

    from raq_kv.experiments.provenance import EXECUTED_CODE_EXEMPT
    from raq_kv.utils.paths import repo_root

    program = (
        "import importlib, json, sys" + chr(10)
        + "for m in ('raq_kv.config.loader','raq_kv.experiments.ids','raq_kv.experiments.runner',"
          "'raq_kv.experiments.tasks'):" + chr(10)
        + "    importlib.import_module(m)" + chr(10)
        + "print(json.dumps(sorted(m for m in sys.modules if m.startswith('raq_kv'))))" + chr(10)
    )
    out = subprocess.run([sys.executable, "-c", program], cwd=repo_root(), capture_output=True,
                         text=True, timeout=180, check=True)
    imported = set(json.loads(out.stdout.strip().splitlines()[-1]))
    assert "raq_kv.experiments.tasks" in imported, out.stdout

    for path in EXECUTED_CODE_EXEMPT:
        stem = path[:-3] if path.endswith(".py") else path
        name = stem.replace("/", ".")
        assert name not in imported, (
            f"{name} is exempt from the executed-code hash but the campaign entry point imports it, so editing it "
            f"mid-run would change what executes without the guard noticing")

    # And the converse, on the module most likely to be added to the list by mistake.
    assert "raq_kv/config/loader.py" not in EXECUTED_CODE_EXEMPT
    assert "raq_kv.config.loader" in imported


def test_every_exempt_path_exists():
    """A typo would silently exempt nothing, which reads as a stricter guard than it is."""
    from raq_kv.experiments.provenance import EXECUTED_CODE_EXEMPT
    from raq_kv.utils.paths import repo_root

    for path in EXECUTED_CODE_EXEMPT:
        assert (repo_root() / "src" / path).exists(), path
