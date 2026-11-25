"""What outside.yml says of its jobs, held by the tests it runs and what they cost (tests/outside_jobs.py says what a job spends):
every job ends inside 5 minutes but one that holds a test needing more alone, and its step ends inside the job's limit. Held here,
on every PR, since a job that outgrows its limit shows on CI as a cancellation and not as the test that grew.

The tests are the outside ones the unit run collects before its layer drops them (conftest's OUTSIDE_TESTS), and what each costs
is tests/shard-costs-outside.json, which names every one of them."""

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from conftest import OUTSIDE_TESTS
from hands.voice import pins
from outside_jobs import ENDING, AFTER, BEFORE, LIMIT, RESTORING, SLOWDOWN, allowed, budget_of, limit_of
from shards import OUTSIDE_COSTS, Costs, Shard, partition

ROOT = Path(__file__).parents[1]
WORKFLOW = yaml.safe_load((ROOT / ".github" / "workflows" / "outside.yml").read_text())


def entries() -> list[dict[str, Any]]:
    return WORKFLOW["jobs"]["tests"]["strategy"]["matrix"]["include"]


def groups_of(entry: dict[str, Any]) -> frozenset[str]:
    return frozenset(name for name in entry["weights"].split(",") if name)


def test_the_restoring_costs_are_of_the_groups_the_pins_name() -> None:
    assert set(RESTORING) == set(pins.GROUPS)


def test_every_pinned_group_is_a_weights_job_and_a_restore_step_of_the_tests_job() -> None:
    assert WORKFLOW["jobs"]["weights"]["strategy"]["matrix"]["group"] == list(pins.GROUPS)
    restores = [step["name"] for step in WORKFLOW["jobs"]["tests"]["steps"] if step.get("uses", "").startswith("actions/cache/restore")]
    assert restores == [f"The {group} weights" for group in pins.GROUPS]


def test_every_job_but_the_tests_ends_inside_5_minutes() -> None:
    limits = {name: job["timeout-minutes"] for name, job in WORKFLOW["jobs"].items() if name != "tests"}
    assert limits == {name: LIMIT for name in limits}
    assert WORKFLOW["jobs"]["tests"]["timeout-minutes"] == "${{ matrix.limit }}"


@pytest.mark.parametrize("entry", entries(), ids=lambda entry: f"{entry['weights'] or 'none'} {entry['shard']}")
def test_a_shard_names_groups_that_are_pinned(entry: dict[str, Any]) -> None:
    assert groups_of(entry) <= set(pins.GROUPS)


def test_the_shards_of_a_set_are_every_number_from_one_to_the_count_of_them_once() -> None:
    by_set: dict[frozenset[str], list[Shard]] = {}
    for entry in entries():
        by_set.setdefault(groups_of(entry), []).append(Shard.parse(entry["shard"]))
    for held in by_set.values():
        assert sorted(shard.index for shard in held) == list(range(1, held[0].total + 1))
        assert {shard.total for shard in held} == {held[0].total}


def test_every_outside_test_the_run_collected_is_priced_and_runs_in_a_job(pytestconfig: pytest.Config) -> None:
    collected = pytestconfig.stash[OUTSIDE_TESTS]
    costs = Costs.load(OUTSIDE_COSTS)
    unpriced = sorted(test for test in collected if test not in costs.tests)
    assert not unpriced, f"{len(unpriced)} outside tests are in no price of {OUTSIDE_COSTS.name}, so no job's budget holds them: {unpriced[:5]} (tests/shards.py writes the file from a run's logs)"
    unrun = sorted({groups for groups in collected.values()} - {groups_of(entry) for entry in entries()}, key=sorted)
    assert not unrun, f"the outside tests that load {[sorted(groups) for groups in unrun]} are in no job of outside.yml"


def test_every_shard_fits_its_step_and_its_step_fits_its_job_which_is_5_minutes_unless_a_test_it_holds_needs_more(pytestconfig: pytest.Config) -> None:
    collected = pytestconfig.stash[OUTSIDE_TESTS]
    costs = Costs.load(OUTSIDE_COSTS)
    over: list[str] = []
    for entry in entries():
        held = groups_of(entry)
        ids = [test for test, groups in collected.items() if groups == held]
        if not ids:
            continue  # a run of some of the files collects none of this set's
        shard = Shard.parse(entry["shard"])
        packed = partition(ids, costs, shard.total)[shard.index - 1]
        name = f"{entry['weights'] or 'none'} {entry['shard']}"
        if entry["budget"] < budget_of(packed.seconds):
            over.append(f"{name}: {packed.seconds:.0f} s packed need {budget_of(packed.seconds)} minutes of step at {SLOWDOWN}x, with its collection; the entry gives {entry['budget']}")
        if entry["limit"] < limit_of(held, entry["budget"]):
            over.append(f"{name}: {BEFORE:.0f} s before its step, {sum(RESTORING[group] for group in held):.0f} restoring, {entry['budget']} minutes of step, {ENDING + AFTER:.0f} s ending and after it, are over its {entry['limit']} minutes")
        if entry["limit"] > allowed(held, packed.tests, costs):
            over.append(f"{name}: {entry['limit']} minutes, where no test it holds needs more than {allowed(held, packed.tests, costs)} alone: cut the set into more shards")
    assert not over, "\n".join(over)


def test_the_cost_file_is_one_of_numbers_in_seconds_that_the_plugin_reads() -> None:
    assert json.loads(OUTSIDE_COSTS.read_text())["tests"]
