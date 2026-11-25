"""What an outside job spends of its minutes, and the jobs outside.yml's matrix has to be for the tests to fit them.

A job pays for its checkout, environment and caches before its tests, the weights' restore for the groups its tests load, pytest's
collection and session start inside the tests' own step, and the report and the post steps after it. The step carries its own
limit under the job's, so a test that outgrew its price ends the step and the report still names it, where the job's limit would
cancel everything and leave a cancellation to read: the step's limit and what the job pays around it fit the job's.

Every number is measured off the steps of outside.yml's dispatched run (docs/testing.md says which run), the slowest of its jobs.

    uv run python tests/outside_jobs.py     the matrix the cost file (tests/shard-costs-outside.json) asks for, as the entries of outside.yml
"""

import math
import sys
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

import pytest

from hands.voice import pins
from shards import OUTSIDE_COSTS, Costs, Shard, partition

# The limit every job is held to, in minutes.
LIMIT = 5
# Checkout, the environment and the caches: what a job pays before it restores weights.
BEFORE = 95.0
# What each group's restore adds to that; the cache of a group is as large as its weights.
RESTORING: Mapping[str, float] = {"streaming": 70.0, "smart_turn": 10.0, "whisper": 35.0, "voiceprint": 10.0}
# pytest's collection and session start, inside the step, which every job pays whatever its shard holds.
COLLECTION = 30.0
# What a step takes to end after its limit, then what the cost report, the upload and the post steps take after the step.
ENDING = 13.0
AFTER = 20.0
# Seconds on a runner for each second a shard is priced at. The prices are the most any run of the cost file's logs gave, so this is
# the room beyond the slowest run seen.
SLOWDOWN = 1.3


def restoring(held: frozenset[str]) -> float:
    return sum(RESTORING[group] for group in held)


def budget_of(packed: float) -> float:
    """The minutes of step that hold a shard priced at `packed` seconds, to the half minute: GitHub reads a limit as a number."""
    return math.ceil((packed * SLOWDOWN + COLLECTION) / 30) / 2


def limit_of(held: frozenset[str], budget: float) -> int:
    """The minutes of job that hold a step of `budget` minutes, the step's ending and what comes after it included."""
    return math.ceil((BEFORE + restoring(held) + budget * 60 + ENDING + AFTER) / 60)


def alone(held: frozenset[str], test: str, costs: Costs) -> int:
    """The minutes of job a test needs on its own."""
    return limit_of(held, budget_of(partition([test], costs, 1)[0].seconds))


@dataclass(frozen=True)
class Entry:
    weights: frozenset[str]
    shard: Shard
    budget: float
    limit: int

    def line(self) -> str:
        return f'          - {{ weights: "{",".join(sorted(self.weights))}", shard: {self.shard}, budget: {self.budget}, limit: {self.limit} }}'


def allowed(held: frozenset[str], tests: frozenset[str], costs: Costs) -> int:
    """The minutes of job a shard may take: LIMIT, or the most the longest test it holds needs on its own, which is the reason a
    job runs past LIMIT: no cut of the tests makes it shorter."""
    return max([LIMIT, *(alone(held, test, costs) for test in tests)])


def sized(tests_by_set: Mapping[frozenset[str], list[str]], costs: Costs) -> list[Entry]:
    """The fewest shards of each set whose every job is inside what it is allowed."""
    entries: list[Entry] = []
    for held, ids in sorted(tests_by_set.items(), key=lambda item: sorted(item[0])):
        for total in range(1, len(ids) + 1):
            made = [Entry(held, Shard(index, total), budget_of(each.seconds), limit_of(held, budget_of(each.seconds))) for index, each in enumerate(partition(ids, costs, total), 1)]
            if all(entry.limit <= allowed(held, each.tests, costs) for entry, each in zip(made, partition(ids, costs, total))):
                entries += made
                break
    return entries


def collected() -> Mapping[frozenset[str], list[str]]:
    """The outside tests by the groups they load, as the unit run's collection finds them."""
    found: dict[str, frozenset[str]] = {}

    class Grab:
        def pytest_collection_finish(self, session: pytest.Session) -> None:
            found.update(session.config.stash[sys.modules["conftest"].OUTSIDE_TESTS])

    pytest.main(["--layer=outside", "--collect-only", "-q", "-p", "no:randomly"], plugins=[Grab()])
    by_set: dict[frozenset[str], list[str]] = defaultdict(list)
    for test, groups in found.items():
        by_set[groups].append(test)
    return by_set


if __name__ == "__main__":
    assert set(RESTORING) == set(pins.GROUPS)
    wanted = sized(collected(), Costs.load(OUTSIDE_COSTS))
    print("--- matrix")
    for entry in wanted:
        print(entry.line())
    print(f"--- {len(wanted)} jobs, {sum(entry.limit for entry in wanted)} job minutes at most")
