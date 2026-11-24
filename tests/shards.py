"""Which tests a shard of a run takes, so a tier's tests can run as several CI jobs that each end inside the job's 5 minutes.

    pytest --shard=K/N ...          the Kth of N shards of the tests the rest of the command line selects
    uv run --no-sync python tests/shards.py LOG...      tests/shard-costs.json, from the cost reports in CI job logs

Every shard of one commit takes the same partition: it is a pure function of the collected tests and tests/shard-costs.json,
both of which the commit holds, so each test runs in exactly one shard and none is dropped by shards that disagree.
"""

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

COSTS = Path(__file__).with_name("shard-costs.json")

# A test that cost less than this is not listed in the file; it is packed at `Costs.fallback`.
LISTED = 0.5
# A report row: total, setup, call and teardown seconds, then the test, as tests/costs.py prints it.
ROW = re.compile(r"^\s*(\d+\.\d{3})\s+(\d+\.\d{3})\s+(\d+\.\d{3})\s+(\d+\.\d{3})\s+(tests/.+?)\s*$")
# What CI puts in front of each line of a job's log.
STAMP = re.compile(r"^\S+Z ")


@dataclass(frozen=True)
class Shard:
    """The `index`th (from 0) of `count` shards."""

    index: int
    count: int

    @classmethod
    def parse(cls, text: str) -> "Shard":
        """[LAW:parse-dont-validate] `K/N` with 1 <= K <= N, the one spelling of a shard the command line has."""
        found = re.fullmatch(r"([1-9][0-9]*)/([1-9][0-9]*)", text)
        if found is None or int(found[1]) > int(found[2]):
            raise argparse.ArgumentTypeError(f"{text!r} is not a shard: K/N with 1 <= K <= N, as 2/3")
        return cls(int(found[1]) - 1, int(found[2]))

    def __str__(self) -> str:
        return f"{self.index + 1}/{self.count}"


def module(test: str) -> str:
    return test.partition("::")[0]


@dataclass(frozen=True)
class Costs:
    """What running a test costs a shard, in seconds. A module's shared fixtures are charged once to the shard that holds any
    of its tests, not to each test: a shard that takes a module's tests pays its models' load, which is what splitting a
    module across shards multiplies. A test the file does not list costs `fallback`."""

    tests: Mapping[str, float]
    setups: Mapping[str, float]
    fallback: float

    @classmethod
    def load(cls, path: Path) -> "Costs":
        record: dict[str, Any] = json.loads(path.read_text())
        return cls(record["tests"], record["setups"], record["fallback"])

    def of(self, test: str) -> float:
        return self.tests.get(test, self.fallback)


def balanced(tests: Sequence[str], costs: Costs, count: int) -> list[list[str]]:
    """`count` bins of `tests`, each test in one, the heaviest first into the lightest bin: the longest-processing-time
    heuristic for multiway number partitioning. Ties go to the lowest bin and the first name, so it is deterministic."""
    loads = [0.0] * count
    held: list[list[str]] = [[] for _ in range(count)]
    for test in sorted(tests, key=lambda test: (-costs.of(test), test)):
        lightest = min(range(count), key=lambda bin: (loads[bin], bin))
        loads[lightest] += costs.of(test)
        held[lightest].append(test)
    return held


def pieces(held: Sequence[str], costs: Costs, share: float, count: int) -> int:
    """How many pieces a module's tests are cut into: one when it fits a share, else the fewest that each fit one, each
    paying the module's setup; when the setup alone fills a share no piece fits in, one test to a piece."""
    if packed(held, costs) <= share:
        return 1
    room = share - costs.setups.get(module(held[0]), 0.0)
    return min(len(held), count, math.ceil(sum(costs.of(test) for test in held) / room)) if room > 0 else len(held)


def partition(tests: Sequence[str], costs: Costs, count: int) -> list[list[str]]:
    """`count` shards of `tests`, each test in exactly one, each about an equal share of the cost. A module is kept whole,
    its setup paid once, unless it is more than a share, and then it is cut into the fewest pieces whose tests, each piece
    paying the setup again, come to no more than a share. The pieces and the whole modules are then packed as units."""
    by_module: dict[str, list[str]] = defaultdict(list)
    for test in tests:
        by_module[module(test)].append(test)
    share = packed(tests, costs) / count
    units: list[list[str]] = []
    for held in by_module.values():
        units += [piece for piece in balanced(held, costs, pieces(held, costs, share, count)) if piece]
    shards: list[list[str]] = [[] for _ in range(count)]
    loads = [0.0] * count
    for unit in sorted(units, key=lambda unit: (-packed(unit, costs), unit[0])):
        lightest = min(range(count), key=lambda shard: (loads[shard], shard))
        loads[lightest] += packed(unit, costs)
        shards[lightest] += unit
    order = {test: place for place, test in enumerate(tests)}
    return [sorted(shard, key=order.__getitem__) for shard in shards]


def packed(tests: Sequence[str], costs: Costs) -> float:
    """What a shard's `tests` are expected to cost it."""
    return sum(costs.of(test) for test in tests) + sum(costs.setups.get(each, 0.0) for each in {module(test) for test in tests})


class Sharding:
    """The plugin that takes a run's shard of the tests the rest of the command line selected, and says at the end how the
    shard's own cost compared with the one it was packed at, so a file gone stale shows in the log before it ends a job."""

    def __init__(self, shard: Shard, costs: Costs) -> None:
        self.shard = shard
        self.costs = costs
        self.expected = 0.0

    # trylast: after -m, -k and the layer have chosen what the run holds, so the shards cut that set and not a wider one.
    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, config: pytest.Config, items: list[pytest.Item]) -> None:
        mine = set(partition([item.nodeid for item in items], self.costs, self.shard.count)[self.shard.index])
        config.hook.pytest_deselected(items=[item for item in items if item.nodeid not in mine])
        items[:] = [item for item in items if item.nodeid in mine]
        self.expected = packed([item.nodeid for item in items], self.costs)

    def pytest_terminal_summary(self, terminalreporter: pytest.TerminalReporter) -> None:
        measured = sum(report.duration for reports in terminalreporter.stats.values() for report in reports if isinstance(report, pytest.TestReport))
        terminalreporter.line(f"shard {self.shard}: packed at {self.expected:.1f} s, cost {measured:.1f} s")


def costs_from(logs: Sequence[str]) -> Costs:
    """The costs of the cost reports (tests/costs.py) in CI job logs. A module's setup is the seconds its tests spent in
    setup, and a test's cost its call and teardown: a fixture the module shares is charged to the first test that asks for it. A
    test too quick to list costs what the quick ones did on average."""
    tests: dict[str, float] = {}
    setups: dict[str, float] = defaultdict(float)
    unlisted: list[float] = []
    for log in logs:
        for line in log.splitlines():
            row = ROW.match(STAMP.sub("", line))
            if row is None:
                continue
            _, setup, call, teardown, test = row.groups()
            setups[module(test)] += float(setup)
            if float(call) + float(teardown) >= LISTED:
                tests[test] = round(float(call) + float(teardown), 1)
            else:
                unlisted.append(float(call) + float(teardown))
    return Costs(dict(sorted(tests.items())), {each: round(seconds, 1) for each, seconds in sorted(setups.items()) if seconds >= LISTED}, round(sum(unlisted) / len(unlisted), 3))


if __name__ == "__main__":
    costs = costs_from([Path(each).read_text(errors="replace") for each in sys.argv[1:]])
    print(json.dumps({"fallback": costs.fallback, "setups": costs.setups, "tests": costs.tests}, indent=1))
