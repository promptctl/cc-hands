"""Which tests a shard of a run takes, so a run's tests can be several CI jobs that each end inside the job's 5 minutes.

    pytest --shard=K/N ...          the Kth of N shards of the tests the rest of the command line selects
    uv run --no-sync python tests/shards.py LOG...      tests/shard-costs.json, from the cost reports in CI job logs

Every shard of one commit takes the same partition: it is a pure function of the collected tests and tests/shard-costs.json,
both of which the commit holds, so each test runs in exactly one shard and none is dropped by shards that disagree.
"""

import argparse
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from costs import costs_in, module, ran_for

COSTS = Path(__file__).with_name("shard-costs.json")

# A test that cost less than this is not listed in the file; it is packed at `Costs.fallback`.
LISTED = 0.5


@dataclass(frozen=True)
class Shard:
    """The `index`th (from 0) of `count` shards."""

    index: int
    count: int

    def __post_init__(self) -> None:
        # [LAW:types-are-the-program] a Shard is one of its count, whoever builds it.
        if not 0 <= self.index < self.count:
            raise ValueError(f"shard index {self.index} is not one of {self.count} shards")

    @classmethod
    def parse(cls, text: str) -> "Shard":
        """[LAW:parse-dont-validate] `K/N` with 1 <= K <= N, the one spelling of a shard the command line has."""
        refusal = argparse.ArgumentTypeError(f"{text!r} is not a shard: K/N with 1 <= K <= N, as 2/3")
        found = re.fullmatch(r"([1-9][0-9]*)/([1-9][0-9]*)", text)
        if found is None:
            raise refusal
        try:
            return cls(int(found[1]) - 1, int(found[2]))
        except ValueError as error:
            raise refusal from error

    def __str__(self) -> str:
        return f"{self.index + 1}/{self.count}"


# The shard of a run that asks for none: all of its tests, the 1st of 1.
WHOLE = Shard(0, 1)


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

    def setup_of(self, name: str) -> float:
        return self.setups.get(name, 0.0)


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
    paying the module's setup. No more pieces than tests or shards: when the setup alone fills a share, pieces only multiply it."""
    if packed(held, costs) <= share:
        return 1
    room = share - costs.setup_of(module(held[0]))
    return min(len(held), count, math.ceil(sum(costs.of(test) for test in held) / room) if room > 0 else count)


def partition(tests: Sequence[str], costs: Costs, count: int) -> list[list[str]]:
    """`count` shards of `tests`, each test in exactly one, each about an equal share of the cost. A module is kept whole,
    its setup paid once, unless it is more than a share, and then it is cut into the fewest pieces whose tests, each piece
    paying the setup again, come to no more than a share. The pieces and the whole modules are then packed as units, each onto
    the lightest shard, which pays a module's setup once however many of its pieces it holds: what `packed` says of the shard."""
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
        shards[lightest] += unit
        loads[lightest] = packed(shards[lightest], costs)
    order = {test: place for place, test in enumerate(tests)}
    return [sorted(shard, key=order.__getitem__) for shard in shards]


def packed(tests: Sequence[str], costs: Costs) -> float:
    """What a shard's `tests` are expected to cost it."""
    return sum(costs.of(test) for test in tests) + sum(costs.setup_of(each) for each in {module(test) for test in tests})


class Sharding:
    """The plugin that takes a run's shard of the tests the rest of the command line selected. It says at the start what the
    shard was packed at, as a job cancelled at its limit never reaches its end, and at the end what the shard cost and which
    tests cost what the file does not list, so a file gone stale shows in the log."""

    def __init__(self, shard: Shard, costs: Costs) -> None:
        self.shard = shard
        self.costs = costs
        self.held = 0
        self.expected = 0.0

    # trylast: after -m, -k and the layer have chosen what the run holds, so the shards cut that set and not a wider one.
    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, config: pytest.Config, items: list[pytest.Item]) -> None:
        mine = set(partition([item.nodeid for item in items], self.costs, self.shard.count)[self.shard.index])
        if items and not mine:
            raise pytest.UsageError(f"shard {self.shard} holds no test: the {len(items)} tests the run selects are too few for {self.shard.count} shards")
        left = [item for item in items if item.nodeid not in mine]
        # As pytest's own -k and -m do: a run that left nothing out has deselected nothing (tests/clockwatch.py reads the hook as "a part").
        if left:
            config.hook.pytest_deselected(items=left)
        items[:] = [item for item in items if item.nodeid in mine]
        self.held = len(items)
        self.expected = packed([item.nodeid for item in items], self.costs)

    def pytest_report_collectionfinish(self) -> str:
        return f"shard {self.shard} holds {self.held} tests, packed at {self.expected:.1f} s"

    def pytest_terminal_summary(self, terminalreporter: pytest.TerminalReporter) -> None:
        reports = [report for each in terminalreporter.stats.values() for report in each if isinstance(report, pytest.TestReport)]
        # A test costs its call and teardown; its setup is its module's, which the file lists apart.
        cost: dict[str, float] = defaultdict(float)
        for report in reports:
            cost[report.nodeid] += 0.0 if report.when == "setup" else report.duration
        unlisted = sorted(test for test, seconds in cost.items() if seconds >= LISTED and test not in self.costs.tests)
        terminalreporter.line(f"shard {self.shard}: packed at {self.expected:.1f} s, cost {sum(report.duration for report in reports):.1f} s")
        terminalreporter.line(f"shard {self.shard}: {len(unlisted)} tests cost {LISTED} s or more and are not in tests/shard-costs.json: {', '.join(unlisted)}")


def costs_from(logs: Mapping[str, str]) -> Costs:
    """The costs of the cost reports (tests/costs.py) in CI job logs, by file name. A module's setup is the seconds its tests
    spent in setup in one log, and the most any log has it spend: a module cut across shards is in several logs, each having
    paid it once. A test's cost is its call and teardown, a fixture the module shares being charged to the first test that
    asks for it, and the most any log has it cost: a shard packed on the slowest runner seen holds on a slow one (one test
    cost 52 s in one job and 117 s in another). A test too quick to list costs what the quick ones did on average. A test the run stopped inside (a job
    cancelled at its limit) was not finished, so it costs what the run spent in it, a floor, until a run that finished it says
    what it costs; the run's seconds are only known of a log that stopped inside the one test."""
    tests: dict[str, float] = {}
    floors: dict[str, float] = {}
    setups: dict[str, float] = {}
    unlisted: list[float] = []
    for name, text in logs.items():
        rows = costs_in(text)
        if not rows:
            raise ValueError(f"{name} holds no cost report rows: it is not a log of tests/costs.py's report, or its run wrote none")
        stopped = [each for each in rows if each.stopped]
        if (ran := ran_for(text)) is not None and len(stopped) == 1:
            floors[stopped[0].test] = ran - sum(each.total for each in rows if not each.stopped) - stopped[0].setup
        in_log: dict[str, float] = defaultdict(float)
        for cost in (each for each in rows if not each.stopped):
            in_log[cost.module] += cost.setup
            if cost.call + cost.teardown >= LISTED:
                tests[cost.test] = max(tests.get(cost.test, 0.0), round(cost.call + cost.teardown, 1))
            else:
                unlisted.append(cost.call + cost.teardown)
        setups |= {each: max(seconds, setups.get(each, 0.0)) for each, seconds in in_log.items()}
    # A run that finished a test, in whichever log, says more of it than one cancelled inside it.
    tests |= {test: round(floor, 1) for test, floor in floors.items() if floor >= LISTED and test not in tests}
    return Costs(dict(sorted(tests.items())), {each: round(seconds, 1) for each, seconds in sorted(setups.items()) if seconds >= LISTED}, round(statistics.fmean(unlisted or [LISTED]), 3))


if __name__ == "__main__":
    logs = {each: Path(each).read_text(errors="replace") for each in sys.argv[1:]}
    costs = costs_from(logs)
    stopped = [cost.test for text in logs.values() for cost in costs_in(text) if cost.stopped]
    print(f"{len(stopped)} tests were stopped inside by their run, and cost no more than the run's seconds in them: {', '.join(stopped)}", file=sys.stderr)
    print(json.dumps({"fallback": costs.fallback, "setups": costs.setups, "tests": costs.tests}, indent=1))
