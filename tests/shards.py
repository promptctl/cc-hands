"""Which tests one CI job runs when the suite is cut across jobs: `--shard=K/N` keeps the tests the Kth of N shards holds.

A shard is a job, and a job's time is its tests' seconds plus a fixed cost the tests do not control (workflows' headers), so
the cut is made to give every job the same seconds. The partition is longest-processing-time packing over what each test
cost on CI (tests/shard-costs.json), and it is a pure function of the collected test ids and that file: every job of a commit
computes it alone and the N of them agree, so each test runs in exactly one. A stale file only unbalances the shards.

The cost model: a test costs its call and teardown; a module's fixtures cost their setup once to each shard that holds any of
its tests; and a module is cut across shards only when it is more than a share, since a cut pays its setup again.

    uv run python tests/shards.py JOBLOG... > tests/shard-costs.json

JOBLOG is the text of a CI job whose log holds tests/costs.py's report (`gh run view --job <id> --log`); a test costs the most
any log gives it, so the shards hold on a slow runner too.
"""

import json
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import pytest

from costs import module, rows

COSTS = Path(__file__).with_name("shard-costs.json")
# A test is named in the file from this many seconds; the ones under it cost what the rest of them did on average.
LISTED = 0.05
# A test of this many seconds the file does not name makes the shard's summary say the file is stale.
UNPRICED = 0.5


@dataclass(frozen=True)
class Shard:
    index: int
    total: int

    def __post_init__(self) -> None:
        if not 1 <= self.index <= self.total:
            raise ValueError(f"shard {self.index}/{self.total}: K is a number from 1 to N")

    @classmethod
    def parse(cls, text: str) -> Self:
        index, slash, total = text.partition("/")
        if not (slash and index.isdecimal() and total.isdecimal()):
            raise ValueError(f"--shard is K/N, as in 2/4, not {text!r}")
        return cls(int(index), int(total))

    def __str__(self) -> str:
        return f"{self.index}/{self.total}"


# The run that is not cut: the one of 1, which every inner session of a test runs as.
WHOLE = Shard(1, 1)


@dataclass(frozen=True)
class Costs:
    tests: Mapping[str, float]
    setups: Mapping[str, float]
    fallback: float

    @classmethod
    def load(cls, path: Path) -> Self:
        data = json.loads(path.read_text())
        return cls(data["tests"], data["setups"], data["fallback"])

    def of(self, test: str) -> float:
        return self.tests.get(test, self.fallback)

    def setup(self, name: str) -> float:
        return self.setups.get(name, 0.0)


@dataclass(frozen=True)
class Packed:
    tests: frozenset[str]
    seconds: float


def partition(tests: Sequence[str], costs: Costs, total: int) -> list[Packed]:
    by_module: dict[str, list[str]] = defaultdict(list)
    for test in sorted(set(tests)):
        by_module[module(test)].append(test)
    share = (sum(costs.of(test) for test in tests) + sum(costs.setup(name) for name in by_module)) / total
    units: list[tuple[float, list[str]]] = []
    for name, held in by_module.items():
        whole = costs.setup(name) + sum(costs.of(test) for test in held)
        units += [(whole, held)] if whole <= share else [(costs.of(test), [test]) for test in held]
    loads = [0.0] * total
    kept: list[set[str]] = [set() for _ in loads]
    holding: list[set[str]] = [set() for _ in loads]
    for _, held in sorted(units, key=lambda unit: (-unit[0], unit[1][0])):
        shard = min(range(total), key=lambda each: (loads[each], each))
        for test in held:
            if module(test) not in holding[shard]:
                holding[shard].add(module(test))
                loads[shard] += costs.setup(module(test))
            loads[shard] += costs.of(test)
            kept[shard].add(test)
    return [Packed(frozenset(held), seconds) for held, seconds in zip(kept, loads)]


@dataclass
class Sharding:
    """The plugin of a run that is one shard of the suite. It runs after every other hook that drops tests, so the partition
    is of the tests this run would have run whole."""

    shard: Shard
    costs: Costs
    packed: Packed = field(default_factory=lambda: Packed(frozenset(), 0.0))
    spent: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, config: pytest.Config, items: list[pytest.Item]) -> None:
        self.packed = partition([item.nodeid for item in items], self.costs, self.shard.total)[self.shard.index - 1]
        kept = [item for item in items if item.nodeid in self.packed.tests]
        # [LAW:no-silent-failure] a shard with no test is a job that passes having run nothing.
        if items and not kept:
            raise pytest.UsageError(f"shard {self.shard} holds none of the {len(items)} tests collected: it is a shard of more jobs than there are tests")
        config.hook.pytest_deselected(items=[item for item in items if item.nodeid not in self.packed.tests])
        items[:] = kept

    def pytest_report_collectionfinish(self, items: Sequence[pytest.Item]) -> str:
        return f"shard {self.shard}: {len(items)} tests, packed at {self.packed.seconds:.1f} s"

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if report.when in ("call", "teardown"):
            self.spent[report.nodeid] += report.duration

    def pytest_terminal_summary(self, terminalreporter: pytest.TerminalReporter) -> None:
        """[LAW:nothing-unseen] what the shard cost against what it was packed at, and the tests that are slow and are not
        in the file: it is stale, and `uv run python tests/shards.py` writes it from these jobs' logs."""
        terminalreporter.section(f"shard {self.shard}")
        terminalreporter.line(f"tests' call and teardown: {sum(self.spent.values()):.1f} s, packed at {self.packed.seconds:.1f} s")
        unpriced = sorted(test for test, seconds in self.spent.items() if seconds >= UNPRICED and test not in self.costs.tests)
        for test in unpriced:
            terminalreporter.line(f"cost {self.spent[test]:.1f} s and not in tests/shard-costs.json: {test}")


def costs_from(logs: Sequence[str]) -> Costs:
    """The most any log gives each test, and the most any gives each module's setup."""
    tests: dict[str, float] = defaultdict(float)
    setups: dict[str, float] = defaultdict(float)
    for log in logs:
        found = rows(log)
        for cost in found:
            tests[cost.test] = max(tests[cost.test], cost.call + cost.teardown)
        by_module: dict[str, float] = defaultdict(float)
        for cost in found:
            by_module[cost.module] += cost.setup
        for name, seconds in by_module.items():
            setups[name] = max(setups[name], seconds)
    quick = [seconds for seconds in tests.values() if seconds < LISTED]
    return Costs(
        {test: round(seconds, 2) for test, seconds in sorted(tests.items()) if seconds >= LISTED},
        {name: round(seconds, 2) for name, seconds in sorted(setups.items()) if seconds >= LISTED},
        round(sum(quick) / len(quick), 4) if quick else 0.0,
    )


if __name__ == "__main__":
    costs = costs_from([Path(each).read_text(encoding="utf-8-sig") for each in sys.argv[1:]])
    print(json.dumps({"fallback": costs.fallback, "setups": costs.setups, "tests": costs.tests}, indent=1))
