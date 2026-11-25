"""Which tests one CI job runs when the suite is cut across jobs: `--shard=K/N` keeps the tests the Kth of N shards holds.

A shard is a job, and a job's time is its tests' seconds plus a fixed cost the tests do not control (workflows' headers), so
the cut is made to give every job the same seconds. The partition is longest-processing-time packing over what each test
cost on CI (tests/shard-costs.json), and it is a pure function of the collected test ids and that file: every job of a commit
computes it alone, and each test runs in exactly one job when the N of them collected the same ids, which each job records a
digest of (`DIGEST`) for test.yml's verdict to hold them to. A stale file loses no test: it unbalances the shards, which each
prints the packed seconds of beside what it cost.

The cost model: a test costs its call and teardown; a module's fixtures cost their setup once to each shard that holds any of
its tests; and a module is cut across shards only when it is more than a share, since a cut pays its setup again.

    uv run python tests/shards.py JOBLOG... > tests/shard-costs.json

JOBLOG is the text of a CI job whose log holds tests/costs.py's report (`gh run view --job <id> --log`), and a log with no row
of it is refused by name; a test costs the most any log gives it, so the shards hold on a slow runner too.
"""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Self

import pytest

from costs import module, rows

COSTS = Path(__file__).with_name("shard-costs.json")
# A test is named in the file from this many seconds; the ones under it cost what the rest of them did on average.
LISTED = 0.05
# A test of this many seconds the file does not name makes the shard's summary say the file is stale.
UNPRICED = 0.5
# The key of pytest's cache a run records the digest of the test ids it collected under: `.pytest_cache/v/shards/collected`.
DIGEST = "shards/collected"


class ShardError(argparse.ArgumentTypeError, ValueError):
    """A `--shard` that is not K/N: an ArgumentTypeError, so argparse prints its text instead of 'invalid parse value'."""


@dataclass(frozen=True)
class Shard:
    index: int
    total: int

    def __post_init__(self) -> None:
        if not 1 <= self.index <= self.total:
            raise ShardError(f"shard {self.index}/{self.total}: K is a number from 1 to N")

    @classmethod
    def parse(cls, text: str) -> Self:
        index, slash, total = text.partition("/")
        if not (slash and index.isdecimal() and total.isdecimal()):
            raise ShardError(f"--shard is K/N, as in 2/4, not {text!r}")
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
        seconds = [data["fallback"], *data["tests"].values(), *data["setups"].values()]
        if not all(isinstance(each, (int, float)) and not isinstance(each, bool) for each in seconds):
            raise ValueError(f"{path}: fallback, and each test's and each module's setup, are seconds, numbers, and nothing else")
        return cls(**data)

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
    share = (sum(costs.of(test) for held in by_module.values() for test in held) + sum(costs.setup(name) for name in by_module)) / total
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
    # None until the collection is cut: a run that stops in collection has packed nothing, and says so by printing no panel.
    packed: Packed | None = None
    digest: str = ""
    spent: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    setups: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, config: pytest.Config, items: list[pytest.Item]) -> None:
        ids = [item.nodeid for item in items]
        self.digest = hashlib.sha256("\n".join(sorted(set(ids))).encode()).hexdigest()
        # [LAW:single-enforcer] "each test is in exactly one shard" holds when every job collected these same ids: the jobs
        # record the digest, and test.yml's verdict holds them to one.
        config.cache.set(DIGEST, self.digest)
        self.packed = packed = partition(ids, self.costs, self.shard.total)[self.shard.index - 1]
        kept = [item for item in items if item.nodeid in packed.tests]
        # [LAW:no-silent-failure] a shard with no test is a job that passes having run nothing.
        if items and not kept:
            raise pytest.UsageError(f"shard {self.shard} holds none of the {len(items)} tests collected: it is a shard of more jobs than there are tests")
        config.hook.pytest_deselected(items=[item for item in items if item.nodeid not in packed.tests])
        items[:] = kept

    def pytest_report_collectionfinish(self, items: Sequence[pytest.Item]) -> str:
        seconds = self.packed.seconds if self.packed else 0.0
        return f"shard {self.shard}: {len(items)} tests, packed at {seconds:.1f} s, of the ids digest {self.digest[:12]}"

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if report.when == "setup":
            self.setups[module(report.nodeid)] += report.duration
        else:
            self.spent[report.nodeid] += report.duration

    def pytest_terminal_summary(self, terminalreporter: pytest.TerminalReporter) -> None:
        """[LAW:nothing-unseen] what the shard cost against what it was packed at, setups and all, and the tests and modules
        that are slow and are not in the file: it is stale, and `uv run python tests/shards.py` writes it from these jobs' logs."""
        if self.packed is None:
            return
        terminalreporter.section(f"shard {self.shard}")
        terminalreporter.line(f"setup, call and teardown: {sum(self.setups.values()) + sum(self.spent.values()):.1f} s, packed at {self.packed.seconds:.1f} s")
        for test in sorted(test for test, seconds in self.spent.items() if seconds >= UNPRICED and test not in self.costs.tests):
            terminalreporter.line(f"cost {self.spent[test]:.1f} s and not in tests/shard-costs.json: {test}")
        for name in sorted(name for name, seconds in self.setups.items() if seconds >= UNPRICED and name not in self.costs.setups):
            terminalreporter.line(f"setup cost {self.setups[name]:.1f} s and not in tests/shard-costs.json: {name}")


def costs_from(logs: Mapping[str, str]) -> Costs:
    """The most any log gives each test, and the most any gives each module's setup. `logs` is each log's text by its name."""
    tests: dict[str, float] = defaultdict(float)
    setups: dict[str, float] = defaultdict(float)
    for name, log in logs.items():
        found = rows(log)
        # [LAW:no-silent-failure] a log with no row is an expired log, a wrong job or a format the reader does not know: from it
        # every test costs the same, and the packing puts them all in one shard.
        if not found:
            raise ValueError(f"{name} holds no row of tests/costs.py's report")
        for cost in found:
            tests[cost.test] = max(tests[cost.test], cost.call + cost.teardown)
        by_module: dict[str, float] = defaultdict(float)
        for cost in found:
            by_module[cost.module] += cost.setup
        for each, seconds in by_module.items():
            setups[each] = max(setups[each], seconds)
    quick = [seconds for seconds in tests.values() if seconds < LISTED]
    return Costs(
        {test: round(seconds, 2) for test, seconds in sorted(tests.items()) if seconds >= LISTED},
        {name: round(seconds, 2) for name, seconds in sorted(setups.items()) if seconds >= LISTED},
        round(sum(quick) / len(quick), 4) if quick else 0.0,
    )


if __name__ == "__main__":
    costs = costs_from({each: Path(each).read_text(encoding="utf-8-sig") for each in sys.argv[1:]})
    print(json.dumps(asdict(costs), indent=1, sort_keys=True))
