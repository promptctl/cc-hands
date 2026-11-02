"""What each test cost, read from the report log every run of the suite writes (pyproject's addopts): the run, each module,
then each test, slowest first. CI prints this into the job's log, so `gh run view <id> --log` reads any run's, cancelled or
red ones too, and reading it starts nothing.

    uv run python tests/costs.py .pytest_cache/report-log.jsonl
"""

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Cost:
    """One test's seconds by the phase it spent them in. Setup stands apart because a fixture shared by a module is charged
    to the first test that asks for it, and a test the run stopped inside has only the phases it finished."""

    test: str
    setup: float = 0.0
    call: float = 0.0
    teardown: float = 0.0

    @property
    def total(self) -> float:
        return self.setup + self.call + self.teardown

    @property
    def module(self) -> str:
        return self.test.partition("::")[0]


def seconds(costs: list[Cost]) -> float:
    return sum(cost.total for cost in costs)


def report(log: str) -> str:
    """The report of a report log's text, which a cancelled run's pytest may still be writing: only a line its writer
    finished is a record, so a partial last line is one more way for the log to stop short."""
    records: list[dict[str, Any]] = [json.loads(line) for line in log.split("\n")[:-1]]
    phases: dict[str, dict[str, float]] = defaultdict(dict)
    for record in records:
        if record["$report_type"] == "TestReport":
            phases[record["nodeid"]][record["when"]] = record["duration"]
    costs = sorted((Cost(test, **finished) for test, finished in phases.items()), key=lambda cost: cost.total, reverse=True)
    modules: dict[str, list[Cost]] = defaultdict(list)
    for cost in costs:
        modules[cost.module].append(cost)
    status = next((record["exitstatus"] for record in records if record["$report_type"] == "SessionFinish"), None)
    ended = "the log stops before the run ended" if status is None else f"the run ended with exit status {status}"
    lines = [f"{len(costs)} tests, {seconds(costs):.3f} s: {ended}", "", f"{'seconds':>9} {'share':>5} {'tests':>5}  module"]
    for module, tests in sorted(modules.items(), key=lambda item: seconds(item[1]), reverse=True):
        lines.append(f"{seconds(tests):9.3f} {seconds(tests) / seconds(costs):5.0%} {len(tests):5}  {module}")
    lines += ["", f"{'total':>9} {'setup':>9} {'call':>9} {'teardown':>9}  test"]
    lines += [f"{cost.total:9.3f} {cost.setup:9.3f} {cost.call:9.3f} {cost.teardown:9.3f}  {cost.test}" for cost in costs]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print(report(Path(sys.argv[1]).read_text()), end="")
