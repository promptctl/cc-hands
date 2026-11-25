"""tests/shards.py cuts the suite across CI's jobs: every test in exactly one shard, the shards equal in packed seconds, and the same
cut in every job, since each job computes it alone."""

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from collections.abc import Generator
import pytest

from costs import report
from shards import COSTS, DIGEST, Costs, Shard, ShardError, Sharding, costs_from, partition
from reportlogs import FINISHED, STARTED, spent


def numbered(module: str, count: int) -> list[str]:
    return [f"tests/{module}.py::test_{n}" for n in range(count)]


def cheap(**tests: float) -> Costs:
    return Costs({f"tests/{name.replace('__', '.py::')}": seconds for name, seconds in tests.items()}, {}, 0.01)


@pytest.mark.parametrize("text, shard", [("1/1", Shard(1, 1)), ("2/4", Shard(2, 4)), ("4/4", Shard(4, 4))])
def test_a_shard_is_k_of_n(text: str, shard: Shard) -> None:
    assert Shard.parse(text) == shard
    assert str(shard) == text


@pytest.mark.parametrize("text", ["", "2", "2/", "/4", "0/4", "5/4", "a/b", "-1/4", "1/0", "1/4/4"])
def test_anything_else_is_no_shard(text: str) -> None:
    with pytest.raises(ShardError):
        Shard.parse(text)


@pytest.mark.parametrize("total", [1, 2, 3, 4, 7, 50])
def test_every_test_is_in_exactly_one_shard_for_any_count_of_shards(total: int) -> None:
    tests = numbered("a", 10) + numbered("b", 3) + numbered("c", 1) + numbered("d", 30)
    costs = Costs({"tests/a.py::test_0": 9.0, "tests/d.py::test_7": 4.0, "tests/b.py::test_1": 2.5}, {"tests/a.py": 1.0, "tests/d.py": 0.5}, 0.01)
    held = Counter(test for packed in partition(tests, costs, total) for test in packed.tests)
    assert held == Counter(tests)
    assert len(partition(tests, costs, total)) == total


def test_the_cut_is_the_same_whatever_order_the_tests_are_collected_in() -> None:
    tests = numbered("a", 10) + numbered("b", 10) + numbered("c", 10)
    costs = Costs({"tests/a.py::test_3": 5.0, "tests/b.py::test_4": 4.0}, {}, 0.5)
    assert partition(tests, costs, 4) == partition(list(reversed(tests)), costs, 4)


def test_modules_that_fit_a_share_are_whole_and_the_shards_equal() -> None:
    costs = cheap(a__t=3, b__t=3, c__t=3, d__t=3)
    packed = partition(["tests/a.py::t", "tests/b.py::t", "tests/c.py::t", "tests/d.py::t"], costs, 2)
    assert [each.seconds for each in packed] == [6, 6]


def test_a_module_more_than_a_share_is_cut_across_the_shards() -> None:
    tests = ["tests/slow.py::test_a", "tests/slow.py::test_b", "tests/quick.py::test_c"]
    packed = partition(tests, cheap(slow__test_a=30, slow__test_b=22, quick__test_c=1), 2)
    assert sorted(each.seconds for each in packed) == [23, 30]
    assert {frozenset(each.tests) for each in packed} == {frozenset({"tests/slow.py::test_a"}), frozenset({"tests/slow.py::test_b", "tests/quick.py::test_c"})}


def test_a_modules_setup_is_charged_once_to_each_shard_that_holds_any_of_it() -> None:
    costs = Costs({"tests/m.py::test_a": 10.0, "tests/m.py::test_b": 10.0}, {"tests/m.py": 4.0}, 0.0)
    tests = ["tests/m.py::test_a", "tests/m.py::test_b"]
    assert [each.seconds for each in partition(tests, costs, 1)] == [24.0]
    assert [each.seconds for each in partition(tests, costs, 2)] == [14.0, 14.0]


def test_a_test_the_file_does_not_name_costs_the_fallback() -> None:
    assert partition(["tests/a.py::t"], Costs({}, {}, 0.25), 1)[0].seconds == 0.25


def test_costs_come_from_the_most_any_log_gives_each_test_and_each_modules_setup() -> None:
    fast = report(STARTED + spent("tests/a.py::test_slow", 1, 5, 1) + spent("tests/a.py::test_quick", 1, 0.01, 0) + spent("tests/b.py::test_x", 0, 0.01, 0) + FINISHED)
    slow = report(STARTED + spent("tests/a.py::test_slow", 3, 9, 1) + spent("tests/a.py::test_quick", 1, 0.02, 0) + FINISHED)
    costs = costs_from({"fast": fast, "slow": slow})
    assert costs.tests == {"tests/a.py::test_slow": 10.0}
    assert costs.setups == {"tests/a.py": 4.0}
    assert costs.fallback == pytest.approx(0.015)


def test_a_log_with_no_row_of_the_report_is_refused_by_name_and_not_made_a_cost_file() -> None:
    with pytest.raises(ValueError, match="expired.log holds no row"):
        costs_from({"expired.log": ""})


def test_no_log_at_all_is_refused_and_not_made_a_cost_file() -> None:
    with pytest.raises(ValueError, match="no log was given"):
        costs_from({})


@pytest.mark.parametrize(
    "text",
    [
        '{"fallback": null, "setups": {}, "tests": {}}',
        '{"fallback": 0, "setups": {}, "tests": {"tests/a.py::t": "1.5"}}',
        '{"fallback": 0, "setups": {"tests/a.py": true}, "tests": {}}',
        '{"fallback": 0, "setups": {}, "tests": {"tests/a.py::t": -1}}',
        '{"fallback": NaN, "setups": {}, "tests": {}}',
        '{"fallback": 0, "setups": {}}',
        '{"fallback": 0, "setups": {}, "tests": {}, "extra": 1}',
        '{"fallback": 0, "setups": [], "tests": []}',
        "<<<<<<< HEAD",
    ],
)
def test_a_cost_file_that_is_not_the_costs_in_seconds_is_refused_naming_the_file(text: str, tmp_path: Path) -> None:
    path = tmp_path / "costs.json"
    path.write_text(text)
    with pytest.raises(ValueError, match="costs.json"):
        Costs.load(path)


def test_the_committed_cost_file_is_the_shape_the_cost_reader_takes() -> None:
    data = json.loads(COSTS.read_text())
    assert set(data) == {"fallback", "setups", "tests"}
    assert Costs.load(COSTS).fallback == data["fallback"]


SUITE = """
def test_{n}():
    pass
"""


def shard_run(pytester: pytest.Pytester, shard: Shard, costs: Costs, *args: str) -> pytest.RunResult:
    for module in ("a", "b", "c"):
        pytester.makepyfile(**{f"test_{module}": "\n".join(SUITE.format(n=n) for n in range(4))})
    return pytester.runpytest_inprocess("-v", "-p", "no:randomly", *args, plugins=[Sharding(shard, costs)])


def test_a_shards_run_keeps_its_tests_and_says_what_it_was_packed_at(pytester: pytest.Pytester) -> None:
    costs = Costs({}, {}, 1.0)
    ran: list[str] = []
    for index in (1, 2, 3):
        result = shard_run(pytester, Shard(index, 3), costs)
        result.assert_outcomes(passed=4, deselected=8)
        result.stdout.fnmatch_lines([f"shard {index}/3: 4 tests, packed at 4.0 s, of the ids digest *"])
        ran += [line.split()[0] for line in result.outlines if " PASSED" in line]
    assert sorted(ran) == sorted(f"test_{module}.py::test_{n}" for module in "abc" for n in range(4))


def test_the_shard_comes_after_the_tests_a_selection_drops(pytester: pytest.Pytester) -> None:
    result = shard_run(pytester, Shard(1, 2), Costs({}, {}, 1.0), "-k", "test_0 or test_1")
    # Six tests are selected, as three modules of two: the shard's two modules, not the eight of a cut of all twelve.
    result.stdout.fnmatch_lines(["shard 1/2: 4 tests, packed at 4.0 s, of the ids digest *"])
    result.assert_outcomes(passed=4, deselected=8)


def test_a_shard_that_would_run_no_test_stops_the_run_naming_it(pytester: pytest.Pytester) -> None:
    result = shard_run(pytester, Shard(3, 3), Costs({}, {}, 1.0), "-k", "test_a and test_0")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*shard 3/3 holds none of the 1 tests collected*"])


class Takes:
    """A plugin that gives each call the seconds a run would have measured, so the suite under test need not spend them."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_makereport(self, item: pytest.Item, call: pytest.CallInfo[None]) -> Generator[None, pytest.TestReport, pytest.TestReport]:
        report = yield
        if call.when == "call":
            report.duration = self.seconds
        return report


def test_a_slow_test_the_file_does_not_name_is_said_to_be_missing_from_it(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_slow="def test_slept():\n    pass\n")
    result = pytester.runpytest_inprocess("-p", "no:randomly", plugins=[Takes(0.6), Sharding(Shard(1, 1), Costs({}, {}, 0.0))])
    result.stdout.fnmatch_lines(["*cost 0.6 s and not in tests/shard-costs.json: test_slow.py::test_slept"])


def test_a_slow_module_setup_the_file_does_not_name_is_said_to_be_missing_from_it(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_fixture="import pytest\n\n@pytest.fixture\ndef slow():\n    pass\n\ndef test_uses(slow):\n    pass\n")
    pytester.makeconftest("import pytest\n\n@pytest.hookimpl(wrapper=True)\ndef pytest_runtest_makereport(call):\n    report = yield\n    if call.when == 'setup':\n        report.duration = 0.7\n    return report\n")
    result = pytester.runpytest_inprocess("-p", "no:randomly", plugins=[Sharding(Shard(1, 1), Costs({}, {}, 0.0))])
    result.stdout.fnmatch_lines(["*setup, call and teardown: 0.7 s, packed at 0.0 s", "*setup cost 0.7 s and not in tests/shard-costs.json: test_fixture.py"])


def test_every_shard_of_one_suite_records_the_same_digest_and_another_suite_another(pytester: pytest.Pytester) -> None:
    digests: list[str] = []
    for index in (1, 2, 3):
        result = shard_run(pytester, Shard(index, 3), Costs({}, {}, 1.0))
        assert result.ret == pytest.ExitCode.OK
        record = json.loads((pytester.path / ".pytest_cache" / "v" / DIGEST).read_text())
        assert (record["shard"], record["total"]) == (index, 3)
        digests.append(record["digest"])
    assert len(set(digests)) == 1
    shard_run(pytester, Shard(1, 3), Costs({}, {}, 1.0), "-k", "test_0")
    assert json.loads((pytester.path / ".pytest_cache" / "v" / DIGEST).read_text())["digest"] not in digests


def test_a_shard_never_cut_says_so_and_not_that_it_packed_nothing() -> None:
    sharding = Sharding(Shard(1, 4), Costs({}, {}, 0.0))
    assert sharding.pytest_report_collectionfinish([]) == "shard 1/4: not cut, the collection stopped before its tests were chosen"


@pytest.mark.outside
def test_a_shard_of_a_run_that_is_not_of_the_unit_layer_is_refused_naming_the_layer() -> None:
    ran = subprocess.run([sys.executable, "-m", "pytest", "--shard=2/4", "--collect-only", "-q", "tests/test_agreement.py"], cwd=Path(__file__).parents[1], capture_output=True, text=True)
    assert ran.returncode == pytest.ExitCode.USAGE_ERROR
    assert "run it with --layer=unit" in ran.stderr
