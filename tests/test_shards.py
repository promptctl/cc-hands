"""A run is cut into shards so each can be a CI job inside the job's 5 minutes: the shards of one run together hold every test
once, a module's setup is paid by the shards that hold it, and the costs the packing reads come from the cost reports CI
prints."""

import argparse
from itertools import chain

import pytest

from costs import Cost, row
from shards import COSTS, Costs, Shard, Sharding, costs_from, packed, partition, pieces

TESTS = [f"tests/test_{each}.py::test_{n}" for each in ("a", "b", "c", "d") for n in range(10)]
COSTS_OF_THESE = Costs({"tests/test_a.py::test_0": 40.0, "tests/test_b.py::test_0": 30.0, "tests/test_c.py::test_0": 20.0}, {"tests/test_d.py": 25.0}, 1.0)


@pytest.mark.parametrize("count", [1, 2, 3, 7, 40, 41])
def test_each_test_is_in_exactly_one_shard_whatever_the_count(count: int) -> None:
    shards = partition(TESTS, COSTS_OF_THESE, count)
    assert len(shards) == count
    assert sorted(chain.from_iterable(shards)) == sorted(TESTS)


def test_the_same_tests_and_costs_make_the_same_shards() -> None:
    assert partition(TESTS, COSTS_OF_THESE, 4) == partition(list(TESTS), Costs(dict(COSTS_OF_THESE.tests), dict(COSTS_OF_THESE.setups), 1.0), 4)


def test_the_costly_tests_are_in_shards_of_their_own_while_there_are_shards_to_give() -> None:
    shards = partition(TESTS, COSTS_OF_THESE, 4)
    heavy = {"tests/test_a.py::test_0", "tests/test_b.py::test_0", "tests/test_c.py::test_0"}
    assert all(len(heavy & set(shard)) <= 1 for shard in shards)


def test_a_module_with_a_setup_costs_each_shard_that_holds_one_of_its_tests_that_setup() -> None:
    d = [test for test in TESTS if test.startswith("tests/test_d.py")]
    assert packed(d[:1], COSTS_OF_THESE) == 26.0
    assert packed(d[:3], COSTS_OF_THESE) == 28.0
    assert packed(d[:2], COSTS_OF_THESE) + packed(d[2:4], COSTS_OF_THESE) == 54.0


def test_a_module_with_a_setup_stays_whole_when_a_shard_can_hold_it() -> None:
    d = {test for test in TESTS if test.startswith("tests/test_d.py")}
    shards = partition(TESTS, COSTS_OF_THESE, 2)
    assert sum(1 for shard in shards if d & set(shard)) == 1


def test_a_module_that_is_more_than_a_share_is_cut_across_the_shards_and_each_piece_pays_its_setup() -> None:
    big = [f"tests/test_big.py::test_{n}" for n in range(20)]
    small = ["tests/test_small.py::test_only"]
    costs = Costs({test: 5.0 for test in big}, {"tests/test_big.py": 10.0}, 1.0)
    shards = partition(big + small, costs, 2)
    assert all(set(shard) & set(big) for shard in shards)
    assert max(packed(shard, costs) for shard in shards) < packed(big, costs)


def test_the_shards_of_a_module_whose_setup_is_more_than_a_share_each_cost_about_what_the_others_do() -> None:
    tests = [f"tests/test_m.py::test_{n}" for n in range(40)]
    costs = Costs({test: 0.6 if n < 30 else 5.0 for n, test in enumerate(tests)}, {"tests/test_m.py": 60.0}, 1.0)
    loads = [packed(shard, costs) for shard in partition(tests, costs, 4)]
    assert max(loads) - min(loads) < 1.0


def test_a_module_is_never_cut_into_more_pieces_than_there_are_shards() -> None:
    tests = [f"tests/test_m.py::test_{n}" for n in range(30)]
    costs = Costs({}, {"tests/test_m.py": 60.0}, 1.0)
    assert pieces(tests, costs, packed(tests, costs) / 4, 4) == 4


@pytest.mark.parametrize("text", ["0/3", "4/3", "1/0", "2", "a/b", "1/2/3", "-1/2", ""])
def test_a_shard_is_k_of_n_and_nothing_else(text: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="not a shard"):
        Shard.parse(text)


@pytest.mark.parametrize("index, count", [(3, 2), (2, 2), (-1, 2), (0, 0)])
def test_a_shard_built_by_hand_is_one_of_its_count_too(index: int, count: int) -> None:
    with pytest.raises(ValueError, match="is not one of"):
        Shard(index, count)


def test_a_shard_reads_as_it_was_written() -> None:
    assert str(Shard.parse("2/3")) == "2/3"
    assert Shard.parse("2/3") == Shard(1, 3)


SUITE = """
def test_a(): pass
def test_b(): pass
def test_c(): pass
def test_d(): pass
def test_e(): pass
"""


def test_the_shards_of_a_run_together_run_each_test_once(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_five=SUITE)
    costs = Costs({}, {}, 1.0)
    ran: list[set[str]] = []
    for index in range(2):
        outcome = pytester.runpytest_inprocess("-p", "no:asyncio", "-v", "-p", "no:cacheprovider", plugins=[Sharding(Shard(index, 2), costs)])
        ran.append({line.split("::")[1].split()[0] for line in outcome.outlines if "PASSED" in line})
        outcome.stdout.fnmatch_lines([f"shard {index + 1}/2: packed at *"])
    assert ran[0].isdisjoint(ran[1])
    assert ran[0] | ran[1] == {"test_a", "test_b", "test_c", "test_d", "test_e"}


def test_a_shard_cuts_what_the_marks_chose_not_what_was_collected(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        test_marked="""
import pytest
pytestmark = pytest.mark.slow
def test_a(): pass
def test_b(): pass
""",
        test_unmarked="""
def test_c(): pass
def test_d(): pass
""",
    )
    pytester.makeini("[pytest]\nmarkers = slow")
    costs = Costs({}, {}, 1.0)
    ran: set[str] = set()
    for index in range(2):
        outcome = pytester.runpytest_inprocess("-p", "no:asyncio", "-v", "-p", "no:cacheprovider", "-m", "slow", plugins=[Sharding(Shard(index, 2), costs)])
        # Each of the two selected tests is in a shard of its own: a cut over the four collected would leave one shard none.
        outcome.assert_outcomes(passed=1)
        ran |= {line.split("::")[1].split()[0] for line in outcome.outlines if "PASSED" in line}
    assert ran == {"test_a", "test_b"}


def test_a_shard_says_what_it_was_packed_at_before_its_first_test_and_which_tests_the_file_has_no_cost_for(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_five=SUITE)
    # Every call takes the 2 s the file does not know of: the clock is not read, so no test waits for it.
    pytester.makeconftest(
        """
import pytest

@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    report = yield
    report.duration = 2.0 if report.when == "call" else 0.0
    return report
"""
    )
    outcome = pytester.runpytest_inprocess("-p", "no:asyncio", "-p", "no:cacheprovider", plugins=[Sharding(Shard(0, 1), Costs({"test_five.py::test_a": 2.0}, {}, 1.0))])
    outcome.stdout.fnmatch_lines(["shard 1/1 holds 5 tests, packed at 6.0 s", "*shard 1/1: packed at 6.0 s, cost 10.0 s", "shard 1/1: 4 tests cost 0.5 s or more and are not in tests/shard-costs.json: test_five.py::test_b, *"])


def test_a_shard_that_holds_no_test_says_why_instead_of_running_none(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_five=SUITE)
    outcome = pytester.runpytest_inprocess("-p", "no:asyncio", "-p", "no:cacheprovider", plugins=[Sharding(Shard(5, 6), Costs({}, {}, 1.0))])
    assert outcome.ret == pytest.ExitCode.USAGE_ERROR
    outcome.stderr.fnmatch_lines(["*shard 6/6 holds no test: the 5 tests the run selects are too few for 6 shards"])


def table(*costs: Cost) -> str:
    """A job's log of the cost report, as CI stamps it."""
    return "\n".join(f"2026-11-24T18:25:20.5Z {row(cost)}" for cost in costs) + "\n"


SHARD_ONE = table(
    Cost("tests/test_streaming.py::test_a[real]", 24.041, 26.1, 0.018),
    Cost("tests/test_streaming.py::test_b[x y]", 0.003, 4.99, 0.007),
    Cost("tests/test_other.py::test_quick", 0.3, 0.05, 0.05),
)
SHARD_TWO = table(Cost("tests/test_streaming.py::test_c", 24.5, 8.0, 0.0))


def test_the_cost_file_is_what_a_job_logs_cost_report_says() -> None:
    costs = costs_from({"one": SHARD_ONE})
    assert costs.tests == {"tests/test_streaming.py::test_a[real]": 26.1, "tests/test_streaming.py::test_b[x y]": 5.0}
    assert costs.setups == {"tests/test_streaming.py": 24.0}


def test_a_module_cut_across_shards_costs_the_setup_one_shard_paid_not_the_sum_of_what_all_paid() -> None:
    costs = costs_from({"one": SHARD_ONE, "two": SHARD_TWO})
    assert costs.setups == {"tests/test_streaming.py": 24.5}
    assert costs.tests["tests/test_streaming.py::test_c"] == 8.0


def test_a_test_the_run_stopped_inside_has_no_cost_and_does_not_weigh_on_the_quick_ones() -> None:
    cut_off = table(Cost("tests/test_streaming.py::test_cut_off", 0.6, 0.0, 0.0, stopped=True), Cost("tests/test_streaming.py::test_quick", 0.0, 0.2, 0.0))
    costs = costs_from({"one": cut_off})
    assert "tests/test_streaming.py::test_cut_off" not in costs.tests
    assert costs.setups == {}
    assert costs.fallback == 0.2


def test_a_log_with_no_cost_rows_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="check-pair-log-3.txt holds no cost report rows"):
        costs_from({"one": SHARD_ONE, "check-pair-log-3.txt": "no report log at .pytest_cache/check-pair-log.jsonl: the run wrote none\n"})


def test_the_costs_the_repository_holds_are_the_ones_the_run_reads() -> None:
    held = Costs.load(COSTS)
    assert held.tests and held.setups and held.fallback > 0
