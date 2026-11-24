"""A run is cut into shards so each can be a CI job inside the job's 5 minutes: the shards of one run together hold every test
once, a module's setup is paid by the shards that hold it, and the costs the packing reads come from the cost reports CI
prints."""

import argparse
from itertools import chain

import pytest

from shards import COSTS, Costs, Shard, Sharding, costs_from, packed, partition

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


@pytest.mark.parametrize("text", ["0/3", "4/3", "1/0", "2", "a/b", "1/2/3", "-1/2", ""])
def test_a_shard_is_k_of_n_and_nothing_else(text: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="not a shard"):
        Shard.parse(text)


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


TABLE = """\
2026-11-24T18:25:20.5Z 906 tests, 1525.780 s: the run ended with exit status 0
2026-11-24T18:25:20.5Z
2026-11-24T18:25:20.5Z     total     setup      call  teardown  test
2026-11-24T18:25:20.5Z   50.165    24.041    26.106     0.018  tests/test_streaming.py::test_a[real]
2026-11-24T18:25:20.5Z    5.000     0.003     4.990     0.007  tests/test_streaming.py::test_b[x y]
2026-11-24T18:25:20.5Z    0.400     0.300     0.050     0.050  tests/test_other.py::test_quick
"""


def test_the_cost_file_is_what_a_job_logs_cost_report_says() -> None:
    costs = costs_from([TABLE])
    assert costs.tests == {"tests/test_streaming.py::test_a[real]": 26.1, "tests/test_streaming.py::test_b[x y]": 5.0}
    assert costs.setups == {"tests/test_streaming.py": 24.0}


def test_the_costs_the_repository_holds_are_the_ones_the_run_reads() -> None:
    held = Costs.load(COSTS)
    assert held.tests and held.setups and held.fallback > 0
