"""conftest's hold on CI: a test skipped there fails the run, with the skip's reason; anywhere else a skip is a skip, and
an expected failure is never one."""

from types import SimpleNamespace

import pytest

import conftest

SUITE = """
import pytest

def test_skipped():
    pytest.skip("no jq on this machine")

@pytest.mark.xfail(strict=True)
def test_expected_to_fail():
    assert False

def test_passed():
    pass
"""


def outcomes(pytester: pytest.Pytester) -> pytest.RunResult:
    pytester.makepyfile(SUITE)
    # The hook alone, as a plugin of the run, not the whole conftest beside it.
    return pytester.runpytest_inprocess(plugins=[SimpleNamespace(pytest_runtest_makereport=conftest.pytest_runtest_makereport)])


def test_a_test_skipped_in_ci_fails_the_run_with_its_reason(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CI", "true")
    ran = outcomes(pytester)
    ran.assert_outcomes(failed=1, xfailed=1, passed=1)
    ran.stdout.fnmatch_lines(["*Skipped: no jq on this machine, in CI, where every test runs*"])


def test_a_test_skipped_outside_ci_is_skipped(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CI", raising=False)
    outcomes(pytester).assert_outcomes(skipped=1, xfailed=1, passed=1)
