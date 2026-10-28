"""conftest's hold on CI: a test skipped there fails the run, with the skip's reason, whether its body, its marks or its
module's collection skipped it; anywhere else a skip is a skip, and an expected failure is never one."""

from types import SimpleNamespace

import pytest

import conftest

SUITE = """
import pytest

def test_skipped_in_its_body():
    pytest.skip("no jq on this machine")

@pytest.mark.skipif(True, reason="a policy on this machine")
def test_skipped_by_its_mark():
    pass

@pytest.mark.xfail(strict=True)
def test_expected_to_fail():
    assert False

def test_passed():
    pass
"""

SKIPPED_MODULE = """
import pytest

pytest.skip("no mlx on this machine", allow_module_level=True)

def test_never_collected():
    pass
"""

# The hooks alone, as a plugin of the run, not the whole conftest beside them.
HOOKS = SimpleNamespace(pytest_runtest_makereport=conftest.pytest_runtest_makereport, pytest_make_collect_report=conftest.pytest_make_collect_report)


def test_a_test_skipped_in_ci_fails_the_run_with_its_reason(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CI", "true")
    pytester.makepyfile(SUITE)
    ran = pytester.runpytest_inprocess(plugins=[HOOKS])
    # A mark skips as the test is set up, so its failure is counted as an error.
    ran.assert_outcomes(failed=1, errors=1, xfailed=1, passed=1)
    ran.stdout.fnmatch_lines_random(["*Skipped: no jq on this machine, in CI, where every test runs*", "*Skipped: a policy on this machine, in CI, where every test runs*"])


def test_a_module_skipped_in_ci_fails_the_run_with_its_reason(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CI", "true")
    pytester.makepyfile(SKIPPED_MODULE)
    ran = pytester.runpytest_inprocess(plugins=[HOOKS])
    ran.assert_outcomes(errors=1)
    ran.stdout.fnmatch_lines(["*no mlx on this machine, in CI, where every test runs*"])


def test_a_skip_outside_ci_is_skipped(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CI", raising=False)
    pytester.makepyfile(test_suite=SUITE, test_skipped_module=SKIPPED_MODULE)
    pytester.runpytest_inprocess(plugins=[HOOKS]).assert_outcomes(skipped=3, xfailed=1, passed=1)
