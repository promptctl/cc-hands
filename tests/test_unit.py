"""The unit layer is what passes with the outside refused by the OS (tests/offline.py, tests/unit.sb): in a `--layer` run a
test cannot start a process, bind a socket, or read the weights the run fetched, and in any other run it can. These are
unit tests, so they run in the layer's own hold and prove it from inside it, since a held process cannot start another
held one; and they pass in the full run, where the same calls are allowed."""

import shutil
import socket
import subprocess

import pytest

from conftest import RUN, layered


def held(config: pytest.Config) -> bool:
    return config.getoption("layer") is not None


def test_a_process_start_is_refused_in_a_layer_run_and_made_in_any_other(pytestconfig: pytest.Config) -> None:
    if held(pytestconfig):
        with pytest.raises(PermissionError):
            subprocess.run(["/usr/bin/true"], check=False)
    else:
        assert subprocess.run(["/usr/bin/true"], check=False).returncode == 0


def test_a_socket_bound_on_loopback_is_refused_in_a_layer_run_and_bound_in_any_other(pytestconfig: pytest.Config) -> None:
    if held(pytestconfig):
        with pytest.raises(PermissionError):
            socket.create_server(("127.0.0.1", 0))
    else:
        socket.create_server(("127.0.0.1", 0)).close()


def test_the_fetched_weights_are_unread_in_a_layer_run_and_read_in_any_other(pytestconfig: pytest.Config) -> None:
    weights = pytestconfig.stash[RUN].wake_word
    if held(pytestconfig):
        with pytest.raises(PermissionError):
            weights.stat()
    else:
        assert weights.is_dir()


def test_a_look_at_a_tool_finds_it_where_a_process_cannot_be_started(pytestconfig: pytest.Config) -> None:
    assert shutil.which("true") is not None


# What each outside reach does as a test, in a session of its own run in process, so it is held as this run is: the
# reach, then its verdict in a layer run and in any other.
REACHES = {
    "starts_a_process": "import subprocess\n\ndef test_it():\n    subprocess.run(['/usr/bin/true'], check=False)\n",
    "binds_a_socket": "import socket\n\ndef test_it():\n    socket.create_server(('127.0.0.1', 0)).close()\n",
    "reads_the_weights": "from pathlib import Path\n\ndef test_it():\n    Path({weights!r}).stat()\n",
}


@pytest.mark.parametrize("reach", REACHES)
def test_a_test_that_reaches_outside_fails_in_a_layer_run_and_passes_in_any_other(reach: str, pytester: pytest.Pytester, pytestconfig: pytest.Config) -> None:
    pytester.makepyfile(test_inner_reach=REACHES[reach].format(weights=str(pytestconfig.stash[RUN].wake_word)))
    outcome = pytester.runpytest_inprocess("-p", "no:asyncio", "-p", "no:cacheprovider").parseoutcomes()
    assert outcome.get("failed" if held(pytestconfig) else "passed") == 1, outcome


def test_a_fixture_that_reaches_outside_fails_every_test_that_uses_it_whichever_is_first(pytester: pytest.Pytester, pytestconfig: pytest.Config) -> None:
    pytester.makepyfile(
        test_inner_shared="import subprocess\nimport pytest\n\n@pytest.fixture(scope='module')\ndef party():\n    subprocess.run(['/usr/bin/true'], check=False)\n\ndef test_first(party):\n    pass\n\ndef test_second(party):\n    pass\n\ndef test_third():\n    pass\n"
    )
    outcome = pytester.runpytest_inprocess("-p", "no:asyncio", "-p", "no:cacheprovider").parseoutcomes()
    if held(pytestconfig):
        assert (outcome.get("errors"), outcome.get("passed")) == (2, 1), outcome
    else:
        assert outcome.get("passed") == 3, outcome


def test_a_layer_run_takes_the_tests_no_mark_sends_outside_or_the_ones_that_it_does(pytester: pytest.Pytester) -> None:
    items = pytester.getitems("import pytest\n\ndef test_in():\n    pass\n\n@pytest.mark.outside\ndef test_out():\n    pass\n")
    unit, audit = list(items), list(items)
    layered(items[0].config, unit, "unit")
    layered(items[0].config, audit, "audit")
    assert [item.name for item in unit] == ["test_in"]
    assert [item.name for item in audit] == ["test_out"]
    (strict,) = audit[0].iter_markers("xfail")
    assert strict.kwargs["strict"] is True
