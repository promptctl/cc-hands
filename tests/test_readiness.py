"""`hands check`: each piece hands needs, found or named as missing, through the same edges a user's machine has."""

import errno
import json
import os
import shutil
import socket
import tempfile
import sys
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import pytest

from conftest import answering, children_on, executable, laid_out, onboard, pin, shell_without_hands_settings, SYSTEM_PATH, Unstarting
from fakeprocesses import PROC_PID, FakeProcesses, Running, processes
from processescontract import PROCARGS2
from fixtures.pins import LAID_OUT
from hands.clock import SYSTEM
from hands.virtualclock import VirtualClock
from hands.core.session import Membership, SessionId
from hands.core.tmux import Behind, Keyboard, NotInTmux, Pane, PaneUnread
from hands.daemon import readiness
from hands.daemon.cli import VERBOSE_HINT, main
from hands.daemon.readiness import Missing, Ready, Unknown
from hands.sessions import audit, child, heartbeat, liveness, wrapper
from hands.sessions.hookconfig import PLUGIN_ID
from hands.sessions.home import Home
from hands.sessions.membership import write_membership
from hands.sessions.payload import Rejected
from hands.sessions.processes import TerminalProcess, TerminalProcesses, Undescribed, terminal_processes
from hands.sessions.wrapper import shim_script
from hands.core.session import PinnedClaude, Runnable
from hands.voice.backends import Account, ClaudeCodeBackend


@pytest.fixture
def root() -> Iterator[Path]:
    # Short on purpose: a test session's fritter socket lives under here, and macOS caps a socket path near 104 bytes.
    root = Path(tempfile.mkdtemp(prefix="ready-", dir="/tmp")).resolve()
    yield root
    shutil.rmtree(root)


def claude_listing(root: Path, plugins: object) -> str:
    """A PATH whose `claude plugin list --json` prints plugins."""
    executable(root / "real" / "claude", f"#!/bin/sh\ncat <<'EOF'\n{json.dumps(plugins)}\nEOF\n")
    return f"{root / 'real'}:{SYSTEM_PATH}"


def listed(id: str, enabled: bool, scope: str = "user") -> dict[str, object]:
    return {"id": id, "version": "f24447053e9c", "scope": scope, "enabled": enabled}


# The plugin


@pytest.mark.outside
def test_an_enabled_plugin_is_ready(root: Path) -> None:
    found = readiness.plugin(children_on(SYSTEM), Home(root / "home"), claude_listing(root, [listed("other@elsewhere", False), listed(PLUGIN_ID, True)]))
    assert isinstance(found, Ready) and "/reload-plugins" in found.said


def test_a_plugin_enabled_at_any_scope_is_ready() -> None:
    assert isinstance(readiness.plugin_listed(json.dumps([listed(PLUGIN_ID, False), listed(PLUGIN_ID, True)])), Ready)


def test_a_disabled_plugin_is_missing_and_says_how_to_enable_it() -> None:
    found = readiness.plugin_listed(json.dumps([listed(PLUGIN_ID, False)]))
    assert isinstance(found, Missing) and f"claude plugin enable --scope user {PLUGIN_ID}" in found.said


@pytest.mark.outside
def test_a_plugin_not_installed_is_missing_and_says_how_to_install_it(root: Path) -> None:
    found = readiness.plugin(children_on(SYSTEM), Home(root / "home"), claude_listing(root, [listed("other@elsewhere", True)]))
    assert isinstance(found, Missing) and f"claude plugin install {PLUGIN_ID}" in found.said


def test_an_install_for_one_project_is_not_one_for_every_session() -> None:
    project = {**listed(PLUGIN_ID, True, "local"), "projectPath": "/code/cc-hands"}
    found = readiness.plugin_listed(json.dumps([project]))
    assert isinstance(found, Missing) and f"`claude plugin install {PLUGIN_ID}`" in found.said


def test_another_plugin_listed_unreadably_says_nothing_of_hands() -> None:
    assert isinstance(readiness.plugin_listed(json.dumps([{"broken": True}, listed(PLUGIN_ID, True)])), Ready)


@pytest.mark.parametrize(
    "printed", ["not json", '{"plugins": []}', '[{"id": "hands@cc-hands", "scope": "user"}]'], ids=["text", "object", "no-enabled"]
)
def test_a_listing_hands_cannot_read_is_unknown_not_missing(printed: str) -> None:
    assert isinstance(readiness.plugin_listed(printed), Unknown)


@pytest.mark.outside
def test_a_claude_that_fails_to_list_is_unknown_and_says_why(root: Path) -> None:
    executable(root / "real" / "claude", "#!/bin/sh\necho 'not logged in' >&2\nexit 3\n")
    found = readiness.plugin(children_on(SYSTEM), Home(root / "home"), f"{root / 'real'}:{SYSTEM_PATH}")
    assert isinstance(found, Unknown) and "exited 3" in found.said and "not logged in" in found.said


def test_a_listing_that_is_not_text_is_unknown(root: Path) -> None:
    executable(root / "real" / "claude", "#!/bin/sh\nprintf '\\377\\376'\n")
    assert isinstance(readiness.plugin(children_on(SYSTEM), Home(root / "home"), f"{root / 'real'}:{SYSTEM_PATH}"), Unknown)


@pytest.mark.outside
def test_the_claude_asked_for_its_plugins_runs_with_its_auto_updater_off(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISABLE_AUTOUPDATER", "0")
    listing = json.dumps([listed(PLUGIN_ID, True)])
    executable(root / "real" / "claude", f"#!/bin/sh\n[ \"$DISABLE_AUTOUPDATER\" = 1 ] || {{ echo \"updater=$DISABLE_AUTOUPDATER\" >&2; exit 3; }}\ncat <<'EOF'\n{listing}\nEOF\n")
    assert isinstance(readiness.plugin(children_on(SYSTEM), Home(root / "home"), f"{root / 'real'}:{SYSTEM_PATH}"), Ready)


def test_no_claude_to_ask_is_unknown(root: Path) -> None:
    assert isinstance(readiness.plugin(children_on(SYSTEM), Home(root / "home"), f"{root / 'empty'}:{SYSTEM_PATH}"), Unknown)


@pytest.mark.outside
def test_the_plugin_is_asked_of_the_shim_as_a_session_would_ask_it(root: Path) -> None:
    # Off a terminal the shim runs the real claude, so the answer is the real one, not fritter's.
    executable(root / "bin" / "fritter", "#!/bin/sh\necho fritter ran >&2\nexit 9\n")
    executable(root / "bin" / "claude", shim_script(root / "bin" / "fritter", root / "wire.sock"))
    path = claude_listing(root, [listed(PLUGIN_ID, True)])
    assert isinstance(readiness.plugin(children_on(SYSTEM), Home(root / "home"), f"{root / 'bin'}:{path}"), Ready)


# The shim


def test_the_shim_first_on_path_is_ready(root: Path, fritter: Path) -> None:
    home = Home(root / "home")
    home.bin.mkdir(parents=True)
    shutil.copy2(fritter, home.bin / "fritter")
    executable(home.shim, shim_script(home.bin / "fritter", home.wire))
    executable(root / "real" / "claude", "#!/bin/sh\n")
    assert isinstance(readiness.shim(home, f"{home.bin}:{root / 'real'}"), Ready)


def test_an_installed_shim_behind_the_real_claude_wants_only_the_path(root: Path) -> None:
    home = Home(root / "home")
    executable(home.shim, shim_script(home.bin / "fritter", home.wire))
    executable(root / "real" / "claude", "#!/bin/sh\n")
    found = readiness.shim(home, f"{root / 'real'}:{home.bin}")
    assert isinstance(found, Missing)
    assert f"`claude` on this PATH is {root / 'real' / 'claude'}" in found.said
    assert f'export PATH="{home.bin}:$PATH"' in found.said and "install-fritter" not in found.said


def test_a_shim_whose_fritter_is_gone_is_missing(root: Path) -> None:
    home = Home(root / "home")
    executable(home.shim, shim_script(home.bin / "fritter", home.wire))
    found = readiness.shim(home, f"{home.bin}")
    assert isinstance(found, Missing) and f"its fritter {home.bin / 'fritter'} is not there to run" in found.said


def test_a_shim_whose_fritter_is_not_the_one_hands_carries_says_to_install_it_again(root: Path) -> None:
    # As after hands is upgraded: the home's copy is the old hands' fritter.
    home = Home(root / "home")
    executable(home.bin / "fritter", "#!/bin/sh\n")
    executable(home.shim, shim_script(home.bin / "fritter", home.wire))
    found = readiness.shim(home, f"{home.bin}")
    assert isinstance(found, Missing)
    assert f"its fritter {home.bin / 'fritter'} is not the one this hands carries, {wrapper.PACKAGED}" in found.detail and found.fix == "`hands install-fritter`"


@pytest.mark.parametrize("installed", [True, False], ids=["shim", "none"])
def test_a_hands_built_without_its_fritter_names_the_rebuild_not_install_fritter(root: Path, monkeypatch: pytest.MonkeyPatch, installed: bool) -> None:
    home = Home(root / "home")
    if installed:
        executable(home.bin / "fritter", "#!/bin/sh\n")
        executable(home.shim, shim_script(home.bin / "fritter", home.wire))
    monkeypatch.setattr(wrapper, "PACKAGED", root / "package" / "bin" / "fritter")
    found = readiness.shim(home, f"{home.bin}")
    assert isinstance(found, Missing) and found.fix.startswith(f"hands' package carries no fritter at {root / 'package' / 'bin' / 'fritter'}: install hands again")


def runs_subcommands_as_sessions(text: str) -> str:
    return text.replace("session=no ;;", ";;")


def before_the_wire(text: str) -> str:
    return "\n".join(text.splitlines()[:3])


@pytest.mark.parametrize("older", [runs_subcommands_as_sessions, before_the_wire])
def test_a_shim_an_older_hands_wrote_says_to_install_it_again(root: Path, fritter: Path, older: Callable[[str], str]) -> None:
    # As after hands is upgraded: the shim is not the script this hands writes, so it may not run what `run` says it does;
    # the second is one written before the wire, with no wire line.
    home = Home(root / "home")
    home.bin.mkdir(parents=True)
    shutil.copy2(fritter, home.bin / "fritter")
    executable(home.shim, older(shim_script(home.bin / "fritter", home.wire)))
    found = readiness.shim(home, f"{home.bin}")
    assert found == Missing("claude shim out of date", f"`claude` on this PATH is hands' shim, {home.shim}, but not the one this hands writes", "`hands install-fritter`")


def test_another_homes_current_shim_is_ready_by_the_wire_it_names(root: Path, fritter: Path) -> None:
    # As a probe's temporary home checked while ~/.hands/bin is first on PATH.
    home, other = Home(root / "home"), Home(root / "other")
    other.bin.mkdir(parents=True)
    shutil.copy2(fritter, other.bin / "fritter")
    executable(other.shim, shim_script(other.bin / "fritter", other.wire))
    assert isinstance(readiness.shim(home, f"{other.bin}"), Ready)


def test_no_shim_installed_says_to_install_it(root: Path) -> None:
    found = readiness.shim(Home(root / "home"), f"{root / 'empty'}")
    assert isinstance(found, Missing)
    assert "`claude` on this PATH is nothing" in found.said and "hands install-fritter" in found.said


# The grant


def test_the_grant_is_ready_when_given_and_missing_with_where_to_give_it_when_not() -> None:
    assert isinstance(readiness.grant(True), Ready)
    missing = readiness.grant(False)
    assert isinstance(missing, Missing) and "Privacy & Security > Input Monitoring" in missing.said


# The running sessions


def untmuxed(pids: Sequence[int]) -> list[Keyboard]:
    """No pane in front of any process: the check never reads the tmux the tests may run inside."""
    return [NotInTmux()] * len(pids)


PANE = Pane(Path("/tmp/tmux-501/default"), "%12", "work", 3)


def joined(home: Home, name: str, pid: int, fritter: Path | None) -> Membership:
    membership = Membership(SessionId(f"0f1e2d3c-aaaa-bbbb-cccc-{name:0>12}"), pid, Path(f"/code/{name}"), Path("/nowhere/t.jsonl"), fritter)
    write_membership(home, membership)
    return membership


def installed(root: Path) -> str:
    """A PATH whose real `claude` is a link into an install directory of versions, as Claude Code installs itself."""
    version = root / "install" / "9.9.9"
    if not version.exists():
        native(version)
        (root / "bin").mkdir()
        (root / "bin" / "claude").symlink_to(version)
    return f"{root / 'bin'}:{SYSTEM_PATH}"


def native(at: Path) -> Path:
    """A file at `at` that is no script, as a native Claude Code's executable is not: nothing here runs it."""
    executable(at, "\x7fELF")
    return at


# This user's Claude Code config, which every session here runs under, and the check with it.
CLAUDE_CONFIG = "/Users/me/.claude"


@pytest.fixture(autouse=True)
def configured_claude(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", CLAUDE_CONFIG)


# The terminal each session here is started on.
TTY = 0x10000003


def at_a_terminal(pid: int, executable: Path, cwd: Path, arguments: tuple[str, ...] = (), piped: bool = False, start: float = 0.0) -> Running:
    """A process started as executable in cwd on a terminal of its own, as a session runs, its stdin a pipe if piped, as
    a shell starts a program: with the directory it starts in as its PWD."""
    environment = {"CLAUDE_CONFIG_DIR": CLAUDE_CONFIG, "PWD": str(cwd)}
    return Running(pid, tty=TTY, in_front=True, start=start, cwd=cwd, path=executable, arguments=arguments, environment=environment, stdin=TTY, stdout=None if piped else TTY)


def test_no_running_session_is_said_as_none_not_left_out(root: Path) -> None:
    assert readiness.sessions(processes(), Home(root / "home"), installed(root), untmuxed) == Ready("no running sessions", "running sessions hands knows of: 0, runs of claude at a terminal that are none: piped 0, subcommand 0, print 0")


@pytest.mark.outside
def test_each_running_session_that_cannot_be_typed_into_is_named_with_why(root: Path) -> None:
    home = Home(root / "home")
    listening = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listening.bind(str(root / "live.sock"))
    listening.listen()
    # One process each, since one process holds one session.
    wrapped, bare, orphaned = 4001, 4002, 4003
    table = processes(Running(wrapped), Running(bare), Running(orphaned))
    try:
        joined(home, "wrapped", wrapped, root / "live.sock")
        joined(home, "unwrapped", bare, None)
        joined(home, "orphaned", orphaned, root / "gone.sock")
        found = readiness.sessions(table, home, installed(root), untmuxed)
    finally:
        listening.close()
    assert isinstance(found, Missing)
    assert found.detail.startswith("running sessions hands knows of: 3, runs of claude at a terminal that are none: piped 0, subcommand 0, print 0\n  hands cannot reach these:")
    assert found.step == "2 sessions unreachable"
    assert f"/code/orphaned (pid {orphaned}) has lost its fritter, whose socket {root / 'gone.sock'} is gone" in found.said
    assert f"/code/unwrapped (pid {bare}) was started outside fritter and runs in no tmux pane, so it cannot be typed into: restart it" in found.said
    assert found.said.endswith(f"\n  these are typed into:\n    /code/wrapped (pid {wrapped}) is typed into through its fritter")


def test_a_session_started_outside_fritter_in_front_of_its_tmux_pane_is_typed_into_through_it(root: Path) -> None:
    home = Home(root / "home")
    bare = 4001
    asked: list[Sequence[int]] = []

    def in_front(pids: Sequence[int]) -> list[Keyboard]:
        asked.append(pids)
        return [PANE] * len(pids)

    joined(home, "unwrapped", bare, None)
    found = readiness.sessions(processes(Running(bare)), home, installed(root), in_front)
    assert asked == [[bare]]
    assert found == Ready("1 running session reachable", f"running sessions hands knows of: 1, runs of claude at a terminal that are none: piped 0, subcommand 0, print 0\n  these are typed into:\n    /code/unwrapped (pid {bare}) is typed into through its tmux pane %12")


def unwrapped(pane: Keyboard) -> readiness.Finding:
    member = Membership(SessionId("0f1e2d3c-aaaa-bbbb-cccc-000000000007"), 7, Path("/code/bare"), Path("/nowhere/t.jsonl"))
    return readiness.sessions_found([member], set(), [], readiness.Unrecorded([], Counter(), []), {7: pane})


def test_a_session_started_outside_fritter_behind_another_program_in_its_pane_is_missing_with_the_way_back() -> None:
    assert unwrapped(Behind(PANE)) == Missing(
        "1 session unreachable",
        "running sessions hands knows of: 1, runs of claude at a terminal that are none: piped 0, subcommand 0, print 0\n  hands cannot reach these:\n"
        "    /code/bare (pid 7) was started outside fritter, and another program has the keyboard of the tmux pane it runs in, %12 (window 3 of work), "
        "so it cannot be typed into: bring it back to the front of its pane, or restart it from a PATH whose `claude` is hands' shim",
        "bring it back to the front of its pane, or restart it from a PATH whose `claude` is hands' shim",
    )


def test_a_session_started_outside_fritter_whose_pane_could_not_be_read_is_unknown_and_says_why() -> None:
    assert unwrapped(PaneUnread("tmux at /tmp/tmux-501/default did not answer list-panes in 5 seconds")) == Unknown(
        "1 session unknown",
        "running sessions hands knows of: 1, runs of claude at a terminal that are none: piped 0, subcommand 0, print 0\n  whether hands can reach these is unknown:\n"
        "    /code/bare (pid 7) was started outside fritter, and which tmux pane it runs in could not be read, so whether it can be typed into is unknown: "
        "tmux at /tmp/tmux-501/default did not answer list-panes in 5 seconds"
    )


def test_a_session_whose_reach_is_unknown_is_not_said_among_those_hands_cannot_reach() -> None:
    lost = Membership(SessionId("0f1e2d3c-aaaa-bbbb-cccc-000000000008"), 8, Path("/code/lost"), Path("/nowhere/t.jsonl"), Path("/gone.sock"))
    bare = Membership(SessionId("0f1e2d3c-aaaa-bbbb-cccc-000000000007"), 7, Path("/code/bare"), Path("/nowhere/t.jsonl"))
    found = readiness.sessions_found([lost, bare], set(), [], readiness.Unrecorded([], Counter(), []), {7: PaneUnread("tmux broke"), 8: NotInTmux()})
    assert found == Missing(
        "1 session unreachable",
        "running sessions hands knows of: 2, runs of claude at a terminal that are none: piped 0, subcommand 0, print 0\n"
        "  hands cannot reach these:\n"
        "    /code/lost (pid 8) has lost its fritter, whose socket /gone.sock is gone, so it cannot be typed into: "
        "restart it from a PATH whose `claude` is hands' shim\n"
        "  whether hands can reach these is unknown:\n"
        "    /code/bare (pid 7) was started outside fritter, and which tmux pane it runs in could not be read, so whether it can be typed into is unknown: tmux broke",
        "restart it from a PATH whose `claude` is hands' shim",
    )


def test_a_session_whose_process_has_ended_is_not_running(root: Path) -> None:
    home = Home(root / "home")
    joined(home, "ended", 4001, None)
    assert readiness.sessions(processes(), home, installed(root), untmuxed) == Ready("no running sessions", "running sessions hands knows of: 0, runs of claude at a terminal that are none: piped 0, subcommand 0, print 0")


def test_an_unreadable_membership_file_is_named_and_left_where_it_is(root: Path) -> None:
    home = Home(root / "home")
    home.memberships.mkdir(parents=True)
    bad = home.membership(SessionId("bad"))
    bad.write_text("{not json")
    found = readiness.sessions(processes(), home, installed(root), untmuxed)
    assert isinstance(found, Missing) and f"{bad} names no session hands can read" in found.said
    assert bad.exists()


def test_a_session_that_never_ran_a_hook_is_named_by_cwd_and_pid_with_the_fix(root: Path) -> None:
    home = Home(root / "home")
    path = installed(root)
    project = root / "project"
    project.mkdir()
    pid = 4001
    found = readiness.sessions(processes(at_a_terminal(pid, root / "install" / "9.9.9", project)), home, path, untmuxed)
    assert isinstance(found, Missing) and f"{project} (pid {pid}) is a session hands has no record of" in found.said


def test_a_claude_printing_or_piped_into_at_a_terminal_is_no_session(root: Path) -> None:
    home = Home(root / "home")
    path = installed(root)
    table = processes(at_a_terminal(4001, root / "install" / "9.9.7", root, ("-p", "hello")), at_a_terminal(4002, root / "install" / "9.9.9", root, piped=True))
    found = readiness.sessions(table, home, path, untmuxed)
    assert found == Ready("no running sessions", "running sessions hands knows of: 0, runs of claude at a terminal that are none: piped 1, subcommand 0, print 1")


def test_a_check_run_from_a_removed_directory_says_its_own_config_cannot_be_told(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gone = root / "gone"
    gone.mkdir()
    monkeypatch.chdir(gone)
    gone.rmdir()
    found = readiness.unrecorded(processes(), Home(root / "home"), installed(root), set())
    assert isinstance(found, readiness.Unfindable) and found.said.startswith("cannot tell which Claude Code config this check runs under")


def test_a_process_the_kernel_will_not_describe_is_named_and_hides_no_other(root: Path) -> None:
    # As kern.procargs2 answers a process caught mid-exec, or on its way out.
    refused, described, ours = 4001, 4002, 4000
    table = processes(
        Running(ours), Running(refused, parent=ours, tty=TTY, refused={PROCARGS2: errno.EIO}), at_a_terminal(described, root / "install" / "9.9.9", root)
    )
    scan = terminal_processes(table)
    assert (scan.unread, [process.pid for process in scan.found]) == ([Undescribed(refused, ours, "kern.procargs2", errno.EIO)], [described])


def test_processes_that_could_not_be_looked_at_leave_unknown_whether_any_is_a_session() -> None:
    unread = [Undescribed(29648, 1, "kern.procargs2", errno.EIO), Undescribed(29649, 1, "proc_pidfdinfo of fd 0", errno.EIO)]
    assert readiness.sessions_found([], set(), [], readiness.Unrecorded([], Counter(), unread), {}) == Unknown(
        "unrecorded sessions unknown",
        "running sessions hands knows of: 0, runs of claude at a terminal that are none: piped 0, subcommand 0, print 0\n  whether hands can reach these is unknown:\n"
        "    the kernel would not describe these processes at a terminal, so whether any is a session hands has no record of is unknown: "
        "pid 29648 (kern.procargs2: Input/output error), pid 29649 (proc_pidfdinfo of fd 0: Input/output error)"
    )


def test_a_process_not_described_is_unknown_unless_hands_knows_it_or_it_is_a_runs_helper() -> None:
    run = terminal(1, "/v/1.0.0")
    stranger, member, helper = (Undescribed(pid, parent, "kern.procargs2", errno.EIO) for pid, parent in ((7, 0), (8, 0), (9, 1)))
    found = readiness.unjoined(Home(Path("/h")), Path("/v/1.0.0"), CHECKED, TerminalProcesses([run], [stranger, member, helper]), {1, 8}, reads_its_terminal)
    assert found.unread == [stranger]


def test_a_run_of_claude_whose_terminal_the_kernel_will_not_say_is_unknown_not_a_failed_scan() -> None:
    told, piped = terminal(1, "/v/1.0.0"), terminal(2, "/v/1.0.0", tty=PIPED)
    refusal = Undescribed(1, 0, "proc_pidfdinfo of fd 0", errno.EIO)

    def refusing(process: TerminalProcess) -> bool | Undescribed:
        return refusal if process is told else reads_its_terminal(process)

    found = readiness.unjoined(Home(Path("/h")), Path("/v/1.0.0"), CHECKED, TerminalProcesses([told, piped], []), set(), refusing)
    assert (found.sessions, found.others, found.unread) == ([], {"piped": 1}, [refusal])


def test_a_reused_pid_of_an_ended_session_hides_no_session_hands_has_no_record_of(root: Path) -> None:
    home = Home(root / "home")
    path = installed(root)
    pid = 4001
    table = processes(at_a_terminal(pid, root / "install" / "9.9.9", root, start=1000.0))
    # Written before the process under its pid started, as an ended session's file left behind is.
    os.utime(home.membership(joined(home, "ended", pid, None).id), (0, 0))
    found = readiness.sessions(table, home, path, untmuxed)
    assert isinstance(found, Missing) and f"(pid {pid}) is a session hands has no record of" in found.said


def test_a_session_on_a_version_pruned_since_it_started_is_still_found(root: Path) -> None:
    home = Home(root / "home")
    path = installed(root)
    # Started on a version no longer installed: the kernel says the path it was started by, which names nothing now
    # (processescontract.py).
    pid = 4001
    found = readiness.sessions(processes(at_a_terminal(pid, root / "install" / "9.9.8", root)), home, path, untmuxed)
    assert isinstance(found, Missing) and f"(pid {pid}) is a session hands has no record of" in found.said


@pytest.mark.outside
def test_a_session_hands_knows_of_is_not_named_as_unknown(root: Path) -> None:
    home = Home(root / "home")
    path = installed(root)
    listening = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listening.bind(str(root / "live.sock"))
    listening.listen()
    pid = 4001
    try:
        joined(home, "known", pid, root / "live.sock")
        found = readiness.sessions(processes(at_a_terminal(pid, root / "install" / "9.9.9", root)), home, path, untmuxed)
    finally:
        listening.close()
    assert found == Ready("1 running session reachable", f"running sessions hands knows of: 1, runs of claude at a terminal that are none: piped 0, subcommand 0, print 0\n  these are typed into:\n    /code/known (pid {pid}) is typed into through its fritter")


CONFIG = Path("/home/.claude")
CHECKED = readiness.config_dir({"CLAUDE_CONFIG_DIR": str(CONFIG)}, Path("/"))
# The terminal a process made by `terminal` reads and writes, unless it is made with another, as a piped one is.
INTERACTIVE, PIPED = 1, 2


def terminal(
    pid: int, executable: str, cwd: str = "/code", parent: int = 0, config: Path = CONFIG, arguments: tuple[str, ...] = (), tty: int = INTERACTIVE
) -> TerminalProcess:
    return TerminalProcess(pid, parent, Path(executable), Path(cwd), {"CLAUDE_CONFIG_DIR": str(config)}, arguments, tty)



def reads_its_terminal(process: TerminalProcess) -> bool:
    return process.tty == INTERACTIVE


def unjoined(claude: str, terminals: list[TerminalProcess], members: set[int] | None = None, home: Home = Home(Path("/h"))) -> list[TerminalProcess]:
    return [session.process for session in readiness.unjoined(home, Path(claude), CHECKED, TerminalProcesses(terminals, []), members or set(), reads_its_terminal).sessions]


def test_a_session_is_a_terminal_process_of_any_version_of_the_real_claudes_install() -> None:
    terminals = [
        terminal(1, "/v/0.9.0", "/code/old"),
        terminal(2, "/v/1.0.0", "/code/joined"),
        terminal(3, "/bin/zsh", "/code/shell"),
        terminal(4, "/elsewhere/v/1.0.0", "/code/other-install"),
    ]
    assert unjoined("/v/1.0.0", terminals, {2}) == [terminals[0]]


def test_a_cask_keeps_each_version_in_a_directory_named_for_it() -> None:
    old = terminal(1, "/opt/homebrew/Caskroom/claude-code/0.9.0/claude")
    assert unjoined("/opt/homebrew/Caskroom/claude-code/1.0.0/claude", [old]) == [old]


def test_a_version_named_directory_above_claude_code_is_part_of_its_install() -> None:
    other = terminal(1, "/tools/1.4.0/bin/claude")
    assert unjoined("/tools/1.5.0/bin/claude", [other]) == []


def test_a_pre_release_is_a_version_of_the_same_install() -> None:
    beta = terminal(1, "/v/1.0.1-beta.1")
    assert unjoined("/v/1.0.0", [beta]) == [beta]


def test_a_plain_claude_is_itself_and_not_its_directorys_other_programs() -> None:
    terminals = [terminal(1, "/usr/local/bin/claude"), terminal(2, "/usr/local/bin/nvim")]
    assert unjoined("/usr/local/bin/claude", terminals) == [terminals[0]]


def test_a_session_under_another_config_as_the_brain_is_has_other_plugins_and_is_not_named() -> None:
    brain = terminal(1, "/v/1.0.0", config=Path("/h/brain"))
    assert unjoined("/v/1.0.0", [brain]) == []


@pytest.mark.parametrize("checked, started", [(".claude", "dotfiles/claude"), ("dotfiles/claude", ".claude")])
def test_a_session_under_the_same_config_by_way_of_a_link_is_named(tmp_path: Path, checked: str, started: str) -> None:
    (tmp_path / "dotfiles" / "claude").mkdir(parents=True)
    (tmp_path / ".claude").symlink_to(tmp_path / "dotfiles" / "claude")
    session = terminal(1, "/v/1.0.0", config=tmp_path / started)
    config = readiness.config_dir({"CLAUDE_CONFIG_DIR": str(tmp_path / checked)}, Path("/"))
    found = readiness.unjoined(Home(Path("/h")), Path("/v/1.0.0"), config, TerminalProcesses([session], []), set(), reads_its_terminal)
    assert [joined.process for joined in found.sessions] == [session]


def test_a_relative_config_is_read_from_the_working_directory_of_the_process_that_names_it(tmp_path: Path) -> None:
    assert readiness.config_dir({"CLAUDE_CONFIG_DIR": "alt"}, tmp_path) == readiness.config_dir({"CLAUDE_CONFIG_DIR": str(tmp_path / "alt")}, Path("/"))


def test_with_no_config_named_a_process_runs_under_the_claude_in_its_own_home(tmp_path: Path) -> None:
    assert readiness.config_dir({"HOME": str(tmp_path)}, Path("/")) == readiness.config_dir({"CLAUDE_CONFIG_DIR": str(tmp_path / ".claude")}, Path("/"))


def test_a_config_named_through_a_link_loop_is_a_config_no_session_runs_under(tmp_path: Path) -> None:
    (tmp_path / "loop").symlink_to(tmp_path / "loop")
    assert readiness.config_dir({"CLAUDE_CONFIG_DIR": str(tmp_path / "loop")}, Path("/")) != readiness.config_dir({"HOME": str(tmp_path)}, Path("/"))


def test_a_sessions_own_helper_run_from_its_executable_is_not_a_session() -> None:
    session, helper = terminal(10, "/v/1.0.0"), terminal(11, "/v/1.0.0", parent=10)
    assert unjoined("/v/1.0.0", [session, helper], {10}) == []


def test_a_claude_the_shim_would_not_have_run_as_a_session_is_not_named() -> None:
    printing, piped = terminal(1, "/v/1.0.0", arguments=("-p", "hello")), terminal(2, "/v/1.0.0", tty=PIPED)
    prompted = terminal(3, "/v/1.0.0", arguments=("--", "-p"))
    controlling = terminal(4, "/v/1.0.0", arguments=("remote-control", "--spawn", "worktree"))
    asked = terminal(5, "/v/1.0.0", arguments=("fix the readme",))
    found = readiness.unjoined(
        Home(Path("/h")), Path("/v/1.0.0"), CHECKED, TerminalProcesses([printing, piped, prompted, controlling, asked], []), set(), reads_its_terminal
    )
    assert [session.process for session in found.sessions] == [prompted, asked]
    assert found.others == {"print": 1, "piped": 1, "subcommand": 1}


def test_the_helper_of_a_claude_that_is_no_session_is_no_session_either() -> None:
    printing, helper = terminal(10, "/v/1.0.0", arguments=("-p", "hello")), terminal(11, "/v/1.0.0", parent=10)
    assert unjoined("/v/1.0.0", [printing, helper]) == []


def test_a_session_hands_has_no_record_of_is_told_to_reload_where_it_can_be_typed_into_and_to_restart_where_not(root: Path) -> None:
    home = Home(root / "home")
    fritter = terminal(5, str(home.fritter))
    inside, outside = terminal(10, "/v/1.0.0", "/code/in", parent=5), terminal(11, "/v/1.0.0", "/code/out", parent=6)
    paned, behind = terminal(12, "/v/1.0.0", "/code/paned", parent=6), terminal(13, "/v/1.0.0", "/code/behind", parent=6)
    unread = terminal(14, "/v/1.0.0", "/code/unread", parent=6)
    sessions = readiness.unjoined(home, Path("/v/1.0.0"), CHECKED, TerminalProcesses([fritter, inside, outside, paned, behind, unread], []), set(), reads_its_terminal)
    # The fritter that wrapped a session types into it whatever its pane is.
    panes: dict[int, Keyboard] = {10: PaneUnread("no tmux"), 11: NotInTmux(), 12: PANE, 13: Behind(PANE), 14: PaneUnread("tmux did not answer")}
    found = readiness.sessions_found([], set(), [], sessions, panes)
    assert isinstance(found, Missing)
    assert "/code/in (pid 10) is a session hands has no record of, so it cannot be reached: /reload-plugins in it" in found.said
    assert "/code/out (pid 11) is a session hands has no record of, started outside fritter and in no tmux pane, so it cannot be typed into: restart it" in found.said
    assert "/code/paned (pid 12) is a session hands has no record of, so it cannot be reached: /reload-plugins in it" in found.said
    assert (
        "/code/behind (pid 13) is a session hands has no record of, started outside fritter, and another program has the keyboard of the tmux pane it runs in, "
        "%12 (window 3 of work), so it cannot be typed into: bring it back to the front of its pane, then /reload-plugins in it"
    ) in found.said
    assert "/code/unread (pid 14) is a session hands has no record of, started outside fritter, and which tmux pane it runs in could not be read (tmux did not answer)" in found.said


def test_with_no_real_claude_a_session_hands_has_no_record_of_cannot_be_found(root: Path) -> None:
    found = readiness.sessions(processes(), Home(root / "home"), "/nowhere", untmuxed)
    assert isinstance(found, Unknown) and "this PATH has no `claude` of its own" in found.said


def test_a_claude_that_is_a_script_cannot_tell_its_sessions(root: Path) -> None:
    (root / "bin").mkdir()
    (root / "bin" / "claude").write_text("#!/usr/bin/env node\n")
    (root / "bin" / "claude").chmod(0o755)
    found = readiness.sessions(processes(), Home(root / "home"), str(root / "bin"), untmuxed)
    assert isinstance(found, Unknown) and "is a script, so its sessions run as its interpreter" in found.said


def test_sessions_that_cannot_be_looked_for_are_said_beside_the_ones_that_cannot_be_reached() -> None:
    unreadable = liveness.Unreadable(Path("/h/sessions/bad.json"), b"{", Rejected("not json"))
    found = readiness.sessions_found([], set(), [unreadable], readiness.Unfindable("why"), {})
    assert isinstance(found, Missing) and "bad.json names no session" in found.said and "cannot be found: why" in found.said


def test_sessions_that_cannot_be_looked_at_are_unknown(root: Path) -> None:
    home = Home(root / "home")
    pid = 4001
    joined(home, "running", pid, None)
    table = processes(Running(pid))
    table.refused = {PROC_PID: errno.EPERM}
    found = readiness.sessions(table, home, "/nowhere", untmuxed)
    assert isinstance(found, Unknown) and found.said == f"cannot look at the running sessions in {home.memberships}: [Errno 1] sysctl could not say when pid {pid} started: Operation not permitted"


def test_the_sessions_line_counts_sessions_and_unreadable_files_apart_with_the_fix_for_each() -> None:
    lost = [Membership(SessionId(f"0f1e2d3c-aaaa-bbbb-cccc-00000000000{pid}"), pid, Path(f"/code/{pid}"), Path("/nowhere/t.jsonl")) for pid in (7, 8)]
    corrupt = liveness.Unreadable(Path("/home/memberships/broken.json"), b"{", Rejected("not JSON"))
    found = readiness.sessions_found(lost, set(), [corrupt], readiness.Unrecorded([], Counter(), []), {7: NotInTmux(), 8: Behind(PANE)})
    assert isinstance(found, Missing) and found.step == "2 sessions unreachable, 1 membership file unreadable"
    assert found.fix == (
        "for each, one of: restart it from a PATH whose `claude` is hands' shim; "
        "bring it back to the front of its pane, or restart it from a PATH whose `claude` is hands' shim; hands run removes it"
    )
    alike = readiness.sessions_found(lost, set(), [], readiness.Unrecorded([], Counter(), []), {7: NotInTmux(), 8: NotInTmux()})
    assert isinstance(alike, Missing) and alike.fix == "for each, restart it from a PATH whose `claude` is hands' shim"


def test_sessions_that_could_not_be_looked_for_are_unknown_by_no_count() -> None:
    found = readiness.sessions_found([], set(), [], readiness.Unfindable("cannot look at this user's processes at a terminal: refused"), {})
    assert isinstance(found, Unknown) and found.step == "unrecorded sessions unknown"


def test_a_claude_code_whose_path_names_no_version_is_named_without_one(root: Path) -> None:
    executable(root / "bin" / "claude", "\x7fELF")
    found = readiness.claude(f"{root / 'bin'}:{SYSTEM_PATH}")
    assert found == Ready("Claude Code", f"Claude Code is installed: {root / 'bin' / 'claude'}")


def test_a_version_named_directory_above_claude_code_is_not_its_version(root: Path) -> None:
    executable(root / "tools" / "1.4.0" / "bin" / "claude", "\x7fELF")
    assert readiness.claude(f"{root / 'tools' / '1.4.0' / 'bin'}:{SYSTEM_PATH}").step == "Claude Code"
    executable(root / "Caskroom" / "claude-code" / "1.0.0" / "claude", "\x7fELF")
    assert readiness.claude(f"{root / 'Caskroom' / 'claude-code' / '1.0.0'}:{SYSTEM_PATH}").step == "Claude Code 1.0.0"


# Claude Code


@pytest.mark.outside
def test_a_native_claude_on_path_is_ready(root: Path) -> None:
    found = readiness.claude(installed(root))
    assert found == Ready("Claude Code 9.9.9", f"Claude Code is installed: {root / 'install' / '9.9.9'}")


def test_no_claude_on_path_is_missing_and_names_its_installer(root: Path) -> None:
    found = readiness.claude(f"{root / 'empty'}:{SYSTEM_PATH}")
    assert isinstance(found, Missing) and "no `claude` of its own" in found.said and "https://claude.ai/install.sh" in found.said


def test_a_claude_that_is_a_script_is_missing_and_names_the_native_one(root: Path) -> None:
    executable(root / "real" / "claude", "#!/bin/sh\n")
    found = readiness.claude(f"{root / 'real'}:{SYSTEM_PATH}")
    assert isinstance(found, Missing) and "is a script" in found.said and found.fix.startswith("`claude install`")


# PortAudio


def test_portaudio_that_pyaudio_loads_is_ready() -> None:
    found = readiness.portaudio()
    assert isinstance(found, Ready) and "PortAudio" in found.said


def test_a_pyaudio_that_cannot_load_is_missing_and_names_homebrews_portaudio(monkeypatch: pytest.MonkeyPatch) -> None:
    # As when Homebrew's portaudio is gone: importing PyAudio's extension fails.
    monkeypatch.setitem(sys.modules, "pyaudio", None)
    found = readiness.portaudio()
    assert isinstance(found, Missing) and "`brew install portaudio`" in found.said


# tmux


def test_a_path_with_tmux_is_ready_and_one_without_is_missing_and_says_how_to_install_it(tmp_path: Path) -> None:
    tmux = tmp_path / "with" / "tmux"
    tmux.parent.mkdir()
    tmux.write_text("#!/bin/sh\n")
    tmux.chmod(0o755)
    assert isinstance(readiness.tmux(str(tmux.parent)), Ready)
    found = readiness.tmux(str(tmp_path / "without"))
    assert isinstance(found, Missing) and "`brew install tmux`" in found.said


# hands on PATH


def hands_printing(root: Path, said: str, code: int = 0) -> str:
    """A PATH whose `hands --version` prints said and exits code, with the `hands-shim` the plugin's hooks run beside it."""
    executable(root / "tools" / "hands", f"#!/bin/sh\necho '{said}'\nexit {code}\n")
    executable(root / "tools" / "hands-shim", "#!/bin/sh\n")
    return f"{root / 'tools'}:{SYSTEM_PATH}"


@pytest.mark.outside
def test_this_hands_on_path_is_ready(root: Path) -> None:
    found = readiness.installed(children_on(SYSTEM), hands_printing(root, f"hands {version('hands')}"))
    assert isinstance(found, Ready) and str(root / "tools" / "hands") in found.said


def test_no_hands_on_path_is_missing_and_says_how_uv_puts_it_there(root: Path) -> None:
    found = readiness.installed(children_on(SYSTEM), f"{root / 'empty'}:{SYSTEM_PATH}")
    assert isinstance(found, Missing) and "`hands-shim`" in found.said and "`uv tool update-shell`" in found.said


@pytest.mark.outside
def test_a_hands_with_no_shim_beside_it_is_missing_since_the_plugin_s_hooks_cannot_start(root: Path) -> None:
    # As with a hands from before the plugin ran `hands-shim`: its `hands` answers, and every hook of the plugin fails.
    path = hands_printing(root, f"hands {version('hands')}")
    (root / "tools" / "hands-shim").unlink()
    found = readiness.installed(children_on(SYSTEM), path)
    assert isinstance(found, Missing) and found.step == "`hands-shim` not beside `hands` on PATH" and "`hands update`" in found.fix


@pytest.mark.outside
def test_a_shim_in_another_directory_is_not_the_one_beside_hands(root: Path) -> None:
    # As with a stale `hands-shim` left on PATH by an older install: it would run in place of this hands' own.
    hands_printing(root, f"hands {version('hands')}")
    (root / "tools" / "hands-shim").unlink()
    executable(root / "other" / "hands-shim", "#!/bin/sh\n")
    found = readiness.installed(children_on(SYSTEM), f"{root / 'tools'}:{root / 'other'}:{SYSTEM_PATH}")
    assert isinstance(found, Missing) and found.step == "`hands-shim` not beside `hands` on PATH" and str(root / "other" / "hands-shim") in found.said


@pytest.mark.outside
def test_another_hands_on_path_is_missing_naming_both(root: Path) -> None:
    # As after an upgrade that left an older install ahead on PATH: sessions would run its hooks.
    found = readiness.installed(children_on(SYSTEM), hands_printing(root, "hands 0.0.1"))
    assert isinstance(found, Missing) and "is hands 0.0.1, not this hands" in found.said


@pytest.mark.outside
def test_a_hands_that_cannot_say_its_version_is_unknown(root: Path) -> None:
    found = readiness.installed(children_on(SYSTEM), hands_printing(root, "broken", 3))
    assert isinstance(found, Unknown) and "exited 3" in found.said


# The backend


@pytest.mark.outside
def test_a_brain_with_no_login_is_missing_and_names_the_command(root: Path, monkeypatch: pytest.MonkeyPatch, fake_claude: Path) -> None:
    monkeypatch.setenv("LOGGED_IN", "0")
    home = Home(root / "home")
    onboard(home.brain)
    found = readiness.configured(children_on(SYSTEM), home, pin(home.brain, fake_claude), shell_without_hands_settings(), heartbeat.NeverRan(home.status))
    assert isinstance(found, Missing) and "`hands login` gives it one" in found.said


def test_settings_hands_cannot_read_are_missing_naming_the_file(root: Path) -> None:
    home = Home(root / "home")
    home.root.mkdir(parents=True)
    home.config.write_text("[llm\n")
    reached = readiness.configured(children_on(SYSTEM), home, LAID_OUT, {}, heartbeat.NeverRan(home.status))
    # The refusal, which names the file, is the detail, and the fix names the file without the refusal again.
    assert isinstance(reached, Missing) and reached.fix == f"correct {home.config}"
    assert reached.detail.startswith(f"hands cannot read its settings: {home.config}: not TOML") and reached.detail.count(str(home.config)) == 1


def test_a_setting_in_the_environment_is_missing_as_the_start_refuses_it(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = Home(root / "home")
    laid_out(home)
    found = readiness.configured(children_on(SYSTEM), home, LAID_OUT, {"HANDS_DEBUG": "1"}, heartbeat.NeverRan(home.status))
    assert isinstance(found, Missing) and "HANDS_DEBUG set, and hands reads no setting from the environment" in found.said


@pytest.mark.parametrize(
    ("outcome", "said"),
    [
        (child.NeverStarted(PermissionError(errno.EACCES, "Permission denied", "claude")), "could not be started: Permission denied"),
        (child.Ran(child.Killed(child.Kill(child.Limit(5))), child.Captured(b""), child.Captured(b"")), child.said(child.Killed(child.Kill(child.Limit(5))))),
    ],
)
def test_a_brain_whose_claude_could_not_be_asked_is_a_start_refused_and_not_a_login_to_make(root: Path, outcome: child.Outcome, said: str) -> None:
    home = Home(root / "home")
    laid_out(home)
    found = readiness.configured(answering(outcome), home, LAID_OUT, {}, heartbeat.NeverRan(home.status))
    # The start refuses on it all the same, so the line says so, with the reason the start gives as its fix.
    assert found == readiness.unstartable(f"`{LAID_OUT.claude(home.brain)} auth status` {said}")


def test_the_brain_says_its_account_and_never_a_key() -> None:
    # The version is the one the brain's claude was verified at.
    found = readiness.reaching(ClaudeCodeBackend(model="claude-sonnet-5", config_dir=Path("/h/brain"), account=Account("claude.ai", "brain@example.com"), brain=Runnable(PinnedClaude(Path("/h/brain/pinned-claude/9.9.9/claude"), "9.9.9"), None)))
    assert found == Ready("brain logged in as brain@example.com (claude.ai)", "the brain is logged in as brain@example.com (claude.ai), and reaches claude-sonnet-5 on Claude Code 9.9.9, hands' own copy: /h/brain/pinned-claude/9.9.9/claude")


# hands running


# The running hands a heartbeat here names.
HANDS = 4242


def hands_running() -> FakeProcesses:
    """The Mac's processes, with the running hands among them, started long before its heartbeat."""
    return processes(Running(HANDS))


def beating(home: Home) -> None:
    home.root.mkdir(parents=True, exist_ok=True)
    heart = heartbeat.Heart(home.status, HANDS, datetime.now(UTC), heartbeat.HEARTBEAT, kick_watchdog=lambda: None, clock=SYSTEM)
    heart.beat("running", None, 0, listening=True, degraded=())


def test_hands_up_is_ready(root: Path) -> None:
    home = Home(root / "home")
    beating(home)
    found = readiness.daemon(heartbeat.look(hands_running(), home.status, datetime.now(UTC)), datetime.now(UTC))
    assert isinstance(found, Ready) and found.said.startswith("hands is up")


def test_hands_that_never_ran_is_missing_and_says_to_run_it(root: Path) -> None:
    found = readiness.daemon(heartbeat.look(hands_running(), Home(root / "home").status, datetime.now(UTC)), datetime.now(UTC))
    assert isinstance(found, Missing) and "hands has not run" in found.said and "`hands run`" in found.said


def test_a_heartbeat_hands_cannot_read_is_unknown(root: Path) -> None:
    home = Home(root / "home")
    home.root.mkdir(parents=True)
    home.status.write_text("not json")
    assert isinstance(readiness.daemon(heartbeat.look(hands_running(), home.status, datetime.now(UTC)), datetime.now(UTC)), Unknown)


# hands check


STEPS = ["claude", "portaudio", "tmux", "hands", "shim", "plugin", "backend", "grant", "running", "sessions"]


def logged_in(_children: object, _llm: object, home: Home, _environment: object, brain: Runnable) -> ClaudeCodeBackend:
    """The brain as its login check finds it logged in: that check is the start's, tested beside it."""
    return ClaudeCodeBackend(model="claude-sonnet-5-5", config_dir=home.brain, account=Account("claude.ai", "brain@example.com"), brain=brain)


def expired_beat(home: Home, degraded: tuple[heartbeat.Degradation, ...] = (heartbeat.LOGIN_EXPIRED,)) -> heartbeat.Status:
    """The heartbeat a running hands writes once its brain's turns said its login expired."""
    home.root.mkdir(parents=True, exist_ok=True)
    clock = VirtualClock()
    heartbeat.Heart(home.status, os.getpid(), clock.wall(), heartbeat.HEARTBEAT, kick_watchdog=lambda: None, clock=clock).beat("running", None, 0, listening=False, degraded=degraded)
    status = heartbeat.read(home.status)
    assert status is not None
    return status


@pytest.mark.parametrize("running", [heartbeat.Up, heartbeat.Unresponsive], ids=["up", "not responding"])
def test_a_login_the_running_brain_said_expired_is_missing_with_hands_fix_and_no_claude_code_is_started(root: Path, running: Callable[[heartbeat.Status], heartbeat.Verdict]) -> None:
    # A daemon that stopped beating still runs its brain, whose turns said what they said.
    home = Home(root / "home")
    laid_out(home)
    found = readiness.configured(Unstarting(SYSTEM, lambda _: None), home, LAID_OUT, {}, running(expired_beat(home)))
    assert found == Missing("brain's login expired", "the brain's login has expired, so every turn fails", "`hands login`, then `hands restart`")
    assert found.said == heartbeat.LOGIN_EXPIRED.said


def never_ran(home: Home) -> heartbeat.Verdict:
    return heartbeat.NeverRan(home.status)


def a_brain_that_said_nothing_of_its_login(home: Home) -> heartbeat.Verdict:
    return heartbeat.Up(expired_beat(home, degraded=()))


def hands_down(home: Home) -> heartbeat.Verdict:
    # Its brain is gone with it, and what it said is no longer said by a brain that runs.
    return heartbeat.Down(expired_beat(home))


@pytest.mark.parametrize(
    "running",
    [never_ran, a_brain_that_said_nothing_of_its_login, hands_down],
    ids=["no hands", "a brain that said nothing of its login", "hands down"],
)
def test_a_login_no_running_brain_said_expired_is_asked_of_its_claude_code(root: Path, monkeypatch: pytest.MonkeyPatch, running: Callable[[Home], heartbeat.Verdict]) -> None:
    home = Home(root / "home")
    laid_out(home)
    asked: list[Home] = []

    def asking(children: object, llm: object, home: Home, environment: object, brain: Runnable) -> ClaudeCodeBackend:
        asked.append(home)
        return logged_in(children, llm, home, environment, brain)

    monkeypatch.setattr(readiness, "resolve", asking)
    found = readiness.configured(children_on(SYSTEM), home, LAID_OUT, {}, running(home))
    assert (asked, found.step) == ([home], "brain logged in as brain@example.com (claude.ai)")


def set_up(root: Path, fritter: Path, monkeypatch: pytest.MonkeyPatch, plugins: object) -> Home:
    """A home on which every step of the README is done, with `plugins` listed by its claude."""
    home = Home(root / "home")
    home.bin.mkdir(parents=True)
    shutil.copy2(fritter, home.bin / "fritter")
    executable(home.shim, shim_script(home.bin / "fritter", home.wire))
    beating(home)
    laid_out(home)
    monkeypatch.setattr(readiness, "resolve", logged_in)
    hands_printing(root, f"hands {version('hands')}")
    executable(root / "tools" / "tmux", "#!/bin/sh\n")
    monkeypatch.setenv("PATH", f"{home.bin}:{root / 'tools'}:{claude_listing(root, plugins)}")
    monkeypatch.setattr("hands.voice.talkkey.granted", lambda: True)

    # The fake claude is a script, whose sessions cannot be told; take it for a native one that runs nowhere.
    def native(_claude: Path | None) -> Path:
        return root / "versions" / "9.9.9"

    monkeypatch.setattr(readiness, "claude_code", native)
    return home


def marks(capsys: pytest.CaptureFixture[str]) -> dict[str, str]:
    lines = [line for line in capsys.readouterr().out.splitlines() if not line.startswith(" ")]
    # A check with a step not done ends on where each step is said in full, and one with every step done does not.
    hinted = lines[-1] == VERBOSE_HINT
    steps = lines[:-1] if hinted else lines
    assert len(steps) == len(STEPS)
    found = {step: line.split()[0] for step, line in zip(STEPS, steps)}
    assert hinted == any(mark != "✓" for mark in found.values())
    return found


@pytest.mark.outside
def test_every_step_done_is_ok_and_exits_0(root: Path, fritter: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    home = set_up(root, fritter, monkeypatch, [listed(PLUGIN_ID, True)])
    assert main(["--home", str(home.root), "check"], pin=LAID_OUT, processes=hands_running()) == 0
    assert marks(capsys) == dict.fromkeys(STEPS, "✓")


# Each undoes one step of a home set_up made.
type Undo = Callable[[Path, Home, pytest.MonkeyPatch], None]


def no_hands(root: Path, _home: Home, _monkeypatch: pytest.MonkeyPatch) -> None:
    (root / "tools" / "hands").unlink()


def unshimmed(_root: Path, home: Home, _monkeypatch: pytest.MonkeyPatch) -> None:
    home.shim.unlink()


def no_plugin(root: Path, _home: Home, _monkeypatch: pytest.MonkeyPatch) -> None:
    claude_listing(root, [])


def logged_out(_root: Path, _home: Home, monkeypatch: pytest.MonkeyPatch) -> None:
    def refused(*_: object) -> object:
        raise Rejected("the brain has no login: `hands login` gives it one")

    monkeypatch.setattr(readiness, "resolve", refused)


def unpinned(_root: Path, home: Home, _monkeypatch: pytest.MonkeyPatch) -> None:
    LAID_OUT.claude(home.brain).unlink()


def ungranted(_root: Path, _home: Home, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("hands.voice.talkkey.granted", lambda: False)


def not_running(_root: Path, home: Home, _monkeypatch: pytest.MonkeyPatch) -> None:
    home.status.unlink()


@pytest.mark.outside
@pytest.mark.parametrize(
    ("undo", "steps"),
    # A brain with no pinned claude is one hands cannot start, which the backend step says of the start.
    [(no_hands, {"hands"}), (unshimmed, {"shim"}), (no_plugin, {"plugin"}), (unpinned, {"backend"}), (logged_out, {"backend"}), (ungranted, {"grant"}), (not_running, {"running"})],
    ids=["hands", "shim", "plugin", "unpinned", "backend", "grant", "running"],
)
def test_a_home_missing_one_step_names_that_step_and_exits_1(
    root: Path, fritter: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], undo: Undo, steps: set[str]
) -> None:
    home = set_up(root, fritter, monkeypatch, [listed(PLUGIN_ID, True)])
    undo(root, home, monkeypatch)
    assert main(["--home", str(home.root), "check"], pin=LAID_OUT, processes=hands_running()) == 1
    assert marks(capsys) == {step: "✗" if step in steps else "✓" for step in STEPS}
    # The command's event carries every step's finding, in the same order.
    [event] = [line for line in map(json.loads, audit.tail(home.audit, 100)[0]) if line.get("event") == "hands.command"]
    assert [finding["type"] for finding in event["facts"]["findings"]] == ["Missing" if each in steps else "Ready" for each in STEPS]
    # It fails saying why: each missing step, the first line of what was found of it, and its fix.
    missing = [finding for finding in event["facts"]["findings"] if finding["type"] == "Missing"]
    assert event["error"] == "exited 1: " + "; ".join(f"{each['step']}: {each['detail'].partition('\n')[0]}: {each['fix'].partition('\n')[0]}" for each in missing)


@pytest.mark.outside
def test_a_step_that_cannot_be_looked_at_exits_2_and_one_missing_outranks_it(
    root: Path, fritter: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    home = set_up(root, fritter, monkeypatch, "unreadable")
    assert main(["--home", str(home.root), "check"], pin=LAID_OUT, processes=hands_running()) == 2
    assert marks(capsys) == {**dict.fromkeys(STEPS, "✓"), "plugin": "?"}
    monkeypatch.setattr("hands.voice.talkkey.granted", lambda: False)
    assert main(["--home", str(home.root), "check"], pin=LAID_OUT, processes=hands_running()) == 1
    assert marks(capsys) == {**dict.fromkeys(STEPS, "✓"), "plugin": "?", "grant": "✗"}
    # Each event's reason names the steps that decided its exit: the one not looked at, then the missing one alone.
    events = [line for line in map(json.loads, audit.tail(home.audit, 100)[0]) if line.get("event") == "hands.command"]
    decided = [[finding for finding in event["facts"]["findings"] if finding["type"] == kind] for event, kind in zip(events, ("Unknown", "Missing"), strict=True)]
    [[unknown], [missing]] = decided
    assert [event["error"] for event in events] == [
        f"exited 2: {unknown['step']}: {unknown['detail'].partition('\n')[0]}",
        f"exited 1: {missing['step']}: {missing['detail'].partition('\n')[0]}: {missing['fix'].partition('\n')[0]}",
    ]


def test_a_missing_step_names_its_fix_on_its_line_and_v_says_each_step_in_full_beneath_it(
    root: Path, fritter: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    home = set_up(root, fritter, monkeypatch, [listed(PLUGIN_ID, True)])
    no_hands(root, home, monkeypatch)
    assert main(["--home", str(home.root), "check"], pin=LAID_OUT, processes=hands_running()) == 1
    *terse, hint = capsys.readouterr().out.splitlines()
    assert hint == VERBOSE_HINT
    [line] = [each for each in terse if each.startswith("✗")]
    named, fix = line.split("  → ")
    assert (named.rstrip(), fix) == ("✗ `hands` not on PATH", "`uv tool update-shell` puts the directory `uv tool install` writes it to on PATH")
    assert not any(each.startswith(" ") for each in terse)
    assert main(["--home", str(home.root), "check", "-v"], pin=LAID_OUT, processes=hands_running()) == 1
    full = capsys.readouterr().out.splitlines()
    # Each step's line is the one the terse check printed, and beneath it, what is said of it in full, less its fix.
    assert [each for each in full if not each.startswith(" ")] == terse
    beneath = full[full.index(line) + 1]
    assert beneath == "    this PATH has no `hands`, so Claude Code cannot run `hands-shim` and no session gets hands' hooks"
    events = [each for each in map(json.loads, audit.tail(home.audit, 100)[0]) if each.get("event") == "hands.command"]
    assert [event["facts"]["arguments.verbose"] for event in events] == [False, True]
