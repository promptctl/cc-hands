"""What is in front on the Mac's screen: the app, and the session it shows, read off the screen, the processes, and tmux."""

from pathlib import Path
from typing import NoReturn

import pytest

from hands.core.front import Candidate, FrontUnread, NoSessionInFront, Shown, SessionInFront, in_front, told
from hands.core.session import SessionId
from hands.sessions.front import _under, read_front  # pyright: ignore[reportPrivateUsage]
from hands.sessions.panes import socket_directory
from hands.sessions.processes import ControllingTerminal, ancestor_terminals
from hands.sessions.screen import FrontApp
from fakeprocesses import USER, Running, processes
from faketerminals import FakePane, FakeScreen, FakeServer, FakeTerminals

# Terminals by device number: two tabs, a tmux client in a third, and the panes behind it.
TAB, OTHER_TAB, CLIENT, PANE, OTHER_PANE, FRITTER = 1, 2, 3, 10, 11, 20
DOCS = Candidate(SessionId("s1"), "hands, docs", frozenset({FRITTER, TAB}))
TESTS = Candidate(SessionId("s2"), "hands, tests", frozenset({21, PANE}))
LAWS = Candidate(SessionId("s3"), "laws", frozenset({22, OTHER_PANE}))


def test_the_session_whose_terminal_the_front_tab_shows_is_in_front() -> None:
    assert in_front(Shown("iTerm2", frozenset({TAB})), {}, [DOCS, TESTS]) == SessionInFront("iTerm2", SessionId("s1"), "hands, docs")


def test_a_tab_running_tmux_shows_the_pane_its_client_is_on_not_the_others_of_the_server() -> None:
    panes = {CLIENT: PANE, 4: OTHER_PANE}
    assert in_front(Shown("iTerm2", frozenset({CLIENT})), panes, [DOCS, TESTS, LAWS]) == SessionInFront("iTerm2", SessionId("s2"), "hands, tests")


def test_an_app_showing_no_session_terminal_is_no_session_in_front_and_says_which_app() -> None:
    assert in_front(Shown("iTerm2", frozenset({OTHER_TAB})), {}, [DOCS]) == NoSessionInFront("iTerm2")
    assert in_front(Shown("Safari", frozenset()), {CLIENT: PANE}, [DOCS, TESTS]) == NoSessionInFront("Safari")


def test_an_app_that_cannot_say_its_front_tab_and_holds_several_sessions_is_left_unread() -> None:
    match in_front(Shown("Ghostty", frozenset({TAB, CLIENT})), {CLIENT: PANE}, [DOCS, TESTS, LAWS]):
        case FrontUnread(reason=reason):
            assert reason == "2 sessions run under the terminals Ghostty shows"
        case other:
            pytest.fail(f"read as {other}")


def test_the_brain_is_told_the_session_in_front_or_that_none_is_and_nothing_of_a_screen_left_unread() -> None:
    assert told(SessionInFront("iTerm2", SessionId("s1"), "hands, docs")) == (
        '[hands] As the user said this, iTerm2 was in front on the Mac\'s screen, showing the session "hands, docs" (id s1). '
        "Mention it only if it bears on what they said."
    )
    assert told(NoSessionInFront("Safari")) == (
        "[hands] As the user said this, Safari was in front on the Mac's screen, showing no session. Mention it only if it bears on what they said."
    )
    assert told(FrontUnread("no window is on screen")) == ""


def test_a_session_runs_under_each_terminal_on_its_line_of_parents_its_own_first_and_its_descendants_are_under_it() -> None:
    # A shell on a tab's terminal, and under it a fritter on the same, which runs a claude on a terminal of its own; the
    # test's own line, from launchd, reaches no terminal.
    tab, fritters = 0x1000_0001, 0x1000_0002
    shell, fritter, claude, other = 4001, 4002, 4003, 4004
    table = processes(
        Running(shell, terminal=ControllingTerminal(tab, foreground=False)), Running(fritter, parent=shell, terminal=ControllingTerminal(tab, foreground=False)), Running(claude, parent=fritter, terminal=ControllingTerminal(fritters, foreground=False)), Running(other)
    ).table()
    assert list(ancestor_terminals(claude, table)) == [fritters, tab, tab]
    assert list(ancestor_terminals(other, table)) == []
    assert sorted(found.pid for found in _under(shell, table)) == [fritter, claude]
    assert list(_under(claude, table)) == []


# An app at pid 900, a shell on a tab of its, and a session under that shell; and a session in a tmux pane.
APP, SHELL, SESSION, IN_PANE = 900, 901, 902, 950
RUNNING = processes(
    Running(APP),
    Running(SHELL, parent=APP, terminal=ControllingTerminal(TAB, foreground=False)),
    Running(SESSION, parent=SHELL, terminal=ControllingTerminal(TAB, foreground=True)),
    Running(IN_PANE, terminal=ControllingTerminal(PANE, foreground=True)),
)
SESSIONS = {SessionId("s1"): (SESSION, "hands, docs"), SessionId("s2"): (IN_PANE, "hands, tests")}


def tmux_holding(tmp_path: Path, clients: dict[int, str]) -> FakeTerminals:
    """A tmux server where tmux puts its socket under `tmp_path`, holding the session in its pane, with `clients`."""
    return FakeTerminals({socket_directory({"TMUX_TMPDIR": str(tmp_path)}, USER) / "default": FakeServer([FakePane("%3", PANE, "tests")], clients)})


async def front(tmp_path: Path, app: FrontApp | FrontUnread, terminals: FakeTerminals | None = None) -> object:
    return await read_front(FakeScreen(app), terminals or FakeTerminals(), RUNNING, SESSIONS, {"TMUX_TMPDIR": str(tmp_path)})


async def test_an_app_that_says_its_front_tab_shows_the_session_on_that_tab_and_none_on_another(tmp_path: Path) -> None:
    assert await front(tmp_path, FrontApp(APP, "iTerm2", tab=TAB)) == SessionInFront("iTerm2", SessionId("s1"), "hands, docs")
    assert await front(tmp_path, FrontApp(APP, "iTerm2", tab=OTHER_TAB)) == NoSessionInFront("iTerm2")


async def test_an_app_that_cannot_be_asked_shows_every_terminal_under_it_and_one_with_none_shows_no_session(tmp_path: Path) -> None:
    assert await front(tmp_path, FrontApp(APP, "Ghostty", tab=None)) == SessionInFront("Ghostty", SessionId("s1"), "hands, docs")
    assert await front(tmp_path, FrontApp(4, "Safari", tab=None)) == NoSessionInFront("Safari")


async def test_a_tab_a_tmux_client_runs_in_shows_the_session_in_the_pane_that_client_shows(tmp_path: Path) -> None:
    terminals = tmux_holding(tmp_path, {CLIENT: "%3"})
    assert await front(tmp_path, FrontApp(APP, "iTerm2", tab=CLIENT), terminals) == SessionInFront("iTerm2", SessionId("s2"), "hands, tests")


async def test_a_screen_unread_is_said_as_it_said_it_and_so_is_a_tmux_server_that_would_not_list_its_clients(tmp_path: Path) -> None:
    assert await front(tmp_path, FrontUnread("no app is in front: lsappinfo said ''")) == FrontUnread("no app is in front: lsappinfo said ''")
    terminals = tmux_holding(tmp_path, {CLIENT: "%3"})
    [server] = terminals.servers.values()
    server.wedged = "did not answer in 2 seconds"
    match await front(tmp_path, FrontApp(APP, "iTerm2", tab=CLIENT), terminals):
        case FrontUnread(reason=reason):
            assert reason.endswith("did not answer: tmux did not answer in 2 seconds")
        case other:
            pytest.fail(f"read as {other}")


class Broken(FakeTerminals):
    """Terminals whose read breaks on a fault of hands' own."""

    def sockets(self, directory: Path) -> NoReturn:
        raise ValueError("not enough values to unpack")


async def test_a_read_that_breaks_leaves_the_screen_unread_saying_why(tmp_path: Path) -> None:
    assert await front(tmp_path, FrontApp(APP, "iTerm2", tab=TAB), Broken()) == FrontUnread("ValueError: not enough values to unpack")
