import pytest

from hands.core.front import Candidate, FrontUnread, NoSessionInFront, Screen, SessionInFront, in_front, told
from hands.core.session import SessionId
from hands.sessions.front import _under  # pyright: ignore[reportPrivateUsage]
from hands.sessions.processes import ancestor_terminals
from fakeprocesses import Running, processes

# Terminals by device number: two tabs, a tmux client in a third, and the panes behind it.
TAB, OTHER_TAB, CLIENT, PANE, OTHER_PANE, FRITTER = 1, 2, 3, 10, 11, 20
DOCS = Candidate(SessionId("s1"), "hands, docs", frozenset({FRITTER, TAB}))
TESTS = Candidate(SessionId("s2"), "hands, tests", frozenset({21, PANE}))
LAWS = Candidate(SessionId("s3"), "laws", frozenset({22, OTHER_PANE}))


def test_the_session_whose_terminal_the_front_tab_shows_is_in_front() -> None:
    assert in_front(Screen("iTerm2", frozenset({TAB})), {}, [DOCS, TESTS]) == SessionInFront("iTerm2", SessionId("s1"), "hands, docs")


def test_a_tab_running_tmux_shows_the_pane_its_client_is_on_not_the_others_of_the_server() -> None:
    panes = {CLIENT: PANE, 4: OTHER_PANE}
    assert in_front(Screen("iTerm2", frozenset({CLIENT})), panes, [DOCS, TESTS, LAWS]) == SessionInFront("iTerm2", SessionId("s2"), "hands, tests")


def test_an_app_showing_no_session_terminal_is_no_session_in_front_and_says_which_app() -> None:
    assert in_front(Screen("iTerm2", frozenset({OTHER_TAB})), {}, [DOCS]) == NoSessionInFront("iTerm2")
    assert in_front(Screen("Safari", frozenset()), {CLIENT: PANE}, [DOCS, TESTS]) == NoSessionInFront("Safari")


def test_an_app_that_cannot_say_its_front_tab_and_holds_several_sessions_is_left_unread() -> None:
    match in_front(Screen("Ghostty", frozenset({TAB, CLIENT})), {CLIENT: PANE}, [DOCS, TESTS, LAWS]):
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
        Running(shell, tty=tab), Running(fritter, parent=shell, tty=tab), Running(claude, parent=fritter, tty=fritters), Running(other)
    ).table()
    assert list(ancestor_terminals(claude, table)) == [fritters, tab, tab]
    assert list(ancestor_terminals(other, table)) == []
    assert sorted(found.pid for found in _under(shell, table)) == [fritter, claude]
    assert list(_under(claude, table)) == []
