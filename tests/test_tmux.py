"""Which tmux pane a session runs in: read over a process table and every tmux server's panes, never kept."""

import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

from conftest import stand_in
from hands.core.tmux import Behind, Listed, NotInTmux, Pane, PaneUnread, Server, Unanswered, keyboard_of, pane_of
from hands.sessions import tmux
from hands.sessions.tmux import Answered
from hands.sessions.terminals import Process, ancestor_terminals, front_terminal

# Terminals by device number: a pane of each of two servers, the pseudoterminal fritter runs a session on, and a tab.
PANE, OTHER_PANE, FRITTER, TAB = 10, 11, 20, 1
WORK, PLAY = Path("/tmp/tmux-501/default"), Path("/tmp/tmux-501/play")
HANDS = Pane(WORK, "%3", "cc-hands", 2)
LAWS = Pane(PLAY, "%0", "laws", 0)
# A session under fritter in a pane of the first server: claude on fritter's pseudoterminal, fritter on the pane's in
# front of its shell. Another, in a pane of the second server, and one stopped there, behind it; and one in a plain tab.
TABLE = {
    process.pid: process
    for process in (
        Process(100, 1, 501, PANE, None),
        Process(101, 100, 501, PANE, PANE),
        Process(102, 101, 501, FRITTER, FRITTER),
        Process(200, 1, 501, OTHER_PANE, OTHER_PANE),
        Process(201, 1, 501, OTHER_PANE, None),
        Process(300, 1, 501, TAB, TAB),
    )
}
SERVERS = [Listed({PANE: HANDS}), Listed({OTHER_PANE: LAWS})]


def test_a_session_under_fritter_in_a_pane_is_in_that_pane_and_one_in_another_server_names_that_servers_socket() -> None:
    assert pane_of(ancestor_terminals(102, TABLE), SERVERS) == HANDS
    assert pane_of(ancestor_terminals(200, TABLE), SERVERS) == LAWS


def test_a_session_in_a_plain_tab_is_not_in_tmux_and_with_no_server_running_none_is() -> None:
    assert pane_of(ancestor_terminals(300, TABLE), SERVERS) == NotInTmux()
    assert pane_of(ancestor_terminals(102, TABLE), []) == NotInTmux()


def test_a_server_that_did_not_answer_leaves_unread_only_the_sessions_no_other_server_holds() -> None:
    servers = [Listed({PANE: HANDS}), Unanswered("tmux at /tmp/tmux-501/play did not answer list-panes in 2 seconds")]
    assert pane_of(ancestor_terminals(102, TABLE), servers) == HANDS
    assert pane_of(ancestor_terminals(300, TABLE), servers) == PaneUnread("tmux at /tmp/tmux-501/play did not answer list-panes in 2 seconds")


def keyboard(pid: int, servers: Sequence[Server] = SERVERS) -> object:
    return keyboard_of(front_terminal(pid, TABLE), ancestor_terminals(pid, TABLE), servers)


def test_keys_reach_a_session_through_its_pane_only_while_it_is_in_front_of_that_pane_itself() -> None:
    assert keyboard(200) == LAWS
    # Stopped, or a background job: its shell has the pane's keys.
    assert keyboard(201) == Behind(LAWS)
    # On a terminal of its own inside the pane - fritter's here, an editor's or ssh's elsewhere - which has the pane's keys.
    assert keyboard(102) == Behind(HANDS)
    # Gone since its pane was read: nothing of it is in front of anything.
    assert keyboard(999) == NotInTmux()
    assert keyboard(300) == NotInTmux()
    assert keyboard(300, [Listed({PANE: HANDS}), Unanswered("tmux at /tmp/tmux-501/play did not answer list-panes in 2 seconds")]) == PaneUnread("tmux at /tmp/tmux-501/play did not answer list-panes in 2 seconds")


def test_a_relative_socket_whose_server_pid_is_now_another_users_names_no_server(tmp_path: Path) -> None:
    # Its server exited and its pid went to a process of root's: launchd's, pid 1, stands in for it. $TMUX is read from
    # a program built for the test, since macOS shows no one /bin/cat's.
    held = subprocess.Popen([stand_in(tmp_path / "stand-in")], env={"TMUX": "mine,1,0"})
    try:
        table = {held.pid: Process(held.pid, 1, os.geteuid(), 1, None), 1: Process(1, 0, 0, None, None)}
        assert tmux.sockets({"TMUX_TMPDIR": str(tmp_path)}, [held.pid], table) == []
    finally:
        held.kill()
        held.wait()


def test_sockets_are_looked_for_where_tmux_puts_them_with_an_empty_tmux_tmpdir_read_as_unset() -> None:
    assert tmux.socket_directory({"TMUX_TMPDIR": "/var/run/mine"}) == Path(f"/var/run/mine/tmux-{os.getuid()}")
    assert tmux.socket_directory({"TMUX_TMPDIR": ""}) == tmux.socket_directory({}) == Path(f"/tmp/tmux-{os.getuid()}")


def test_a_pane_closed_between_its_listing_and_the_look_at_its_terminal_leaves_the_rest_of_its_server_read() -> None:
    lines = ["/dev/null\t%1\t0\twork", "/dev/hands-no-such-terminal\t%2\t1\twork"]
    assert tmux.listed(Answered(WORK, lines)) == Listed({os.stat("/dev/null").st_rdev: Pane(WORK, "%1", "work", 0)})
