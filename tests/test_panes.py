"""Which tmux pane a session runs in: read over a process table and every tmux server's panes, never kept."""

import os
from collections.abc import Sequence
from pathlib import Path

from fakeprocesses import USER, Running, processes
from faketerminals import FakePane, FakeServer, FakeTerminals
from hands.core.tmux import Behind, Listed, NotInTmux, Pane, PaneUnread, Server, Unanswered, keyboard_of, pane_of
from hands.sessions import panes, tmux
from hands.sessions.processes import ControllingTerminal, ancestor_terminals, foreground_terminal

# Terminals by device number: a pane of each of two servers, the pseudoterminal fritter runs a session on, and a tab.
PANE, OTHER_PANE, FRITTER, TAB = 10, 11, 20, 1
WORK, PLAY = Path("/tmp/tmux-501/default"), Path("/tmp/tmux-501/play")
HANDS = Pane(WORK, "%3", "cc-hands", 2)
LAWS = Pane(PLAY, "%0", "laws", 0)
# A session under fritter in a pane of the first server: claude on fritter's pseudoterminal, fritter on the pane's in
# front of its shell. Another, in a pane of the second server, and one stopped there, behind it; and one in a plain tab.
TABLE = processes(
    Running(100, terminal=ControllingTerminal(PANE, foreground=False)),
    Running(101, parent=100, terminal=ControllingTerminal(PANE, foreground=True)),
    Running(102, parent=101, terminal=ControllingTerminal(FRITTER, foreground=True)),
    Running(200, terminal=ControllingTerminal(OTHER_PANE, foreground=True)),
    Running(201, terminal=ControllingTerminal(OTHER_PANE, foreground=False)),
    Running(300, terminal=ControllingTerminal(TAB, foreground=True)),
).table()
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
    return keyboard_of(foreground_terminal(pid, TABLE), ancestor_terminals(pid, TABLE), servers)


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
    # Its server exited and its pid went to a process of root's: launchd's, pid 1, stands in for it.
    held = 4242
    table = processes(Running(held, terminal=ControllingTerminal(1, foreground=False), environment={"TMUX": "mine,1,0"}))
    assert panes.sockets(FakeTerminals(), table, {"TMUX_TMPDIR": str(tmp_path)}, [held], table.table()) == []


def test_a_relative_socket_is_where_its_server_runs_while_that_server_started_before_the_pane_that_names_it(tmp_path: Path) -> None:
    # A server started in /code, and a pane's shell it started; and a pane's shell whose server exited and left its pid to
    # a process of the user's that started after the shell did.
    table = processes(
        Running(500, start=100.0, cwd=Path("/code")),
        Running(501, parent=500, terminal=ControllingTerminal(10, foreground=True), start=200.0, environment={"TMUX": "work,500,0"}),
        Running(600, start=900.0, cwd=Path("/elsewhere")),
        Running(601, terminal=ControllingTerminal(11, foreground=True), start=200.0, environment={"TMUX": "play,600,0"}),
    )
    environment = {"TMUX_TMPDIR": str(tmp_path)}
    assert panes.sockets(FakeTerminals(), table, environment, [501], table.table()) == [Path("/code/work")]
    assert panes.sockets(FakeTerminals(), table, environment, [601], table.table()) == []


def test_a_server_found_in_its_directory_and_by_the_tmux_a_pane_names_is_asked_once_by_its_path_there(tmp_path: Path) -> None:
    socket = panes.socket_directory({"TMUX_TMPDIR": str(tmp_path)}, USER) / "default"
    table = processes(Running(101, terminal=ControllingTerminal(PANE, foreground=True), environment={"TMUX": f"{os.path.realpath(socket)},100,0"}))
    terminals = FakeTerminals({socket: FakeServer([FakePane("%3", PANE, "cc-hands", 2)])})
    assert panes.sockets(terminals, table, {"TMUX_TMPDIR": str(tmp_path)}, [101], table.table()) == [socket]


async def test_a_session_is_read_in_the_pane_of_a_server_only_its_tmux_names_and_left_unread_where_a_server_would_not_say(tmp_path: Path) -> None:
    elsewhere = Path("/srv/tmux/work")
    table = processes(Running(101, terminal=ControllingTerminal(PANE, foreground=True), environment={"TMUX": f"{elsewhere},100,0"}), Running(300, terminal=ControllingTerminal(TAB, foreground=True)))
    environment = {"TMUX_TMPDIR": str(tmp_path)}
    terminals = FakeTerminals({elsewhere: FakeServer([FakePane("%3", PANE, "cc-hands", 2)])})
    assert await panes.panes(terminals, table, [101, 300], environment) == [Pane(elsewhere, "%3", "cc-hands", 2), NotInTmux()]
    terminals.servers[elsewhere].wedged = "did not answer in 2 seconds"
    assert await panes.keyboards(terminals, table, [101, 300], environment) == [PaneUnread(f"tmux at {elsewhere} did not answer: tmux did not answer in 2 seconds")] * 2


def test_sockets_are_looked_for_where_tmux_puts_them_with_an_empty_tmux_tmpdir_read_as_unset() -> None:
    assert panes.socket_directory({"TMUX_TMPDIR": "/var/run/mine"}, USER) == Path(f"/var/run/mine/tmux-{USER}")
    assert panes.socket_directory({"TMUX_TMPDIR": ""}, USER) == panes.socket_directory({}, USER) == Path(f"/tmp/tmux-{USER}")


def test_a_pane_closed_between_its_listing_and_the_look_at_its_terminal_leaves_the_rest_of_its_server_read() -> None:
    lines = ["/dev/null\t%1\t0\twork", "/dev/hands-no-such-terminal\t%2\t1\twork"]
    assert tmux.listed(WORK, lines) == Listed({os.stat("/dev/null").st_rdev: Pane(WORK, "%1", "work", 0)})
