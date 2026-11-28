"""The `start_session` tool: `claude` started in a tmux window of the tmux session named for its folder, as from a terminal
outside any session, and the session that joined hands named."""

import asyncio
import json
import os
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from itertools import count
from pathlib import Path
from typing import Any

from fakechildren import NOT_THERE, FakeChildren
from fakeprocesses import FakeProcesses, Running, processes
from faketerminals import DEAD, FakePane, FakeServer, FakeTerminals, Wedged

from hands.core.session import Membership, SessionId
from hands.sessions.audit import AuditLog, segment
from hands.sessions.home import Home
from hands.sessions.startsession import JOIN_SECONDS, LOOK_SECONDS, SESSION_GIVEN, SETTINGS, as_from_a_terminal, descends
from hands.sessions.terminals import Terminals
from hands.sessions.tmux import NoTmux, TmuxTerminals
from hands.virtualclock import VirtualClock
from hands.voice.tool import Result
from hands.voice.tools import start_session_tool


def home_in(tmp_path: Path) -> Home:
    home = Home(tmp_path / "home")
    home.memberships.mkdir(parents=True)
    home.bin.mkdir(parents=True)
    return home


class Starts:
    """What a start reaches, as a test sets it up: the terminals a window opens in, the processes, the sessions that have
    joined hands, and the clock the start waits for one on."""

    def __init__(self, home: Home, opened: Callable[["Starts", FakePane, Sequence[str], Mapping[str, str]], None] | None = None) -> None:
        self.home = home
        self.clock = VirtualClock()
        self.processes: FakeProcesses = processes()
        self.members: list[Membership] = []
        # What each window opened ran, and the environment it was opened with.
        self.ran: list[tuple[Sequence[str], Mapping[str, str]]] = []
        self.fake = FakeTerminals(opened=self._opened)
        self.terminals: Terminals = self.fake
        self._then = opened
        self._joining = count(9000)

    def _opened(self, pane: FakePane, command: Sequence[str], environment: Mapping[str, str]) -> None:
        self.ran.append((command, environment))
        self.processes.add(Running(pane.pid, terminal=None))
        if self._then is not None:
            self._then(self, pane, command, environment)

    def joins(self, pane: FakePane, fritter: bool = True) -> None:
        """A session joins hands from a process the pane's process started, under fritter or not."""
        pid = next(self._joining)
        self.processes.add(Running(pid, parent=pane.pid))
        self.members.append(Membership(SessionId(f"s{pane.id[1:]}"), pid, Path("/code"), Path("/code/t.jsonl"), Path("/tmp/f.sock") if fritter else None))

    async def start(self, folder: Path, model: str = "", environment: Mapping[str, str] = os.environ) -> dict[str, Any]:
        """The tool called as the model calls it, in the environment hands was started in."""
        start_session = start_session_tool(self.clock, self.terminals, self.processes, self.home, AuditLog(self.home.audit, clock=datetime.now).record, environment, lambda: self.members)

        async def called() -> Result:
            return await start_session.body(folder=str(folder), model=model)

        return dict(await called())


def started(home: Home) -> list[dict[str, Any]]:
    return [line for line in map(json.loads, segment(home.audit, 0).read_text().splitlines()) if line.get("event") == "session.start"]


def folder_in(tmp_path: Path) -> Path:
    folder = tmp_path / "billing"
    folder.mkdir()
    return folder


def test_a_process_descends_from_itself_and_what_it_started_never_from_a_sibling() -> None:
    table = processes(*(Running(pid, parent) for pid, parent in {10: 1, 11: 10, 12: 11, 20: 1}.items())).table()
    assert [descends(pid, 10, table) for pid in (10, 11, 12, 20, 1, 99)] == [True, True, True, False, False, False]


async def test_a_folder_that_is_not_there_is_a_symlink_loop_or_is_not_a_whole_path_is_said(tmp_path: Path) -> None:
    starts = Starts(home_in(tmp_path))
    loop = tmp_path / "loop"
    loop.symlink_to(loop)
    for folder in (tmp_path / "nowhere", loop):
        assert f"there is no folder {folder}" in (await starts.start(folder))["error"]
    # Never the daemon's own working directory.
    for folder in (Path("billing"), Path("")):
        assert f"{folder} is no absolute folder, nor one from ~" in (await starts.start(folder))["error"]
    assert [event["outcome"] for event in started(starts.home)] == ["failed"] * 4
    assert starts.ran == []


def test_a_terminal_outside_any_session_has_nothing_of_one_and_keeps_the_users_own_setup() -> None:
    inside = {
        **{name: "given" for name in SESSION_GIVEN},
        # hands' own Hugging Face token is withheld too: the brain reads these windows.
        "HF_TOKEN": "a-hub-token",
        # fritter's tap of the session hands was started in, and the proxy it replaced.
        "FRITTER_TAP": "http://127.0.0.1:9", "HTTPS_PROXY": "http://127.0.0.1:9", "FRITTER_OUTER_HTTPS_PROXY": "http://proxy.lan:3128",
    }
    users = {"PATH": "/Users/me/.hands/bin:/usr/bin", "CLAUDE_CONFIG_DIR": "/Users/me/.claude.work", "ANTHROPIC_API_KEY": "sk-mine", "HANDS_HOME": "/elsewhere"}
    assert as_from_a_terminal({**inside, **users}) == {**users, "HTTPS_PROXY": "http://proxy.lan:3128"}


async def test_a_session_is_started_in_a_window_of_the_tmux_session_named_for_its_folder_and_named_once_it_joins(tmp_path: Path) -> None:
    starts = Starts(home_in(tmp_path), lambda starts, pane, _command, _environment: starts.joins(pane))
    folder = folder_in(tmp_path)
    inside = {"PATH": "/usr/bin", "CLAUDECODE": "1", "HF_TOKEN": "a-hub-token"}
    assert await starts.start(folder, "opus", inside) == {"session": "s0", "tmux_session": "billing", "pane": "%0"}
    assert await starts.start(folder, environment=inside) == {"session": "s1", "tmux_session": "billing", "pane": "%1"}
    [(command, environment), _] = starts.ran
    # The window is handed the home its session reports to, and runs hands' own claude as from a terminal outside any
    # session, trusting its project's MCP servers, whose dialog would hold it before it joined.
    [server] = starts.fake.servers.values()
    assert [pane.environment for pane in server.panes] == [{"HANDS_HOME": str(starts.home.root)}] * 2
    assert list(command[-4:]) == ["claude", "--settings", '{"enableAllProjectMcpServers": true}', "--model=opus"]
    assert environment == {"PATH": "/usr/bin"}
    # The start's event says which settings its session was launched with.
    assert [(event["facts"]["made_tmux_session"], event["facts"]["session"], event["facts"]["settings"]) for event in started(starts.home)] == [(True, "s0", SETTINGS), (False, "s1", SETTINGS)]


async def test_a_session_started_where_no_tmux_is_on_path_answers_that_it_was_not_started(tmp_path: Path) -> None:
    starts = Starts(home_in(tmp_path))
    starts.terminals = NoTmux("no tmux is on the PATH /nowhere")
    reason = "hands starts a session in a tmux window, and none was opened: no tmux is on the PATH /nowhere"
    assert await starts.start(folder_in(tmp_path)) == {"error": reason}
    [event] = started(starts.home)
    assert (event["outcome"], event["error"]) == ("failed", f"NotStarted: {reason}")


async def test_a_claude_that_ends_before_it_joins_is_said_with_what_its_pane_showed_or_that_its_window_closed(tmp_path: Path) -> None:
    def dies(starts: Starts, pane: FakePane, _command: Sequence[str], _environment: Mapping[str, str]) -> None:
        pane.screen = "error: unknown option '--settings'"
        pane.end()

    def closes(starts: Starts, pane: FakePane, _command: Sequence[str], _environment: Mapping[str, str]) -> None:
        [server] = starts.fake.servers.values()
        server.close(pane)

    folder = folder_in(tmp_path)
    # What it said as it ended is told, though tmux's notice that it is dead took its screen.
    assert (await Starts(home_in(tmp_path / "dies"), dies).start(folder))["error"] == f"`claude` in {folder} ended before it joined hands; the last of what tmux pane %0 drew:\nerror: unknown option '--settings'\n{DEAD} (signal kill)"
    assert (await Starts(home_in(tmp_path / "closes"), closes).start(folder))["error"] == f"`claude` in {folder} ended before it joined hands, and its window closed with it"


async def test_a_claude_still_running_unjoined_once_its_time_is_up_is_said_with_what_its_pane_shows(tmp_path: Path) -> None:
    def holds(starts: Starts, pane: FakePane, _command: Sequence[str], _environment: Mapping[str, str]) -> None:
        pane.screen = "Do you trust the files in this folder?"

    starts = Starts(home_in(tmp_path), holds)
    folder = folder_in(tmp_path)
    started_at = starts.clock.at
    answer = asyncio.ensure_future(starts.start(folder))
    # Looked at once every LOOK_SECONDS, until the time to join is past: the look after that says so.
    while starts.clock.at - started_at <= JOIN_SECONDS:
        await starts.clock.sleeping(starts.clock.at + LOOK_SECONDS)
        starts.clock.advance(LOOK_SECONDS)
    assert (await answer)["error"] == f"`claude` in {folder} has not joined hands in 30 seconds, and is still running in tmux pane %0, which shows:\nDo you trust the files in this folder?"


async def test_a_session_that_joins_outside_fritter_is_said_to_be_one_hands_cannot_type_into(tmp_path: Path) -> None:
    starts = Starts(home_in(tmp_path), lambda starts, pane, _command, _environment: starts.joins(pane, fritter=False))
    error = (await starts.start(folder_in(tmp_path)))["error"]
    assert error.startswith("session s0 joined hands in tmux pane %0, but not under fritter, so hands cannot type into it")


async def test_a_session_whose_tmux_could_not_be_started_answers_that_it_was_not_started(tmp_path: Path) -> None:
    starts = Starts(home_in(tmp_path))
    # A tmux the OS never starts.
    starts.terminals = TmuxTerminals(FakeChildren(VirtualClock(), {str(tmp_path / "bin" / "tmux"): NOT_THERE}), tmp_path / "bin" / "tmux")
    answer = await starts.start(folder_in(tmp_path), environment={"PATH": "/usr/bin", "TMUX_TMPDIR": str(tmp_path)})
    assert answer.keys() == {"error"}
    assert answer["error"].startswith("hands starts a session in a tmux window, and none was opened: tmux could not be started: ")


async def test_a_tmux_that_gives_no_answer_to_the_opening_is_said_to_maybe_have_opened_the_window(tmp_path: Path) -> None:
    starts = Starts(home_in(tmp_path))
    environment = {"PATH": "/usr/bin", "TMUX_TMPDIR": str(tmp_path)}
    starts.fake.servers[tmp_path / f"tmux-{starts.processes.user}" / "default"] = FakeServer(answering=Wedged("did not exit in 5.0 s, so it was killed"))
    reason = "hands starts a session in a tmux window, and tmux did not say whether it opened one, so `claude` may be starting in tmux session billing now"
    assert (await starts.start(folder_in(tmp_path), environment=environment))["error"].startswith(reason)
    assert starts.ran == []


async def test_a_session_is_started_in_the_server_the_tmux_it_was_started_inside_names(tmp_path: Path) -> None:
    starts = Starts(home_in(tmp_path), lambda starts, pane, _command, _environment: starts.joins(pane))
    await starts.start(folder_in(tmp_path), environment={"PATH": "/usr/bin", "TMUX": "/srv/tmux/work,100,0"})
    assert list(starts.fake.servers) == [Path("/srv/tmux/work")]


async def test_a_folder_with_no_name_to_give_its_tmux_session_is_said(tmp_path: Path) -> None:
    starts = Starts(home_in(tmp_path))
    assert (await starts.start(Path("/")))["error"] == "/ has no name to give the tmux session a session there is started in: give a folder inside it"
    assert starts.ran == []
