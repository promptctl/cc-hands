"""The user's terminals as values, for a test to hand to anything that takes `Terminals`, and the Mac's screen, for one that
takes `Screen`: tmux servers by socket, each with its panes and the clients attached to it; the fritters that wrapped
sessions; and what a session's program read of what was typed into it, as tmux and fritter type it
(tests/terminalscontract.py holds the fake to both)."""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path

from hands.core.effects import Command, Fritter, Input, Key, Text, Type
from hands.core.front import FrontUnread
from hands.core.session import Keystroke
from hands.core.tmux import Listed, Pane, Server, Unanswered
from hands.sessions.screen import FrontApp
from hands.sessions.terminals import Opened, PaneState
from hands.sessions.typing import Untyped

# What the default server of a test listens at: tmux's own default socket of the uid fakeprocesses' user is.
SOCKET = Path("/tmp/tmux-501/default")
# How a program that asked for bracketed paste reads a paste, as tmux and fritter bracket it.
PASTE_START, PASTE_END = "\x1b[200~", "\x1b[201~"
# The bytes each chord reaches a program on a terminal in raw mode as.
CHORDS: Mapping[Keystroke, str] = {
    "escape": "\x1b", "enter": "\r", "ctrl_c": "\x03", "ctrl_e": "\x05", "ctrl_h": "\x08", "ctrl_k": "\x0b", "ctrl_u": "\x15", "ctrl_q": "\x11",
    "up": "\x1b[A", "down": "\x1b[B", "tab": "\t", "shift_tab": "\x1b[Z",
}


@dataclass
class FakePane:
    """A pane of a fake tmux server: its id, the device of its terminal, the tmux session and window it is in, the pid of
    the process it runs, whether that process ended and the pane was kept (`dead`), what it shows, and what its program
    read of what was typed into it."""

    id: str
    terminal: int
    session: str
    window: int = 0
    pid: int = 0
    dead: bool = False
    screen: str = ""
    read: str = ""


@dataclass
class FakeServer:
    """A fake tmux server: its panes; each client attached to it on a terminal, by its terminal's device, to the id of
    the pane it shows; and why it does not answer, for one that is wedged."""

    panes: list[FakePane] = field(default_factory=list[FakePane])
    clients: dict[int, str] = field(default_factory=dict[int, str])
    wedged: str | None = None

    def pane(self, id: str) -> FakePane | None:
        return next((pane for pane in self.panes if pane.id == id), None)


def ignored(pane: FakePane, command: Sequence[str], environment: Mapping[str, str]) -> None:
    pass


class FakeTerminals:
    """Terminals that hold `servers` by socket and the sockets of servers that exited (`left`); `fritters`, the wrapped
    pid each fritter's socket types into; and with `tmux` False, no tmux to ask. A window opened is a new pane of the
    server at `opens_at`, handed to `opened` with what it runs, for a test to put its process in a fake table."""

    def __init__(
        self,
        servers: Mapping[Path, FakeServer] | None = None,
        left: Iterable[Path] = (),
        fritters: Mapping[Path, int] | None = None,
        tmux: bool = True,
        opens_at: Path = SOCKET,
        opened: Callable[[FakePane, Sequence[str], Mapping[str, str]], None] = ignored,
    ) -> None:
        self.servers = dict(servers or {})
        self.left = list(left)
        self.fritters = dict(fritters or {})
        # What each fritter's program read of what was typed into it, by its socket.
        self.read: dict[Path, str] = {}
        # Each Type typed, in order.
        self.typed: list[Type[Input]] = []
        self._tmux = tmux
        self._opens_at = opens_at
        self._opened = opened
        self._ids = count(max((int(pane.id[1:]) + 1 for server in self.servers.values() for pane in server.panes), default=0))
        self._ttys = count(0x10000100)
        self._pids = count(7000)

    def sockets(self, directory: Path) -> list[Path]:
        return [socket for socket in [*self.servers, *self.left] if socket.parent == directory]

    async def panes(self, socket: Path) -> Server:
        match self._asked(socket):
            case Unanswered() as unanswered:
                return unanswered
            case None:
                return Listed({})
            case FakeServer(panes=panes):
                return Listed({pane.terminal: Pane(socket, pane.id, pane.session, pane.window) for pane in panes if not pane.dead})

    async def clients(self, socket: Path) -> Mapping[int, int] | Unanswered:
        match self._asked(socket):
            case Unanswered() as unanswered:
                return unanswered
            case None:
                return {}
            case FakeServer() as server:
                return {client: shown.terminal for client, id in server.clients.items() if (shown := server.pane(id)) is not None}

    async def shown(self, socket: Path, pane: str) -> str | Unanswered:
        match self._pane(socket, pane):
            case FakePane(screen=screen):
                return screen
            case Unanswered() as unanswered:
                return Unanswered(f"tmux at {socket} did not show pane {pane}: {unanswered.reason}")

    async def drawn(self, socket: Path, pane: str) -> str | Unanswered:
        return await self.shown(socket, pane)

    async def state(self, socket: Path, pane: str) -> PaneState | Unanswered:
        match self._asked(socket):
            case Unanswered() as unanswered:
                return unanswered
            case None:
                return "closed"
            case FakeServer() as server:
                match server.pane(pane):
                    case None:
                        return "closed"
                    case FakePane(dead=dead):
                        return "dead" if dead else "running"

    async def open(self, session: str, folder: Path, command: Sequence[str], environment: Mapping[str, str]) -> Opened | Unanswered:
        if not self._tmux:
            return Unanswered(f"no tmux is on the PATH to open a window of tmux session {session}")
        server = self.servers.setdefault(self._opens_at, FakeServer())
        if server.wedged is not None:
            return Unanswered(f"tmux new-session {server.wedged}")
        windows = [pane.window for pane in server.panes if pane.session == session]
        pane = FakePane(f"%{next(self._ids)}", next(self._ttys), session, max(windows, default=-1) + 1, next(self._pids))
        server.panes.append(pane)
        self._opened(pane, command, environment)
        return Opened(not windows, pane.id, pane.pid, self._opens_at)

    async def type(self, typed: Type[Input]) -> None:
        match typed.writer:
            case Fritter(socket=socket, pid=pid):
                if self.fritters.get(socket) != pid:
                    raise Untyped(f"cannot talk to the fritter wrapping {pid} at {socket}: [Errno 2] No such file or directory")
                self.read[socket] = self.read.get(socket, "") + keyed(typed.input)
            case Pane(socket=socket, id=id):
                match self._pane(socket, id):
                    case Unanswered(reason=reason):
                        raise Untyped(f"tmux did not type into session {typed.session}: tmux at {socket} did not type into pane {id}: {reason}")
                    case FakePane() as pane:
                        pane.read += keyed(typed.input)
        self.typed.append(typed)

    def _asked(self, socket: Path) -> FakeServer | Unanswered | None:
        """The server at `socket`, none where none listens, or why it did not answer."""
        if not self._tmux:
            return Unanswered(f"no tmux is on the PATH to ask tmux at {socket}")
        match self.servers.get(socket):
            case FakeServer(wedged=str(wedged)):
                return Unanswered(f"tmux at {socket} did not answer: tmux {wedged}")
            case server:
                return server

    def _pane(self, socket: Path, id: str) -> FakePane | Unanswered:
        match self._asked(socket):
            case Unanswered() as unanswered:
                return unanswered
            case None:
                return Unanswered(f"no server running on {socket}")
            case FakeServer() as server:
                match server.pane(id):
                    case FakePane() as pane:
                        return pane
                    case None:
                        return Unanswered(f"can't find pane: {id}")


def keyed(input: Input) -> str:
    """What a program that asked for bracketed paste reads of `input` typed into it: a prompt pasted and sent with Return,
    a command's name pressed and its arguments pasted behind it, and a chord as its bytes."""
    match input:
        case Text() as text:
            return f"{PASTE_START}{text.typed}{PASTE_END}\r"
        case Command(args=None) as command:
            return f"{command.word}\r"
        case Command(args=args) as command:
            return f"{command.word} {PASTE_START}{args}{PASTE_END}\r"
        case Key(key=key):
            return CHORDS[key]


@dataclass
class FakeScreen:
    """A screen that answers `front` when asked."""

    front_app: FrontApp | FrontUnread

    async def front(self) -> FrontApp | FrontUnread:
        return self.front_app
