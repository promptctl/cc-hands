"""The user's terminals as values, for a test to hand to anything that takes `Terminals` or `Fritters`, and the Mac's screen,
for one that takes `Screen`: tmux servers by socket, each with its panes and the clients attached to it; the fritters that
wrapped sessions; and what a session's program read of what was typed into it, as tmux and fritter type it
(tests/terminalscontract.py holds the fake to both)."""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path

from hands.core.effects import Command, Fritter, Input, Key, Text
from hands.core.front import FrontUnread
from hands.core.session import Keystroke
from hands.core.tmux import Listed, Pane, Server, Unanswered
from hands.sessions.screen import FrontApp, FrontTab
from hands.sessions.terminals import Opened, PaneState, Unopened, Untyped

# What the default server of a test listens at: tmux's own default socket of the uid fakeprocesses' user is.
SOCKET = Path("/tmp/tmux-501/default")
# How a program that asked for bracketed paste reads a paste, as tmux and fritter bracket it.
PASTE_START, PASTE_END = "\x1b[200~", "\x1b[201~"
# How tmux 3.8 begins the notice it writes on a dead pane's screen: "Pane is dead (signal term, <when>)".
DEAD = "Pane is dead"
# The rows of a pane's screen, as tmux sizes a detached session's window.
ROWS = 24


@dataclass
class FakePane:
    """A pane of a fake tmux server: its id, the device of its terminal, the tmux session and window it is in, the pid of
    the process it runs, what its screen shows and what scrolled off it into history, whether its process ended and the
    pane was kept (`dead`), and what its program read of what was typed into it."""

    id: str
    terminal: int
    session: str
    window: int = 0
    pid: int = 0
    screen: str = ""
    history: str = ""
    dead: bool = False
    read: str = ""

    def end(self) -> None:
        """Its program ends and the pane is kept, as remain-on-exit keeps one in tmux 3.8: the screen, ROWS rows, scrolls
        a row into history, and tmux writes its notice on the bottom row, under every row the program left empty."""
        top, *rest = [*(rows := self.screen.split("\n")), *[""] * (ROWS - len(rows))]
        self.history = "\n".join(part for part in (self.history, top) if part)
        self.screen = "\n".join([*rest, f"{DEAD} (signal kill)"])
        self.dead = True


@dataclass(frozen=True)
class Answering:
    """A server that answers what it is asked."""


@dataclass(frozen=True)
class Refusing:
    """A server whose socket refuses tmux a connection, and what tmux says of it."""

    said: str


@dataclass(frozen=True)
class Wedged:
    """A server that answers nothing, and what tmux's wait for it ended with."""

    said: str


@dataclass
class FakeServer:
    """A fake tmux server: its panes; each client attached to it on a terminal, by its terminal's device, to the id of
    the pane it shows; and whether it answers."""

    panes: list[FakePane] = field(default_factory=list[FakePane])
    clients: dict[int, str] = field(default_factory=dict[int, str])
    answering: Answering | Refusing | Wedged = Answering()

    def pane(self, id: str) -> FakePane | None:
        return next((pane for pane in self.panes if pane.id == id), None)

    def close(self, pane: FakePane) -> None:
        """`pane`'s program ends and its window closes with it."""
        self.panes.remove(pane)


def ignored(pane: FakePane, command: Sequence[str], environment: Mapping[str, str]) -> None:
    pass


class FakeTerminals:
    """Terminals that hold `servers` by socket and the sockets of servers that exited (`left`). A window opened is a new
    pane of the server at the socket it is opened at, handed to `opened` with what it runs, for a test to put its process
    in a fake table."""

    def __init__(self, servers: Mapping[Path, FakeServer] | None = None, left: Iterable[Path] = (), opened: Callable[[FakePane, Sequence[str], Mapping[str, str]], None] = ignored) -> None:
        self.servers = dict(servers or {})
        self.left = list(left)
        self._opened = opened
        self._ids = count(max((int(pane.id[1:]) + 1 for server in self.servers.values() for pane in server.panes), default=0))
        self._ttys = count(0x10000100)
        self._pids = count(7000)

    def sockets(self, directory: Path) -> list[Path]:
        return [socket for socket in [*self.servers, *self.left] if socket.parent == directory]

    async def panes(self, socket: Path) -> Server:
        match self._asked(socket, "list-panes"):
            case Unanswered() as unanswered:
                return unanswered
            case None:
                return Listed({})
            case FakeServer(panes=panes):
                return Listed({pane.terminal: Pane(socket, pane.id, pane.session, pane.window) for pane in panes if not pane.dead})

    async def clients(self, socket: Path) -> Mapping[int, int] | Unanswered:
        match self._asked(socket, "list-clients"):
            case Unanswered() as unanswered:
                return unanswered
            case None:
                return {}
            case FakeServer() as server:
                return {client: shown.terminal for client, id in server.clients.items() if (shown := server.pane(id)) is not None}

    async def shown(self, socket: Path, pane: str) -> str | Unanswered:
        match self._pane(socket, pane, "did not show pane"):
            case FakePane(screen=screen):
                return screen.rstrip()
            case Unanswered() as unanswered:
                return unanswered

    async def tail(self, socket: Path, pane: str, lines: int) -> str | Unanswered:
        match self._pane(socket, pane, "did not show pane"):
            case FakePane(history=history, screen=screen):
                return "\n".join([line for part in (history, screen) for line in part.split("\n") if line.strip()][-lines:])
            case Unanswered() as unanswered:
                return unanswered

    async def state(self, socket: Path, pane: str) -> PaneState | Unanswered:
        match self._asked(socket, "list-panes"):
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

    async def open(self, socket: Path, session: str, folder: Path, command: Sequence[str], environment: Mapping[str, str]) -> Opened | Unopened | Unanswered:
        server = self.servers.setdefault(socket, FakeServer())
        match server.answering:
            case Refusing(said=said):
                return Unopened(f"tmux could not make session {session}: {said}")
            case Wedged(said=said):
                return Unanswered(f"tmux at {socket} did not answer new-session: tmux {said}")
            case Answering():
                windows = [pane.window for pane in server.panes if pane.session == session]
                pane = FakePane(f"%{next(self._ids)}", next(self._ttys), session, max(windows, default=-1) + 1, next(self._pids))
                server.panes.append(pane)
                self._opened(pane, command, environment)
                return Opened(not windows, pane.id, pane.pid)

    async def type(self, pane: Pane, input: Input) -> None:
        match self._pane(pane.socket, pane.id, "did not type into pane"):
            case Unanswered(reason=reason):
                raise Untyped(reason)
            case FakePane() as typed:
                typed.read += keyed(input)

    def _asked(self, socket: Path, command: str) -> FakeServer | Unanswered | None:
        """The server at `socket`, none where none listens, or why it did not answer `command`."""
        match self.servers.get(socket):
            case FakeServer(answering=Refusing(said=said) | Wedged(said=said)):
                return Unanswered(f"tmux at {socket} did not answer {command}: {said}")
            case server:
                return server

    def _pane(self, socket: Path, id: str, asked: str) -> FakePane | Unanswered:
        match self._asked(socket, "capture-pane"):
            case Unanswered(reason=reason):
                return Unanswered(f"tmux at {socket} {asked} {id}: {reason}")
            case None:
                return Unanswered(f"tmux at {socket} {asked} {id}: no server running on {socket}")
            case FakeServer() as server:
                match server.pane(id):
                    case FakePane() as pane:
                        return pane
                    case None:
                        return Unanswered(f"tmux at {socket} {asked} {id}: can't find pane: {id}")


class FakeFritters:
    """Fritters listening at the sockets of `listening`, each typing into the pid it wrapped, and what each one's program
    read of what was typed into it, by its socket."""

    def __init__(self, listening: Mapping[Path, int] | None = None) -> None:
        self.listening = dict(listening or {})
        self.read: dict[Path, str] = {}

    async def type(self, fritter: Fritter, input: Input) -> None:
        if self.listening.get(fritter.socket) != fritter.pid:
            raise Untyped(f"cannot talk to the fritter wrapping {fritter.pid} at {fritter.socket}: [Errno 2] No such file or directory")
        self.read[fritter.socket] = self.read.get(fritter.socket, "") + keyed(input)


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
            return chord(key)


def chord(key: Keystroke) -> str:
    """The bytes a program on a terminal in raw mode reads `key` as: pyright holds every Keystroke to one."""
    match key:
        case "escape":
            return "\x1b"
        case "enter":
            return "\r"
        case "ctrl_c":
            return "\x03"
        case "ctrl_e":
            return "\x05"
        case "ctrl_h":
            return "\x08"
        case "ctrl_k":
            return "\x0b"
        case "ctrl_u":
            return "\x15"
        case "ctrl_q":
            return "\x11"
        case "up":
            return "\x1b[A"
        case "down":
            return "\x1b[B"
        case "tab":
            return "\t"
        case "shift_tab":
            return "\x1b[Z"


@dataclass
class FakeScreen:
    """A screen that answers `front` when asked."""

    front_app: FrontTab | FrontApp | FrontUnread

    async def front(self) -> FrontTab | FrontApp | FrontUnread:
        return self.front_app
