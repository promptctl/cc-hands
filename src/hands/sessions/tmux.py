"""The user's tmux servers as tmux answers for them: the one real adapter of `hands.sessions.terminals.Terminals`, which
runs tmux against a server's socket.

What hands asks a terminal is decided in `hands.sessions.panes` and `hands.sessions.startsession`; the tmux commands that
ask it are here alone.
"""

import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from hands.core.effects import Command, Input, Key, Text
from hands.core.session import Keystroke
from hands.core.tmux import Listed, Pane, Server, Unanswered
from hands.sessions import child
from hands.sessions.child import Children
from hands.sessions.terminals import Opened, PaneState, Terminals, Unopened, Untyped, device

# What tmux says, on stderr, for a socket no server listens on any more, as a server that exited leaves behind, and
# for one that is gone; never "error connecting to" for any other reason, as a path too long for a socket.
_NO_SERVER = re.compile(rb"no server running on |error connecting to .* \(No such file or directory\)$")
# What tmux says for a pane it holds none of, as one closed with its window (tmux 3.8).
_NO_PANE = re.compile(rb"can't find pane: ")
# How long a server has to answer: one that does not is wedged.
ANSWER_SECONDS = 2.0
# How long tmux has to open a window, which may start its server first: one that does not is wedged.
OPEN_SECONDS = 5.0
# What tmux says when the session a new one would be named for is there already (tmux 3.6): the contract's run against
# tmux (tests/test_terminals_contract_tmux.py) opens two windows in one session, so a tmux that rewords it fails there.
DUPLICATE = b"duplicate session"
# tmux's -P answer to opening a window, in the format `_PRINTED` asks for.
_PRINTED = "#{pane_id} #{pane_pid}"
_OPENED = re.compile(rb"(?P<pane>%\d+) (?P<pid>\d+)\n?")


@dataclass(frozen=True)
class TmuxExit:
    """How a tmux command exited, and what it wrote."""

    code: int
    out: bytes
    err: bytes


def on_path(children: Children, path: str) -> Terminals:
    """The user's tmux servers, asked through the tmux on `path` as it is found now: [LAW:parse-dont-validate] once, by
    the composition root, which holds the answer as the adapter it builds."""
    match shutil.which("tmux", path=path):
        case str(tmux):
            return TmuxTerminals(children, Path(tmux))
        case None:
            return NoTmux(f"no tmux is on the PATH {path}")


class TmuxTerminals:
    """The user's tmux servers, asked through the tmux at `tmux`."""

    def __init__(self, children: Children, tmux: Path) -> None:
        self._children = children
        self._tmux = tmux

    def sockets(self, directory: Path) -> list[Path]:
        return _sockets(directory)

    async def panes(self, socket: Path) -> Server:
        # A dead pane, kept by remain-on-exit, runs nothing and keeps the name of a terminal that is gone or reused.
        # The session's name last: it is the one field that may hold a tab.
        match await self._lines(socket, ("list-panes", "-a", "-f", "#{?pane_dead,0,1}", "-F", "#{pane_tty}\t#{pane_id}\t#{window_index}\t#{session_name}")):
            case Unanswered() as unanswered:
                return unanswered
            case lines:
                return listed(socket, lines)

    async def clients(self, socket: Path) -> Mapping[int, int] | Unanswered:
        match await self._lines(socket, ("list-clients", "-F", "#{client_tty}\t#{pane_tty}")):
            case Unanswered() as unanswered:
                return unanswered
            case lines:
                # A client with no terminal, as one in control mode over a pipe, is on no screen.
                pairs = [(client, pane) for client, pane in (line.split("\t") for line in lines) if client]
                match _devices(socket, [tty for pair in pairs for tty in pair]):
                    case Unanswered() as unanswered:
                        return unanswered
                    case devices:
                        # Nor is one gone since it was listed, or showing a pane whose terminal is.
                        return {terminal: shown for client, pane in pairs if (terminal := devices[client]) is not None and (shown := devices[pane]) is not None}

    async def shown(self, socket: Path, pane: str) -> str | Unanswered:
        return await _capture(self._children, self._tmux, socket, pane, ("-p",))

    async def tail(self, socket: Path, pane: str, lines: int) -> str | Unanswered:
        # From the start of its history: any number of empty rows may lie under the last `lines` that hold text.
        match await _capture(self._children, self._tmux, socket, pane, ("-p", "-S", "-")):
            case str(drawn):
                return "\n".join([line for line in drawn.split("\n") if line.strip()][-lines:])
            case Unanswered() as unanswered:
                return unanswered

    async def state(self, socket: Path, pane: str) -> PaneState | Unanswered:
        # list-panes, since display-message answers for a pane that is gone as though it were there (tmux 3.6).
        match await ran_at(self._children, self._tmux, socket, ("list-panes", "-t", pane, "-f", f"#{{==:#{{pane_id}},{pane}}}", "-F", "#{pane_dead}")):
            case TmuxExit(code=0, out=b"0\n"):
                return "running"
            case TmuxExit(code=0, out=b"1\n"):
                return "dead"
            case TmuxExit(err=err) if _NO_PANE.match(err) or _NO_SERVER.match(err):
                # Its window closed with it, and with the last window, its server.
                return "closed"
            case TmuxExit() as answered:
                return Unanswered(f"tmux at {socket} did not say whether pane {pane} runs: it answered {answered.out!r}, {_said(answered)}")
            case Unanswered() as unanswered:
                return unanswered

    async def open(self, socket: Path, session: str, folder: Path, command: Sequence[str], environment: Mapping[str, str]) -> Opened | Unopened | Unanswered:
        # The directory tmux makes for its own default socket, where -S names it.
        socket.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        # [LAW:no-ambient-temporal-coupling] made or not is what tmux answers the making with, never a look beforehand
        # that a second open could make stale.
        match await self._opening(socket, environment, "new-session", "-s", session, "-c", str(folder), *command):
            case TmuxExit(code=0) as made:
                return _opened(True, made)
            case TmuxExit(err=err) if err.startswith(DUPLICATE):
                match await self._opening(socket, environment, "new-window", "-t", f"={session}:", "-c", str(folder), *command):
                    case TmuxExit(code=0) as window:
                        return _opened(False, window)
                    case TmuxExit() as refused:
                        return Unopened(f"tmux could not open a window in session {session}: {_said(refused)}")
                    case Unanswered() as unanswered:
                        return unanswered
            case TmuxExit() as refused:
                return Unopened(f"tmux could not make session {session}: {_said(refused)}")
            case Unanswered() as unanswered:
                return unanswered

    async def type(self, pane: Pane, input: Input) -> None:
        """Type `input` into `pane` as someone at its keyboard would.

        [LAW:no-ambient-temporal-coupling] one tmux command list, so what is typed and the Return that sends it happen
        together or not at all: no half-typed prompt is left in the input for a retry to double. And an Escape is answered
        only once it has been read alone, so whatever is typed next cannot be read with it as one chord.
        """
        # A buffer of its own per paste, so two sends cannot paste each other's text; loaded from a file, since a command
        # tmux is handed is far shorter than a prompt can be.
        buffer = f"hands-{self._children.clock.uuid()}"
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", prefix="hands-paste-", delete_on_close=False) as pasted:
            pasted.write(_pasted(input))
            pasted.close()
            match await ran_at(self._children, self._tmux, pane.socket, _typing(pane.id, input, buffer, pasted.name)):
                case TmuxExit(code=0):
                    await self._children.clock.sleep(_quiet(input))
                case TmuxExit() as refused:
                    raise Untyped(f"tmux at {pane.socket} did not type into pane {pane.id}: {_said(refused)}")
                case Unanswered(reason=reason):
                    raise Untyped(f"tmux at {pane.socket} did not type into pane {pane.id}: {reason}")

    async def _lines(self, socket: Path, arguments: Sequence[str]) -> list[str] | Unanswered:
        """The lines the server at `socket` answered `arguments` with; none from a socket no server listens on."""
        match await ran_at(self._children, self._tmux, socket, arguments):
            case TmuxExit(code=0, out=out):
                # By newline alone: a session's name, which is a folder's, may hold any other line separator.
                return [line for line in out.decode(errors="replace").split("\n") if line]
            case TmuxExit(err=err) if _NO_SERVER.match(err):
                return []
            case TmuxExit() as refused:
                return Unanswered(f"tmux at {socket} did not answer {arguments[0]}: {_said(refused)}")
            case Unanswered() as unanswered:
                return unanswered

    async def _opening(self, socket: Path, environment: Mapping[str, str], *arguments: str) -> TmuxExit | Unanswered:
        # A server this starts takes `environment` as every window's, so it holds what the caller gave it and nothing else.
        return await ran_at(self._children, self._tmux, socket, (arguments[0], "-d", "-P", "-F", _PRINTED, *arguments[1:]), OPEN_SECONDS, environment)


class NoTmux:
    """The user's tmux servers where no tmux is found to ask them: each ask answers `reason`."""

    def __init__(self, reason: str) -> None:
        self._reason = reason

    def sockets(self, directory: Path) -> list[Path]:
        return _sockets(directory)

    async def panes(self, socket: Path) -> Server:
        return Unanswered(self._reason)

    async def clients(self, socket: Path) -> Mapping[int, int] | Unanswered:
        return Unanswered(self._reason)

    async def shown(self, socket: Path, pane: str) -> str | Unanswered:
        return Unanswered(self._reason)

    async def tail(self, socket: Path, pane: str, lines: int) -> str | Unanswered:
        return Unanswered(self._reason)

    async def state(self, socket: Path, pane: str) -> PaneState | Unanswered:
        return Unanswered(self._reason)

    async def open(self, socket: Path, session: str, folder: Path, command: Sequence[str], environment: Mapping[str, str]) -> Opened | Unopened | Unanswered:
        return Unopened(self._reason)

    async def type(self, pane: Pane, input: Input) -> None:
        raise Untyped(self._reason)


def listed(socket: Path, lines: Sequence[str]) -> Server:
    """The panes a server's list-panes answer names, by the device number of each one's terminal."""
    panes = [(tty, Pane(socket, id, session, int(window))) for tty, id, window, session in (line.split("\t", 3) for line in lines)]
    match _devices(socket, [tty for tty, _ in panes]):
        case Unanswered() as unanswered:
            return unanswered
        case devices:
            return Listed({terminal: pane for tty, pane in panes if (terminal := devices[tty]) is not None})


def _sockets(directory: Path) -> list[Path]:
    return [path for path in directory.glob("*") if path.is_socket()]


def _devices(socket: Path, ttys: Sequence[str]) -> dict[str, int | None] | Unanswered:
    """The device of each terminal the server at `socket` named in `ttys`, none for one closed since; or why one could not
    be read."""
    try:
        return {tty: device(tty) for tty in ttys}
    except OSError as error:
        return Unanswered(f"a terminal tmux at {socket} named could not be read: {error}")


def _opened(made: bool, opened: TmuxExit) -> Opened | Unanswered:
    match _OPENED.fullmatch(opened.out):
        case re.Match() as printed:
            return Opened(made, printed["pane"].decode(), int(printed["pid"]))
        case None:
            # [LAW:parse-dont-validate] tmux opened something it did not name: a window may be running.
            return Unanswered(f"tmux opened a window and named it {opened.out!r}, which is no pane and pid")


def _said(ran: TmuxExit) -> str:
    return ran.err.decode(errors="replace").strip()


async def styled(children: Children, tmux: Path, socket: Path, pane: str) -> str | Unanswered:
    """What pane `pane` of the server at `socket` shows, with the attributes each character is drawn in kept as SGR
    (`-e`): what tells Claude Code's dim placeholder from words (hands.core.screen)."""
    return await _capture(children, tmux, socket, pane, ("-p", "-e"))


async def _capture(children: Children, tmux: Path, socket: Path, pane: str, how: tuple[str, ...]) -> str | Unanswered:
    # [LAW:no-silent-failure] a server gone since the pane was named is said, never read as a blank screen.
    match await ran_at(children, tmux, socket, ("capture-pane", *how, "-t", pane)):
        case TmuxExit(code=0, out=out):
            return out.decode(errors="replace").rstrip()
        case TmuxExit() as refused:
            return Unanswered(f"tmux at {socket} did not show pane {pane}: {_said(refused)}")
        case Unanswered() as unanswered:
            return unanswered


def _pasted(input: Input) -> str:
    """What of `input` is pasted rather than pressed: a prompt, a command's arguments, nothing of a key."""
    match input:
        case Text() as text:
            return text.typed
        case Command(args=str() as args):
            return args
        case Command() | Key():
            return ""


def _quiet(input: Input) -> float:
    """How long the keys of `input` keep the session's input to themselves: an Escape until it is read alone."""
    match input:
        case Key(key="escape"):
            return LONE_ESCAPE
        case Text() | Command() | Key():
            return 0.0


def _typing(pane: str, input: Input, buffer: str, pasted: str) -> list[str]:
    """The tmux command list that types `input` into `pane`, the file `pasted` holding what of it is pasted.

    Something is typed into the pane before anything is loaded - a command's name, or nothing ahead of a prompt - so a
    pane gone since it was read stops the list before the prompt sits in a buffer that only a paste would delete. A
    paste is bracketed (-p), so its newlines stay in the prompt rather than send it; pasted as they are rather than as
    carriage returns (-r); and its buffer is deleted once pasted (-d). A command's name and the space behind it are
    pressed as keys and its arguments pasted after them, as fritter types one, so a long paste Claude Code folds into a
    placeholder cannot fold the command in with it.
    """
    paste = ["load-buffer", "-b", buffer, pasted, ";", "paste-buffer", "-p", "-r", "-d", "-b", buffer, "-t", pane, ";"]
    enter = ["send-keys", "-t", pane, "Enter"]
    match input:
        case Text():
            return ["send-keys", "-t", pane, "-l", "", ";", *paste, *enter]
        case Command(args=None) as command:
            return ["send-keys", "-t", pane, "-l", command.word, ";", *enter]
        case Command() as command:
            return ["send-keys", "-t", pane, "-l", f"{command.word} ", ";", *paste, *enter]
        case Key(key=key):
            return ["send-keys", "-t", pane, tmux_key(key)]


# How long an Escape keeps the session's input to itself: fritter's loneEscape (fritter/control.go).
LONE_ESCAPE = 0.1


def tmux_key(key: Keystroke) -> str:
    """[LAW:types-are-the-program] each chord hands speaks, as tmux names its key: pyright holds every Keystroke to one."""
    match key:
        case "escape":
            return "Escape"
        case "enter":
            return "Enter"
        case "ctrl_c":
            return "C-c"
        case "ctrl_e":
            return "C-e"
        case "ctrl_h":
            return "C-h"
        case "ctrl_k":
            return "C-k"
        case "ctrl_u":
            return "C-u"
        case "ctrl_q":
            return "C-q"
        case "up":
            return "Up"
        case "down":
            return "Down"
        case "tab":
            return "Tab"
        case "shift_tab":
            return "BTab"


async def ran_at(children: Children, tmux: Path, socket: Path, arguments: Sequence[str], limit: float = ANSWER_SECONDS, environment: Mapping[str, str] | None = None) -> TmuxExit | Unanswered:
    """How the tmux server at `socket` exited from `arguments`, or why it gave no answer within `limit`: wedged, or
    `tmux` not run. `environment` is the one tmux runs in, the daemon's where none is given."""
    ran = await children.run(str(tmux), "-S", str(socket), *arguments, limit=limit, env=environment)
    match ran:
        case child.Ran(ending=child.Exited(code=code), out=out, err=err):
            return TmuxExit(code, out.data, err.data)
        case child.Ran() | child.NeverStarted() | child.Unreaped() | child.Lost():
            return Unanswered(f"tmux at {socket} did not answer {arguments[0]}: tmux {child.said(ran)}")
