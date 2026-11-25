"""The user's terminals as tmux and fritter answer for them: the one real adapter of `hands.sessions.terminals.Terminals`,
which runs tmux against a server's socket and asks a fritter at its own.

What hands asks a terminal is decided in `hands.sessions.panes` and `hands.sessions.startsession`; the tmux commands that
ask it, and the stat of each terminal tmux names, are here alone.
"""

import asyncio
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from hands.core.effects import Command, Fritter, Input, Key, Text, Type
from hands.core.session import Keystroke
from hands.core.tmux import Listed, Pane, Server, Unanswered
from hands.sessions import child
from hands.sessions.child import Children
from hands.sessions.terminals import Opened, PaneState
from hands.sessions.typing import Typist, Untyped

# What tmux says, on stderr, for a socket no server listens on any more, as a server that exited leaves behind, and
# for one that is gone; never "error connecting to" for any other reason, as a path too long for a socket.
_NO_SERVER = re.compile(rb"no server running on |error connecting to .* \(No such file or directory\)$")
# How long a server has to answer: one that does not is wedged.
ANSWER_SECONDS = 2.0
# How long tmux has to open a window, which may start its server first: one that does not is wedged.
OPEN_SECONDS = 5.0
# What tmux says when the session a new one would be named for is there already (tmux 3.6).
DUPLICATE = "duplicate session"


@dataclass(frozen=True)
class TmuxExit:
    """How a tmux command exited, and what it wrote."""

    code: int
    out: bytes
    err: bytes


class TmuxTerminals:
    """The user's tmux servers, asked through the tmux on `path`, the PATH it is looked up on at each ask, and the
    fritters that wrapped sessions, asked at their sockets."""

    def __init__(self, children: Children, path: str | None) -> None:
        self._children = children
        self._path = path

    def sockets(self, directory: Path) -> list[Path]:
        return [path for path in directory.glob("*") if path.is_socket()]

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
                try:
                    # A client with no terminal, as one in control mode over a pipe, is on no screen.
                    pairs = [(device(client), device(pane)) for client, pane in (line.split("\t") for line in lines) if client]
                except OSError as error:
                    return Unanswered(f"a terminal of a client of tmux at {socket} could not be read: {error}")
                # Nor is one gone since it was listed, or showing a pane whose terminal is.
                return {client: pane for client, pane in pairs if client is not None and pane is not None}

    async def shown(self, socket: Path, pane: str) -> str | Unanswered:
        tmux = self._tmux()
        if tmux is None:
            return Unanswered(f"no tmux is on the PATH to read pane {pane} of tmux at {socket}")
        return await captured(self._children, tmux, socket, pane)

    async def drawn(self, socket: Path, pane: str) -> str | Unanswered:
        tmux = self._tmux()
        if tmux is None:
            return Unanswered(f"no tmux is on the PATH to read pane {pane} of tmux at {socket}")
        return await _capture(self._children, tmux, socket, pane, ("-p", "-S", "-"))

    async def state(self, socket: Path, pane: str) -> PaneState | Unanswered:
        tmux = self._tmux()
        if tmux is None:
            return Unanswered(f"no tmux is on the PATH to ask tmux at {socket} whether pane {pane} runs")
        # list-panes, since display-message answers for a pane that is gone as though it were there (tmux 3.6).
        match await ran_at(self._children, tmux, socket, ("list-panes", "-t", pane, "-f", f"#{{==:#{{pane_id}},{pane}}}", "-F", "#{pane_dead}")):
            case Unanswered() as unanswered:
                return unanswered
            case TmuxExit(code=code) if code != 0:
                # Its window closed with it, and with the last window, its server.
                return "closed"
            case TmuxExit(out=out) if out.strip() == b"1":
                return "dead"
            case TmuxExit():
                return "running"

    async def open(self, session: str, folder: Path, command: Sequence[str], environment: Mapping[str, str]) -> Opened | Unanswered:
        tmux = self._tmux()
        if tmux is None:
            return Unanswered(f"no tmux is on the PATH to open a window of tmux session {session}")
        printed = ("-d", "-P", "-F", "#{pane_id} #{pane_pid} #{socket_path}")
        # [LAW:no-ambient-temporal-coupling] made or not is what tmux answers the making with, never a look beforehand
        # that a second open could make stale.
        match await self._opening(tmux, environment, "new-session", *printed, "-s", session, "-c", str(folder), *command):
            case TmuxExit(code=0) as made:
                return _opened(True, made)
            case TmuxExit() as refused if not _said(refused).startswith(DUPLICATE):
                return Unanswered(f"tmux could not make session {session}: {_said(refused)}")
            case Unanswered() as unanswered:
                return unanswered
            case TmuxExit():
                pass
        match await self._opening(tmux, environment, "new-window", *printed, "-t", f"={session}:", "-c", str(folder), *command):
            case TmuxExit(code=0) as window:
                return _opened(False, window)
            case TmuxExit() as refused:
                return Unanswered(f"tmux could not open a window in session {session}: {_said(refused)}")
            case Unanswered() as unanswered:
                return unanswered

    async def type(self, typed: Type[Input]) -> None:
        match typed.writer:
            case Fritter(socket=path, pid=pid):
                await asyncio.to_thread(_by_fritter, Typist(path, pid), typed.input)
            case Pane() as pane:
                match await self._typed(pane, typed.input):
                    case Unanswered(reason=reason):
                        raise Untyped(f"tmux did not type into session {typed.session}: {reason}")
                    case None:
                        return

    def _tmux(self) -> str | None:
        return shutil.which("tmux", path=self._path)

    async def _lines(self, socket: Path, arguments: Sequence[str]) -> list[str] | Unanswered:
        """The lines the server at `socket` answered `arguments` with; none from a socket no server listens on."""
        tmux = self._tmux()
        if tmux is None:
            return Unanswered(f"no tmux is on the PATH to ask tmux at {socket}")
        match await ran_at(self._children, tmux, socket, arguments):
            case TmuxExit(code=0, out=out):
                return out.decode(errors="replace").splitlines()
            case TmuxExit(err=err) if _NO_SERVER.match(err):
                return []
            case TmuxExit(err=err):
                return Unanswered(f"tmux at {socket} did not answer {arguments[0]}: {err.decode(errors='replace').strip()}")
            case Unanswered() as unanswered:
                return unanswered

    async def _opening(self, tmux: str, environment: Mapping[str, str], *arguments: str) -> TmuxExit | Unanswered:
        # A server this starts takes `environment` as every window's, so it holds what the caller gave it and nothing else.
        ran = await self._children.run(tmux, *arguments, limit=OPEN_SECONDS, env=environment)
        match ran:
            case child.Ran(ending=child.Exited(code=code), out=out, err=err):
                return TmuxExit(code, out.data, err.data)
            case child.Ran() | child.NeverStarted() | child.Unreaped() | child.Lost():
                return Unanswered(f"tmux {arguments[0]} {child.said(ran)}")

    async def _typed(self, pane: Pane, input: Input) -> Unanswered | None:
        """Type `input` into `pane` as someone at its keyboard would, or say why it was not.

        [LAW:no-ambient-temporal-coupling] one tmux command list, so what is typed and the Return that sends it happen
        together or not at all: no half-typed prompt is left in the input for a retry to double. And an Escape is answered
        only once it has been read alone, so whatever is typed next cannot be read with it as one chord.
        """
        tmux = self._tmux()
        if tmux is None:
            return Unanswered(f"no tmux is on the PATH to type into pane {pane.id} of tmux at {pane.socket}")
        # A buffer of its own per paste, so two sends cannot paste each other's text; loaded from a file, since a command
        # tmux is handed is far shorter than a prompt can be.
        buffer = f"hands-{self._children.clock.uuid()}"
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", prefix="hands-paste-", delete_on_close=False) as pasted:
            pasted.write(_pasted(input))
            pasted.close()
            match await ran_at(self._children, tmux, pane.socket, _typing(pane.id, input, buffer, pasted.name)):
                case TmuxExit(code=0):
                    await self._children.clock.sleep(_quiet(input))
                    return None
                case TmuxExit(err=err):
                    return Unanswered(f"tmux at {pane.socket} did not type into pane {pane.id}: {err.decode(errors='replace').strip()}")
                case Unanswered() as unanswered:
                    return unanswered


def listed(socket: Path, lines: Sequence[str]) -> Server:
    """The panes a server's list-panes answer names, by the device number of each one's terminal."""
    panes = [(tty, Pane(socket, id, session, int(window))) for tty, id, window, session in (line.split("\t", 3) for line in lines)]
    try:
        terminals = [(device(tty), pane) for tty, pane in panes]
    except OSError as error:
        return Unanswered(f"a terminal of a pane of tmux at {socket} could not be read: {error}")
    return Listed({device: pane for device, pane in terminals if device is not None})


def device(tty: str) -> int | None:
    """The device number of a terminal tmux named, of a pane or a client; none for one closed since it was named, which
    is gone and shows or runs nothing."""
    try:
        return os.stat(tty).st_rdev
    except FileNotFoundError:
        return None


def _opened(made: bool, opened: TmuxExit) -> Opened:
    # The socket last: the one field that may hold a space. Every tmux hands runs on names it: #{socket_path} is 2.2's,
    # and the -f filters hands lists panes by are 3.2's.
    pane, pid, socket = opened.out.decode().rstrip("\n").split(" ", 2)
    return Opened(made, pane, int(pid), Path(socket))


def _said(ran: TmuxExit) -> str:
    return ran.err.decode(errors="replace").strip()


def _by_fritter(typist: Typist, input: Input) -> None:
    match input:
        case Text() as typed:
            # A wrapped session runs on Claude Code's own keybindings, where Return sends the prompt.
            typist.type(typed.typed, "enter")
        case Command() as command:
            typist.command(command)
        case Key(key=key):
            typist.press(key)


async def captured(children: Children, tmux: str, socket: Path, pane: str) -> str | Unanswered:
    """Pane `pane` of the server at `socket` as it shows now, read with the `tmux` named, down to its last line drawn on,
    the blank rows below left off, or why it could not be read."""
    return await _capture(children, tmux, socket, pane, ("-p",))


async def styled(children: Children, tmux: str, socket: Path, pane: str) -> str | Unanswered:
    """What `captured` returns, with the attributes each character is drawn in kept as SGR (`-e`): what tells Claude
    Code's dim placeholder from words (hands.core.screen)."""
    return await _capture(children, tmux, socket, pane, ("-p", "-e"))


async def _capture(children: Children, tmux: str, socket: Path, pane: str, how: tuple[str, ...]) -> str | Unanswered:
    # [LAW:no-silent-failure] a server gone since the pane was named is said, never read as a blank screen.
    match await ran_at(children, tmux, socket, ("capture-pane", *how, "-t", pane)):
        case TmuxExit(code=0, out=out):
            return out.decode(errors="replace").rstrip()
        case TmuxExit(err=err):
            return Unanswered(f"tmux at {socket} did not show pane {pane}: {err.decode(errors='replace').strip()}")
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
            return ["send-keys", "-t", pane, KEYS[key]]


# How long an Escape keeps the session's input to itself: fritter's loneEscape (fritter/control.go).
LONE_ESCAPE = 0.1

# [LAW:types-are-the-program] each chord hands speaks, as tmux names its key: every Keystroke has one.
KEYS: Mapping[Keystroke, str] = {
    "escape": "Escape",
    "enter": "Enter",
    "ctrl_c": "C-c",
    "ctrl_e": "C-e",
    "ctrl_h": "C-h",
    "ctrl_k": "C-k",
    "ctrl_u": "C-u",
    "ctrl_q": "C-q",
    "up": "Up",
    "down": "Down",
    "tab": "Tab",
    "shift_tab": "BTab",
}


async def ran_at(children: Children, tmux: str, socket: Path, arguments: Sequence[str]) -> TmuxExit | Unanswered:
    """How the tmux server at `socket` exited from `arguments`, or why it gave no answer: wedged, or `tmux` not run."""
    ran = await children.run(tmux, "-S", str(socket), *arguments, limit=ANSWER_SECONDS)
    match ran:
        case child.Ran(ending=child.Exited(code=code), out=out, err=err):
            return TmuxExit(code, out.data, err.data)
        case child.Ran() | child.NeverStarted() | child.Unreaped() | child.Lost():
            return Unanswered(f"tmux at {socket} did not answer {arguments[0]}: tmux {child.said(ran)}")
