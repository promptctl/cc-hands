"""Typing into a session through the fritter that wrapped it: the one real adapter of
`hands.sessions.terminals.Fritters`.

fritter runs a session's `claude` on a pseudo-terminal and listens on a unix socket
beside it; text asked for there is typed into that session's input and sent with Return,
as someone at the keyboard would. This is the client side of that socket. The other half
of hands' dependence on fritter is the name `FRITTER_SOCKET`, which
`hands.sessions.shim` reads out of a wrapped session's environment.

It decides nothing about *what* to type. Whether a draft is ready, whether a session can
be sent to, and whether a leading slash is escaped are all settled in the core before anything
reaches here `[LAW:single-enforcer]`.
"""

import asyncio
import json
import socket
from dataclasses import dataclass
from pathlib import Path

from hands.core.effects import Command, Fritter, Input, Key, Text
from hands.core.session import Keystroke, PromptText, Submit
from hands.sessions.terminals import Unconfirmed, Untyped

# How long the exchange may take. fritter answers as soon as its one write is done.
ANSWER_TIMEOUT = 5.0


@dataclass(frozen=True)
class Typist:
    """A session fritter wrapped: where its socket is, and the process fritter wrapped. Which session that process is in
    is the process's to change, by a /clear or a /resume, so a typist names none."""

    socket: Path
    # fritter types only into the process it wrapped, and its address is inherited, so a
    # session started from inside a wrapped one carries its parent's. Every request names this.
    pid: int
    # How long fritter has to answer each ask.
    answer: float = ANSWER_TIMEOUT

    def type(self, text: PromptText, submit: Submit) -> None:
        """Type text into the session's input and press `submit`, the chord that sends it: Return at Claude Code's
        defaults, the brain's own chord for the brain (`hands.brain.process.SUBMIT`)."""
        self._ask({"pid": self.pid, "kind": "text", "text": str(text), "submit": submit})

    def command(self, command: Command) -> None:
        """Type a slash command into the session's input and press Return: the command as keys, and its arguments
        pasted behind it, so a long paste Claude Code folds into a placeholder cannot fold the command in with it."""
        self._ask({"pid": self.pid, "kind": "command", "command": command.word, "text": "" if command.args is None else str(command.args)})

    def press(self, key: Keystroke) -> None:
        """Press one named chord."""
        self._ask({"pid": self.pid, "kind": "key", "key": key})

    def pasting(self) -> None:
        """Returns if the session has turned bracketed paste on, which Claude Code does once its input is up, and raises
        Untyped saying why if it has not. Types nothing."""
        self._ask({"pid": self.pid, "kind": "pasting"})

    def _ask(self, request: dict[str, object]) -> None:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(self.answer)
            try:
                connection.connect(str(self.socket))
                # fritter acts on a whole line: one cut short before its newline is never typed.
                connection.sendall(json.dumps(request).encode() + b"\n")
            except OSError as error:
                raise Untyped(f"cannot talk to the fritter wrapping {self.pid} at {self.socket}: {error}") from error
            try:
                said = connection.makefile("rb").readline()
            except OSError as error:
                # [LAW:no-silent-failure] fritter has the request, and may have typed it before its answer was lost.
                raise Unconfirmed(f"the fritter wrapping {self.pid} at {self.socket} was asked to type and never said whether it did: {error}") from error
        match _answer(said):
            case {"ok": True}:
                return
            case {"ok": False, "reason": str() as reason}:
                raise Untyped(f"the fritter wrapping {self.pid} did not type: {reason}")
            case b"":
                raise Unconfirmed(f"the fritter wrapping {self.pid} at {self.socket} was asked to type and hung up without saying whether it did")
            case other:
                raise Unconfirmed(f"the fritter wrapping {self.pid} at {self.socket} was asked to type and answered {other!r}, which says neither")


def _answer(said: bytes) -> object:
    """What fritter answered: the JSON it said, or what it said where that is none."""
    try:
        return json.loads(said)
    except ValueError:
        return said


class FritterSockets:
    """The fritters that wrapped sessions, each asked at its control socket and given `answer` seconds to answer."""

    def __init__(self, answer: float = ANSWER_TIMEOUT) -> None:
        self._answer = answer

    async def type(self, fritter: Fritter, input: Input) -> None:
        await asyncio.to_thread(_by_fritter, Typist(fritter.socket, fritter.pid, self._answer), input)


def _by_fritter(typist: Typist, input: Input) -> None:
    match input:
        case Text() as typed:
            # A wrapped session runs on Claude Code's own keybindings, where Return sends the prompt.
            typist.type(typed.typed, "enter")
        case Command() as command:
            typist.command(command)
        case Key(key=key):
            typist.press(key)
