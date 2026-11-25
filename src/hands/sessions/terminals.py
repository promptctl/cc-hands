"""The terminals hands types into and reads past the Mac's processes: the user's tmux servers, the panes each holds and
the clients attached to it (`Terminals`), and the fritters that wrapped sessions (`Fritters`). Each has one real adapter,
`hands.sessions.tmux.TmuxTerminals` and `hands.sessions.typing.FritterSockets`, which only a composition root builds;
every other module is handed the interface."""

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from hands.core.effects import Fritter, Input, Type
from hands.core.tmux import Pane, Server, Unanswered


class Untyped(Exception):
    """Nothing was typed: fritter or tmux could not be reached, or refused, or could not write. The message says which."""


class Unconfirmed(Exception):
    """fritter or tmux was handed the input and never said whether it typed it: it may be in the session. The message
    says why no answer came."""


@dataclass(frozen=True)
class Opened:
    """A window opened in a tmux session: whether the session was made for it, the socket of the server it opened in by
    its real path, as tmux names it, the window's pane (`%12`), and the pid of the process the pane runs."""

    made: bool
    socket: Path
    pane: str
    pid: int


@dataclass(frozen=True)
class Unopened:
    """tmux answered that it opened no window, and why."""

    reason: str


# A pane as its server says it is: running its process, dead (its process ended, the pane kept by remain-on-exit), or
# closed, with its window, since its process ended.
PaneState = Literal["running", "dead", "closed"]


class Terminals(Protocol):
    """The user's tmux servers, as they stand when asked."""

    def sockets(self, directory: Path) -> list[Path]:
        """Every socket in `directory`, where tmux puts the sockets of the servers it starts, by its real path, as tmux
        names a server in each pane's $TMUX and Opened: one a server that exited left behind among them."""
        ...

    async def panes(self, socket: Path) -> Server:
        """The live panes the server at `socket` holds, by the device of each one's terminal: none where no server
        listens; a pane whose terminal closed since it was listed is left out, as it shows and runs nothing."""
        ...

    async def clients(self, socket: Path) -> Mapping[int, int] | Unanswered:
        """Each client attached to the server at `socket` on a terminal, by its terminal's device, to the device of the
        pane it shows: none where no server listens; a client with no terminal, as one in control mode, is on no screen."""
        ...

    async def shown(self, socket: Path, pane: str) -> str | Unanswered:
        """The text pane `pane` of the server at `socket` shows now, down to its last line drawn on: of a dead pane,
        tmux's notice that it is dead, which takes a line of what its program drew off the screen (tmux 3.8)."""
        ...

    async def tail(self, socket: Path, pane: str, lines: int) -> str | Unanswered:
        """The last `lines` lines holding text that pane `pane` of the server at `socket` has drawn, from its history and
        then its screen: what a dead pane's program drew before it ended among them, and none of the empty rows between
        that and the notice tmux writes on the screen's bottom row."""
        ...

    async def state(self, socket: Path, pane: str) -> PaneState | Unanswered:
        """Whether pane `pane` of the server at `socket` still runs its process."""
        ...

    async def open(self, session: str, folder: Path, command: Sequence[str], window: Mapping[str, str], environment: Mapping[str, str]) -> Opened | Unopened | Unanswered:
        """Open a detached window of tmux session `session`, in `folder`, running `command` with the variables of `window`
        set, making the session where there is none, in the server a `tmux` run in `environment` reaches, as tmux picks
        it: the one its $TMUX names, or else its default one. A server this starts takes `environment` as every window's.
        Unanswered where tmux gave no answer, so a window may have opened."""
        ...

    async def type(self, pane: Pane, input: Input) -> None:
        """Type `input` into `pane` as someone at its keyboard would. Raises Untyped saying why nothing was typed, or
        Unconfirmed where tmux gave no answer, so it may have been."""
        ...


class Fritters(Protocol):
    """The fritters that wrapped sessions, each at its control socket."""

    async def type(self, fritter: Fritter, input: Input) -> None:
        """Type `input` into the session `fritter` wrapped, as someone at its keyboard would. Raises Untyped saying why
        nothing was typed, or Unconfirmed where fritter was asked and gave no answer, so it may have been."""
        ...


async def type_into(terminals: Terminals, fritters: Fritters, typed: Type[Input]) -> None:
    """Type `typed`'s input into its session by its writer: the fritter that wrapped it, or the tmux pane it runs in."""
    match typed.writer:
        case Fritter() as fritter:
            await fritters.type(fritter, typed.input)
        case Pane() as pane:
            await terminals.type(pane, typed.input)


def device(tty: str) -> int | None:
    """The device number of the terminal at `tty`, as tmux or an app names one; none for one closed since it was named,
    which is gone and shows or runs nothing. Raises OSError where it could not be read."""
    try:
        return os.stat(tty).st_rdev
    except FileNotFoundError:
        return None
