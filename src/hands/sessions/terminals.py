"""The terminals hands reaches past the Mac's processes: the user's tmux servers, the panes each holds and the clients
attached to it, and the fritters that wrapped sessions. One real adapter, `hands.sessions.tmux.TmuxTerminals`, which only
a composition root builds; every other module is handed a `Terminals`."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from hands.core.effects import Input, Type
from hands.core.tmux import Server, Unanswered


@dataclass(frozen=True)
class Opened:
    """A window opened in a tmux session: whether the session was made for it, the window's pane (`%12`), the pid of the
    process the pane runs, and the socket of the server holding it."""

    made: bool
    pane: str
    pid: int
    socket: Path


# A pane as its server says it is: running its process, dead (its process ended, the pane kept by remain-on-exit), or
# closed, with its window, since its process ended.
PaneState = Literal["running", "dead", "closed"]


class Terminals(Protocol):
    """The user's tmux servers, and the fritters that wrapped sessions, as they stand when asked."""

    def sockets(self, directory: Path) -> list[Path]:
        """Every socket in `directory`, where tmux puts the sockets of the servers it starts: one a server that exited
        left behind among them."""
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
        """The text pane `pane` of the server at `socket` shows now, down to its last line drawn on."""
        ...

    async def drawn(self, socket: Path, pane: str) -> str | Unanswered:
        """Everything pane `pane` of the server at `socket` has drawn that its server keeps, its history and then its
        screen, down to its last line drawn on: what a dead pane's program drew before it ended, which tmux moves off the
        pane's screen for its notice that the pane is dead (tmux 3.8)."""
        ...

    async def state(self, socket: Path, pane: str) -> PaneState | Unanswered:
        """Whether pane `pane` of the server at `socket` still runs its process."""
        ...

    async def open(self, session: str, folder: Path, command: Sequence[str], environment: Mapping[str, str]) -> Opened | Unanswered:
        """Open a detached window of tmux session `session`, in `folder`, running `command`, making the session where
        there is none: a server this starts takes `environment` as every window's."""
        ...

    async def type(self, typed: Type[Input]) -> None:
        """Type `typed`'s input into its session, by its writer, as someone at the session's keyboard would. Raises
        hands.sessions.typing.Untyped saying why it was not typed."""
        ...
