"""What hands relies on of the user's terminals, as one contract held to two parties: the fake terminals
(tests/faketerminals.py), in `make check` (test_terminals_contract.py), and tmux and fritter themselves through
`hands.sessions.tmux.TmuxTerminals`, on a tmux server of the test's own and the fritter this checkout carries
(test_terminals_contract_tmux.py, outside). The cases are written once, here, and each of those modules runs all of them
against its own `world`, so each asserts the same of both: that is what holds the fake to tmux and fritter. A claim they no
longer keep fails the outside run, and the fake is changed to match."""

import re
from pathlib import Path
from typing import Protocol

import pytest

from faketerminals import keyed
from hands.core.effects import Command, Fritter, Input, Key, Text, Type, Writer
from hands.core.session import CommandName, PromptText, SessionId
from hands.core.tmux import Listed, Pane, Unanswered
from hands.sessions.terminals import Opened, Terminals
from hands.sessions.typing import Untyped

SESSION = SessionId("0f1e2d3c-aaaa-bbbb-cccc-000000000002")
# What each window's program draws once it is up.
DRAWN = "hello from the pane"
# Each kind of input a session is typed: a prompt, its newline kept in it; a command, with arguments and without; a chord.
INPUTS = [
    Text(PromptText("first\nsecond")),
    Command(CommandName("model"), PromptText("opus")),
    Command(CommandName("clear"), None),
    Key("escape"),
]


class World(Protocol):
    """A tmux server of the world's own, the windows the contract opens in it, each running a program that draws DRAWN,
    asks its terminal for bracketed paste, and reads what is typed into it raw; and fritters wrapping such a program."""

    @property
    def terminals(self) -> Terminals: ...

    @property
    def socket(self) -> Path:
        """The socket of the server `window` opens windows at, by its real path."""
        ...

    async def window(self, session: str) -> Opened:
        """A window opened in tmux session `session` through `terminals`, once its program is up."""
        ...

    def pid(self, window: Opened) -> int:
        """The pid of the program `window` runs."""
        ...

    def device(self, window: Opened) -> int:
        """The device of the terminal `window`'s program runs on."""
        ...

    async def end(self, window: Opened, kept: bool) -> None:
        """End `window`'s program, its pane kept by remain-on-exit where `kept`; returns once its server has seen it end."""
        ...

    async def attach(self, window: Opened) -> int:
        """A client attached to the server, showing `window`: the device of its terminal."""
        ...

    async def stop(self) -> None:
        """The server exits as one killed does, leaving its socket behind."""
        ...

    async def wrapped(self) -> Fritter:
        """A fritter wrapping such a program, in a window of its own, once the program has asked for bracketed paste."""
        ...

    def unheard(self) -> Fritter:
        """A fritter's address that nobody listens at, as a session that ended leaves."""
        ...

    async def reads(self, writer: Writer, expected: str) -> str:
        """What the program `writer` types into read, once it has read `expected` or has read for long enough."""
        ...


class TerminalsContract:
    async def test_a_window_opened_makes_its_session_once_and_runs_its_program_in_a_pane_its_server_lists(self, world: World) -> None:
        first, second = await world.window("work"), await world.window("work")
        assert (first.made, second.made) == (True, False)
        assert first.socket == second.socket == world.socket
        assert [first.pid, second.pid] == [world.pid(first), world.pid(second)]
        assert world.socket in world.terminals.sockets(world.socket.parent)
        assert await world.terminals.panes(world.socket) == Listed({world.device(first): Pane(world.socket, first.pane, "work", 0), world.device(second): Pane(world.socket, second.pane, "work", 1)})

    async def test_a_pane_shows_what_its_program_drew(self, world: World) -> None:
        window = await world.window("work")
        assert await world.terminals.shown(world.socket, window.pane) == DRAWN

    async def test_a_pane_runs_until_its_program_ends_then_is_dead_where_kept_and_closed_where_not(self, world: World) -> None:
        # A third window, so its server outlives the two that end.
        kept, closed, running = await world.window("work"), await world.window("work"), await world.window("work")
        assert [await world.terminals.state(world.socket, window.pane) for window in (kept, closed)] == ["running", "running"]
        await world.end(kept, kept=True)
        await world.end(closed, kept=False)
        assert [await world.terminals.state(world.socket, window.pane) for window in (kept, closed)] == ["dead", "closed"]
        # A dead pane runs nothing and is listed as none, and keeps what its program drew before it ended.
        match await world.terminals.panes(world.socket):
            case Listed(panes=panes):
                assert {pane.id for pane in panes.values()} == {running.pane}
            case other:
                pytest.fail(f"listed as {other}")
        match await world.terminals.drawn(world.socket, kept.pane):
            case str(drawn):
                assert drawn.startswith(DRAWN)
            case Unanswered() as unanswered:
                pytest.fail(f"drawn unread: {unanswered}")
        assert isinstance(await world.terminals.shown(world.socket, closed.pane), Unanswered)

    async def test_a_server_killed_leaves_its_socket_behind_holding_no_panes_and_no_clients(self, world: World) -> None:
        window = await world.window("work")
        await world.attach(window)
        await world.stop()
        assert world.socket in world.terminals.sockets(world.socket.parent)
        assert (await world.terminals.panes(world.socket), await world.terminals.clients(world.socket)) == (Listed({}), {})
        assert await world.terminals.state(world.socket, window.pane) == "closed"

    async def test_a_client_is_listed_by_its_terminal_with_the_pane_it_shows(self, world: World) -> None:
        _, shown = await world.window("work"), await world.window("work")
        client = await world.attach(shown)
        assert await world.terminals.clients(world.socket) == {client: world.device(shown)}

    @pytest.mark.parametrize("input", INPUTS, ids=["text", "command", "bare command", "key"])
    async def test_each_input_typed_into_a_pane_reaches_its_program_as_its_keys(self, world: World, input: Input) -> None:
        window = await world.window("work")
        pane = Pane(world.socket, window.pane, "work", 0)
        await world.terminals.type(Type(SESSION, pane, input))
        assert await world.reads(pane, keyed(input)) == keyed(input)

    async def test_typing_into_a_pane_that_closed_raises_untyped_naming_the_pane(self, world: World) -> None:
        window, _ = await world.window("work"), await world.window("work")
        await world.end(window, kept=False)
        with pytest.raises(Untyped, match=re.escape(f"did not type into pane {window.pane}")):
            await world.terminals.type(Type(SESSION, Pane(world.socket, window.pane, "work", 0), Text(PromptText("hello"))))

    @pytest.mark.parametrize("input", INPUTS, ids=["text", "command", "bare command", "key"])
    async def test_each_input_typed_into_a_fritter_reaches_the_program_it_wrapped_as_its_keys(self, world: World, input: Input) -> None:
        fritter = await world.wrapped()
        await world.terminals.type(Type(SESSION, fritter, input))
        assert await world.reads(fritter, keyed(input)) == keyed(input)

    async def test_typing_into_a_fritter_nobody_listens_for_raises_untyped_naming_its_address(self, world: World) -> None:
        fritter = world.unheard()
        with pytest.raises(Untyped, match=re.escape(str(fritter.socket))):
            await world.terminals.type(Type(SESSION, fritter, Text(PromptText("hello"))))

