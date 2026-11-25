"""What hands relies on of the user's terminals, as one contract held to two parties: the fake terminals and fritters
(tests/faketerminals.py), in `make check` (test_terminals_contract.py), and tmux and fritter themselves through
`hands.sessions.tmux.TmuxTerminals` and `hands.sessions.typing.FritterSockets`, on a tmux server of the test's own and the
fritter this checkout carries (test_terminals_contract_tmux.py, outside). The cases are written once, here, and each of those
modules runs all of them against its own `world`, so each asserts the same of both: that is what holds the fake to tmux and
fritter. A claim they no longer keep fails the outside run, and the fake is changed to match."""

import asyncio
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, Protocol, get_args

import pytest

from faketerminals import DEAD, keyed
from hands.core.effects import Command, Fritter, Input, Key, Text, Writer
from hands.core.session import CommandName, Keystroke, PromptText
from hands.core.tmux import Listed, Pane, Unanswered
from hands.sessions.terminals import Fritters, Opened, Terminals, Unconfirmed, Unopened, Untyped

# What each window's program draws once it is up.
DRAWN = "hello from the pane"
# The variable each window is opened with, beyond its server's environment, and its value.
VARIABLE, VALUE = "HANDS_CONTRACT_WINDOW", "the window's own"
# Each kind of input a session is typed: a prompt, its newline kept in it; a command, with arguments and without; every chord.
INPUTS = [Text(PromptText("first\nsecond")), Command(CommandName("model"), PromptText("opus")), Command(CommandName("clear"), None), *(Key(key) for key in get_args(Keystroke))]
TYPED = ["text", "command", "bare command", *get_args(Keystroke)]
# How a pane's program ends: its pane kept by remain-on-exit, or its window closed with it.
Ending = Literal["dead", "closed"]


class World(Protocol):
    """A tmux server of the world's own, the windows the contract opens in it, each running a program that draws DRAWN,
    asks its terminal for bracketed paste, and reads what is typed into it raw, opened with VARIABLE set to VALUE; and
    fritters wrapping such a program."""

    @property
    def terminals(self) -> Terminals: ...

    @property
    def fritters(self) -> Fritters: ...

    @property
    def socket(self) -> Path:
        """The socket of the world's server, by its real path."""
        ...

    @property
    def environment(self) -> Mapping[str, str]:
        """The environment the world's windows are opened in, outside tmux, which reaches the world's server."""
        ...

    @property
    def directory(self) -> Path:
        """Where tmux puts the world's sockets, as hands names it from `environment`: through any link on its path."""
        ...

    async def window(self, session: str, environment: Mapping[str, str] | None = None) -> Opened:
        """A window opened through `terminals` in tmux session `session`, in `environment` over the world's own, once
        its program is up."""
        ...

    def given(self, window: Opened) -> str | None:
        """What `window`'s program was given as VARIABLE."""
        ...

    def pid(self, window: Opened) -> int:
        """The pid of the program `window` runs."""
        ...

    def device(self, window: Opened) -> int:
        """The device of the terminal `window`'s program runs on."""
        ...

    async def end(self, window: Opened, ending: Ending) -> None:
        """End `window`'s program as `ending` says; returns once its server has seen it end."""
        ...

    async def attach(self, window: Opened) -> int:
        """A client attached to the server, showing `window`: the device of its terminal."""
        ...

    async def stop(self) -> None:
        """The server exits as one killed does, leaving its socket behind."""
        ...

    async def refuse(self) -> None:
        """The server's socket refuses every connection, as one the user may not write to does."""
        ...

    async def wedge(self) -> None:
        """The server answers nothing, as one stopped does; `terminals` gives up on it at once."""
        ...

    async def wrapped(self) -> Fritter:
        """A fritter wrapping such a program, in a window of its own, once the program has asked for bracketed paste."""
        ...

    def unheard(self) -> Fritter:
        """A fritter's address that nobody listens at, as a session that ended leaves."""
        ...

    async def stall(self, fritter: Fritter) -> None:
        """`fritter` takes what it is asked and answers nothing, as one stopped does; `fritters` gives up on it at once."""
        ...

    async def reads(self, writer: Writer, expected: str) -> str:
        """What the program `writer` types into read, once it has read as many bytes as `expected` holds."""
        ...


class TerminalsContract:
    async def test_a_window_opened_makes_its_session_once_and_runs_its_program_in_a_pane_its_server_lists(self, world: World) -> None:
        first, second = await world.window("work"), await world.window("work")
        assert (first.made, second.made) == (True, False)
        # In the server tmux picks outside tmux, named as tmux and the socket's directory name it.
        assert (first.socket, second.socket) == (world.socket, world.socket)
        assert world.terminals.sockets(world.directory) == [world.socket]
        assert [first.pid, second.pid] == [world.pid(first), world.pid(second)]
        assert await world.terminals.panes(world.socket) == Listed({world.device(first): Pane(world.socket, first.pane, "work", 0), world.device(second): Pane(world.socket, second.pane, "work", 1)})

    async def test_a_window_opened_inside_tmux_opens_in_the_server_its_tmux_names(self, world: World) -> None:
        await world.window("work")
        inside = await world.window("play", {"TMUX": f"{world.socket},1,0", "TMUX_TMPDIR": str(world.directory / "elsewhere")})
        assert (inside.socket, inside.made) == (world.socket, True)
        # tmux reads an empty $TMUX as none, so a window opened under one is in the default server.
        assert (await world.window("rest", {"TMUX": ""})).socket == world.socket

    async def test_a_window_runs_its_program_with_the_variables_it_was_opened_with(self, world: World) -> None:
        assert world.given(await world.window("work")) == VALUE

    async def test_a_pane_shows_what_its_program_drew(self, world: World) -> None:
        window = await world.window("work")
        assert await world.terminals.shown(world.socket, window.pane) == DRAWN
        assert await world.terminals.tail(world.socket, window.pane, 5) == DRAWN

    async def test_a_pane_runs_until_its_program_ends_then_is_dead_where_kept_and_closed_where_not(self, world: World) -> None:
        # A third window, so its server outlives the two that end.
        kept, closed, running = await world.window("work"), await world.window("work"), await world.window("work")
        assert [await world.terminals.state(world.socket, window.pane) for window in (kept, closed)] == ["running", "running"]
        await world.end(kept, "dead")
        await world.end(closed, "closed")
        assert [await world.terminals.state(world.socket, window.pane) for window in (kept, closed)] == ["dead", "closed"]
        # A dead pane runs nothing and is listed as none.
        match await world.terminals.panes(world.socket):
            case Listed(panes=panes):
                assert {pane.id for pane in panes.values()} == {running.pane}
            case other:
                pytest.fail(f"listed as {other}")
        assert isinstance(await world.terminals.shown(world.socket, closed.pane), Unanswered)

    async def test_a_dead_pane_shows_tmuxs_notice_in_place_of_what_its_program_drew_which_its_tail_keeps(self, world: World) -> None:
        window, _ = await world.window("work"), await world.window("work")
        await world.end(window, "dead")
        match await world.terminals.shown(world.socket, window.pane), await world.terminals.tail(world.socket, window.pane, 5):
            case str(shown), str(tail):
                assert shown.split("\n")[-1].startswith(DEAD) and DRAWN not in shown
                assert tail.split("\n")[0] == DRAWN and tail.split("\n")[-1].startswith(DEAD)
                assert await world.terminals.tail(world.socket, window.pane, 1) == tail.split("\n")[-1]
            case other:
                pytest.fail(f"read as {other}")

    async def test_a_server_killed_leaves_its_socket_behind_holding_no_panes_and_no_clients(self, world: World) -> None:
        window = await world.window("work")
        await world.attach(window)
        await world.stop()
        assert world.terminals.sockets(world.directory) == [world.socket]
        assert (await world.terminals.panes(world.socket), await world.terminals.clients(world.socket)) == (Listed({}), {})
        assert await world.terminals.state(world.socket, window.pane) == "closed"
        # A window opened there starts a server again, at the socket left behind.
        again = await world.window("work")
        assert (again.socket, again.made) == (world.socket, True)

    async def test_a_server_whose_socket_refuses_answers_nothing_says_why_and_opens_no_window(self, world: World) -> None:
        window = await world.window("work")
        await world.refuse()
        asked = await asyncio.gather(world.terminals.panes(world.socket), world.terminals.clients(world.socket), world.terminals.shown(world.socket, window.pane), world.terminals.tail(world.socket, window.pane, 5), world.terminals.state(world.socket, window.pane))
        assert [type(answer) for answer in asked] == [Unanswered] * 5
        assert all(str(world.socket) in answer.reason for answer in asked if isinstance(answer, Unanswered))
        assert isinstance(await world.terminals.open("work", world.directory, ("true",), {}, world.environment), Unopened)
        with pytest.raises(Untyped, match=re.escape(f"did not type into pane {window.pane}")):
            await world.terminals.type(Pane(world.socket, window.pane, "work", 0), Text(PromptText("hello")))

    async def test_a_wedged_server_answers_nothing_so_what_it_was_asked_to_do_may_have_been_done(self, world: World) -> None:
        window = await world.window("work")
        await world.wedge()
        pane = Pane(world.socket, window.pane, "work", 0)
        asked = await asyncio.gather(world.terminals.panes(world.socket), world.terminals.clients(world.socket), world.terminals.shown(world.socket, window.pane), world.terminals.state(world.socket, window.pane), world.terminals.type(pane, Text(PromptText("hello"))), return_exceptions=True)
        assert [type(answer) for answer in asked] == [Unanswered] * 4 + [Unconfirmed]
        assert all(str(world.socket) in str(answer) for answer in asked)
        assert isinstance(await world.terminals.open("work", world.directory, ("true",), {}, world.environment), Unanswered)

    async def test_a_client_is_listed_by_its_terminal_with_the_pane_it_shows(self, world: World) -> None:
        _, shown = await world.window("work"), await world.window("work")
        client = await world.attach(shown)
        assert await world.terminals.clients(world.socket) == {client: world.device(shown)}

    @pytest.mark.parametrize("input", INPUTS, ids=TYPED)
    async def test_each_input_typed_into_a_pane_reaches_its_program_as_its_keys(self, world: World, input: Input) -> None:
        window = await world.window("work")
        pane = Pane(world.socket, window.pane, "work", 0)
        await world.terminals.type(pane, input)
        assert await world.reads(pane, keyed(input)) == keyed(input)

    @pytest.mark.parametrize("ending", get_args(Ending))
    @pytest.mark.parametrize("input", INPUTS, ids=TYPED)
    async def test_typing_into_a_pane_whose_program_ended_raises_untyped_naming_the_pane(self, world: World, ending: Ending, input: Input) -> None:
        # A dead pane takes keys without a word and refuses a paste: neither is typed, whatever tmux exits with.
        window, _ = await world.window("work"), await world.window("work")
        await world.end(window, ending)
        with pytest.raises(Untyped, match=re.escape(f"did not type into pane {window.pane}")):
            await world.terminals.type(Pane(world.socket, window.pane, "work", 0), input)

    @pytest.mark.parametrize("input", INPUTS, ids=TYPED)
    async def test_each_input_typed_into_a_fritter_reaches_the_program_it_wrapped_as_its_keys(self, world: World, input: Input) -> None:
        fritter = await world.wrapped()
        await world.fritters.type(fritter, input)
        assert await world.reads(fritter, keyed(input)) == keyed(input)

    async def test_typing_into_a_fritter_nobody_listens_for_raises_untyped_naming_its_address(self, world: World) -> None:
        fritter = world.unheard()
        with pytest.raises(Untyped, match=re.escape(str(fritter.socket))):
            await world.fritters.type(fritter, Text(PromptText("hello")))

    async def test_typing_into_a_fritter_that_never_answers_raises_unconfirmed_naming_its_address(self, world: World) -> None:
        fritter = await world.wrapped()
        await world.stall(fritter)
        with pytest.raises(Unconfirmed, match=re.escape(str(fritter.socket))):
            await world.fritters.type(fritter, Text(PromptText("hello")))
