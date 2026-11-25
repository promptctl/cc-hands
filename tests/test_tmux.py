"""The tmux hands asks the user's servers through: the one found on the PATH as the run starts, or none."""

from pathlib import Path

import pytest

from fakechildren import FakeChildren, exited
from hands.core.effects import Command, Text
from hands.core.session import CommandName, PromptText
from hands.core.tmux import Pane, Unanswered
from hands.sessions.terminals import Unconfirmed, Unopened, Untyped
from hands.sessions.tmux import TYPING, NoTmux, TmuxTerminals, on_path
from hands.virtualclock import VirtualClock

SOCKET = Path("/tmp/tmux-501/default")


def test_the_tmux_asked_is_the_first_runnable_one_on_the_path(tmp_path: Path) -> None:
    unrunnable, runnable = tmp_path / "unrunnable", tmp_path / "runnable"
    for folder, mode in ((unrunnable, 0o644), (runnable, 0o755)):
        folder.mkdir()
        (folder / "tmux").touch(mode)
    assert isinstance(on_path(FakeChildren(VirtualClock()), f"{unrunnable}:{runnable}"), TmuxTerminals)
    assert isinstance(on_path(FakeChildren(VirtualClock()), str(unrunnable)), NoTmux)


async def test_with_no_tmux_on_the_path_every_ask_says_so_and_nothing_is_opened_or_typed(tmp_path: Path) -> None:
    terminals = on_path(FakeChildren(VirtualClock()), str(tmp_path))
    reason = f"no tmux is on the PATH {tmp_path}"
    assert [await terminals.panes(SOCKET), await terminals.clients(SOCKET), await terminals.shown(SOCKET, "%0"), await terminals.state(SOCKET, "%0")] == [Unanswered(reason)] * 4
    assert await terminals.open("billing", tmp_path, ["claude"], {}, {}) == Unopened(reason)
    with pytest.raises(Untyped, match=reason):
        await terminals.type(Pane(SOCKET, "%0", "billing", 0), Text(PromptText("hello")))


async def test_a_tmux_that_cannot_be_run_opens_and_types_nothing_and_answers_nothing(tmp_path: Path) -> None:
    # [LAW:no-ambient-temporal-coupling] a tmux never started is never waited on: no real time passes.
    terminals = TmuxTerminals(FakeChildren(VirtualClock()), tmp_path / "tmux")
    match await terminals.open("billing", tmp_path, ["claude"], {}, {}):
        case Unopened(reason=reason):
            assert reason.startswith("tmux could not be started: ")
        case other:
            pytest.fail(f"opened as {other}")
    with pytest.raises(Untyped, match="could not be started"):
        await terminals.type(Pane(SOCKET, "%0", "billing", 0), Text(PromptText("hello")))
    match await terminals.panes(SOCKET):
        case Unanswered(reason=reason):
            assert reason.startswith("tmux could not be started: ")
        case other:
            pytest.fail(f"listed as {other}")


async def test_a_typing_tmux_stopped_after_its_first_key_may_have_typed_and_one_stopped_before_typed_nothing(tmp_path: Path) -> None:
    # tmux keeps what a list ran before the command it stopped at (test_terminals_contract_tmux.py), so the name of a
    # command can be in the input when its arguments' paste fails.
    pane, command = Pane(SOCKET, "%3", "billing", 0), Command(CommandName("model"), PromptText("opus"))
    partway = FakeChildren(VirtualClock(), {str(tmp_path / "tmux"): exited(1, out=f"{TYPING}\n".encode(), err=b"no buffer hands-1\n")})
    with pytest.raises(Unconfirmed, match=f"tmux at {SOCKET} began typing into pane %3 and stopped partway: no buffer hands-1"):
        await TmuxTerminals(partway, tmp_path / "tmux").type(pane, command)
    before = FakeChildren(VirtualClock(), {str(tmp_path / "tmux"): exited(1, err=b"can't find pane: %3\n")})
    with pytest.raises(Untyped, match=f"tmux at {SOCKET} did not type into pane %3: can't find pane: %3"):
        await TmuxTerminals(before, tmp_path / "tmux").type(pane, command)
