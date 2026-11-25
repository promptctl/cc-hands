"""The tmux hands asks the user's servers through: the one found on the PATH as the run starts, or none."""

from pathlib import Path

import pytest

from conftest import children_on
from hands.clock import SYSTEM
from hands.core.effects import Text
from hands.core.session import PromptText
from hands.core.tmux import Pane, Unanswered
from hands.sessions.terminals import Unopened, Untyped
from hands.sessions.tmux import NoTmux, TmuxTerminals, on_path
from hands.virtualclock import VirtualClock

SOCKET = Path("/tmp/tmux-501/default")


def test_the_tmux_asked_is_the_first_runnable_one_on_the_path(tmp_path: Path) -> None:
    unrunnable, runnable = tmp_path / "unrunnable", tmp_path / "runnable"
    for folder, mode in ((unrunnable, 0o644), (runnable, 0o755)):
        folder.mkdir()
        (folder / "tmux").touch(mode)
    assert isinstance(on_path(children_on(SYSTEM), f"{unrunnable}:{runnable}"), TmuxTerminals)
    assert isinstance(on_path(children_on(SYSTEM), str(unrunnable)), NoTmux)


async def test_with_no_tmux_on_the_path_every_ask_says_so_and_nothing_is_opened_or_typed(tmp_path: Path) -> None:
    terminals = on_path(children_on(SYSTEM), str(tmp_path))
    reason = f"no tmux is on the PATH {tmp_path}"
    assert [await terminals.panes(SOCKET), await terminals.clients(SOCKET), await terminals.shown(SOCKET, "%0"), await terminals.state(SOCKET, "%0")] == [Unanswered(reason)] * 4
    assert await terminals.open("billing", tmp_path, ["claude"], {}, {}) == Unopened(reason)
    with pytest.raises(Untyped, match=reason):
        await terminals.type(Pane(SOCKET, "%0", "billing", 0), Text(PromptText("hello")))


async def test_a_tmux_that_cannot_be_run_opens_and_types_nothing_and_answers_nothing(tmp_path: Path) -> None:
    # [LAW:no-ambient-temporal-coupling] a tmux never started is never waited on: no real time passes.
    terminals = TmuxTerminals(children_on(VirtualClock()), tmp_path / "tmux")
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
