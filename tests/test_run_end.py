"""Why a run stopped: every cause is told to one Stop, and the run is one `hands.run` event that names them all and the
end read off them, which is what the run returns or raises."""

import asyncio
import os
import signal
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from conftest import unedited
from hands.daemon.cli import launch
from hands.daemon.restart import RESTART_SIGNAL
from hands.daemon.run import stopping, until_told
from hands.daemon.starting import QUIT_SIGNALS, Cause, Ended, Failed, QuitKey, Signalled, Start, Stop, TaskFailed
from hands.sessions import audit, heartbeat
from hands.sessions.wide import WideEvent

NOW = datetime(2026, 10, 23, 12, 0, 0, tzinfo=UTC)
FAILED = TaskFailed("the namer", OSError("the disk is gone"))
TEARDOWN = ConnectionError("the device went away")

# What the run under test does: tell its Stop a cause, or raise.
type Step = Cause | BaseException


def events(recorded: list[audit.Entry], name: str) -> list[WideEvent]:
    return [entry for entry in recorded if isinstance(entry, WideEvent) and entry.event == name]


def ran(recorded: list[audit.Entry]) -> list[tuple[str, object, object]]:
    """Each hands.run event's outcome, causes, and ending."""
    return [(event.outcome, event.facts["stopped_by"], event.facts["ending"]) for event in events(recorded, "hands.run")]


def stopped(heart: heartbeat.Heart) -> bool:
    status = heartbeat.read(heart.path)
    return status is not None and status.pipeline == "stopped"


class Launched:
    """A launch whose run does each of `now` as it begins, waits to be told to stop, and then does each of `then`."""

    def __init__(self, tmp_path: Path, now: list[Step], then: list[Step], ready: bool = True) -> None:
        self.heart = heartbeat.Heart(tmp_path / "status.json", pid=4242, started_at=NOW, period=timedelta(milliseconds=10))
        self.recorded: list[audit.Entry] = []
        self.running = asyncio.Event()
        run_start = Start(restarted=False)

        async def run(stop: Stop) -> Ended:
            if ready:
                run_start.ended(self.recorded.append, None)
            self.running.set()
            for step in now:
                do(step, stop)
            await stop.wait()
            for step in then:
                do(step, stop)
            return Ended(None, 0)

        self.task = asyncio.create_task(launch(lambda: run, self.heart, lambda: (), unedited, self.recorded.append, run_start))


def do(step: Step, stop: Stop) -> None:
    match step:
        case BaseException():
            raise step
        case cause:
            stop.tell(cause)


@pytest.mark.parametrize("number", QUIT_SIGNALS)
async def test_each_quit_signal_ends_a_run_as_a_quit_named_by_the_signal(tmp_path: Path, number: signal.Signals) -> None:
    launched = Launched(tmp_path, [], [])
    await launched.running.wait()
    os.kill(os.getpid(), number)
    assert await launched.task == "quit"
    assert ran(launched.recorded) == [("ok", (number.name,), "quit")]
    assert stopped(launched.heart)


async def test_the_q_key_ends_a_run_as_a_quit_named_by_the_key(tmp_path: Path) -> None:
    launched = Launched(tmp_path, [QuitKey()], [])
    assert await launched.task == "quit"
    assert ran(launched.recorded) == [("ok", ("q key",), "quit")]
    [event] = events(launched.recorded, "hands.run")
    assert event.facts["pid"] == os.getpid()


async def test_a_failed_task_ends_a_run_failed_with_its_error(tmp_path: Path) -> None:
    launched = Launched(tmp_path, [FAILED], [])
    with pytest.raises(OSError, match="the disk is gone"):
        await launched.task
    assert ran(launched.recorded) == [("failed", ("the namer",), "failed")]
    [event] = events(launched.recorded, "hands.run")
    assert event.error == "OSError: the disk is gone"
    # Not stopped: a run that failed reads as down.
    assert not stopped(launched.heart)


@pytest.mark.parametrize(("first", "name"), [(QuitKey(), "q key"), (Signalled(RESTART_SIGNAL), "SIGUSR1"), (Signalled(signal.SIGTERM), "SIGTERM")])
async def test_a_task_that_fails_as_the_run_winds_down_fails_it_whatever_stopped_it_first(tmp_path: Path, first: Cause, name: str) -> None:
    # Failed, and so not restarted: the event's ending is what the run raises.
    launched = Launched(tmp_path, [first], [FAILED])
    with pytest.raises(OSError, match="the disk is gone"):
        await launched.task
    assert ran(launched.recorded) == [("failed", (name, "the namer"), "failed")]
    assert not stopped(launched.heart)


async def test_a_teardown_that_raises_after_the_restart_signal_fails_the_run(tmp_path: Path) -> None:
    launched = Launched(tmp_path, [], [TEARDOWN])
    await launched.running.wait()
    os.kill(os.getpid(), RESTART_SIGNAL)
    with pytest.raises(ConnectionError, match="the device went away"):
        await launched.task
    assert ran(launched.recorded) == [("failed", ("SIGUSR1", "escaped"), "failed")]
    assert not stopped(launched.heart)


async def test_a_task_failed_and_a_teardown_that_raises_are_both_raised_and_both_on_the_event(tmp_path: Path) -> None:
    launched = Launched(tmp_path, [FAILED], [TEARDOWN])
    with pytest.RaisesGroup(pytest.RaisesExc(OSError, match="the disk is gone"), pytest.RaisesExc(ConnectionError, match="the device went away")):
        await launched.task
    assert ran(launched.recorded) == [("failed", ("the namer", "escaped"), "failed")]
    [event] = events(launched.recorded, "hands.run")
    assert event.error == "ExceptionGroup: OSError: the disk is gone; ConnectionError: the device went away (2 sub-exceptions)"
    assert "OSError: the disk is gone" in event.trace and "ConnectionError: the device went away" in event.trace


async def test_a_run_that_fails_before_it_was_ready_fails_its_start_too(tmp_path: Path) -> None:
    launched = Launched(tmp_path, [TaskFailed("the pipeline", RuntimeError("the pipeline ended without being told to stop"))], [], ready=False)
    with pytest.raises(RuntimeError):
        await launched.task
    [start] = events(launched.recorded, "hands.start")
    assert (start.outcome, start.error) == ("failed", "RuntimeError: the pipeline ended without being told to stop")
    assert ran(launched.recorded) == [("failed", ("the pipeline",), "failed")]


async def test_a_run_that_returns_with_no_cause_told_fails_saying_so(tmp_path: Path) -> None:
    heart = heartbeat.Heart(tmp_path / "status.json", pid=4242, started_at=NOW, period=timedelta(milliseconds=10))
    recorded: list[audit.Entry] = []

    async def run(_stop: Stop) -> Ended:
        return Ended(None, 0)

    with pytest.raises(LookupError, match="no cause told"):
        await launch(lambda: run, heart, lambda: (), unedited, recorded.append, Start(restarted=False))
    assert ran(recorded) == [("failed", ("escaped",), "failed")]


async def test_a_pipeline_that_ends_on_its_own_is_told_before_the_teardown_and_a_quit_after_it_does_not_hide_it() -> None:
    # As converse runs it: the pipeline told on, then waited on beside the stop.
    stop = Stop()

    async def ends() -> None:
        pass

    pipeline = asyncio.create_task(until_told(ends(), stop), name="the pipeline")
    pipeline.add_done_callback(stopping(stop))
    told = asyncio.create_task(stop.wait())
    await asyncio.wait({pipeline, told}, return_when=asyncio.FIRST_COMPLETED)
    # The teardown begins here, with the pipeline's end told already: a Ctrl-C during it is listed after.
    assert stop.named == ("the pipeline",)
    stop.tell(Signalled(signal.SIGINT))
    match stop.end:
        case Failed(errors=(RuntimeError() as error,)):
            assert str(error) == "the pipeline ended without being told to stop"
        case end:
            raise AssertionError(f"the run ended {end}")
    assert stop.named == ("the pipeline", "SIGINT")
    told.cancel()


async def test_a_pipeline_told_to_stop_ends_without_failing() -> None:
    stop = Stop()
    stop.tell(QuitKey())

    async def ends() -> None:
        pass

    pipeline = asyncio.create_task(until_told(ends(), stop), name="the pipeline")
    pipeline.add_done_callback(stopping(stop))
    await asyncio.wait({pipeline})
    assert (stop.named, stop.end) == (("q key",), "quit")
