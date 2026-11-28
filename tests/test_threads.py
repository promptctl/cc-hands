"""A serial thread: its work runs one piece at a time, in order, and a close ends the thread once all of it has run."""

import threading

import pytest

from conftest import taken
from hands.threads import SerialThread


def named(name: str) -> list[threading.Thread]:
    return [thread for thread in threading.enumerate() if thread.name == name]


async def test_a_close_returns_once_everything_given_before_it_has_run_and_the_thread_has_ended() -> None:
    serial, ran = SerialThread("closing"), list[int]()
    for piece in range(3):
        serial.give(lambda piece=piece: ran.append(piece))
    await serial.close()
    assert ran == [0, 1, 2]
    assert named("closing") == []


async def test_work_given_to_a_closed_thread_is_refused_rather_than_never_run() -> None:
    serial = SerialThread("closed")
    await serial.close()
    with pytest.raises(RuntimeError, match="closed is closed"):
        serial.give(lambda: None)


async def test_a_second_close_returns_at_once() -> None:
    serial = SerialThread("closed twice")
    await serial.close()
    await serial.close()
    assert named("closed twice") == []


def test_a_call_is_answered_once_everything_given_before_it_has_run() -> None:
    serial, ran = SerialThread("calling"), list[int]()
    for piece in range(2):
        serial.give(lambda piece=piece: ran.append(piece))
    assert serial.call(lambda: list(ran)) == [0, 1]


def test_work_that_raises_raises_on_the_thread_that_called_for_it_and_the_thread_runs_on() -> None:
    serial = SerialThread("raising")

    def fails() -> None:
        raise ValueError("no weights")

    with pytest.raises(ValueError, match="no weights"):
        serial.call(fails)
    assert serial.call(lambda: "after") == "after"


def test_work_that_calls_for_more_work_on_its_own_thread_is_refused_rather_than_waiting_for_ever() -> None:
    serial = SerialThread("itself")
    with pytest.raises(RuntimeError, match="work on itself called for more work on itself"):
        serial.call(lambda: serial.call(lambda: None))


def test_the_work_waiting_is_what_was_given_behind_the_work_running() -> None:
    serial, running, let_go = SerialThread("waiting"), threading.Event(), threading.Event()

    def held() -> None:
        running.set()
        let_go.wait()

    serial.give(held)
    try:
        running.wait()
        serial.give(lambda: None)
        serial.give(lambda: None)
        assert serial.waiting == 2
    finally:
        let_go.set()
    serial.call(lambda: None)
    assert serial.waiting == 0


def test_work_submitted_and_cancelled_before_its_turn_never_runs_and_the_work_behind_it_does() -> None:
    serial, let_go, ran = SerialThread("cancelled"), threading.Event(), list[str]()
    serial.submit(let_go.wait)
    try:
        # As an executor's work is, when the task awaiting it is cancelled: Pipecat's verdicts are given this way.
        cancelled = serial.submit(lambda: ran.append("cancelled"))
        behind = serial.submit(lambda: ran.append("behind"))
        assert cancelled.cancel()
    finally:
        let_go.set()
    behind.result()
    assert (ran, cancelled.cancelled()) == (["behind"], True)


def test_only_the_thread_itself_is_here() -> None:
    serial = SerialThread("here")
    assert (serial.here(), serial.call(serial.here)) == (False, True)


def test_work_asked_to_settle_where_no_loop_runs_is_refused_before_it_is_queued() -> None:
    serial, ran = SerialThread("loopless"), list[str]()
    with pytest.raises(RuntimeError, match="no running event loop"):
        serial.settles(lambda: ran.append("settled"))
    # Work given after it runs once everything given before it has.
    serial.call(lambda: None)
    assert ran == []


async def test_work_that_fails_once_nobody_waits_for_it_says_so(errors: list[str]) -> None:
    serial, let_go = SerialThread("abandoned"), threading.Event()

    def fails() -> None:
        let_go.wait()
        raise ValueError("no weights")

    waited = serial.settles(fails)
    waited.cancel()
    let_go.set()
    # What is given after it settles after it, so its failure has been said by then.
    await serial.settles(lambda: None)
    assert taken(errors) == ["work on abandoned failed after its waiter had gone"]
