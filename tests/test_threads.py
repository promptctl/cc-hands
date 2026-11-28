"""A serial thread: its work runs one piece at a time, in order, and a close ends the thread once all of it has run."""

import threading

import pytest

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
