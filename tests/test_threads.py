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
