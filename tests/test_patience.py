import asyncio
import itertools
import time

import conftest
import pytest
from conftest import patiently


async def test_a_step_that_holds_the_loop_past_the_patience_is_not_counted_against_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(conftest, "PATIENCE_SECS", 0.1)
    finished: list[None] = []

    def step() -> None:
        time.sleep(0.3)
        finished.append(None)

    asyncio.get_running_loop().call_later(0.05, step)
    await patiently(lambda: bool(finished), lambda: 0)


async def test_a_wait_whose_end_comes_on_the_look_that_spends_its_patience_ends_there(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(conftest, "PATIENCE_SECS", 0.05)
    looks = itertools.count()
    await patiently(lambda: next(looks) == round(conftest.PATIENCE_SECS / conftest.LOOK_SECS), lambda: 0)


async def test_a_wait_on_a_free_loop_where_nothing_moves_fails_once_its_patience_is_spent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(conftest, "PATIENCE_SECS", 0.1)
    with pytest.raises(TimeoutError, match="nothing moved through 0.1s of looking, at 0"):
        await patiently(lambda: False, lambda: 0)
