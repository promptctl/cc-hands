import asyncio

from hands.tasks import cancel_and_wait


async def test_every_task_is_cancelled_in_the_order_given_and_told_before_the_wait_returns() -> None:
    heard: list[str] = []
    told: list[str] = []

    async def runs(name: str) -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            heard.append(name)
            raise

    names = [f"t{index}" for index in range(6)]
    tasks = {name: asyncio.create_task(runs(name)) for name in names}
    for name, task in tasks.items():
        task.add_done_callback(lambda _done, name=name: told.append(name))
    await asyncio.sleep(0)
    given = names[::-1]
    # One already done among them is waited on as done.
    done = asyncio.ensure_future(asyncio.sleep(0))
    await done
    await cancel_and_wait(*(tasks[name] for name in given), done)
    assert (heard, sorted(told), all(task.cancelled() for task in tasks.values())) == (given, sorted(names), True)
