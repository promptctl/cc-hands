"""The fan-out: one stream told to every listener subscribed as each item is published."""

from hands.fanout import Fanout


async def test_each_listener_hears_what_is_published_while_it_listens_in_order_and_a_new_subscription_only_what_follows() -> None:
    fanout: Fanout[int] = Fanout()
    # Nobody listens: it reaches nobody.
    fanout.publish(0)
    async with fanout.subscribed() as first:
        async with fanout.subscribed() as second:
            fanout.publish(1)
            fanout.publish(2)
            assert [await anext(second), await anext(second)] == [1, 2]
        fanout.publish(3)
        assert [await anext(first), await anext(first), await anext(first)] == [1, 2, 3]
    async with fanout.subscribed() as again:
        fanout.publish(4)
        assert await anext(again) == 4
