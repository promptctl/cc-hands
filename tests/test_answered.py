"""A door's answer to a call answered with an effect after it: written first, and the effect run once it is; where it
cannot be written, the effect run or dropped as the answer says, and the unit failed by what stopped it."""

import pytest
from aiohttp import web
from aiohttp.test_utils import make_mocked_request

from hands.virtualclock import VirtualClock
from hands.sessions.audit import Entry
from hands.sessions.wide import WideEvent, begun, unit
from hands.voice.answered import answered_then
from hands.voice.tool import After, Unanswered


class Unwritable(web.Response):
    """An answer whose client went away before it could be written."""

    async def prepare(self, request: web.BaseRequest) -> None:
        raise ConnectionResetError("the client went away")


@pytest.mark.parametrize(("unanswered", "ran"), [("dropped", []), ("run", ["kept"])])
async def test_an_effect_whose_answer_cannot_be_written_is_run_or_dropped_as_the_answer_says_and_the_unit_fails(unanswered: Unanswered, ran: list[str]) -> None:
    recorded: list[Entry] = []
    effects: list[str] = []

    async def effect() -> None:
        effects.append("kept")

    with pytest.raises(ConnectionResetError), unit("tool.after", recorded.append, begun(VirtualClock())):
        await answered_then(make_mocked_request("POST", "/"), Unwritable(), After({}, effect, unanswered), "use_model")
    assert effects == ran
    [event] = [entry for entry in recorded if isinstance(entry, WideEvent)]
    assert (event.outcome, event.error) == ("failed", "ConnectionResetError: the client went away")
