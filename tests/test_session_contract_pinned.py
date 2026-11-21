"""The pinned Claude Code's order of a session's end and the next session's start, which hands' brain is built on: on a
/clear or a /resume at its keyboard, Claude Code runs the start of the session it enters only once the end of the one it
left is answered, so the brain hears every end under the session it ends (hands.brain.process.Brain._moved_by)."""

import asyncio
import json
import time
from collections.abc import Mapping

import pytest

from pinnedbrain import TURN, Hands, MockApi
from pinnedbrain import claude_code, hands, mock_api, mock_binary  # pyright: ignore[reportUnusedImport]
from conftest import events
from hands.brain.process import Brain

pytestmark = pytest.mark.outside

# How long past the brain's answer an end's reply is held: far longer than Claude Code takes to run a start it does not
# wait to run, and far inside the time it waits on the end (BRAIN_HOOKS).
HOLD_SECONDS = 2.0


@pytest.fixture
def moves(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, float]]:
    """Each start and end the brain is posted, as its hook, its session and when: an end's reply, held HOLD_SECONDS past
    the brain's answer, and a start's arrival. Asked for before `hands`, so the brain's launch is posted through it."""
    seen: list[tuple[str, str, float]] = []
    posted = Brain._posted  # pyright: ignore[reportPrivateUsage]

    async def holding(brain: Brain, event: str, body: bytes) -> Mapping[str, object]:
        if event == "SessionStart":
            seen.append((event, json.loads(body)["session_id"], time.monotonic()))
        answer = await posted(brain, event, body)
        if event == "SessionEnd":
            await asyncio.sleep(HOLD_SECONDS)
            seen.append((event, json.loads(body)["session_id"], time.monotonic()))
        return answer

    monkeypatch.setattr(Brain, "_posted", holding)
    return seen


async def test_a_clear_and_a_resume_start_the_session_entered_only_once_the_end_of_the_one_left_is_answered(moves: list[tuple[str, str, float]], mock_api: MockApi, hands: Hands) -> None:
    # A turn first, so the conversation the brain launched in has a transcript a /resume can find.
    mock_api.script("1-hello", {"lastUserContains": ["say hello"], "bodyContains": [TURN]}, {"type": "text", "text": "Hello."})
    await hands.connected()
    await hands.say("say hello")
    await hands.until(lambda: "Hello." in "".join(hands.out.said()))
    [(_, launched, _)] = moves
    await hands.keyed("-l", "/clear")
    await hands.keyed("Enter")
    await hands.until(lambda: len(moves) == 3)
    await hands.keyed("-l", f"/resume {launched}")
    await hands.keyed("Enter")
    await hands.until(lambda: len(moves) == 5)
    _, (_, cleared_from, cleared), (_, entered, entered_at), (_, resumed_from, resumed), (_, back, back_at) = moves
    assert (cleared_from, resumed_from, back) == (launched, entered, launched) and entered != launched
    # Each start came only once the end before it was answered, however long that answer was held.
    assert entered_at > cleared and back_at > resumed
    assert [(event.facts["left"], event.facts["entered"], event.facts["source"]) for event in events(hands.recorded, "brain.session")] == [
        (launched, launched, "startup"), (launched, entered, "clear"), (entered, launched, "resume"),
    ]
