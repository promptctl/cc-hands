"""The pinned Claude Code's order of a session's end, the next session's start, and that session's first request, which
hands' brain is built on: on a /clear or a /resume at its keyboard, Claude Code runs the start of the session it enters
only once the end of the one it left is answered, so the brain hears every end under the session it ends
(hands.brain.process.Brain._moved_by); and it sends the entered session's first request only once that start is
answered, so the brain admits it as its own (hands.brain.context.Kept)."""

import asyncio
import json
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
def moves(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Each start and end the brain is posted, as its hook and its session, in the order seen: an end once its reply,
    held HOLD_SECONDS past the brain's answer, goes, and a start as it arrives. Asked for before `hands`, so the brain's
    launch is posted through it."""
    seen: list[tuple[str, str]] = []
    posted = Brain._posted  # pyright: ignore[reportPrivateUsage]

    async def holding(brain: Brain, event: str, body: bytes) -> Mapping[str, object]:
        if event == "SessionStart":
            seen.append((event, json.loads(body)["session_id"]))
        answer = await posted(brain, event, body)
        if event == "SessionEnd":
            await asyncio.sleep(HOLD_SECONDS)
            seen.append((event, json.loads(body)["session_id"]))
        return answer

    monkeypatch.setattr(Brain, "_posted", holding)
    return seen


async def test_a_clear_and_a_resume_start_the_session_entered_only_once_the_end_of_the_one_left_is_answered(moves: list[tuple[str, str]], mock_api: MockApi, hands: Hands) -> None:
    for words in ("say hello", "say cleared", "say resumed"):
        mock_api.script(f"1-{words}", {"lastUserContains": [words], "bodyContains": [TURN]}, {"type": "text", "text": f"{words.title()}."})
    await hands.connected()
    # A turn first, so the conversation the brain launched in has a transcript a /resume can find.
    await hands.say("say hello")
    await hands.until(lambda: "Say Hello." in "".join(hands.out.said()))
    [(_, launched)] = moves
    await hands.keyed("-l", "/clear")
    await hands.keyed("Enter")
    await hands.until(lambda: len(moves) == 3)
    # The entered session's first request is the brain's: the turn spoken there is answered.
    await hands.say("say cleared")
    await hands.until(lambda: "Say Cleared." in "".join(hands.out.said()))
    await hands.keyed("-l", f"/resume {launched}")
    await hands.keyed("Enter")
    await hands.until(lambda: len(moves) == 5)
    await hands.say("say resumed")
    await hands.until(lambda: "Say Resumed." in "".join(hands.out.said()))
    entered = moves[2][1]
    # Each start came only once the end before it was answered, however long that answer was held.
    assert moves == [("SessionStart", launched), ("SessionEnd", launched), ("SessionStart", entered), ("SessionEnd", entered), ("SessionStart", launched)]
    assert entered != launched
    assert [(event.facts["left"], event.facts["entered"], event.facts["source"]) for event in events(hands.recorded, "brain.session")] == [
        (launched, launched, "startup"), (launched, entered, "clear"), (entered, launched, "resume"),
    ]
