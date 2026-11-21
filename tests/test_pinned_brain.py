"""The assembled brain, wired: a turn said to hands goes through hands' proxy to the mock of Anthropic's API, and the brain's
reply is spoken. The happy path and nothing else, the one smoke test of the brain (promptctl-testing-fxnj caps the smoke
layer): every decision hands makes of what Claude Code does is asserted in the unit layer, on Claude Code's turns as
recorded, and what hands relies on of its terminal is tests/test_terminal_contract.py's. Beside it, what only the pinned
Claude Code can say: that a brain moved to another conversation at its pane still answers."""

import pytest

from conftest import events
from hands.core.events import Ended
from pinnedbrain import TURN, Hands, MockApi
from pinnedbrain import claude_code, hands, mock_api, mock_binary  # pyright: ignore[reportUnusedImport]

pytestmark = pytest.mark.outside


async def test_a_turn_said_to_hands_goes_through_its_proxy_to_the_mock_and_the_reply_is_spoken(mock_api: MockApi, hands: Hands) -> None:
    mock_api.script("1-hello", {"lastUserContains": ["say hello"], "bodyContains": [TURN]}, {"type": "text", "text": "Hello."})
    await hands.connected()
    await hands.say("say hello")
    await hands.until(lambda: "Hello." in "".join(hands.out.said()))
    # The turn reached the model through hands' proxy: the brain's request is one it observed.
    assert any(TURN in str(request.body) and "say hello" in str(request.body) for request in hands.sent)


async def test_a_brain_cleared_at_its_pane_answers_the_next_turn_said_to_hands_from_the_session_it_moved_to(mock_api: MockApi, hands: Hands) -> None:
    mock_api.script("1-hello", {"lastUserContains": ["say hello"], "bodyContains": [TURN]}, {"type": "text", "text": "Hello."})
    mock_api.script("2-again", {"lastUserContains": ["say again"], "bodyContains": [TURN]}, {"type": "text", "text": "Again."})
    await hands.connected()
    await hands.say("say hello")
    await hands.until(lambda: "Hello." in "".join(hands.out.said()))
    # The start of each session is the shim's, run as a command hook in the brain's pane.
    await hands.keyed("/clear", "Enter")
    await hands.until(lambda: len(events(hands.recorded, "brain.start")) == 2)
    await hands.say("say again")
    await hands.until(lambda: "Again." in "".join(hands.out.said()))
    launched, cleared = (event.facts for event in events(hands.recorded, "brain.start"))
    assert (launched["source"], cleared["source"], cleared["left"]) == ("startup", "clear", launched["session"])
    # The turn after the move went under the session the brain moved to, and was the brain's to hear.
    assert any(TURN in str(request.body) and "say again" in str(request.body) and request.session == cleared["session"] for request in hands.sent)
    # The end of the session left was posted, and before the start: no hook of it came once the brain had moved.
    assert [event.facts["applied"] for event in events(hands.recorded, "applied") if isinstance(event.facts["applied"], Ended)] == [Ended(launched["session"], "clear")]  # pyright: ignore[reportArgumentType]
    assert events(hands.recorded, "brain.hook") == []
