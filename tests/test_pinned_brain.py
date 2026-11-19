"""The assembled brain, wired: a turn said to hands goes through hands' proxy to the mock of Anthropic's API, and the brain's
reply is spoken. The happy path and nothing else, the one smoke test of the brain (promptctl-testing-fxnj caps the smoke
layer): every decision hands makes of what Claude Code does is asserted in the unit layer, on Claude Code's turns as
recorded, and what hands relies on of its terminal is tests/test_terminal_contract.py's."""

import pytest

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
