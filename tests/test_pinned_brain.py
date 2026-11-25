"""The assembled brain, wired: a turn said to hands goes through brioche and hands' proxy to the mock of Anthropic's API,
and the brain's reply is spoken. The happy path and nothing else, the one smoke test of the brain (promptctl-testing-fxnj caps the smoke
layer): every decision hands makes of what Claude Code does is asserted in the unit layer, on Claude Code's turns as
recorded, and what hands relies on of its terminal is tests/test_terminal_contract.py's."""

import pytest

from pinnedbrain import TURN, Hands, MockApi
from hands.core.wire import MainTurn
from hands.sessions.audit import BriocheSaid
from pinnedbrain import claude_code, hands, mock_api, mock_binary  # pyright: ignore[reportUnusedImport]

pytestmark = pytest.mark.outside


async def test_a_turn_said_to_hands_goes_through_its_proxy_to_the_mock_and_the_reply_is_spoken(mock_api: MockApi, hands: Hands) -> None:
    mock_api.script("1-hello", {"lastUserContains": ["say hello"], "bodyContains": [TURN]}, {"type": "text", "text": "Hello."})
    await hands.connected()
    await hands.say("say hello")
    await hands.until(lambda: "Hello." in "".join(hands.out.said()))
    # The turn reached the model through hands' proxy: the brain's request is one it observed.
    assert any(TURN in str(request.body) and "say hello" in str(request.body) for request in hands.sent)
    # The brain took brioche's certificate for Anthropic's API, and so took brioche for that API, as only its own turns
    # show: each names the prompt it answers, which Claude Code names to Anthropic's own API alone.
    assert BriocheSaid("brioche: answering the client's TLS for api.anthropic.com:443") in hands.recorded
    turns = [request.kind for request in hands.sent if isinstance(request.kind, MainTurn)]
    assert turns and all(turn.prompt is not None for turn in turns), turns
    # Each exchange is observed once: brioche records none of its own.
    exchanges = [request.exchange for request in hands.sent]
    assert len(exchanges) == len(set(exchanges)), exchanges
