"""The assembled brain, wired: a turn said to hands goes through hands' proxy to the mock of Anthropic's API, and the brain's
reply is spoken. The happy path and nothing else, the one smoke test of the brain (promptctl-testing-fxnj caps the smoke
layer): every decision hands makes of what Claude Code does is asserted in the unit layer, on Claude Code's turns as
recorded, and what hands relies on of its terminal is tests/test_terminal_contract.py's."""

from collections import Counter

import pytest

from pinnedbrain import TURN, Hands, MockApi
from pinnedbrain import claude_code, hands, mock_api, mock_binary  # pyright: ignore[reportUnusedImport]
from hands.core.wire import MainTurn

pytestmark = pytest.mark.outside


async def test_a_turn_said_to_hands_goes_through_its_proxy_to_the_mock_and_the_reply_is_spoken(mock_api: MockApi, hands: Hands) -> None:
    mock_api.script("1-hello", {"lastUserContains": ["say hello"], "bodyContains": [TURN]}, {"type": "text", "text": "Hello."})
    await hands.connected()
    await hands.say("say hello")
    await hands.until(lambda: "Hello." in "".join(hands.out.said()))
    # The turn reached the model through hands' proxy: the brain's request is one it observed.
    [turn] = [request for request in hands.sent if TURN in str(request.body) and "say hello" in str(request.body)]
    # Its main turn names its prompt, as Claude Code names it only to Anthropic's own API: the brain, given hands' proxy as
    # its HTTPS proxy and no base URL, reached https://api.anthropic.com through the proxy and trusted the authority the
    # proxy ended its TLS with, so it took the proxy for the API itself.
    assert isinstance(turn.kind, MainTurn) and turn.kind.prompt is not None, turn.kind
    # Each exchange is recorded once, the turn's among them, by the one recorder every request through the proxy passes.
    await hands.until(lambda: any(ended.exchange == turn.exchange for ended in hands.exchanged))
    assert all(count == 1 for count in Counter(ended.exchange for ended in hands.exchanged).values()), hands.exchanged
    assert {ended.exchange for ended in hands.exchanged} <= {request.exchange for request in hands.sent}
