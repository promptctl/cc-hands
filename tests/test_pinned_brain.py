"""The assembled brain, wired: a turn said to hands goes through hands' proxy to the mock of Anthropic's API, and the brain's
reply is spoken. The happy path and nothing else, the one smoke test of the brain (promptctl-testing-fxnj caps the smoke
layer): every decision hands makes of what Claude Code does is asserted in the unit layer, on Claude Code's turns as
recorded, and what hands relies on of its terminal is tests/test_terminal_contract.py's."""

from datetime import timedelta

import pytest

from hands.sessions.wide import WideEvent

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
    # [LAW:no-ambient-temporal-coupling] the pin ran the brain's SessionStart command and, slowed, its input came up
    # without it, but the turn's request waited on it: the turn ended only after hands had answered the start, which it
    # does once it holds the session the start names (docs/claude-code-behavior.md, measured on 2.1.308).
    [started] = [entry for entry in hands.recorded if isinstance(entry, WideEvent) and entry.event == "brain.hook"]
    [turn] = [entry for entry in hands.recorded if isinstance(entry, WideEvent) and entry.event == "brain.turn"]
    assert (started.outcome, started.facts["source"]) == ("ok", "startup")
    assert turn.started_at < started.started_at <= turn.started_at + timedelta(milliseconds=turn.duration_ms)
