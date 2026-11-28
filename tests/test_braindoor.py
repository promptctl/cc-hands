"""A voice turn: the acts, the same to any caller, served the brain with what a call of each is to the turn."""

from pathlib import Path

import pytest

from conftest import new_sessions
from hands.clock import SYSTEM
from hands.sessions.home import Home
from hands.brain.usage import Usage
from hands.core.session import SessionId
from hands.voice.player import Player
from hands.voice.refocus import Refocus
from hands.core.playback import Labelled
from hands.voice.modelswitch import Switch, Switches
from hands.voice.tool import After, Answer, Result, result_of, tool
from hands.voice.tools import usage_tool
from hands.voice.braindoor import Part, BrainDoor, defaulting_to_focus, brain_door
from served import every_act, voiced


def test_the_acts_are_the_brains_tools_but_the_voices_own_and_none_says_what_only_a_voice_turn_does(tmp_path: Path) -> None:
    sessions, home = new_sessions(clock=SYSTEM), Home(tmp_path)
    acts = every_act(sessions, home)
    turn = brain_door(sessions, home, Player(lambda _entry: None), Refocus(sessions, home, lambda _entry: None), lambda: None, Switches(Player(lambda _entry: None)))
    served = {voiced.tool.name: voiced for voiced in turn.serve(acts)}
    assert set(served) - {act.name for act in acts} == {"focus_session", "resume", "skip", "repeat", "stay_silent"}
    for act in acts:
        # With no voice turn there is no focus to stand in for a session, so an act that takes one requires it.
        assert "session" not in act.properties or "session" in act.required, act.name
    for name, part in turn.parts.items():
        for said in part.says:
            assert said in served[name].tool.description
            assert not [act.name for act in acts if said in act.description], said


def test_a_voice_turn_refuses_a_part_for_an_act_it_is_not_served(tmp_path: Path) -> None:
    turn = BrainDoor({"use_model": Part("silence")}, (), Home(tmp_path), lambda: None, Switches(Player(lambda _entry: None)))
    with pytest.raises(ValueError, match="use_model"):
        turn.serve([])


async def test_every_tool_a_voice_turn_serves_plays_the_working_cue_as_it_is_called_but_staying_silent(tmp_path: Path) -> None:
    sessions, home = new_sessions(clock=SYSTEM), Home(tmp_path)
    cued: list[str] = []
    turn = brain_door(sessions, home, Player(lambda _entry: None), Refocus(sessions, home, lambda _entry: None), lambda: cued.append("working"), Switches(Player(lambda _entry: None)))
    served = {name: each.tool for name, each in voiced(turn, [usage_tool(Usage(lambda: SessionId("brain")))]).items()}
    # context_usage was the one act left uncued, and is cued like every other now.
    for name, arguments in (("context_usage", {}), ("repeat", {}), ("stay_silent", {})):
        cued.clear()
        await served[name].body(**arguments)
        assert cued == ([] if name == "stay_silent" else ["working"]), name


def test_an_act_whose_session_is_optional_is_refused_as_the_tools_are_built(tmp_path: Path) -> None:
    async def peek(session: str = "") -> Result:
        """Look at a session.

        Args:
            session: The session's id.
        """
        return {}

    with pytest.raises(TypeError, match="peek takes a session"):
        defaulting_to_focus(tool(peek), Home(tmp_path))


def test_a_voice_turn_refuses_an_act_named_as_one_of_its_own_tools(tmp_path: Path) -> None:
    async def repeat() -> Result:
        """Repeat the last command."""
        return {}

    turn = brain_door(new_sessions(clock=SYSTEM), Home(tmp_path), Player(lambda _entry: None), Refocus(new_sessions(clock=SYSTEM), Home(tmp_path), lambda _entry: None), lambda: None, Switches(Player(lambda _entry: None)))
    with pytest.raises(ValueError, match="two tools of one name: repeat"):
        voiced(turn, [tool(repeat)])


class Hearing:
    """The speaker as a switch reads it: each line said, and heard to its end."""

    def __init__(self) -> None:
        self.said: list[str] = []

    async def heard(self, line: Labelled) -> bool:
        self.said.append(line.text)
        return True


async def test_a_switch_of_model_any_act_answers_is_said_before_it_is_answered_and_any_other_effect_is_served_as_it_was(tmp_path: Path) -> None:
    hearing = Hearing()

    async def keep() -> None:
        pass

    async def unswitched() -> None:
        pass

    async def retune() -> Answer:
        """Retune hands."""
        return Switch("claude-opus-5-5", "claude-sonnet-5-5", keep).answer

    async def later() -> Answer:
        """Do something once answered."""
        return After({"done": True}, unswitched, "dropped")

    turn = BrainDoor({}, (), Home(tmp_path), lambda: None, Switches(hearing))
    served = {each.tool.name: each.tool for each in turn.serve([tool(retune), tool(later)])}
    switched = await served["retune"].body()
    assert hearing.said == ["Switching to Opus 5.5. I'll be back in a few seconds."]
    assert isinstance(switched, After) and switched.unanswered == "run"
    assert result_of(switched)["said"] == hearing.said[0]
    assert await served["later"].body() == After({"done": True}, unswitched, "dropped")
    assert hearing.said == ["Switching to Opus 5.5. I'll be back in a few seconds."]
