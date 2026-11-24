"""What a call of one of hands' tools is to a voice turn, and the voice's own tools: the brain door's part of a tool,
which the act never holds.

[LAW:types-are-the-program] an act (`hands.voice.tool.Tool`) is built with no player, no refocus and no cue, so its
body cannot speak a voice turn's line, wait on a heard line, or move the focus. The `BrainDoor` does those around the
calls it serves the brain: it plays the working cue, lets a call naming no session act on the focus, says the voice
turn's own sentences in a description, has every switch of model heard before it is kept, and serves the tools that
mean nothing outside a voice turn.
"""

import asyncio
import functools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

from hands.core import playback
from hands.core.playback import as_told
from hands.core.session import SessionId
from hands.sessions.focus import Unreadable, focused
from hands.sessions.home import Home
from hands.sessions.payload import Rejected
from hands.sessions.registry import Sessions
from hands.voice.modelswitch import BargedIn, Switch, Switches
from hands.voice.player import Player
from hands.voice.readback import spoken_name
from hands.voice.refocus import NotRunning, Refocus, move_focus
from hands.voice.tool import After, Answer, Body, Result, Tool, result_of, tool, with_result

# What a call means to the conversation. "reply": the model is asked to go on once it has the result. "silence": the call
# is the whole reply and hands says something for it: a draft read back, playback resumed, the model switched. "nothing":
# the call is the whole reply and nothing is said for it, which only words not said to the model may end in. Either whole
# reply that hands back an error is the model's to answer: a refused call did nothing, and only the model can retry it.
Then = Literal["reply", "silence", "nothing"]


@dataclass(frozen=True)
class Voiced:
    """A tool as the brain is served it, and what a call of it means to the turn."""

    tool: Tool[Answer]
    then: Then


def whole(silences: Sequence[bool]) -> bool:
    """Whether a reply's calls were the whole of it: every one of them silent. One that replied or was refused is the model's to answer."""
    return bool(silences) and all(silences)


def silent(voiced: Voiced, result: Result) -> bool:
    """Whether the call is the whole reply: a silence or nothing tool's, unless it was refused."""
    return voiced.then != "reply" and "error" not in result


def _plain(body: Body[Answer]) -> Body[Answer]:
    return body


@dataclass(frozen=True)
class Part:
    """What a call of one act is to a voice turn."""

    then: Then = "reply"
    # Paragraphs true only in a voice turn, said after the act's own description.
    says: tuple[str, ...] = ()
    # What the voice does around the act's call.
    around: Callable[[Body[Answer]], Body[Answer]] = field(default=_plain)


@dataclass(frozen=True)
class BrainDoor:
    """The brain door's part of hands' tools: what a call of each act is to a voice turn, by the act's name, and the
    voice's own tools."""

    parts: Mapping[str, Part]
    own: Sequence[Voiced]
    home: Home
    # Says that hands is acting, each time the model calls an act.
    acting: Callable[[], None]
    # Says a switch of model, one under way at a time.
    switches: Switches

    def serve(self, acts: Sequence[Tool[Answer]]) -> list[Voiced]:
        """The acts as the brain is served them, then the voice's own tools. [LAW:single-enforcer] every act is served
        the same way: a session left unnamed taken for the focus, its own part around it, and any switch of model it
        answers heard first; and every tool served is cued but the one whose call is no act. Raises ValueError for a part that names none of the acts, which a renamed
        act would leave behind, served as a reply with nothing said of it."""
        unserved = self.parts.keys() - {act.name for act in acts}
        if unserved:
            raise ValueError(f"the voice turn has parts for tools it is not served: {', '.join(sorted(unserved))}")
        served = [*(self._voiced(act, self.parts.get(act.name, Part())) for act in acts), *self.own]
        names = [each.tool.name for each in served]
        if len(set(names)) != len(names):
            # [LAW:no-silent-failure] the doors take tools by name, so a second of one name would hide the first.
            raise ValueError(f"the voice turn would serve two tools of one name: {', '.join(sorted({name for name in names if names.count(name) > 1}))}")
        # Not cued: a call that is the whole reply with nothing said for it, staying silent, is the choice not to act.
        return [each if each.then == "nothing" else replace(each, tool=_cued(each.tool, self.acting)) for each in served]

    def _voiced(self, act: Tool[Answer], part: Part) -> Voiced:
        told = replace(act, description="\n\n".join((act.description, *part.says)), body=part.around(_heard_first(self.switches, act.body)))
        return Voiced(defaulting_to_focus(told, self.home), part.then)


def brain_door(sessions: Sessions, home: Home, player: Player, refocus: Refocus, acting: Callable[[], None], switches: Switches) -> BrainDoor:
    """The voice of the run: what its speaker says around a call, the focus it moves, and the cue it plays. A part's
    sentences are true only of a voice turn, or only of the brain, which is hands."""
    whole_reply = "Calling it is the whole reply: add no words of your own."
    parts = {
        "stage_draft": Part("silence", (f"Hands reads the draft back to the user as it will be typed. {whole_reply}",)),
        "amend_draft": Part("silence", (f"Hands reads the new draft back as it will be typed, or, when it is long, the sentences that changed as they now read. {whole_reply}",)),
        "use_model": Part("silence", ("Hands says it is switching, and switches once the user has heard that: calling it is the whole reply, so add no words of your own.",)),
        "tell_turn": Part("reply", ("The session told becomes the focus, as a session whose turn hands tells as it finishes does.",), _refocusing(refocus)),
        "expand": Part(says=("Tell it in your own words, in spoken form. There is no word-for-word reading: say what code does and what a command found, and never read out code, output, paths, or hashes as written, however much the user asks for.",)),
        "attention": Part(says=("For quiet on, say in a few words that you will keep quiet, never nothing, so they know it took.",)),
        "hear_voices": Part(says=("Say nothing before calling it. What you say after it is heard after the last voice, so keep it to a few words asking which they would like. After a barge-in too, call it again to hear them again: going back or repeating says the words again in the voice you speak in, not in theirs.",)),
        "use_voice": Part(says=("What you say next is in it.",)),
        "context_usage": Part(says=("Call it too when you need to know how full your own context is: a token total in your system reminders is not your usage. A subagent you start has its own context, which this does not measure.",)),
    }
    own = [
        Voiced(focus_session_tool(sessions, home), "reply"),
        *(Voiced(each, "silence") for each in playback_tools(player)),
        stay_silent_tool(),
    ]
    return BrainDoor(parts, own, home, acting, switches)


def _cued(tool: Tool[Answer], acting: Callable[[], None]) -> Tool[Answer]:
    """The tool, saying by `acting` that hands is acting each time the model calls it, before its body runs."""

    @functools.wraps(tool.body)
    async def call(**arguments: object) -> Answer:
        acting()
        return await tool.body(**arguments)

    return replace(tool, body=call)


def _heard_first(switches: Switches, body: Body[Answer]) -> Body[Answer]:
    """The act, a switch of model it answers said, and answered once the user has heard it, so the keep that restarts
    hands never cuts the line off; any other answer as it was. A barge-in over the line leaves the switch unkept, and
    the brain is told to ask again; a switch refused as another is under way is never said.

    [LAW:dataflow-not-control-flow] taken by what the act answers, never by its name: whichever act answers a switch, it
    is heard before it restarts hands.
    """

    @functools.wraps(body)
    async def call(**arguments: object) -> Answer:
        match await body(**arguments):
            case After(effect=Switch() as switch):
                try:
                    return await switches.announced(switch)
                except BargedIn as error:
                    return {"error": f"{error} Ask whether they still want it."}
                except Rejected as error:
                    return {"error": str(error)}
            case answer:
                return answer

    return call


def _refocusing(refocus: Refocus) -> Callable[[Body[Answer]], Body[Answer]]:
    """tell_turn in a voice turn: told as a turn that finishes is told, so the focus moves to the session told as it
    does then, recorded as that move is. A refused call told nothing, and moves nothing."""

    def around(body: Body[Answer]) -> Body[Answer]:
        # [LAW:one-source-of-truth] the session told is the one the call names: the focus has stood in for it by now.
        @functools.wraps(body)
        async def call(session: str, **arguments: object) -> Answer:
            answer = await body(session=session, **arguments)
            match result_of(answer):
                case {"error": _}:
                    pass
                case _:
                    await refocus(SessionId(session))
            return answer

        return call

    return around


def defaulting_to_focus(tool: Tool[Answer], home: Home) -> Tool[Answer]:
    """The tool, where its schema requires a session, with the session made optional: a call that names none acts on the
    focused one. A tool that acts on no one session is served as it is."""
    # [LAW:single-enforcer] the one place a session left unnamed becomes the focus, for every tool that acts on one, so
    # what "the focus" means is said once and the model never has to remember which session that is.
    # [LAW:one-source-of-truth] which tools act on a session is read off their schemas, never listed beside them.
    match tool.properties.get("session"):
        case None:
            return tool
        case {"description": str(line)} as named if "session" in tool.required:
            pass
        case named:
            # [LAW:no-silent-failure] refused as the daemon builds its tools: an act whose session is optional, or
            # undescribed, would never be given the focus.
            raise TypeError(f"{tool.name} takes a session {named!r} the focus cannot stand in for: it must be required and described")

    async def call(session: object = "", **arguments: object) -> Answer:
        if not _unnamed(session):
            return await tool.body(session=session, **arguments)
        match await asyncio.to_thread(focused, home):
            case None:
                return {"error": "no session was named and none is focused: ask the user which session they mean"}
            case Unreadable(reason):
                return {"error": f"no session was named, and which one is focused cannot be read: {reason}"}
            case focus:
                # [LAW:nothing-unseen] the session the focus stood in for rides on the result, so its event says where the call went.
                return with_result(await tool.body(session=focus, **arguments), lambda result: {**result, "focused_session": focus})

    properties = {**tool.properties, "session": {**named, "description": f"{line} Empty for the focused session."}}
    return replace(tool, properties=properties, required=tuple(name for name in tool.required if name != "session"), body=call)


def _unnamed(session: object) -> bool:
    """Whether the model named no session: empty, or null, which some models send for an argument left empty."""
    return session in ("", None)


def focus_session_tool(sessions: Sessions, home: Home) -> Tool[Result]:
    async def focus_session(session: str) -> Result:
        """Make a session the one the user is talking to: what they say for a session without naming one goes to it.

        Call this when the user says to focus a session, switch or move to one, or work in one, such as "switch to
        cc-hands" or "focus the laws session". Do not ask them to confirm it. It holds from the next thing they say, and
        across restarts, until they move it; a session they name still gets what they say to it, focused or not. Say
        the returned readback.

        Args:
            session: The session's id, from list_sessions. Empty to focus none, when the user says to stop working in one.
        """
        try:
            to = None if session == "" else SessionId(session)
            await move_focus(sessions, home, to)
        except (Rejected, NotRunning, OSError) as error:
            return {"error": str(error)}
        return {"readback": "No session is focused now." if to is None else f"Now on {spoken_name(sessions, to)}."}

    return tool(focus_session)


def playback_tools(player: Player) -> list[Tool[Result]]:
    """Going back over what was said: hands says it again from where the speaker was, never the model from memory.

    [LAW:nothing-unseen] each call's result is what hands said for it and how many cut-off readings still wait, so its
    event holds what was heard and where it left playback.
    """

    async def resume() -> Result:
        """Go back to what you were saying when the user cut in, and say it from where it stopped.

        Hands says it, exactly as it was said, from the start of the sentence that was cut off. Calling it is the whole
        reply: add no words of your own, and never retell what you remember saying.
        """
        return {"said": as_told(await player.act(playback.resume)), "waiting": player.waiting}

    async def skip() -> Result:
        """Skip the sentence the user cut in on, and go on with what you were saying from the one after it.

        Hands says the rest. Calling it is the whole reply: add no words of your own.
        """
        return {"said": as_told(await player.act(playback.skip)), "waiting": player.waiting}

    async def repeat() -> Result:
        """Say again the last thing you said, whole, exactly as it was said.

        Hands says it. Calling it is the whole reply: add no words of your own, and never say it again yourself.
        """
        return {"said": as_told(await player.act(playback.repeat)), "waiting": player.waiting}

    return [tool(resume), tool(skip), tool(repeat)]


def stay_silent_tool() -> Voiced:
    async def stay_silent() -> Result:
        """End your turn with nothing said, because what was just heard was not said to you.

        In a turn the user asked and did not speak over, if you acted and they have heard nothing since, hands refuses the
        silence once, and you say what you did.
        """
        # [LAW:no-silent-failure] the choice not to answer is still a tool's event in the audit log, and with the model
        # not run on the result, nothing follows it to the speaker.
        return {"silent": True}

    # [LAW:types-are-the-program] the one whole reply nothing is said for: the stage refuses it to a turn that acted.
    return Voiced(tool(stay_silent), "nothing")
