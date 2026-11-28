"""A switch of hands' model as a run says it: one under way at a time, from its line until hands restarts on it, so a
second is refused before it is said, and one that ends without a restart frees the way."""

import re

import pytest

from hands.core.playback import Labelled
from hands.core.runningon import RunningOn
from hands.sessions.payload import Rejected
from hands.voice.microphone import Devices
from hands.voice.modelswitch import BargedIn, Switch, Switches
from hands.voice.offeredmodels import OFFERED_MODELS, OfferedModel
from hands.voice.system import Started, system_text


class Hearing:
    """The speaker as a switch reads it: each line said, heard to its end unless `heard` says otherwise."""

    def __init__(self, *heard: bool) -> None:
        self.heard_to_end = list(heard)
        self.said: list[str] = []

    async def heard(self, line: Labelled) -> bool:
        self.said.append(line.text)
        return self.heard_to_end.pop(0) if self.heard_to_end else True


def switch(to: OfferedModel, kept: list[OfferedModel], refusal: str | None = None) -> Switch:
    async def keep() -> None:
        if refusal is not None:
            raise Rejected(refusal)
        kept.append(to)

    return Switch(to, "claude-sonnet-5-5", keep)


async def test_a_switch_asked_for_while_one_is_under_way_is_refused_before_it_is_said_and_after_that_one_is_kept_too() -> None:
    hearing, kept = Hearing(), list[OfferedModel]()
    switches = Switches(hearing)
    first = await switches.announced(switch("claude-opus-5-5", kept))
    with pytest.raises(Rejected, match="hands is switching to claude-opus-5-5 already"):
        await switches.announced(switch("claude-fable-5-1", kept))
    await first.effect()
    # Kept, hands is restarting: a switch asked for now would be cut off by it.
    with pytest.raises(Rejected, match="hands is switching to claude-opus-5-5 already"):
        await switches.announced(switch("claude-fable-5-1", kept))
    assert (hearing.said, kept) == (["Switching to Opus 5.5. I'll be back in a few seconds."], ["claude-opus-5-5"])


async def test_a_switch_talked_over_or_whose_keep_fails_frees_the_way_for_the_next() -> None:
    hearing, kept = Hearing(False), list[OfferedModel]()
    switches = Switches(hearing)
    with pytest.raises(BargedIn):
        await switches.announced(switch("claude-opus-5-5", kept))
    failing = await switches.announced(switch("claude-opus-5-5", kept, refusal="config.toml was saved by hand"))
    with pytest.raises(Rejected):
        await failing.effect()
    await (await switches.announced(switch("claude-fable-5-1", kept))).effect()
    assert kept == ["claude-fable-5-1"]
    assert hearing.said == [
        "Switching to Opus 5.5. I'll be back in a few seconds.",
        "Switching to Opus 5.5. I'll be back in a few seconds.",
        "The switch to Opus 5.5 did not happen: config.toml was saved by hand",
        "Switching to Fable 5.1. I'll be back in a few seconds.",
    ]


@pytest.mark.parametrize("model", OFFERED_MODELS)
async def test_a_switch_and_one_that_did_not_happen_say_the_model_by_the_name_hands_says_it_by_as_it_starts(model: OfferedModel) -> None:
    hearing = Hearing()
    failing = await Switches(hearing).announced(switch(model, [], refusal="config.toml was saved by hand"))
    with pytest.raises(Rejected):
        await failing.effect()
    # The name is the one the start says, read off its line, so a change to how hands says a model as it starts is a
    # change to the switch's lines too.
    started = Started(after_crash=False, running_on=RunningOn(model, None), devices=Devices(input="MacBook Pro Microphone", output="MacBook Pro Speakers"))
    [name] = re.findall(r"^hands is up on (.+?), ", system_text(started))
    assert hearing.said == [f"Switching to {name}. I'll be back in a few seconds.", f"The switch to {name} did not happen: config.toml was saved by hand"]
    assert not any(model in line for line in hearing.said)
