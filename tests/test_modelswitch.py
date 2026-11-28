"""A switch of hands' model as a run says it: one under way at a time, from its line until hands restarts on it, so a
second is refused before it is said, and one that ends without a restart frees the way."""

import pytest

from hands.core.playback import Labelled
from hands.sessions.payload import Rejected
from hands.voice.modelswitch import BargedIn, OfferedModel, Switch, Switches


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
    assert (hearing.said, kept) == (["Switching to claude-opus-5-5. I'll be back in a few seconds."], ["claude-opus-5-5"])


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
        "Switching to claude-opus-5-5. I'll be back in a few seconds.",
        "Switching to claude-opus-5-5. I'll be back in a few seconds.",
        "The switch to claude-opus-5-5 did not happen: config.toml was saved by hand",
        "Switching to claude-fable-5-1. I'll be back in a few seconds.",
    ]
