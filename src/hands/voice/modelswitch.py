"""The switch of the model hands runs on, as the voice tool and the phone's page both make it: weighed, said aloud, and
kept once the user has heard it and the caller has its answer.

Neutral to its callers: the brain's tools (`hands.voice.tools` weighs, `hands.voice.braindoor` says) and the phone's
gallery page (`hands.voice.gallerypage`) import it, so none depends on another.
"""

import functools
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from hands.core.playback import Labelled, hands_says
from hands.core.spoken import spoken_model
from hands.sessions.payload import Rejected
from hands.threads import off_loop
from hands.voice.offeredmodels import OfferedModel
from hands.voice.tool import After, Result


class Weighed(Protocol):
    """A model weighed for hands (hands.daemon.config.ModelChoice): its id, and the keep that runs hands on it."""

    @property
    def to(self) -> OfferedModel: ...

    def keep(self) -> None: ...


class OwnModel(Protocol):
    """The model hands itself runs on (hands.daemon.config.OwnModel): hands has no session id, so what targets it is a
    tool of its own, never a session's."""

    def running(self) -> OfferedModel: ...

    def weigh(self, model: OfferedModel) -> Weighed: ...


class BargedIn(Rejected):
    """A switch the user talked over while its line was said: hands did not switch."""


class Heard(Protocol):
    """The speaker as a model switch reads it: a line said, and whether the user heard it to its end."""

    async def heard(self, line: Labelled) -> bool: ...


@dataclass(frozen=True)
class Switch:
    """A switch of hands' model, weighed: the model it switches to, the one it runs on, and the keep, which restarts
    hands. [LAW:effects-at-boundaries] an effect run once the caller has its answer, holding nothing until then."""

    to: OfferedModel
    running_on: OfferedModel
    keep: Callable[[], Awaitable[None]]

    @property
    def result(self) -> Result:
        return {"running_on": self.running_on, "switching_to": self.to}

    @property
    def answer(self) -> After:
        """The switch as a caller is answered it: kept once the answer is written, and dropped where it cannot be, since
        the answer is the only telling of it."""
        return After(self.result, self, "dropped")

    async def __call__(self) -> None:
        await self.keep()


async def chosen(own: OwnModel, model: OfferedModel) -> Switch:
    """The switch to `model`, weighed. Raises Rejected where the switch is refused, and OSError where the file the model
    is kept in cannot be read."""
    choice = await off_loop(functools.partial(own.weigh, model), "weighing a model chosen for hands")
    switch = Switch(choice.to, own.running(), functools.partial(off_loop, choice.keep, "keeping a model chosen for hands"))
    return switch


class Switches:
    """The switches of hands' model a run says, by voice or on the phone's page: one under way at a time, from the
    moment its line is said until hands restarts on it, or it ends without a restart.

    [LAW:single-enforcer] the one owner of a switch under way, for every door that says one: a second said while one is
    under way would be cut off by the first one's restart, never kept and never said not to have happened, so it is
    refused before it is said.
    """

    def __init__(self, player: Heard) -> None:
        self._player = player
        self._underway: OfferedModel | None = None

    async def announced(self, switch: Switch) -> After:
        """The switch, said, once the user has heard it to its end: the line rides on its result, and the switch is kept
        as they were told, whether or not its answer can be written; a keep that then fails is said too. Raises
        Rejected where another switch is under way, and BargedIn where they talked over the line; neither is kept.

        [LAW:one-source-of-truth] the one order of a switch of model, by voice or on the phone's page alike: weighed
        (`chosen`), said, answered, then kept.
        """
        if self._underway is not None:
            raise Rejected(f"hands is switching to {self._underway} already, and is back on it in a few seconds")
        # [LAW:no-ambient-temporal-coupling] claimed with no await after the check, so no other switch on the loop comes
        # between them; held past a keep that succeeds, since hands is restarting then.
        self._underway = switch.to
        # [LAW:one-source-of-truth] the model said by the name hands says it by as it starts; the brain's answer keeps its id.
        line = f"Switching to {spoken_model(switch.to)}. I'll be back in a few seconds."
        try:
            # Kept only once heard: the keep restarts hands, which would cut the line off.
            if not await self._player.heard(hands_says(line)):
                raise BargedIn(f"The user spoke over hands saying it would switch to {switch.to}, so it did not switch.")
        except BaseException:
            self._underway = None
            raise
        return After({**switch.result, "said": line}, functools.partial(self._kept, switch), "run")

    async def _kept(self, switch: Switch) -> None:
        try:
            await switch()
        except Exception as error:
            self._underway = None
            # [LAW:no-silent-failure] a switch the user heard said, which did not happen, is heard not to have, whatever
            # stopped it; the door's event holds what it raised.
            await self._player.heard(hands_says(f"The switch to {spoken_model(switch.to)} did not happen: {error}"))
            raise
