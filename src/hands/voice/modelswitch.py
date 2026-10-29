"""The switch of the model hands runs on, as the voice tool and the phone's page both make it: weighed, said aloud, and
kept once the user has heard it.

Neutral to both callers: the brain's tools (`hands.voice.tools`) and the phone's gallery page
(`hands.voice.gallerypage`) import it, so neither depends on the other.
"""

import asyncio
import functools
from collections.abc import Callable
from typing import Protocol

from hands.sessions.payload import Rejected
from hands.threads import off_loop
from hands.voice.tool import Result


class OwnModel(Protocol):
    """The model hands itself runs on (hands.daemon.config.OwnModel): hands has no session id, so what targets it is a
    tool of its own, never a session's."""

    # [LAW:single-enforcer] one lock over every switch of the model: a switch is weighed, said, and kept before another
    # is weighed, so each weighs the file as the one before it left it.
    changing: asyncio.Lock

    def running(self) -> str: ...

    def weigh(self, model: str) -> Callable[[], None]: ...


class Heard(Protocol):
    """The speaker as a model switch reads it: a line said, and whether the user heard it to its end."""

    async def heard(self, sentence: str) -> bool: ...


async def switch_model(own: OwnModel, player: Heard, model: str) -> Result:
    """Run hands on `model`: weighed, the switch said aloud, and kept once the user has heard it, since keeping restarts hands.

    [LAW:one-source-of-truth] the one policy for a switch of model, by voice or on the phone's page alike: the order of
    weighing, saying, and keeping is the same whoever asks.
    """
    async with own.changing:
        try:
            keep = await off_loop(functools.partial(own.weigh, model), "weighing a model chosen by voice")
        except Rejected as error:
            return {"error": str(error)}
        line = f"Switching to {model.strip()}. I'll be back in a few seconds."
        # [LAW:no-ambient-temporal-coupling] written only once heard: the edit restarts hands, which would cut the line off.
        if not await player.heard(line):
            return {"error": f"The user spoke over hands saying it would switch to {model.strip()}, so it did not switch.", "spoken_over": True}
        try:
            await off_loop(keep, "keeping a model chosen by voice")
        except (Rejected, OSError) as error:
            return {"error": str(error)}
    return {"said": line, "running_on": own.running(), "switching_to": model.strip()}
