"""The desk served: its driver (`hands.voice.engaged`) wired to the gate, the level's owner, the speaker's cues, and the
models it hears with, as the daemon runs it once the pipeline has opened its streams."""

from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from functools import partial
from typing import Protocol

from loguru import logger

from hands.clock import Clock
from hands.sessions.wide import WideEvent, annotate, begun, unit
from hands.voice import wake
from hands.voice.wakemodels import WakeModels
from hands.voice.cues import Cue, Played, level_cue
from hands.voice.desk import Leveled, Made
from hands.voice.engaged import Hears, drive, loaded
from hands.voice.ptt import PushToTalk
from hands.voice.shortcut import KeyEvent, Move
from hands.voice.listening import Driving
from hands.voice.turnlife import Ears
from hands.voice.wakeword import Word


class Microphone(Protocol):
    """The desk's microphone, as the desk hears it between turns."""

    @property
    def sample_rate(self) -> int: ...

    def overheard(self) -> AbstractAsyncContextManager[AsyncIterator[bytes]]: ...


class Speaker(Protocol):
    """The speaker attached now: it plays a cue at once."""

    def cue(self, cue: Cue) -> Played: ...


async def serve_desk(
    *,
    key: PushToTalk,
    levels: Driving,
    tapped: Callable[[Clock, Callable[[KeyEvent], None]], AbstractAsyncContextManager[None]],
    microphone: Microphone,
    speaker: Callable[[], Speaker],
    over_hands: Callable[[], bool],
    built_in: Callable[[], bool],
    models: WakeModels,
    word: Word,
    taken: Callable[[Move], Awaitable[None]],
    beat: Callable[[], None],
    record: Callable[[WideEvent], None],
    clock: Clock,
) -> None:
    """Drive the desk until cancelled: each move it makes goes to the gate, and what the gate takes it as to `taken`; each
    change of level plays its cue and is beaten, the driver having told `levels` of it as it stepped it. `over_hands`
    says whether hands' voice may be in what a turn opened now would keep, and `built_in` whether the desk's streams are
    on the Mac's own microphone and speakers: what the wake word is heard with (`hands.voice.wake.listening`)."""

    async def at_desk(made: Made) -> None:
        # [LAW:single-enforcer] a clip is ended only through `PushToTalk.end`, which the gate lets only the trigger that opened it do;
        # the desk names the turn open now, until `Made` carries the turn it decided about (promptctl-phone-app-9d9y.3hd).
        match made.move:
            case "stop" | "drop" | "timeout" as how:
                moves = key.end(key.gate.turn, how, made.by)
            case move:
                moves = key.move(move, made.by)
        # A Shift at the desk while the turn is the phone's made no moves: nothing of it is cued, said, or beaten.
        for move in moves:
            await taken(move)

    async def at_level(leveled: Leveled) -> None:
        cue = level_cue(leveled)
        logger.info(cue.line)
        speaker().cue(cue)
        beat()

    @asynccontextmanager
    async def hearing() -> AsyncGenerator[tuple[Ears, Hears]]:
        rate = microphone.sample_rate
        # [LAW:no-silent-failure] the models every level hears with are fetched and loaded once, beside the talk key as
        # the desk starts: one that cannot be is the run's failure, said with what could not be loaded, never a level
        # that fails as it is chosen.
        # [LAW:nothing-unseen] the fetch is its own unit of work, naming the files the wake word is heard with, those it
        # fetched, none once they are kept, and those it wrote over for bytes that were not the pinned ones.
        with unit("desk.fetched", record, begun(clock)):
            ensured = await models.ensured(word)
            files = ensured.files
            annotate(**ensured.facts())
        async with loaded(clock, rate, record, "desk.loaded") as ears, wake.loaded(clock, rate, files, word, record) as heard:
            yield ears, wake.listening(clock, heard, over_hands, built_in, record)

    await drive(clock, partial(tapped, clock), microphone.overheard, hearing, levels, at_desk, at_level, record)
