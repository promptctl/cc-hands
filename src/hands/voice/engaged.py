"""The desk's ears and its driver: the models that hear the room between turns, and the one task that steps the desk
(`hands.voice.desk`) with everything it hears.

Two models listen to the desk's microphone, heard through the echo canceller as every buffer is: Silero's voice
activity detector says where speech starts and stops, a silence shorter than `STOP_SECS` being inside the speech, and
at each stop Smart Turn says whether it is the end of the turn or a pause inside it, from how the speech ended. A pause
it judges a thought still going holds the turn open; if the silence then runs on to Smart Turn's own `stop_secs`, the
turn ends there. They hear for engaged conversation and for the wake word alike, and the wake word's own model beside
them (`hands.voice.wake`).

The desk moves the gate with the held key's own moves, so the gate, the hold owner, and the cues take a turn the voice
opened as they take a held one: speech starting arms the microphone, speech confirmed opens the turn, a start that was
only a noise disarms it, and the end of the turn sends it. Listening, the desk keeps the second before each turn opens, so
the words said while the detector made sure of them are in it.
"""

import asyncio
import time
from collections import deque
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from typing import Any, Protocol, cast, get_args

import numpy as np
from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState
from pipecat.audio.turn.smart_turn.base_smart_turn import BaseSmartTurn
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams, VADState
from pipecat.metrics.metrics import TurnMetricsData

from hands.sessions.wide import WideEvent, annotate, count, unit
from hands.voice import loads
from hands.threads import settles_off_loop
from hands.voice.desk import LEVELS, Chosen, Level, Desk, Event, Heard, Keyed, Leveled, Made, SpeechStarted, SpeechStarting, SpeechStopped, TurnEnded, Woke, Woken, deadline, listens, step
from hands.voice.hold import Instant, KeyEvent, Move
from hands.voice.holds import Edge
from hands.voice.trigger import Choice


class Ears(Protocol):
    """What the edge asks of the models that listen: Silero's detector and Smart Turn, in one."""

    async def detect(self, audio: bytes) -> VADState: ...
    def heard(self, audio: bytes, speech: bool) -> bool: ...
    async def judge(self) -> tuple[bool, TurnMetricsData | None]: ...
    def clear(self) -> None: ...
    def afresh(self) -> None: ...


# The silence that ends speech: a pause between words is shorter, so it is inside the speech. At Pipecat's 0.2 s, tuned
# for Smart Turn, Smart Turn judged every such pause, and a phrase said before one read to it as complete, so each
# phrase was a turn (the real-model test in tests/test_engaged.py fails at 0.2). The price is that every turn ends this
# long after the speech does, before Smart Turn is asked.
STOP_SECS = 0.8


# Smart Turn v3.2 ported to MLX, fetched from the Hugging Face hub into its cache on the first load, at the revision
# measured for it.
SMART_TURN = "mlx-community/smart-turn-v3"
SMART_TURN_REVISION = "74af1538ebace37b0fcc667a6a8a3d902d73eee0"
# The port's weights, under the name its repository gives them: in the hub's cache once the port is fetched.
SMART_TURN_WEIGHTS = "model.safetensors"


class Unported(Exception):
    """Smart Turn's port could not be fetched or loaded, and why."""


def ported() -> Any:
    """Smart Turn's MLX port, fetched into the hub's cache unless it is there already, with every weight loaded."""
    # mlx-audio ships no types.
    from mlx_audio.vad import load  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]

    try:
        # [LAW:no-silent-failure] strict: a weight the port's loader no longer maps fails the load, where it would be
        # left at random and every verdict after it noise.
        return cast(Any, load(SMART_TURN, revision=SMART_TURN_REVISION, strict=True))
    except Exception as error:
        # [LAW:single-enforcer] every way the port is not got is one error, naming the port, which a switch is refused
        # with: the hub's errors, its client's, a strict load's, and MLX's on a damaged file are each a different class,
        # and mlx-audio, untyped, says which of them it raises nowhere.
        raise Unported(f"Smart Turn, {SMART_TURN} at {SMART_TURN_REVISION}: {error}") from error


class SmartTurn(BaseSmartTurn):
    """Smart Turn v3.2 on MLX: Pipecat's analyzer keeps the speech, times the silence, and cuts the segment, and the port
    judges it. The port is converted from the full-precision build, which Pipecat's benchmark puts ahead of the quantized
    one its own local analyzer runs on the CPU, and judges a segment in a fifth of the time."""

    def __init__(self, sample_rate: int) -> None:
        super().__init__(sample_rate=sample_rate)
        self._model = ported()

    def _predict_endpoint(self, audio_array: np.ndarray) -> dict[str, Any]:
        verdict = self._model.predict_endpoint(audio_array, sample_rate=self.sample_rate)
        return {"prediction": verdict.prediction, "probability": verdict.probability}


class Models:
    """Silero's voice activity detector, on the CPU as Pipecat ships it, and Smart Turn on MLX."""

    def __init__(self, sample_rate: int) -> None:
        self._vad = SileroVADAnalyzer(sample_rate=sample_rate, params=VADParams(stop_secs=STOP_SECS))
        self._vad.set_sample_rate(sample_rate)
        self._turn = SmartTurn(sample_rate)
        self._turn.set_sample_rate(sample_rate)
        # As Pipecat syncs it at each start: the speech Silero takes to be sure of is kept before the segment judged.
        self._turn.update_vad_start_secs(self._vad.params.start_secs)

    @property
    def vad_stop_secs(self) -> float:
        return self._vad.params.stop_secs

    async def detect(self, audio: bytes) -> VADState:
        return await self._vad.analyze_audio(audio)

    def heard(self, audio: bytes, speech: bool) -> bool:
        """Give Smart Turn the audio as it is heard; True once the silence after speech has run past `stop_secs`."""
        return self._turn.append_audio(audio, speech) == EndOfTurnState.COMPLETE

    async def judge(self) -> tuple[bool, TurnMetricsData | None]:
        state, metrics = await self._turn.analyze_end_of_turn()
        return state == EndOfTurnState.COMPLETE, metrics if isinstance(metrics, TurnMetricsData) else None

    def clear(self) -> None:
        self._turn.clear()

    def afresh(self) -> None:
        """Hear on as if nothing came before: Silero is quiet until sure of speech again, and Smart Turn holds none."""
        # Setting the detector's parameters puts it back to quiet, with no speech counted toward starting or stopping.
        self._vad.set_params(self._vad.params)
        self._turn.clear()

    async def close(self) -> None:
        """Shut down the thread each model runs on."""
        await self._vad.cleanup()
        await self._turn.cleanup()


@asynccontextmanager
async def loaded(sample_rate: int, emit: Callable[[WideEvent], None]) -> AsyncGenerator[Models]:
    """Both models, loaded off the loop as their own unit of work, for as long as the desk hears with them."""
    async with AsyncExitStack() as held:
        with loads.loading("desk.loaded", emit):
            # [LAW:nothing-unseen] the Smart Turn being loaded, before the load, so one that fails says which; and the
            # pause the detector ends speech at, as it was loaded with it.
            annotate(smart_turn=SMART_TURN, smart_turn_revision=SMART_TURN_REVISION)
            # On a thread of its own the process does not join at exit (hands.threads), as every model's load is; started
            # inside the unit, so the load's time and growth are all on it.
            loading = settles_off_loop(lambda: Models(sample_rate), "Models")
            held.push_async_callback(_let_go, loading)
            annotate(vad_stop_secs=(await asyncio.shield(loading)).vad_stop_secs)
        yield loading.result()


async def _let_go(loading: asyncio.Future[Models]) -> None:
    """A switch away while they load waits the load out, so what it built is let go of too."""
    await (await loading).close()


# [LAW:one-source-of-truth] each move the desk makes is counted on the desk's event, by the move's name and by the edge
# that made it, and each change of level by the level it went to.
_COUNTED: tuple[str, ...] = (
    *(name for literal in get_args(Move) for name in get_args(literal)),
    *(f"by_{edge.replace(' ', '_')}" for edge in get_args(Edge)),
    *(f"to_{level.replace(' ', '_')}" for level in LEVELS),
    "held_open", "ended_on_silence", "heard", "muted",
)

# Whether a buffer of the desk's microphone ends with the wake word (`hands.voice.wake`), heard afresh from it when its
# level has just started.
Hears = Callable[[bytes, bool], Awaitable[bool]]


@dataclass(frozen=True)
class Overheard:
    """A buffer of the desk's microphone, with the models it is heard with."""

    audio: bytes
    ears: Ears
    woken: Hears


async def drive(
    tapped: Callable[[Callable[[KeyEvent], None]], AbstractAsyncContextManager[None]],
    overheard: Callable[[], AbstractAsyncContextManager[AsyncIterator[bytes]]],
    hearing: Callable[[], AbstractAsyncContextManager[tuple[Ears, Hears]]],
    chosen: asyncio.Queue[Choice],
    on_move: Callable[[Made], Awaitable[None]],
    on_level: Callable[[Leveled], Awaitable[None]],
    emit: Callable[[WideEvent], None],
    clock: Callable[[], Instant] = time.monotonic,
) -> None:
    """Hand every move the desk makes to `on_move`, and every change of level to `on_level`, until cancelled.

    One task steps the desk, in the order things were heard, so the speech, the key, the taps, the wake word, the
    tools' choices, and Smart Turn's verdicts never race each other: a verdict is stepped as it is given, before any
    audio heard while the model thought, so it ends the turn it judged and no other. The key is read from the start;
    the room is heard once `hearing` has loaded the models it is heard with, and one that cannot be loaded fails the
    drive.
    """
    loop = asyncio.get_running_loop()
    events: asyncio.Queue[Event | Overheard | Choice] = asyncio.Queue()

    async def overhearing() -> None:
        # [LAW:no-ambient-temporal-coupling] each buffer carries the models it is heard with, so none is heard before
        # they are loaded, and the key never waits on them.
        async with hearing() as (ears, woken), overheard() as heard:
            async for audio in heard:
                events.put_nowait(Overheard(audio, ears, woken))

    async def choosing() -> None:
        while True:
            events.put_nowait(await chosen.get())

    desk = Desk()
    # The silence the last buffer was heard in, as the detector said it.
    detected = VADState.QUIET
    # The models the last buffer was heard with: None until they are loaded, while Smart Turn has heard nothing.
    ears: Ears | None = None
    heard: deque[Event] = deque()
    # The level the last buffer was heard at: off, where nothing has heard it, as the desk starts.
    overheard_at: Level = "off"
    async with tapped(events.put_nowait), asyncio.TaskGroup() as group:
        group.create_task(overhearing(), name="the desk's microphone, overheard")
        group.create_task(choosing(), name="the levels the tools chose")
        while True:
            # [LAW:nothing-unseen] one event for each stint at a level, counting every move, edge, and change of level
            # in it, emitted as the level changes.
            with unit("desk", emit, counts=_COUNTED):
                level = desk.level
                annotate(level=level)
                while desk.level == level:
                    if not heard:
                        match await events.get():
                            # [LAW:types-are-the-program] off, the microphone is closed between turns, and nothing
                            # hears the room.
                            case Overheard() if not listens(desk.level):
                                overheard_at = desk.level
                            case Overheard(audio=audio, ears=ears, woken=woken):
                                # Each model hears from where its level started, with nothing of the room before it.
                                begun = desk.level != overheard_at
                                overheard_at = desk.level
                                if begun:
                                    ears.afresh()
                                    detected = VADState.QUIET
                                state = await ears.detect(audio)
                                # Smart Turn hears speech only where the detector is sure of it, as Pipecat feeds it,
                                # and only with no key down, so the room it hears otherwise is trimmed as silence is.
                                silent_too_long = ears.heard(audio, not isinstance(desk.phase, Keyed) and state == VADState.SPEAKING)
                                # The wake word's model is run only where it can open a turn.
                                word = desk.level == "wake word" and await woken(audio, begun)
                                heard.extend([*_changed(detected, state, clock), *([Woken(clock())] if word else []), *([TurnEnded("silence")] if silent_too_long else [])])
                                detected = state
                            case Choice(level=chose, was=was_level):
                                # A tool call cancelled while it waited, the user cutting the brain off, is answered by
                                # no one; its choice is the user's all the same.
                                if not was_level.cancelled():
                                    was_level.set_result(desk.level)
                                heard.append(Chosen(chose))
                            case event:
                                heard.append(event)
                        continue
                    each = heard.popleft()
                    was = desk
                    desk, acts = step(desk, each)
                    # [LAW:types-are-the-program] the wake word's own speech is no part of what is asked: heard afresh
                    # from where it opened the turn, what is asked starts as any speech does, and Smart Turn judges it alone.
                    if isinstance(desk.phase, Woke) and not isinstance(was.phase, Woke):
                        # Only a buffer heard with the models wakes the desk.
                        assert ears is not None
                        ears.afresh()
                        detected = VADState.QUIET
                    # Speech the desk ignored while the key was down starts for it as the key comes up, so a tap of
                    # Shift as the user began does not lose what they say.
                    if isinstance(was.phase, Keyed) and not isinstance(desk.phase, Keyed):
                        detected = VADState.QUIET
                    for act in acts:
                        match act:
                            case "judge":
                                # Only speech the detector heard is judged, and it is heard with the models.
                                assert ears is not None
                                if await _judged(ears, emit):
                                    heard.appendleft(TurnEnded("verdict"))
                                else:
                                    count(held_open=1)
                            case Made(move=move, by=by):
                                count(**{move: 1, f"by_{by.replace(' ', '_')}": 1})
                                if move == "stop" and each == TurnEnded("silence"):
                                    count(ended_on_silence=1)
                                # [LAW:no-ambient-temporal-coupling] Smart Turn starts each turn with no speech of the
                                # last one in it, as Pipecat clears it as each turn ends; before the models load it has
                                # heard none.
                                if move in ("stop", "drop", "timeout") and ears is not None:
                                    ears.clear()
                                await on_move(act)
                            case Leveled(was=was_level, now=now, by=chooser):
                                count(**{f"to_{now.replace(' ', '_')}": 1})
                                # [LAW:nothing-unseen] each change of level is its own event: from, to, and by.
                                with unit("desk.leveled", emit):
                                    annotate(was=was_level, now=now, by=chooser)
                                await on_level(act)
                    for at, later in deadline(was.phase, desk.phase):
                        loop.call_at(at, events.put_nowait, later)


def _changed(was: VADState, now: VADState, clock: Callable[[], Instant]) -> tuple[Heard, ...]:
    """What the detector's state moving from `was` to `now` says of the speech."""
    match was, now:
        case VADState.QUIET, VADState.STARTING:
            return (SpeechStarting(),)
        case VADState.QUIET | VADState.STARTING, VADState.SPEAKING:
            return (SpeechStarted(clock()),)
        case VADState.STARTING | VADState.SPEAKING | VADState.STOPPING, VADState.QUIET:
            return (SpeechStopped(),)
        case _:
            return ()


async def _judged(ears: Ears, emit: Callable[[WideEvent], None]) -> bool:
    """Ask Smart Turn whether the speech that just stopped ends the turn: its own unit of work, with its verdict."""
    with unit("desk.judged", emit):
        complete, metrics = await ears.judge()
        # [LAW:nothing-unseen] the verdict, and how sure of it the model was; no metrics is a segment with no audio in it.
        annotate(complete=complete, probability=None if metrics is None else metrics.probability, inference_ms=None if metrics is None else metrics.e2e_processing_time_ms)
        return complete
