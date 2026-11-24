"""The desk's ears and its driver: the models that hear the room between turns, and the one task that steps the desk
(`hands.voice.desk`) with everything it hears.

Two models listen to the desk's microphone, heard through the echo canceller as every buffer is: Silero's voice
activity detector says where speech starts and stops, a silence shorter than `STOP_SECS` being inside the speech, and
at each stop Smart Turn says whether it is the end of the turn or a pause inside it, from how the speech ended. A pause
it judges a thought still going holds the turn open; if the silence then runs on to Smart Turn's own `stop_secs`, the
turn ends there. They hear for engaged conversation and for the wake word alike, and the wake word's own model beside
them (`hands.voice.wake`). What is done with the two models for a turn the voice opened is `hands.voice.turnlife`, which
the desk runs and anything else that hears a person speak can; this module is the desk's own part, its key, its taps and
its level.

The desk moves the gate with the shortcut's own moves, so the gate, the clip owner, and the cues take a turn the voice
opened as they take a held one: speech starting arms the microphone, speech confirmed opens the turn, a start that was
only a noise disarms it, and the end of the turn sends it. Listening, the desk keeps the second before each turn opens, so
the words said while the detector made sure of them are in it.
"""

import asyncio
from collections import deque
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from typing import Any, Literal, cast, get_args

import numpy as np
from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState
from pipecat.audio.turn.smart_turn.base_smart_turn import BaseSmartTurn
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams, VADState
from pipecat.metrics.metrics import TurnMetricsData

from hands.clock import Clock
from hands.sessions import wide
from hands.sessions.wide import WideEvent, annotate, count, unit
from hands.voice import loads, voiceturn
from hands.threads import settles_off_loop
from hands.voice.clips import Trigger
from hands.voice.desk import Chosen, Desk, Event, Keyed, Leveled, Made, ModelsLoaded, Woken, listens, step, turn_of
from hands.voice.listening import LEVELS, Choice, Driving, Level
from hands.voice.shortcut import KeyEvent, Move
from hands.voice.turnlife import Ears, Life
from hands.voice.voiceturn import Judging, TurnEnded


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

    def __init__(self, clock: Clock, sample_rate: int) -> None:
        super().__init__(sample_rate=sample_rate)
        self.set_sample_rate(sample_rate)
        self._model = ported()
        # [LAW:no-ambient-temporal-coupling] a warm-up: the port builds its Metal kernels, and MLX its stream for the
        # thread, on the first verdict there, so one is given on silence as it loads, on the thread every verdict runs on,
        # and a person's first turn is judged at every turn's latency. The port pads or trims every segment to the same
        # frames, so silence builds exactly the kernels every verdict runs.
        started = clock.now()
        self._model_executor.submit(self._predict_endpoint, np.zeros(sample_rate, dtype=np.float32)).result()
        self.warmup_secs = clock.now() - started

    def _predict_endpoint(self, audio_array: np.ndarray) -> dict[str, Any]:
        verdict = self._model.predict_endpoint(audio_array, sample_rate=self.sample_rate)
        return {"prediction": verdict.prediction, "probability": verdict.probability}


class Models:
    """Silero's voice activity detector, on the CPU as Pipecat ships it, and Smart Turn on MLX."""

    def __init__(self, clock: Clock, sample_rate: int) -> None:
        self._vad = SileroVADAnalyzer(sample_rate=sample_rate, params=VADParams(stop_secs=STOP_SECS))
        self._vad.set_sample_rate(sample_rate)
        self._turn = SmartTurn(clock, sample_rate)
        # As Pipecat syncs it at each start: the speech Silero takes to be sure of is kept before the segment judged.
        self._turn.update_vad_start_secs(self._vad.params.start_secs)

    @property
    def vad_stop_secs(self) -> float:
        return self._vad.params.stop_secs

    @property
    def smart_turn_warmup_secs(self) -> float:
        return self._turn.warmup_secs

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
async def loaded(clock: Clock, sample_rate: int, emit: Callable[[WideEvent], None], name: Literal["desk.loaded", "phone.loaded"]) -> AsyncGenerator[Models]:
    """Both models, loaded off the loop as their own unit of work, `name`, Smart Turn judging at a turn's latency from the
    first verdict, for as long as what hears with them does: the desk (`desk.loaded`) or the phone (`phone.loaded`)."""
    async with AsyncExitStack() as held:
        with loads.loading(clock, name, emit):
            # [LAW:nothing-unseen] the Smart Turn being loaded, before the load, so one that fails says which; and the
            # pause the detector ends speech at, as it was loaded with it, and how long Smart Turn's warm-up took.
            annotate(smart_turn=SMART_TURN, smart_turn_revision=SMART_TURN_REVISION)
            # On a thread of its own the process does not join at exit (hands.threads), as every model's load is; started
            # inside the unit, so the load's time and growth are all on it.
            loading = settles_off_loop(lambda: Models(clock, sample_rate), "Models")
            held.push_async_callback(_let_go, loading)
            models = await asyncio.shield(loading)
            annotate(vad_stop_secs=models.vad_stop_secs, smart_turn_warmup_secs=models.smart_turn_warmup_secs)
        yield models


async def _let_go(loading: asyncio.Future[Models]) -> None:
    """A switch away while they load waits the load out, so what it built is let go of too."""
    await (await loading).close()


# [LAW:one-source-of-truth] each move the desk makes is counted on the desk's event, by the move's name and by the trigger
# that made it, and each change of level by the level it went to.
_COUNTED: tuple[str, ...] = (
    *(name for literal in get_args(Move) for name in get_args(literal)),
    *(f"by_{trigger.replace(' ', '_')}" for trigger in get_args(Trigger)),
    *(f"to_{level.replace(' ', '_')}" for level in LEVELS),
    "held_open", "ended_on_silence", "heard", "wake_words",
)

# The wake word a buffer of the desk's microphone ends with, if it does (`hands.voice.wake`), heard afresh from it when
# its level has just started.
Hears = Callable[[bytes, bool], Awaitable[Woken | None]]


@dataclass(frozen=True)
class Overheard:
    """A buffer of the desk's microphone, with the life of the models it is heard with."""

    audio: bytes
    life: Life
    woken: Hears


def _models(life: Life | None) -> Life:
    """The life of the models a voice turn is made of: what it is heard and judged with is heard with them, so none opens
    or is judged before they are loaded."""
    # [LAW:no-silent-failure] a turn that moved with no models loaded is a desk that was stepped wrongly, and says so.
    if life is None:
        raise RuntimeError("a voice turn moved or was judged before the models that hear it were loaded")
    return life


async def drive(
    clock: Clock,
    tapped: Callable[[Callable[[KeyEvent], None]], AbstractAsyncContextManager[None]],
    overheard: Callable[[], AbstractAsyncContextManager[AsyncIterator[bytes]]],
    hearing: Callable[[], AbstractAsyncContextManager[tuple[Ears, Hears]]],
    levels: Driving,
    on_move: Callable[[Made], Awaitable[None]],
    on_level: Callable[[Leveled], Awaitable[None]],
    emit: Callable[[WideEvent], None],
) -> None:
    """Hand every move the desk makes to `on_move`, and every change of level to `on_level`, until cancelled, the desk
    made at the level `levels` says the run starts at, and `levels` told the desk after every step.

    One task steps the desk, in the order things were heard, so the speech, the key, the taps, the wake word, the
    tools' choices, and Smart Turn's verdicts never race each other: a verdict is stepped as it is given, before any
    audio heard while the model thought, so it ends the turn it judged and no other. The key is read from the start;
    the room is heard once `hearing` has loaded the models it is heard with, and one that cannot be loaded fails the
    drive. A level chosen before then, by the start or anyone, is listened at, with its cue, from then.
    """
    events: asyncio.Queue[Event | Overheard | Choice] = asyncio.Queue()
    # What has been told and is yet to be stepped, in the order it is stepped.
    heard: deque[Event] = deque()

    async def overhearing() -> None:
        # [LAW:no-ambient-temporal-coupling] each buffer carries the life of the models it is heard with, so none is heard
        # before they are loaded, and the key never waits on them.
        async with hearing() as (ears, woken), overheard() as room:
            # Ahead of the first buffer, so the level chosen while the models loaded is heard from it on.
            events.put_nowait(ModelsLoaded())
            life = Life(clock, ears, events.put_nowait, heard.appendleft, emit, "desk")
            async for audio in room:
                events.put_nowait(Overheard(audio, life, woken))

    async def choosing() -> None:
        while True:
            events.put_nowait(await levels.taken())

    # [LAW:dataflow-not-control-flow] the level the run starts at is the start's choice, held as any choice is while the
    # models load, and then the gate is told the desk listens, its cue plays, and its change is said, as for a tap; off, a
    # cold start's, is no change.
    desk = Desk.starting(levels.last_chosen)
    # The life of the models the last buffer was heard with: None until they are loaded, while Smart Turn has heard nothing.
    life: Life | None = None
    # Whether the key has come up since the last buffer was heard: speech the desk ignored while it was down starts for it
    # as the key comes up, so a tap of Shift as the user began does not lose what they say.
    key_came_up = False
    # The level the last buffer was heard at: off, where nothing has heard it, as the desk listens while its models load.
    overheard_at: Level = "off"
    async with tapped(events.put_nowait), asyncio.TaskGroup() as group:
        group.create_task(overhearing(), name="the desk's microphone, overheard")
        group.create_task(choosing(), name="the levels the tools chose")
        while True:
            # [LAW:nothing-unseen] one event for each stint at a level, counting every move, trigger, and change of level
            # in it, emitted as the level changes.
            with unit("desk", emit, wide.begun(clock), _COUNTED):
                level = desk.level
                annotate(level=level)
                while desk.level == level:
                    if not heard:
                        match await events.get():
                            # [LAW:types-are-the-program] off, the microphone is closed between turns, and nothing
                            # hears the room.
                            case Overheard() if not listens(desk.level):
                                overheard_at = desk.level
                            case Overheard(audio=audio, life=life, woken=woken):
                                count(heard=1)
                                # Each model hears from where its level started, with nothing of the room before it.
                                begun = desk.level != overheard_at
                                overheard_at = desk.level
                                if begun:
                                    life.afresh()
                                if key_came_up:
                                    life.quiet()
                                key_came_up = False
                                # Smart Turn hears speech only with no key down, so the room it hears otherwise is
                                # trimmed as silence is.
                                detected = await life.hear(audio, not isinstance(desk.phase, Keyed), turn_of(desk.phase))
                                # The wake word's model is run only where it can open a turn.
                                wake = await woken(audio, begun) if desk.level == "wake word" else None
                                heard.extend(detected.told(() if wake is None else (wake,)))
                            case Choice(level=chose, by=asker) as choice:
                                levels.answer(choice)
                                heard.append(Chosen(chose, asker))
                            case event:
                                heard.append(event)
                        continue
                    each = heard.popleft()
                    was = desk
                    desk, acts = step(desk, each)
                    # [LAW:no-ambient-temporal-coupling] the owner is told before any act is awaited, so a drive cancelled
                    # inside one leaves the owner at the level the desk is at, which a restart then carries.
                    levels.stepped(desk)
                    was_turn, turn = turn_of(was.phase), turn_of(desk.phase)
                    if was_turn != turn:
                        _models(life).moved(was_turn, turn)
                        if voiceturn.closed(was_turn, turn) is not None and isinstance(each, TurnEnded) and each.by == "silence":
                            count(ended_on_silence=1)
                    key_came_up = key_came_up or (isinstance(was.phase, Keyed) and not isinstance(desk.phase, Keyed))
                    for act in acts:
                        match act:
                            case Judging(turn=judged):
                                # Only speech the detector heard is judged, and it is heard with the models.
                                if not await _models(life).judge(judged):
                                    count(held_open=1)
                            case Made(move=move, by=by):
                                count(**{move: 1, f"by_{by.replace(' ', '_')}": 1})
                                await on_move(act)
                            case Leveled(was=was_level, now=now, by=chooser):
                                count(**{f"to_{now.replace(' ', '_')}": 1})
                                # [LAW:nothing-unseen] each change of level is its own event: from, to, and by.
                                with unit("desk.leveled", emit, wide.begun(clock)):
                                    annotate(was=was_level, now=now, by=chooser)
                                await on_level(act)
