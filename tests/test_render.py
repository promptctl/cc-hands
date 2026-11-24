"""hands' TTS rendered to audio: each saying made by the service in its voice, in order, or the reason it was not made."""

import asyncio
from collections.abc import AsyncGenerator, Callable, Sequence

import pytest
from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame, TTSUpdateSettingsFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.pocket_tts.tts import PocketTTSService, PocketTTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.types import require_given

from hands.clock import Clock
from hands.core.playback import hands_says
from hands.sessions.wide import begun
from hands.virtualclock import VirtualClock
from hands.voice.clips import RATE, Synthesis
from hands.voice.player import LineFrame, Unsaid
from hands.voice.render import NotRendered, Saying, render
from hands.voice.spoken import SpokenForm
from hands.voice.synthesis import Made, SpeakingIn
from hands.voice.voicechoice import parse_voice
from hands.voice.voices import SaidInTTS

TTS_RATE = 24_000
ALBA, MARIUS = parse_voice("alba"), parse_voice("marius")

# What the TTS says behind a synthesis it opened.
Makes = Callable[[Synthesis], Sequence[Frame]]


def sound(seconds: float, rate: int = TTS_RATE, channels: int = 1) -> Frame:
    return TTSAudioRawFrame(b"\x01\x00" * round(seconds * rate) * channels, rate, channels)


def raises(_: Synthesis) -> Sequence[Frame]:
    raise RuntimeError("the filter raised")


def made(seconds: float) -> Makes:
    return lambda synthesis: [sound(seconds), Made(synthesis, 0.01, seconds, None)]


class Spoke(FrameProcessor):
    """A TTS service as the speaker meets it: a voice changed by a settings frame, and a line made into a synthesis that
    `makes` says the sound of."""

    def __init__(self, clock: Clock, makes: Makes, said: Callable[[str], str] = lambda text: text) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
        self._clock, self._makes, self._said = clock, makes, said
        self._voice = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        match frame:
            case TTSUpdateSettingsFrame(delta=PocketTTSSettings(voice=voice)):
                self._voice = require_given(voice, "voice")
            case LineFrame(text=text):
                synthesis = Synthesis(self._voice, self._said(text), begun(self._clock))
                for made_frame in (SpeakingIn(synthesis), *self._makes(synthesis)):
                    # Pipecat's own: an error goes upstream to the pipeline's worker, never down to what follows.
                    match made_frame:
                        case ErrorFrame():
                            await self.push_error_frame(made_frame)
                        case _:
                            await self.push_frame(made_frame)
            case _:
                await self.push_frame(frame, direction)


async def test_each_saying_is_made_in_its_voice_in_order_at_the_recording_rate() -> None:
    fetched: list[object] = []
    sayings = [Saying(ALBA, "Hands here."), Saying(MARIUS, "Which review?"), Saying(ALBA, "Hands here.")]
    clock = VirtualClock()
    out = await render(clock, Spoke(clock, made(0.5)), sayings, fetched.append)
    assert [(one.synthesis.voice, one.synthesis.text) for one in out] == [("alba", "Hands here."), ("marius", "Which review?"), ("alba", "Hands here.")]
    assert [len(one.audio) for one in out] == [RATE // 2] * 3
    # A voice is fetched once, before any is said.
    assert fetched == [ALBA, MARIUS]


def failed(synthesis: Synthesis) -> Sequence[Frame]:
    return [sound(0.1), Made(synthesis, 0.01, 0.1, "Unknown error")]


def silent(synthesis: Synthesis) -> Sequence[Frame]:
    return [Made(synthesis, 0.01, 0.0, None)]


def at_two_rates(synthesis: Synthesis) -> Sequence[Frame]:
    return [sound(0.1), sound(0.1, 16_000), Made(synthesis, 0.01, 0.2, None)]


def in_stereo(synthesis: Synthesis) -> Sequence[Frame]:
    return [sound(0.1, channels=2), Made(synthesis, 0.01, 0.1, None)]


def of_another(synthesis: Synthesis) -> Sequence[Frame]:
    return [sound(0.1), Made(Synthesis(synthesis.voice, synthesis.text, synthesis.begun), 0.01, 0.1, None)]


def unsaid(_: Synthesis) -> Sequence[Frame]:
    return [Unsaid(hands_says("Hands here."))]


def errors(_: Synthesis) -> Sequence[Frame]:
    return [ErrorFrame("the model raised")]


@pytest.mark.parametrize(
    ("makes", "reason"),
    [
        (failed, "Unknown error"),
        (silent, "no sound"),
        (at_two_rates, "Hz"),
        (in_stereo, "Hz"),
        (of_another, "did not open"),
        (unsaid, "as nothing"),
        (errors, "the model raised"),
        (raises, "the filter raised"),
    ],
    ids=["the TTS says it failed", "it made no sound", "its sound is at two rates", "its sound is in stereo", "it closes another synthesis", "the line is said as nothing", "it pushes an error", "a frame fails to process"],
)
async def test_a_saying_that_is_not_made_is_named_with_the_reason(makes: Makes, reason: str) -> None:
    clock = VirtualClock()
    with pytest.raises(NotRendered, match=f"'Hands here.' in alba: .*{reason}"):
        await render(clock, Spoke(clock, makes), [Saying(ALBA, "Hands here.")], lambda _: None)


async def test_a_tts_that_says_nothing_is_named_when_the_wait_ends() -> None:
    clock = VirtualClock()
    rendering = asyncio.create_task(render(clock, Spoke(clock, lambda _: []), [Saying(ALBA, "Hands here.")], lambda _: None, timeout=5.0))
    await clock.scheduled(5.0)
    clock.advance(5.0)
    with pytest.raises(NotRendered, match="'Hands here.' in alba: nothing came of it in 5 s"):
        await rendering


async def test_a_saying_the_tts_changed_is_not_the_line_that_was_asked_for() -> None:
    clock = VirtualClock()
    with pytest.raises(NotRendered, match="it said 'x.' in alba"):
        await render(clock, Spoke(clock, made(0.1), said=lambda _: "x."), [Saying(ALBA, "`tests/x.py`")], lambda _: None)


async def test_a_voice_that_cannot_be_fetched_is_refused_before_any_is_said() -> None:
    def refused(voice: object) -> None:
        raise OSError(f"no {voice}")

    clock = VirtualClock()
    with pytest.raises(OSError, match="no alba"):
        await render(clock, Spoke(clock, made(0.1)), [Saying(ALBA, "Hands here.")], refused)


class Modelless(SaidInTTS):
    """hands' own TTS service, `run_tts` and the filter as built, with Pocket's model under them taken out."""

    def __init__(self, clock: Clock) -> None:
        TTSService.__init__(self, settings=PocketTTSSettings(model=None, voice="alba", language=Language.EN), text_filters=[SpokenForm()])  # pyright: ignore[reportUnknownMemberType]
        self._clock = clock


def model_makes(monkeypatch: pytest.MonkeyPatch, frames: Sequence[Frame]) -> None:
    async def run_tts(_service: PocketTTSService, _text: str, _context_id: str) -> AsyncGenerator[Frame | None, None]:
        for frame in frames:
            yield frame

    monkeypatch.setattr(PocketTTSService, "run_tts", run_tts)


async def test_the_service_hands_speaks_with_is_rendered_in_the_order_it_makes_a_synthesis(monkeypatch: pytest.MonkeyPatch) -> None:
    model_makes(monkeypatch, [sound(0.5)])
    clock = VirtualClock()
    sayings = [Saying(ALBA, "Hands here."), Saying(MARIUS, "Which review?")]
    out = await render(clock, Modelless(clock), sayings, lambda _: None)
    assert [(one.synthesis.voice, one.synthesis.text, len(one.audio)) for one in out] == [("alba", "Hands here.", RATE // 2), ("marius", "Which review?", RATE // 2)]


async def test_the_service_hands_speaks_with_says_what_its_model_failed_with(monkeypatch: pytest.MonkeyPatch) -> None:
    model_makes(monkeypatch, [ErrorFrame("the model raised")])
    clock = VirtualClock()
    with pytest.raises(NotRendered, match="'Hands here.' in alba: .*the model raised"):
        await render(clock, Modelless(clock), [Saying(ALBA, "Hands here.")], lambda _: None)


async def test_the_service_hands_speaks_with_filters_a_line_before_it_is_rendered(monkeypatch: pytest.MonkeyPatch) -> None:
    model_makes(monkeypatch, [sound(0.5)])
    clock = VirtualClock()
    with pytest.raises(NotRendered, match="it said .* in alba"):
        await render(clock, Modelless(clock), [Saying(ALBA, "Open `tests/x.py`.")], lambda _: None)
