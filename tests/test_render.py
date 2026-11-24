"""hands' TTS rendered to audio: each saying made by the service in its voice, in order, or the reason it was not made."""

import asyncio
from collections.abc import Callable, Sequence

import pytest
from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame, TTSUpdateSettingsFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.pocket_tts.tts import PocketTTSSettings
from pipecat.utils.types import require_given

from hands.clock import SYSTEM
from hands.core.playback import hands_says
from hands.sessions.wide import begun
from hands.virtualclock import VirtualClock
from hands.voice.clips import RATE, Synthesis
from hands.voice.player import LineFrame, Unsaid
from hands.voice.render import NotRendered, Saying, render
from hands.voice.synthesis import Made, SpeakingIn
from hands.voice.voicechoice import parse_voice

TTS_RATE = 24_000
ALBA, MARIUS = parse_voice("alba"), parse_voice("marius")

# What the TTS says behind a synthesis it opened.
Makes = Callable[[Synthesis], Sequence[Frame]]


def sound(seconds: float, rate: int = TTS_RATE) -> Frame:
    return TTSAudioRawFrame(b"\x01\x00" * round(seconds * rate), rate, 1)


def raises(_: Synthesis) -> Sequence[Frame]:
    raise RuntimeError("the filter raised")


def made(seconds: float) -> Makes:
    return lambda synthesis: [sound(seconds), Made(synthesis, 0.01, seconds, None)]


class Spoke(FrameProcessor):
    """A TTS service as the speaker meets it: a voice changed by a settings frame, and a line made into a synthesis that
    `makes` says the sound of."""

    def __init__(self, makes: Makes, said: Callable[[str], str] = lambda text: text) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
        self._makes, self._said = makes, said
        self._voice = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        match frame:
            case TTSUpdateSettingsFrame(delta=PocketTTSSettings(voice=voice)):
                self._voice = require_given(voice, "voice")
            case LineFrame(text=text):
                synthesis = Synthesis(self._voice, self._said(text), begun(SYSTEM))
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
    out = await render(VirtualClock(), Spoke(made(0.5)), sayings, fetched.append)
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


def unsaid(_: Synthesis) -> Sequence[Frame]:
    return [Unsaid(hands_says("Hands here."))]


def errors(_: Synthesis) -> Sequence[Frame]:
    return [ErrorFrame("the model raised")]


@pytest.mark.parametrize(
    ("makes", "reason"),
    [(failed, "Unknown error"), (silent, "no sound"), (at_two_rates, "Hz"), (unsaid, "as nothing"), (errors, "the model raised"), (raises, "the filter raised")],
    ids=["the TTS says it failed", "it made no sound", "its sound is at two rates", "the line is said as nothing", "it pushes an error", "a frame fails to process"],
)
async def test_a_saying_that_is_not_made_is_named_with_the_reason(makes: Makes, reason: str) -> None:
    with pytest.raises(NotRendered, match=f"'Hands here.' in alba: .*{reason}"):
        await render(VirtualClock(), Spoke(makes), [Saying(ALBA, "Hands here.")], lambda _: None)


async def test_a_tts_that_says_nothing_is_named_when_the_wait_ends() -> None:
    clock = VirtualClock()
    rendering = asyncio.create_task(render(clock, Spoke(lambda _: []), [Saying(ALBA, "Hands here.")], lambda _: None, timeout=5.0))
    await clock.scheduled(5.0)
    clock.advance(5.0)
    with pytest.raises(NotRendered, match="'Hands here.' in alba: nothing came of it in 5 s"):
        await rendering


async def test_a_saying_the_tts_changed_is_not_the_line_that_was_asked_for() -> None:
    with pytest.raises(NotRendered, match="it said 'x.' in alba"):
        await render(VirtualClock(), Spoke(made(0.1), said=lambda _: "x."), [Saying(ALBA, "`tests/x.py`")], lambda _: None)


async def test_a_voice_that_cannot_be_fetched_is_refused_before_any_is_said() -> None:
    def refused(voice: object) -> None:
        raise OSError(f"no {voice}")

    with pytest.raises(OSError, match="no alba"):
        await render(VirtualClock(), Spoke(made(0.1)), [Saying(ALBA, "Hands here.")], refused)
