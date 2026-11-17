"""hands' own words never reach the brain, on what hands heard at the desk (promptctl-voice-i15t.99s.zoj, the fixtures
test_streaming.py describes): the real streaming pair hears each clip with hands playing over it, hands' own voiceprint
identifies its speakers at its end, and Whisper, at its pinned weights, given what that came to, says what a person said
in it."""

import asyncio

import pytest

from hands.clock import SYSTEM
from hands.voice import streaming
from hands.voice.clips import Clip, Identified
from hands.voice.whisper import Whisper
from test_streaming import HANDS_ALONE, OVER_HANDS, PLAYED, SENTENCES, Overheard, Samples, fixture_clips, pcm
from test_streaming import hands_print, overheard, pair, paired  # pyright: ignore[reportUnusedImport]  (fixtures, which pytest finds by name)

pytestmark = pytest.mark.outside


@pytest.fixture(autouse=True)
def no_whisper_weights() -> None:
    """These tests are about the weights: the suite's stand-in for the model is not put in its place here."""


@pytest.fixture(scope="module")
def whisper() -> Whisper:
    async def unprimed() -> str | None:
        return None

    return Whisper(clock=SYSTEM, prompt=unprimed, record=lambda _: None)


async def said(overheard: Overheard, whisper: Whisper, number: int, clip: Samples) -> str | None:
    """What a turn of `clip` sends: Whisper's words, given its last identification as the clip owner gives it."""
    last: asyncio.Future[Identified] = asyncio.get_running_loop().create_future()
    last.set_result(await overheard.held(number, clip))
    return await whisper.transcribed(Clip(number, "wake word"), pcm(clip), last)


async def test_a_clip_of_hands_own_echo_sends_nothing(overheard: Overheard, whisper: Whisper) -> None:
    [echo] = fixture_clips("hands_echo_uncancelled.flac")
    starts = [round((PLAYED + start) * streaming.RATE) for start in SENTENCES]
    # Each sentence a clip of its own, as the wake word or engaged conversation would open one on hands' own voice.
    sent = [await said(overheard, whisper, number, echo[start:end]) for number, (start, end) in enumerate(zip(starts, [*starts[1:], len(echo)]), 1)]
    assert sent == [None] * len(SENTENCES), overheard.identified()


async def test_a_person_over_hands_sends_their_words_and_hands_alone_after_them_sends_nothing(overheard: Overheard, whisper: Whisper) -> None:
    [mic] = fixture_clips("hands_and_another.flac")
    over, alone = (mic[round(start * streaming.RATE) : round(end * streaming.RATE)] for start, end in (OVER_HANDS, HANDS_ALONE))
    # The second voice's lines each say who is speaking ("...the second speaker...").
    theirs = await said(overheard, whisper, 1, over)
    assert theirs is not None and "second speaker" in theirs.lower(), (theirs, overheard.identified())
    assert await said(overheard, whisper, 2, alone) is None, overheard.identified()
