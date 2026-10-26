"""The cues: a turn's edges show and play a terminal line and a short tone for each, and nothing for Shift; the cues for
silence wait for nobody to be speaking and for their spacing, and play each kind once however often it was owed."""

import asyncio
from typing import Literal

import numpy as np

from conftest import ManualClock, patiently
from hands.sessions.audit import Cued, Entry
from hands.sessions.audit import Played
from hands.voice.cues import CUE_LEVEL, CUE_SECONDS, DROPPED, OPENED, RECEIVED, SENT, WORKING, Cue, QuietCue, QuietCues, cues, sound

RATE = 24000


def test_a_turn_shows_its_edges_and_nothing_of_shift() -> None:
    assert [cue.line for move in ("arm", "start", "stop") for cue in cues(move)] == ["turn: started", "turn: ended"]
    assert [cue.line for move in ("arm", "start", "drop") for cue in cues(move)] == ["turn: started", "turn: dropped"]
    assert [cue.line for move in ("arm", "start", "timeout") for cue in cues(move)] == ["turn: started", "turn: ended, open 600s"]
    assert cues("arm") + cues("disarm") == ()


def samples(cue: Cue, channels: int = 1) -> np.ndarray:
    return np.frombuffer(sound(cue, RATE, channels), np.int16).astype(np.float64) / 32767


def pitch(tone: np.ndarray) -> float:
    """Hz, from the zero crossings of the tone's loud middle."""
    middle = tone[len(tone) // 4 : 3 * len(tone) // 4]
    crossings = np.count_nonzero(np.diff(np.signbit(middle)))
    return crossings / 2 / (len(middle) / RATE)


def test_each_cue_is_short_soft_and_starts_and_ends_in_silence() -> None:
    for cue in (OPENED, SENT, DROPPED, RECEIVED.cue, WORKING.cue):
        audio = samples(cue)
        assert len(audio) == round(RATE * CUE_SECONDS) * 2 * len(cue.glides)
        assert 0.9 * CUE_LEVEL < np.abs(audio).max() <= CUE_LEVEL
        assert np.abs(audio[:5]).max() < 0.01 and np.abs(audio[-5:]).max() < 0.01  # no click at either end


def test_opening_rises_sending_falls_and_dropping_is_low_and_twice() -> None:
    n = round(RATE * CUE_SECONDS)
    opened, sent, dropped = samples(OPENED)[:n], samples(SENT)[:n], samples(DROPPED)
    assert pitch(opened[: n // 2]) < pitch(opened[n // 2 :])
    assert pitch(sent[: n // 2]) > pitch(sent[n // 2 :])
    assert pitch(dropped[:n]) < 400 and np.abs(dropped[2 * n : 3 * n]).max() > 0.9 * CUE_LEVEL


def test_every_channel_carries_the_tone() -> None:
    stereo = samples(OPENED, channels=2)
    assert np.array_equal(stereo[0::2], stereo[1::2]) and np.array_equal(stereo[0::2], samples(OPENED))


def test_receiving_is_high_and_rising_in_steps_and_working_is_one_steady_tone() -> None:
    n = round(RATE * CUE_SECONDS)
    received, working = samples(RECEIVED.cue), samples(WORKING.cue)
    # Each of its tones steady, which none of the key's glides is, and above all of them.
    first, second = received[:n], received[2 * n : 3 * n]
    assert abs(pitch(first[: n // 2]) - pitch(first[n // 2 :])) < 100
    assert 1000 < pitch(first) < pitch(second)
    assert len(working) == 2 * n and abs(pitch(working[: n // 2]) - pitch(working[n // 2 : n])) < 100
    # One steady tone, where every other cue is a glide or a pair.
    assert [len(cue.glides) == 1 and cue.glides[0][0] == cue.glides[0][1] for cue in (OPENED, SENT, DROPPED, RECEIVED.cue, WORKING.cue)] == [False, False, False, False, True]


class Quiet(asyncio.Event):
    """Whether nobody is speaking, which also counts each time the player comes to wait for it."""

    def __init__(self) -> None:
        super().__init__()
        self.awaited = 0

    async def wait(self) -> Literal[True]:
        self.awaited += 1
        return await super().wait()


class Speaker:
    def __init__(self) -> None:
        self.quiet = Quiet()
        self.played: list[Cue] = []

    def cue(self, cue: Cue) -> Played:
        self.played.append(cue)
        return "desk"

    async def heard(self, count: int) -> list[Cue]:
        """What it has played, once it has played `count` cues."""
        await patiently(lambda: len(self.played) >= count, lambda: len(self.played))
        return self.played

    async def waited_for(self) -> list[Cue]:
        """What it has played, once the player has come to wait for it to be quiet."""
        await patiently(lambda: self.quiet.awaited > 0, lambda: len(self.played))
        return self.played


async def test_a_cue_for_silence_waits_out_speech_and_each_kind_plays_once() -> None:
    clock = ManualClock()
    owed, speaker, recorded = QuietCues(clock.now, clock.pause), Speaker(), list[Entry]()
    cueing = asyncio.create_task(owed.keep_playing(speaker, recorded.append))
    try:
        owed.owe(RECEIVED)
        clock.advance(0.5)
        owed.owe(WORKING)
        owed.owe(WORKING)
        assert await speaker.waited_for() == []  # someone is speaking
        clock.advance(1.5)
        speaker.quiet.set()
        assert await speaker.heard(2) == [RECEIVED.cue, WORKING.cue]
        assert recorded == [Cued("turn: received", 1, 2.0, "desk"), Cued("working", 2, 1.5, "desk")]
        owed.owe(RECEIVED)
        assert await speaker.heard(3) == [RECEIVED.cue, WORKING.cue, RECEIVED.cue]  # quiet and unspaced, so at once
    finally:
        cueing.cancel()


async def test_a_burst_of_acts_in_silence_is_one_tone_per_spacing() -> None:
    clock = ManualClock()
    owed, speaker, recorded = QuietCues(clock.now, clock.pause), Speaker(), list[Entry]()
    speaker.quiet.set()
    spaced = QuietCue(WORKING.cue, 0.05)
    cueing = asyncio.create_task(owed.keep_playing(speaker, recorded.append))
    try:
        owed.owe(spaced)
        await speaker.heard(1)
        owed.owe(spaced)
        owed.owe(spaced)
        owed.owe(RECEIVED)
        # The receipt is never held behind the working tone's spacing; the two acts after the first wait it out as one,
        # and the clock has not moved, so its spacing is not up.
        assert await speaker.heard(2) == [WORKING.cue, RECEIVED.cue]
        clock.advance(0.05)
        assert await speaker.heard(3) == [WORKING.cue, RECEIVED.cue, WORKING.cue]
        assert [(entry.line, entry.folded) for entry in recorded if isinstance(entry, Cued)] == [("working", 1), ("turn: received", 1), ("working", 2)]
        clock.advance(10.0)
        owed.owe(RECEIVED)
        # A cue still due would have played ahead of the receipt.
        assert await speaker.heard(4) == [WORKING.cue, RECEIVED.cue, WORKING.cue, RECEIVED.cue]
    finally:
        cueing.cancel()
