"""hands' own voice through the room and a person over it, recorded at the desk for promptctl-voice-i15t.99s.zoj, and
what hands played as each was recorded.

hands_cancelled_and_another.flac is the desk heard through hands' echo canceller (scripts/record-double-talk.py): its
channels as heard after the canceller, as captured before it, and the canceller's reference; three of hands' sentences
in 'charles', and 'alba' saying two lines over them that repeat hands' words where hands says them. Its .json keeps
which sentence each 10 ms was matched with, so what hands played over it is known, not read off.
hands_no_echo_and_another.flac is the same take with hands' voice in no speaker, as in headphones: the canceller was told
it, and the microphone heard only the second voice and the room. hands_over_an_opening.flac and
hands_no_echo_over_an_opening.flac are another script's two takes the same two ways: a reply hands says alone, a 4 s
pause, and 'alba' cutting into the next reply as it opens.

The takes of promptctl-voice-i15t.65f.gp1 are the same two ways again where a person and hands meet otherwise:
hands_and_the_same_words and hands_no_echo_and_the_same_words ('alba' saying words of hands' as hands says them),
hands_and_short_words and hands_no_echo_and_short_words ('alba' saying "No.", "Stop." and "Yes." over hands),
hands_in_two_voices and hands_no_echo_in_two_voices (hands' replies in 'charles' and 'marius'). hands_long_and_slow.flac is
two long sentences of hands' with no one speaking, and hands_then_a_person_alone.flac a reply and then 'alba' alone, at
the desk with the speaker up. Each .json also names the voice of each of hands' sentences."""

import json
from pathlib import Path

import numpy as np

from hands.sessions.wide import begun
from hands.virtualclock import VirtualClock
from hands.voice.clips import RATE, SLOT, Played, Reference, Playhead, Sound, Synthesis
from hands.voice.echo import EchoCanceller
from speech import audio

# hands' eight sentences, as its TTS was given them: read back off the recordings one sentence at a time, the names as
# hands' vocabulary spells them.
READ = (
    "Hands here. Starting houdini-mcp and opening macklebox now.",
    "Phosphene finished its render, and the build is green.",
    "I can hear you. Say the next step when you are ready.",
    "The diarizer reports one speaker in this hold.",
    "Checking the speech models before we continue with the release.",
    "Turn ended, sending the words to the brain now.",
    "Would you like me to read the summary aloud?",
    "That is everything for this run. Thank you.",
)


def played(onsets: tuple[float, ...], stopped: float, start: float, seconds: float) -> Played:
    """What the speaker played over `seconds` of a recording from `start`, one reference for each 10 ms: each of hands'
    sentences, a synthesis of its own, from its onset in `onsets` to the next one's, and nothing once the play stopped at
    `stopped`."""
    said = tuple(Synthesis("charles", text, begun(VirtualClock())) for text in READ)

    def at(moment: float) -> Reference | None:
        playing = tuple(Playhead(synthesis, moment - onset) for onset, end, synthesis in zip(onsets, [*onsets[1:], stopped], said) if onset <= moment < end)
        return Reference(bytes(2 * SLOT), RATE, 1, playing, True) if playing else None

    return tuple(at(start + slot * SLOT / RATE) for slot in range(round(seconds * RATE / SLOT)))


CANCELLED = Path(__file__).parent / "fixtures" / "hands_cancelled_and_another"
NO_ECHO = Path(__file__).parent / "fixtures" / "hands_no_echo_and_another"
OPENING = Path(__file__).parent / "fixtures" / "hands_over_an_opening"
NO_ECHO_OPENING = Path(__file__).parent / "fixtures" / "hands_no_echo_over_an_opening"
# What a person and hands do at the desk beyond those: a person saying words hands is saying, a short "no", "stop" or "yes"
# over hands, hands in two voices, hands' long and slow sentences with no one speaking, and a person alone after hands.
# Each but the last two is recorded at the desk and again in headphones.
SAME_WORDS = Path(__file__).parent / "fixtures" / "hands_and_the_same_words"
NO_ECHO_SAME_WORDS = Path(__file__).parent / "fixtures" / "hands_no_echo_and_the_same_words"
SHORT_WORDS = Path(__file__).parent / "fixtures" / "hands_and_short_words"
NO_ECHO_SHORT_WORDS = Path(__file__).parent / "fixtures" / "hands_no_echo_and_short_words"
TWO_VOICES = Path(__file__).parent / "fixtures" / "hands_in_two_voices"
NO_ECHO_TWO_VOICES = Path(__file__).parent / "fixtures" / "hands_no_echo_in_two_voices"
LONG_AND_SLOW = Path(__file__).parent / "fixtures" / "hands_long_and_slow"
PERSON_ALONE = Path(__file__).parent / "fixtures" / "hands_then_a_person_alone"
# The second voice's lines on hands_cancelled_and_another.flac, "the build is" and "Say the next step" said where hands
# says them.
OVER_CANCELLED = ("No, the build is broken.", "Say the next step again.")


def cut_in(fixture: Path) -> float:
    """When the second voice first speaks on a take of scripts/record-double-talk.py, in seconds."""
    return json.loads(fixture.with_suffix(".json").read_text())["other"][0]["start"]


CANCELLED_ALONE = cut_in(CANCELLED)


def take(fixture: Path) -> tuple[np.ndarray, np.ndarray, tuple[tuple[Playhead, ...], ...]]:
    """A take of scripts/record-double-talk.py as the microphone captured it and as the speaker played it, and the
    playheads of hands' sentence over each 10 ms of it."""
    described = json.loads(fixture.with_suffix(".json").read_text())
    said = tuple(Synthesis(voice, text, begun(VirtualClock())) for text, voice in zip(described["hands"], described["voices"], strict=True))
    samples, _ = audio(fixture.with_suffix(".flac"))
    playheads = tuple(() if slot is None else (Playhead(said[int(slot[0])], float(slot[1])),) for slot in described["played"])
    return samples[: len(playheads) * SLOT, 1], samples[: len(playheads) * SLOT, 2], playheads


def replayed(captured: np.ndarray, reference: np.ndarray, playheads: tuple[tuple[Playhead, ...], ...]) -> Sound:
    """What the microphone `captured` heard again through a fresh echo canceller, told the speaker's `reference` with
    its `playheads` 20 ms at a time ahead of each 20 ms heard, as the desk's two devices give them."""
    canceller, heard, played = EchoCanceller(), bytearray(), list[Reference | None]()
    buffers = min(len(captured), len(reference)) // (2 * SLOT)
    for buffer in range(buffers):
        at = slice(2 * SLOT * buffer, 2 * SLOT * (buffer + 1))
        canceller.played(reference[at].tobytes(), RATE, 1, tuple(dict.fromkeys(playheads[2 * buffer] + playheads[2 * buffer + 1])))
        cancelled = canceller.heard(captured[at].tobytes(), RATE, 1)
        heard += cancelled.heard
        played += cancelled.played
    return Sound(bytes(heard), captured[: 2 * SLOT * buffers].tobytes(), tuple(played))
