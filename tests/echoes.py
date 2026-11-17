"""hands' own voice through the room, uncancelled, and a person over it, recorded at the desk for
promptctl-voice-i15t.99s.zoj, and what hands played as each was recorded.

hands_echo_uncancelled.flac is the MacBook's microphone, no canceller, as hands said eight sentences ('charles', its
default voice) out of its speakers; hands_and_another.flac is another play of the same sentences, with a second voice
('alba') saying five lines over hands until about 14.3 s, and hands alone after. hands_tts.flac is hands' own TTS output
in that voice, the first three sentences, which the print is made from as hands makes its own. The list hands played
from was not kept, so when each sentence began is read off each recording.

hands_cancelled_and_another.flac is the desk heard through hands' echo canceller (scripts/record-double-talk.py): its
channels as heard after the canceller, as captured before it, and the canceller's reference; three of hands' sentences
in 'charles', and 'alba' saying two lines over them that repeat hands' words where hands says them. Its .json keeps
which sentence each 10 ms was matched with, so what hands played over it is known, not read off."""

import json
from pathlib import Path

from hands.sessions.wide import begun
from hands.virtualclock import VirtualClock
from hands.voice.clips import RATE, SLOT, Played, Reference, Playhead, Sound, Synthesis
from speech import audio

# When each of hands' sentences began on hands_echo_uncancelled.flac, in seconds from PLAYED.
PLAYED = 0.5
SENTENCES = (0.0, 4.92, 8.56, 12.04, 15.68, 19.56, 23.36, 26.52)
# And when its play stopped.
STOPPED = PLAYED + 29.7

# When each began on hands_and_another.flac, in seconds from its start: where Whisper first hears it, unprimed where it
# heard it alone ("I can hear you." 11.7 s, the fourth to sixth sentences' segments), primed where the person hides its
# start ("starting" 5.0 s, past "Hands here.", "phosphene" 8.4 s); the seventh and eighth, and the stop, past the end of
# the recording at 24 s, by the uncancelled play's gaps. Its play is not the uncancelled one's: 1.4 to 3.5 s later.
ANOTHER = (4.0, 8.3, 11.6, 15.0, 18.2, 21.4, 25.2, 28.4)
ANOTHER_STOPPED = 31.6
OVER_HANDS, HANDS_ALONE = (0.0, 14.6), (15.0, 24.0)

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


def played(onsets: tuple[float, ...], stopped: float, start: float, clip: bytes) -> Played:
    """What the speaker played over `clip`, a recording from `start`, one reference for each 10 ms: each of hands'
    sentences, a synthesis of its own, from its onset in `onsets` to the next one's, and nothing once the play stopped at
    `stopped`. What was played was not kept, so each reference's sound is the clip's own over its 10 ms: it is read for
    whether hands' voice reached the microphone, and on these recordings it did."""
    said = tuple(Synthesis("charles", text, begun(VirtualClock())) for text in READ)

    def at(slot: int) -> Reference | None:
        moment = start + slot * SLOT / RATE
        playing = tuple(Playhead(synthesis, moment - onset) for onset, end, synthesis in zip(onsets, [*onsets[1:], stopped], said) if onset <= moment < end)
        return Reference(clip[2 * SLOT * slot : 2 * SLOT * (slot + 1)], RATE, 1, playing) if playing else None

    return tuple(at(slot) for slot in range(len(clip) // (2 * SLOT)))


FIXTURES = Path(__file__).parent / "fixtures"
CANCELLED = FIXTURES / "hands_cancelled_and_another"
# hands_no_echo_and_another.flac: the same, recorded with hands' voice in headphones, none of it out of the speaker.
NO_ECHO = FIXTURES / "hands_no_echo_and_another"
# The second voice's lines on both, "the build is" and "Say the next step" said where hands says them.
OVER_CANCELLED = ("No, the build is broken.", "Say the next step again.")
# When the second voice first speaks on hands_cancelled_and_another.flac, in seconds.
CANCELLED_ALONE = json.loads(CANCELLED.with_suffix(".json").read_text())["other"][0]["start"]


def take(fixture: Path) -> Sound:
    """A take of scripts/record-double-talk.py as the desk heard it, and what the speaker played over each 10 ms of it."""
    described = json.loads(fixture.with_suffix(".json").read_text())
    said = tuple(Synthesis("charles", text, begun(VirtualClock())) for text in described["hands"])
    samples, _ = audio(fixture.with_suffix(".flac"))
    reference = samples[:, 2].tobytes()
    played = tuple(
        None if slot is None else Reference(reference[2 * SLOT * at : 2 * SLOT * (at + 1)], RATE, 1, (Playhead(said[int(slot[0])], float(slot[1])),))
        for at, slot in enumerate(described["played"])
    )
    return Sound(samples[: len(played) * SLOT, 0].tobytes(), samples[: len(played) * SLOT, 1].tobytes(), played)
