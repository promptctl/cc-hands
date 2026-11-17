"""hands' own voice through the room, uncancelled, and a person over it, recorded at the desk for
promptctl-voice-i15t.99s.zoj, and what hands played as each was recorded.

hands_echo_uncancelled.flac is the MacBook's microphone, no canceller, as hands said eight sentences ('charles', its
default voice) out of its speakers; hands_and_another.flac is another play of the same sentences, with a second voice
('alba') saying five lines over hands until about 14.3 s, and hands alone after. hands_tts.flac is hands' own TTS output
in that voice, the first three sentences, which the print is made from as hands makes its own. The list hands played
from was not kept, so when each sentence began is read off each recording."""

from hands.sessions.wide import begun
from hands.virtualclock import VirtualClock
from hands.voice.clips import RATE, SLOT, Played, Reference, Playhead, Synthesis

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


def played(onsets: tuple[float, ...], stopped: float, start: float, seconds: float) -> Played:
    """What the speaker played over `seconds` of a recording from `start`, one reference for each 10 ms: each of hands'
    sentences, a synthesis of its own, from its onset in `onsets` to the next one's, and nothing once the play stopped at
    `stopped`."""
    said = tuple(Synthesis("charles", text, begun(VirtualClock())) for text in READ)

    def at(moment: float) -> Reference | None:
        playing = tuple(Playhead(synthesis, moment - onset) for onset, end, synthesis in zip(onsets, [*onsets[1:], stopped], said) if onset <= moment < end)
        return Reference(bytes(2 * SLOT), RATE, 1, playing) if playing else None

    return tuple(at(start + slot * SLOT / RATE) for slot in range(round(seconds * RATE / SLOT)))
