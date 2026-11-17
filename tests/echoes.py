"""hands' own voice through the room, uncancelled, and a person over it, recorded at the desk for
promptctl-voice-i15t.99s.zoj, and what hands played as each was recorded.

hands_echo_uncancelled.flac is the MacBook's microphone, no canceller, as hands said eight sentences ('charles', its
default voice) out of its speakers, play starting at PLAYED; hands_and_another.flac is the same with a second voice
('alba') saying four lines over hands from 1 s in, until about 14.3 s, and hands alone after. hands_tts.flac is hands' own
TTS output in that voice, the first three sentences, which the print is made from as hands makes its own."""

from hands.sessions.wide import begun
from hands.virtualclock import VirtualClock
from hands.voice.clips import RATE, SLOT, Played, Reference, Synthesis

PLAYED = 0.5
# When each of hands' sentences began, in seconds from PLAYED, as heard on hands_echo_uncancelled.flac. On
# hands_and_another.flac hands' words land 1.4 to 2.2 s later than these from the third sentence on (Whisper's word
# timestamps), so a play there is only as near as one sentence's onset to the next.
SENTENCES = (0.0, 4.92, 8.56, 12.04, 15.68, 19.56, 23.36, 26.52)
OVER_HANDS, HANDS_ALONE = (0.0, 14.6), (15.0, 24.0)

# hands' eight sentences, as its TTS was given them: read back off the recording one sentence at a time, the names as
# hands' vocabulary spells them, since the list hands played from was not kept.
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
# When hands' last sentence stopped playing: the play was 29.7 s long.
STOPPED = PLAYED + 29.7

# Each of hands' sentences as it played, one synthesis each.
SAID = tuple(Synthesis("charles", text, begun(VirtualClock())) for text in READ)


def played(start: float, seconds: float) -> Played:
    """What the speaker played over `seconds` of a recording from `start`, one reference for each 10 ms: each of hands'
    sentences from its onset to the next one's, and nothing once the play stopped."""
    onsets = [PLAYED + onset for onset in SENTENCES]

    def at(moment: float) -> Reference | None:
        playing = tuple(synthesis for onset, end, synthesis in zip(onsets, [*onsets[1:], STOPPED], SAID) if onset <= moment < end)
        return Reference(bytes(2 * SLOT), RATE, 1, playing) if playing else None

    return tuple(at(start + slot * SLOT / RATE) for slot in range(round(seconds * RATE / SLOT)))
