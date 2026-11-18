"""Record a person speaking over hands, through hands' echo canceller: a fixture's .flac and .json under tests/fixtures,
each take of one of the scripts in `TAKES`, for what a person and hands actually do at the desk: a person saying hands' own
words over the middle of its sentences, over the opening of a reply that follows a pause, a short "yes", "no" or "stop"
over it, over a reply in a second of its voices, and alone after it; and hands alone, in long sentences. Each with hands out
of the speaker, as at the desk (`hands_<what>`), and where a person speaks over hands, out of no speaker, as in headphones
(`hands_no_echo_<what>`).

hands' sentences are made by hands' own TTS, each in the voice its script gives, and played out of the default speaker, the
canceller told each as hands tells it; a second voice ('alba'), not told to the canceller, says the person's lines. hands'
voice goes to the speaker at a gain, 1 at the desk and 0 as in headphones, and to the canceller whole either way. The default
microphone is heard through the canceller as the desk hears it. The .flac holds, at RATE, the audio as heard after the
canceller, as captured before it, and the reference each 10 ms of it was matched with; the .json, hands' sentences with the
voice of each, which of them each 10 ms was matched with and how far into it, and the lines of the person with when each began.

The speaker is kept muted: a take unmutes it for the playing and mutes it again as it was.

Run by hand in a quiet room, never by a test: each take plays up to 30 s of speech aloud, and the microphone hears the room too,
so a take with anyone else's voice in it is recorded again. `scripts/measure-echo.py raw` shows it: a word that is neither
hands' nor the person's, where hands is out of the speaker.

    uv run python scripts/record-double-talk.py hands_cancelled_and_another
    uv run python scripts/record-double-talk.py        # lists the takes
"""

import json
import subprocess
import sys
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyaudio
import soundfile

from hands.sessions.wide import begun
from hands.virtualclock import VirtualClock
from hands.voice.clips import RATE, SLOT, Playhead, Synthesis
from hands.voice.echo import EchoCanceller
from hands.voice.recorder import track
from hands.voice.voices import DEFAULT, parse_voice, pocket, source

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


@dataclass(frozen=True)
class Reply:
    """One of hands' sentences: what it says, in which voice, and the pause before it (the first's is `LEAD`)."""

    text: str
    pause: float = 0.3
    voice: str = str(DEFAULT)


@dataclass(frozen=True)
class Alongside:
    """The person's line says `words`, which hands' reply says too, where hands says them."""

    reply: int
    words: str


@dataclass(frozen=True)
class Into:
    """The person's line begins `seconds` into hands' reply."""

    reply: int
    seconds: float


@dataclass(frozen=True)
class After:
    """The person's line begins `seconds` after hands' reply ended."""

    reply: int
    seconds: float


@dataclass(frozen=True)
class Said:
    """A line of the person's: what it says, and where against hands' replies it begins."""

    text: str
    placed: Alongside | Into | After


@dataclass(frozen=True)
class Script:
    """What a take plays: hands' replies, the first begun after `LEAD`, and the person's lines."""

    hands: tuple[Reply, ...]
    person: tuple[Said, ...]


MIDDLE = (
    Reply("Hands here. Starting houdini-mcp and opening macklebox now."),
    Reply("Phosphene finished its render, and the build is green."),
    Reply("I can hear you. Say the next step when you are ready."),
)
# The person says some of hands' own words where hands says them.
DOUBLE_TALK = Script(MIDDLE, (Said("No, the build is broken.", Alongside(1, "the build is")), Said("Say the next step again.", Alongside(2, "Say the next step"))))
# The person says only hands' own words, the whole of each line, as hands says them.
SAME_WORDS = Script(MIDDLE, (Said("the build is green", Alongside(1, "the build is green")), Said("Say the next step when you are ready.", Alongside(2, "Say the next step when you are ready."))))
# A reply hands says alone, and after a pause longer than any between sentences of one reply, another a person cuts into as
# it opens.
OPENING = Script(
    (Reply("Hands here. Starting houdini-mcp and opening macklebox now."), Reply("Checking the speech models before we continue with the release.", 4.0)),
    (Said("Wait, stop, that is not what I asked.", Into(1, 0.0)),),
)
# A "no", a "stop" and a "yes", one word over each reply.
SHORT_WORDS = Script(MIDDLE, (Said("No.", Into(0, 2.0)), Said("Stop.", Into(1, 2.5)), Said("Yes.", Into(2, 2.5))))
# A reply in a second of hands' voices, the person over the middle of it saying words of their own.
TWO_VOICES = Script(
    (
        Reply("Hands here. Starting houdini-mcp and opening macklebox now."),
        Reply("The render finished, and the review you asked for is ready to read.", 0.8, "marius"),
        Reply("Tell me which file you want first and I will open it.", 0.5),
    ),
    (Said("Wait, which review do you mean?", Into(1, 2.0)),),
)
# Long sentences, with no person anywhere: only hands, to be taken out.
LONG_AND_SLOW = Script(
    (
        Reply("I have read the three files you pointed me at, and before I change anything, I want to walk through what I found, slowly, one piece at a time, so that you can stop me if I have anything wrong."),
        Reply(
            "The first file builds the list of sessions, the second decides which of them should speak next, and the third, which is the one I am least sure about, writes the answer back to the terminal where you are working.",
            1.5,
        ),
    ),
    (),
)
# A reply, and after it hands silent while the person speaks, a long sentence and then a short word.
ALONE = Script(
    (Reply("Hands here. Starting houdini-mcp and opening macklebox now."),),
    (Said("The build is broken on the branch I pushed this morning, so please hold the deploy until I have looked at it.", After(0, 2.5)), Said("Yes.", After(0, 13.0))),
)
# Each take's script, and hands' gain to the speaker: 1 at the desk, 0 as in headphones.
TAKES = {
    "hands_cancelled_and_another": (DOUBLE_TALK, 1.0),
    "hands_no_echo_and_another": (DOUBLE_TALK, 0.0),
    "hands_over_an_opening": (OPENING, 1.0),
    "hands_no_echo_over_an_opening": (OPENING, 0.0),
    "hands_and_the_same_words": (SAME_WORDS, 1.0),
    "hands_no_echo_and_the_same_words": (SAME_WORDS, 0.0),
    "hands_and_short_words": (SHORT_WORDS, 1.0),
    "hands_no_echo_and_short_words": (SHORT_WORDS, 0.0),
    "hands_in_two_voices": (TWO_VOICES, 1.0),
    "hands_no_echo_in_two_voices": (TWO_VOICES, 0.0),
    "hands_long_and_slow": (LONG_AND_SLOW, 1.0),
    "hands_then_a_person_alone": (ALONE, 1.0),
}
LEAD, TAIL = 1.0, 2.0
BUFFER = 2 * SLOT  # 20 ms, as hands opens both streams


def osascript(line: str) -> str:
    return subprocess.run(["osascript", "-e", line], check=True, capture_output=True, text=True).stdout.strip()


@contextmanager
def speaker_up() -> Generator[None]:
    """The desk's speaker unmuted for what runs inside, and muted again as it was after, whatever happens in between."""
    was = osascript("output muted of (get volume settings)")
    # [LAW:parse-dont-validate] an output with no mute control reports `missing value`: there is no state to put back.
    if was not in ("true", "false"):
        raise RuntimeError(f"the default output has no mute to unmute and restore: it reports {was!r}")
    osascript("set volume output muted false")
    try:
        yield
    finally:
        osascript(f"set volume output muted {was}")


def main(fixture: Path, script: Script, gain: float) -> None:
    say = pocket()
    made = [say(source(parse_voice(reply.voice)), reply.text) for reply in script.hands]
    onsets = list(np.cumsum([LEAD, *(len(audio) / RATE + reply.pause for audio, reply in zip(made[:-1], script.hands[1:], strict=True))]))
    lines = [(line, say(source(parse_voice("alba")), line.text)) for line in script.person]

    def begins(line: Said, audio: np.ndarray) -> float:
        match line.placed:
            case Alongside(reply, words):
                # The line's words begin where hands', placed by its characters at Pocket TTS's own pace, begin.
                text = script.hands[reply].text
                return onsets[reply] + text.index(words) / len(text) * len(made[reply]) / RATE - line.text.index(words) / len(line.text) * len(audio) / RATE
            case Into(reply, seconds):
                return onsets[reply] + seconds
            case After(reply, seconds):
                return onsets[reply] + len(made[reply]) / RATE + seconds

    starts = [begins(line, audio) for line, audio in lines]
    ends = [start + len(audio) / RATE for start, (_, audio) in zip(starts, lines)]
    # [LAW:no-silent-failure] a line begun before the take, or over the person's own line before it, is a script that does not say what it plays.
    if any(start < 0 or start < before for start, before in zip(starts, [0.0, *ends], strict=False)):
        raise ValueError("the person's lines do not follow one another from the start of the take: " + "; ".join(f"{line.text!r} {start:.2f}-{end:.2f} s" for (line, _), start, end in zip(lines, starts, ends)))
    length = round((max([onsets[-1] + len(made[-1]) / RATE, *ends]) + TAIL) * RATE)
    hands = np.zeros(length, np.float32)
    sentence = np.full(length, -1)
    for index, (onset, audio) in enumerate(zip(onsets, made)):
        at = round(onset * RATE)
        hands[at : at + len(audio)] = audio
        sentence[at : at + len(audio)] = index
    other = np.zeros(length, np.float32)
    for start, (_, audio) in zip(starts, lines):
        at = round(start * RATE)
        other[at : at + len(audio)] = audio[: length - at]
    syntheses = [Synthesis(reply.voice, reply.text, begun(VirtualClock())) for reply in script.hands]
    out = np.clip((hands * gain + other) * 32767, -32768, 32767).astype(np.int16).tobytes()
    reference = np.clip(hands * 32767, -32768, 32767).astype(np.int16).tobytes()

    echo = EchoCanceller()
    heard, captured, played = bytearray(), bytearray(), []
    position = 0
    done = threading.Event()

    def speaker(_in: bytes | None, frames: int, _time: object, _status: int) -> tuple[bytes, int]:
        nonlocal position
        start, stop = position, position + frames
        position = stop
        chunk = out[2 * start : 2 * stop].ljust(2 * frames, b"\0")
        # As hands' speaker gives the canceller: a playhead for each sentence the buffer holds any of, where in it the
        # buffer's first sample of it is.
        playheads = tuple(Playhead(syntheses[at], max(start / RATE - onsets[at], 0.0)) for at in dict.fromkeys(sentence[start:stop].tolist()) if at >= 0)
        echo.played(reference[2 * start : 2 * stop].ljust(2 * frames, b"\0"), RATE, 1, playheads)
        if stop >= length:
            done.set()
        return chunk, pyaudio.paContinue if stop < length else pyaudio.paComplete

    def microphone(in_data: bytes, _frames: int, _time: object, _status: int) -> tuple[None, int]:
        cancelled = echo.heard(in_data, RATE, 1)
        captured.extend(in_data)
        heard.extend(cancelled.heard)
        played.extend(cancelled.played)
        return None, pyaudio.paContinue

    with speaker_up():
        audio = pyaudio.PyAudio()
        listening = audio.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=BUFFER, stream_callback=microphone)
        time.sleep(0.2)
        speaking = audio.open(format=pyaudio.paInt16, channels=1, rate=RATE, output=True, frames_per_buffer=BUFFER, stream_callback=speaker)
        done.wait()
        time.sleep(0.5)
        speaking.close()
        listening.close()
        audio.terminate()
        echo.close()

    as_float = lambda raw: np.frombuffer(bytes(raw), np.int16).astype(np.float32) / 32768  # noqa: E731
    channels = np.stack([as_float(heard), as_float(captured), as_float(track(tuple(played)))], axis=1)
    soundfile.write(f"{fixture}.flac", channels, RATE, subtype="PCM_16")
    # Where the speaker's timeline falls on the recording's, which began as the microphone opened: the first slot the
    # canceller matched with hands' speech, less how far into its sentence that was and when the sentence began: to
    # within a 20 ms buffer, and early by it, not late, as the buffer a sentence begins in is placed at its start.
    first = next(reference for reference in played if reference is not None)
    offset = played.index(first) * SLOT / RATE - first.playheads[0].into - onsets[syntheses.index(first.playheads[0].synthesis)]
    slots = [None if reference is None else [syntheses.index(reference.playheads[0].synthesis), round(reference.playheads[0].into, 3)] for reference in played]
    described = {
        "hands": [reply.text for reply in script.hands],
        "voices": [reply.voice for reply in script.hands],
        "other": [{"text": line.text, "start": round(start + offset, 3)} for (line, _), start in zip(lines, starts)],
        "played": slots,
    }
    fixture.with_suffix(".json").write_text(json.dumps(described) + "\n")
    print(f"{fixture}.flac: {len(heard) / 2 / RATE:.2f} s, canceller {echo.counts()}")


if __name__ == "__main__":
    match sys.argv[1:]:
        case [fixture] if fixture in TAKES:
            main(FIXTURES / fixture, *TAKES[fixture])
        case _:
            sys.exit(f"usage: record-double-talk.py FIXTURE (one of {', '.join(TAKES)})")
