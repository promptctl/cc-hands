"""Record a person saying hands' own words over hands, through hands' echo canceller: a fixture's .flac and .json under
tests/fixtures, hands_cancelled_and_another for promptctl-voice-i15t.8vg.csd, with hands out of the speaker, and
hands_no_echo_and_another for promptctl-voice-i15t.8vg.0gs, with hands out of no speaker, as in headphones.

Three of hands' sentences are made by hands' own TTS ('charles', its default voice) and played out of the default
speaker, the canceller told each as hands tells it; a second voice ('alba'), not told to the canceller, says two lines
over the second and third that repeat hands' words where hands says them, as a person contradicting hands does. hands'
voice goes to the speaker at a gain, 1 at the desk and 0 as in headphones, and to the canceller whole either way. The
default microphone is heard through the canceller as the desk hears it. The .flac holds, at RATE, the audio as heard
after the canceller, as captured before it, and the reference each 10 ms of it was matched with; the .json, which of
hands' sentences each 10 ms was matched with and how far into it, and the lines of both voices with when each began.

Run by hand at a quiet desk with the speaker up, never by a test: it plays about 16 s of speech aloud.

    uv run python scripts/record-double-talk.py hands_cancelled_and_another 1
    uv run python scripts/record-double-talk.py hands_no_echo_and_another 0
"""

import json
import sys
import threading
import time
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
HANDS = (
    "Hands here. Starting houdini-mcp and opening macklebox now.",
    "Phosphene finished its render, and the build is green.",
    "I can hear you. Say the next step when you are ready.",
)
# Each line of the other voice: what it says, over which of hands' sentences, and which of hands' words in it it says
# where hands says them.
OTHER = (
    ("No, the build is broken.", 1, "the build is"),
    ("Say the next step again.", 2, "Say the next step"),
)
LEAD, GAP, TAIL = 1.0, 0.3, 2.0
BUFFER = 2 * SLOT  # 20 ms, as hands opens both streams


def main(fixture: Path, gain: float) -> None:
    say = pocket()
    made = [say(source(DEFAULT), text) for text in HANDS]
    onsets = list(np.cumsum([LEAD, *(len(audio) / RATE + GAP for audio in made[:-1])]))
    length = round((onsets[-1] + len(made[-1]) / RATE + TAIL) * RATE)
    hands = np.zeros(length, np.float32)
    sentence = np.full(length, -1)
    for index, (onset, audio) in enumerate(zip(onsets, made)):
        at = round(onset * RATE)
        hands[at : at + len(audio)] = audio
        sentence[at : at + len(audio)] = index
    other = np.zeros(length, np.float32)
    lines = []
    for text, over, words in OTHER:
        audio = say(source(parse_voice("alba")), text)
        # The other voice's words begin where hands', placed by its characters at Pocket TTS's own pace, begin.
        begins = onsets[over] + HANDS[over].index(words) / len(HANDS[over]) * len(made[over]) / RATE - text.index(words) / len(text) * len(audio) / RATE
        at = round(begins * RATE)
        other[at : at + len(audio)] = audio[: length - at]
        lines.append((text, begins))
    syntheses = [Synthesis(DEFAULT, text, begun(VirtualClock())) for text in HANDS]
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
    fixture.with_suffix(".json").write_text(
        json.dumps({"hands": list(HANDS), "other": [{"text": text, "start": round(begins + offset, 3)} for text, begins in lines], "played": slots}) + "\n"
    )
    print(f"{fixture}.flac: {len(heard) / 2 / RATE:.2f} s, canceller {echo.counts()}")


if __name__ == "__main__":
    main(FIXTURES / sys.argv[1], float(sys.argv[2]))
