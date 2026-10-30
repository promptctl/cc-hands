"""Voiceprints: who a speaker in a hold is, told by their voice against hands' own, on 3D-Speaker's ERes2Net speaker
embedding run by sherpa-onnx on the CPU; nothing here loads Pipecat.

A voiceprint is the model's embedding of a stretch of speech, scaled to unit length, so two compare by their cosine
similarity. hands' own print is made by hands, from its own TTS output in each voice it speaks in (`HandsVoice`), never
from a person's voice and never by an enrolment the user performs. The diarizer (`hands.voice.streaming`) splits a hold
into speakers; its numbers are not identities, since one number can be two voices and overlapped speech gets both, so each
speaker in each hold is scored here, from that hold's audio alone.

Measured on zoj's recordings (promptctl-voice-i15t.99s.zoj, re-measured by scripts/measure-voiceprints.py): hands' echo
through the room scores 0.79 to 0.93 over long windows, another voice -0.04 to 0.16, and the two overlap below about a
second of speech, so how much of a speaker was heard decides what their score can say (`FLOORS`).
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from hands.sessions.audit import Record
from hands.sessions.wide import annotate, unit
from hands.voice import loads
from hands.voice.holds import RATE, Hands, Person, Samples, Unheard, Unsure, Who

# The model, fetched from the Hugging Face hub into its cache, at the revision measured for it.
MODEL = "csukuangfj/speaker-embedding-models"
REVISION = "0743f301363dec56491a490f6d6cbc9d67f9a3bf"
FILE = "3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx"

# A speaker's embedding, of unit length.
type Print = npt.NDArray[np.float32]

# The model: a stretch of mono speech at RATE, as its print.
type Embed = Callable[[Samples], Print]


def score(a: Print, b: Print) -> float:
    """How alike two voiceprints are: their cosine similarity, 1 for the same direction."""
    return float(np.dot(a, b))


@dataclass(frozen=True)
class Floor:
    """From `seconds` of a speaker's voice on, a score under `score` is not hands: under what hands' own echo scored at
    that length."""

    seconds: float
    score: float


# [LAW:one-source-of-truth] the one table a speaker is decided by, fitted where it is applied, on windows of each length
# from a speech onset (promptctl-voice-i15t.8vg.di1's plan, from zoj's captured.wav and mic2.wav, and dp5's people):
# from 1.0 s, people alone scored at most 0.18 and hands' echo at least 0.37; from 1.5 s, people over hands at most 0.66
# but mostly far under, and the echo at least 0.64; from 3 s, people over hands at most 0.64 and the echo at least 0.74.
# Under a second nothing is decided: at 0.8 s the echo scored as low as 0.19, a person alone as high as 0.13. One
# room, one voice: hands records each hold it hears (`hands.voice.recorder`) so the table is refitted on real use.
FLOORS = (Floor(1.0, 0.25), Floor(1.5, 0.55), Floor(3.0, 0.70))
ALONE, PERSON, HANDS = FLOORS

# The most of a speaker's voice, their latest, a score is taken on: the table's longest row, past which nothing was
# measured, and the bound on what scoring costs, about 18 ms for each second scored, at every hearing.
WINDOW = HANDS.seconds


def decided(scored: float, seconds: float) -> Who:
    """Who a speaker is whose latest voice scored `scored` against hands', `seconds` of them heard in all.
    [LAW:dataflow-not-control-flow] one row of the table decides, the longest the speaker has reached."""
    floor = next(row for row in reversed(FLOORS) if row.seconds <= seconds)
    match scored < floor.score, seconds >= HANDS.seconds:
        case True, _:
            return Person(scored, seconds)
        case False, True:
            return Hands(scored, seconds)
        case False, False:
            return Unsure(scored, seconds)


# Who a speaker is, from all the hold has of their voice and how many seconds of it there were.
type Identify = Callable[[Samples, float], Who]


class HandsVoice:
    """hands' own voice as voiceprints, one for each voice it has spoken in this run, each made from that voice's own TTS
    output: whichever it speaks in now, and the one before a change, whose echo may still be in the room.

    [LAW:no-shared-mutable-globals] enrolment, off the event loop, is the one writer, and replaces the prints whole; the
    streaming listener's thread reads them. No print is kept on disk: each run makes its own from the TTS it runs.
    """

    def __init__(self, voice: str, embed: Embed, synthesize: Callable[[str, str], Samples], record: Record) -> None:
        self._embed = embed
        self._synthesize = synthesize
        self._record = record
        self._prints: dict[str, Print] = {}
        self.enrol(voice)

    def enrol(self, voice: str) -> None:
        """Make `voice`'s print, unless this run has: ENROLMENT said in it by hands' TTS, as one unit of work. Slow, about
        3.5 s on this Mac with the TTS model's own load: run off the event loop."""
        if voice in self._prints:
            return
        # [LAW:nothing-unseen] each print made is a unit: which voice, and how long the speech it was made from.
        with unit("voiceprints.enrolled", self._record):
            annotate(voice=voice)
            said = self._synthesize(voice, ENROLMENT)
            annotate(seconds=round(len(said) / RATE, 3))
            self._prints = {**self._prints, voice: self._embed(said)}

    def score(self, voice: Print) -> float:
        """How like hands a print is: its highest score against any voice hands has spoken in."""
        return max(score(voice, each) for each in self._prints.values())

    def identify(self, voice: Samples, seconds: float) -> Who:
        """Who a speaker is, `voice` the latest WINDOW of all `seconds` the hold has of them."""
        match seconds < ALONE.seconds:
            case True:
                return Unheard(seconds)
            case False:
                return decided(self.score(self._embed(voice[-round(WINDOW * RATE) :])), seconds)


# What hands says to make its own print in a voice: about twelve seconds of the kind of thing it says, never played.
ENROLMENT = (
    "Hands here. The build finished, and two sessions are waiting on you. One of them asked a question about the release, "
    "and the other is ready to move on. Say which one you want to hear first, or hold the key to talk."
)


def load(record: Record) -> Embed:
    """The model, from the Hugging Face cache, fetched there if it never was, as one unit of work.

    [LAW:no-silent-failure] a model that does not load fails hands' start, naming the file: without it hands cannot tell
    its own voice from a person's."""
    # sherpa-onnx ships no types, and the hub's download types one argument incompletely.
    import sherpa_onnx  # pyright: ignore[reportMissingTypeStubs]
    from huggingface_hub import hf_hub_download  # pyright: ignore[reportUnknownVariableType]

    with loads.loading("voiceprints.loaded", record):
        annotate(model=MODEL, revision=REVISION, file=FILE)
        try:
            path = hf_hub_download(MODEL, FILE, revision=REVISION)
            extractor = cast(Any, sherpa_onnx).SpeakerEmbeddingExtractor(cast(Any, sherpa_onnx).SpeakerEmbeddingExtractorConfig(model=path, num_threads=2))
        except Exception as error:
            raise RuntimeError(f"the voiceprint model {FILE} from {MODEL} at {REVISION} did not load: {type(error).__name__}: {error}") from error
        annotate(dimensions=extractor.dim)

    def embed(voice: Samples) -> Print:
        stream = extractor.create_stream()
        stream.accept_waveform(sample_rate=RATE, waveform=voice)
        stream.input_finished()
        embedded = np.asarray(extractor.compute(stream), np.float32)
        return embedded / np.linalg.norm(embedded)

    return embed
