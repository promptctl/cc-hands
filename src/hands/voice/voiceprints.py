"""Voiceprints: who a speaker in a hold is, told by their voice against hands' own, on 3D-Speaker's ERes2Net speaker
embedding run by sherpa-onnx on the CPU; nothing here loads Pipecat.

A voiceprint is the model's embedding of a stretch of speech, scaled to unit length, so two compare by their cosine
similarity. hands' own voiceprint is made by hands, from its own TTS output in each voice it speaks in
(`Voiceprints`), never from a person's voice and never by an enrolment the user performs. The diarizer
(`hands.voice.streaming`) splits a hold into speakers; its numbers are not identities, since one number can be two
voices and overlapped speech gets both, so each speaker in each hold is scored here, from that hold's audio alone.

Measured on zoj's recordings (promptctl-voice-i15t.99s.zoj, re-measured by scripts/measure-voiceprints.py): hands' echo
through the room scores 0.79 to 0.93 over long windows, another voice -0.04 to 0.16, and the two overlap below about a
second of speech, so how much of a speaker was heard decides what their score can say (`FLOORS`).

Who each speaker is is measured beside the cut and decides nothing yet. It is worked out on a lane of its own, after the
words it is about have gone out, so no hearing's words ever wait on a voiceprint.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from hands.sessions.audit import Record
from hands.sessions.wide import Begun, annotate, begun, ended
from hands.threads import SerialThread
from hands.voice import loads
from hands.voice.holds import RATE, Hold, Samples

# The model, fetched from the Hugging Face hub into its cache, at the revision measured for it.
MODEL = "csukuangfj/speaker-embedding-models"
REVISION = "0743f301363dec56491a490f6d6cbc9d67f9a3bf"
FILE = "3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx"

# A speaker's embedding, of unit length.
type Print = npt.NDArray[np.float32]

# The model: a stretch of mono speech at RATE, as its voiceprint.
type Embed = Callable[[Samples], Print]

# hands' own TTS: a text said in one of its voices, as mono speech at RATE.
type Synthesize = Callable[[str, str], Samples]


def score(a: Print, b: Print) -> float:
    """How alike two voiceprints are: their cosine similarity, 1 for the same direction."""
    return float(np.dot(a, b))


@dataclass(frozen=True)
class Person:
    """A speaker who is not hands: their voice scored `score` against hands' own voiceprint, on `seconds` of it."""

    score: float
    seconds: float


@dataclass(frozen=True)
class Hands:
    """A speaker who is hands' own voice, come back through the microphone."""

    score: float
    seconds: float


@dataclass(frozen=True)
class Unsure:
    """A speaker heard long enough to score, whose score is in hands' range at that length, and who is not yet heard
    long enough to be called hands."""

    score: float
    seconds: float


@dataclass(frozen=True)
class Unheard:
    """A speaker heard too briefly for any score to tell them from hands."""

    seconds: float


@dataclass(frozen=True)
class Unidentified:
    """A speaker nothing could be scored for, and why: the model did not load, or hands has no voiceprint yet of the
    voice it speaks in."""

    seconds: float
    why: str


# Who a speaker of a hold is, by their voiceprint against hands' own.
type Who = Person | Hands | Unsure | Unheard | Unidentified


@dataclass(frozen=True)
class Floor:
    """From `seconds` of a speaker's voice on, a score under `score` is not hands: under what hands' own echo scored at
    that length."""

    seconds: float
    score: float


# [LAW:one-source-of-truth] the one table a speaker is decided by, fitted where it is applied, on windows of each length
# from a speech onset (scripts/measure-voiceprints.py on zoj's captured.wav and mic2.wav, and dp5's people): from 1.0 s,
# people alone scored at most 0.19 and hands' echo at least 0.37; from 1.5 s, people over hands at most 0.66 but mostly
# far under, and the echo at least 0.64; from 3 s the echo at least 0.74, and the one line of a person over hands that
# long 0.36. Under a second nothing is decided: at 0.8 s the echo scored as low as 0.19, a person alone as high as 0.21.
# One room, one voice: hands records each hold it hears (`hands.voice.recorder`) so the table is refitted on real use.
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


@dataclass(frozen=True)
class Voiced:
    """One speaker in a hold, as a hearing found them: their number in the run, the words they added, None for none,
    the latest WINDOW of their voice in the hold, and how many seconds of them it has in all."""

    speaker: int
    words: str | None
    voice: Samples
    seconds: float


@dataclass(frozen=True)
class Speaker:
    """One speaker in a hold, identified: their number in the run, the words they added, and who they are."""

    number: int
    words: str | None
    who: Who


def scored(embed: Embed, prints: tuple[Print, ...], voiced: Voiced) -> Who:
    """Who a speaker is against hands' `prints`, the highest score of any deciding. Under a second, nothing is scored."""
    match voiced.seconds < ALONE.seconds:
        case True:
            return Unheard(voiced.seconds)
        case False:
            voiceprint = embed(voiced.voice)
            return decided(max(score(voiceprint, each) for each in prints), voiced.seconds)


# What hands says to make its own voiceprint in a voice: about twelve seconds of the kind of thing it says, never played.
ENROLMENT = (
    "Hands here. The build finished, and two sessions are waiting on you. One of them asked a question about the release, "
    "and the other is ready to move on. Say which one you want to hear first, or hold the key to talk."
)


@dataclass(frozen=True)
class Unloaded:
    """A model that did not load, and why."""

    why: str


@dataclass(frozen=True)
class Unprinted:
    """A voice hands has no voiceprint of, and why: not made yet, or its enrolment failed."""

    why: str


def failure(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


class Voiceprints:
    """hands' voiceprint lane: the model, hands' own prints, and every identification, on a thread of their own, and
    hands' TTS for its prints on another, so neither a hearing's words nor anything hands says waits on them.

    hands keeps the voiceprint of the voice it speaks in now, and of the one before the last change, whose echo may
    still be in the room; a voice only sampled is never one hands speaks in (`hands.voice.voices.Voices.hear`), and no
    voiceprint is kept on disk: each run makes its own from the TTS it runs.

    [LAW:no-shared-mutable-globals] the model, the prints, the voices hands speaks in, and who was a person in the hold
    are the lane's alone, read and written only by the work given to it, in order; the TTS is the enrolment thread's.
    [LAW:no-silent-failure] a model that does not load, or a voiceprint that cannot be made, fails its own unit with
    why, and every speaker identified after is Unidentified with that why, never scored against what is not there.
    """

    def __init__(self, load: Callable[[], Embed], speech: Callable[[], Synthesize], record: Record) -> None:
        self._record = record
        self._lane = SerialThread("Voiceprints")
        self._enrolling = SerialThread("Enrolment")
        # Set by each thread's first work, and read only by the work behind it on the same thread.
        self._embed: Embed | Unloaded
        self._say: Synthesize | Unloaded
        self._prints: dict[str, Print | Unprinted] = {}
        # The voice before the last change, and the one now: at most two, the last the one hands speaks in.
        self._speaking: tuple[str, ...] = ()
        # The hold the people below were found in, and each speaker found a person in it, which they stay for it.
        self._people: tuple[int, dict[int, Person]] = (0, {})
        self._lane.give(lambda: self._loaded(load))
        self._enrolling.give(lambda: self._speech_loaded(speech))

    def _loaded(self, load: Callable[[], Embed]) -> None:
        """[LAW:nothing-unseen] the model's load is a unit of its own, as each speech model's is."""
        try:
            with loads.loading("voiceprints.loaded", self._record):
                annotate(model=MODEL, revision=REVISION, file=FILE)
                self._embed = load()
        except Exception as error:
            self._embed = Unloaded(f"the voiceprint model did not load: {failure(error)}")

    def _speech_loaded(self, speech: Callable[[], Synthesize]) -> None:
        """hands' TTS for its prints, one model loaded once for the run: a unit of its own."""
        try:
            with loads.loading("enrolment.loaded", self._record):
                self._say = speech()
        except Exception as error:
            self._say = Unloaded(f"hands' TTS for its prints did not load: {failure(error)}")

    def speaking_in(self, voice: str) -> None:
        """hands speaks in `voice` from now: its voiceprint is made, off everything hands says, and until it is, every
        speaker is Unidentified. Returns at once, and cannot fail: a voiceprint not made fails its own unit."""
        began = begun()
        self._lane.give(lambda: self._spoken_in(voice))
        self._enrolling.give(lambda: self._enrol(voice, began))

    def _spoken_in(self, voice: str) -> None:
        # The voice now, behind whichever other one hands spoke in last; the prints of every other voice let go of.
        self._speaking = (*[each for each in self._speaking if each != voice][-1:], voice)
        self._prints = {each: voiceprint for each, voiceprint in self._prints.items() if each in self._speaking}

    def _enrol(self, voice: str, began: Begun) -> None:
        """ENROLMENT said in `voice`, on the enrolment thread; its voiceprint made on the lane."""
        said: Samples | Exception
        match self._say:
            case Unloaded(why=why):
                said = RuntimeError(why)
            case say:
                try:
                    said = say(voice, ENROLMENT)
                except Exception as error:
                    said = error
        self._lane.give(lambda: self._enrolled(voice, said, began))

    def _enrolled(self, voice: str, said: Samples | Exception, began: Begun) -> None:
        """[LAW:nothing-unseen] each voiceprint made, or not, is one `voiceprints.enrolled` unit: which voice, how long the
        speech it was made from, and why where it failed."""
        made: Print | Unprinted
        raised: Exception | None = None
        try:
            match said, self._embed:
                case Exception() as error, _:
                    raise error
                case _, Unloaded(why=why):
                    raise RuntimeError(why)
                case samples, embed:
                    made = embed(samples)
        except Exception as error:
            made, raised = Unprinted(f"hands' voiceprint of {voice} could not be made: {failure(error)}"), error
        seconds = None if isinstance(said, Exception) else round(len(said) / RATE, 3)
        ended("voiceprints.enrolled", self._record, began, raised, voice=voice, seconds=seconds)
        self._prints = {each: voiceprint for each, voiceprint in {**self._prints, voice: made}.items() if each in self._speaking}

    def identify(self, hold: Hold, voiced: tuple[Voiced, ...]) -> None:
        """Who each speaker in `voiced` is, worked out on the lane and told as one `voiceprints.identified` unit, which
        hangs off the unit open here. Returns at once."""
        began = begun()
        self._lane.give(lambda: self._identified(hold, voiced, began))

    def _identified(self, hold: Hold, voiced: tuple[Voiced, ...], began: Begun) -> None:
        try:
            speakers = tuple(Speaker(each.speaker, each.words, self._who(hold, each)) for each in voiced)
        except Exception as error:
            ended("voiceprints.identified", self._record, began, error, hold=hold.number)
            return
        ended("voiceprints.identified", self._record, began, None, hold=hold.number, speakers=speakers)

    def _who(self, hold: Hold, voiced: Voiced) -> Who:
        """[LAW:single-enforcer] once a person in a hold, a speaker is one for the rest of it."""
        number, people = self._people
        people = people if number == hold.number else {}
        match people.get(voiced.speaker):
            case Person() as known:
                who: Who = known
            case None:
                who = self._judged(voiced)
        match who:
            case Person():
                people = {**people, voiced.speaker: who}
            case _:
                pass
        self._people = (hold.number, people)
        return who

    def _judged(self, voiced: Voiced) -> Who:
        match self._embed, self._kept():
            case Unloaded(why=why), _:
                return Unidentified(voiced.seconds, why)
            case _, Unprinted(why=why):
                return Unidentified(voiced.seconds, why)
            case embed, tuple() as prints:
                return scored(embed, prints, voiced)

    def _kept(self) -> tuple[Print, ...] | Unprinted:
        """Every voiceprint hands keeps, so long as it has the one of the voice it speaks in now."""
        match self._speaking:
            case (*_, now):
                match self._prints.get(now, Unprinted(f"no print yet of {now}")):
                    case Unprinted() as missing:
                        return missing
                    case _:
                        return tuple(voiceprint for voiceprint in self._prints.values() if not isinstance(voiceprint, Unprinted))
            case _:
                return Unprinted("hands speaks in no voice yet")


def load() -> Embed:
    """The model, from the Hugging Face cache, fetched there if it never was.

    [LAW:no-silent-failure] a model that does not load raises naming the file: without it hands cannot tell its own voice
    from a person's."""
    # sherpa-onnx ships no types, and the hub's download types one argument incompletely.
    import sherpa_onnx  # pyright: ignore[reportMissingTypeStubs]
    from huggingface_hub import hf_hub_download  # pyright: ignore[reportUnknownVariableType]

    try:
        path = hf_hub_download(MODEL, FILE, revision=REVISION)
        extractor = cast(Any, sherpa_onnx).SpeakerEmbeddingExtractor(cast(Any, sherpa_onnx).SpeakerEmbeddingExtractorConfig(model=path, num_threads=2))
    except Exception as error:
        raise RuntimeError(f"the voiceprint model {FILE} from {MODEL} at {REVISION} did not load: {failure(error)}") from error

    def embed(voice: Samples) -> Print:
        stream = extractor.create_stream()
        stream.accept_waveform(sample_rate=RATE, waveform=voice)
        stream.input_finished()
        embedded = np.asarray(extractor.compute(stream), np.float32)
        return embedded / np.linalg.norm(embedded)

    return embed
