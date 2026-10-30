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
from pathlib import Path
from typing import Any, Protocol, cast
from urllib.parse import quote

import numpy as np
import numpy.typing as npt

from hands.sessions.audit import Record
from hands.sessions.wide import Begun, annotate, begun, ended, unit
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
class Window:
    """What a speaker's score was taken on: `seconds` of their latest voice, `shared` of them in frames the diarizer
    heard another speaker in too, so a refit is fitted on the very input the table is applied to."""

    seconds: float
    shared: float


@dataclass(frozen=True)
class Person:
    """A speaker who is not hands: their voice scored `score` against hands' own voiceprint, on `window` of it, with
    `seconds` of them heard in all."""

    score: float
    seconds: float
    window: Window


@dataclass(frozen=True)
class Hands:
    """A speaker who is hands' own voice, come back through the microphone."""

    score: float
    seconds: float
    window: Window


@dataclass(frozen=True)
class Unsure:
    """A speaker heard long enough to score, whose score is in hands' range at that length, and who is not yet heard
    long enough to be called hands."""

    score: float
    seconds: float
    window: Window


@dataclass(frozen=True)
class Unheard:
    """A speaker heard too briefly for any score to tell them from hands."""

    seconds: float


@dataclass(frozen=True)
class Unidentified:
    """A speaker nothing could be scored for, and why: the model did not load, or hands holds no voiceprint of the
    voice it speaks in, or of one it played over them."""

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


# [LAW:one-source-of-truth] the one table a speaker is decided by, fitted where it is applied. A first estimate, from
# contiguous clean windows from a speech onset, not from the diarizer's frames a speaker is scored on here
# (scripts/measure-voiceprints.py on zoj's captured.wav and mic2.wav, and dp5's people): from 1.0 s, people alone scored
# at most 0.19 and hands' echo at least 0.37; from 1.5 s, people over hands at most 0.66 but mostly far under, and the
# echo at least 0.64; from 3 s the echo at least 0.74, and the one line of a person over hands that long 0.36. Under a
# second nothing is decided: at 0.8 s the echo scored as low as 0.19, a person alone as high as 0.21. One room, one
# voice: hands records each hold it hears (`hands.voice.recorder`), and each identification says the window it scored,
# so the table is refitted on real use (promptctl-voice-i15t.8vg.xvr).
FLOORS = (Floor(1.0, 0.25), Floor(1.5, 0.55), Floor(3.0, 0.70))
ALONE, PERSON, HANDS = FLOORS

# The most of a speaker's voice, their latest, a score is taken on: the table's longest row, past which nothing was
# measured, and the bound on what scoring costs, about 18 ms for each second scored.
WINDOW = HANDS.seconds


def reached(seconds: float) -> int:
    """How many rows of the table `seconds` of a speaker has reached: a speaker is scored as they reach each."""
    return sum(row.seconds <= seconds for row in FLOORS)


def decided(scored: float, seconds: float, window: Window) -> Who:
    """Who a speaker is whose latest voice, `window` of it, scored `scored` against hands', `seconds` of them heard in
    all. [LAW:dataflow-not-control-flow] one row of the table decides, the longest the speaker has reached."""
    floor = next(row for row in reversed(FLOORS) if row.seconds <= seconds)
    match scored < floor.score, seconds >= HANDS.seconds:
        case True, _:
            return Person(scored, seconds, window)
        case False, True:
            return Hands(scored, seconds, window)
        case False, False:
            return Unsure(scored, seconds, window)


@dataclass(frozen=True)
class Voiced:
    """One speaker in a hold, as the stream heard them: their number in the run, the latest WINDOW of their voice in
    the hold, how many seconds of them it has in all, how many of the latest were frames shared with another speaker,
    and every voice of hands' TTS played over any of them."""

    speaker: int
    voice: Samples
    seconds: float
    shared: float
    played: tuple[str, ...]


@dataclass(frozen=True)
class Speaker:
    """One speaker in a hold, identified: their number in the run, and who they are."""

    number: int
    who: Who


def scored(embed: Embed, prints: tuple[Print, ...], voiced: Voiced) -> Who:
    """Who a speaker is against hands' `prints`, the highest score of any deciding. Under a second, nothing is scored."""
    match voiced.seconds < ALONE.seconds:
        case True:
            return Unheard(voiced.seconds)
        case False:
            voiceprint = embed(voiced.voice)
            window = Window(round(len(voiced.voice) / RATE, 3), round(voiced.shared, 3))
            return decided(max(score(voiceprint, each) for each in prints), voiced.seconds, window)


class Identifier(Protocol):
    """What identifies the speakers the stream hears: each as they reach a row of the table, and every one at the end
    of their hold."""

    def identify(self, hold: Hold, voiced: tuple[Voiced, ...]) -> None: ...
    def ended(self, hold: Hold, voiced: tuple[Voiced, ...]) -> None: ...


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
    """A voice hands holds no voiceprint of, and why."""

    why: str


def failure(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


def stored(directory: Path, voice: str) -> Path:
    """Where hands keeps its voiceprint of `voice`: one file for each voice, in a directory for each revision of the
    model, since a print is only comparable with another of the same model."""
    return directory / REVISION / f"{quote(voice, safe='')}.npy"


def kept(path: Path, voiceprint: Print) -> None:
    """`voiceprint` written whole to `path`, or not at all."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    with partial.open("wb") as out:
        np.save(out, voiceprint)
    partial.replace(path)


class Voiceprints:
    """hands' voiceprint lane: the model, hands' own prints, and every identification, on a thread of their own, and
    hands' TTS for its prints on another, so neither a hearing's words nor anything hands says waits on them.

    Each voice is enrolled once ever: its print, made by hands from its own TTS and never from a person's voice, is
    kept in `kept`, read back from there each run it is spoken in, and never made again, so choosing a voice hands has
    a print of, the one it speaks in included, makes nothing. Enrolment loads a Pocket TTS model of its own for each
    print it makes, and lets it go once made. hands holds in memory the print of the voice it speaks in now, and of the
    one before the last change, whose echo may still be in the room; a voice only sampled is never one hands speaks
    in (`hands.voice.voices.Voices.hear`).

    [LAW:no-shared-mutable-globals] the model, the prints, the voices hands speaks in, and who was a person in each
    hold are the lane's alone, read and written only by the work given to it, in order.
    [LAW:no-silent-failure] a model that does not load, or a voiceprint that cannot be made or kept, fails its own
    unit with why, and every speaker it leaves nothing to score against is Unidentified with that why.
    """

    def __init__(self, load: Callable[[], Embed], speech: Callable[[], Synthesize], kept: Path, record: Record) -> None:
        self._speech = speech
        self._kept_in = kept
        self._record = record
        self._lane = SerialThread("Voiceprints")
        self._enrolment = SerialThread("Enrolment")
        # Set by the lane's first work, and read only by the work behind it.
        self._embed: Embed | Unloaded
        self._prints: dict[str, Print | Unprinted] = {}
        # The voices whose print the enrolment thread is making.
        self._enrolling: set[str] = set()
        # The voice before the last change, and the one now: at most two, the last the one hands speaks in.
        self._speaking: tuple[str, ...] = ()
        # [LAW:one-source-of-truth] each speaker found a person in each hold, which they stay for it, by the hold's
        # number: a hold's last identification, at its end, may come after the next hold's first.
        self._people: dict[int, dict[int, Person]] = {}
        self._lane.give(lambda: self._loaded(load))

    def _loaded(self, load: Callable[[], Embed]) -> None:
        """[LAW:nothing-unseen] the model's load is a unit of its own, as each speech model's is."""
        try:
            with loads.loading("voiceprints.loaded", self._record):
                annotate(model=MODEL, revision=REVISION, file=FILE)
                self._embed = load()
        except Exception as error:
            self._embed = Unloaded(f"the voiceprint model did not load: {failure(error)}")

    def speaking_in(self, voice: str) -> None:
        """hands speaks in `voice` from now: its voiceprint is read back, or made, off everything hands says, and until
        it is, every speaker is Unidentified. Returns at once, and cannot fail: a print not made fails its own unit."""
        began = begun()
        self._lane.give(lambda: self._spoken_in(voice, began))

    def _spoken_in(self, voice: str, began: Begun) -> None:
        # The voice now, behind whichever other one hands spoke in last; the prints of every other voice let go of.
        self._speaking = (*[each for each in self._speaking if each != voice][-1:], voice)
        self._prints = {each: voiceprint for each, voiceprint in self._prints.items() if each in self._speaking}
        match self._prints.get(voice):
            case np.ndarray():
                return
            case _ if voice in self._enrolling:
                return
            case _:
                pass
        path = stored(self._kept_in, voice)
        if path.exists() and self._recalled(voice, path):
            return
        match self._embed:
            case Unloaded(why=why):
                ended("voiceprints.enrolled", self._record, began, RuntimeError(why), voice=voice)
                self._prints[voice] = Unprinted(f"hands' voiceprint of {voice} could not be made: {why}")
            case _:
                self._enrolling.add(voice)
                self._enrolment.give(lambda: self._enrol(voice, began))

    def _recalled(self, voice: str, path: Path) -> bool:
        """`voice`'s print read back from where hands keeps it; one that cannot be read fails its unit, and is made
        again."""
        try:
            with unit("voiceprints.recalled", self._record):
                annotate(voice=voice, file=path)
                self._prints[voice] = np.load(path).astype(np.float32)
        except Exception:
            return False
        return True

    def _enrol(self, voice: str, began: Begun) -> None:
        """ENROLMENT said in `voice`, on the enrolment thread, by a Pocket TTS model loaded for it and let go of after;
        its voiceprint made on the lane."""
        said: Samples | Exception
        try:
            with loads.loading("enrolment.loaded", self._record):
                annotate(voice=voice)
                say = self._speech()
            said = say(voice, ENROLMENT)
        except Exception as error:
            said = error
        self._lane.give(lambda: self._enrolled(voice, said, began))

    def _enrolled(self, voice: str, said: Samples | Exception, began: Begun) -> None:
        """[LAW:nothing-unseen] each voiceprint made, or not, is one `voiceprints.enrolled` unit: which voice, how long
        the speech it was made from, where it is kept, and why where it failed. A print made and not kept is still
        held for the run."""
        self._enrolling.discard(voice)
        path = stored(self._kept_in, voice)
        made: Print | Unprinted
        raised: Exception | None = None
        try:
            made = self._made(said)
        except Exception as error:
            made, raised = Unprinted(f"hands' voiceprint of {voice} could not be made: {failure(error)}"), error
        else:
            try:
                kept(path, made)
            except Exception as error:
                raised = error
        seconds = None if isinstance(said, Exception) else round(len(said) / RATE, 3)
        ended("voiceprints.enrolled", self._record, began, raised, voice=voice, seconds=seconds, file=path)
        if voice in self._speaking:
            self._prints[voice] = made

    def _made(self, said: Samples | Exception) -> Print:
        match said, self._embed:
            case Exception() as error, _:
                raise error
            case _, Unloaded(why=why):
                raise RuntimeError(why)
            case samples, embed:
                return embed(samples)

    def identify(self, hold: Hold, voiced: tuple[Voiced, ...]) -> None:
        """Who each speaker in `voiced` is, as they reach a row of the table, worked out on the lane and told as one
        `voiceprints.identified` unit, which hangs off the unit open here. Returns at once."""
        began = begun()
        self._lane.give(lambda: self._identified(hold, voiced, began))

    def ended(self, hold: Hold, voiced: tuple[Voiced, ...]) -> None:
        """Who each speaker in `hold` is, at its end: its last identification, after which no speaker of it is
        remembered."""
        began = begun()

        def last() -> None:
            self._identified(hold, voiced, began)
            self._people.pop(hold.number, None)

        self._lane.give(last)

    def spanned(self, span: int, voiced: Voiced) -> None:
        """Who the one voice of echo span `span` is (`hands.voice.recorder`), hands' echo heard with no hold open,
        worked out on the lane and told as a `voiceprints.identified` unit of the span's. Returns at once."""
        began = begun()

        def identified() -> None:
            try:
                speakers = (Speaker(voiced.speaker, self._judged(voiced)),)
            except Exception as error:
                ended("voiceprints.identified", self._record, began, error, span=span)
                return
            ended("voiceprints.identified", self._record, began, None, span=span, speakers=speakers)

        self._lane.give(identified)

    def _identified(self, hold: Hold, voiced: tuple[Voiced, ...], began: Begun) -> None:
        if not voiced:
            return
        try:
            speakers = tuple(Speaker(each.speaker, self._who(hold, each)) for each in voiced)
        except Exception as error:
            ended("voiceprints.identified", self._record, began, error, hold=hold.number)
            return
        ended("voiceprints.identified", self._record, began, None, hold=hold.number, speakers=speakers)

    def _who(self, hold: Hold, voiced: Voiced) -> Who:
        """[LAW:single-enforcer] once a person in a hold, a speaker is one for the rest of it."""
        people = self._people.setdefault(hold.number, {})
        match people.get(voiced.speaker):
            case Person() as known:
                return known
            case None:
                who = self._judged(voiced)
                match who:
                    case Person():
                        people[voiced.speaker] = who
                    case _:
                        pass
                return who

    def _judged(self, voiced: Voiced) -> Who:
        match self._embed, self._against(voiced.played):
            case Unloaded(why=why), _:
                return Unidentified(voiced.seconds, why)
            case _, Unprinted(why=why):
                return Unidentified(voiced.seconds, why)
            case embed, tuple() as prints:
                return scored(embed, prints, voiced)

    def _against(self, played: tuple[str, ...]) -> tuple[Print, ...] | Unprinted:
        """Every voiceprint hands holds, so long as it holds the one of the voice it speaks in now and of each voice it
        played over the speaker: a speaker is never scored against prints that leave out a voice they may be."""
        needed = tuple(dict.fromkeys((*self._speaking[-1:], *played)))
        if not needed:
            return Unprinted("hands speaks in no voice yet")
        for voice in needed:
            match self._prints.get(voice):
                case np.ndarray():
                    pass
                case Unprinted() as missing:
                    return missing
                case None if voice in self._enrolling or voice in self._speaking:
                    return Unprinted(f"no print yet of {voice}")
                case None:
                    return Unprinted(f"hands played {voice} over them and holds no print of it")
        return tuple(voiceprint for voiceprint in self._prints.values() if isinstance(voiceprint, np.ndarray))


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
