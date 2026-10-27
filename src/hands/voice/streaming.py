"""Streaming speech recognition with diarization, on NVIDIA's Nemotron Speech Streaming paired with Nemotron 3
Diarization, in hands' own process: the words of a hold heard as they are spoken, while it is still open, each with the
speaker who said it; nothing here loads Pipecat.

The pair is mlx-audio's speaker streaming session. The diarizer scores up to eight speakers in every 10 ms frame, its
speakers numbered by when each was first heard; the session masks each speaker's log-mel frames by those decisions and
runs one cache-aware FastConformer-RNNT stream per speaker, so a word comes out of exactly one speaker's stream, and never
taken back. hands keeps one session for the run and feeds it only the holds, so a speaker keeps its number from the first
hold to the last: a new session, or the session's own end, numbers everyone from 0 again (promptctl-voice-i15t.99s.gaq).

It is one listener on the hold owner (`hands.voice.holdowner`), which cuts hands off on the first words it hears in a hold
still open, so a turn engaged conversation opened cuts while the user is still speaking over hands, not once they stop.
Its words are never sent: the model is weak on conversational speech (about 20% WER on AMI, against 6 to 8% for the
batch models measured beside it), so the transcriber (`hands.voice.whisper`) says what each hold held.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

import numpy as np
import numpy.typing as npt

from hands.sessions.audit import Record
from hands.sessions.wide import annotate, unit
from hands.threads import SerialThread
from hands.voice.holds import RATE, Fate, Heard, Hearings, Hold, Sound, decoded

# The models, fetched from the Hugging Face hub into its cache on the first load, at the revisions measured for them.
MODEL = "nvidia/nemotron-speech-streaming-en-0.6b"
REVISION = "ebe59e5a817142986528bbbee5dba8db7b38ed50"
DIARIZATION = "mlx-community/Nemotron-3-Diarization"
DIARIZATION_REVISION = "59ed2dbfc1346dcea9d423c71306a3a2499c568f"

# How the pair cuts the run, which `load` reads off the models and refuses to differ: a mel frame every MEL_HOP samples,
# each needing N_FFT // 2 samples past its centre, and an encoder frame every SUBSAMPLING mel frames.
MEL_HOP = 160
N_FFT = 512
SUBSAMPLING = 8
# The recognizer's 160 ms mode, one encoder frame of look-ahead: a chunk of CHUNK mel frames.
LOOKAHEAD = 1
CHUNK = SUBSAMPLING * (1 + LOOKAHEAD)
# The diarizer's NVIDIA preset, its step committing STEP mel frames once it has heard AHEAD past them: 0.24 s, looking
# 0.08 s ahead. Measured on two speakers (AMI), the run started at five phases of the grid: every word given the speaker
# the transformers path gave it at its 1.04 s preset, where the 0.64 s preset missed two in three of the five; and a
# hold's first words 0.17 to 0.4 s after they are said, where the 1.04 s preset's took up to 1.8 s.
PRESET = "ultra_low"
STEP = 3 * SUBSAMPLING
AHEAD = 1 * SUBSAMPLING
# [LAW:one-source-of-truth] how many chunks a speaker's stream is still stepped after the last chunk they were heard in,
# which is also how long after a hold the session is fed silence: every speaker of the hold is gated off, its last words
# decoded, before the next hold's first sample. mlx-audio's default of 2 froze a speaker before their last word came out,
# and it surfaced in their next hold.
GATING = 4


def steps(samples: int) -> int:
    """How many steps the diarizer has committed once the run has been fed `samples`. The session decodes a chunk only
    once the diarizer has committed its frames, and every chunk they complete in the feed that commits them, so a
    hearing on that feed finds its words, and one before it finds none."""
    first = (STEP + AHEAD - 1) * MEL_HOP + N_FFT // 2
    return 0 if samples < first else 1 + (samples - first) // (STEP * MEL_HOP)


def through(samples: int) -> int:
    """How far the run is fed, silence after a hold that ended at `samples`: until the diarizer has committed GATING
    chunks past the chunk holding the hold's last sample, which the step committing them waits for."""
    last = (samples - 1 + N_FFT // 2) // MEL_HOP
    committed = (last // CHUNK + 1 + GATING) * CHUNK
    step = -(-committed // STEP)
    return max(samples, (STEP * step + AHEAD - 1) * MEL_HOP + N_FFT // 2)


type Samples = npt.NDArray[np.float32]


@dataclass(frozen=True)
class Said:
    """Text one speaker's stream decoded: the speaker, by when the run first heard them, 0 the first."""

    speaker: int
    text: str


# What a feed decoded, speaker by speaker, in the order decoded.
type Spoken = tuple[Said, ...]


class Session(Protocol):
    """The pair's one session for the run: fed samples, it says what each speaker's stream decoded in the chunks they
    complete."""

    def decoded(self, samples: Samples) -> Spoken: ...


def words(text: str) -> str | None:
    """The words in text the model decoded, from its first word on; None where it holds no word. A sentence's full stop
    comes out only once the model hears what follows it, so a hold's text can open with the punctuation that ends the
    last."""
    first = next((at for at, character in enumerate(text) if character.isalnum()), None)
    return None if first is None else text[first:].strip()


def joined(spoken: Spoken) -> str | None:
    """Every speaker's words, as one say: None where nobody said any."""
    return " ".join(said.text for said in spoken) or None


@dataclass
class _Held:
    """The hold open, as the stream heard it: each speaker's text decoded since it opened, in the order they were first
    heard in it, how much of each speaker's words hearings have said, and how many samples of it were heard."""

    text: dict[int, str] = field(default_factory=dict[int, str])
    said: dict[int, int] = field(default_factory=dict[int, int])
    samples: int = 0


@dataclass(frozen=True)
class Ended:
    """A hold, once the stream heard the silence after it: every word each speaker said in it, and how many seconds of it
    were heard."""

    words: Spoken
    seconds: float


class Stream:
    """The run's streaming recognition: one session of the pair for the whole run, fed only holds.

    [LAW:one-source-of-truth] a hold's words are what the session decoded from its opening to the silence after it, so a
    word belongs to one hold and one speaker, and a speaker keeps its number from the first hold to the last.
    """

    def __init__(self, session: Session) -> None:
        self._session = session
        self._held = _Held()

    def opened(self, samples: Samples) -> None:
        """A hold opens, with `samples`, what it kept from before it opened: its words start with what they decode to."""
        self._held = _Held()
        self.heard(samples)

    def heard(self, samples: Samples) -> None:
        """More of the hold open."""
        self._held.samples += len(samples)
        self._decoded(samples)

    def silence(self, samples: int) -> None:
        """`samples` of silence after the hold open, so every word in it is decoded before the next hold opens."""
        self._decoded(np.zeros(samples, np.float32))

    def _decoded(self, samples: Samples) -> None:
        for said in self._session.decoded(samples):
            self._held.text[said.speaker] = self._held.text.get(said.speaker, "") + said.text

    def said(self) -> Spoken:
        """The words each speaker added to the hold open since this was last asked."""
        held, added = self._held, list[Said]()
        for speaker, text in held.text.items():
            known = words(text) or ""
            new, held.said[speaker] = words(known[held.said.get(speaker, 0) :]), len(known)
            added += [Said(speaker, new)] if new else []
        return tuple(added)

    def ended(self) -> Ended:
        """The hold open, heard to the silence after it."""
        spoken = tuple(Said(speaker, said) for speaker, text in self._held.text.items() if (said := words(text)))
        return Ended(spoken, round(self._held.samples / RATE, 3))


@dataclass(frozen=True)
class Broken:
    """The run's streaming recognition where the models did not load, or decoding failed and left their caches in a state
    nothing vouches for: nothing hears what it is given from then on, and each hold ends failed with why.
    [LAW:one-type-per-behavior] it is given holds as a Stream is, so the listener is the same either way."""

    error: str

    def opened(self, samples: Samples) -> None:
        """A hold no model hears."""

    def heard(self, samples: Samples) -> None:
        """Samples no model hears."""

    def silence(self, samples: int) -> None:
        """Silence no model hears."""

    def said(self) -> Spoken:
        """No word, which the hold's end says why of."""
        return ()

    def ended(self) -> Ended:
        raise RuntimeError(self.error)


class StreamingRecognizer:
    """The run's stream on a thread of its own: a listener on the hold owner, given each hold's audio as it is spoken and
    told when it ends, which nothing the owner does waits on.

    [LAW:no-ambient-temporal-coupling] what moves the stream is queued on the thread inside the call that brings it, so a
    hold's end lands behind its last audio and ahead of the next hold's first; a hearing only waits on what the stream says
    at its place in that line.
    """

    name = "streaming"

    # The run's stream, set by the thread's first work and read only by the work queued behind it, on the same thread.
    _stream: Stream | Broken

    def __init__(self, model: Callable[[], Session], record: Record) -> None:
        self._record = record
        # The models load on the thread first, so holds spoken while they load wait behind them, and nothing else does.
        self._thread = SerialThread("Streaming")
        self._thread.give(lambda: self._load(model))
        # The samples the run's stream has been given, the silence after each hold included: the diarizer's steps they
        # complete are where the stream's words change.
        self._fed = 0

    def _load(self, model: Callable[[], Session]) -> None:
        """The run's stream, on the pair; or, where it fails to load, one that fails each hold with why.
        [LAW:nothing-unseen] the load is a unit of its own, as the transcriber's is."""
        try:
            with unit("streaming.loaded", self._record):
                annotate(
                    model=MODEL,
                    revision=REVISION,
                    transcription_delay_ms=CHUNK * MEL_HOP * 1000 // RATE,
                    diarization=DIARIZATION,
                    diarization_revision=DIARIZATION_REVISION,
                    preset=PRESET,
                    gating=GATING,
                )
                self._stream = Stream(model())
        except Exception as error:
            self._stream = Broken(f"the models did not load: {type(error).__name__}: {error}")

    def opened(self, hold: Hold, before: Sound) -> Hearings:
        """`hold` opens: what it kept from before it opened is the start of its words."""
        samples = decoded(before.heard)
        self._thread.give(lambda: self._moved(lambda stream: stream.opened(samples)))
        return self._fed_with(len(samples))

    def heard(self, sound: Sound) -> Hearings:
        """More of the hold open, decoded on the thread behind everything given before it."""
        samples = decoded(sound.heard)
        self._thread.give(lambda: self._moved(lambda stream: stream.heard(samples)))
        return self._fed_with(len(samples))

    def _moved(self, move: Callable[[Stream | Broken], None]) -> None:
        """Move the run's stream, on the thread. [LAW:no-silent-failure] decoding that raises leaves the models' caches,
        and the grid hearings are timed on, in a state nothing vouches for: the stream is broken from there, and the hold
        it broke in and every one after it end failed with why."""
        try:
            move(self._stream)
        except Exception as error:
            self._stream = Broken(f"decoding failed: {type(error).__name__}: {error}")

    def _ended(self, silence: int) -> Ended:
        """The hold open, heard through the silence after it, on the thread."""
        self._moved(lambda stream: stream.silence(silence))
        return self._stream.ended()

    def _fed_with(self, samples: int) -> Hearings:
        """A hearing of the hold open where `samples` more complete a step of the diarizer's, and none where they do not:
        only a step committed decodes a word."""
        completed = steps(self._fed + samples) - steps(self._fed)
        self._fed += samples
        match completed:
            case 0:
                return ()
            case _:
                said = self._thread.settles(lambda: self._stream.said())

                async def hear() -> Heard:
                    added = await said
                    # [LAW:nothing-unseen] the words each speaker added in this hearing: the hold's whole is on its end's.
                    annotate(added=added)
                    return Heard(joined(added))

                return (hear,)

    def ended(self, hold: Hold, fate: Fate) -> Hearings:
        """The hold open ends, sent or dropped: every word each speaker said in it, the last of them after it closed,
        which cut nothing; told so what hands heard live can be read beside what the transcriber heard of the same hold."""
        silence = through(self._fed) - self._fed
        ended = self._thread.settles(lambda: self._ended(silence))
        self._fed += silence

        async def settle() -> Heard:
            annotate(fate=fate)
            # [LAW:single-enforcer] a hold the broken stream did not hear raises, and the owner fails this hearing with why.
            heard = await ended
            annotate(words=heard.words, seconds=heard.seconds)
            return Heard(joined(heard.words))

        return (settle,)


def speaker(label: str) -> int:
    """[LAW:parse-dont-validate] the session's `speaker_<n>`, as the number it is."""
    match label.split("_"):
        case ["speaker", number] if number.isdigit():
            return int(number)
        case _:
            raise ValueError(f"the session named a speaker {label!r}, not speaker_<n>")


@dataclass
class _Paired:
    """The run's session of the pair, mlx-audio's own: fed any partition of the run, it decodes every chunk the diarizer
    has committed."""

    session: Any

    def decoded(self, samples: Samples) -> Spoken:
        return tuple(Said(speaker(delta.speaker), delta.text) for delta in self.session.feed(samples))


def diarizer() -> Any:
    """Nemotron 3 Diarization at PRESET, fetched if it never was.

    [LAW:parse-dont-validate] a diarizer that cuts the run other than `steps` says is refused here, so no hearing is timed
    on a grid it does not commit on."""
    # mlx-audio ships no types.
    from mlx_audio.vad import load as loaded  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]

    model = cast(Any, loaded(DIARIZATION, revision=DIARIZATION_REVISION))
    model.set_streaming_config(PRESET)
    modules, processor, factor = model.config.modules_config, model.config.processor_config, model.config.encoder_config.subsampling_factor
    cut = (processor.sampling_rate, processor.hop_length, processor.n_fft, modules.chunk_len * factor, modules.chunk_right_context * factor)
    if cut != (RATE, MEL_HOP, N_FFT, STEP, AHEAD):
        raise ValueError(f"{DIARIZATION} at {PRESET} cuts the run as (rate, mel hop, FFT, step, look-ahead) {cut}; hands times its hearings on {(RATE, MEL_HOP, N_FFT, STEP, AHEAD)}")
    return model


def load() -> Session:
    """The run's session of the pair, the models fetched if they never were: a few seconds on this Mac from the cache, on
    MLX's default device, which is the CPU where there is no Metal."""
    from mlx_audio.stt import load as loaded  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]

    model = cast(Any, loaded(MODEL, revision=REVISION))
    session = model.create_speaker_streaming_session(
        diarizer(), att_context_size=[model.default_att_context_size[0], LOOKAHEAD], cache_gating_buffer_size=GATING
    )
    # [LAW:parse-dont-validate] the session checks the two models hear the same rate and hop; this, the chunk GATING
    # counts in.
    if session.chunk_mel != CHUNK:
        raise ValueError(f"{MODEL}'s session takes chunks of {session.chunk_mel} mel frames; hands gates speakers in chunks of {CHUNK}")
    return _Paired(session)
