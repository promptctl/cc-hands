"""Streaming speech recognition on NVIDIA's Nemotron Speech Streaming, in hands' own process: the words of a hold heard as
they are spoken, while it is still open; nothing here loads Pipecat.

The model is a cache-aware FastConformer-RNNT: it decodes a chunk at a time, each once the look-ahead past it has been
heard, and never takes back a word it has given, so a hold's words only grow. hands keeps one stream of it for the run and
feeds it only the holds, as the diarizer's (`hands.voice.diarization`). It is one listener on the hold owner
(`hands.voice.holdowner`), which cuts hands off on the first words it hears in a hold still open, so a turn engaged
conversation opened cuts while the user is still speaking over hands, not once they stop. Its words are never sent: it is
weak on conversational speech (about 20% WER on AMI, against 6 to 8% for the batch models measured beside it), so the
transcriber (`hands.voice.whisper`) says what each hold held.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, cast

import numpy as np
import numpy.typing as npt

from hands.sessions.audit import Record
from hands.sessions.wide import annotate, unit
from hands.threads import SerialThread
from hands.voice.holds import RATE, Fate, Heard, Hearings, Hold, Sound, decoded

# The model, fetched from the Hugging Face hub into its cache on the first load, at the revision measured for it.
MODEL = "nvidia/nemotron-speech-streaming-en-0.6b"
REVISION = "ebe59e5a817142986528bbbee5dba8db7b38ed50"

# How the model cuts the run into chunks, which `load` reads off the model and refuses to differ: a mel frame every
# MEL_HOP samples, each needing N_FFT // 2 samples past its centre; an encoder frame every SUBSAMPLING mel frames; and
# each chunk decoded once LOOKAHEAD encoder frames past it are heard. One frame of look-ahead is the model's 160 ms
# mode (mlx-audio's `transcription_delay_ms`, a chunk of CHUNK mel frames), the one the speech rework measured: 67 ms of compute per 0.5 s of audio on this Mac, and a hold's first word out
# 0.6 s into it.
MEL_HOP = 160
N_FFT = 512
SUBSAMPLING = 8
LOOKAHEAD = 1
# The mel frames of the run's first chunk, and of each after it, as mlx-audio's own streaming session cuts them.
FIRST_CHUNK = 1 + SUBSAMPLING * LOOKAHEAD
CHUNK = SUBSAMPLING * (1 + LOOKAHEAD)

# The silence fed behind a hold's last sample: two chunks, so its last word is decoded before the next hold's first
# sample is fed. Measured over 24 holds of one reader in one stream: no hold's last word was left for the next.
TRAILING = 2 * CHUNK * MEL_HOP


def chunks(samples: int) -> int:
    """How many chunks the model has decoded once the run has been fed `samples`: each is decoded in the feed that
    completes it, so a hearing on that feed finds its words, and one before it finds none."""
    mel = (samples - N_FFT // 2) // MEL_HOP + 1 if samples >= N_FFT // 2 else 0
    return 0 if mel < FIRST_CHUNK else 1 + (mel - FIRST_CHUNK) // CHUNK


type Samples = npt.NDArray[np.float32]


class Session(Protocol):
    """The model's one stream for the run: fed samples, it says what the chunks they complete decode to."""

    def decoded(self, samples: Samples) -> str: ...


def words(text: str) -> str | None:
    """The words in text the model decoded, from its first word on; None where it holds no word. A sentence's full stop
    comes out only once the model hears what follows it, so a hold's text can open with the punctuation that ends the
    last."""
    first = next((at for at, character in enumerate(text) if character.isalnum()), None)
    return None if first is None else text[first:].strip()


@dataclass
class _Held:
    """The hold open, as the stream heard it: the text decoded since it opened, how much of its words hearings have said,
    and how many samples of it were heard."""

    text: str = ""
    said: int = 0
    samples: int = 0


@dataclass(frozen=True)
class Ended:
    """A hold, once the stream heard the silence after it: every word it decoded, None where it decoded none, and how many
    seconds of it were heard."""

    words: str | None
    seconds: float


class Stream:
    """The run's streaming recognition: one session of the model for the whole run, fed only holds, as the diarizer's is.

    [LAW:one-source-of-truth] a hold's words are what the session decoded from its opening to the silence after it, so a
    word belongs to one hold, and a speaker the diarizer pairs with it keeps its number from the first hold to the last.
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
        self._held.text += self._session.decoded(samples)

    def silence(self) -> None:
        """The silence after the hold open, so its last word is decoded before the next hold opens."""
        self._held.text += self._session.decoded(np.zeros(TRAILING, np.float32))

    def said(self) -> str | None:
        """The words the hold open added since this was last asked; None where it added none."""
        held = self._held
        known = words(held.text) or ""
        added, held.said = known[held.said :], len(known)
        return words(added)

    def ended(self) -> Ended:
        """The hold open, heard to the silence after it."""
        return Ended(words(self._held.text), round(self._held.samples / RATE, 3))


@dataclass(frozen=True)
class Broken:
    """The run's streaming recognition where the model did not load, or decoding failed and left its caches in a state
    nothing vouches for: nothing hears what it is given from then on, and each hold ends failed with why.
    [LAW:one-type-per-behavior] it is given holds as a Stream is, so the listener is the same either way."""

    error: str

    def opened(self, samples: Samples) -> None:
        """A hold no model hears."""

    def heard(self, samples: Samples) -> None:
        """Samples no model hears."""

    def silence(self) -> None:
        """Silence no model hears."""

    def said(self) -> str | None:
        """No word, which the hold's end says why of."""
        return None

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
        # The model loads on the thread first, so holds spoken while it loads wait behind it, and nothing else does.
        self._thread = SerialThread("Streaming")
        self._thread.give(lambda: self._load(model))
        # The samples the run's stream has been given, the silence after each hold included: the chunks they complete are
        # where the stream's words change.
        self._fed = 0

    def _load(self, model: Callable[[], Session]) -> None:
        """The run's stream, on the model; or, where it fails to load, one that fails each hold with why.
        [LAW:nothing-unseen] the load is a unit of its own, as the transcriber's and the diarizer's are."""
        try:
            with unit("streaming.loaded", self._record):
                annotate(model=MODEL, revision=REVISION, transcription_delay_ms=CHUNK * MEL_HOP * 1000 // RATE)
                self._stream = Stream(model())
        except Exception as error:
            self._stream = Broken(f"the model did not load: {type(error).__name__}: {error}")

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
        """Move the run's stream, on the thread. [LAW:no-silent-failure] decoding that raises leaves the model's caches,
        and the grid hearings are timed on, in a state nothing vouches for: the stream is broken from there, and the hold
        it broke in and every one after it end failed with why."""
        try:
            move(self._stream)
        except Exception as error:
            self._stream = Broken(f"decoding failed: {type(error).__name__}: {error}")

    def _ended(self) -> Ended:
        """The hold open, heard through the silence after it, on the thread."""
        self._moved(lambda stream: stream.silence())
        return self._stream.ended()

    def _fed_with(self, samples: int) -> Hearings:
        """A hearing of the hold open where `samples` more complete a chunk of the run's, and none where they do not: only
        a chunk completed decodes a word."""
        completed = chunks(self._fed + samples) - chunks(self._fed)
        self._fed += samples
        match completed:
            case 0:
                return ()
            case _:
                said = self._thread.settles(lambda: self._stream.said())

                async def hear() -> Heard:
                    added = await said
                    # [LAW:nothing-unseen] the words this hearing added: the hold's whole is on its end's event.
                    annotate(added=added)
                    return Heard(added)

                return (hear,)

    def ended(self, hold: Hold, fate: Fate) -> Hearings:
        """The hold open ends, sent or dropped: every word its stream decoded, the last of them after it closed, which cut
        nothing; told so what hands heard live can be read beside what the transcriber heard of the same hold."""
        ended = self._thread.settles(self._ended)
        self._fed += TRAILING

        async def settle() -> Heard:
            annotate(fate=fate)
            # [LAW:single-enforcer] a hold the broken stream did not hear raises, and the owner fails this hearing with why.
            heard = await ended
            annotate(words=heard.words, seconds=heard.seconds)
            return Heard(heard.words)

        return (settle,)


@dataclass
class _Decoder:
    """The run's stream on mlx-audio's own streaming parts, as its NemotronStreamingSession puts them together: the log-mel
    frontend, the encoder's caches, and the greedy RNN-T decoder, each fed in turn, every chunk a feed completes decoded in
    it."""

    model: Any
    frontend: Any
    encoder: Any
    decoder: Any
    # The encoder frames decoded so far: the clock each token is stamped on.
    frame: int = 0

    def decoded(self, samples: Samples) -> str:
        from mlx import core as mx

        text = ""
        for encoded in self.encoder.push(self.frontend.push(mx.array(samples))):
            prompted = self.model.apply_prompt(encoded, self.model.default_language)
            self.encoder.materialize(prompted)
            text += "".join(token.text for token in self.decoder.decode(prompted, self.frame))
            self.frame += prompted.shape[1]
        return text


def load() -> Session:
    """The run's session of the model, the model fetched if it never was: about 10 s on this Mac from the cache, on MLX's
    default device, which is the CPU where there is no Metal.

    [LAW:parse-dont-validate] a model that cuts the run other than `chunks` says is refused here, so no hearing is timed
    on a grid the model does not decode on."""
    # mlx-audio ships no types.
    from mlx_audio.stt import load as loaded  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
    from mlx_audio.stt.models.nemotron_asr.audio import StreamingLogMelSpectrogram  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
    from mlx_audio.stt.models.nemotron_asr.rnnt import GreedyDecoderState  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
    from mlx_audio.stt.models.nemotron_asr.streaming import ConformerStreamingState  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]

    model = cast(Any, loaded(MODEL, revision=REVISION))
    cut = (model.preprocessor_config.sample_rate, model.preprocessor_config.hop_length, model.preprocessor_config.n_fft, model.encoder_config.subsampling_factor)
    if cut != (RATE, MEL_HOP, N_FFT, SUBSAMPLING):
        raise ValueError(f"{MODEL} cuts the run as (rate, mel hop, FFT, subsampling) {cut}; hands times its hearings on {(RATE, MEL_HOP, N_FFT, SUBSAMPLING)}")
    encoder = cast(Any, ConformerStreamingState(model.encoder, att_context_size=[model.default_att_context_size[0], LOOKAHEAD], first_chunk_mel=FIRST_CHUNK))
    if encoder.chunk_mel != CHUNK:
        raise ValueError(f"{MODEL}'s encoder takes chunks of {encoder.chunk_mel} mel frames; hands times its hearings on {CHUNK}")
    return _Decoder(model, StreamingLogMelSpectrogram(model.preprocessor_config), encoder, GreedyDecoderState(model))
