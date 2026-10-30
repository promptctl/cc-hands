"""Streaming speech recognition with diarization, on NVIDIA's Nemotron Speech Streaming paired with Nemotron 3
Diarization, in hands' own process: the words of a hold heard as they are spoken, while it is still open, each with the
speaker who said it; nothing here loads Pipecat.

The pair is mlx-audio's speaker streaming session. The diarizer scores up to eight speakers in every 10 ms frame, its
speakers numbered by when each was first heard; the session masks each speaker's log-mel frames by those decisions and
runs one cache-aware FastConformer-RNNT stream per speaker, so a word comes out of exactly one speaker's stream, and never
taken back. hands keeps one session for the run and feeds it only the holds, so a speaker keeps its number from the first
hold to the last: a new session, or the session's own end, numbers everyone from 0 again (promptctl-voice-i15t.99s.gaq).

Each hearing hands each speaker's voice in the hold, the frames the diarizer heard them in, to be identified against
hands' own (`hands.voice.voiceprints`), after its words have gone out: who they were decides nothing yet.

It is one listener on the hold owner (`hands.voice.holdowner`), which cuts hands off on the first words it hears in a hold
still open, so a turn engaged conversation opened cuts while the user is still speaking over hands, not once they stop.
Its words are never sent: the model is weak on conversational speech (about 20% WER on AMI, against 6 to 8% for the
batch models measured beside it), so the transcriber (`hands.voice.whisper`) says what each hold held.
"""

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

import numpy as np
from loguru import logger

from hands.sessions.audit import Record
from hands.sessions.wide import annotate
from hands.voice import loads
from hands.threads import SerialThread
from hands.voice.holds import RATE, Fate, Heard, Hearings, Hold, Samples, Sound, decoded
from hands.voice.voiceprints import WINDOW, Voiced

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
# How many chunks a speaker's stream is stepped from the last chunk the diarizer heard them in, that one included: the
# GATING - 1 after it are their log-mel frames masked to zero, which the model does not hear as silence, so it decides
# how a hold's last words come out. It does not decide whether they stay in the hold: the hold's end closes every
# stream (`Session.closed`). 1 steps a stream only where its speaker is heard. Measured by scripts/measure-gating.py on
# 578 AMI turns with known words that end in silence, held and in engaged conversation, against 2 and 4: 1 ends the most
# turns on their last word and invents the fewest words past it, in both shapes, and gives as few to another speaker as
# 2, within one. What it costs: a chunk the diarizer misses its speaker in is one their stream never hears, and one
# LibriVox reader came out 37 words short of the 12851 heard at 4 (docs/features.md).
GATING = 1
# The diarizer's decisions, as the session thresholds them: one for each encoder frame, every SUBSAMPLING mel frames.
FRAME = SUBSAMPLING * MEL_HOP


def steps(samples: int) -> int:
    """How many steps the diarizer has committed once the run has been fed `samples`. The session decodes a chunk only
    once the diarizer has committed its frames, and every chunk they complete in the feed that commits them, so a
    hearing on that feed finds its words, and one before it finds none."""
    first = (STEP + AHEAD - 1) * MEL_HOP + N_FFT // 2
    return 0 if samples < first else 1 + (samples - first) // (STEP * MEL_HOP)


def through(samples: int) -> int:
    """How far the run is fed, silence after a hold that ended at `samples`: until the diarizer has committed the chunk
    holding the hold's last sample, which the step committing it waits for, so the session decodes it before the hold
    closes. Silence fed past it is heard as nobody: it reaches a stream only as the GATING - 1 masked chunks after its
    speaker's last, and otherwise only puts the close off."""
    last = (samples - 1 + N_FFT // 2) // MEL_HOP
    committed = (last // CHUNK + 1) * CHUNK
    step = -(-committed // STEP)
    return max(samples, (STEP * step + AHEAD - 1) * MEL_HOP + N_FFT // 2)


@dataclass(frozen=True)
class Said:
    """Text one speaker's stream decoded: the speaker, by when the run first heard them, 0 the first."""

    speaker: int
    text: str


# Words by speaker: a feed's in the order the session decoded them, a hold's each speaker's together, in the order they
# were first heard in it.
type Spoken = tuple[Said, ...]


@dataclass(frozen=True)
class Closed:
    """A hold's close: the speakers whose streams it closed, and the last words they decoded. A speaker can be closed
    with no word, as one the diarizer marked only for a moment is."""

    speakers: frozenset[int]
    words: Spoken


@dataclass(frozen=True)
class Active:
    """The diarizer marked `speaker` in the run's encoder frame `frame`, the FRAME samples from frame * FRAME: in the
    session's own threshold, the one its streams are masked by, so where a speaker is active is where their words are."""

    speaker: int
    frame: int


@dataclass(frozen=True)
class Decoded:
    """What a feed decoded: the text each speaker's stream added, and every frame the diarizer committed in it, by each
    speaker active in it, two in one frame where they overlap."""

    words: Spoken
    active: tuple[Active, ...]


class Session(Protocol):
    """The pair's one session for the run: fed samples, it says what each speaker's stream decoded in the chunks they
    complete, and where the diarizer heard each speaker; closed, it decodes every stream's last words and ends them, so
    none comes out in the next hold."""

    def decoded(self, samples: Samples) -> Decoded: ...

    def closed(self) -> Closed: ...


def words(text: str) -> str | None:
    """The words in text the model decoded, from its first word on; None where it holds no word. A sentence's full stop
    comes out only once the model hears what follows it, so a hold's text can open with the punctuation that ends the
    last."""
    first = next((at for at, character in enumerate(text) if character.isalnum()), None)
    return None if first is None else text[first:].strip()


def joined(texts: Iterable[str | None]) -> str | None:
    """Every speaker's words, as one say: None where nobody said any."""
    return " ".join(text for text in texts if text) or None


class _Tape:
    """The hold open's samples, from where the run was when it opened: grown in place, so a frame of it is a slice."""

    def __init__(self, start: int) -> None:
        self.start = start
        self._samples = np.zeros(RATE, np.float32)
        self.length = 0

    def add(self, samples: Samples) -> None:
        wanted = self.length + len(samples)
        if wanted > len(self._samples):
            # Doubled as it fills: a hold of any length is copied a few times, not at every frame.
            grown = np.zeros(max(wanted, 2 * len(self._samples)), np.float32)
            grown[: self.length] = self._samples[: self.length]
            self._samples = grown
        self._samples[self.length : wanted] = samples
        self.length = wanted

    def frame(self, frame: int) -> Samples:
        """The samples of the run's encoder frame `frame` that are the hold's: none for a frame before it opened or in
        the silence after it."""
        start = min(max(frame * FRAME - self.start, 0), self.length)
        return self._samples[start : min(max((frame + 1) * FRAME - self.start, 0), self.length)]


@dataclass
class _Voice:
    """One speaker of the hold open: the latest WINDOW of where the diarizer heard them, overlap included, and how many
    seconds of them in all."""

    latest: Samples
    seconds: float


def _unheard() -> _Voice:
    """A speaker before the diarizer has marked them in any of the hold's own frames."""
    return _Voice(np.zeros(0, np.float32), 0.0)


@dataclass
class _Held:
    """The hold open, as the stream heard it: each speaker's text decoded since it opened, in the order they were first
    heard in it, how much of each speaker's words hearings have said, its samples, each speaker's voice in it, and its
    close, which closes no stream until the silence after it is heard."""

    tape: _Tape
    text: dict[int, str] = field(default_factory=dict[int, str])
    said: dict[int, int] = field(default_factory=dict[int, int])
    voices: dict[int, _Voice] = field(default_factory=dict[int, _Voice])
    closed: Closed = Closed(frozenset(), ())


@dataclass(frozen=True)
class Ended:
    """A hold, once the stream heard the silence after it: every word each speaker said in it, how many seconds of it
    were heard, its close, whose words are among them, and every speaker heard in it, with their voice in it."""

    words: Spoken
    seconds: float
    closed: Closed
    voices: tuple[Voiced, ...]


class Stream:
    """The run's streaming recognition: one session of the pair for the whole run, fed only holds, and each speaker's
    voice in each hold.

    [LAW:one-source-of-truth] a hold's words are what the session decoded from its opening to the silence after it, so a
    word belongs to one hold and one speaker, and a speaker keeps its number from the first hold to the last. A speaker's
    voice is where the session's own diarizer heard them, so their words and the voice they are identified by are the
    same frames.
    """

    def __init__(self, session: Session) -> None:
        self._session = session
        self._held = _Held(_Tape(0))

    def opened(self, samples: Samples, start: int) -> None:
        """A hold opens, `start` samples into the run, with `samples`, what it kept from before it opened: its words start
        with what they decode to."""
        self._held = _Held(_Tape(start))
        self.heard(samples)

    def heard(self, samples: Samples) -> None:
        """More of the hold open."""
        self._held.tape.add(samples)
        self._decoded(samples)

    def silence(self, samples: int) -> None:
        """`samples` of silence after the hold open, and its close, so every word in it is decoded before the next hold
        opens."""
        self._decoded(np.zeros(samples, np.float32))
        self._held.closed = self._session.closed()
        self._added(self._held.closed.words)

    def _decoded(self, samples: Samples) -> None:
        decoded = self._session.decoded(samples)
        self._added(decoded.words)
        for active in decoded.active:
            self._voiced(active)

    def _added(self, spoken: Spoken) -> None:
        for said in spoken:
            self._held.text[said.speaker] = self._held.text.get(said.speaker, "") + said.text

    def _voiced(self, active: Active) -> None:
        """A frame more of a speaker's voice, where it falls in the hold: none of it before the hold or after it."""
        heard = self._held.tape.frame(active.frame)
        voice = self._held.voices.setdefault(active.speaker, _unheard())
        voice.latest = np.concatenate([voice.latest, heard])[-round(WINDOW * RATE) :]
        voice.seconds += len(heard) / RATE

    def _voice(self, speaker: int, words: str | None) -> Voiced:
        voice = self._held.voices.get(speaker, _unheard())
        return Voiced(speaker, words, voice.latest, voice.seconds)

    def said(self) -> tuple[Voiced, ...]:
        """The words each speaker added to the hold open since this was last asked, each with their voice in it so far."""
        held, added = self._held, list[Voiced]()
        for speaker, text in held.text.items():
            known = words(text) or ""
            new, held.said[speaker] = words(known[held.said.get(speaker, 0) :]), len(known)
            added += [self._voice(speaker, new)] if new else []
        return tuple(added)

    def ended(self) -> Ended:
        """The hold open, heard to the silence after it."""
        held = self._held
        spoken = tuple(Said(speaker, said) for speaker, text in held.text.items() if (said := words(text)))
        voices = tuple(self._voice(speaker, words(held.text.get(speaker, ""))) for speaker in {**held.text, **held.voices})
        return Ended(spoken, round(held.tape.length / RATE, 3), held.closed, voices)


@dataclass(frozen=True)
class Broken:
    """The run's streaming recognition where the models did not load, or decoding failed and left their caches in a state
    nothing vouches for: nothing hears what it is given from then on, and each hold ends failed with why.
    [LAW:one-type-per-behavior] it is given holds as a Stream is, so the listener is the same either way."""

    error: str

    def opened(self, samples: Samples, start: int) -> None:
        """A hold no model hears."""

    def heard(self, samples: Samples) -> None:
        """Samples no model hears."""

    def silence(self, samples: int) -> None:
        """Silence no model hears."""

    def said(self) -> tuple[Voiced, ...]:
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

    def __init__(self, model: Callable[[int], Session], identify: Callable[[Hold, tuple[Voiced, ...]], None], record: Record) -> None:
        self._record = record
        self._identify = identify
        # The models load on the thread first, so holds spoken while they load wait behind them, and nothing else does.
        self._thread = SerialThread("Streaming")
        self._thread.give(lambda: self._load(model))
        # [LAW:one-source-of-truth] the samples the run's stream has been given, the silence after each hold included:
        # the diarizer's steps they complete are where the stream's words change, and where a frame falls in a hold.
        self._fed = 0
        # The hold open, or the last one, as the owner told it.
        self._hold = Hold(0, "held key")

    def _load(self, model: Callable[[int], Session]) -> None:
        """The run's stream, on the pair made at GATING; or, where it fails to load, one that fails each hold with why.
        [LAW:nothing-unseen] the load is a unit of its own, as the transcriber's is."""
        try:
            with loads.loading("streaming.loaded", self._record):
                annotate(
                    model=MODEL,
                    revision=REVISION,
                    transcription_delay_ms=CHUNK * MEL_HOP * 1000 // RATE,
                    diarization=DIARIZATION,
                    diarization_revision=DIARIZATION_REVISION,
                    preset=PRESET,
                    gating=GATING,
                )
                # [LAW:one-source-of-truth] the gating the event says is the one the session is made with.
                self._stream = Stream(model(GATING))
        except Exception as error:
            self._stream = Broken(f"the models did not load: {type(error).__name__}: {error}")

    def opened(self, hold: Hold, before: Sound) -> Hearings:
        """`hold` opens: what it kept from before it opened is the start of its words."""
        samples, start = decoded(before.heard), self._fed
        self._hold = hold
        self._thread.give(lambda: self._moved(lambda stream: stream.opened(samples, start)))
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
            # Each hold from here fails saying why; where in the pair it broke is said once, here.
            logger.opt(exception=error).error("the streaming session's decoding broke; no hold is heard live from here")
            self._stream = Broken(f"decoding failed: {type(error).__name__}: {error}")

    def _ended(self, silence: int) -> Ended:
        """The hold open, heard through the silence after it, on the thread."""
        self._moved(lambda stream: stream.silence(silence))
        return self._stream.ended()

    def _identified(self, hold: Hold, voiced: tuple[Voiced, ...]) -> None:
        """Hand the speakers a hearing found on to be identified; a hearing that found none hands on nothing."""
        match voiced:
            case ():
                pass
            case _:
                self._identify(hold, voiced)

    def _fed_with(self, samples: int) -> Hearings:
        """A hearing of the hold open where `samples` more complete a step of the diarizer's, and none where they do not:
        only a step committed decodes a word."""
        completed = steps(self._fed + samples) - steps(self._fed)
        self._fed += samples
        match completed:
            case 0:
                return ()
            case _:
                said, hold = self._thread.settles(lambda: self._stream.said()), self._hold

                async def hear() -> Heard:
                    added = await said
                    # [LAW:nothing-unseen] the words each speaker added in this hearing: the hold's whole is on its end's.
                    annotate(added=tuple(Said(voiced.speaker, voiced.words) for voiced in added if voiced.words))
                    # Who said them is worked out on the voiceprints' own lane: the words go out without waiting on it.
                    self._identified(hold, added)
                    return Heard(joined(voiced.words for voiced in added))

                return (hear,)

    def ended(self, hold: Hold, fate: Fate, sound: Sound) -> Hearings:
        """The hold open ends, sent or dropped: every word each speaker said in it, the last of them after it closed,
        which cut nothing; told so what hands heard live can be read beside what the transcriber heard of the same hold."""
        silence = through(self._fed) - self._fed
        ended = self._thread.settles(lambda: self._ended(silence))
        self._fed += silence

        async def settle() -> Heard:
            # [LAW:nothing-unseen] the silence after a hold differs with where on the grid it ended, and costs decode.
            annotate(fate=fate, silence=round(silence / RATE, 3))
            # [LAW:single-enforcer] a hold the broken stream did not hear raises, and the owner fails this hearing with why.
            heard = await ended
            # [LAW:nothing-unseen] the close's words are among the hold's, told apart: what the floor decoded, not the feed.
            annotate(words=heard.words, seconds=heard.seconds, closed=heard.closed.speakers, closing=heard.closed.words)
            self._identified(hold, heard.voices)
            return Heard(joined(said.text for said in heard.words))

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
    has committed, and says where the diarizer heard each speaker in it."""

    session: Any
    # The frames the diarizer marked a speaker in, since the last feed returned.
    _active: list[Active] = field(default_factory=list[Active])

    def __post_init__(self) -> None:
        """Hear each chunk's speakers as the session takes it.

        [LAW:locality-or-seam] exception: mlx-audio thresholds the diarizer's probabilities as it masks each chunk and
        keeps the decision nowhere, so this wraps the session's own step that takes the chunks, here alone, on the version
        pyproject pins exactly, and thresholds the same probabilities with the session's own function, factor and
        threshold, so the frames a speaker is active in are those their stream heard."""
        import mlx.core as mx  # pyright: ignore[reportMissingTypeStubs]
        from mlx_audio.stt.models.nemotron_asr.speaker_streaming import speaker_activity  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]

        session = self.session
        push = session._push_features

        def pushed(mel: Any, probs: Any, *, final: bool = False) -> Any:
            first, pending = session._mel_offset, session._probs
            updates = push(mel, probs, final=final)
            taken = session._mel_offset - first
            match taken:
                case 0:
                    # Most feeds complete no chunk: nothing to threshold.
                    return updates
                case _:
                    heard = mx.concatenate([pending, probs], axis=0)[:taken]
                    frames, speakers = np.nonzero(np.array(speaker_activity(heard, session.factor, session.threshold)))
                    self._active += [Active(int(each), first // session.factor + int(frame)) for frame, each in zip(frames, speakers, strict=True)]
                    return updates

        session._push_features = pushed

    def decoded(self, samples: Samples) -> Decoded:
        words = tuple(Said(speaker(delta.speaker), delta.text) for delta in self.session.feed(samples))
        active, self._active = tuple(self._active), []
        return Decoded(words, active)

    def closed(self) -> Closed:
        """Every speaker's stream given a chunk of silence as its last and ended, its last token decoded, and the gate
        history emptied, so a stream is made again only for a speaker the diarizer marks in a later hold; the diarizer is
        left as it is, so speakers keep their numbers.

        [LAW:locality-or-seam] exception: mlx-audio closes only the whole session (`feed(final=True)`, the diarizer
        with it), so this reaches its per-speaker state, here alone, on the version pyproject pins exactly;
        promptctl-voice-i15t.qg8.35b proposes the close upstream."""
        import mlx.core as mx  # pyright: ignore[reportMissingTypeStubs]

        session, last = self.session, list[Said]()
        speakers, config = frozenset[int](session._encoders), session.model.preprocessor_config
        # [LAW:one-source-of-truth] silence as the model's frontend hears it, the log of zero power plus its guard (the
        # chunked frontend takes only unnormalized models): one chunk, the one frame of right context a stream needs past
        # its speaker's last chunk. More invents words: on AMI's fifth hold at GATING 4, two chunks gave its second
        # speaker "Wall.", four "Walk.", where one gives them none.
        floor = mx.full((1, session.chunk_mel, config.features), math.log(config.log_zero_guard_value))
        for number in sorted(speakers):
            encoder, decoder, start, text = session._encoders.pop(number), session._decoders.pop(number), session._mel_offset // session.factor, ""
            for encoded in encoder.push(floor, final=True):
                prompted = session.model.apply_prompt(encoded, session.language)
                encoder.materialize(prompted)
                text += "".join(token.text for token in decoder.decode(prompted, start))
                start += prompted.shape[1]
            last += [Said(number, text)] if text else []
        session._history.clear()
        return Closed(speakers, tuple(last))


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


def load(gating: int) -> Session:
    """The run's session of the pair, each speaker's stream stepped `gating` chunks from the last the diarizer heard them
    in, the models fetched if they never were: a few seconds on this Mac from the cache, on MLX's default device, which
    is the CPU where there is no Metal."""
    from mlx_audio.stt import load as loaded  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]

    model = cast(Any, loaded(MODEL, revision=REVISION))
    # [LAW:parse-dont-validate] the session checks the two models hear the same rate and hop, and `diarizer` that they
    # are hands'; this, the recognizer's own FFT, which places a hold's last mel frame for `through`, and the chunk
    # GATING counts in.
    if model.preprocessor_config.n_fft != N_FFT:
        raise ValueError(f"{MODEL} frames the run with an FFT of {model.preprocessor_config.n_fft}; hands places a hold's last mel frame on {N_FFT}")
    session = model.create_speaker_streaming_session(
        diarizer(), att_context_size=[model.default_att_context_size[0], LOOKAHEAD], cache_gating_buffer_size=gating
    )
    if session.chunk_mel != CHUNK:
        raise ValueError(f"{MODEL}'s session takes chunks of {session.chunk_mel} mel frames; hands gates speakers in chunks of {CHUNK}")
    return _Paired(session)
