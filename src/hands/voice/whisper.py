"""Whisper, the transcriber and one listener on the hold owner (`hands.voice.holdowner`): it transcribes each hold the key
sent, whole, as it ends (hands.voice.transcription), and hears a hold still open where its cut waits on words.

A hold whose cut waits on words (`hands.voice.trigger.turn_start`) is heard while it is still open, every
`OVERHEARING_SECONDS` of its audio, until Whisper hears words in it: the hold owner cuts hands off on those, so hands is
cut off while the user speaks over it rather than once they stop, which in engaged conversation is only after
end-of-turn detection closes the hold. The same filters judge what counts as words, so hands' echo or a cough still cuts
nothing. Each hearing hears the audio since the one before it was returned, two hops of it: a word straddling one
hearing's end is whole in the next, and a hearing costs what the model takes to hear a hop, however long the hold has
been open."""

import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal

from hands.sessions.audit import HoldHeard, Record, Unsaid
from hands.sessions.wide import annotate, unit
from hands.threads import SerialThread
from hands.voice import transcription
from hands.voice.holds import SECOND, Fate, Heard, Hearings, Hold, Sound
from hands.voice.trigger import turn_start

# Whisper's own compression_ratio_threshold: above it, the decoder was repeating itself, as it does over noise primed
# with a vocabulary, sure of every word ("and turnstop, and turnstop, ..." read 17.1 to 23.8; speech no more than 0.92,
# hands-dictation-2bs.1xq).
_REPEATING = 2.4

# The average log probability below which a segment is Whisper guessing. Primed noise came back as "and slow-talking."
# at -2.8, which the compression ratio (0.68) does not catch; "okay" said quietly at -25 dBFS read
# -0.89 to -1.11 (hands-dictation-d7i).
_GUESSED = -1.5

# How much more audio a hold still open gathers before Whisper hears it again, where its cut waits on words: a word or two
# said, and about a third of a second for the model to hear it, so hands is cut off within a second of the user speaking
# over it. The audit log had engaged turns cut 0.5 to 12.5 s after they opened, each just after the hold closed, all
# that time talked over (hands-voice-o6o).
OVERHEARING_SECONDS = 0.5

# Silence after a hold's last sample: a segment that ends right where the speech stopped has its last word dropped or
# garbled, so Whisper hears half a second of it past the end, as Pipecat padded every segment it gave a transcriber.
_TRAILING_SILENCE = bytes(SECOND // 2)


# Where hearing a hold while it is open stands: between hearings; one returned and not yet heard; or over, since words were
# heard, the hold closed, a hearing failed, or its cut never waited on words.
Overhearing = Literal["listening", "queued", "over"]


@dataclass(frozen=True)
class _Unread:
    """The vocabulary, not yet read for the hearings of a hold still open."""


@dataclass
class _Open:
    """The hold the key last opened, as Whisper hears it while it is open: the hop before its last hearing was returned
    and how much was gathered since, in bytes, where hearing it stands, the audio its next hearing hears, and the
    vocabulary every hearing of it is primed with, read for the first."""

    hold: Hold
    hop: int
    gathered: int
    overhearing: Overhearing
    heard: bytearray = field(default_factory=bytearray)
    captured: bytearray = field(default_factory=bytearray)
    prompt: str | None | _Unread = _Unread()


def _overhearing(hold: Hold) -> Overhearing:
    """Where hearing `hold` while it is open starts: listening where its cut waits on words, over where it cut as it opened."""
    match turn_start(hold.opener):
        case "on words":
            return "listening"
        case "on the hold":
            return "over"


class Whisper:
    """Whisper large-v3-turbo on MLX: the hold owner's transcriber, and the listener that hears a hold still open."""

    name = "whisper"

    def __init__(self, *, prompt: Callable[[], Awaitable[str | None]], record: Record) -> None:
        # [LAW:nothing-unseen] the load is a unit of work of its own: how long the start waited on it, and on what model.
        with unit("whisper.loaded", record):
            transcription.load()
            annotate(model=transcription.MODEL)
        # The vocabulary each hold is transcribed with, read as it is: see hands.voice.vocabulary.
        self._prompt = prompt
        self._record = record
        # [LAW:single-enforcer] the one thread the model runs on, for hearings and transcriptions alike: a run given up
        # on is still running, and the next must not run beside it.
        self._model = SerialThread("Whisper")
        self._opened = _Open(Hold(0, "held key"), 0, 0, "over")

    def opened(self, hold: Hold, before: Sound) -> Hearings:
        # What the hold kept from before it opened is its first hop.
        self._opened = _Open(hold, len(before.heard), 0, _overhearing(hold), bytearray(before.heard), bytearray(before.captured))
        return ()

    def heard(self, sound: Sound) -> Hearings:
        opened = self._opened
        opened.heard += sound.heard
        opened.captured += sound.captured
        opened.gathered += len(sound.heard)
        return self._overhear()

    def ended(self, hold: Hold, fate: Fate) -> Hearings:
        # Transcribed whole next, or thrown away: the model is not spent on a hearing of it still waiting.
        self._opened.overhearing = "over"
        return ()

    def _overhear(self) -> Hearings:
        """The hold open's last two hops to be heard, where its cut waits on words, none was heard in it yet, no hearing of
        it is waiting, and it has gathered `OVERHEARING_SECONDS` of audio since it opened or was last heard."""
        opened = self._opened
        match opened.overhearing:
            case "listening" if opened.gathered >= OVERHEARING_SECONDS * SECOND:
                heard = opened.hop + opened.gathered
                sound = Sound(bytes(opened.heard[-heard:]), bytes(opened.captured[-heard:]))
                opened.hop, opened.gathered, opened.overhearing = opened.gathered, 0, "queued"
                # Only the hop just gathered is heard again, by the next hearing.
                del opened.heard[: -opened.hop]
                del opened.captured[: -opened.hop]
                return (lambda: self._overheard(opened, sound),)
            case _:
                return ()

    async def _overheard(self, opened: _Open, sound: Sound) -> Heard:
        """The words heard in a hold still open, on the hold owner's event for this hearing: none where nothing was said in
        it lately, and the model not run at all where the hold closed before the model took it."""
        annotate(seconds=round(sound.seconds, 3), levels=sound.levels)
        match opened.overhearing:
            case "over":
                # Its own transcription says what it held: the model is spent on that instead.
                annotate(skipped="the hold closed before the model took it")
                return Heard(None)
            case "listening" | "queued":
                pass
        try:
            heard = await self._heard(opened.hold.number, sound.heard, await self._primed(opened))
        except BaseException:
            # Failed, or given up on: not heard again while it is open; its own transcription as it closes says what it held.
            opened.overhearing = "over"
            raise
        annotate(hearing=heard)
        # [LAW:single-enforcer] what counts as words is what `_heard` keeps: noise, a cough, or hands' echo left over by the
        # canceller is none, and cuts nothing. Heard words end the hearings of the hold; none, and it is heard on.
        opened.overhearing = "listening" if heard.said is None and opened.overhearing == "queued" else "over"
        return Heard(heard.said)

    async def _primed(self, opened: _Open) -> str | None:
        """The vocabulary the hearings of the hold open are primed with: read for its first hearing, kept for the rest."""
        match opened.prompt:
            case _Unread():
                prompt = await self._prompt()
                opened.prompt = prompt
                return prompt
            case prompt:
                return prompt

    async def transcribed(self, hold: Hold, audio: bytes) -> str | None:
        """The words said in a hold of the key's, primed with the vocabulary as it is now; recorded, so "I spoke and nothing
        happened" can be looked into."""
        heard = await self._heard(hold.number, audio, await self._prompt())
        self._record(heard)
        return heard.said

    async def _heard(self, hold: int, audio: bytes, prompt: str | None) -> HoldHeard:
        """What was said in a hold's audio, heard with `_TRAILING_SILENCE` after it, primed with `prompt`, the vocabulary.

        A segment with no word in it is dropped, as are one the decoder repeated itself in and one Whisper only guessed
        at: that is what a primed Whisper makes of noise. A hold with nothing said in it comes back with no segment.
        """
        # Set once nobody waits for this hold's transcription any more: the model is not run on it if it has yet to be.
        given_up = threading.Event()
        try:
            # Off the loop: the model runs for a third of a second a hold, and the speaker and the key go on meanwhile.
            scored = await self._model.run(lambda: [] if given_up.is_set() else transcription.segments(audio + _TRAILING_SILENCE, prompt))
        finally:
            given_up.set()
        said: list[str] = []
        dropped: list[Unsaid] = []
        for segment in scored:
            worded = any(character.isalnum() for character in segment.text)
            if worded and segment.compression_ratio <= _REPEATING and segment.avg_logprob >= _GUESSED:
                said.append(segment.text)
            else:
                dropped.append(segment)
        return HoldHeard(hold, " ".join(said).strip() or None, tuple(dropped))
