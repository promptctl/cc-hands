"""What the real pair decoded for a run, kept and served again: the fake of `streaming.Session`, so the logic that reads the
pair (`streaming.Stream`) is tested without it. [LAW:one-source-of-truth] a recording is made one way, by `Recorder` on the
real pair in a `real` case that passed (`pytest --record-decodes`), and read one way, by `Replay`.

A recording is the pair's answers by the run's position, in samples fed: what each feed that decoded something returned, and
what each close returned, with a hash of all the audio fed up to it. It holds no samples, so a run fed other audio than it
was made from (a changed fixture, a clip cut elsewhere) fails naming the recording, and a run fed the same audio in other
feeds than it was made from fails too: a feed ending where none did, with a decode recorded before it, would be served nothing
it should have."""

import hashlib
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hands.voice.streaming import Active, Closed, Decoded, Said, Samples, Session

RECORDINGS = Path(__file__).parent / "fixtures" / "decodes"


@dataclass(frozen=True)
class Feed:
    """What the feed that ended `end` samples into the run decoded."""

    end: int
    decoded: Decoded


@dataclass(frozen=True)
class Close:
    """What the close `end` samples into the run returned, and the hash of all the audio fed before it."""

    end: int
    audio: str
    closed: Closed


@dataclass(frozen=True)
class Recording:
    """Every feed that decoded anything and every close of a run, and how many samples the run was fed in all."""

    feeds: tuple[Feed, ...]
    closes: tuple[Close, ...]
    total: int

    def text(self) -> str:
        """One line a feed or a close, so a change to a recording is a change to a line."""
        feeds = [[feed.end, [[said.speaker, said.text] for said in feed.decoded.words], [[active.speaker, active.frame] for active in feed.decoded.active]] for feed in self.feeds]
        closes = [[close.end, close.audio, sorted(close.closed.speakers), [[said.speaker, said.text] for said in close.closed.words]] for close in self.closes]
        feeds_text, closes_text = ",\n".join(map(json.dumps, feeds)), ",\n".join(map(json.dumps, closes))
        return f'{{"total": {self.total},\n"feeds": [\n{feeds_text}\n],\n"closes": [\n{closes_text}\n]}}\n'

    @staticmethod
    def parsed(text: str) -> "Recording":
        raw = json.loads(text)
        feeds = tuple(Feed(end, Decoded(tuple(Said(*said) for said in words), tuple(Active(*active) for active in active))) for end, words, active in raw["feeds"])
        closes = tuple(Close(end, audio, Closed(frozenset(speakers), tuple(Said(*said) for said in words))) for end, audio, speakers, words in raw["closes"])
        return Recording(feeds, closes, raw["total"])


class Audio:
    """The hash of all the samples fed so far."""

    def __init__(self) -> None:
        self._hash = hashlib.sha256()

    def add(self, samples: Samples) -> None:
        self._hash.update(np.ascontiguousarray(samples, np.float32).tobytes())

    def digest(self) -> str:
        return self._hash.hexdigest()


class Recorder:
    """The real pair's session, as the run is fed it, remembered."""

    def __init__(self, session: Session) -> None:
        self._session = session
        self._total = 0
        self._audio = Audio()
        self._feeds: list[Feed] = []
        self._closes: list[Close] = []

    def decoded(self, samples: Samples) -> Decoded:
        self._total += len(samples)
        self._audio.add(samples)
        decoded = self._session.decoded(samples)
        if decoded.words or decoded.active:
            self._feeds.append(Feed(self._total, decoded))
        return decoded

    def closed(self) -> Closed:
        closed = self._session.closed()
        self._closes.append(Close(self._total, self._audio.digest(), closed))
        return closed

    def recording(self) -> Recording:
        return Recording(tuple(self._feeds), tuple(self._closes), self._total)


class Replay:
    """A recording served as the real pair's session: each feed that decoded something gets what it decoded, every other
    feed nothing, and each close what it returned. A run fed up to a point serves the recording up to it; one fed past the
    end of it, or any other audio or feed than it was made from, fails naming `name`."""

    def __init__(self, name: str, recording: Recording) -> None:
        self._name = name
        self._recording = recording
        self._feeds = deque(recording.feeds)
        self._closes = deque(recording.closes)
        self._total = 0
        self._audio = Audio()

    def _stale(self, why: str) -> AssertionError:
        return AssertionError(f"the recording {self._name!r} is not of this run: {why}; run `pytest --record-decodes -k real` to make it again")

    def decoded(self, samples: Samples) -> Decoded:
        self._total += len(samples)
        self._audio.add(samples)
        if self._total > self._recording.total:
            raise self._stale(f"the run is fed {self._total} samples, the recording ends at {self._recording.total}")
        if self._feeds and self._feeds[0].end < self._total:
            raise self._stale(f"a feed ended at {self._total} samples, with the decode recorded at {self._feeds[0].end} before it")
        if self._feeds and self._feeds[0].end == self._total:
            return self._feeds.popleft().decoded
        return Decoded((), ())

    def closed(self) -> Closed:
        if not self._closes:
            raise self._stale(f"the run closes at {self._total} samples, past the recording's last close")
        close = self._closes.popleft()
        if close.end != self._total:
            raise self._stale(f"the run closes at {self._total} samples, the recording at {close.end}")
        if close.audio != self._audio.digest():
            raise self._stale(f"the audio fed up to the close at {self._total} samples is not the audio recorded")
        return close.closed


def path(name: str) -> Path:
    return RECORDINGS / f"{name}.json"


def read(name: str) -> Replay:
    """The recording `name`, served."""
    return Replay(name, Recording.parsed(path(name).read_text()))


def write(name: str, recording: Recording) -> None:
    RECORDINGS.mkdir(parents=True, exist_ok=True)
    path(name).write_text(recording.text())
