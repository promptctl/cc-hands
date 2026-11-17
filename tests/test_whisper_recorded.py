"""Whisper as hands runs it, on speech recorded once and committed (tests/speech.py): the real model at its pinned weights,
primed with a vocabulary of the shape hands primes with, given each clip a held key sends."""

import re
from difflib import SequenceMatcher

import numpy as np
import pytest

from hands.clock import SYSTEM
from hands.voice import vocabulary
from hands.voice.clips import Clip
from hands.voice.whisper import Whisper
from speech import RATE, quiet, spoken

pytestmark = pytest.mark.outside


@pytest.fixture(autouse=True)
def no_whisper_weights() -> None:
    """These tests are about the weights: the suite's stand-in for the model is not put in its place here."""


# A vocabulary as hands primes with one: the focused repository's newest file names, then the running sessions'
# projects, 40 words in all (hands.voice.vocabulary), as promptctl-voice-i15t.a5s.8z9 measured the release pass with.
VOCABULARY = vocabulary.prompt(
    [
        *"atdesk backends backlogpage brainpage briefing clipowner clips conversation conversationpage coreaudio cues".split(),
        *"devicechoice engaged floor gallerypage keyed longpoll microphone milestone modelswitch narrator phoneaddress".split(),
        *"readback recorder refocus summarising synthesis talkkey taps transcript turnstart turnstop utterance".split(),
        *"voiceprints wakeword whisper working houdini-mcp macklebox phosphene".split(),
    ]
)

# Sentences naming words of the vocabulary no dictionary holds, and those words. Unprimed, Whisper spelled none of them
# as named: "clip counter", "MacLeBox", "Phosphine", "Houdini MCP", "the clog page", "read back".
NAMED = {
    "Open clipowner and look at how turnstop ends a hold.": {"clipowner", "turnstop"},
    "Switch to macklebox and run the tests again.": {"macklebox"},
    "Ask phosphene whether the release build finished.": {"phosphene"},
    "File a ticket in houdini-mcp for the broken export.": {"houdini-mcp"},
    "The voiceprints change broke backlogpage, so revert it.": {"voiceprints", "backlogpage"},
    "Move the narrator code next to readback and summarising.": {"readback", "summarising"},
}

PROSE = [
    "I want the summary to say what changed, not how we got there.",
    "Keep the old behavior behind the flag until the migration lands.",
    "When the build fails on the second machine, read the log before you rerun anything.",
    "Write the test first, then make the smallest change that passes it.",
    "The release notes should name every setting a user has to change by hand.",
    "If the cache is cold, the first request takes almost four seconds.",
    "Rename the helper so its name says what it returns, and drop the comment.",
    "We agreed on Tuesday that the export runs nightly, not every hour.",
    "Tell me which of the three branches still has work nobody merged.",
    "Before lunch, send the draft to the reviewers and ask for comments by Friday.",
    "The installer should refuse to run when the disk has less than a gigabyte free.",
    "Leave the retry loop alone, and log each attempt with its delay.",
    "After the meeting, close the tickets we decided not to build.",
    "Read the error out loud to me, then suggest one fix and wait.",
]

# The pauses between a dictation's sentences, in seconds, in turn: a person stopping to think.
PAUSES = [1.0, 3.0, 2.0, 5.0, 1.5, 4.0]


def whisper(prompt: str | None) -> Whisper:
    async def primed() -> str | None:
        return prompt

    return Whisper(clock=SYSTEM, prompt=primed, record=lambda _: None)


def words(text: str) -> list[str]:
    return re.findall(r"[\w'-]+", text.lower())


def room(seconds: float, dbfs: float) -> bytes:
    """A quiet room as the microphone hands it over: noise at `dbfs`, the same on every run."""
    noise = np.random.default_rng(0).standard_normal(round(RATE * seconds)) * 10 ** (dbfs / 20) * 32768
    return noise.astype(np.int16).tobytes()


@pytest.mark.parametrize("text", NAMED)
async def test_a_name_dictated_comes_back_spelled_as_the_vocabulary_spells_it(text: str) -> None:
    said = await whisper(VOCABULARY).transcribed(Clip(1, "shortcut"), spoken(text))

    assert said is not None and NAMED[text] <= set(words(said)), said


# The user's silent holds were digital silence, -103 to -118 dBFS; a room is louder.
@pytest.mark.parametrize("audio", [quiet(3), quiet(10), room(3, -60)], ids=["3 s of silence", "10 s of silence", "3 s of a quiet room"])
@pytest.mark.parametrize("prompt", [None, VOCABULARY], ids=["unprimed", "primed"])
async def test_a_hold_nobody_spoke_in_sends_nothing(audio: bytes, prompt: str | None) -> None:
    # Whisper heard "Thank you." in each unprimed, and primed, read back the vocabulary's last words.
    assert await whisper(prompt).transcribed(Clip(1, "shortcut"), audio) is None


async def test_a_long_dictation_comes_back_whole() -> None:
    # Two minutes held, past four of Whisper's 30 s windows, each decoded on its own.
    sentences = [*NAMED, *PROSE]
    audio = b"".join(spoken(text) + quiet(PAUSES[index % len(PAUSES)]) for index, text in enumerate(sentences))

    said = await whisper(VOCABULARY).transcribed(Clip(1, "shortcut"), audio)

    assert said is not None
    # Every sentence is there, in order, near enough word for word: a window lost loses whole sentences, while Whisper
    # spelling a word its own way ("re-run", "4") or losing one at a window's edge loses a word.
    spoken_words = [(index, word) for index, text in enumerate(sentences) for word in words(text)]
    matcher = SequenceMatcher(None, [word for _, word in spoken_words], words(said), autojunk=False)
    heard = {position for block in matcher.get_matching_blocks() for position in range(block.a, block.a + block.size)}
    kept = [sum(position in heard for position, (index, _) in enumerate(spoken_words) if index == sentence) / len(words(text)) for sentence, text in enumerate(sentences)]
    assert min(kept) >= 0.75, (said, list(zip(sentences, kept)))
