"""What the brain is sent of each take the harness holds (tests/echoscore.py), through the desk's canceller and the real
Whisper at its pinned weights: every word of the person's, and none of hands' own, with no rule about whose a word is by
its text or its time. The harness scores the streaming pair's dated words; Whisper's come back as one string a clip, so
the same two numbers are read here undated, over the stretch hands played alone and over the whole take. numba times
each compile by the machine's clock, so this module stays in UNMOVED in tests/clockwatch.py."""

from difflib import SequenceMatcher
from pathlib import Path

import pytest

import echoes
from echoscore import TAKES, Line, Take, normal, recorded
from listening import LEAD
from hands.virtualclock import VirtualClock
from hands.voice import vocabulary
from hands.voice.clips import RATE, SLOT, Clip, Sound
from hands.voice.whisper import Whisper

pytestmark = [pytest.mark.outside, pytest.mark.weights("whisper")]


@pytest.fixture(autouse=True)
def no_whisper_weights() -> None:
    """These tests are about the weights: the suite's stand-in for the model is not put in its place here."""


# A vocabulary as hands primes with one (see tests/test_whisper_recorded.py).
VOCABULARY = vocabulary.prompt(
    [
        *"atdesk backends backlogpage brainpage briefing clipowner clips conversation conversationpage coreaudio cues".split(),
        *"devicechoice engaged floor gallerypage keyed longpoll microphone milestone modelswitch narrator phoneaddress".split(),
        *"readback recorder refocus summarising synthesis talkkey taps transcript turnstart turnstop utterance".split(),
        *"voiceprints wakeword whisper working houdini-mcp macklebox phosphene".split(),
    ]
)


def desk(take: Take, first: float, last: float) -> Sound:
    """The stretch of a take from `first` to `last` seconds, as the desk hears it: through the echo canceller."""
    sound = echoes.replayed(take.captured, take.reference, take.playheads)
    start, stop = (round(min(max(second, 0.0), len(take.playheads) * SLOT / RATE) * RATE / SLOT) for second in (first, last))
    return Sound(sound.heard[2 * SLOT * start : 2 * SLOT * stop], sound.captured[2 * SLOT * start : 2 * SLOT * stop], (sound.played or ())[start:stop])


async def sent(sound: Sound) -> list[str]:
    """The words the brain is sent of a clip with `sound`."""

    async def primed() -> str | None:
        return VOCABULARY

    said = await Whisper(clock=VirtualClock(), prompt=primed, record=lambda _: None).transcribed(Clip(1, "shortcut"), sound)
    return normal(said or "")


def judged(line: Line, said: list[str]) -> tuple[list[str], list[str]]:
    """The words of a `line` the person said that `said` lost, and the words sent that are none of the line's: hands' leaked."""
    spoken = normal(line.text)
    lost, leaked = list[str](), list[str]()
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, spoken, said, autojunk=False).get_opcodes():
        # A word heard in place of one said is that one misheard: lost, and none of hands'.
        lost += spoken[i1:i2] if tag != "equal" else []
        leaked += said[j1 + (i2 - i1) if tag == "replace" else j1 : j2] if tag in ("replace", "insert") else []
    return lost, leaked


@pytest.mark.parametrize("fixture", TAKES, ids=lambda fixture: fixture.name)
async def test_hands_alone_sends_nothing_on_any_recorded_take(fixture: Path) -> None:
    take = recorded(fixture)
    assert await sent(desk(take, 0.0, take.cut_in)) == []


@pytest.mark.parametrize("fixture", [fixture for fixture in TAKES if recorded(fixture).person], ids=lambda fixture: fixture.name)
async def test_a_person_over_hands_is_sent_whole_and_hands_is_not_on_any_recorded_take(fixture: Path) -> None:
    take = recorded(fixture)
    ends = [*[line.start - 0.3 for line in take.person[1:]], float("inf")]
    for line, end in zip(take.person, ends, strict=True):
        # Each line its own clip, from LEAD before its first word to the next line's, as a voice-opened turn is.
        lost, leaked = judged(line, await sent(desk(take, line.start - LEAD, end)))
        assert (lost, leaked) == ([], []), line.text


@pytest.mark.parametrize("fixture", TAKES, ids=lambda fixture: fixture.name)
async def test_a_whole_take_is_sent_every_word_the_person_said_in_it_and_none_of_hands(fixture: Path) -> None:
    take = recorded(fixture)
    # One clip from the take's first sample, as a held key that the person speaks late into is: the quiet before them is in it.
    said = " ".join(line.text for line in take.person)
    lost, leaked = judged(Line(said, 0.0), await sent(desk(take, 0.0, float("inf"))))
    assert (lost, leaked) == ([], [])
