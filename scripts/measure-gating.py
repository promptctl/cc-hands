"""Measure how a clip's last words come out at each GATING (hands.voice.streaming), on turns with known words that end in
silence, fed in clips as the listener feeds them.

    uv run scripts/measure-gating.py DIR [GATING ...]    # fetches what it needs into DIR once; GATING 1 2 4 if none given

The turns: AMI Meeting Corpus's sixteen test meetings (Mix-Headset, CC BY 4.0), cut by AMI's manual word-level
transcripts into each turn one person says alone (`measuring.turns`) that nobody sounds in for BEFORE ahead of it or for
engaged.STOP_SECS past it, as a turn ends. Each is a clip in the two shapes a clip takes, cut and fed as the real-pair
tests cut and feed theirs (tests/listening.py): the shortcut, let go just after the last word, and engaged conversation,
closed once STOP_SECS of the silence after it are heard. One session per meeting, as a run has one, loaded at each GATING.

A clip's end is scored against its known words, aligned: whether its last word is right; insertions past the last known
word; deletions from the end; words given to anyone but the person whose turn it is; and the word error rate.
"""

import difflib
import re
import sys
import zipfile
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from hands.voice import engaged, streaming
from measuring import AMI_AUDIO, AMI_TEST, RATE, SHAPES, Live, Stretch, fetched, mono, turn, turns

AMI_WORDS = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
PEOPLE = "ABCD"
BEFORE = 0.3
# What the transcript times that is heard. Not its disfluency markers: each marks where someone broke off, and over the 70
# turns one ran on past, the stretch it is timed over is as quiet as the silence after it, a median 1.7 dB above it.
SOUNDS = ("w", "vocalsound", "gap")


def transcribed(archive: zipfile.ZipFile, meeting: str) -> tuple[list[Stretch], list[Stretch]]:
    """A meeting's words, each a stretch of its own, and every sound its transcript times: words, vocal sounds, gaps."""
    words, sounds = list[Stretch](), list[Stretch]()
    for person in PEOPLE:
        for element in ElementTree.fromstring(archive.read(f"words/{meeting}.{person}.words.xml")):
            if element.tag not in SOUNDS or element.get("starttime") is None or element.get("endtime") is None or element.get("punc") == "true":
                continue
            stretch = Stretch(float(element.attrib["starttime"]), float(element.attrib["endtime"]), person)
            sounds.append(stretch)
            if element.tag == "w":
                words.append(Stretch(stretch.start, stretch.end, person, (element.text or "",)))
    return words, sounds


def normal(words: Iterable[str]) -> list[str]:
    """Words as they are scored, known and heard alike: lowercase, only their letters, digits and apostrophes, and none left
    with nothing."""
    return [word for word in (re.sub(r"[^\w']", "", word.lower()) for word in words) if word]


def scored(known: tuple[str, ...], ended: streaming.Ended) -> Counter[str]:
    """A clip's end against the words said in it: the clip's own person is the speaker who said most of its words."""
    by = Counter({said.speaker: len(said.text.split()) for said in ended.words})
    own = by.most_common(1)[0][0] if by else None
    heard = normal(word for said in ended.words if said.speaker == own for word in said.text.split())
    truth = normal(known)
    edits = difflib.SequenceMatcher(None, truth, heard, autojunk=False).get_opcodes()
    tag, i1, i2, j1, j2 = edits[-1]
    return Counter(
        clips=1,
        words=len(truth),
        last=tag == "equal",
        insertions=j2 - j1 if tag == "insert" else max(0, (j2 - j1) - (i2 - i1)) if tag == "replace" else 0,
        deletions=i2 - i1 if tag == "delete" else max(0, (i2 - i1) - (j2 - j1)) if tag == "replace" else 0,
        others=sum(count for speaker, count in by.items() if speaker != own),
        errors=sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in edits if tag != "equal"),
    )


def measured(gating: int, meetings: list[list[tuple[tuple[str, ...], dict[str, np.ndarray]]]]) -> None:
    """Every meeting's clips at `gating`, on a pair loaded for it alone, let go as it ends: no two pairs are held at once."""
    pair = streaming.load(gating)
    for shape in SHAPES:
        total = Counter[str]()
        for clips in meetings:
            # A run per meeting, as hands has one: its session afresh, nobody heard yet, as the real-pair tests reset it.
            pair.reset()
            run = Live(pair)
            for words, clip in clips:
                total += scored(words, run.spoken(clip[shape])[1])
        print(
            f"{shape} GATING {gating}: {total['clips']} clips; last word right {total['last']}; insertions past the last word {total['insertions']}, "
            f"deletions from the end {total['deletions']}, given to anyone else {total['others']}; WER {total['errors'] / total['words']:.3f}",
            flush=True,
        )


def main() -> None:
    directory = Path(sys.argv[1])
    directory.mkdir(parents=True, exist_ok=True)
    gatings = [int(gating) for gating in sys.argv[2:]] or [1, 2, 4]
    archive = zipfile.ZipFile(fetched(directory, AMI_WORDS))
    # Each meeting's clips in each shape, and the words said in them, copied out so its whole audio, a few hundred MB, is let
    # go.
    meetings = list[list[tuple[tuple[str, ...], dict[str, np.ndarray]]]]()
    for name in AMI_TEST:
        audio = mono(fetched(directory, AMI_AUDIO.format(name)))
        alone = turns(*transcribed(archive, name), len(audio) / RATE, before=BEFORE, after=engaged.STOP_SECS)
        meetings.append([(said.words, {shape: turn(audio, said.start, said.end, after).copy() for shape, after in SHAPES.items()}) for said in alone])
    for gating in gatings:
        measured(gating, meetings)


if __name__ == "__main__":
    main()
