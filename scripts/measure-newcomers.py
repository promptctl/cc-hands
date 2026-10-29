"""Measure whether a newcomer is given their own speaker in a hold they share with a voice the run knows, by how much of
each the run heard alone first, on hands' diarizer (hands.voice.streaming.diarizer) fed as the listener feeds the pair.

    uv run scripts/measure-newcomers.py DIR    # fetches what it needs into DIR once, then measures

The voices: AMI Meeting Corpus's sixteen test meetings (Mix-Headset, CC BY 4.0), cut into each turn one person says
alone (`measuring.turns`) that nobody sounds in for LEAD ahead of it or for engaged.STOP_SECS past it, per pyannote's
AMI-diarization-setup references, and that is no longer than LONGEST, as a hold is. Each ordered pair of people in a
meeting is a user, the voice the run knows, and a newcomer. A run hears the user alone in each of USERS of their turns,
then the newcomer alone in each of NEWCOMERS of theirs, then one hold of the next turn of each, joined, both orders: the
same joined hold for every amount heard. Each hold is followed by the silence `through` says, as the listener follows it.

Scored on the diarizer's decisions, not on words: the session masks each speaker's stream by them, so they decide whose
every word is, and the recognizer's own misses (about 20% of AMI's words, and all of a quiet person's) are not counted
as attribution. A stretch's speaker is the one the session hears in most of its chunks. The joined hold is right where
the user's stretch is the speaker the user's holds alone were, and the newcomer's is the speaker theirs were, another.
"""

import itertools
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from hands.voice import engaged, streaming
from hands.voice.streaming import MEL_HOP, SUBSAMPLING, through
from measuring import AMI_AUDIO, AMI_TEST, LEAD, RATE, fetched, mono, referenced, turn, turns

USERS = (1, 2, 5)
NEWCOMERS = (1, 3)
LONGEST = 6.0


def decided(diarizer: Any, holds: list[np.ndarray]) -> list[np.ndarray]:
    """The diarizer's decisions on each frame of each hold, the holds fed in turn as one run."""
    state, fed, spans, frames = diarizer.init_streaming_state(), 0, list[tuple[int, int]](), list[np.ndarray]()
    for samples in holds:
        silence = np.zeros(through(fed + len(samples)) - fed - len(samples), np.float32)
        result, state = diarizer.feed(np.concatenate([samples, silence]), state)
        frames.append(np.array(result.speaker_probs))
        spans.append((fed // MEL_HOP, (fed + len(samples)) // MEL_HOP))
        fed += len(samples) + len(silence)
    run = np.concatenate(frames)
    return [run[start:end] for start, end in spans]


def speaker(decisions: np.ndarray) -> int | None:
    """The speaker the session hears in most of these frames' chunks; None where it hears nobody.

    [LAW:one-source-of-truth] who speaks in a chunk is decided as the session decides it, the one decision that masks
    each speaker's stream: mlx-audio's own `speaker_activity`, at the session's threshold."""
    import inspect

    import mlx.core as mx  # pyright: ignore[reportMissingTypeStubs]
    from mlx_audio.stt.models.nemotron_asr.speaker_streaming import SpeakerStreamingSession, speaker_activity  # pyright: ignore[reportMissingTypeStubs]

    # hands leaves the session's threshold at mlx-audio's default (`streaming.load`).
    threshold = inspect.signature(SpeakerStreamingSession).parameters["threshold"].default
    marked = np.array(speaker_activity(mx.array(decisions), SUBSAMPLING, threshold)).sum(axis=0)
    return int(marked.argmax()) if marked.any() else None


def main() -> None:
    directory = Path(sys.argv[1])
    directory.mkdir(parents=True, exist_ok=True)
    diarizer = streaming.diarizer()
    # (user's turns, newcomer's turns, order) -> how each joined hold came out; and the joined holds each pair got wrong.
    tally: dict[tuple[int, int, str], Counter[str]] = {}
    missed: Counter[tuple[str, str, str]] = Counter()
    for name in AMI_TEST:
        audio = mono(fetched(directory, AMI_AUDIO.format(name)))
        segments = referenced(directory, name)
        alone = [t for t in turns(segments, segments, len(audio) / RATE, before=LEAD, after=engaged.STOP_SECS) if t.end - t.start <= LONGEST]
        for user, newcomer in itertools.permutations(sorted({t.person for t in alone}), 2):
            mine, theirs = [t for t in alone if t.person == user], [t for t in alone if t.person == newcomer]
            if len(mine) <= max(USERS) or len(theirs) <= max(NEWCOMERS):
                continue
            for users, newcomers, order in itertools.product(USERS, NEWCOMERS, ("newcomer first", "user first")):
                shared = (theirs[max(NEWCOMERS)], mine[max(USERS)])
                first, second = shared if order == "newcomer first" else shared[::-1]
                joined = (turn(audio, first.start, first.end, 0.0), turn(audio, second.start, second.end, engaged.STOP_SECS))
                held = [turn(audio, t.start, t.end, engaged.STOP_SECS) for t in (*mine[:users], *theirs[:newcomers])]
                *each, together = decided(diarizer, [*held, np.concatenate(joined)])
                known, new = speaker(np.concatenate(each[:users])), speaker(np.concatenate(each[users:]))
                stretches = speaker(together[: len(joined[0]) // MEL_HOP]), speaker(together[len(joined[0]) // MEL_HOP :])
                got_new, got_user = stretches if order == "newcomer first" else stretches[::-1]
                outcome = (
                    "unknown voices" if known is None or new in (None, known)
                    else "right" if (got_user, got_new) == (known, new)
                    else "newcomer as the user" if got_new == known
                    else "user as another" if got_user != known
                    else "newcomer as another"
                )  # fmt: skip
                tally.setdefault((users, newcomers, order), Counter())[outcome] += 1
                missed[(name, user, newcomer)] += outcome != "right"
        print(f"{name}: done", file=sys.stderr, flush=True)
    for (users, newcomers, order), outcomes in sorted(tally.items()):
        print(f"user heard alone in {users} turns, newcomer in {newcomers}, {order}: {outcomes['right']}/{outcomes.total()} right; {dict(outcomes - Counter(right=outcomes['right']))}")
    print(f"pairs: {sum(1 for n in missed.values() if n)} of {len(missed)} with a joined hold not right")
    for (name, user, newcomer), n in missed.most_common():
        if n:
            print(f"  {name} user {user} newcomer {newcomer}: {n} of {len(USERS) * len(NEWCOMERS) * 2}")


if __name__ == "__main__":
    main()
