"""Measure whether a newcomer is given their own speaker in a hold they share with a voice the run knows, by how much of
each the run heard alone first, on hands' diarizer (hands.voice.streaming.diarizer) fed as the listener feeds the pair.

    uv run scripts/measure-newcomers.py DIR    # fetches what it needs into DIR once, then measures

The voices: AMI Meeting Corpus's sixteen test meetings (Mix-Headset, CC BY 4.0), cut into each turn one person says
alone (`measuring.turns`) that nobody sounds in, laughs and coughs included, for LEAD ahead of it or for
engaged.STOP_SECS past it, per pyannote's AMI-diarization-setup references, and that is no longer than LONGEST, as a hold
is. Each ordered pair of people in a meeting is a user, the voice the run knows, and a newcomer. A run hears the user
alone in each of USERS of their turns, then the newcomer alone in each of NEWCOMERS of theirs, then one hold of the next
turn of each, joined, both orders: the same joined hold for every amount heard. Each hold is followed by the silence
`through` says, as the listener follows it.

Scored on the diarizer's decisions, not on words: the session masks each speaker's stream by them, so they decide whose
every word is, and the recognizer's own misses (about 20% of AMI's words, and all of a quiet person's) are not counted
as attribution. A stretch's speaker is the one the session hears in most of its encoder frames. The joined hold is right
where the user's stretch is the speaker every one of the user's holds alone was, and the newcomer's is the speaker every
one of theirs was, another.
"""

import itertools
import math
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


def decided(session: Any, holds: list[tuple[np.ndarray, ...]]) -> list[list[np.ndarray]]:
    """Which speakers `session` hears in each encoder frame of each stretch of each hold, the holds fed to its diarizer in
    turn as one run, each a stretch or more joined.

    [LAW:one-source-of-truth] decided as the session decides it, the one decision that masks each speaker's stream:
    mlx-audio's own `speaker_activity`, at the session's threshold, on the run's own grid of encoder frames."""
    import mlx.core as mx  # pyright: ignore[reportMissingTypeStubs]
    from mlx_audio.stt.models.nemotron_asr.speaker_streaming import speaker_activity  # pyright: ignore[reportMissingTypeStubs]

    diarizer = session.diarization_model
    state, fed, cuts, probs = diarizer.init_streaming_state(), 0, list[list[int]](), list[np.ndarray]()
    for stretches in holds:
        ends = list(itertools.accumulate(map(len, stretches), initial=fed))
        silence = np.zeros(through(ends[-1]) - ends[-1], np.float32)
        result, state = diarizer.feed(np.concatenate([*stretches, silence]), state)
        probs.append(np.array(result.speaker_probs))
        # A mel frame is the stretch's its centre is in, and an encoder frame the stretch's its first mel frame is.
        cuts.append([math.ceil(math.ceil(sample / MEL_HOP) / SUBSAMPLING) for sample in ends])
        fed = ends[-1] + len(silence)
    heard = np.array(speaker_activity(mx.array(np.concatenate(probs)), SUBSAMPLING, session.threshold))
    return [[heard[start:end] for start, end in itertools.pairwise(hold)] for hold in cuts]


def speaker(heard: np.ndarray) -> int | None:
    """The one speaker heard in more of these encoder frames than any other; None where nobody is, or two are as often."""
    counts = heard.sum(axis=0)
    most = np.flatnonzero(counts == counts.max())
    return int(most[0]) if counts.max() and len(most) == 1 else None


def judged(mine: set[int | None], theirs: set[int | None], got_user: int | None, got_new: int | None) -> str:
    """How a joined hold came out, from the speakers the user's holds alone and the newcomer's were each heard as, and
    each one's stretch of the joined hold."""
    if None in mine | theirs:
        return "nobody heard alone"
    if len(mine) > 1 or len(theirs) > 1:
        return "heard as two alone"
    (known,), (new,) = mine, theirs
    return (
        "newcomer as the user alone" if new == known
        else "right" if (got_user, got_new) == (known, new)
        else "newcomer as the user" if got_new == known
        else "nobody heard together" if None in (got_user, got_new)
        else "user as another" if got_user != known
        else "newcomer as another"
    )  # fmt: skip


def main() -> None:
    directory = Path(sys.argv[1])
    directory.mkdir(parents=True, exist_ok=True)
    # The run's session as hands builds it, so its diarizer and threshold are hands'; its recognizer is left unused.
    session = streaming.load(streaming.GATING).session
    # (user's turns, newcomer's turns, order) -> how each joined hold came out; and the joined holds each pair got wrong.
    tally: dict[tuple[int, int, str], Counter[str]] = {}
    missed: Counter[tuple[str, str, str]] = Counter()
    for name in AMI_TEST:
        audio = mono(fetched(directory, AMI_AUDIO.format(name)))
        spoken, sounds = referenced(directory, name, "only_words"), referenced(directory, name, "word_and_vocalsounds")
        alone = [t for t in turns(spoken, sounds, len(audio) / RATE, before=LEAD, after=engaged.STOP_SECS) if t.end - t.start <= LONGEST]
        for user, newcomer in itertools.permutations(sorted({t.person for t in alone}), 2):
            mine, theirs = [t for t in alone if t.person == user], [t for t in alone if t.person == newcomer]
            if len(mine) <= max(USERS) or len(theirs) <= max(NEWCOMERS):
                continue
            for users, newcomers, order in itertools.product(USERS, NEWCOMERS, ("newcomer first", "user first")):
                shared = (theirs[max(NEWCOMERS)], mine[max(USERS)])
                first, second = shared if order == "newcomer first" else shared[::-1]
                joined = (turn(audio, first.start, first.end, 0.0), turn(audio, second.start, second.end, engaged.STOP_SECS))
                held = [(turn(audio, t.start, t.end, engaged.STOP_SECS),) for t in (*mine[:users], *theirs[:newcomers])]
                *each, together = decided(session, [*held, joined])
                got_new, got_user = map(speaker, together if order == "newcomer first" else together[::-1])
                outcome = judged({speaker(e) for e, in each[:users]}, {speaker(e) for e, in each[users:]}, got_user, got_new)
                tally.setdefault((users, newcomers, order), Counter())[outcome] += 1
                missed[(name, user, newcomer)] += outcome != "right"
        print(f"{name}: done", file=sys.stderr, flush=True)
    for (users, newcomers, order), outcomes in sorted(tally.items()):
        print(f"user heard alone in {users} turns, newcomer in {newcomers}, {order}: {outcomes['right']}/{outcomes.total()} right; {dict(outcomes - Counter(right=outcomes['right']))}")
    people = {(user, newcomer) for _, user, newcomer in missed}
    print(f"pairs: {len(missed)} of a meeting's people, {len(people)} of people; {sum(1 for n in missed.values() if n)} with a joined hold not right")
    for (name, user, newcomer), n in missed.most_common():
        if n:
            print(f"  {name} user {user} newcomer {newcomer}: {n} of {len(USERS) * len(NEWCOMERS) * 2}")


if __name__ == "__main__":
    main()
