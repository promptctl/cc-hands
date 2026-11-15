"""The milestones of a user turn: the moments between the key's release and the first sound after it.

One vocabulary for whoever is told of them: the latency log, which says how long after the release each came, and the
phone's page, which shows the user holding it what became of what they said.
"""

from dataclasses import dataclass
from typing import Literal

from hands.voice.clips import Opener

# In the order a turn that is answered passes them, after its release. `no words` is a turn Whisper found nothing said
# in, and `failed` a stage failing so that the release goes unanswered, each of which ends the wait. `first audio` is the
# speaker's first sound after the release, whatever hands said with it.
Milestone = Literal["transcript", "no words", "failed", "first LLM token", "first audio"]


@dataclass(frozen=True)
class ClipReleased:
    """The release: a clip taken, to be transcribed and sent, by what opened it.

    [LAW:types-are-the-program] the one milestone of a clip rather than of its turn, so it names the clip: a clip is
    sent at a place hands leaves while its button is still held, so the phone's page tells its own from another place's
    by what opened it, never by where it falls."""

    opener: Opener


Told = ClipReleased | Milestone
