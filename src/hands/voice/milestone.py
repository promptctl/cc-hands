"""The milestones of a user turn: the moments between the key's release and the first sound after it.

One vocabulary for whoever is told of them: the latency log, which says how long after the release each came, and the
phone's page, which shows the user holding it what became of what they said.
"""

from dataclasses import dataclass
from typing import Literal

from hands.voice.clips import Clip

# In the order a turn that is answered passes them, after its release. `no words` is a turn Whisper found nothing said
# in, and `failed` a stage failing so that the release goes unanswered, each of which ends the wait. `first audio` is the
# speaker's first sound after the release, whatever hands said with it.
Milestone = Literal["transcript", "no words", "failed", "first LLM token", "first audio"]


@dataclass(frozen=True)
class ClipOpened:
    """A clip of the key's opened, numbered and by what opened it, as the gate's `turn`: the press that opened the turn
    learned it as it was made, so the call tells the page which clip its press is by it, never by when it opened."""

    clip: Clip
    turn: int


@dataclass(frozen=True)
class ClipReleased:
    """A clip taken, to be transcribed and sent: the release its turn's milestones are measured from.

    [LAW:types-are-the-program] the release names its clip, since a move between places can send the phone's clip while
    its button is still held: the page tells its own from another clip by number, never by where the release falls."""

    clip: Clip


Told = ClipOpened | ClipReleased | Milestone
