"""Hands' own voice, as the speaker plays it: each time it starts, and each time it falls quiet.

Pipecat's output transport says hands stopped, for a TTS's audio, only on the stop frame the TTS sends behind the last
audio of a response, or once it has had nothing more to say for its `stop_frame_timeout_s`: falling quiet is the end of
what hands said, played, and not a pause between its sentences.
"""

from dataclasses import dataclass

from hands.core.place import Place
from hands.voice.shortcut import Instant


@dataclass(frozen=True)
class HandsSpeaking:
    """Hands started speaking."""


@dataclass(frozen=True)
class HandsQuiet:
    """Hands fell quiet, at `at`, having spoken at `place`; `deaf` where the desk has no microphone to hear an answer."""

    at: Instant
    place: Place
    deaf: bool


HandsVoice = HandsSpeaking | HandsQuiet
