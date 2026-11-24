"""hands' tools as a test reads them: every act there is, and the tools a voice turn serves the brain."""

from collections.abc import Sequence
from dataclasses import replace

from hands.brain import pinned
from hands.brain.usage import Usage
from hands.core.session import SessionId
from hands.daemon.config import Config, OwnModel, SampleLines, Settings
from hands.sessions.home import Home
from hands.sessions.registry import Sessions
from hands.sessions.summaries import Summaries
from hands.spotify import Catalogue, Missing
from hands.voice.library import Library
from hands.voice.narrator import Recounts
from hands.voice.player import Player
from hands.voice.listening import Levels
from hands.voice.ptt import PushToTalk
from hands.voice.summaries import SummaryStore
from hands.voice.tool import Answer, Tool
from hands.voice.tools import acts, usage_tool
from hands.voice.voicechoice import DEFAULT
from hands.voice.voices import Voices
from hands.voice.braindoor import Voiced, BrainDoor
from hands.voice.wakeword import Pretrained


def every_act(sessions: Sessions, home: Home) -> list[Tool[Answer]]:
    """Every act a voice turn is served, as the daemon builds them, on `home` and with no Spotify credentials."""
    settings = Settings(None, Config())
    given = acts(
        sessions, SummaryStore(Summaries(home.root / "summaries.db")), home, Recounts(), PushToTalk(sessions.clock, lambda _entry: None).switch, Levels("off"), lambda: None, Pretrained(), OwnModel(home, pinned.PIN, {}, settings),
        Voices(home, Player(lambda _entry: None).lines, lambda _voice: None, Library(sessions.clock, lambda: []), DEFAULT, SampleLines(home, settings)), Catalogue(Missing("no credentials in tests"), sessions.clock), {"TMUX_TMPDIR": str(home.root)},
        lambda _entry: None,
    )  # fmt: skip
    return [*given, usage_tool(Usage(lambda: SessionId("brain")))]


def voiced(turn: BrainDoor, acts: Sequence[Tool[Answer]]) -> dict[str, Voiced]:
    """Some of the acts as the voice turn serves the brain them, and the voice's own tools, by name: the turn's parts for
    the acts given, since a part for an act not served is refused."""
    names = {act.name for act in acts}
    narrowed = replace(turn, parts={name: part for name, part in turn.parts.items() if name in names})
    return {served.tool.name: served for served in narrowed.serve(acts)}
