"""hands' own model: chosen by voice as an edit to config.toml, made only once the user has heard hands say it is switching."""

import asyncio
import threading
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
from pipecat.frames.frames import InterruptionFrame, TTSSpeakFrame
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.frames.frames import Frame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from conftest import laid_out, new_sessions, Running, running
from fixtures.pins import LAID_OUT
from hands.clock import SYSTEM
from hands.daemon import config
from hands.daemon.config import Claude, Config
from hands.sessions.audit import Entry, SettingsEdited
from hands.sessions.home import Home
from hands.sessions.payload import Rejected
from hands.core.playback import hands_says
from hands.voice.modelswitch import Switches
from hands.voice.offeredmodels import OFFERED_MODELS, OfferedModel
from hands.voice.player import Marker, Markers, Player
from hands.voice.tool import After, Answer, Called, Effect, Tool, audited, result_of
from hands.sessions.wide import WideEvent
from hands.voice.tools import model_tools
from hands.voice.braindoor import Voiced, silent, brain_door
from hands.voice.refocus import Refocus
from served import voiced
from test_playback import Heard
from fixtures.fakemachine import UNMANAGED

SPOKEN_FILE = """# hands, by hand
[llm]
backend = "claude"  # the brain
model = "claude-sonnet-5-5"

[telemetry]
collector = "http://otel.example:4318"
"""


@pytest.fixture(autouse=True)
def _laid_out(tmp_path: Path) -> None:
    # Each test's home is tmp_path, whose brain the start's preflight passes on LAID_OUT.
    laid_out(Home(tmp_path))


def _own(home: Home) -> config.OwnModel:
    return config.OwnModel(home, LAID_OUT, UNMANAGED, {}, config.load(home))


def test_a_model_chosen_is_an_edit_to_the_file_that_keeps_every_other_line_as_written(tmp_path: Path) -> None:
    home = Home(tmp_path)
    home.config.write_text(SPOKEN_FILE)
    _own(home).weigh("claude-opus-5-5").keep()
    assert home.config.read_text() == SPOKEN_FILE.replace('model = "claude-sonnet-5-5"', 'model = "claude-opus-5-5"')
    assert config.load(home).config == Config(llm=Claude(model="claude-opus-5-5"), collector="http://otel.example:4318")


def test_a_model_chosen_with_no_file_is_the_default_backend_on_that_model(tmp_path: Path) -> None:
    home = Home(tmp_path)
    _own(home).weigh("claude-opus-5-5").keep()
    assert home.config.read_text() == '[llm]\nmodel = "claude-opus-5-5"\n'
    assert config.load(home).config == Config(llm=Claude(model="claude-opus-5-5"))


def test_the_model_running_is_the_one_the_run_started_on_not_the_file_since(tmp_path: Path) -> None:
    home = Home(tmp_path)
    home.config.write_text('[llm]\nmodel = "claude-fable-5-1"\n')
    own = _own(home)
    home.config.write_text('[llm]\nmodel = "claude-opus-5-5"\n')
    assert own.running() == "claude-fable-5-1"


def test_the_model_hands_runs_on_already_is_refused_and_nothing_is_written(tmp_path: Path) -> None:
    home = Home(tmp_path)
    # A personality saved since the start and not taken yet is no reason to say otherwise.
    home.config.write_text(SPOKEN_FILE)
    own = _own(home)
    home.config.write_text(SPOKEN_FILE + '[talk]\npersonality = "Dry."\n')
    with pytest.raises(Rejected, match="^hands runs on claude-sonnet-5-5 already$"):
        own.weigh("claude-sonnet-5-5")
    assert home.config.read_text() == SPOKEN_FILE + '[talk]\npersonality = "Dry."\n'


async def test_a_model_the_file_names_already_since_the_start_is_chosen_and_the_run_starts_again_on_it(tmp_path: Path) -> None:
    home = Home(tmp_path)
    home.config.write_text(SPOKEN_FILE)
    running_on = config.load(home)
    # Saved while the run went on, and not taken yet.
    on_opus = SPOKEN_FILE.replace("claude-sonnet-5-5", "claude-opus-5-5")
    home.config.write_text(on_opus)
    config.OwnModel(home, LAID_OUT, UNMANAGED, {}, running_on).weigh("claude-opus-5-5").keep()
    assert home.config.read_text() == on_opus
    assert await asyncio.wait_for(config.edited(home, LAID_OUT, UNMANAGED, {}, lambda _entry: None, running_on, SYSTEM, period=0.01), 2.0) == SettingsEdited(path=str(home.config), refused=None)


def test_a_file_whose_one_fault_is_its_model_is_mended_by_a_model_chosen(tmp_path: Path) -> None:
    home = Home(tmp_path)
    home.config.write_text(SPOKEN_FILE)
    own = _own(home)
    # Saved by hand while the run went on, naming a model hands does not offer: refused as it was saved.
    home.config.write_text(SPOKEN_FILE.replace("claude-sonnet-5-5", "claude-opus-9"))
    own.weigh("claude-opus-5-5").keep()
    assert config.load(home).config == Config(llm=Claude(model="claude-opus-5-5"), collector="http://otel.example:4318")


def test_a_model_on_a_home_a_start_would_refuse_is_refused_before_it_is_written(tmp_path: Path) -> None:
    home = Home(tmp_path)
    home.config.write_text(SPOKEN_FILE)
    (home.brain / "settings.json").write_text('{"syncClaudeAiPlugins": false}')
    with pytest.raises(Rejected, match="^the brain would load its account's syncClaudeAiSkills: "):
        _own(home).weigh("claude-opus-5-5")
    assert home.config.read_text() == SPOKEN_FILE


def test_a_model_chosen_on_a_file_that_is_not_toml_is_refused_naming_the_file(tmp_path: Path) -> None:
    home = Home(tmp_path)
    home.config.write_text("[llm\n")
    with pytest.raises(Rejected, match=f"^{home.config}: not TOML"):
        _own(home).weigh("claude-opus-5-5")
    assert home.config.read_text() == "[llm\n"


def test_a_save_by_hand_after_the_choice_was_weighed_is_not_written_over(tmp_path: Path) -> None:
    home = Home(tmp_path)
    home.config.write_text(SPOKEN_FILE)
    choice = _own(home).weigh("claude-opus-5-5")
    assert choice.to == "claude-opus-5-5"
    home.config.write_text(SPOKEN_FILE + "# saved by hand\n")
    with pytest.raises(Rejected, match="changed after the choice was made, by another choice kept first or a save by hand"):
        choice.keep()
    assert home.config.read_text() == SPOKEN_FILE + "# saved by hand\n"


def test_of_two_choices_in_flight_the_one_kept_first_is_kept_and_the_other_refused(tmp_path: Path) -> None:
    home = Home(tmp_path)
    home.config.write_text(SPOKEN_FILE)
    own = _own(home)
    # Both weighed before either is kept, as two switches heard at once are.
    opus, fable = own.weigh("claude-opus-5-5"), own.weigh("claude-fable-5-1")
    opus.keep()
    with pytest.raises(Rejected, match="by another choice kept first"):
        fable.keep()
    assert home.config.read_text() == SPOKEN_FILE.replace("claude-sonnet-5-5", "claude-opus-5-5")


class Watched:
    """The lock every write of config.toml holds, counting each writer that comes to it."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.come = threading.Semaphore(0)

    def __enter__(self) -> None:
        self.come.release()
        self.lock.acquire()

    def __exit__(self, *_exc: object) -> None:
        self.lock.release()


def test_a_model_kept_while_a_sample_line_is_written_waits_for_the_write_and_is_refused_on_the_file_it_left(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = Home(tmp_path)
    home.config.write_text(SPOKEN_FILE)
    choice = _own(home).weigh("claude-opus-5-5")
    lines = config.SampleLines(home, config.load(home))
    writing, written = threading.Event(), threading.Event()
    replace_whole = config.replace_whole

    def held_open(target: Path, text: str, mode: int) -> None:
        # The line's write, held open under the lock until the model's keep has come to it.
        writing.set()
        written.wait()
        replace_whole(target, text, mode)

    watched = Watched()
    monkeypatch.setattr(config, "replace_whole", held_open)
    monkeypatch.setattr(config, "_WRITING", watched)
    refused: list[str] = []

    def keep_model() -> None:
        try:
            choice.keep()
        except Rejected as error:
            refused.append(str(error))

    line, model = threading.Thread(target=lines.keep, args=("mary", "Mary here.")), threading.Thread(target=keep_model)
    line.start()
    try:
        writing.wait()
        model.start()
        # Both writers have come to the lock, the line's holding it, before its write ends.
        watched.come.acquire()
        watched.come.acquire()
    finally:
        # Ended however the test does, so a keep that never comes to the lock fails the test and never hangs the run.
        written.set()
    line.join()
    model.join()
    assert refused == [f"{home.config} changed after the choice was made, by another choice kept first or a save by hand, so the choice was not written over it"]
    assert "Mary here." in home.config.read_text()
    assert config.load(home).config.llm.model == "claude-sonnet-5-5"


def test_a_file_kept_behind_a_link_is_edited_through_it_and_the_link_stays(tmp_path: Path) -> None:
    kept = tmp_path / "dotfiles" / "hands.toml"
    kept.parent.mkdir()
    kept.write_text(SPOKEN_FILE)
    home = Home(tmp_path)
    home.config.symlink_to(kept)
    _own(home).weigh("claude-opus-5-5").keep()
    assert home.config.is_symlink()
    assert 'model = "claude-opus-5-5"' in kept.read_text()


def test_a_model_chosen_while_a_model_flag_outranks_the_file_is_refused_naming_the_flag(tmp_path: Path) -> None:
    home = Home(tmp_path)
    home.config.write_text(SPOKEN_FILE)
    flagged = config.OwnModel(home, LAID_OUT, UNMANAGED, {}, config.load(home, "claude-fable-5-1"))
    assert flagged.running() == "claude-fable-5-1"
    with pytest.raises(Rejected, match=f"^hands was started with --model claude-fable-5-1, which it keeps over {home.config} until it is started without it$"):
        flagged.weigh("claude-opus-5-5")
    assert home.config.read_text() == SPOKEN_FILE


async def test_a_model_kept_is_an_edit_the_run_starts_again_on(tmp_path: Path) -> None:
    home = Home(tmp_path)
    home.config.write_text(SPOKEN_FILE)
    running_on = config.load(home)
    watching = asyncio.create_task(asyncio.wait_for(config.edited(home, LAID_OUT, UNMANAGED, {}, lambda _entry: None, running_on, SYSTEM, period=0.01), 2.0))
    config.OwnModel(home, LAID_OUT, UNMANAGED, {}, running_on).weigh("claude-opus-5-5").keep()
    assert await watching == SettingsEdited(path=str(home.config), refused=None)


@dataclass(frozen=True)
class FakeChoice:
    to: OfferedModel
    own: "FakeOwn"

    def keep(self) -> None:
        if self.own.unkept is not None:
            raise Rejected(self.own.unkept)
        self.own.kept.append(self.to)


class FakeOwn:
    """hands' own model, kept in memory: what was kept, or the choice refused as it is weighed, or as it is kept."""

    def __init__(self, refusal: str | None = None, unkept: str | None = None) -> None:
        self.refusal = refusal
        self.unkept = unkept
        self.kept: list[OfferedModel] = []

    def running(self) -> OfferedModel:
        return "claude-sonnet-5-5"

    def weigh(self, model: OfferedModel) -> FakeChoice:
        if self.refusal is not None:
            raise Rejected(self.refusal)
        return FakeChoice(model, self)


@asynccontextmanager
async def spoken(player: Player, speaker: FrameProcessor | None = None) -> AsyncGenerator[tuple[Running, Heard]]:
    """The player's lines through a speaker and output transport, each reduced to a processor that lets frames through,
    with the markers played behind the output as the daemon stands them."""
    speaker, output, heard = speaker or IdentityFilter(), IdentityFilter(), Heard()
    async with running([player.lines, speaker, output, Markers(), heard], [player.watching(speaker, output)]) as run:
        yield run, heard


def _voiced(own: FakeOwn, player: Player) -> dict[str, Voiced]:
    """The model tools as a voice turn serves the brain them."""
    sessions, home = new_sessions(clock=SYSTEM), Home(Path("/nonexistent"))
    return voiced(brain_door(sessions, home, player, Refocus(sessions, home, lambda _entry: None), lambda: None, Switches(player)), model_tools(own))


def _tools(own: FakeOwn, player: Player) -> dict[str, Tool[Answer]]:
    return {name: each.tool for name, each in _voiced(own, player).items()}


def _effect(answer: Answer) -> Effect:
    assert isinstance(answer, After), answer
    return answer.effect


async def test_a_line_heard_to_its_end_is_said_so() -> None:
    player = Player(lambda _entry: None)
    async with spoken(player):
        assert await asyncio.wait_for(player.heard(hands_says("Switching.")), 2.0) is True


class Unfinished(FrameProcessor):
    """An output transport whose speaker never finishes what it is handed: it passes on every frame but a marker."""

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        match frame:
            case Marker():
                pass
            case _:
                await self.push_frame(frame, direction)


async def test_a_line_is_heard_once_the_output_passes_its_marker_on_whatever_takes_it_after() -> None:
    player = Player(lambda _entry: None)
    # No Markers behind the output: a barge-in empties its queue, so what counts is the output's push, seen in order.
    speaker, output, heard = IdentityFilter(), IdentityFilter(), Heard()
    async with running([player.lines, speaker, output, heard], [player.watching(speaker, output)]) as run:
        hearing = asyncio.create_task(player.heard(hands_says("Switching.")))
        await heard.until(1, Marker)
        await run.worker.queue_frame(InterruptionFrame())
        assert await asyncio.wait_for(hearing, 2.0) is True


async def test_a_line_cut_off_by_a_barge_in_is_said_so() -> None:
    player = Player(lambda _entry: None)
    # The marker is never passed on, so the line is never heard to its end: only a barge-in can settle it.
    speaker, output, heard = IdentityFilter(), Unfinished(), Heard()
    async with running([player.lines, speaker, output, heard], [player.watching(speaker, output)]) as run:
        hearing = asyncio.create_task(player.heard(hands_says("Switching.")))
        await heard.until(1, TTSSpeakFrame)
        await run.worker.queue_frame(InterruptionFrame())
        assert await asyncio.wait_for(hearing, 2.0) is False


async def test_use_model_says_it_is_switching_and_keeps_the_model_once_answered_and_is_the_whole_reply() -> None:
    own, player = FakeOwn(), Player(lambda _entry: None)
    served = _voiced(own, player)
    async with spoken(player) as (_, heard):
        answer = await served["use_model"].tool.body(model="claude-opus-5-5")
        said = await heard.until(1, TTSSpeakFrame)
    line = "Switching to claude-opus-5-5. I'll be back in a few seconds."
    assert [frame.text for frame in said if isinstance(frame, TTSSpeakFrame)] == [line]
    assert result_of(answer) == {"said": line, "running_on": "claude-sonnet-5-5", "switching_to": "claude-opus-5-5"}
    assert silent(served["use_model"], result_of(answer))
    # Kept only once the door runs its effect, after the answer is written.
    assert own.kept == []
    await _effect(answer)()
    assert own.kept == ["claude-opus-5-5"]


async def test_use_model_alone_says_nothing_and_keeps_nothing_until_its_effect_is_run() -> None:
    """The act as any caller is answered it: no line is spoken, since no player is there to speak it, and an answer left
    with its effect unrun holds nothing."""
    own = FakeOwn()
    [_, use_model] = model_tools(own)
    answer = await use_model.body(model="claude-opus-5-5")
    assert result_of(answer) == {"running_on": "claude-sonnet-5-5", "switching_to": "claude-opus-5-5"}
    assert own.kept == []
    await _effect(answer)()
    assert own.kept == ["claude-opus-5-5"]


async def test_use_model_shows_the_brain_the_models_hands_offers_and_refuses_another_before_it_is_weighed() -> None:
    own = FakeOwn(refusal="weighed")
    [_, use_model] = model_tools(own)
    assert use_model.properties["model"]["enum"] == list(OFFERED_MODELS)
    # Asked for by a name it does not offer, the brain is told the ones it does, and nothing is weighed or kept.
    assert await use_model.body(model="claude-opus-9") == {"error": f"'claude-opus-9' is no model; it is one of {', '.join(OFFERED_MODELS)}"}
    assert own.kept == []


async def test_a_switch_said_but_refused_as_it_is_kept_is_said_aloud() -> None:
    own, player = FakeOwn(unkept="the home's brain is not logged in"), Player(lambda _entry: None)
    served = _voiced(own, player)
    async with spoken(player) as (_, heard):
        answer = await served["use_model"].tool.body(model="claude-opus-5-5")
        await heard.until(1, TTSSpeakFrame)
        with pytest.raises(Rejected):
            await _effect(answer)()
        said = await heard.until(2, TTSSpeakFrame)
    assert [frame.text for frame in said if isinstance(frame, TTSSpeakFrame)][1] == "The switch to claude-opus-5-5 did not happen: the home's brain is not logged in"
    assert own.kept == []


async def test_use_model_refused_says_nothing_and_switches_nothing_and_the_model_answers() -> None:
    own, player = FakeOwn(refusal="the brain is not logged in"), Player(lambda _entry: None)
    served = _voiced(own, player)
    async with spoken(player) as (_, heard):
        result = await served["use_model"].tool.body(model="claude-opus-5-5")
    assert result == {"error": "the brain is not logged in"}
    assert not [frame for frame in heard.frames if isinstance(frame, TTSSpeakFrame)]
    assert own.kept == []
    assert not silent(served["use_model"], result_of(result))


async def test_use_model_cut_off_before_its_line_is_heard_switches_nothing() -> None:
    own, player = FakeOwn(), Player(lambda _entry: None)
    tools = _tools(own, player)
    speaker, output, heard = IdentityFilter(), Unfinished(), Heard()
    async with running([player.lines, speaker, output, heard], [player.watching(speaker, output)]) as run:
        call = asyncio.ensure_future(tools["use_model"].body(model="claude-opus-5-5"))
        await heard.until(1, TTSSpeakFrame)
        await run.worker.queue_frame(InterruptionFrame())
        result = await call
    assert result == {"error": "The user spoke over hands saying it would switch to claude-opus-5-5, so it did not switch. Ask whether they still want it."}
    assert own.kept == []


async def test_model_in_use_is_the_model_hands_runs_on() -> None:
    tools = _tools(FakeOwn(), Player(lambda _entry: None))
    assert await tools["model_in_use"].body() == {"running_on": "claude-sonnet-5-5"}


async def test_each_model_tool_call_is_one_event_naming_the_model_run_on_and_the_one_switched_to() -> None:
    recorded: list[Entry] = []
    def audited_tools(player: Player) -> dict[str, Tool[Answer]]:
        return {name: audited(SYSTEM, tool, recorded.append) for name, tool in _tools(FakeOwn(), player).items()}

    tools = audited_tools(player := Player(lambda _entry: None))
    async with spoken(player):
        await tools["model_in_use"].body()
        await tools["use_model"].body(model="claude-opus-5-5")
    # A player of its own: its lines stand in one pipeline.
    tools = audited_tools(player := Player(lambda _entry: None))
    speaker, output, heard = IdentityFilter(), Unfinished(), Heard()
    async with running([player.lines, speaker, output, heard], [player.watching(speaker, output)]) as run:
        call = asyncio.ensure_future(tools["use_model"].body(model="claude-opus-5-5"))
        await heard.until(1, TTSSpeakFrame)
        await run.worker.queue_frame(InterruptionFrame())
        await asyncio.wait_for(call, 2.0)
    events = [entry for entry in recorded if isinstance(entry, WideEvent)]
    assert [(event.outcome, cast(Called, event.facts["called"]).result) for event in events] == [
        ("ok", {"running_on": "claude-sonnet-5-5"}),
        ("ok", {"said": "Switching to claude-opus-5-5. I'll be back in a few seconds.", "running_on": "claude-sonnet-5-5", "switching_to": "claude-opus-5-5"}),
        ("failed", {"error": "The user spoke over hands saying it would switch to claude-opus-5-5, so it did not switch. Ask whether they still want it."}),
    ]
