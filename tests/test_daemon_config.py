"""The process boundary: the settings file, parsed once, and the secret each backend it names reaches its model with."""

import argparse
import asyncio
import json
import os
import re
import shutil
import tempfile
import threading
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path

import pytest

from conftest import children_on, laid_out, new_sessions, onboard, patiently, pin, shell_without_hands_settings
from fixtures.fakemachine import UNMANAGED
from fakeprocesses import processes
from faketerminals import FakeFritters, FakeScreen, FakeTerminals
from hands.core.front import FrontUnread
from fixtures.pins import LAID_OUT
from hands.brain import configdir, pinned, process
from hands.core.session import PinnedClaude, Runnable
from hands.clock import SYSTEM, Clock
from hands.daemon import cli, config, preflight, run
from hands.daemon.config import ANTHROPIC_MODEL, CLAUDE_MODELS, Claude, Config
from hands.daemon.restart import RESTART_SIGNAL
from hands.daemon.starting import CannotStart, Ended, Handover, QuitKey, Signalled, Start, Stop, Failing, start
from hands.voice.desk import Desk, Loaded, Resting
from hands.voice.listening import Driving, Level, Levels, Listening
from hands.daemon.backend import backend
from hands.sessions import audit, heartbeat
from hands.sessions.audit import Entry, SettingsEdited
from hands.sessions.hookconfig import DISPLAY_PATH
from hands.sessions.home import Home
from hands.sessions.payload import Rejected
from hands.sessions.processes import Processes
from hands.sessions.registry import Sessions
from hands.sessions.wide import WideEvent
from hands.core.wire import UPSTREAM
from hands.voice import voicechoice
from hands.voice.wakeword import Pretrained, Trained
from hands.voice.devicechoice import Auto, Choices, Named, Off
from hands.voice.backends import Account, ClaudeCodeBackend

HOME = Home(Path("/Users/someone/.hands"))


def test_no_file_is_the_brain_on_sonnet(tmp_path: Path) -> None:
    assert config.load(Home(tmp_path)) == config.Settings(None, Config(llm=Claude(model=ANTHROPIC_MODEL), collector=None))
    assert config.parse("") == Config()


def test_the_file_names_the_backend_and_its_model(tmp_path: Path) -> None:
    home = Home(tmp_path)
    home.config.write_text('[llm]\nbackend = "claude"\nmodel = "claude-opus-5-5"\n')
    settings = config.load(home)
    assert settings.config == Config(llm=Claude(model="claude-opus-5-5"))
    assert settings.path(home) == home.config
    assert config.parse('[llm]\nmodel = "claude-opus-5-5"\n').llm == Claude(model="claude-opus-5-5")


@pytest.mark.outside
def test_the_file_names_how_hands_comes_across_in_the_users_words(fake_claude: Path, tmp_path: Path) -> None:
    home = Home(tmp_path / ".hands")
    onboard(home.brain)
    brain = pin(home.brain, fake_claude)
    home.config.write_text('[talk]\npersonality = """\n  Dry and wry.\n"""\n')
    assert run.configured_from(children_on(SYSTEM), home, config.load(home), preflight.preflight(home, brain, UNMANAGED, shell_without_hands_settings())).voice.personality == "Dry and wry."
    # Left out, hands comes across as its own.
    assert config.parse("").personality is None


def test_the_file_names_the_wake_word_one_of_openwakewords_own_or_the_users_own_model() -> None:
    # Left out, it is Hey Jarvis; one of openWakeWord's own is matched however it is capitalised and spaced.
    assert config.parse("").wake == Pretrained("Hey Jarvis")
    assert config.parse('[talk]\nwake_word = "hey  mycroft"\n').wake == Pretrained("Hey Mycroft")
    assert config.parse('[talk]\nwake_word = "ALEXA"\n').wake == Pretrained("Alexa")
    # The user's own is said as written, spaced as one of openWakeWord's is, and its model's path may start at the home
    # directory.
    assert config.parse('[talk]\nwake_word = "Hey Computer"\nwake_word_model = "~/wake/hey_computer.onnx"\n').wake == Trained("Hey Computer", Path.home() / "wake" / "hey_computer.onnx")
    assert config.parse('[talk]\nwake_word = " Hey   Computer "\nwake_word_model = "/wake/hey_computer.onnx"\n').wake == Trained("Hey Computer", Path("/wake/hey_computer.onnx"))


@pytest.mark.outside
def test_the_file_names_the_audio_devices_each_auto_off_or_a_device_by_name(fake_claude: Path, tmp_path: Path) -> None:
    # Left out, each side follows the system's default, as auto does, however it is capitalised.
    assert config.parse("").devices == Choices(Auto(), Auto())
    assert config.parse('[audio]\ninput_device = "Auto"\noutput_device = "auto"\n').devices == Choices(Auto(), Auto())
    # Off opens no device, however it is capitalised; any other text is a device's name, as the system gives it, matched as written.
    assert config.parse('[audio]\ninput_device = "Off"\noutput_device = "off"\n').devices == Choices(Off(), Off())
    named = config.parse('[audio]\ninput_device = " MacBook Pro Microphone "\noutput_device = "Off Stage Monitor"\n').devices
    assert named == Choices(Named("MacBook Pro Microphone"), Named("Off Stage Monitor"))
    assert config.parse('[audio]\noutput_device = "MacBook Pro Speakers"\n').devices == Choices(Auto(), Named("MacBook Pro Speakers"))
    # And the voice is built on them.
    home = Home(tmp_path / ".hands")
    onboard(home.brain)
    brain = pin(home.brain, fake_claude)
    home.config.write_text('[audio]\ninput_device = "MacBook Pro Microphone"\n')
    assert run.configured_from(children_on(SYSTEM), home, config.load(home), preflight.preflight(home, brain, UNMANAGED, shell_without_hands_settings())).voice.devices == Choices(Named("MacBook Pro Microphone"), Auto())


@pytest.mark.parametrize(
    ("text", "said"),
    [
        ("[llm\n", "not TOML"),
        ('[llm]\nbackend = "local"\n', "backend 'local' is not one hands runs on: it runs on claude, the brain"),
        # A model API reached with a key is no harness hands drives.
        ('[llm]\nbackend = "anthropic"\n', "backend 'anthropic' is not one hands runs on: it runs on claude, the brain, a Claude Code of its own, never on a model API reached with a key"),
        ('[llm]\nbackend = "openai"\n', "backend 'openai' is not one hands runs on"),
        # A url for the brain would be ignored, its requests going through hands' proxy, so it is refused.
        ('[llm]\nbackend = "claude"\nurl = "http://localhost:8080"\n', "[llm] for claude has no 'url'; it takes backend, model"),
        ('[llm]\nurl = "https://api.anthropic.com"\n', "[llm] for claude has no 'url'; it takes backend, model"),
        # A key misspelled is refused, not a setting silently left at its default.
        ('[llm]\nmodle = "m"\n', "has no 'modle'"),
        ('voice = "charles"\n', "the file has no 'voice'"),
        ("[llm]\nmodel = 4\n", "model should be a non-empty string, got 4"),
        ('llm = "claude"\n', "llm should be a table"),
        ('[talk]\npersonality = "  "\n', "[talk] personality should be a non-empty string"),
        ('[talk]\nmood = "dry"\n', "[talk] has no 'mood'; it takes personality, wake_word, wake_word_model"),
        # A phrase openWakeWord has no model of could never be heard, so it is refused, naming those it has.
        ('[talk]\nwake_word = "Hey Computer"\n', "[talk] wake_word 'Hey Computer' is not one of openWakeWord's own: Hey Jarvis, Hey Mycroft, Hey Rhasspy, Alexa"),
        ('[talk]\nwake_word = " "\n', "[talk] wake_word should be a non-empty string"),
        ('[talk]\nwake_word_model = "/models/hey_computer.onnx"\n', "wake_word_model needs wake_word"),
        ('[talk]\nwake_word = "Hey Computer"\nwake_word_model = "models/hey_computer.onnx"\n', "is not a full path to an ONNX model"),
        ('[talk]\nwake_word = "Hey Computer"\nwake_word_model = "/models/hey_computer.tflite"\n', "is not a full path to an ONNX model"),
        ('[audio]\nmicrophone = "MacBook Pro Microphone"\n', "[audio] has no 'microphone'; it takes input_device, output_device"),
        ('[audio]\ninput_device = " "\n', "[audio] input_device should be a non-empty string"),
        ("[audio]\noutput_device = false\n", "[audio] output_device should be a non-empty string, got False"),
        ('audio = "off"\n', "audio should be a table"),
    ],
)
def test_a_file_that_does_not_parse_is_refused_saying_what_is_wrong(text: str, said: str) -> None:
    with pytest.raises(Rejected) as refused:
        config.parse(text)
    assert said in str(refused.value)


def test_a_refused_file_is_refused_naming_itself(tmp_path: Path) -> None:
    home = Home(tmp_path)
    home.config.write_text('[llm]\nbackend = "local"\n')
    with pytest.raises(Rejected, match=f"{home.config}: \\[llm\\] backend 'local'"):
        config.load(home)


def test_the_collector_is_an_http_address_spelled_without_a_trailing_slash() -> None:
    assert config.parse("").collector is None
    assert config.parse('[telemetry]\ncollector = "http://otel.example:4318/"\n').collector == "http://otel.example:4318"
    assert config.parse('[telemetry]\ncollector = "https://[::1]:4318/otel"\n').collector == "https://[::1]:4318/otel"
    # No scheme, a bracket left open, no host, a port out of range, the traces endpoint itself, a query, credentials
    # urllib would take for part of the host and the log would hold: none is a
    # base address a batch could ever be posted under, so each is refused as the file is read, not batch by batch.
    for unusable in ("otel.example:4317", "http://[::1:4318", "http://:4318", "http://otel.example:99999", "http://otel.example:4318/v1/traces", "http://otel.example:4318?x=1", "http://user:secret@otel.example:4318"):
        with pytest.raises(Rejected, match="is not an OTLP/HTTP collector's base address") as refused:
            config.parse(f'[telemetry]\ncollector = "{unusable}"\n')
        # A refused edit is a log line: the address it refused is not in it.
        assert "secret" not in str(refused.value)
    with pytest.raises(Rejected, match="\\[telemetry\\] has no 'endpoint'"):
        config.parse('[telemetry]\nendpoint = "http://otel.example:4318"\n')


def test_a_file_naming_a_transcription_server_or_a_whisper_model_is_refused() -> None:
    # Either would be a setting silently not applied: hands transcribes with its own Whisper, on one model.
    with pytest.raises(Rejected, match="the file has no 'transcription'"):
        config.parse('[transcription]\nurl = "http://127.0.0.1:8610/v1"\n')
    with pytest.raises(Rejected, match="the file has no 'whisper'"):
        config.parse('[whisper]\nmodel = "mlx-community/whisper-large-v3-turbo"\n')


@pytest.mark.outside
def test_a_brain_it_cannot_reach_stops_the_start_naming_what_is_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_claude: Path) -> None:
    monkeypatch.setenv("LOGGED_IN", "0")
    home = Home(tmp_path / ".hands")
    onboard(home.brain)
    brain = pin(home.brain, fake_claude)
    with pytest.raises(CannotStart, match="`hands login` gives it one"):
        run.configured_from(children_on(SYSTEM), home, config.load(home), preflight.preflight(home, brain, UNMANAGED, shell_without_hands_settings()))


def test_a_hands_setting_left_in_the_environment_stops_the_start_naming_the_file(tmp_path: Path) -> None:
    # The variables settings used to be: one still exported would run hands on the default backend, silently.
    home = Home(tmp_path)
    laid_out(home)
    environment = {"HANDS_HOME": str(tmp_path), "HANDS_LLM": "claude", "HANDS_WHISPER_MODEL": "w"}
    with pytest.raises(CannotStart, match=f"^HANDS_LLM, HANDS_WHISPER_MODEL set, .* put the setting in {home.config}"):
        run.configured_from(children_on(SYSTEM), home, config.load(home), preflight.preflight(home, LAID_OUT, UNMANAGED, environment))


@pytest.mark.outside
def test_claude_is_the_brain_on_the_login_in_hands_own_config_dir_with_no_key(fake_claude: Path, tmp_path: Path) -> None:
    home = Home(tmp_path / ".hands")
    onboard(home.brain)
    assert backend(children_on(SYSTEM), Claude(), home, os.environ, Runnable(PinnedClaude(fake_claude, pinned.PIN.version), None)) == ClaudeCodeBackend(model=ANTHROPIC_MODEL, config_dir=home.brain, account=Account("claude.ai", "brain@example.com"), brain=Runnable(PinnedClaude(fake_claude, pinned.PIN.version), None))
    assert backend(children_on(SYSTEM), Claude(model="claude-opus-5-5"), home, os.environ, Runnable(PinnedClaude(fake_claude, pinned.PIN.version), None)).model == "claude-opus-5-5"


@pytest.mark.outside
@pytest.mark.parametrize("model", CLAUDE_MODELS)
def test_claude_runs_on_each_model_on_offer(fake_claude: Path, tmp_path: Path, model: str) -> None:
    home = Home(tmp_path / ".hands")
    onboard(home.brain)
    assert backend(children_on(SYSTEM), Claude(model=model), home, os.environ, Runnable(PinnedClaude(fake_claude, pinned.PIN.version), None)).model == model


def test_a_claude_model_not_on_offer_is_refused_as_the_file_is_parsed() -> None:
    # A model by its spoken name, one that does not exist, and real ones hands does not offer: each is refused naming the
    # ones it does.
    for model in ("opus", "claude-opus-9", "claude-sonnet-5", "claude-haiku-4-5-20251001"):
        with pytest.raises(Rejected, match=f"^hands runs Claude on {', '.join(CLAUDE_MODELS)}, not {model}$"):
            config.parse(f'[llm]\nbackend = "claude"\nmodel = "{model}"\n')


def test_a_model_flag_outranks_the_files_on_the_files_backend(tmp_path: Path) -> None:
    home = Home(tmp_path)
    assert config.load(home, "claude-opus-5-5") == config.Settings(None, Config(llm=Claude(model="claude-opus-5-5")), "claude-opus-5-5")
    home.config.write_text('[llm]\nbackend = "claude"\nmodel = "claude-fable-5-1"\n[telemetry]\ncollector = "http://otel.example:4318"\n')
    assert config.load(home, "claude-opus-5-5").config == Config(llm=Claude(model="claude-opus-5-5"), collector="http://otel.example:4318")
    # The file's model, outranked, is never read: one hands no longer offers is no reason to refuse the run.
    home.config.write_text('[llm]\nbackend = "claude"\nmodel = "claude-sonnet-4"\n')
    assert config.load(home, "claude-opus-5-5").config == Config(llm=Claude(model="claude-opus-5-5"))


def test_a_model_flag_hands_cannot_run_on_is_refused_naming_the_flag_not_the_file(tmp_path: Path) -> None:
    home = Home(tmp_path)
    home.config.write_text('[llm]\nbackend = "claude"\n')
    with pytest.raises(Rejected) as refused:
        config.load(home, "opus")
    assert str(refused.value) == f"--model 'opus': hands runs Claude on {', '.join(CLAUDE_MODELS)}, not opus"


def test_a_blank_model_flag_is_refused_as_the_command_line_is_parsed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exited:
        cli.main(["--home", str(tmp_path), "run", "--model", "  "], machine=UNMANAGED, processes=processes())
    assert exited.value.code == 2
    assert f"argument --model: a model's id, such as {ANTHROPIC_MODEL}, not '  '" in capsys.readouterr().err


def test_a_start_says_the_model_flag_it_was_given_though_refused_before_the_settings_are_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from hands.voice import talkkey

    home = Home(tmp_path)
    monkeypatch.setattr(talkkey, "granted", lambda: False)
    monkeypatch.setattr(talkkey, "ask", lambda: None)
    for given in ((), ("--model", " claude-opus-5-5 ")):
        assert cli.main(["--home", str(home.root), "run", *given], machine=UNMANAGED, processes=processes()) == 1
    capsys.readouterr()
    starts = [line for line in map(json.loads, audit.tail(home.audit, 100)[0]) if line.get("event") == "hands.start"]
    assert [(start["outcome"], start["facts"]["model_flag"]) for start in starts] == [("failed", None), ("failed", "claude-opus-5-5")]


def test_a_model_flag_hands_cannot_run_on_refuses_the_start_at_the_door(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from hands.voice import talkkey

    home = Home(tmp_path / ".hands")
    monkeypatch.setattr(talkkey, "granted", lambda: True)
    assert cli.door(home, LAID_OUT, UNMANAGED, {}, "claude-opus-5-5", Start(False, SYSTEM)) == config.load(home, "claude-opus-5-5")
    with pytest.raises(CannotStart, match="^--model 'opus': "):
        cli.door(home, LAID_OUT, UNMANAGED, {}, "opus", Start(False, SYSTEM))


@pytest.mark.parametrize(("given", "loads", "carried"), [((), None, ()), (("--model", " -local-model "), "-local-model", ("--model=-local-model",))])
def test_a_restart_runs_again_on_the_model_flag_the_run_was_given_listening_at_the_level_its_desk_was_at(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, given: tuple[str, ...], loads: str | None, carried: tuple[str, ...]) -> None:
    home = Home(tmp_path)
    loaded_on: list[str | None] = []
    # What each run's desk does before the restart: the first engages; the second is restarted before its desk steps the
    # level it starts at, as a restart told while the models load is.
    stepped: list[tuple[Level, ...]] = [("engaged",), ()]
    # The level each run's owner held as it loaded: what nothing listens at yet, and what a restart then would carry.
    started: list[tuple[Level, Level]] = []

    def door(_home: Home, _pin: pinned.Pin, _machine: configdir.Machine, _environment: Mapping[str, str], model: str | None, _run_start: Start) -> config.Settings:
        # Only what the restart is given is weighed here: an id that begins with a dash is no model hands offers.
        return config.Settings(None, Config(), model)

    class Execed(Exception):
        pass

    def execv(argv: list[str]) -> None:
        raise Execed(argv)

    def hold(_clock: Clock, _home: Home, _record: object, *, restarted: bool) -> None:
        pass

    def loaded(clock: Clock, _processes: Processes, home: Home, pin: pinned.Pin, machine: configdir.Machine, environment: Mapping[str, str], settings: config.Settings, heart: heartbeat.Heart, record: audit.Record, degraded: Callable[[], tuple[heartbeat.Degradation, ...]], after_crash: bool, run_start: Start, listening: Listening, driving: Driving) -> cli.Run:
        loaded_on.append(settings.model)
        started.append((listening.level, driving.last_chosen))

        async def restarted(stop: Stop) -> Ended:
            for level in stepped.pop(0):
                driving.stepped(Desk(Loaded(Resting(level))))
            stop.tell(Signalled(RESTART_SIGNAL))
            return Ended(None, 0)

        return restarted

    monkeypatch.setattr(cli, "hold", hold)
    monkeypatch.setattr(cli, "door", door)
    monkeypatch.setattr(cli, "loaded", loaded)
    # The indicator each run shows is 42, and the one the run before showed is ended at once.
    def shown(_home: Home) -> int:
        return 42

    def kept(_indicator: int) -> None:
        pass

    def retired(_indicator: int, _clock: Clock) -> cli.Retired:
        return "ended"

    monkeypatch.setattr(cli, "start_indicator", shown)
    monkeypatch.setattr(cli, "reap", kept)
    monkeypatch.setattr(cli, "retire", retired)
    monkeypatch.setattr(cli, "execv", execv)
    with pytest.raises(Execed) as first:
        cli.main(["--home", str(home.root), "run", *given], machine=UNMANAGED, processes=processes())
    again = first.value.args[0][-3 - len(carried):]
    assert again == ["--restarted", "42", "--level=engaged", *carried]
    # The restart's own command line parses back to the run it restarts, an id that begins with a dash and all, its desk to
    # start at the level the run before's desk was at; restarted before it steps it, it carries that level on again.
    with pytest.raises(Execed) as second:
        cli.main(["--home", str(home.root), "run", *again], machine=UNMANAGED, processes=processes())
    assert second.value.args[0][-3 - len(carried):] == again
    assert (loaded_on, started) == ([loads, loads], [("off", "off"), ("off", "engaged")])
    lines = list(map(json.loads, audit.tail(home.audit, 100)[0]))
    # Each start says the --level it was given and the level its desk starts at, and each run the level its desk listened
    # at and the one last chosen.
    assert [(line["facts"]["level_flag"], line["facts"]["level_chosen"]) for line in lines if line.get("event") == "hands.start"] == [(None, "off"), ("engaged", "engaged")]
    assert [(line["facts"]["listening_level"], line["facts"]["level_chosen"]) for line in lines if line.get("event") == "hands.run"] == [("engaged", "engaged"), ("off", "engaged")]


def test_a_level_given_to_a_start_that_is_no_restart_is_refused_as_the_command_line_is_parsed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    for given, refused in ((("--level=engaged",), "--level is a restart's"), (("--restarted", "42", "--level=loud"), "argument --level: hands listens at off, wake word, engaged, not 'loud'")):
        with pytest.raises(SystemExit) as exited:
            cli.main(["--home", str(tmp_path), "run", *given], machine=UNMANAGED, processes=processes())
        assert exited.value.code == 2
        assert refused in capsys.readouterr().err


def test_a_restart_that_names_no_level_comes_back_at_off_as_one_exec_d_by_a_hands_from_before_the_level_was_carried() -> None:
    # The first restart onto this hands is exec'd by the one before it, which passed no --level: neither refused, which
    # would leave hands down, nor taken for a level given, since its start says level_flag None.
    assert cli.handed_over(argparse.ArgumentParser(), 42, None) == Handover(42, "off")
    assert cli.handed_over(argparse.ArgumentParser(), None, None) is None


async def test_an_edit_to_the_files_model_is_no_edit_while_a_model_flag_outranks_it_and_one_to_anything_else_is(tmp_path: Path) -> None:
    home = Home(tmp_path)
    laid_out(home)
    home.config.write_text('[llm]\nbackend = "claude"\n')
    recorded: list[Entry] = []
    running = config.load(home, "claude-opus-5-5")
    watching = asyncio.create_task(asyncio.wait_for(config.edited(home, LAID_OUT, UNMANAGED, {}, recorded.append, running, SYSTEM, period=0.01), 0.3))
    await asyncio.sleep(0.05)
    home.config.write_text('[llm]\nbackend = "claude"\nmodel = "claude-fable-5-1"\n')
    await asyncio.sleep(0.05)
    # Nor is one to a model hands does not offer: outranked, it is never read, so nothing is refused.
    home.config.write_text('[llm]\nbackend = "claude"\nmodel = "opus"\n')
    with pytest.raises(TimeoutError):
        await watching
    watching = asyncio.create_task(asyncio.wait_for(config.edited(home, LAID_OUT, UNMANAGED, {}, recorded.append, running, SYSTEM, period=0.01), 2.0))
    await asyncio.sleep(0.05)
    home.config.write_text('[llm]\nbackend = "claude"\nmodel = "claude-fable-5-1"\n[talk]\npersonality = "Dry."\n')
    assert await watching == SettingsEdited(path=str(home.config), refused=None)
    assert recorded == []






@pytest.mark.outside
def test_a_brain_with_no_login_stops_the_run_before_the_voice_loads_naming_the_command(monkeypatch: pytest.MonkeyPatch, fake_claude: Path) -> None:
    monkeypatch.setenv("LOGGED_IN", "0")
    with pytest.raises(Rejected, match="`hands login` gives it one"):
        backend(children_on(SYSTEM), Claude(), HOME, os.environ, Runnable(PinnedClaude(fake_claude, pinned.PIN.version), None))


def _starting(tmp_path: Path) -> tuple[Home, Sessions, heartbeat.Heart, run.VoiceConfig]:
    heart = heartbeat.Heart(tmp_path / "status.json", pid=4242, started_at=datetime.now(UTC), period=timedelta(seconds=0.01), kick_watchdog=lambda: None, clock=SYSTEM)
    sessions = new_sessions(clock=SYSTEM)
    config = run.VoiceConfig(llm=ClaudeCodeBackend(model=ANTHROPIC_MODEL, config_dir=tmp_path / "brain", account=Account("claude.ai", "brain@example.com"), brain=Runnable(PinnedClaude(pinned.PIN.claude(tmp_path / "brain"), pinned.PIN.version), None)), voice=voicechoice.DEFAULT, personality="Dry and wry.", wake=Trained("Hey Computer", Path("/wake/hey_computer.onnx")), devices=Choices(Named("MacBook Pro Microphone"), Auto()))
    return Home(tmp_path), sessions, heart, config


async def test_the_start_beats_while_the_configuration_is_read(tmp_path: Path) -> None:
    # A login check answered slowly is a start that is waiting, never one that looks stuck.
    home, sessions, heart, config = _starting(tmp_path)
    answered = threading.Event()

    def prompted(_checked: preflight.Preflight) -> run.Configured:
        answered.wait()
        return run.Configured(config, run.Settings(b"", Config(collector="http://otel.example:4318")))

    run_start = Start(restarted=False, clock=SYSTEM)
    # A collector already failing while hands starts is said by the start's beats, as it is once hands runs.
    exporting_down = heartbeat.Degradation("export", "can't export traces", "the collector has not taken all of its traces")
    starting = asyncio.create_task(start(lambda: run.configured(prompted, lambda _: None, processes(), home, LAID_OUT, UNMANAGED, {}, sessions, run_start), heart, sessions.live_count, lambda: (exporting_down,), Stop(), SYSTEM))
    # The prompt is answered only once the start has said "starting" three times while it waited.
    beats: set[datetime] = set()
    while len(beats) < 3:
        status = heartbeat.read(heart.path)
        if status is not None and status.pipeline == "starting":
            assert status.degraded == (exporting_down,)
            beats.add(status.written_at)
        await asyncio.sleep(0.005)
    answered.set()
    read = await starting
    assert read is not None and read.voice == config
    # The start's event says which file the settings came from, the collector they name, which server and
    # model the run reaches and on what account, the voice it speaks in, the personality it comes across in, and the
    # wake word it listens for, with the model of the user's own it is heard with, and the devices config.toml chooses.
    recorded: list[Entry] = []
    run_start.ended(recorded.append, None)
    [event] = recorded
    assert isinstance(event, WideEvent) and (event.event, event.outcome) == ("hands.start", "ok")
    # How big hands was as its start ended.
    assert isinstance(event.facts["footprint_mb"], float) and event.facts["footprint_mb"] > 0
    chosen = {name: event.facts[name] for name in ("settings", "collector", "backend", "model", "account", "voice", "personality", "wake_word", "wake_word_model", "input_device", "output_device")}
    assert chosen == {
        "settings": home.config, "collector": "http://otel.example:4318",
        "backend": "ClaudeCodeBackend", "model": ANTHROPIC_MODEL, "account": Account("claude.ai", "brain@example.com"), "voice": voicechoice.DEFAULT,
        "personality": "Dry and wry.", "wake_word": "Hey Computer", "wake_word_model": "/wake/hey_computer.onnx",
        "input_device": Named("MacBook Pro Microphone"), "output_device": Auto(),
    }


async def test_a_stop_during_the_configuration_read_ends_the_start(tmp_path: Path) -> None:
    home, sessions, heart, _config = _starting(tmp_path)
    never = threading.Event()

    def prompted(_checked: preflight.Preflight) -> run.Configured:
        never.wait()
        raise AssertionError("the prompt was never answered")

    stop = Stop()
    stop.tell(QuitKey())
    assert await start(lambda: run.configured(prompted, lambda _: None, processes(), home, LAID_OUT, UNMANAGED, {}, sessions, Start(restarted=False, clock=SYSTEM)), heart, sessions.live_count, lambda: (), stop, SYSTEM) is None
    never.set()


def test_a_refused_configuration_stops_the_start(tmp_path: Path) -> None:
    # Run as the CLI runs it, so only asyncio.run's caller sees it.
    home, sessions, heart, _config = _starting(tmp_path)

    def refused(_checked: preflight.Preflight) -> run.Configured:
        raise CannotStart("no key")

    surveyed: list[run.Configured | CannotStart] = []
    with pytest.raises(CannotStart, match="no key"):
        asyncio.run(start(lambda: run.configured(refused, surveyed.append, processes(), home, LAID_OUT, UNMANAGED, {}, sessions, Start(restarted=False, clock=SYSTEM)), heart, sessions.live_count, lambda: (), Stop(), SYSTEM))
    # Every other step is still said: the readiness check is given the refusal, as the backend's step.
    [refusal] = surveyed
    assert isinstance(refusal, CannotStart) and str(refusal) == "no key"


def test_a_restart_lists_its_running_sessions_before_the_configuration_is_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home, sessions, heart, config = _starting(tmp_path)
    order: list[str] = []

    async def swept(*_: object) -> None:
        order.append("sweep")

    def configure(_checked: preflight.Preflight) -> run.Configured:
        order.append("configure")
        return run.Configured(config, run.Settings(b"", Config()))

    monkeypatch.setattr(run, "sweep", swept)
    asyncio.run(start(lambda: run.configured(configure, lambda _: order.append("survey"), processes(), home, LAID_OUT, UNMANAGED, {}, sessions, Start(restarted=False, clock=SYSTEM)), heart, sessions.live_count, lambda: (), Stop(), SYSTEM))
    assert order == ["sweep", "configure", "survey"]


@pytest.mark.outside
async def test_a_start_says_where_the_run_listens_as_it_serves_each(monkeypatch: pytest.MonkeyPatch) -> None:
    # A unix socket path is capped near 104 bytes on macOS, so not under pytest's long tmp_path.
    root = Path(tempfile.mkdtemp(prefix="hands-"))
    home = Home(root)
    # The system picks the display's port, so a hands running on this machine keeps its own.
    monkeypatch.setattr(run, "DISPLAY_PORT", 0)
    heart = heartbeat.Heart(root / "status.json", pid=4242, started_at=datetime.now(UTC), period=timedelta(seconds=0.01), kick_watchdog=lambda: None, clock=SYSTEM)

    def refused(_checked: preflight.Preflight) -> run.Configured:
        raise CannotStart("no key")

    run_start = Start(restarted=False, clock=SYSTEM)
    recorded: list[Entry] = []
    levels = Levels("off")
    try:
        with pytest.raises(CannotStart, match="no key"), run_start.ending(recorded.append):
            await run.run(refused, lambda _: None, processes(), FakeTerminals(), FakeFritters(), FakeScreen(FrontUnread("no screen in tests")), home, heart, recorded.append, lambda: (), Stop(), False, {}, LAID_OUT, UNMANAGED, run_start, config.OwnModel(home, LAID_OUT, UNMANAGED, {}, _NO_FILE), SYSTEM, levels, levels)
    finally:
        shutil.rmtree(root)
    # Refused at its settings, after every server was up: the start says where each listened.
    [event] = [entry for entry in recorded if isinstance(entry, WideEvent) and entry.event == "hands.start"]
    proxy, display = event.facts["proxy"], event.facts["display"]
    assert (event.outcome, event.facts["hooks"], event.facts["upstream"], event.facts["tap"]) == ("failed", home.socket, UPSTREAM, home.wire)
    assert isinstance(proxy, str) and re.fullmatch(r"http://127\.0\.0\.1:[1-9]\d*", proxy)
    # The port the display was bound on, not the 0 it was asked for.
    assert isinstance(display, str) and re.fullmatch(rf"http://127\.0\.0\.1:[1-9]\d*{DISPLAY_PATH}", display)


def test_a_start_refused_says_why_in_the_audit_log_and_in_hands_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    # Started from a launcher whose terminal nobody watches, the reason is still on record: the run's first read of its
    # configuration, through the CLI's start, refusing a setting left in the environment.
    from hands.voice import talkkey

    home = Home(tmp_path)
    laid_out(home)
    monkeypatch.setattr(talkkey, "granted", lambda: True)
    # No indicator is shown.
    def unshown(_home: Home) -> int:
        return 0

    def kept(*_: object) -> None:
        pass

    monkeypatch.setattr(cli, "start_indicator", unshown)
    monkeypatch.setattr(cli, "reap", kept)
    # The brain is logged in, so the setting is the one refusal: the login is asked beside every refusal of the start.
    def logged_in(*_: object) -> Account:
        return Account("claude.ai", "brain@example.com")

    monkeypatch.setattr(process, "logged_in", logged_in)
    # Exported in the process, so the refusal is of what `main` hands down to the run, not of an environment the test makes up.
    monkeypatch.setenv("HANDS_LLM", "claude")

    def loaded(clock: Clock, processes: Processes, home: Home, pin: pinned.Pin, machine: configdir.Machine, environment: Mapping[str, str], settings: config.Settings, heart: heartbeat.Heart, record: audit.Record, _degraded: Callable[[], tuple[heartbeat.Degradation, ...]], _after_crash: bool, run_start: Start, _listening: Listening, _driving: Driving) -> cli.Run:
        sessions = new_sessions(clock=SYSTEM, record=record)

        async def refused(stop: Stop) -> Ended:
            configure = partial(run.configured_from, children_on(clock), home, settings)
            await start(lambda: run.configured(configure, lambda _: None, processes, home, pin, UNMANAGED, environment, sessions, run_start), heart, sessions.live_count, lambda: (), stop, clock)
            raise AssertionError("a start with HANDS_LLM set went on")

        return refused

    monkeypatch.setattr(cli, "loaded", loaded)
    assert cli.main(["--home", str(home.root), "run"], pin=LAID_OUT, machine=UNMANAGED, processes=processes()) == 1
    reason = f"HANDS_LLM set, and hands reads no setting from the environment: unset HANDS_LLM, and put the setting in {home.config}; then run hands again"
    assert capsys.readouterr().err == f"hands: {reason}\n"
    # The start's one event is failed with the reason, and says which run it was.
    [refused] = [line for line in map(json.loads, audit.tail(home.audit, 100)[0]) if line.get("event") == "hands.start"]
    assert (refused["level"], refused["outcome"], refused["error"]) == ("error", "failed", f"CannotStart: {reason}")
    assert (refused["facts"]["pid"], refused["facts"]["restarted"], refused["facts"]["after_crash"]) == (os.getpid(), False, False)
    assert cli.main(["--home", str(home.root), "status"], machine=UNMANAGED, processes=processes()) == 1
    assert capsys.readouterr().out == f"hands refused to start 0s ago: {reason}\n"


def test_a_start_refused_as_a_task_fails_is_still_said_and_the_failure_raised(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from hands.voice import talkkey

    home = Home(tmp_path)
    monkeypatch.setattr(talkkey, "granted", lambda: True)

    def unshown(_home: Home) -> int:
        return 0

    def kept(*_: object) -> None:
        pass

    monkeypatch.setattr(cli, "start_indicator", unshown)
    monkeypatch.setattr(cli, "reap", kept)

    def loaded(*_: object) -> cli.Run:
        async def refused(stop: Stop) -> Ended:
            stop.tell(Failing("the namer", OSError("the disk is gone")))
            raise CannotStart("no key")

        return refused

    monkeypatch.setattr(cli, "loaded", loaded)
    with pytest.RaisesGroup(pytest.RaisesExc(OSError, match="the disk is gone")):
        cli.main(["--home", str(home.root), "run"], machine=UNMANAGED, processes=processes())
    assert capsys.readouterr().err == "hands: no key\n"
    assert cli.main(["--home", str(home.root), "status"], machine=UNMANAGED, processes=processes()) == 1
    assert capsys.readouterr().out == "hands refused to start 0s ago: no key\n"


def test_a_start_that_fails_before_its_launch_is_still_one_event(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Past the door, before the run's launch: the indicator cannot be started.
    from hands.voice import talkkey

    home = Home(tmp_path)
    monkeypatch.setattr(talkkey, "granted", lambda: True)

    def unshown(_home: Home) -> int:
        raise OSError("no indicator")

    monkeypatch.setattr(cli, "start_indicator", unshown)
    with pytest.raises(OSError, match="no indicator"):
        cli.main(["--home", str(home.root), "run"], machine=UNMANAGED, processes=processes())
    [failed] = [line for line in map(json.loads, audit.tail(home.audit, 100)[0]) if line.get("event") == "hands.start"]
    assert (failed["outcome"], failed["error"]) == ("failed", "OSError: no indicator")


@pytest.mark.outside
def test_a_settings_file_it_cannot_read_refuses_the_start_at_the_door_and_a_restarts_in_its_heartbeat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    # Before its first heartbeat a run holds none: the one there, here a crash's, is left for the next run to read.
    from hands.voice import talkkey

    home = Home(tmp_path)
    home.config.write_text('[llm]\nbackend = "nonesuch"\n')
    monkeypatch.setattr(talkkey, "granted", lambda: True)
    # The crashed run's pid, which no process holds now.
    gone, table = 4343, processes()
    crashed = heartbeat.Heart(home.status, gone, datetime.now(UTC), heartbeat.HEARTBEAT, kick_watchdog=lambda: None, clock=SYSTEM)
    crashed.beat("running", None, 0, listening=False, degraded=())
    assert cli.main(["--home", str(home.root), "run"], machine=UNMANAGED, processes=table) == 1
    reason = capsys.readouterr().err.removeprefix("hands: ").rstrip("\n")
    assert str(home.config) in reason
    # Refused at the door, before the settings name a collector: the hold's event and the start's are on the log alone.
    assert [(line["event"], line["error"]) for line in map(json.loads, audit.tail(home.audit, 10)[0])] == [("hands.hold", None), ("hands.start", f"CannotStart: {reason}")]
    assert cli.crashed_before(SYSTEM, table, home)
    # A restart's run holds the heartbeat from the outset, the run before it having beat starting under its pid.
    assert cli.main(["--home", str(home.root), "run", "--restarted", "0"], machine=UNMANAGED, processes=table) == 1
    capsys.readouterr()
    assert cli.main(["--home", str(home.root), "status"], machine=UNMANAGED, processes=table) == 1
    assert capsys.readouterr().out == f"hands refused to start 0s ago: {reason}\n"


@pytest.mark.outside
def test_the_voice_is_charles_until_one_is_chosen_and_the_chosen_one_after_a_restart(fake_claude: Path, tmp_path: Path) -> None:
    """Charles by the name the installed pocket_tts resolves itself, until the user chooses another, which the next run
    is built in.

    A voice state is a cache computed by one network and noise under another, and the loader does not check; so the
    default is a catalogue name, which the package maps to the state primed by the weights it loads. The membership
    check here is the one get_state_for_audio_prompt makes, so a dependency bump that drops the name fails in its commit.
    """
    from pocket_tts.utils.utils import _ORIGINS_OF_PREDEFINED_VOICES  # pyright: ignore[reportPrivateUsage]

    logged_in = shell_without_hands_settings()
    home = Home(tmp_path / ".hands")
    onboard(home.brain)
    brain = pin(home.brain, fake_claude)
    assert run.configured_from(children_on(SYSTEM), home, config.load(home), preflight.preflight(home, brain, UNMANAGED, logged_in)).voice.voice == "charles"
    assert "charles" in _ORIGINS_OF_PREDEFINED_VOICES
    voicechoice.keep(home, voicechoice.parse_voice("Bill Boerst"))
    assert run.configured_from(children_on(SYSTEM), home, config.load(home), preflight.preflight(home, brain, UNMANAGED, logged_in)).voice.voice == "bill_boerst"
    # A kept name the installed pocket_tts no longer has stops the start, naming the file to fix.
    home.voice.write_text("zed\n")
    with pytest.raises(CannotStart, match=f"{home.voice} says 'zed'"):
        run.configured_from(children_on(SYSTEM), home, config.load(home), preflight.preflight(home, brain, UNMANAGED, logged_in))
    # One it cannot read stops it the same way, naming the file.
    home.voice.unlink()
    home.voice.mkdir()
    with pytest.raises(CannotStart, match=str(home.voice)):
        run.configured_from(children_on(SYSTEM), home, config.load(home), preflight.preflight(home, brain, UNMANAGED, logged_in))


# The settings of a run started with no file: every default.
_NO_FILE = config.Settings(None, Config())


async def _edited_within(home: Home, recorded: list[Entry], seconds: float = 0.5) -> SettingsEdited | None:
    try:
        return await asyncio.wait_for(config.edited(home, LAID_OUT, UNMANAGED, {}, recorded.append, config.load(home), SYSTEM, period=0.01), seconds)
    except TimeoutError:
        return None


async def test_an_edit_that_parses_is_heard_and_said(tmp_path: Path) -> None:
    home = Home(tmp_path)
    laid_out(home)
    recorded: list[Entry] = []
    watching = asyncio.create_task(_edited_within(home, recorded))
    await asyncio.sleep(0.05)
    home.config.write_text('[llm]\nmodel = "claude-opus-5-5"\n')
    assert await watching == SettingsEdited(path=str(home.config), refused=None)
    assert recorded == []


async def test_a_file_saved_unchanged_is_no_edit(tmp_path: Path) -> None:
    home = Home(tmp_path)
    home.config.write_text('[llm]\nmodel = "claude-opus-5-5"\n')
    recorded: list[Entry] = []
    watching = asyncio.create_task(_edited_within(home, recorded, seconds=0.2))
    await asyncio.sleep(0.05)
    home.config.write_text('[llm]\nmodel = "claude-opus-5-5"\n')
    assert await watching is None
    assert recorded == []


async def test_an_edit_that_does_not_parse_is_said_and_outlived_until_one_that_does(tmp_path: Path) -> None:
    home = Home(tmp_path)
    laid_out(home)
    recorded: list[Entry] = []
    watching = asyncio.create_task(_edited_within(home, recorded, seconds=2.0))
    await asyncio.sleep(0.05)
    home.config.write_text('[llm]\nbackend = "local"\n')
    while not recorded:
        await asyncio.sleep(0.01)
    assert not watching.done()
    assert recorded == [SettingsEdited(path=str(home.config), refused=f"{home.config}: [llm] backend 'local' is not one hands runs on: it runs on claude, the brain, a Claude Code of its own, never on a model API reached with a key")]
    assert audit.level(recorded[0]) == "error"
    home.config.write_text('[llm]\nmodel = "claude-fable-5-1"\n')
    assert await watching == SettingsEdited(path=str(home.config), refused=None)
    assert len(recorded) == 1


async def test_settings_removed_are_an_edit_back_to_the_defaults(tmp_path: Path) -> None:
    home = Home(tmp_path)
    laid_out(home)
    home.config.write_text('[llm]\nmodel = "claude-opus-5-5"\n')
    recorded: list[Entry] = []
    watching = asyncio.create_task(_edited_within(home, recorded))
    await asyncio.sleep(0.05)
    home.config.unlink()
    assert await watching == SettingsEdited(path=str(home.config), refused=None)
    assert recorded == []


async def test_an_edit_undone_back_to_the_settings_the_run_is_on_is_no_edit(tmp_path: Path) -> None:
    home = Home(tmp_path)
    home.config.write_text('[llm]\nmodel = "claude-opus-5-5"\n')
    recorded: list[Entry] = []
    watching = asyncio.create_task(_edited_within(home, recorded, seconds=0.5))
    await asyncio.sleep(0.05)
    home.config.write_text('[llm]\nbackend = "local"\n')
    while not recorded:
        await asyncio.sleep(0.01)
    home.config.write_text('[llm]\nmodel = "claude-opus-5-5"\n')
    assert await watching is None
    assert [type(entry) for entry in recorded] == [SettingsEdited] and audit.level(recorded[0]) == "error"


async def test_a_file_that_cannot_be_read_is_a_refused_edit_and_outlived(tmp_path: Path) -> None:
    home = Home(tmp_path)
    laid_out(home)
    home.config.write_text('[llm]\nmodel = "claude-opus-5-5"\n')
    recorded: list[Entry] = []
    watching = asyncio.create_task(_edited_within(home, recorded, seconds=2.0))
    await asyncio.sleep(0.05)
    home.config.unlink()
    home.config.mkdir()
    while not recorded:
        await asyncio.sleep(0.01)
    assert not watching.done()
    assert recorded == [SettingsEdited(path=str(home.config), refused=f"{home.config} could not be read: [Errno 21] Is a directory: '{home.config}'")]
    home.config.rmdir()
    home.config.write_text('[llm]\nmodel = "claude-fable-5-1"\n')
    assert await watching == SettingsEdited(path=str(home.config), refused=None)


async def test_a_save_written_in_two_steps_is_weighed_once_whole(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = Home(tmp_path)
    laid_out(home)
    # What each poll after the run's read reads, as an editor that truncates and then writes leaves it: one poll lands between the two.
    reads = iter([b'[llm]\nbackend = "cla', b'[llm]\nmodel = "claude-opus-5-5"\n', b'[llm]\nmodel = "claude-opus-5-5"\n', b'[llm]\nmodel = "claude-opus-5-5"\n'])

    def held(_home: Home) -> bytes | None:
        return next(reads)

    monkeypatch.setattr(config, "_held", held)
    recorded: list[Entry] = []
    assert await config.edited(home, LAID_OUT, UNMANAGED, {}, recorded.append, _NO_FILE, SYSTEM, period=0) == SettingsEdited(path=str(home.config), refused=None)
    assert recorded == []


async def test_a_comment_added_is_no_edit(tmp_path: Path) -> None:
    home = Home(tmp_path)
    home.config.write_text('[llm]\nmodel = "claude-opus-5-5"\n')
    recorded: list[Entry] = []
    watching = asyncio.create_task(_edited_within(home, recorded, seconds=0.2))
    await asyncio.sleep(0.05)
    home.config.write_text('# opus for now\n[llm]\nmodel = "claude-opus-5-5"\n')
    assert await watching is None
    assert recorded == []


async def test_a_sample_line_edited_is_no_edit_since_it_is_read_where_it_is_said(tmp_path: Path) -> None:
    home = Home(tmp_path)
    home.config.write_text('[llm]\nmodel = "claude-opus-5-5"\n')
    recorded: list[Entry] = []
    watching = asyncio.create_task(_edited_within(home, recorded, seconds=0.2))
    await asyncio.sleep(0.05)
    config.SampleLines(home, config.load(home)).keep(voicechoice.Voice("mary"), "Mary here.")
    assert await watching is None
    assert recorded == []
    assert config.SampleLines(home, config.load(home)).read() == {voicechoice.Voice("mary"): "Mary here."}


def test_a_sample_line_kept_is_an_edit_to_the_file_that_keeps_every_other_line_as_written(tmp_path: Path) -> None:
    home = Home(tmp_path)
    written = '# hands, by hand\n[llm]\nmodel = "claude-opus-5-5"  # for now\n\n[sample_lines]\nalba = "Alba here."  # hers\n'
    home.config.write_text(written)
    lines = config.SampleLines(home, config.load(home))
    alba, library = voicechoice.Voice("alba"), voicechoice.LibraryVoice("vctk/p225_023_enhanced")
    lines.keep(library, "  From the library.  ")
    assert home.config.read_text() == written + '"vctk/p225_023_enhanced" = "From the library."\n'
    lines.keep(alba, "Alba, again.")
    assert lines.read() == {alba: "Alba, again.", library: "From the library."}
    assert home.config.read_text().startswith('# hands, by hand\n[llm]\nmodel = "claude-opus-5-5"  # for now\n\n[sample_lines]\nalba = "Alba, again."')
    lines.keep(alba, "")
    lines.keep(library, " ")
    # The last line taken out takes its table with it: the file is the user's as it was before any line.
    assert home.config.read_text() == '# hands, by hand\n[llm]\nmodel = "claude-opus-5-5"  # for now\n\n'
    assert lines.read() == {}


def test_sample_lines_are_none_with_no_file_and_one_that_is_not_text_refuses_the_file() -> None:
    assert config.parse('[sample_lines]\nmary = "Mary here."\n') == Config()
    with pytest.raises(Rejected, match=r"\[sample_lines\] mary should be a non-empty string"):
        config.parse("[sample_lines]\nmary = 3\n")
    with pytest.raises(Rejected, match="sample_lines should be a table"):
        config.parse('sample_lines = "Mary here."\n')


def test_a_sample_line_key_that_names_no_voice_refuses_the_file_naming_the_key_and_so_does_one_voice_named_twice() -> None:
    with pytest.raises(Rejected, match=r"\[sample_lines\] hands has no voice called 'Zed!'"):
        config.parse('[sample_lines]\nmary = "Mary here."\n"Zed!" = "Who?"\n')
    with pytest.raises(Rejected, match=r"\[sample_lines\] names one voice twice, spelled two ways: 'bill_boerst' and 'Bill Boerst'"):
        config.parse('[sample_lines]\nbill_boerst = "One."\n"Bill Boerst" = "Two."\n')


def test_a_line_kept_for_a_voice_replaces_the_line_the_file_keeps_for_it_under_whatever_spelling_it_holds(tmp_path: Path) -> None:
    home = Home(tmp_path)
    bill = voicechoice.Voice("bill_boerst")
    home.config.write_text('[sample_lines]\n"Bill Boerst" = "Hand-written."\nalba = "Alba here."\n')
    lines = config.SampleLines(home, config.load(home))
    assert lines.keep(bill, "Saved from the page.") == voicechoice.SampleLine("Bill Boerst", "Saved from the page.")
    assert home.config.read_text() == '[sample_lines]\n"Bill Boerst" = "Saved from the page."\nalba = "Alba here."\n'
    assert lines.read() == {bill: "Saved from the page.", voicechoice.Voice("alba"): "Alba here."}
    # A blank line takes the line out from under that spelling too, so no key is left for the voice.
    assert lines.keep(bill, "") == voicechoice.SampleLine("Bill Boerst", "")
    assert home.config.read_text() == '[sample_lines]\nalba = "Alba here."\n'


def test_a_line_is_never_kept_under_a_key_a_start_would_refuse(tmp_path: Path) -> None:
    home = Home(tmp_path)
    home.config.write_text('[sample_lines]\nalba = "Alba here."\n')
    # A voice no crossing made: the file is read back as a start reads it before it is written.
    with pytest.raises(Rejected, match=r"\[sample_lines\] hands has no voice called 'Zed!'"):
        config.SampleLines(home, config.load(home)).keep(voicechoice.Voice("Zed!"), "Who?")
    assert home.config.read_text() == '[sample_lines]\nalba = "Alba here."\n'


def test_a_sample_line_is_never_kept_in_a_file_a_start_would_refuse(tmp_path: Path) -> None:
    home = Home(tmp_path)
    home.config.write_text('[llm]\nmodel = "gpt-9"\n')
    with pytest.raises(Rejected, match="not gpt-9"):
        config.SampleLines(home, _NO_FILE).keep(voicechoice.Voice("mary"), "Mary here.")
    assert home.config.read_text() == '[llm]\nmodel = "gpt-9"\n'
    # A start on --model never reads the file's model, so a line is kept in a file whose model only it outranks.
    config.SampleLines(home, config.load(home, "claude-opus-5-5")).keep(voicechoice.Voice("mary"), "Mary here.")
    assert config.SampleLines(home, _NO_FILE).read() == {voicechoice.Voice("mary"): "Mary here."}
    assert config.SampleLines(Home(tmp_path / "none"), _NO_FILE).read() == {}


async def test_an_edit_saved_over_while_it_is_weighed_is_not_taken(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = Home(tmp_path)
    laid_out(home)
    recorded: list[Entry] = []
    weighed: list[bytes] = []

    def overwritten(weighing: Home, pin: pinned.Pin, machine: configdir.Machine, environment: Mapping[str, str]) -> preflight.Preflight:
        weighed.append(weighing.config.read_bytes())
        # The first edit is saved over with a typo while the home is weighed for it.
        if weighed == [b'[llm]\nmodel = "claude-opus-5-5"\n']:
            home.config.write_text('[llm]\nbackend = "claud"\n')
        return preflight.preflight(weighing, pin, machine, environment)

    monkeypatch.setattr(config, "preflight", overwritten)
    # Polled, not spun: a watch with no period would hold the loop against the thread this test waits on.
    watching = asyncio.create_task(config.edited(home, LAID_OUT, UNMANAGED, {}, recorded.append, config.load(home), SYSTEM, period=0.01))
    try:
        home.config.write_text('[llm]\nmodel = "claude-opus-5-5"\n')
        # Moved by each weigh, the hop to a thread that a loaded machine is slowest to take.
        await patiently(lambda: bool(recorded) or watching.done(), lambda: len(weighed))
        # Had the edit saved over been taken, the watch would have ended on it before ever weighing the typo.
        assert not watching.done(), watching.result()
        assert [type(entry) for entry in recorded] == [SettingsEdited] and "'claud'" in str(recorded[0])
        # And it is still watching: the next edit is the one it takes.
        home.config.write_text('[llm]\nmodel = "claude-fable-5-1"\n')
        await patiently(watching.done, lambda: len(weighed))
        assert watching.result() == SettingsEdited(path=str(home.config), refused=None)
    finally:
        watching.cancel()


async def test_an_edit_saved_over_while_weighed_and_back_again_is_taken(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = Home(tmp_path)
    laid_out(home)
    edit, other = b'[llm]\nmodel = "claude-opus-5-5"\n', b'[llm]\nmodel = "claude-fable-5-1"\n'
    # Seen, settled, saved over as its backend is checked, and back before the next poll.
    reads = iter([edit, edit, other, edit, edit])

    def held(_home: Home) -> bytes | None:
        return next(reads)

    monkeypatch.setattr(config, "_held", held)
    recorded: list[Entry] = []
    assert await config.edited(home, LAID_OUT, UNMANAGED, {}, recorded.append, _NO_FILE, SYSTEM, period=0) == SettingsEdited(path=str(home.config), refused=None)
    assert recorded == []
