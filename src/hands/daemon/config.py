"""The settings file, `<home>/config.toml`: read once, when a run starts, into a frozen Config; an edit to it while the
run runs starts the run again, on the file as edited.

    [llm]
    backend = "claude"           # the brain, a Claude Code of hands' own: the default, and the one there is
    model = "claude-sonnet-5-5"  # one of CLAUDE_MODELS

    [telemetry]
    collector = "http://otel.example:4318"   # an OpenTelemetry collector's OTLP/HTTP address; none by default

    [talk]
    personality = "Dry and wry, with a bit of wit."   # how hands comes across, in the user's words; hands' own by default
    wake_word = "Hey Mycroft"     # what the desk listens for at its wake word level: Hey Jarvis (the default), Hey Mycroft, Hey Rhasspy, or Alexa
    wake_word_model = "~/wake/hey_computer.onnx"      # or a model of your own, trained with openWakeWord on wake_word, said as written

    [audio]
    input_device = "MacBook Pro Microphone"   # auto (the default) follows the system's default; a device's name keeps to it; off opens none
    output_device = "MacBook Pro Speakers"    # the same for the speaker

    [sample_lines]
    charles = "Charles here. A session is waiting for you."   # what a voice says when it is heard, by its name; hands' own line by default
    "vctk/p225_023_enhanced" = "And this is a voice from Kyutai's library."   # a library voice by its path there

A file left out, or a key, is the default. Secrets are not settings: the brain's login is in its own config directory. The voice is not either: the user chooses it by voice,
and it changes while the daemon runs (hands.voice.voices). The sample lines are settings that change while it runs as
well: they are read from the file each time a voice is heard (SampleLines), so an edit to them alone is no edit. The model is a setting the user may also choose by voice:
that choice is an edit to this file (OwnModel), taken as any edit is. `hands run --model` names one that outranks the
file's for as long as that run runs, its restarts included.

[LAW:no-mode-explosion] the settings cap: every key names a variant or a value, and there are no flags. A key that
would be a flag is a variant with a real alternative, or it does not exist.
"""

import asyncio
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit

import tomlkit

from hands.clock import Clock
from hands.daemon.preflight import Refused, Startable, beside, preflight
from hands.sessions.audit import Record, SettingsEdited
from hands.sessions.files import Unread, replace_whole
from hands.sessions.home import Home
from hands.sessions.payload import Rejected
from hands.threads import off_loop
from hands.voice.devicechoice import Auto, Choice, Choices, Named, Off
from hands.voice.wakemodels import PRETRAINED
from hands.voice.wakeword import Pretrained, Trained, Word

# How late an edit to the file is heard.
EDIT_SECONDS = 1.0

ANTHROPIC_MODEL = "claude-sonnet-5-5"
# The models hands runs on: one Haiku, Sonnet, Opus, and Fable each. A model outside them is refused where the file is
# parsed, since by voice a turn that fails on it could never choose another.
CLAUDE_MODELS = ("claude-haiku-4-5-20251001", ANTHROPIC_MODEL, "claude-opus-5-5", "claude-fable-5-1")


@dataclass(frozen=True)
class Claude:
    """Claude through a slim Claude Code of hands' own, on its own login; its requests go through hands' proxy, so it has no URL."""

    model: str = ANTHROPIC_MODEL


@dataclass(frozen=True)
class Config:
    """`collector` is the OpenTelemetry collector each wide event is also sent to, over OTLP/HTTP; None sends them nowhere
    but the audit log. `personality` is how hands comes across, in the user's own words; None is hands' own. `wake` is
    the wake word the desk listens for at its wake word level. `devices` are the audio devices the desk hears and
    speaks on."""

    llm: Claude = Claude()
    collector: str | None = None
    personality: str | None = None
    wake: Word = Pretrained()
    devices: Choices = Choices()


@dataclass(frozen=True)
class Settings:
    """The settings a run starts on: the file's bytes as it read them, None where there was no file, and what they hold
    with `model` in place of the file's, the one `hands run --model` named, None where the run was given none."""

    held: bytes | None
    config: Config
    model: str | None = None

    def path(self, home: Home) -> Path | None:
        """The file the settings were read from, None where they are every default."""
        return None if self.held is None else home.config


def load(home: Home, model: str | None = None) -> Settings:
    """The settings `home` holds, on `model` where one is given, read once as a run starts; raises Rejected naming the
    file and what is wrong in it, or --model and why hands cannot run on it."""
    held = _readable(home, _held(home))
    return Settings(held, _settings(home, held, model), model)


async def edited(home: Home, record: Record, running: Settings, clock: Clock, period: float = EDIT_SECONDS) -> SettingsEdited:
    """The edit, once the file holds settings other than `running`, the ones the run was started on: written, rewritten,
    or removed, and a start on it would refuse none of the home's files.

    [LAW:no-ambient-temporal-coupling] weighed against the bytes the run read, not a read of its own, so an edit is never
    missed, however late after the run's read the watch begins. An edit is weighed once its bytes read the same on two
    polls, so a save an editor writes in two steps is weighed whole, and it is taken only while the file still holds it
    once weighed. One that does not parse, or that the start's preflight refuses, is said and outlived, the run keeping
    the settings it has, and weighed again at each poll while its bytes stand, so one refused on the home is taken once
    the home is mended: said whenever what refuses it changes, every refusal standing then. One whose settings are the
    run's, a comment or a revert, is no edit. A login lost since the start is not weighed (promptctl-phone-app-9d9y.h4s):
    an edit made then is restarted on, and the restart refused.
    """
    seen: bytes | Unread | None = running.held
    # The bytes whose verdict holds whatever the home holds: the run's own, or settings that are the run's.
    settled = seen
    said: tuple[str, ...] = ()
    while True:
        await clock.sleep(period)
        if (now := _held(home)) != seen:
            seen, said = now, ()
            continue
        if now == settled:
            continue
        try:
            # [LAW:one-source-of-truth] weighed on the run's --model, as the run's own settings were, so an edit to the
            # file's model alone is no edit while one outranks it.
            settings = _settings(home, _readable(home, now), running.model)
        except Rejected as error:
            refusals: tuple[str, ...] = await off_loop(partial(beside, home, error), "weighing a settings edit")
        else:
            if settings == running.config:
                settled = now
                continue
            match (await off_loop(partial(preflight, home), "weighing a settings edit")).verdict:
                # [LAW:escape-local-minima] exception: a login lost since the start passes here, until
                # promptctl-phone-app-9d9y.h4s (blocked by promptctl-brain-uoz0.gi7) weighs it with no Claude Code started.
                case Startable():
                    if _held(home) == now:
                        return SettingsEdited(path=str(home.config), refused=None)
                    # Saved over while weighed: no verdict, so these bytes are weighed again should they come back.
                    continue
                case Refused(refusals=refusals):
                    pass
        # [LAW:nothing-unseen] a refusal that clears and comes back is a change, and said again.
        if refusals != said:
            record(SettingsEdited(path=str(home.config), refused="; ".join(refusals)))
            said = refusals


class OwnModel:
    """hands' own model: the one the run is on, and the one the user chooses for hands by voice.

    [LAW:one-source-of-truth] config.toml is the one place the model is chosen, by hand or by voice, and `hands run
    --model` the one thing that outranks it, for that run alone: a choice by voice is refused while it does. A choice by
    voice is an edit to the file, its every other line kept as written, and the run takes it as it takes any edit: by
    starting again on it once `edited` has weighed it.
    """

    def __init__(self, home: Home, running: Settings) -> None:
        self._home = home
        self._running = running
        # [LAW:single-enforcer] the one lock over a switch of the model, taken by `hands.voice.modelswitch.switch_model`.
        self.changing = asyncio.Lock()

    def running(self) -> str:
        """The model the run is on: the one --model named, or the file's as it started."""
        return self._running.config.llm.model

    def weigh(self, model: str) -> Callable[[], None]:
        """The edit that runs hands on `model`, written by calling it. Raises Rejected where the file could not take it, or
        the start's preflight refuses the home: weighed here as `edited` weighs an edit by hand, so a choice refused on
        the home's files is refused while the user is there to hear why. A login lost since the start is not weighed
        (promptctl-phone-app-9d9y.h4s): a choice made then is said, written, and its restart refused. Hashes a pinned
        claude this process has not hashed since it changed, so it is called off the loop."""
        if self._running.model is not None:
            # [LAW:one-source-of-truth] the file's model is not the run's while --model outranks it, so an edit to it would
            # change nothing until hands is started without the flag.
            raise Rejected(f"hands was started with --model {self._running.model}, which it keeps over {self._home.config} until it is started without it")
        model = model.strip()
        # The file is taken whole first, so the edit is only ever made to settings a start would take: on the run's own
        # --model, as the start weighs it, and one that does not parse is refused beside the home's other refusals.
        try:
            held = _readable(self._home, _held(self._home))
            on_file = _settings(self._home, held, self._running.model)
        except Rejected as error:
            raise Rejected("; ".join(beside(self._home, error))) from error
        edited = _with_model(held, model)
        chosen = _settings(self._home, edited, self._running.model)
        if chosen == self._running.config:
            raise Rejected(f"hands runs on {model} already")
        if chosen == on_file:
            # The file names it, and the run is not on it: an edit `edited` has yet to take, or one it refused, which the
            # same bytes written again would not change. Which of the two is not known here, so the reason names both.
            raise Rejected(f"{self._home.config} names {model} already and hands is not on it yet: an edit it is about to take, or one it refused when saved, which `hands log` says and `hands restart` starts on")
        # [LAW:escape-local-minima] exception: a login lost since the start passes here, until promptctl-phone-app-9d9y.h4s.
        match preflight(self._home).verdict:
            case Refused(refusals=refusals):
                raise Rejected("; ".join(refusals))
            case Startable():
                return partial(_keep, self._home, held, edited)


def _with_model(held: bytes | None, model: str) -> bytes:
    # Only read once `_settings` has taken the file, so it is TOML and [llm], where there is one, is a table.
    document = tomlkit.parse(b"" if held is None else held)
    llm = cast(dict[str, object], document.setdefault("llm", tomlkit.table()))
    llm["model"] = model
    return tomlkit.dumps(document).encode()


def _keep(home: Home, held: bytes | None, edited: bytes) -> None:
    # [LAW:one-source-of-truth] one writer at a time: a save by hand since the choice was weighed is the user's, and is
    # never written over.
    if _held(home) != held:
        raise Rejected(f"{home.config} was saved after the choice was made, so the choice was not written over it")
    # Through a link to the file, as dotfiles keep one, so the link stays and the file it names takes the edit.
    target = home.config.resolve()
    replace_whole(target, edited.decode(), 0o644 if held is None else target.stat().st_mode & 0o777)


def _held(home: Home) -> bytes | Unread | None:
    # The bytes, not the mtime: a save that changes nothing, or a touch, is no edit.
    try:
        return home.config.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as error:
        return Unread(str(error))


def _readable(home: Home, held: bytes | Unread | None) -> bytes | None:
    match held:
        case Unread(reason=reason):
            raise Rejected(f"{home.config} could not be read: {reason}")
        case _:
            return held


def _settings(home: Home, held: bytes | None, model: str | None) -> Config:
    # [LAW:single-enforcer] the start and the watch weigh the file's bytes here, the same bytes each compared, on the
    # same --model.
    try:
        return parse("" if held is None else held.decode(), model)
    except UnicodeDecodeError as error:
        raise Rejected(f"{home.config} could not be read: {error}") from error
    except ModelFlagRejected:
        raise
    except Rejected as error:
        raise Rejected(f"{home.config}: {error}") from error


class ModelFlagRejected(Rejected):
    """A model --model named that hands cannot run on: the flag's fault, never the file's, so it is said without it."""


def parse(text: str, model: str | None = None) -> Config:
    """The settings `text` holds, on `model`, the one --model names, where one is given: it outranks the file's, which
    is then never read, so one hands does not offer refuses nothing and an edit to it alone is no edit."""
    # [LAW:parse-dont-validate] the one crossing from the file: past it, a setting is a variant that holds exactly its fields.
    try:
        top = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise Rejected(f"not TOML: {error}") from error
    _known(top, "the file", ("llm", "telemetry", "talk", "audio", "sample_lines"))
    # Checked as the file is read, so a line hands could not say is refused at the start or the edit; read where it is said.
    _sample_lines(top)
    telemetry = _table(top, "telemetry")
    _known(telemetry, "[telemetry]", ("collector",))
    talk = _table(top, "talk")
    _known(talk, "[talk]", ("personality", "wake_word", "wake_word_model"))
    audio = _table(top, "audio")
    _known(audio, "[audio]", ("input_device", "output_device"))
    devices = Choices(input=_device(audio, "input_device"), output=_device(audio, "output_device"))
    return Config(llm=_llm(_table(top, "llm"), model), collector=_collector(telemetry), personality=_personality(talk), wake=_wake(talk), devices=devices)


def _sample_lines(top: Mapping[str, object]) -> dict[str, str]:
    """The line each voice named says when it is heard, by the voice's name as written; which voice a name is, is the
    voices' to say (hands.voice.voices.parse_lines)."""
    table = _table(top, "sample_lines")
    return {name: _text(table, "[sample_lines]", name, "") for name in table}


class SampleLines:
    """The lines voices say when they are heard, as config.toml holds them now, and the edit that sets one.

    [LAW:one-source-of-truth] the file is the one place a line is kept, by hand or from the gallery page; nothing holds
    a copy, so each hearing reads it afresh and no edit to it starts the run again (`edited` weighs Config, which holds
    none). An edit from the page is an edit to the file, its every other line kept as written, as OwnModel's is.
    """

    def __init__(self, home: Home, running: Settings) -> None:
        self._home = home
        self._running = running

    def read(self) -> dict[str, str]:
        held = _readable(self._home, _held(self._home))
        try:
            return _sample_lines(tomllib.loads("" if held is None else held.decode()))
        except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
            raise Rejected(f"{self._home.config} could not be read: {error}") from error
        except Rejected as error:
            raise Rejected(f"{self._home.config}: {error}") from error

    def keep(self, voice: str, line: str) -> None:
        """`voice` says `line` from its next hearing on; a blank line takes its line out, so it says hands' own again."""
        held = _readable(self._home, _held(self._home))
        # The file is taken whole first, on the run's --model as a start takes it, so an edit is only ever made to
        # settings a start would take.
        _settings(self._home, held, self._running.model)
        _keep(self._home, held, _with_line(held, voice, line.strip()))


def _with_line(held: bytes | None, voice: str, line: str) -> bytes:
    # Only read once `_settings` has taken the file, so it is TOML and [sample_lines], where there is one, is a table.
    document = tomlkit.parse(b"" if held is None else held)
    lines = cast(dict[str, object], document.setdefault("sample_lines", tomlkit.table()))
    if line:
        # Set in place, so a line edited keeps its place in the table.
        lines[voice] = line
    else:
        lines.pop(voice, None)
    if not lines:
        # A table emptied by the last line taken out goes with it, so the file reads as one never given a line.
        del document["sample_lines"]
    return tomlkit.dumps(document).encode()


def _device(table: Mapping[str, object], key: str) -> Choice:
    """auto, the system's default as it moves; off, no device; or any other name, the device the system calls that, matched
    as written. Whether it is there is the transport's to find each time it opens."""
    said = _text(table, "[audio]", key, "auto")
    match said.casefold():
        case "auto":
            return Auto()
        case "off":
            return Off()
        case _:
            return Named(said)


def _wake(table: Mapping[str, object]) -> Word:
    """The wake word: one of openWakeWord's own, matched however it is capitalised, or, with a model, the user's own,
    said as written. Whether the model is there is the desk's to find as it starts (hands.daemon.run), as it is for
    openWakeWord's own."""
    if "wake_word" not in table:
        if "wake_word_model" in table:
            raise Rejected("[talk] wake_word_model needs wake_word, the phrase the model was trained on, which hands tells you to say")
        return Pretrained()
    said = " ".join(_text(table, "[talk]", "wake_word", "").split())
    if "wake_word_model" in table:
        model = Path(_text(table, "[talk]", "wake_word_model", "")).expanduser()
        if not model.is_absolute() or model.suffix != ".onnx":
            raise Rejected(f"[talk] wake_word_model {str(model)!r} is not a full path to an ONNX model, such as ~/wake/hey_computer.onnx")
        return Trained(phrase=said, model=model)
    for phrase in PRETRAINED:
        if said.casefold() == phrase.casefold():
            return Pretrained(phrase)
    raise Rejected(f"[talk] wake_word {said!r} is not one of openWakeWord's own: {', '.join(PRETRAINED)}; a wake word of your own takes wake_word_model too, the model you trained on it")


def _personality(table: Mapping[str, object]) -> str | None:
    return _text(table, "[talk]", "personality", "") if "personality" in table else None


def _collector(table: Mapping[str, object]) -> str | None:
    if "collector" not in table:
        return None
    return _base(_text(table, "[telemetry]", "collector", ""), "[telemetry] collector", "an OTLP/HTTP collector", "http://host:4318", "/v1/traces")


def _base(url: str, where: str, server: str, example: str, appended: str) -> str:
    """[LAW:parse-dont-validate] a server's base address, spelled one way from here on: the base `appended` is appended
    to, with no trailing slash."""
    url = url.rstrip("/")
    # Not echoed: a refused edit is a log line, and the address may hold credentials.
    unusable = Rejected(f"{where} is not {server}'s base address, as {example}, to which hands appends {appended}, with no credentials in it")
    try:
        parts = urlsplit(url)
        # Read for what it raises: a port that is not a number, or out of range.
        parts.port
    except ValueError as error:
        raise unusable from error
    if parts.scheme not in ("http", "https") or not parts.hostname or parts.username is not None or parts.password is not None or parts.query or parts.fragment or parts.path.endswith(appended):
        raise unusable
    return url


def _llm(table: Mapping[str, object], model: str | None) -> Claude:
    # [LAW:one-type-per-behavior] hands reaches its model through a harness it drives, as it drives Claude Code: the brain.
    # Another harness joins as another backend here; a model API reached with a key is none.
    match backend := _text(table, "[llm]", "backend", "claude"):
        case "claude":
            # A url would be ignored, its requests going through hands' proxy to Anthropic's API, so it is refused.
            _known(table, "[llm] for claude", ("backend", "model"))
            return _on(Claude(), table, model)
        case _:
            raise Rejected(f"[llm] backend {backend!r} is not one hands runs on: it runs on claude, the brain, a Claude Code of its own, never on a model API reached with a key")


def _on(llm: Claude, table: Mapping[str, object], flag: str | None) -> Claude:
    """`llm` on the model `table` names, its default where it names none, or on `flag`, the --model that outranks it."""
    if flag is None:
        return replace(llm, model=_offered(_text(table, "[llm]", "model", llm.model)))
    try:
        return replace(llm, model=_offered(flag))
    except Rejected as error:
        raise ModelFlagRejected(f"--model {flag!r}: {error}") from error


def _offered(model: str) -> str:
    # [LAW:single-enforcer] the one place a model is refused that hands does not offer, whether the file, --model, or a
    # choice by voice named it.
    if model not in CLAUDE_MODELS:
        raise Rejected(f"hands runs Claude on {', '.join(CLAUDE_MODELS)}, not {model}")
    return model


def _table(top: Mapping[str, object], name: str) -> Mapping[str, object]:
    match value := top.get(name, {}):
        case dict():
            return cast(dict[str, object], value)
        case _:
            raise Rejected(f"{name} should be a table, got {type(value).__name__}")


def _known(table: Mapping[str, object], where: str, keys: tuple[str, ...]) -> None:
    # [LAW:no-silent-failure] a key misspelled, or one its variant has no use for, would be a setting silently not applied.
    if unknown := sorted(set(table) - set(keys)):
        raise Rejected(f"{where} has no {', '.join(map(repr, unknown))}; it takes {', '.join(keys)}")


def _text(table: Mapping[str, object], where: str, key: str, default: str) -> str:
    match value := table.get(key, default):
        # A blank value is no value: left in, it would be a server or a model named "".
        case str() if value.strip():
            return value.strip()
        case _:
            raise Rejected(f"{where} {key} should be a non-empty string, got {value!r}")
