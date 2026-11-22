"""The start's preflight: every refusal a start makes on its home's files, said at once by the start and by each weigh of
an edit or a model chosen by voice, starting no Claude Code."""

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from conftest import children_on, located, new_sessions, startable
from hands.brain import pinned
from hands.virtualclock import VirtualClock
from hands.daemon import cli, config, readiness, run
from hands.daemon.preflight import Preflight, Refused, Startable, preflight
from hands.daemon.starting import CannotStart, Start
from hands.sessions import wrapper
from hands.sessions.audit import Entry, SettingsEdited
from hands.sessions.home import Home
from hands.sessions.payload import Rejected
from hands.sessions.wide import WideEvent

EDIT = b'[llm]\nmodel = "claude-opus-5-5"\n'
KEPT_OUT = {"syncClaudeAiSkills": False, "syncClaudeAiPlugins": False}


class Outlived(Exception):
    """The scripted file's reads ran out with the watch still running."""


def _reads(script: list[bytes | None], at: dict[int, Callable[[], object]] | None = None) -> Callable[[Home], bytes | None]:
    """config.toml as each poll reads it, in turn, doing what `at` names at that read first; Outlived once they are spent."""
    reads = iter(enumerate(script))

    def held(_home: Home) -> bytes | None:
        for index, read in reads:
            (at or {}).get(index, lambda: None)()
            return read
        raise Outlived

    return held


def _stale_shim(home: Home) -> None:
    # As an older hands wrote it: without what this one writes into it.
    home.bin.mkdir(parents=True, exist_ok=True)
    home.shim.write_text(wrapper.shim_script(home.fritter, home.wire).replace("export DISABLE_AUTOUPDATER=1\n", ""))


def _unpinned(home: Home, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pinned, "claude", located(home.brain / "pinned-claude" / pinned.VERSION / "claude"))


def _untrusted(home: Home, _monkeypatch: pytest.MonkeyPatch) -> None:
    # Onboarded, and the directory above the brain's, where its working directory is, not trusted.
    (home.brain / ".claude.json").write_text(json.dumps({"hasCompletedOnboarding": True, "projects": {str(home.root.resolve()): {"hasTrustDialogAccepted": False}}}))


def _unreadable(path: Path) -> None:
    path.chmod(0)


def _dangling_fritter(home: Home) -> None:
    # A link to nothing: there, and nothing to read.
    home.bin.mkdir()
    home.fritter.symlink_to(home.root / "nowhere")


def _unreadable_shim(home: Home) -> None:
    home.bin.mkdir(parents=True, exist_ok=True)
    home.shim.write_text(wrapper.shim_script(home.fritter, home.wire))
    _unreadable(home.shim)


# One row a refusal: what breaks the home, and what the refusal says.
FAULTS: list[tuple[str, Callable[[Home, pytest.MonkeyPatch], object], str]] = [
    ("pin missing", _unpinned, f"Claude Code {pinned.VERSION}, the brain's, is not installed at "),
    ("pin wrong", lambda home, _: (home.root / "pinned" / "claude").write_text("#!/bin/sh\nexit 0\n"), f"of Claude Code {pinned.VERSION}, the brain's: run `hands install-brain` to fetch it again"),
    ("hands built without its fritter", lambda home, monkeypatch: monkeypatch.setattr(wrapper, "PACKAGED", home.root / "package" / "bin" / "fritter"), "/package/bin/fritter: install hands again"),
    ("stale fritter copy", lambda home, _: (home.bin.mkdir(), home.fritter.write_bytes(b"#!/bin/sh\n")), "is not the fritter this hands carries"),
    ("fritter copy unreadable", lambda home, _: _dangling_fritter(home), "/fritter'): run `hands install-fritter` to copy it again"),
    ("stale shim", lambda home, _: _stale_shim(home), "is not the shim this hands writes, so sessions started through it may not do what this hands asks of them: run `hands install-fritter`"),
    ("shim unreadable", lambda home, _: _unreadable_shim(home), "/claude'): run `hands install-fritter` to write it again"),
    ("first screens unanswered", lambda home, _: (home.brain / ".claude.json").write_text("{}"), "first screens (its onboarding unfinished; "),
    ("its working directory untrusted", _untrusted, "/cwd untrusted): `hands login` answers them"),
    ("first screens unreadable", lambda home, _: _unreadable(home.brain / ".claude.json"), ".claude.json'): correct it, or remove it and run `hands login` to make it again"),
    ("a sync not kept out", lambda home, _: (home.brain / "settings.json").write_text('{"syncClaudeAiPlugins": false}'), "the brain would load its account's syncClaudeAiSkills: "),
    # Left out, each switch is on.
    ("no settings.json", lambda home, _: (home.brain / "settings.json").unlink(), 'its account\'s syncClaudeAiSkills and syncClaudeAiPlugins: set "syncClaudeAiSkills": false and "syncClaudeAiPlugins": false in'),
    ("a bad defaultMode", lambda home, _: (home.brain / "settings.json").write_text(json.dumps({**KEPT_OUT, "permissions": {"defaultMode": "acceptedits"}})), 'a defaultMode Claude Code does not take, "acceptedits": name one of acceptEdits, auto, bypassPermissions, default, dontAsk, plan, manual in'),
    ("a kept voice hands has no voice for", lambda home, _: home.voice.write_text("zed\n"), "says 'zed', which names no voice hands has"),
]


@pytest.mark.parametrize(("fault", "said"), [(fault, said) for _, fault, said in FAULTS], ids=[name for name, _, _ in FAULTS])
async def test_each_refusal_of_the_start_refuses_an_edit_and_a_voice_choice_which_the_watch_outlives(fault: Callable[[Home, pytest.MonkeyPatch], object], said: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = Home(tmp_path)
    mark = startable(tmp_path, monkeypatch)
    assert isinstance(preflight(home).verdict, Startable)
    fault(home, monkeypatch)
    match preflight(home).verdict:
        case Refused(refusals=refusals):
            assert [refusal for refusal in refusals if said in refusal] != [], refusals
        case passed:
            raise AssertionError(f"{passed} passed the preflight")
    # A model chosen by voice is refused with it, and nothing is written.
    with pytest.raises(Rejected) as refused:
        config.OwnModel(home, config.load(home)).weigh("claude-opus-5-5")
    assert said in str(refused.value)
    assert not home.config.exists()
    # An edit is refused with it, said once, and the watch is still weighing the edit when the reads run out.
    running = config.load(home)
    monkeypatch.setattr(config, "_held", _reads([EDIT] * 4))
    recorded: list[Entry] = []
    with pytest.raises(Outlived):
        await config.edited(home, recorded.append, running, VirtualClock(), period=0)
    [entry] = recorded
    assert isinstance(entry, SettingsEdited) and entry.refused is not None and said in entry.refused
    assert not mark.exists()


def _faulted(home: Home, monkeypatch: pytest.MonkeyPatch) -> tuple[str, ...]:
    """A home with several faults at once, an unreadable settings.json among them; what each refusal says."""
    _unpinned(home, monkeypatch)
    (home.brain / "settings.json").write_text("{")
    _stale_shim(home)
    home.voice.write_text("zed\n")
    return (f"is not installed at {home.brain}", f"the brain's settings could not be read (", "is not the shim this hands writes", "says 'zed'")


def test_a_start_refused_on_its_home_names_every_refusal_at_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = Home(tmp_path)
    startable(tmp_path, monkeypatch)
    each = _faulted(home, monkeypatch)
    checked = preflight(home)
    with pytest.raises(CannotStart) as refused:
        run.configured_from(children_on(VirtualClock()), home, config.load(home), os.environ, checked)
    assert [said for said in each if said not in str(refused.value)] == []
    # Said once, after them all: the fix of each is made before hands is run again.
    assert str(refused.value).endswith("; then run hands again")
    # An unreadable settings.json is its own refusal, never also a first screen unanswered.
    assert "first screens" not in str(refused.value)
    assert checked.facts() == {"fritter": "absent", "shim": "stale"}


async def test_the_start_records_what_its_preflight_found_before_its_configuration_is_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = Home(tmp_path)
    startable(tmp_path, monkeypatch)
    _dangling_fritter(home)
    run_start = Start(restarted=False, clock=VirtualClock())

    def configure(checked: Preflight) -> run.Configured:
        raise CannotStart("; ".join(cast(Refused, checked.verdict).refusals))

    with pytest.raises(CannotStart):
        await run.configured(configure, lambda _: None, home, new_sessions(clock=VirtualClock()), run_start)
    events: list[WideEvent] = []
    run_start.ended(events.append, None)
    assert [(event.facts["fritter"], event.facts["shim"]) for event in events] == [("unreadable", "not-a-shim")]


def test_an_unreadable_record_of_api_key_answers_is_said_beside_an_unreadable_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = Home(tmp_path)
    startable(tmp_path, monkeypatch)
    (home.brain / "settings.json").write_text(json.dumps({**KEPT_OUT, "env": "x"}))
    state = home.brain / ".claude.json"
    state.write_text(json.dumps({**json.loads(state.read_text()), "customApiKeyResponses": []}))
    match preflight(home).verdict:
        case Refused(refusals=refusals):
            assert any("an env hands cannot read" in refusal for refusal in refusals), refusals
            assert any(f"{state} unreadable" in refusal for refusal in refusals), refusals
        case passed:
            raise AssertionError(f"{passed} passed the preflight")


async def test_an_edit_refused_on_its_home_names_every_refusal_at_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = Home(tmp_path)
    mark = startable(tmp_path, monkeypatch)
    each = _faulted(home, monkeypatch)
    running = config.load(home)
    monkeypatch.setattr(config, "_held", _reads([EDIT] * 3))
    recorded: list[Entry] = []
    with pytest.raises(Outlived):
        await config.edited(home, recorded.append, running, VirtualClock(), period=0)
    [entry] = recorded
    assert isinstance(entry, SettingsEdited) and entry.refused is not None
    assert [said for said in each if said not in entry.refused] == []
    assert not mark.exists()


def test_the_door_passes_a_home_whose_brain_is_refused_so_the_start_says_it_beside_the_survey(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from hands.voice import talkkey

    home = Home(tmp_path)
    monkeypatch.setattr(talkkey, "granted", lambda: True)
    _stale_shim(home)
    assert cli.door(home, None) == config.load(home)


async def test_an_edit_refused_on_its_home_is_taken_once_the_home_is_mended_its_bytes_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = Home(tmp_path)
    mark = startable(tmp_path, monkeypatch)
    settings = home.brain / "settings.json"
    settings.write_text('{"syncClaudeAiPlugins": false}')
    # Seen, then weighed at each poll after while its bytes stand; the home is mended before the sixth read.
    running = config.load(home)
    monkeypatch.setattr(config, "_held", _reads([EDIT] * 8, {5: lambda: settings.write_text(json.dumps(KEPT_OUT))}))
    recorded: list[Entry] = []
    assert await config.edited(home, recorded.append, running, VirtualClock(), period=0) == SettingsEdited(path=str(home.config), refused=None)
    assert [entry.refused for entry in recorded if isinstance(entry, SettingsEdited)] == [f'the brain would load its account\'s syncClaudeAiSkills: set "syncClaudeAiSkills": false and "syncClaudeAiPlugins": false in {settings}']
    assert not mark.exists()


async def test_an_edit_whose_refusals_change_while_its_bytes_stand_is_said_again_with_those_standing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = Home(tmp_path)
    startable(tmp_path, monkeypatch)
    _stale_shim(home)
    home.voice.write_text("zed\n")
    # Refused on both; the voice mended before the fifth read, then broken again before the seventh.
    running = config.load(home)
    monkeypatch.setattr(config, "_held", _reads([EDIT] * 8, {4: lambda: home.voice.unlink(), 6: lambda: home.voice.write_text("zed\n")}))
    recorded: list[Entry] = []
    with pytest.raises(Outlived):
        await config.edited(home, recorded.append, running, VirtualClock(), period=0)
    said = [(("is not the shim" in refused), ("says 'zed'" in refused)) for entry in recorded if isinstance(entry, SettingsEdited) and (refused := entry.refused) is not None]
    assert said == [(True, True), (True, False), (True, True)]


async def test_an_edit_refused_reverted_and_saved_again_is_refused_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = Home(tmp_path)
    startable(tmp_path, monkeypatch)
    _stale_shim(home)
    # Refused, weighed again, reverted to the run's own bytes, and saved again: refused again.
    running = config.load(home)
    monkeypatch.setattr(config, "_held", _reads([EDIT, EDIT, EDIT, None, None, EDIT, EDIT, EDIT]))
    recorded: list[Entry] = []
    with pytest.raises(Outlived):
        await config.edited(home, recorded.append, running, VirtualClock(), period=0)
    assert len([entry for entry in recorded if isinstance(entry, SettingsEdited) and "is not the shim" in (entry.refused or "")]) == 2


async def test_an_edit_and_a_voice_choice_are_weighed_without_starting_a_claude_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = Home(tmp_path)
    mark = startable(tmp_path, monkeypatch)
    held = config._held  # pyright: ignore[reportPrivateUsage]
    running = config.load(home)
    monkeypatch.setattr(config, "_held", _reads([EDIT, EDIT, EDIT]))
    assert await config.edited(home, lambda _entry: None, running, VirtualClock(), period=0) == SettingsEdited(path=str(home.config), refused=None)
    monkeypatch.setattr(config, "_held", held)
    config.OwnModel(home, config.load(home)).weigh("claude-opus-5-5")()
    assert config.load(home).config.llm.model == "claude-opus-5-5"
    assert not mark.exists()


def test_a_home_with_the_fritter_and_shim_this_hands_installs_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = Home(tmp_path)
    startable(tmp_path, monkeypatch)
    wrapper.install(home)
    checked = preflight(home)
    assert isinstance(checked.verdict, Startable)
    assert checked.facts() == {"fritter": "current", "shim": "current"}


def _api_key(home: Home, key: str, answered: list[str]) -> None:
    (home.brain / "settings.json").write_text(json.dumps({**KEPT_OUT, "env": {"ANTHROPIC_API_KEY": key}}))
    state = home.brain / ".claude.json"
    state.write_text(json.dumps({**json.loads(state.read_text()), "customApiKeyResponses": {"approved": answered}}))


@pytest.mark.parametrize(("key", "answered", "asked"), [
    # Claude Code asks of no blank key, and answers a key by its last 20 characters once trimmed (docs/claude-code-behavior.md).
    ("", [], False),
    ("  ", [], True),
    ("sk-ant-" + "k" * 30 + "\n", ["k" * 20], False),
    ("sk-ant-" + "k" * 30, ["sk-ant-" + "k" * 13], True),
], ids=["blank", "whitespace", "answered once trimmed", "unanswered"])
def test_an_api_key_the_brain_settings_set_is_unanswered_as_claude_code_reads_it(key: str, answered: list[str], asked: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = Home(tmp_path)
    startable(tmp_path, monkeypatch)
    _api_key(home, key, answered)
    match preflight(home).verdict:
        case Refused(refusals=refusals):
            assert asked and any("the API key" in refusal and "unanswered" in refusal for refusal in refusals), refusals
        case Startable():
            assert not asked


async def test_settings_that_do_not_parse_hide_none_of_the_homes_refusals_from_the_start_the_survey_or_an_edit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from hands.voice import talkkey

    home = Home(tmp_path)
    mark = startable(tmp_path, monkeypatch)
    _stale_shim(home)
    running = config.load(home)
    home.config.write_text("[llm\n")
    monkeypatch.setattr(talkkey, "granted", lambda: True)
    each = (str(home.config), "is not the shim this hands writes")
    # The start, refused at its door, before the settings name where the run is recorded.
    with pytest.raises(CannotStart) as refused:
        cli.door(home, None)
    assert [said for said in each if said not in str(refused.value)] == []
    # The survey.
    match readiness.configured(children_on(VirtualClock()), home, os.environ):
        case readiness.Missing(detail=detail):
            assert [said for said in each if said not in detail] == []
        case found:
            raise AssertionError(f"the survey found {found}")
    # An edit to them, said once, the watch outliving it.
    monkeypatch.setattr(config, "_held", _reads([b"[llm\n"] * 3))
    recorded: list[Entry] = []
    with pytest.raises(Outlived):
        await config.edited(home, recorded.append, running, VirtualClock(), period=0)
    [entry] = recorded
    assert isinstance(entry, SettingsEdited) and entry.refused is not None
    assert [said for said in each if said not in entry.refused] == [], entry.refused
    assert not mark.exists()
