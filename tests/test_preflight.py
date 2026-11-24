"""The start's preflight: every refusal a start makes on its home's files, said at once by the start and by each weigh of
an edit or a model chosen by voice, starting no Claude Code."""

import json
import platform
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from conftest import children_on, laid_out, new_sessions, shell_without_hands_settings
from fixtures import pins
from fixtures.pins import LAID_OUT
from hands.brain import process
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


def _unpinned(home: Home, _monkeypatch: pytest.MonkeyPatch) -> None:
    LAID_OUT.claude(home.brain).unlink()


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
    ("pin missing", _unpinned, f"Claude Code {LAID_OUT.version}, the brain's, is not installed at "),
    ("pin wrong", lambda home, _: LAID_OUT.claude(home.brain).write_text("#!/bin/sh\nexit 0\n"), f"of Claude Code {LAID_OUT.version}, the brain's: run `hands install-brain` to fetch it again"),
    ("hands built without its fritter", lambda home, monkeypatch: monkeypatch.setattr(wrapper, "PACKAGED", home.root / "package" / "bin" / "fritter"), "/package/bin/fritter: install hands again"),
    ("stale fritter copy", lambda home, _: (home.bin.mkdir(), home.fritter.write_bytes(b"#!/bin/sh\n")), "is not the fritter this hands carries"),
    ("fritter copy unreadable", lambda home, _: _dangling_fritter(home), "/fritter'): run `hands install-fritter` to copy it again"),
    ("stale shim", lambda home, _: _stale_shim(home), "is not the shim this hands writes, so sessions started through it may not do what this hands asks of them: run `hands install-fritter`"),
    ("shim unreadable", lambda home, _: _unreadable_shim(home), "/claude'): run `hands install-fritter` to write it again"),
    ("first screens unanswered", lambda home, _: (home.brain / ".claude.json").write_text("{}"), "first screens (its onboarding unfinished; "),
    ("its working directory untrusted", _untrusted, "/cwd untrusted): `hands login` answers them"),
    ("first screens unreadable", lambda home, _: _unreadable(home.brain / ".claude.json"), ".claude.json unreadable: [Errno 13] Permission denied"),
    ("a sync not kept out", lambda home, _: (home.brain / "settings.json").write_text('{"syncClaudeAiPlugins": false}'), "the brain would load its account's syncClaudeAiSkills: "),
    # Left out, each switch is on.
    ("no settings.json", lambda home, _: (home.brain / "settings.json").unlink(), 'its account\'s syncClaudeAiSkills and syncClaudeAiPlugins: set "syncClaudeAiSkills": false and "syncClaudeAiPlugins": false in'),
    ("a bad defaultMode", lambda home, _: (home.brain / "settings.json").write_text(json.dumps({**KEPT_OUT, "permissions": {"defaultMode": "acceptedits"}})), 'a defaultMode Claude Code does not take, "acceptedits": name one of acceptEdits, auto, bypassPermissions, default, dontAsk, plan, manual in'),
    ("a kept voice hands has no voice for", lambda home, _: home.voice.write_text("zed\n"), "says 'zed', which names no voice hands has"),
]


@pytest.mark.parametrize(("fault", "said"), [(fault, said) for _, fault, said in FAULTS], ids=[name for name, _, _ in FAULTS])
async def test_each_refusal_of_the_start_refuses_an_edit_and_a_voice_choice_which_the_watch_outlives(fault: Callable[[Home, pytest.MonkeyPatch], object], said: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = Home(tmp_path)
    mark = laid_out(home)
    assert isinstance(preflight(home, LAID_OUT, {}).verdict, Startable)
    fault(home, monkeypatch)
    match preflight(home, LAID_OUT, {}).verdict:
        case Refused(refusals=refusals):
            assert [refusal for refusal in refusals if said in refusal] != [], refusals
        case passed:
            raise AssertionError(f"{passed} passed the preflight")
    # A model chosen by voice is refused with it, and nothing is written.
    with pytest.raises(Rejected) as refused:
        config.OwnModel(home, LAID_OUT, {}, config.load(home)).weigh("claude-opus-5-5")
    assert said in str(refused.value)
    assert not home.config.exists()
    # An edit is refused with it, said once, and the watch is still weighing the edit when the reads run out.
    running = config.load(home)
    monkeypatch.setattr(config, "_held", _reads([EDIT] * 4))
    recorded: list[Entry] = []
    with pytest.raises(Outlived):
        await config.edited(home, LAID_OUT, {}, recorded.append, running, VirtualClock(), period=0)
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
    laid_out(home)
    each = _faulted(home, monkeypatch)
    checked = preflight(home, LAID_OUT, {})
    with pytest.raises(CannotStart) as refused:
        run.configured_from(children_on(VirtualClock()), home, config.load(home), checked)
    assert [said for said in each if said not in str(refused.value)] == []
    # Said once, after them all: the fix of each is made before hands is run again.
    assert str(refused.value).endswith("; then run hands again")
    # An unreadable settings.json is its own refusal, never also a first screen unanswered.
    assert "first screens" not in str(refused.value)
    assert checked.facts() == {"fritter": "absent", "shim": "stale", "claude_code": LAID_OUT.version, "hands_settings": "none"}


async def test_the_start_records_what_its_preflight_found_before_its_configuration_is_read(tmp_path: Path) -> None:
    home = Home(tmp_path)
    laid_out(home)
    _dangling_fritter(home)
    run_start = Start(restarted=False, clock=VirtualClock())

    def configure(checked: Preflight) -> run.Configured:
        raise CannotStart("; ".join(cast(Refused, checked.verdict).refusals))

    with pytest.raises(CannotStart):
        await run.configured(configure, lambda _: None, home, LAID_OUT, {}, new_sessions(clock=VirtualClock()), run_start)
    events: list[WideEvent] = []
    run_start.ended(events.append, None)
    assert [(event.facts["fritter"], event.facts["shim"]) for event in events] == [("unreadable", "not-a-shim")]


STRAY = {"HANDS_HOME": "/elsewhere", "HANDS_LLM": "claude", "HANDS_WHISPER_MODEL": "w"}
SAID_STRAY = "HANDS_LLM, HANDS_WHISPER_MODEL set, and hands reads no setting from the environment: unset HANDS_LLM, HANDS_WHISPER_MODEL, and put the setting in "


def test_a_start_refused_on_a_hands_setting_in_the_environment_says_it_beside_every_refusal_on_the_homes_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = Home(tmp_path)
    laid_out(home)
    each = _faulted(home, monkeypatch)
    checked = preflight(home, LAID_OUT, STRAY)
    # The start's event names the settings it refused, the one a person asks "which?" of.
    assert checked.facts()["hands_settings"] == "HANDS_LLM, HANDS_WHISPER_MODEL"
    # The brain's own settings are among the refused, and the environment is still said.
    match checked.verdict:
        case Refused(refusals=refusals):
            assert [said for said in (*each, f"{SAID_STRAY}{home.config}") if not any(said in refusal for refusal in refusals)] == [], refusals
        case passed:
            raise AssertionError(f"{passed} passed the preflight")
    with pytest.raises(CannotStart) as refused:
        run.configured_from(children_on(VirtualClock()), home, config.load(home), checked)
    assert [said for said in (*each, SAID_STRAY) if said not in str(refused.value)] == []
    # The survey says the same refusal.
    match readiness.configured(children_on(VirtualClock()), home, LAID_OUT, STRAY):
        case readiness.Missing(fix=fix):
            assert SAID_STRAY in fix
        case found:
            raise AssertionError(f"the survey found {found}")


def test_hands_home_is_the_one_variable_of_hands_the_preflight_leaves_in_the_environment(tmp_path: Path) -> None:
    home = Home(tmp_path)
    laid_out(home)
    assert isinstance(preflight(home, LAID_OUT, {"HANDS_HOME": str(tmp_path), "HOME": "/h"}).verdict, Startable)


async def test_an_edit_and_a_voice_choice_a_restart_would_refuse_on_a_hands_setting_are_refused_and_the_watch_outlives_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = Home(tmp_path)
    mark = laid_out(home)
    running = config.load(home)
    with pytest.raises(Rejected) as chosen:
        config.OwnModel(home, LAID_OUT, STRAY, running).weigh("claude-opus-5-5")
    assert SAID_STRAY in str(chosen.value)
    assert not home.config.exists()
    monkeypatch.setattr(config, "_held", _reads([EDIT] * 4))
    recorded: list[Entry] = []
    with pytest.raises(Outlived):
        await config.edited(home, LAID_OUT, STRAY, recorded.append, running, VirtualClock(), period=0)
    [entry] = recorded
    assert isinstance(entry, SettingsEdited) and entry.refused is not None and SAID_STRAY in entry.refused
    assert not mark.exists()


def test_an_unreadable_record_of_api_key_answers_is_said_beside_an_unreadable_env(tmp_path: Path) -> None:
    home = Home(tmp_path)
    laid_out(home)
    (home.brain / "settings.json").write_text(json.dumps({**KEPT_OUT, "env": "x"}))
    state = home.brain / ".claude.json"
    state.write_text(json.dumps({**json.loads(state.read_text()), "customApiKeyResponses": []}))
    match preflight(home, LAID_OUT, {}).verdict:
        case Refused(refusals=refusals):
            assert any("an env hands cannot read" in refusal for refusal in refusals), refusals
            assert any(f"{state} unreadable" in refusal for refusal in refusals), refusals
        case passed:
            raise AssertionError(f"{passed} passed the preflight")


async def test_an_edit_refused_on_its_home_names_every_refusal_at_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = Home(tmp_path)
    mark = laid_out(home)
    each = _faulted(home, monkeypatch)
    running = config.load(home)
    monkeypatch.setattr(config, "_held", _reads([EDIT] * 3))
    recorded: list[Entry] = []
    with pytest.raises(Outlived):
        await config.edited(home, LAID_OUT, {}, recorded.append, running, VirtualClock(), period=0)
    [entry] = recorded
    assert isinstance(entry, SettingsEdited) and entry.refused is not None
    assert [said for said in each if said not in entry.refused] == []
    assert not mark.exists()


def test_the_door_passes_a_home_whose_brain_is_refused_so_the_start_says_it_beside_the_survey(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from hands.voice import talkkey

    home = Home(tmp_path)
    monkeypatch.setattr(talkkey, "granted", lambda: True)
    _stale_shim(home)
    assert cli.door(home, LAID_OUT, {}, None, Start(False, VirtualClock())) == config.load(home)


async def test_an_edit_refused_on_its_home_is_taken_once_the_home_is_mended_its_bytes_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = Home(tmp_path)
    mark = laid_out(home)
    settings = home.brain / "settings.json"
    settings.write_text('{"syncClaudeAiPlugins": false}')
    # Seen, then weighed at each poll after while its bytes stand; the home is mended before the sixth read.
    running = config.load(home)
    monkeypatch.setattr(config, "_held", _reads([EDIT] * 8, {5: lambda: settings.write_text(json.dumps(KEPT_OUT))}))
    recorded: list[Entry] = []
    assert await config.edited(home, LAID_OUT, {}, recorded.append, running, VirtualClock(), period=0) == SettingsEdited(path=str(home.config), refused=None)
    assert [entry.refused for entry in recorded if isinstance(entry, SettingsEdited)] == [f'the brain would load its account\'s syncClaudeAiSkills: set "syncClaudeAiSkills": false and "syncClaudeAiPlugins": false in {settings}']
    assert not mark.exists()


async def test_an_edit_whose_refusals_change_while_its_bytes_stand_is_said_again_with_those_standing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = Home(tmp_path)
    laid_out(home)
    _stale_shim(home)
    home.voice.write_text("zed\n")
    # Refused on both; the voice mended before the fifth read, then broken again before the seventh.
    running = config.load(home)
    monkeypatch.setattr(config, "_held", _reads([EDIT] * 8, {4: lambda: home.voice.unlink(), 6: lambda: home.voice.write_text("zed\n")}))
    recorded: list[Entry] = []
    with pytest.raises(Outlived):
        await config.edited(home, LAID_OUT, {}, recorded.append, running, VirtualClock(), period=0)
    said = [(("is not the shim" in refused), ("says 'zed'" in refused)) for entry in recorded if isinstance(entry, SettingsEdited) and (refused := entry.refused) is not None]
    assert said == [(True, True), (True, False), (True, True)]


async def test_an_edit_refused_reverted_and_saved_again_is_refused_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = Home(tmp_path)
    laid_out(home)
    _stale_shim(home)
    # Refused, weighed again, reverted to the run's own bytes, and saved again: refused again.
    running = config.load(home)
    monkeypatch.setattr(config, "_held", _reads([EDIT, EDIT, EDIT, None, None, EDIT, EDIT, EDIT]))
    recorded: list[Entry] = []
    with pytest.raises(Outlived):
        await config.edited(home, LAID_OUT, {}, recorded.append, running, VirtualClock(), period=0)
    assert len([entry for entry in recorded if isinstance(entry, SettingsEdited) and "is not the shim" in (entry.refused or "")]) == 2


async def test_an_edit_and_a_voice_choice_are_weighed_without_starting_a_claude_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = Home(tmp_path)
    mark = laid_out(home)
    held = config._held  # pyright: ignore[reportPrivateUsage]
    running = config.load(home)
    monkeypatch.setattr(config, "_held", _reads([EDIT, EDIT, EDIT]))
    assert await config.edited(home, LAID_OUT, {}, lambda _entry: None, running, VirtualClock(), period=0) == SettingsEdited(path=str(home.config), refused=None)
    monkeypatch.setattr(config, "_held", held)
    config.OwnModel(home, LAID_OUT, {}, config.load(home)).weigh("claude-opus-5-5")()
    assert config.load(home).config.llm.model == "claude-opus-5-5"
    assert not mark.exists()


def test_a_home_with_the_fritter_and_shim_this_hands_installs_passes(tmp_path: Path) -> None:
    home = Home(tmp_path)
    laid_out(home)
    wrapper.install(home)
    checked = preflight(home, LAID_OUT, {})
    assert isinstance(checked.verdict, Startable)
    assert checked.facts() == {"fritter": "current", "shim": "current", "claude_code": LAID_OUT.version, "hands_settings": "none"}


def test_two_starts_in_one_process_each_weigh_the_brain_against_the_pin_they_are_handed(tmp_path: Path) -> None:
    home = Home(tmp_path)
    laid_out(home)
    wrapper.install(home)
    # The same build in the place another release's pin puts it, as a copy of the wrong release would be.
    other = pins.of(b"#!/bin/sh\nexit 0\n")
    other.claude(home.brain).parent.mkdir(parents=True)
    other.claude(home.brain).symlink_to(LAID_OUT.claude(home.brain))
    # Each start weighs the build against its own pin, whichever was weighed before it.
    assert isinstance(preflight(home, LAID_OUT, {}).verdict, Startable)
    match preflight(home, other, {}).verdict:
        case Refused(refusals=(unpinned,)):
            assert f"not the {other.build(platform.machine()).checksum} of Claude Code {other.version}, the brain's: run `hands install-brain` to fetch it again" in unpinned
        case verdict:
            raise AssertionError(f"{verdict} is not the brain refused on the other pin's digest alone")
    assert isinstance(preflight(home, LAID_OUT, {}).verdict, Startable)


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
def test_an_api_key_the_brain_settings_set_is_unanswered_as_claude_code_reads_it(key: str, answered: list[str], asked: bool, tmp_path: Path) -> None:
    home = Home(tmp_path)
    laid_out(home)
    _api_key(home, key, answered)
    match preflight(home, LAID_OUT, {}).verdict:
        case Refused(refusals=refusals):
            assert asked and any("the API key" in refusal and "unanswered" in refusal for refusal in refusals), refusals
        case Startable():
            assert not asked


async def test_settings_that_do_not_parse_hide_none_of_the_homes_refusals_from_the_start_the_survey_or_an_edit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from hands.voice import talkkey

    home = Home(tmp_path)
    mark = laid_out(home)
    _stale_shim(home)
    running = config.load(home)
    home.config.write_text("[llm\n")
    monkeypatch.setattr(talkkey, "granted", lambda: True)
    each = (str(home.config), "is not the shim this hands writes")
    # The start, refused at its door, before the settings name where the run is recorded, its event saying what the
    # preflight beside the settings found, as a start past its door says it.
    run_start = Start(False, VirtualClock())
    with pytest.raises(CannotStart) as refused:
        cli.door(home, LAID_OUT, {}, None, run_start)
    assert [said for said in each if said not in str(refused.value)] == []
    # Said as the start's other refusals are, ending in the one sentence they end in.
    assert str(refused.value).endswith("; then run hands again")
    events: list[WideEvent] = []
    run_start.ended(events.append, refused.value)
    assert [{fact: event.facts[fact] for fact in ("fritter", "shim", "claude_code", "hands_settings")} for event in events] == [preflight(home, LAID_OUT, {}).facts()]
    # The survey.
    match readiness.configured(children_on(VirtualClock()), home, LAID_OUT, shell_without_hands_settings()):
        case readiness.Missing(detail=detail):
            assert [said for said in each if said not in detail] == []
        case found:
            raise AssertionError(f"the survey found {found}")
    # A model chosen by voice on them.
    with pytest.raises(Rejected) as chosen:
        config.OwnModel(home, LAID_OUT, {}, running).weigh("claude-opus-5-5")
    assert [said for said in each if said not in str(chosen.value)] == []
    # An edit to them, said once, the watch outliving it.
    monkeypatch.setattr(config, "_held", _reads([b"[llm\n"] * 3))
    recorded: list[Entry] = []
    with pytest.raises(Outlived):
        await config.edited(home, LAID_OUT, {}, recorded.append, running, VirtualClock(), period=0)
    [entry] = recorded
    assert isinstance(entry, SettingsEdited) and entry.refused is not None
    assert [said for said in each if said not in entry.refused] == [], entry.refused
    assert not mark.exists()


def _logged_out(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """The brain's login, asked of the claude it is run with, answered with none; each claude it was asked of."""
    asked: list[Path] = []

    def logged_in(_children: object, claude: Path, *_: object) -> object:
        asked.append(claude)
        raise process.NotLoggedIn("the brain holds no login: run `hands login`")

    monkeypatch.setattr(process, "logged_in", logged_in)
    return asked


def test_a_start_refused_on_the_homes_other_files_names_the_brains_lost_login_beside_them(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = Home(tmp_path)
    laid_out(home)
    _stale_shim(home)
    asked = _logged_out(monkeypatch)
    each = ("is not the shim this hands writes", "the brain holds no login: run `hands login`")
    with pytest.raises(CannotStart) as refused:
        run.configured_from(children_on(VirtualClock()), home, config.load(home), preflight(home, LAID_OUT, {}))
    assert [said for said in each if said not in str(refused.value)] == []
    assert str(refused.value).endswith("; then run hands again")
    match readiness.configured(children_on(VirtualClock()), home, LAID_OUT, {}):
        case readiness.Missing(fix=fix):
            assert [said for said in each if said not in fix] == []
        case found:
            raise AssertionError(f"the survey found {found}")
    assert len(asked) == 2


def test_a_brain_whose_own_config_directory_is_refused_is_asked_no_login(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The run that asks would load the account's skills and plugins its settings do not keep out.
    home = Home(tmp_path)
    laid_out(home)
    (home.brain / "settings.json").write_text("{}")
    asked = _logged_out(monkeypatch)
    with pytest.raises(CannotStart):
        run.configured_from(children_on(VirtualClock()), home, config.load(home), preflight(home, LAID_OUT, {}))
    assert isinstance(readiness.configured(children_on(VirtualClock()), home, LAID_OUT, {}), readiness.Missing)
    assert asked == []


async def test_settings_that_do_not_parse_are_said_again_as_the_homes_refusals_beside_them_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = Home(tmp_path)
    laid_out(home)
    running = config.load(home)
    monkeypatch.setattr(config, "_held", _reads([b"[llm\n"] * 4, at={2: lambda: _stale_shim(home)}))
    recorded: list[Entry] = []
    with pytest.raises(Outlived):
        await config.edited(home, LAID_OUT, {}, recorded.append, running, VirtualClock(), period=0)
    said = [cast(SettingsEdited, entry).refused or "" for entry in recorded]
    assert [("is not the shim this hands writes" in refused) for refused in said] == [False, True]
