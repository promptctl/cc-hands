"""`hands install-plugin`: hands writes its plugin into a marketplace in its home and installs it from there, asking
nothing, on a first install, over an older one, and over a marketplace of its name added from elsewhere.

Claude Code is the fake built from recordings (claudecode_fake): a marketplace added from a directory, and a plugin
installed or updated to the version that directory's manifest names, as a real one does.
"""

import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

import pytest

from claudecode_fake import Config, Fake, Held, config_dir
from hands.daemon import cli
from hands.sessions import claudecode
from hands.sessions.audit import segment
from hands.sessions.claudecode import Installed, Instance, Unreachable
from hands.sessions.home import Home
from hands.sessions.hookconfig import PLUGIN_ID
from hands.sessions.marketplace import plugin

VERSION = plugin(sys.executable).version
INSTALLED = [f"plugin install {PLUGIN_ID}", f"plugin update {PLUGIN_ID}", "plugin list"]


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Home:
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path.resolve() / "config"))
    return Home(tmp_path.resolve() / "home")


def install(home: Home, fake: Fake) -> int:
    return cli.install_plugin(fake, home, cli.audit_log_of(home).record)


def person(fake: Fake) -> Config:
    return fake.configs.setdefault(config_dir(os.environ), Config())


def events(home: Home) -> list[dict[str, Any]]:
    lines = (json.loads(line) for line in segment(home.audit, 0).read_text().splitlines())
    return [line for line in lines if line.get("event") == "plugin.install"]


def asked(fake: Fake) -> list[str]:
    return [line for line in fake.transcript if not line.startswith("shown: ")]


def rendered(home: Home) -> dict[str, object]:
    return {"config": os.environ["CLAUDE_CONFIG_DIR"], "plugin": str(home.plugins / VERSION), "version": VERSION}


def test_a_first_install_adds_the_home_s_marketplace_and_installs_this_hands_plugin_asking_nothing(home: Home, capsys: pytest.CaptureFixture[str]) -> None:
    fake = Fake()
    assert install(home, fake) == 0
    assert f"the plugin {PLUGIN_ID} {VERSION} is installed and enabled for every session" in capsys.readouterr().out
    assert asked(fake) == ["plugin marketplace list", f"plugin marketplace add {home.plugins}", *INSTALLED]
    assert person(fake) == Config(plugin=Held(VERSION), marketplace=home.plugins)
    # [LAW:nothing-unseen] the install's event: where, what it rendered, how it found the marketplace, each exit, and after.
    [event] = events(home)
    facts = {**rendered(home), "written": True, "marketplace": "missing", "marketplace_add_exit": 0, "install_exit": 0, "update_exit": 0, "after": "ready"}
    assert (event["outcome"], event["facts"]) == ("ok", facts)


def test_an_install_over_another_hands_plugin_moves_it_onto_this_one(home: Home) -> None:
    fake = Fake()
    person(fake).marketplace, person(fake).plugin = home.plugins, Held("0123456789abcdef")
    assert install(home, fake) == 0
    assert asked(fake) == ["plugin marketplace list", "plugin marketplace update cc-hands", *INSTALLED]
    assert person(fake).plugin == Held(VERSION)
    [event] = events(home)
    assert event["facts"] == {**rendered(home), "written": True, "marketplace": "ready", "marketplace_update_exit": 0, "install_exit": 0, "update_exit": 0, "after": "ready"}


def test_a_disabled_plugin_is_enabled(home: Home) -> None:
    fake = Fake()
    person(fake).marketplace, person(fake).plugin = home.plugins, Held(VERSION, enabled=False)
    assert install(home, fake) == 0
    assert person(fake).plugin == Held(VERSION)


def test_a_marketplace_of_its_name_added_from_elsewhere_is_replaced_by_the_home_s(home: Home) -> None:
    # As hands installed its plugin before: from GitHub, through a command the person had to accept.
    fake = Fake()
    person(fake).marketplace, person(fake).plugin = "github", Held("f24447053e9c")
    assert install(home, fake) == 0
    assert asked(fake) == ["plugin marketplace list", "plugin marketplace remove cc-hands", f"plugin marketplace add {home.plugins}", *INSTALLED]
    assert person(fake) == Config(plugin=Held(VERSION), marketplace=home.plugins)
    [event] = events(home)
    replaced = {"marketplace": "elsewhere", "marketplace_remove_exit": 0, "marketplace_add_exit": 0}
    assert event["facts"] == {**rendered(home), "written": True, **replaced, "install_exit": 0, "update_exit": 0, "after": "ready"}


def test_a_second_install_writes_nothing_new_and_runs_the_same_commands(home: Home) -> None:
    fake = Fake()
    assert install(home, fake) == 0
    fake.transcript.clear()
    assert install(home, fake) == 0
    assert asked(fake) == ["plugin marketplace list", "plugin marketplace update cc-hands", *INSTALLED]
    assert events(home)[-1]["facts"]["written"] is False


def test_a_command_that_fails_stops_the_install_naming_it(home: Home, capsys: pytest.CaptureFixture[str]) -> None:
    fake = Fake(refuses_marketplace=True)
    assert install(home, fake) == 1
    assert f"hands install-plugin: `claude plugin marketplace add {home.plugins}` failed (1)" in capsys.readouterr().err
    assert person(fake).plugin is None
    [event] = events(home)
    assert (event["outcome"], event["facts"]) == ("failed", {**rendered(home), "written": True, "marketplace": "missing", "marketplace_add_exit": 1})


def test_a_command_that_cannot_be_run_stops_the_install_saying_why(home: Home, capsys: pytest.CaptureFixture[str]) -> None:
    fake = Fake(unrunnable={f"plugin update {PLUGIN_ID}"})
    assert install(home, fake) == 1
    assert f"hands install-plugin: `claude plugin update {PLUGIN_ID}` could not be run" in capsys.readouterr().err
    [event] = events(home)
    assert (event["outcome"], event["facts"]["update_exit"]) == ("failed", "unreachable")


def test_a_command_that_hangs_is_stopped_and_said(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The real adapter, so the command's limit can be a second, not the minutes a real one is given.
    hangs = tmp_path / "claude"
    hangs.write_text("#!/bin/sh\nexec sleep 30\n")
    hangs.chmod(hangs.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(claudecode, "PLUGIN_COMMAND_SECONDS", 1.0)
    with pytest.raises(Unreachable, match=f"`claude plugin update --scope user {PLUGIN_ID}` did not finish in 1s"):
        Installed().update_plugin(Instance(hangs, {}), PLUGIN_ID)


def test_a_plugin_claude_code_did_not_move_is_said_without_sending_the_person_back_here(home: Home, capsys: pytest.CaptureFixture[str]) -> None:
    fake = Fake(stale=True)
    person(fake).marketplace, person(fake).plugin = home.plugins, Held("0123456789abcdef")
    assert install(home, fake) == 1
    said = capsys.readouterr().err
    assert "is 0123456789abcdef, not this hands'" in said and "exited 0 without making it so" in said
    assert "`hands install-plugin`" not in said


def test_an_install_run_from_a_removed_directory_installs_nothing_and_says_why(home: Home, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    gone = tmp_path / "gone"
    gone.mkdir()
    monkeypatch.chdir(gone)
    gone.rmdir()
    fake = Fake()
    assert install(home, fake) == 2
    assert "hands install-plugin: cannot tell which Claude Code config this runs under" in capsys.readouterr().err
    assert fake.transcript == []


def test_a_claude_that_cannot_list_its_plugins_is_said_and_fails(home: Home, capsys: pytest.CaptureFixture[str]) -> None:
    fake = Fake(unrunnable={"plugin list"})
    assert install(home, fake) == 1
    assert "cannot ask `claude plugin list`" in capsys.readouterr().err
    [event] = events(home)
    assert (event["outcome"], event["facts"]["after"]) == ("failed", "unknown")


def test_an_install_under_the_brain_s_config_installs_nothing(home: Home, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    # The brain's every command inherits its CLAUDE_CONFIG_DIR, which none of the person's sessions run under.
    home.brain.mkdir(parents=True)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(home.brain))
    fake = Fake()
    assert install(home, fake) == 2
    assert f"Claude Code here runs under the brain's config, {home.brain}" in capsys.readouterr().err
    assert fake.transcript == [] and not home.plugins.exists()
    [event] = events(home)
    assert (event["outcome"], event["facts"]) == ("failed", {"config": str(home.brain)})
