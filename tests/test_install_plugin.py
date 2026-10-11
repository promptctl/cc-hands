"""`hands install-plugin`: hands writes its plugin into a marketplace in its home and installs it from there, asking
nothing, on a first install, over an older one, and over a marketplace of its name added from elsewhere.

`claude` is a stand-in that records each call and keeps its marketplaces and plugins in files, as Claude Code does: a
marketplace added from a directory, and a plugin installed or updated to the version that directory's manifest names.
Its stdin is never a terminal, and it reads none.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from hands.daemon import cli
from hands.sessions.audit import segment
from hands.sessions.home import Home
from hands.sessions.hookconfig import PLUGIN_ID
from hands.sessions.marketplace import plugin

CLAUDE = r"""#!/bin/bash
echo "claude $*" >>"$ROOT/calls"
[ ! -t 0 ] || { echo "claude was given a terminal" >&2; exit 9; }
json() { "$PYTHON" -c "import json, sys; print($1)" "${@:2}"; }
# The version the marketplace added from a directory names for its plugin, as an install or update records it.
offered() {
  local root
  root=$(json 'json.load(open(sys.argv[1]))[0]["path"]' "$ROOT/marketplaces")
  json 'json.load(open(sys.argv[1] + "/" + json.load(open(sys.argv[1] + "/.claude-plugin/marketplace.json"))["plugins"][0]["source"] + "/.claude-plugin/plugin.json"))["version"]' "$root"
}
listed() { printf '[{"id": "%s", "scope": "user", "enabled": %s, "version": "%s"}]' "$PLUGIN_ID" "$1" "$2" >"$ROOT/listed"; }
case "$1 $2 $3" in
  "plugin list --json")
    [ ! -e "$ROOT/list-fails" ] || { echo "cannot list" >&2; exit 3; }
    cat "$ROOT/listed" 2>/dev/null || echo "[]" ;;
  "plugin marketplace list") cat "$ROOT/marketplaces" 2>/dev/null || echo "[]" ;;
  "plugin marketplace add")
    [ ! -e "$ROOT/add-fails" ] || exit 4
    [ -f "$4/.claude-plugin/marketplace.json" ] || { echo "no marketplace at $4" >&2; exit 5; }
    printf '[{"name": "cc-hands", "source": "directory", "path": "%s"}]' "$4" >"$ROOT/marketplaces" ;;
  # Removing a marketplace uninstalls what was installed from it.
  "plugin marketplace remove") echo "[]" >"$ROOT/marketplaces"; rm -f "$ROOT/listed" ;;
  "plugin marketplace update") ;;
  # Over an install, enabled or not, an install enables it and changes no version.
  "plugin install --scope")
    if [ -e "$ROOT/listed" ]; then
      listed true "$(json 'json.load(open(sys.argv[1]))[0]["version"]' "$ROOT/listed")"
    else
      listed true "$(offered)"
    fi ;;
  "plugin update --scope")
    [ ! -e "$ROOT/update-hangs" ] || exec sleep 30
    # An update that exits 0 and moves nothing, as over a manifest Claude Code kept from before.
    [ ! -e "$ROOT/update-stale" ] || { listed true stale; exit 0; }
    listed true "$(offered)" ;;
  *) echo "unexpected: $*" >&2; exit 8 ;;
esac
"""
VERSION = plugin(sys.executable).version
INSTALLED = [f"claude plugin install --scope user {PLUGIN_ID}", f"claude plugin update --scope user {PLUGIN_ID}", "claude plugin list --json"]


@pytest.fixture
def root() -> Iterator[Path]:
    root = Path(tempfile.mkdtemp(prefix="plug-", dir="/tmp")).resolve()
    stand_in = root / "bin" / "claude"
    stand_in.parent.mkdir()
    stand_in.write_text(CLAUDE)
    stand_in.chmod(0o755)
    (root / "person").mkdir()
    yield root
    shutil.rmtree(root)


def person(root: Path) -> dict[str, str]:
    """The person's own environment: no CLAUDE_CONFIG_DIR, and the stand-in first on PATH."""
    inherited = {name: value for name, value in os.environ.items() if name != "CLAUDE_CONFIG_DIR"}
    return {
        **inherited,
        "PATH": f"{root / 'bin'}:/usr/bin:/bin",
        "HOME": str(root / "person"),
        "ROOT": str(root),
        "PYTHON": sys.executable,
        "PLUGIN_ID": PLUGIN_ID,
        "HANDS_HOME": str(root / "home"),
    }


def install(root: Path, **environment: str) -> subprocess.CompletedProcess[str]:
    given = {**person(root), **environment}
    return subprocess.run([sys.executable, "-m", "hands.daemon", "install-plugin"], env=given, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60)


def events(root: Path) -> list[dict[str, Any]]:
    lines = (json.loads(line) for line in segment(Home(root / "home").audit, 0).read_text().splitlines())
    return [line for line in lines if line.get("event") == "plugin.install"]


def calls(root: Path) -> list[str]:
    return (root / "calls").read_text().splitlines()


def listed(root: Path) -> list[dict[str, Any]]:
    return json.loads((root / "listed").read_text())


def rendered(root: Path) -> dict[str, object]:
    return {"config": str(root / "person" / ".claude"), "plugin": str(Home(root / "home").plugins / VERSION), "version": VERSION}


def added_from(root: Path, source: dict[str, str]) -> None:
    (root / "marketplaces").write_text(json.dumps([{"name": "cc-hands", **source}]))


def test_a_first_install_adds_the_home_s_marketplace_and_installs_this_hands_plugin_asking_nothing(root: Path) -> None:
    ran = install(root)
    assert ran.returncode == 0, ran.stderr
    assert f"the plugin {PLUGIN_ID} {VERSION} is installed and enabled for every session" in ran.stdout
    plugins = Home(root / "home").plugins
    assert calls(root) == ["claude plugin marketplace list --json", f"claude plugin marketplace add {plugins}", *INSTALLED]
    assert listed(root) == [{"id": PLUGIN_ID, "scope": "user", "enabled": True, "version": VERSION}]
    # [LAW:nothing-unseen] the install's event: where, what it rendered, how it found the marketplace, each exit, and after.
    [event] = events(root)
    facts = {**rendered(root), "written": True, "marketplace": "missing", "marketplace_add_exit": 0, "install_exit": 0, "update_exit": 0, "after": "ready"}
    assert (event["outcome"], event["facts"]) == ("ok", facts)


def test_an_install_over_another_hands_plugin_moves_it_onto_this_one(root: Path) -> None:
    added_from(root, {"source": "directory", "path": str(Home(root / "home").plugins)})
    (root / "listed").write_text(json.dumps([{"id": PLUGIN_ID, "scope": "user", "enabled": True, "version": "0123456789abcdef"}]))
    ran = install(root)
    assert ran.returncode == 0, ran.stderr
    assert calls(root) == ["claude plugin marketplace list --json", "claude plugin marketplace update cc-hands", *INSTALLED]
    assert listed(root)[0]["version"] == VERSION
    [event] = events(root)
    assert event["facts"] == {**rendered(root), "written": True, "marketplace": "ready", "marketplace_update_exit": 0, "install_exit": 0, "update_exit": 0, "after": "ready"}


def test_a_disabled_plugin_is_enabled(root: Path) -> None:
    added_from(root, {"source": "directory", "path": str(Home(root / "home").plugins)})
    (root / "listed").write_text(json.dumps([{"id": PLUGIN_ID, "scope": "user", "enabled": False, "version": VERSION}]))
    assert install(root).returncode == 0
    assert listed(root) == [{"id": PLUGIN_ID, "scope": "user", "enabled": True, "version": VERSION}]


def test_a_marketplace_of_its_name_added_from_elsewhere_is_replaced_by_the_home_s(root: Path) -> None:
    # As hands installed its plugin before: from GitHub, through a command the person had to accept.
    added_from(root, {"source": "github", "repo": "promptctl/cc-hands"})
    (root / "listed").write_text(json.dumps([{"id": PLUGIN_ID, "scope": "user", "enabled": True, "version": "unknown"}]))
    ran = install(root)
    assert ran.returncode == 0, ran.stderr
    plugins = Home(root / "home").plugins
    assert calls(root) == ["claude plugin marketplace list --json", "claude plugin marketplace remove cc-hands", f"claude plugin marketplace add {plugins}", *INSTALLED]
    assert listed(root)[0]["version"] == VERSION
    [event] = events(root)
    replaced = {"marketplace": "elsewhere", "marketplace_remove_exit": 0, "marketplace_add_exit": 0}
    assert event["facts"] == {**rendered(root), "written": True, **replaced, "install_exit": 0, "update_exit": 0, "after": "ready"}


def test_a_second_install_writes_nothing_new_and_runs_the_same_commands(root: Path) -> None:
    assert install(root).returncode == 0
    (root / "calls").unlink()
    again = install(root)
    assert again.returncode == 0, again.stderr
    assert calls(root) == ["claude plugin marketplace list --json", "claude plugin marketplace update cc-hands", *INSTALLED]
    assert events(root)[-1]["facts"]["written"] is False


def test_a_command_that_fails_stops_the_install_naming_it(root: Path) -> None:
    (root / "add-fails").touch()
    ran = install(root)
    assert ran.returncode == 1
    plugins = Home(root / "home").plugins
    assert f"hands install-plugin: `claude plugin marketplace add {plugins}` failed (4)" in ran.stderr
    assert not (root / "listed").exists()
    [event] = events(root)
    assert (event["outcome"], event["facts"]) == ("failed", {**rendered(root), "written": True, "marketplace": "missing", "marketplace_add_exit": 4})


def test_a_command_that_hangs_is_stopped_and_said(root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    # In this process, so the command's limit can be a second, not the minutes a real one is given.
    (root / "update-hangs").touch()
    # setenv, not a new os.environ: the `claude` this runs inherits the process's own environment.
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    for name, value in person(root).items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(cli, "PLUGIN_COMMAND_TIMEOUT_SECONDS", 1.0)
    assert cli.main(["install-plugin"]) == 1
    assert f"hands install-plugin: `claude plugin update --scope user {PLUGIN_ID}` did not finish in 1s" in capsys.readouterr().err
    [event] = events(root)
    assert (event["outcome"], event["facts"]["update_exit"]) == ("failed", "timeout")


def test_a_plugin_claude_code_did_not_move_is_said_without_sending_the_person_back_here(root: Path) -> None:
    (root / "update-stale").touch()
    ran = install(root)
    assert ran.returncode == 1
    assert "is stale, not this hands'" in ran.stderr and "exited 0 without making it so" in ran.stderr
    assert "`hands install-plugin`" not in ran.stderr


def test_a_claude_that_cannot_list_its_plugins_is_said_and_fails(root: Path) -> None:
    (root / "list-fails").touch()
    ran = install(root)
    assert ran.returncode == 1
    assert "`claude plugin list` failed (3)" in ran.stderr and "cannot list" in ran.stderr
    [event] = events(root)
    assert (event["outcome"], event["facts"]["after"]) == ("failed", "unknown")


def test_an_install_under_the_brain_s_config_installs_nothing(root: Path) -> None:
    # The brain's every command inherits its CLAUDE_CONFIG_DIR, which none of the person's sessions run under.
    brain = Home(root / "home").brain
    brain.mkdir(parents=True)
    ran = install(root, CLAUDE_CONFIG_DIR=str(brain))
    assert ran.returncode == 2
    assert f"Claude Code here runs under the brain's config, {brain}" in ran.stderr
    assert not (root / "calls").exists() and not Home(root / "home").plugins.exists()
    [event] = events(root)
    assert (event["outcome"], event["facts"]) == ("failed", {"config": str(brain)})
