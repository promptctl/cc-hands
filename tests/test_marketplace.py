"""hands' plugin, rendered into the marketplace in its home for the interpreter of the hands that renders it."""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from hands.sessions.hookconfig import LAUNCHER, MARKETPLACE_NAME
from hands.sessions.home import Home
from hands.sessions.marketplace import MARKETPLACE_MANIFEST, PACKAGED, PLUGIN_MANIFEST, launcher, plugin, render


def files(root: Path) -> dict[str, bytes]:
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def marketplace(home: Home) -> dict[str, Any]:
    return json.loads((home.plugins / MARKETPLACE_MANIFEST).read_text())


def test_a_render_writes_the_package_s_plugin_with_its_launcher_and_the_marketplace_names_it(tmp_path: Path) -> None:
    home = Home(tmp_path / "home")
    rendered = render(home, sys.executable)
    assert (rendered.plugin, rendered.version, rendered.written) == (home.plugins / rendered.version, plugin(sys.executable).version, True)
    packaged = files(PACKAGED)
    manifest = json.loads(packaged[PLUGIN_MANIFEST])
    # Every file the package carries, the launcher running this interpreter, and the manifest stamped with the version.
    assert files(rendered.plugin) == {**packaged, LAUNCHER: launcher(sys.executable).encode(), PLUGIN_MANIFEST: files(rendered.plugin)[PLUGIN_MANIFEST]}
    assert json.loads(files(rendered.plugin)[PLUGIN_MANIFEST]) == {**manifest, "version": rendered.version}
    assert os.access(rendered.plugin / LAUNCHER, os.X_OK)
    assert marketplace(home) == {
        "name": MARKETPLACE_NAME,
        "description": "hands, hands-free control of Claude Code, as the hands installed writes it.",
        "owner": manifest["author"],
        "plugins": [{"name": manifest["name"], "description": manifest["description"], "source": f"./{rendered.version}"}],
    }


def test_a_second_render_writes_no_second_copy(tmp_path: Path) -> None:
    home = Home(tmp_path)
    first = render(home, sys.executable)
    again = render(home, sys.executable)
    assert (again.plugin, again.written) == (first.plugin, False)
    assert sorted(path.name for path in home.plugins.iterdir()) == [".claude-plugin", first.version]


def test_another_interpreter_is_another_plugin_and_the_marketplace_moves_onto_it(tmp_path: Path) -> None:
    home = Home(tmp_path)
    here = render(home, sys.executable)
    elsewhere = "/Users/someone else/.local/share/uv/tools/hands/bin/python"
    there = render(home, elsewhere)
    assert here.version != there.version and there.written
    assert (there.plugin / LAUNCHER).read_text() == launcher(elsewhere)
    # The plugin before stays where sessions loaded it from; the marketplace names the newest.
    assert here.plugin.is_dir()
    [named] = marketplace(home)["plugins"]
    assert named["source"] == f"./{there.version}"


def test_a_launcher_whose_interpreter_is_gone_says_which_and_how_to_bring_the_plugin_back(tmp_path: Path) -> None:
    gone = tmp_path / "uninstalled hands" / "bin" / "python"
    rendered = render(Home(tmp_path / "home"), str(gone)).plugin
    ran = subprocess.run([rendered / LAUNCHER, "-m", "hands.sessions.shim"], capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"}, check=False)
    assert (ran.returncode, ran.stdout) == (1, "")
    assert ran.stderr == f"hands: the plugin runs {gone}, which is gone; install hands again\n"


def full_disk(staged: Path, plugin: Path) -> None:
    raise OSError(28, "No space left on device")


def test_a_render_that_fails_leaves_nothing_staged_and_no_marketplace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = Home(tmp_path)
    monkeypatch.setattr(Path, "rename", full_disk)
    with pytest.raises(OSError, match="No space left"):
        render(home, sys.executable)
    assert list(home.plugins.iterdir()) == []


def test_a_plugin_rendered_already_is_kept_as_it_is(tmp_path: Path) -> None:
    home = Home(tmp_path)
    first = render(home, sys.executable)
    (first.plugin / "marker").write_text("the one already there")
    again = render(home, sys.executable)
    assert (again.plugin, again.written) == (first.plugin, False)
    assert (first.plugin / "marker").read_text() == "the one already there"
