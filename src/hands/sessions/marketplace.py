"""hands' Claude Code plugin, written by the installed hands into a marketplace in its home that Claude Code installs from.

The plugin's files are in hands' package, beside this module. The one thing a package cannot carry is which interpreter
runs it, so the launcher every hook and skill names is written here, for the interpreter that runs this. Claude Code loads
a plugin from a local marketplace in place, from the directory the marketplace's manifest names, so each plugin rendered
is a directory of its own, named by its content, and the manifest is moved onto the newest.
"""

import hashlib
import json
import os
import shlex
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from hands.sessions.hookconfig import LAUNCHER, MARKETPLACE_NAME
from hands.sessions.home import Home

# The plugin's files as hands' package carries them: everything but its launcher.
PACKAGED = Path(__file__).parent / "plugin"
# Where a plugin's manifest is within it, and a marketplace's within its root.
PLUGIN_MANIFEST = ".claude-plugin/plugin.json"
MARKETPLACE_MANIFEST = ".claude-plugin/marketplace.json"


@dataclass(frozen=True)
class Plugin:
    """The plugin for one interpreter, before it is written."""

    # [LAW:one-source-of-truth] the digest of the package's files and the launcher: the directory's name, and the version
    # its manifest carries, so what Claude Code lists of an installed plugin says which render it is.
    version: str
    files: Mapping[str, bytes]


@dataclass(frozen=True)
class Rendered:
    plugin: Path
    version: str
    # False when an install before had rendered this same plugin.
    written: bool


def launcher(interpreter: str) -> str:
    """hooks/python: the interpreter given, run on whatever module a hook or skill names."""
    # exec, so the hook is the process Claude Code spawned (hooks.json is exec form, with no shell between), and the
    # shim's parent is the claude whose pid it records. -I keeps the session's directory, where Claude Code runs the
    # hook, and the user's PYTHON* variables off the path, so a project's own json.py or hands/ cannot stand in for
    # hands' modules.
    # [LAW:no-silent-failure] a plugin outlives the interpreter it names when hands is uninstalled or its venv is
    # rebuilt; the hook then says which interpreter is gone and what brings the plugin back, not the shell's exit 127.
    quoted = shlex.quote(interpreter)
    return (
        "#!/bin/sh\n"
        f"[ -x {quoted} ] || {{ echo {shlex.quote(f'hands: the plugin runs {interpreter}, which is gone; install hands again')} >&2; exit 1; }}\n"
        f'exec {quoted} -I "$@"\n'
    )


def plugin(interpreter: str) -> Plugin:
    """The plugin whose launcher runs `interpreter`."""
    files = {path.relative_to(PACKAGED).as_posix(): path.read_bytes() for path in PACKAGED.rglob("*") if path.is_file()}
    files[LAUNCHER] = launcher(interpreter).encode()
    version = digest(files)
    manifest = json.loads(files[PLUGIN_MANIFEST])
    return Plugin(version, {**files, PLUGIN_MANIFEST: json.dumps({**manifest, "version": version}, indent=2).encode()})


def render(home: Home, interpreter: str) -> Rendered:
    """Write the plugin, its launcher running `interpreter`, into the home's marketplace, and point the marketplace at it."""
    rendering = plugin(interpreter)
    rendered = _written(home.plugins / rendering.version, rendering)
    manifest = json.loads(rendering.files[PLUGIN_MANIFEST])
    marketplace = {
        "name": MARKETPLACE_NAME,
        "description": "hands, hands-free control of Claude Code, as the hands installed writes it.",
        "owner": manifest["author"],
        "plugins": [{"name": manifest["name"], "description": manifest["description"], "source": f"./{rendered.plugin.name}"}],
    }
    _replace(home.plugins / MARKETPLACE_MANIFEST, json.dumps(marketplace, indent=2).encode())
    return rendered


def _written(directory: Path, rendering: Plugin) -> Rendered:
    # Sessions load the plugin from this directory, so one already there is never written again.
    if directory.is_dir():
        return Rendered(directory, rendering.version, written=False)
    directory.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=".staged-", dir=directory.parent))
    try:
        for name, content in rendering.files.items():
            (staged / name).parent.mkdir(parents=True, exist_ok=True)
            (staged / name).write_bytes(content)
        (staged / LAUNCHER).chmod(0o755)
        staged.chmod(0o755)
        # [LAW:no-ambient-temporal-coupling] a directory is staged whole and renamed into place, so Claude Code never
        # loads one half written, and one already there has these same files.
        try:
            staged.rename(directory)
        except OSError:
            if not directory.is_dir():
                raise
            return Rendered(directory, rendering.version, written=False)
        return Rendered(directory, rendering.version, written=True)
    finally:
        # Renamed into place, the staged directory is gone; anything else it became, failed or redundant, goes with it.
        if staged.exists():
            shutil.rmtree(staged)


def _replace(path: Path, content: bytes) -> None:
    """Write `path` whole, beside it and renamed into place, so a reader never sees half of it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    staging.write_bytes(content)
    staging.replace(path)


def digest(files: Mapping[str, bytes]) -> str:
    """A name for a directory's files: each one's path within it and its bytes."""
    hashed = hashlib.sha256()
    for name in sorted(files):
        hashed.update(f"{name}\0".encode() + files[name] + b"\0")
    return hashed.hexdigest()[:16]
