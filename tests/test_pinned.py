"""The brain runs one Claude Code release, hands' own copy of it, and no other: never the person's claude."""

import hashlib
import json
import os
import platform
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from aiohttp import web

from hands.brain import pinned
from hands.brain.process import Unstartable, brain_claude
from hands.daemon.cli import main
from hands.sessions.audit import segment
from hands.sessions.home import Home

RELEASE = b"#!/bin/sh\necho 'the pinned release'\n"
OTHER = b"#!/bin/sh\necho 'another release'\n"


def pin_bytes(monkeypatch: pytest.MonkeyPatch, release: bytes) -> None:
    """Pins `release`'s bytes in place of the real release's, under the real VERSION."""
    monkeypatch.setattr(pinned, "CHECKSUMS", dict.fromkeys(pinned.CHECKSUMS, hashlib.sha256(release).hexdigest()))


def placed(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    path.chmod(0o755)
    return path


def test_the_brain_runs_exactly_the_pinned_version_in_its_own_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pin_bytes(monkeypatch, RELEASE)
    config_dir = tmp_path / "brain"
    placed(config_dir / "pinned-claude" / "2.1.289" / "claude", RELEASE)
    # Another version beside it is never the brain's.
    placed(config_dir / "pinned-claude" / "2.1.299" / "claude", RELEASE)
    assert brain_claude(config_dir) == config_dir / "pinned-claude" / "2.1.289" / "claude"


def test_a_brain_whose_pinned_release_is_not_installed_refuses_naming_the_version_and_the_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pin_bytes(monkeypatch, RELEASE)
    config_dir = tmp_path / "brain"
    # Another version in its place is not the pin, and is never run for it.
    placed(config_dir / "pinned-claude" / "2.1.281" / "claude", RELEASE)
    expected = config_dir / "pinned-claude" / "2.1.289" / "claude"
    with pytest.raises(Unstartable) as refused:
        brain_claude(config_dir)
    assert str(refused.value) == f"Claude Code 2.1.289, the brain's, is not installed at {expected}: run `hands install-brain`"


def test_a_copy_that_is_not_the_pinned_release_refuses_naming_its_checksum_and_the_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pin_bytes(monkeypatch, RELEASE)
    config_dir = tmp_path / "brain"
    copy = placed(pinned.claude(config_dir), OTHER)
    with pytest.raises(Unstartable) as refused:
        brain_claude(config_dir)
    assert str(refused.value) == (
        f"{copy} hashes to {hashlib.sha256(OTHER).hexdigest()}, not the {hashlib.sha256(RELEASE).hexdigest()} of Claude Code 2.1.289, the brain's: run `hands install-brain` to fetch it again"
    )
    # Written over with the pin, it is the brain's: what was hashed before is not taken for what is there now.
    placed(copy, RELEASE)
    assert brain_claude(config_dir) == copy


def test_nothing_on_path_or_in_the_persons_own_install_is_the_brains_claude(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The person's claude, as Claude Code installs itself, holding the very bytes of the pin: on PATH, at
    # ~/.local/bin/claude, and in ~/.local/share/claude/versions.
    pin_bytes(monkeypatch, RELEASE)
    home = tmp_path / "person"
    versions = placed(home / ".local" / "share" / "claude" / "versions" / "2.1.289", RELEASE)
    (home / ".local" / "bin").mkdir(parents=True)
    (home / ".local" / "bin" / "claude").symlink_to(versions)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", f"{home / '.local' / 'bin'}{os.pathsep}{os.environ['PATH']}")
    config_dir = tmp_path / "brain"
    with pytest.raises(Unstartable, match="is not installed at"):
        brain_claude(config_dir)


@pytest.fixture
async def releases(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[list[str]]:
    """A stand-in for Anthropic's release downloads, serving RELEASE for every build: the paths asked for."""
    asked: list[str] = []

    async def download(request: web.Request) -> web.Response:
        asked.append(request.path)
        return web.Response(body=RELEASE)

    app = web.Application()
    app.router.add_get("/{version}/{build}/claude", download)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    [(_, port)] = runner.addresses
    monkeypatch.setattr(pinned, "RELEASES", f"http://127.0.0.1:{port}")
    yield asked
    await runner.cleanup()


async def test_install_fetches_the_pinned_build_once_and_replaces_a_copy_that_is_not_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, releases: list[str]) -> None:
    pin_bytes(monkeypatch, RELEASE)
    config_dir = tmp_path / "brain"
    built = pinned.build(platform.machine())
    assert await pinned.install(config_dir) == pinned.Installed(config_dir / "pinned-claude" / "2.1.289" / "claude", fetched=True)
    assert releases == [f"/2.1.289/{built}/claude"]
    assert brain_claude(config_dir).read_bytes() == RELEASE and os.access(brain_claude(config_dir), os.X_OK)
    # There already, it is not fetched again.
    assert await pinned.install(config_dir) == pinned.Installed(pinned.claude(config_dir), fetched=False)
    assert len(releases) == 1
    # A copy that is not the pin is fetched again, whole.
    pinned.claude(config_dir).write_bytes(OTHER)
    assert await pinned.install(config_dir) == pinned.Installed(pinned.claude(config_dir), fetched=True)
    assert pinned.claude(config_dir).read_bytes() == RELEASE and len(releases) == 2


async def test_a_download_that_is_not_the_pinned_release_is_refused_and_leaves_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, releases: list[str]) -> None:
    pin_bytes(monkeypatch, OTHER)
    config_dir = tmp_path / "brain"
    with pytest.raises(pinned.Uninstallable, match=f"hashes to {hashlib.sha256(RELEASE).hexdigest()}, not the {hashlib.sha256(OTHER).hexdigest()} Claude Code 2.1.289 is pinned to"):
        await pinned.install(config_dir)
    assert list(pinned.claude(config_dir).parent.iterdir()) == []


def test_install_brain_says_what_it_did_and_is_one_event_each_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    pin_bytes(monkeypatch, RELEASE)
    home = Home(tmp_path / "home")
    target = placed(pinned.claude(home.brain), RELEASE)
    assert main(["--home", str(home.root), "install-brain"]) == 0
    assert capsys.readouterr().out == f"already there: Claude Code 2.1.289, the brain's, at {target}\n"
    # A release that cannot be fetched fails the install, saying why.
    target.unlink()
    monkeypatch.setattr(pinned, "RELEASES", "http://127.0.0.1:1")
    assert main(["--home", str(home.root), "install-brain"]) == 1
    assert capsys.readouterr().err.startswith(f"hands install-brain: could not fetch http://127.0.0.1:1/2.1.289/{pinned.build(platform.machine())}/claude: ")
    # [LAW:nothing-unseen] each install's event: the release it pins, where it put it, and whether it fetched it.
    events = [line for line in (json.loads(line) for line in segment(home.audit, 0).read_text().splitlines()) if line.get("event") == "brain.install"]
    assert [(event["outcome"], event["facts"]) for event in events] == [
        ("ok", {"version": "2.1.289", "claude": str(target), "fetched": False}),
        ("failed", {"version": "2.1.289"}),
    ]
