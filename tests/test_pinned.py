"""The brain runs one Claude Code release, hands' own copy of it, and no other: never the person's claude."""

import asyncio
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
# Releases that are never the pin, whatever version it is moved to.
ANOTHER_VERSION = f"{pinned.VERSION}.1"
AN_OLDER_VERSION = f"{pinned.VERSION}.0"


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
    placed(config_dir / "pinned-claude" / pinned.VERSION / "claude", RELEASE)
    # Another version beside it is never the brain's.
    placed(config_dir / "pinned-claude" / ANOTHER_VERSION / "claude", RELEASE)
    assert brain_claude(config_dir) == config_dir / "pinned-claude" / pinned.VERSION / "claude"


def test_a_brain_whose_pinned_release_is_not_installed_refuses_naming_the_version_and_the_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pin_bytes(monkeypatch, RELEASE)
    config_dir = tmp_path / "brain"
    # Another version in its place is not the pin, and is never run for it.
    placed(config_dir / "pinned-claude" / AN_OLDER_VERSION / "claude", RELEASE)
    expected = config_dir / "pinned-claude" / pinned.VERSION / "claude"
    with pytest.raises(Unstartable) as refused:
        brain_claude(config_dir)
    assert str(refused.value) == f"Claude Code {pinned.VERSION}, the brain's, is not installed at {expected}: run `hands install-brain`"


def test_a_copy_that_is_not_the_pinned_release_refuses_naming_its_checksum_and_the_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pin_bytes(monkeypatch, RELEASE)
    config_dir = tmp_path / "brain"
    copy = placed(pinned.claude(config_dir), OTHER)
    with pytest.raises(Unstartable) as refused:
        brain_claude(config_dir)
    assert str(refused.value) == (
        f"{copy} hashes to {hashlib.sha256(OTHER).hexdigest()}, not the {hashlib.sha256(RELEASE).hexdigest()} of Claude Code {pinned.VERSION}, the brain's: run `hands install-brain` to fetch it again"
    )
    # Written over with the pin, it is the brain's: what was hashed before is not taken for what is there now.
    placed(copy, RELEASE)
    assert brain_claude(config_dir) == copy
    # Nor is it when other bytes of the same length are written into the same file and its modified time is set back.
    held = copy.stat()
    copy.write_bytes(RELEASE.replace(b"pinned", b"PINNED"))
    os.utime(copy, ns=(held.st_atime_ns, held.st_mtime_ns))
    assert (copy.stat().st_ino, copy.stat().st_size, copy.stat().st_mtime_ns) == (held.st_ino, held.st_size, held.st_mtime_ns)
    with pytest.raises(Unstartable, match="hashes to"):
        brain_claude(config_dir)


def test_nothing_on_path_or_in_the_persons_own_install_is_the_brains_claude(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The person's claude, as Claude Code installs itself, holding the very bytes of the pin: on PATH, at
    # ~/.local/bin/claude, and in ~/.local/share/claude/versions.
    pin_bytes(monkeypatch, RELEASE)
    home = tmp_path / "person"
    versions = placed(home / ".local" / "share" / "claude" / "versions" / pinned.VERSION, RELEASE)
    (home / ".local" / "bin").mkdir(parents=True)
    (home / ".local" / "bin" / "claude").symlink_to(versions)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", f"{home / '.local' / 'bin'}{os.pathsep}{os.environ['PATH']}")
    config_dir = tmp_path / "brain"
    with pytest.raises(Unstartable, match="is not installed at"):
        brain_claude(config_dir)


def test_a_copy_that_cannot_be_read_or_run_refuses_naming_the_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pin_bytes(monkeypatch, RELEASE)
    config_dir = tmp_path / "brain"
    copy = placed(pinned.claude(config_dir), RELEASE)
    # Not a file that can be read: refused with what reading it raised, not raised as it.
    copy.chmod(0o000)
    with pytest.raises(Unstartable, match=r"cannot be read as Claude Code .*Permission denied.*: run `hands install-brain` to fetch it again"):
        brain_claude(config_dir)
    # The pin's bytes, but not a file this process may run: refused, never left to fail at the spawn.
    copy.chmod(0o644)
    with pytest.raises(Unstartable) as refused:
        brain_claude(config_dir)
    assert str(refused.value) == f"{copy}, Claude Code {pinned.VERSION}, the brain's, cannot be run: run `hands install-brain` to fetch it again"
    # Where the version's directory should be, a file: the same.
    copy.unlink()
    copy.parent.rmdir()
    copy.parent.write_bytes(b"")
    with pytest.raises(Unstartable, match=r"cannot be read as Claude Code .*Not a directory.*: run `hands install-brain`"):
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
    target = config_dir / "pinned-claude" / pinned.VERSION / "claude"
    assert await pinned.install(config_dir) == pinned.Installed(target, unpinned=f"Claude Code {pinned.VERSION}, the brain's, is not installed at {target}: run `hands install-brain`")
    assert releases == [f"/{pinned.VERSION}/{built}/claude"]
    assert brain_claude(config_dir).read_bytes() == RELEASE and os.access(brain_claude(config_dir), os.X_OK)
    # There already, it is not fetched again.
    assert await pinned.install(config_dir) == pinned.Installed(pinned.claude(config_dir), unpinned=None)
    assert len(releases) == 1
    # A copy that is not the pin is fetched again, whole, saying what the copy there hashed to.
    pinned.claude(config_dir).write_bytes(OTHER)
    installed = await pinned.install(config_dir)
    assert installed.fetched and installed.unpinned is not None and f"hashes to {hashlib.sha256(OTHER).hexdigest()}" in installed.unpinned
    assert pinned.claude(config_dir).read_bytes() == RELEASE and len(releases) == 2


async def test_a_download_that_is_not_the_pinned_release_is_refused_and_leaves_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, releases: list[str]) -> None:
    pin_bytes(monkeypatch, OTHER)
    config_dir = tmp_path / "brain"
    with pytest.raises(pinned.Uninstallable, match=f"hashes to {hashlib.sha256(RELEASE).hexdigest()}, not the {hashlib.sha256(OTHER).hexdigest()} Claude Code {pinned.VERSION} is pinned to"):
        await pinned.install(config_dir)
    assert list(pinned.claude(config_dir).parent.iterdir()) == []


async def test_install_replaces_a_copy_of_the_pin_that_cannot_be_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, releases: list[str]) -> None:
    pin_bytes(monkeypatch, RELEASE)
    config_dir = tmp_path / "brain"
    placed(pinned.claude(config_dir), RELEASE).chmod(0o644)
    installed = await pinned.install(config_dir)
    assert installed.unpinned is not None and installed.unpinned.endswith("cannot be run: run `hands install-brain` to fetch it again")
    assert brain_claude(config_dir) == pinned.claude(config_dir) and len(releases) == 1


async def test_two_installs_at_once_each_write_their_own_download_and_leave_the_pin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, releases: list[str]) -> None:
    pin_bytes(monkeypatch, RELEASE)
    config_dir = tmp_path / "brain"
    first, second = await asyncio.gather(pinned.install(config_dir), pinned.install(config_dir))
    assert first.fetched and second.fetched and len(releases) == 2
    # Neither removed or renamed the other's download, and what is left is the pin alone.
    assert list(pinned.claude(config_dir).parent.iterdir()) == [pinned.claude(config_dir)]
    assert brain_claude(config_dir) == pinned.claude(config_dir)


@pytest.fixture
async def stalling(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[asyncio.Event]:
    """A stand-in for Anthropic's release downloads that sends the start of a release and then nothing: set once it has."""
    began, done = asyncio.Event(), asyncio.Event()

    async def download(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse()
        await response.prepare(request)
        await response.write(RELEASE[:8])
        began.set()
        await done.wait()
        return response

    app = web.Application()
    app.router.add_get("/{version}/{build}/claude", download)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    [(_, port)] = runner.addresses
    monkeypatch.setattr(pinned, "RELEASES", f"http://127.0.0.1:{port}")
    yield began
    done.set()
    await runner.cleanup()


async def test_a_stalled_download_fails_the_install_and_leaves_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stalling: asyncio.Event) -> None:
    pin_bytes(monkeypatch, RELEASE)
    monkeypatch.setattr(pinned, "STALL_SECONDS", 0.5)
    config_dir = tmp_path / "brain"
    with pytest.raises(pinned.Uninstallable, match="could not fetch"):
        await pinned.install(config_dir)
    assert stalling.is_set() and list(pinned.claude(config_dir).parent.iterdir()) == []


async def test_an_install_cut_short_leaves_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stalling: asyncio.Event) -> None:
    pin_bytes(monkeypatch, RELEASE)
    config_dir = tmp_path / "brain"
    installing = asyncio.create_task(pinned.install(config_dir))
    await stalling.wait()
    # The start of the release has reached the disk by the time it is cut short.
    while not list(pinned.claude(config_dir).parent.glob("claude.*.partial")):
        await asyncio.sleep(0.01)
    installing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await installing
    assert list(pinned.claude(config_dir).parent.iterdir()) == []


def test_install_brain_says_what_it_did_and_is_one_event_each_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    pin_bytes(monkeypatch, RELEASE)
    built = pinned.build(platform.machine())
    home = Home(tmp_path / "home")
    target = placed(pinned.claude(home.brain), RELEASE)
    assert main(["--home", str(home.root), "install-brain"]) == 0
    assert capsys.readouterr().out == f"already there: Claude Code {pinned.VERSION}, the brain's, at {target}\n"
    # A release that cannot be fetched fails the install, saying why.
    target.unlink()
    monkeypatch.setattr(pinned, "RELEASES", "http://127.0.0.1:1")
    assert main(["--home", str(home.root), "install-brain"]) == 1
    unfetched = capsys.readouterr().err
    assert unfetched.startswith(f"hands install-brain: could not fetch http://127.0.0.1:1/{pinned.VERSION}/{built}/claude: ")
    # One that fetches says so, and its event why the copy there was not the pin.
    unpinned = f"Claude Code {pinned.VERSION}, the brain's, is not installed at {target}: run `hands install-brain`"

    async def fetched(config_dir: Path) -> pinned.Installed:
        return pinned.Installed(placed(pinned.claude(config_dir), RELEASE), unpinned)

    monkeypatch.setattr(pinned, "install", fetched)
    assert main(["--home", str(home.root), "install-brain"]) == 0
    assert capsys.readouterr().out == f"fetched: Claude Code {pinned.VERSION}, the brain's, at {target}\n"
    # [LAW:nothing-unseen] each install's event: the release it pins, the build of it this Mac runs, where it put it,
    # whether it fetched it, and why.
    events = [line for line in (json.loads(line) for line in segment(home.audit, 0).read_text().splitlines()) if line.get("event") == "brain.install"]
    assert [(event["outcome"], event["facts"]) for event in events] == [
        ("ok", {"version": pinned.VERSION, "build": built, "claude": str(target), "fetched": False, "unpinned": None}),
        ("failed", {"version": pinned.VERSION, "build": built}),
        ("ok", {"version": pinned.VERSION, "build": built, "claude": str(target), "fetched": True, "unpinned": unpinned}),
    ]
    # The failed one's command says why, in the words it said it.
    commands = [line for line in (json.loads(line) for line in segment(home.audit, 0).read_text().splitlines()) if line.get("event") == "hands.command"]
    assert [command["error"] for command in commands] == [None, f"exited 1: {unfetched.removeprefix('hands install-brain: ').removesuffix('\n')}", None]
