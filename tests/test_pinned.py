"""The brain runs one Claude Code release, hands' own copy of it, and no other: never the person's claude."""

import asyncio
import hashlib
import json
import os
import platform
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from aiohttp import web

from hands.brain import pinned
from hands.daemon.cli import main
from hands.sessions.audit import segment
from hands.sessions.home import Home
from fixtures import pins

RELEASE = b"#!/bin/sh\necho 'the pinned release'\n"
OTHER = b"#!/bin/sh\necho 'another release'\n"
# Releases that are never the pin, whatever version it is moved to.
ANOTHER_VERSION = f"{pinned.PIN.version}.1"
AN_OLDER_VERSION = f"{pinned.PIN.version}.0"
# RELEASE's bytes pinned in place of the real release's, under the real version.
PIN = pins.of(RELEASE)
VERSION = PIN.version


def placed(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    path.chmod(0o755)
    return path


def test_the_brain_runs_exactly_the_pinned_version_in_its_own_directory(tmp_path: Path) -> None:
    config_dir = tmp_path / "brain"
    placed(config_dir / "pinned-claude" / VERSION / "claude", RELEASE)
    # Another version beside it is never the brain's.
    placed(config_dir / "pinned-claude" / ANOTHER_VERSION / "claude", RELEASE)
    assert PIN.verified(config_dir) == config_dir / "pinned-claude" / VERSION / "claude"


def test_two_pins_in_one_process_each_weigh_a_brain_against_their_own_checksums(tmp_path: Path) -> None:
    config_dir = tmp_path / "brain"
    copy = placed(PIN.claude(config_dir), RELEASE)
    other = pins.of(OTHER)
    assert PIN.verified(config_dir) == copy
    with pytest.raises(pinned.Unpinned, match=f"hashes to {hashlib.sha256(RELEASE).hexdigest()}, not the {hashlib.sha256(OTHER).hexdigest()}"):
        other.verified(config_dir)
    # Weighing one leaves the other as it was.
    assert PIN.verified(config_dir) == copy


def test_a_brain_whose_pinned_release_is_not_installed_refuses_naming_the_version_and_the_install(tmp_path: Path) -> None:
    config_dir = tmp_path / "brain"
    # Another version in its place is not the pin, and is never run for it.
    placed(config_dir / "pinned-claude" / AN_OLDER_VERSION / "claude", RELEASE)
    expected = config_dir / "pinned-claude" / VERSION / "claude"
    with pytest.raises(pinned.Unpinned) as refused:
        PIN.verified(config_dir)
    assert str(refused.value) == f"Claude Code {VERSION}, the brain's, is not installed at {expected}: run `hands install-brain`"


def test_a_copy_that_is_not_the_pinned_release_refuses_naming_its_checksum_and_the_install(tmp_path: Path) -> None:
    config_dir = tmp_path / "brain"
    copy = placed(PIN.claude(config_dir), OTHER)
    with pytest.raises(pinned.Unpinned) as refused:
        PIN.verified(config_dir)
    assert str(refused.value) == (
        f"{copy} hashes to {hashlib.sha256(OTHER).hexdigest()}, not the {hashlib.sha256(RELEASE).hexdigest()} of Claude Code {VERSION}, the brain's: run `hands install-brain` to fetch it again"
    )
    # Written over with the pin, it is the brain's: what was hashed before is not taken for what is there now.
    placed(copy, RELEASE)
    assert PIN.verified(config_dir) == copy
    # Nor is it when other bytes of the same length are written into the same file and its modified time is set back.
    held = copy.stat()
    copy.write_bytes(RELEASE.replace(b"pinned", b"PINNED"))
    os.utime(copy, ns=(held.st_atime_ns, held.st_mtime_ns))
    assert (copy.stat().st_ino, copy.stat().st_size, copy.stat().st_mtime_ns) == (held.st_ino, held.st_size, held.st_mtime_ns)
    with pytest.raises(pinned.Unpinned, match="hashes to"):
        PIN.verified(config_dir)


def test_nothing_on_path_or_in_the_persons_own_install_is_the_brains_claude(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The person's claude, as Claude Code installs itself, holding the very bytes of the pin: on PATH, at
    # ~/.local/bin/claude, and in ~/.local/share/claude/versions.
    home = tmp_path / "person"
    versions = placed(home / ".local" / "share" / "claude" / "versions" / VERSION, RELEASE)
    (home / ".local" / "bin").mkdir(parents=True)
    (home / ".local" / "bin" / "claude").symlink_to(versions)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", f"{home / '.local' / 'bin'}{os.pathsep}{os.environ['PATH']}")
    config_dir = tmp_path / "brain"
    with pytest.raises(pinned.Unpinned, match="is not installed at"):
        PIN.verified(config_dir)


def test_a_copy_that_cannot_be_read_or_run_refuses_naming_the_install(tmp_path: Path) -> None:
    config_dir = tmp_path / "brain"
    copy = placed(PIN.claude(config_dir), RELEASE)
    # Not a file that can be read: refused with what reading it raised, not raised as it.
    copy.chmod(0o000)
    with pytest.raises(pinned.Unpinned, match=r"cannot be read as Claude Code .*Permission denied.*: run `hands install-brain` to fetch it again"):
        PIN.verified(config_dir)
    # The pin's bytes, but not a file this process may run: refused, never left to fail at the spawn.
    copy.chmod(0o644)
    with pytest.raises(pinned.Unpinned) as refused:
        PIN.verified(config_dir)
    assert str(refused.value) == f"{copy}, Claude Code {VERSION}, the brain's, cannot be run: run `hands install-brain` to fetch it again"
    # Where the version's directory should be, a file: the same.
    copy.unlink()
    copy.parent.rmdir()
    copy.parent.write_bytes(b"")
    with pytest.raises(pinned.Unpinned, match=r"cannot be read as Claude Code .*Not a directory.*: run `hands install-brain`"):
        PIN.verified(config_dir)


@dataclass(frozen=True)
class Served:
    """A stand-in for Anthropic's release downloads: the source it is, and the paths asked of it."""

    source: pinned.Source
    asked: list[str]


@pytest.fixture
async def releases() -> AsyncIterator[Served]:
    """A stand-in for Anthropic's release downloads, serving RELEASE for every build."""
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
    yield Served(pinned.released(f"http://127.0.0.1:{port}"), asked)
    await runner.cleanup()


@pytest.mark.outside
async def test_install_fetches_the_pinned_build_once_and_replaces_a_copy_that_is_not_it(tmp_path: Path, releases: Served) -> None:
    config_dir = tmp_path / "brain"
    built = PIN.build(platform.machine())
    target = config_dir / "pinned-claude" / VERSION / "claude"
    assert await pinned.install(config_dir, PIN, releases.source) == pinned.Installed(target, unpinned=f"Claude Code {VERSION}, the brain's, is not installed at {target}: run `hands install-brain`")
    assert releases.asked == [f"/{VERSION}/{built}/claude"]
    assert PIN.verified(config_dir).read_bytes() == RELEASE and os.access(PIN.verified(config_dir), os.X_OK)
    # There already, it is not fetched again.
    assert await pinned.install(config_dir, PIN, releases.source) == pinned.Installed(PIN.claude(config_dir), unpinned=None)
    assert len(releases.asked) == 1
    # A copy that is not the pin is fetched again, whole, saying what the copy there hashed to.
    PIN.claude(config_dir).write_bytes(OTHER)
    installed = await pinned.install(config_dir, PIN, releases.source)
    assert installed.fetched and installed.unpinned is not None and f"hashes to {hashlib.sha256(OTHER).hexdigest()}" in installed.unpinned
    assert PIN.claude(config_dir).read_bytes() == RELEASE and len(releases.asked) == 2


@pytest.mark.outside
async def test_a_download_that_is_not_the_pinned_release_is_refused_and_leaves_nothing(tmp_path: Path, releases: Served) -> None:
    config_dir = tmp_path / "brain"
    with pytest.raises(pinned.Uninstallable, match=f"hashes to {hashlib.sha256(RELEASE).hexdigest()}, not the {hashlib.sha256(OTHER).hexdigest()} Claude Code {VERSION} is pinned to"):
        await pinned.install(config_dir, pins.of(OTHER), releases.source)
    assert list(PIN.claude(config_dir).parent.iterdir()) == []


@pytest.mark.outside
async def test_install_replaces_a_copy_of_the_pin_that_cannot_be_run(tmp_path: Path, releases: Served) -> None:
    config_dir = tmp_path / "brain"
    placed(PIN.claude(config_dir), RELEASE).chmod(0o644)
    installed = await pinned.install(config_dir, PIN, releases.source)
    assert installed.unpinned is not None and installed.unpinned.endswith("cannot be run: run `hands install-brain` to fetch it again")
    assert PIN.verified(config_dir) == PIN.claude(config_dir) and len(releases.asked) == 1


@pytest.mark.outside
async def test_two_installs_at_once_each_write_their_own_download_and_leave_the_pin(tmp_path: Path, releases: Served) -> None:
    config_dir = tmp_path / "brain"
    first, second = await asyncio.gather(pinned.install(config_dir, PIN, releases.source), pinned.install(config_dir, PIN, releases.source))
    assert first.fetched and second.fetched and len(releases.asked) == 2
    # Neither removed or renamed the other's download, and what is left is the pin alone.
    assert list(PIN.claude(config_dir).parent.iterdir()) == [PIN.claude(config_dir)]
    assert PIN.verified(config_dir) == PIN.claude(config_dir)


def serving(*chunks: bytes | Exception) -> pinned.Source:
    """A source of the pin other than Anthropic's release, writing `chunks` for every build, or raising one that is an
    exception, and failing as a connection does."""

    async def fetch(url: str, write: pinned.Write) -> None:
        for chunk in chunks:
            if isinstance(chunk, Exception):
                raise chunk
            write(chunk)

    return pinned.Source(url=lambda _version, built: f"elsewhere:{built}", fetch=fetch, failures=(ConnectionError,))


async def test_install_holds_every_source_to_the_pin_and_leaves_nothing_of_one_that_is_not_it(tmp_path: Path) -> None:
    config_dir = tmp_path / "brain"
    built = PIN.build(platform.machine())
    with pytest.raises(pinned.Uninstallable) as refused:
        await pinned.install(config_dir, PIN, serving(OTHER[:4], OTHER[4:]))
    assert str(refused.value) == f"elsewhere:{built} hashes to {hashlib.sha256(OTHER).hexdigest()}, not the {hashlib.sha256(RELEASE).hexdigest()} Claude Code {VERSION} is pinned to"
    assert list(PIN.claude(config_dir).parent.iterdir()) == []
    # A failure the source names is its fetch's, an OSError among them, never the disk's.
    with pytest.raises(pinned.Uninstallable) as refused:
        await pinned.install(config_dir, PIN, serving(RELEASE[:4], ConnectionResetError(54, "Connection reset by peer")))
    assert str(refused.value) == f"could not fetch elsewhere:{built}: ConnectionResetError(54, 'Connection reset by peer')"
    assert list(PIN.claude(config_dir).parent.iterdir()) == []
    # One it does not name is a defect, raised as itself.
    with pytest.raises(TypeError, match="a defect"):
        await pinned.install(config_dir, PIN, serving(RELEASE[:4], TypeError("a defect")))
    assert list(PIN.claude(config_dir).parent.iterdir()) == []
    # Its bytes the pin's, written in pieces, it is the brain's.
    installed = await pinned.install(config_dir, PIN, serving(RELEASE[:4], RELEASE[4:]))
    assert installed.fetched and PIN.verified(config_dir).read_bytes() == RELEASE


@dataclass(frozen=True)
class Stalling:
    """A stand-in for Anthropic's release downloads: the source it is, and an event set once it has sent all it sends."""

    source: pinned.Source
    began: asyncio.Event


@pytest.fixture
async def stalling() -> AsyncIterator[Stalling]:
    """A stand-in for Anthropic's release downloads that sends the start of a release and then nothing."""
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
    yield Stalling(pinned.released(f"http://127.0.0.1:{port}"), began)
    done.set()
    await runner.cleanup()


@pytest.mark.outside
async def test_a_stalled_download_fails_the_install_and_leaves_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stalling: Stalling) -> None:
    monkeypatch.setattr(pinned, "STALL_SECONDS", 0.5)
    config_dir = tmp_path / "brain"
    with pytest.raises(pinned.Uninstallable, match="could not fetch"):
        await pinned.install(config_dir, PIN, stalling.source)
    assert stalling.began.is_set() and list(PIN.claude(config_dir).parent.iterdir()) == []


@pytest.mark.outside
async def test_an_install_cut_short_leaves_nothing(tmp_path: Path, stalling: Stalling) -> None:
    config_dir = tmp_path / "brain"
    installing = asyncio.create_task(pinned.install(config_dir, PIN, stalling.source))
    await stalling.began.wait()
    # The start of the release has reached the disk by the time it is cut short.
    while not list(PIN.claude(config_dir).parent.glob("claude.*.partial")):
        await asyncio.sleep(0.01)
    installing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await installing
    assert list(PIN.claude(config_dir).parent.iterdir()) == []


def test_install_brain_says_what_it_did_and_is_one_event_each_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    built = PIN.build(platform.machine())
    home = Home(tmp_path / "home")
    target = placed(PIN.claude(home.brain), RELEASE)
    assert main(["--home", str(home.root), "install-brain"], pin=PIN) == 0
    assert capsys.readouterr().out == f"already there: Claude Code {VERSION}, the brain's, at {target}\n"
    # A release that cannot be fetched fails the install, saying why.
    target.unlink()
    monkeypatch.setattr(pinned, "RELEASE", pinned.released("http://127.0.0.1:1"))
    assert main(["--home", str(home.root), "install-brain"], pin=PIN) == 1
    unfetched = capsys.readouterr().err
    assert unfetched.startswith(f"hands install-brain: could not fetch http://127.0.0.1:1/{VERSION}/{built}/claude: ")
    # One that fetches says so, and its event why the copy there was not the pin.
    unpinned = f"Claude Code {VERSION}, the brain's, is not installed at {target}: run `hands install-brain`"

    async def fetched(config_dir: Path, pin: pinned.Pin, source: pinned.Source) -> pinned.Installed:
        return pinned.Installed(placed(pin.claude(config_dir), RELEASE), unpinned)

    monkeypatch.setattr(pinned, "install", fetched)
    assert main(["--home", str(home.root), "install-brain"], pin=PIN) == 0
    assert capsys.readouterr().out == f"fetched: Claude Code {VERSION}, the brain's, at {target}\n"
    # [LAW:nothing-unseen] each install's event: the release it pins, the build of it this Mac runs, where it put it,
    # whether it fetched it, and why.
    events = [line for line in (json.loads(line) for line in segment(home.audit, 0).read_text().splitlines()) if line.get("event") == "brain.install"]
    assert [(event["outcome"], event["facts"]) for event in events] == [
        ("ok", {"version": VERSION, "build": built, "claude": str(target), "fetched": False, "unpinned": None}),
        ("failed", {"version": VERSION, "build": built}),
        ("ok", {"version": VERSION, "build": built, "claude": str(target), "fetched": True, "unpinned": unpinned}),
    ]
    # The failed one's command says why, in the words it said it.
    commands = [line for line in (json.loads(line) for line in segment(home.audit, 0).read_text().splitlines()) if line.get("event") == "hands.command"]
    assert [command["error"] for command in commands] == [None, f"exited 1: {unfetched.removeprefix('hands install-brain: ').removesuffix('\n')}", None]
