"""The run is offline once its fetches are done (conftest, offline.py): nothing a test does, nor anything a test starts, connects
past loopback, and loopback is as open as it was; and the machine's Claude Code policy is nowhere either looks."""

import errno
import os
import re
import socket
import subprocess
import sys
from pathlib import Path

import pytest

import offline
from offline import MANAGED_PREFERENCES, POLICY_FOLDER, POLICY_PLIST, UNROUTED, policy_found, sandboxable


def test_a_connection_a_test_makes_past_loopback_is_refused_by_the_os() -> None:
    with pytest.raises(PermissionError) as refused:
        socket.create_connection(UNROUTED, timeout=5)
    assert refused.value.errno == errno.EPERM


def test_a_connection_a_test_s_child_makes_past_loopback_is_refused_by_the_os() -> None:
    tried = subprocess.run(["nc", "-zv", "-w", "5", *map(str, UNROUTED)], capture_output=True, text=True, timeout=30)
    assert tried.returncode != 0
    assert "Operation not permitted" in tried.stderr, tried.stderr


def test_a_name_off_this_machine_is_not_resolved() -> None:
    with pytest.raises(socket.gaierror):
        socket.getaddrinfo("example.com", 443)


def test_loopback_is_reachable_by_address_and_by_localhost() -> None:
    with socket.create_server(("127.0.0.1", 0)) as server:
        port = server.getsockname()[1]
        for host in ("127.0.0.1", "localhost"):
            socket.create_connection((host, port), timeout=5).close()


def test_the_machine_s_claude_code_policy_is_not_found_by_a_test_or_its_child() -> None:
    assert policy_found() == []
    assert not os.access(POLICY_FOLDER, os.F_OK)
    assert subprocess.run(["test", "-e", str(POLICY_FOLDER)], timeout=30).returncode == 1
    # CI writes a policy into its folder and both plists before the run, so there each run shows the profile hides one
    # that is there: the folders that hold them still list them, as the profile leaves a listing alone.
    if os.environ.get("CI") == "true":
        assert POLICY_FOLDER.name in os.listdir(POLICY_FOLDER.parent)
        assert len([folder for folder, _, files in os.walk(MANAGED_PREFERENCES) if POLICY_PLIST in files]) == 2


def test_a_policy_in_any_of_its_places_is_found_by_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder, preferences = tmp_path / "ClaudeCode", tmp_path / "Managed Preferences"
    monkeypatch.setattr(offline, "POLICY_FOLDER", folder)
    monkeypatch.setattr(offline, "MANAGED_PREFERENCES", preferences)
    assert policy_found() == []
    plists = [preferences / POLICY_PLIST, preferences / "an account" / POLICY_PLIST]
    (preferences / "an account").mkdir(parents=True)
    folder.mkdir()
    for plist in plists:
        plist.touch()
    assert policy_found() == sorted(map(str, [folder, *plists]))


def test_a_place_that_refuses_the_look_is_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(offline, "POLICY_FOLDER", tmp_path / "refused" / "ClaudeCode")
    monkeypatch.setattr(offline, "MANAGED_PREFERENCES", tmp_path / "Managed Preferences")
    (tmp_path / "refused").mkdir(mode=0)
    try:
        assert policy_found() == [str(tmp_path / "refused" / "ClaudeCode")]
    finally:
        (tmp_path / "refused").chmod(0o700)


@pytest.mark.parametrize("account", ["", "an account"])
def test_a_folder_of_managed_preferences_that_refuses_the_look_is_found(account: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(offline, "POLICY_FOLDER", tmp_path / "ClaudeCode")
    monkeypatch.setattr(offline, "MANAGED_PREFERENCES", tmp_path / "Managed Preferences")
    refused = tmp_path / "Managed Preferences" / account
    refused.mkdir(parents=True)
    refused.chmod(0)
    try:
        assert policy_found() == [str(refused)]
    finally:
        refused.chmod(0o700)


def test_a_run_that_finds_a_policy_stops_naming_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(offline, "POLICY_FOLDER", tmp_path)
    with pytest.raises(pytest.UsageError, match=re.escape(f"Claude Code policy found: {[str(tmp_path)]}")):
        offline.hold()


@pytest.mark.parametrize("release", ["15.0", "15.7.9", "26.5.2"])
def test_a_macos_whose_sandbox_holds_loopback_starts_the_run(release: str) -> None:
    sandboxable(release)


def test_a_macos_whose_sandbox_refuses_loopback_at_random_stops_the_run_saying_so() -> None:
    with pytest.raises(pytest.UsageError, match=r"refuses loopback at random before macOS 15: this is macOS 14\.8\.9$"):
        sandboxable("14.8.9")


def test_a_run_started_inside_the_held_run_is_held_and_says_how() -> None:
    held = subprocess.run([sys.executable, "-c", "import offline; print(offline.hold())"], cwd=Path(offline.__file__).parent, capture_output=True, text=True, timeout=30)
    assert held.returncode == 0, held.stderr
    assert held.stdout.startswith(str(offline.OFFLINE)), held.stdout
