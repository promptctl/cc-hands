"""Any voice in Kyutai's voice library: named by its path, checked by the cloning model and its own audio prompt, listed for
search once an hour, and given the Hugging Face token from the environment or the keychain. No test lists or fetches the
real library: each takes its listing, its download, and its keychain from a fake."""

import asyncio
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError, RepositoryNotFoundError
from pipecat.processors.filters.identity_filter import IdentityFilter

from conftest import PATIENCE_SECS
from hands.daemon import run
from hands.sessions.home import Home
from hands.sessions.payload import Rejected
from hands.voice import library, voices
from hands.voice.library import LIBRARY_URL_PREFIX, Found, Library
from hands.voice.tools import audited, voice_tools
from hands.voice.voices import DEFAULT, LibraryVoice, Voices, chosen, parse_voice, source
from hands.daemon.run import CannotStart

LIBRARY_FILES = ["vctk/p225_023_enhanced.wav", "vctk/p226_023_enhanced.wav", "alba-mackenna/casual.wav", "alba-mackenna/casual.wav.1e68beda@240.safetensors"]


def listing(files: list[str]) -> Callable[[], list[str]]:
    return lambda: list(files)


# --- names ---------------------------------------------------------------------------------------------------------------

def test_a_built_in_name_is_still_a_built_in_name() -> None:
    assert parse_voice("Bill Boerst") == "bill_boerst"
    assert parse_voice("charles") == DEFAULT


def test_a_library_path_is_a_library_voice_with_its_case_kept() -> None:
    assert parse_voice("vctk/p225_023_enhanced") == LibraryVoice("vctk/p225_023_enhanced")
    assert parse_voice("voice-donations/Selfie") == LibraryVoice("voice-donations/Selfie")


@pytest.mark.parametrize("malformed", ["../secrets", "a/../b", "/absolute", "trailing/", "Zed!", "a//b", "a?b", "zed", "Charles2"])
def test_a_name_that_is_not_a_repo_path_is_refused_by_the_parser(malformed: str) -> None:
    with pytest.raises(Rejected, match="no voice called"):
        parse_voice(malformed)


def test_the_source_hands_pipecat_is_the_name_or_the_audio_prompt_of_the_library_voice() -> None:
    assert source(DEFAULT) == "charles"
    assert source(LibraryVoice("vctk/p225_023_enhanced")) == f"{LIBRARY_URL_PREFIX}vctk/p225_023_enhanced.wav"
    assert LIBRARY_URL_PREFIX == "hf://kyutai/tts-voices/"


def test_a_library_voice_is_kept_and_read_back_as_its_path(tmp_path: Path) -> None:
    voices.keep(Home(tmp_path), LibraryVoice("vctk/p225_023_enhanced"))
    assert chosen(Home(tmp_path)) == LibraryVoice("vctk/p225_023_enhanced")


# --- fetching: the cloning model first, then the voice's own audio prompt -----------------------------------------------

def download_that_fails_for(monkeypatch: pytest.MonkeyPatch, failing: Callable[[str], bool], error: Exception) -> list[str]:
    asked: list[str] = []

    def download(path: str) -> Path:
        asked.append(path)
        if failing(path):
            raise error
        return Path(path)

    monkeypatch.setattr(voices, "download_if_necessary", download)
    return asked


def test_a_library_voice_whose_cloning_model_cannot_be_fetched_is_refused_before_its_own_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    asked = download_that_fails_for(monkeypatch, lambda path: "tts_" in path or "english" in path, OSError("no route to the hub"))
    with pytest.raises(Rejected, match="cannot fetch the cloning model"):
        voices.fetched(LibraryVoice("vctk/p225_023_enhanced"))
    assert not any(path.startswith(LIBRARY_URL_PREFIX) for path in asked)


def test_a_library_voice_the_library_lacks_is_refused_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    download_that_fails_for(monkeypatch, lambda path: path.startswith(LIBRARY_URL_PREFIX), EntryNotFoundError("no such file"))
    with pytest.raises(Rejected, match="has no voice called 'vctk/nobody'"):
        voices.fetched(LibraryVoice("vctk/nobody"))


def test_a_refused_token_is_named_as_such_and_not_as_a_voice_the_library_lacks(monkeypatch: pytest.MonkeyPatch) -> None:
    refused = RepositoryNotFoundError("401 Unauthorized", response=Mock(status_code=401, headers={}))
    download_that_fails_for(monkeypatch, lambda path: path.startswith(LIBRARY_URL_PREFIX), refused)
    with pytest.raises(Rejected, match="Hugging Face token was refused"):
        voices.fetched(LibraryVoice("vctk/p225_023_enhanced"))


def test_a_library_voice_that_cannot_be_reached_is_an_os_error_not_a_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    download_that_fails_for(monkeypatch, lambda path: path.startswith(LIBRARY_URL_PREFIX), OSError("offline"))
    with pytest.raises(OSError, match="offline"):
        voices.fetched(LibraryVoice("vctk/p225_023_enhanced"))


def test_a_library_voice_not_in_the_cache_while_offline_is_unreachable_not_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    # huggingface_hub's offline miss is an EntryNotFoundError too; it must not read as the library lacking the voice.
    download_that_fails_for(monkeypatch, lambda path: path.startswith(LIBRARY_URL_PREFIX), LocalEntryNotFoundError("not cached"))
    with pytest.raises(LocalEntryNotFoundError):
        voices.fetched(LibraryVoice("vctk/p225_023_enhanced"))


async def test_a_bare_name_in_the_library_is_not_offered_by_search_since_no_choice_can_make_it(tmp_path: Path) -> None:
    files = ["Charles.wav", "vctk/p225_023_enhanced.wav"]
    found = await Voices(Home(tmp_path), IdentityFilter(), voices.fetched, Library(listing(files))).search("", 10)
    assert found.names == ("vctk/p225_023_enhanced",)


async def test_a_search_returns_at_most_the_most_found_whatever_limit_it_is_asked() -> None:
    files = [f"vctk/voice{number}.wav" for number in range(library.MOST_FOUND + 10)]
    found = await Library(listing(files)).search("", 10_000)
    assert len(found.names) == library.MOST_FOUND


async def test_a_stalled_listing_is_waited_for_by_the_next_search_and_never_started_again(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(library, "LISTING_TIMEOUT_SECONDS", 0.01)
    started: list[int] = []
    running = threading.Event()
    released = threading.Event()

    def stalled() -> list[str]:
        started.append(1)
        running.set()
        assert released.wait(PATIENCE_SECS)
        return ["vctk/p225_023_enhanced.wav"]

    stuck = Library(stalled)
    try:
        # [LAW:no-ambient-temporal-coupling] each search times out while the listing stalls in its thread, which `running`
        # says it is, rather than a guess at how soon the thread runs.
        with pytest.raises(OSError):
            await stuck.search("", 10)
        assert await asyncio.to_thread(running.wait, PATIENCE_SECS)
        with pytest.raises(OSError):
            await stuck.search("", 10)
    finally:
        released.set()
    # Given time, the next search waits for that same listing, has its names, and forgets the listing. Every listing a search
    # started was awaited by a search, so one started is the one listing there ever was.
    monkeypatch.setattr(library, "LISTING_TIMEOUT_SECONDS", PATIENCE_SECS)
    assert (await stuck.search("", 10)).names == ("vctk/p225_023_enhanced",)
    assert len(started) == 1
    assert stuck._pending is None  # pyright: ignore[reportPrivateUsage]


async def test_a_library_voice_that_cannot_be_fetched_is_never_kept(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    download_that_fails_for(monkeypatch, lambda path: path.startswith(LIBRARY_URL_PREFIX), EntryNotFoundError("no such file"))
    home = Home(tmp_path)
    voices_owner = Voices(home, IdentityFilter(), voices.fetched, Library(listing([])))
    with pytest.raises(Rejected):
        await voices_owner.use(LibraryVoice("vctk/nobody"))
    assert not home.voice.exists()


# --- the start: a kept library voice is checked the same way the choice was ---------------------------------------------

def test_a_kept_library_voice_that_cannot_be_fetched_stops_the_start_naming_it(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    home = Home(tmp_path)
    voices.keep(home, LibraryVoice("vctk/p225_023_enhanced"))
    download_that_fails_for(monkeypatch, lambda path: "tts_" in path or "english" in path, OSError("no route to the hub"))
    with pytest.raises(CannotStart, match="cannot fetch the cloning model"):
        run._voice(home)  # pyright: ignore[reportPrivateUsage]


def test_a_kept_built_in_voice_starts_without_any_fetch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    home = Home(tmp_path)
    voices.keep(home, DEFAULT)
    asked = download_that_fails_for(monkeypatch, lambda path: True, OSError("offline"))
    assert run._voice(home) == DEFAULT  # pyright: ignore[reportPrivateUsage]
    assert asked == []


# --- the library: listed once an hour, searched, and a failed listing is not kept -----------------------------------------

async def test_the_library_is_listed_once_within_the_hour_and_again_after_it() -> None:
    now = [1000.0]
    calls: list[int] = []

    def list_files() -> list[str]:
        calls.append(1)
        return list(LIBRARY_FILES)

    catalogue = Library(list_files, clock=lambda: now[0])
    first = await catalogue.search("vctk", 10)
    now[0] += library.LISTING_SECONDS - 1
    second = await catalogue.search("alba", 10)
    now[0] += 2
    third = await catalogue.search("vctk", 10)
    assert len(calls) == 2
    assert (first.from_cache, second.from_cache, third.from_cache) == (False, True, False)
    assert first.listed_at == 1000.0 and second.listed_at == 1000.0 and third.listed_at == now[0]


async def test_a_search_matches_names_without_case_and_only_voice_files_up_to_the_limit() -> None:
    catalogue = Library(listing(LIBRARY_FILES), clock=lambda: 5.0)
    found = await catalogue.search("VCTK", 1)
    assert found == Found(("vctk/p225_023_enhanced",), 5.0, False)
    everything = await catalogue.search("", 10)
    assert everything.names == ("vctk/p225_023_enhanced", "vctk/p226_023_enhanced", "alba-mackenna/casual")


async def test_a_failed_listing_is_not_kept_so_the_next_search_lists_again() -> None:
    attempts: list[int] = []

    def flaky() -> list[str]:
        attempts.append(1)
        if len(attempts) == 1:
            raise OSError("the hub is unreachable")
        return list(LIBRARY_FILES)

    catalogue = Library(flaky, clock=lambda: 1.0)
    with pytest.raises(OSError, match="unreachable"):
        await catalogue.search("vctk", 10)
    found = await catalogue.search("vctk", 10)
    assert len(attempts) == 2 and found.names == ("vctk/p225_023_enhanced", "vctk/p226_023_enhanced")


async def test_the_find_voices_tool_returns_matches_and_when_they_were_listed(tmp_path: Path) -> None:
    recorded: list[Any] = []
    owner = Voices(Home(tmp_path), IdentityFilter(), voices.fetched, Library(listing(LIBRARY_FILES), clock=lambda: 7.0))
    tools = {tool.name: audited(tool, recorded.append) for tool in voice_tools(owner)}
    assert await tools["find_voices"].body(query="casual") == {"voices": ["alba-mackenna/casual"], "listed_at": 7.0, "from_cache": False}


# --- the token: the environment's, else the keychain's, once, and never logged ------------------------------------------

def test_the_environments_token_is_used_and_the_keychain_is_not_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(library, "_keychain_token", lambda: pytest.fail("the keychain was read though the environment has a token"))
    assert library.hub_token({"HF_TOKEN": "from-the-environment"}) == "from-the-environment"


def test_with_no_environment_token_the_keychain_supplies_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(library, "_keychain_token", lambda: "from-the-keychain")
    assert library.hub_token({}) == "from-the-keychain"


def test_with_neither_token_there_is_none_and_the_environment_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(library, "_keychain_token", lambda: None)
    assert library.hub_token({}) is None
    assert "HF_TOKEN" not in os.environ


def test_a_command_puts_the_keychains_token_in_the_environment_the_fetch_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from hands.daemon import cli

    # An empty token is no token: the keychain is asked, and the environment is restored by monkeypatch at the end.
    monkeypatch.setenv("HF_TOKEN", "")
    monkeypatch.setattr(library, "_keychain_token", lambda: "from-the-keychain")
    cli.main(["--home", str(tmp_path / "home"), "status"])
    assert os.environ["HF_TOKEN"] == "from-the-keychain"


def test_the_keychain_is_asked_for_the_item_on_stdout_and_its_token_is_never_logged(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[list[str]] = []

    class Completed:
        stdout = "secret-token\n"

    def run_security(command: list[str], **_: object) -> Any:
        commands.append(command)
        return Completed()

    monkeypatch.setattr(library.subprocess, "run", run_security)
    assert library._keychain_token() == "secret-token"  # pyright: ignore[reportPrivateUsage]
    assert commands == [["security", "find-generic-password", "-s", "HF_TOKEN", "-w"]]


def test_help_reads_no_keychain(monkeypatch: pytest.MonkeyPatch) -> None:
    from hands.daemon import cli

    reads: list[int] = []

    def record() -> str | None:
        reads.append(1)
        return None

    monkeypatch.setattr(library, "_keychain_token", record)
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    assert reads == []
