"""The runtime the phone page runs the wake word's models on: each of its files fetched on the first ask at its version,
served only from what the fetch returned, and fetched once. The fetch is the test's: no socket opens and no thread runs
(the file's digest is held by `hands.voice.pinnedfile`, tested there)."""

import asyncio
import urllib.error
from http.client import IncompleteRead
from pathlib import Path

import pytest

from hands.voice.ortweb import FILES, PREFIX, RELEASE, VERSION, Runtime, Unserved

PINNED = {"runtime.mjs": "digest of the script", "runtime.wasm": "digest of the webassembly"}


class Fetching:
    """The fetch hands makes, as the test answers it: each call's arguments, and what it does."""

    def __init__(self) -> None:
        self.calls = list[tuple[Path, str, str, str, str]]()
        self.failure: Exception | None = None
        # Set as a call begins, and the call finishes only once `release` is: so a test holds a fetch open.
        self.begun, self.release = asyncio.Event(), asyncio.Event()
        self.release.set()

    async def __call__(self, directory: Path, name: str, url: str, digest: str, at: str) -> Path:
        self.calls.append((directory, name, url, digest, at))
        self.begun.set()
        await self.release.wait()
        if self.failure is not None:
            raise self.failure
        return directory / name


async def test_a_file_is_fetched_from_its_release_at_its_digest_on_the_first_ask_and_held_there_after(tmp_path: Path) -> None:
    fetching = Fetching()
    runtime = Runtime(tmp_path, PINNED, fetching)
    first = await runtime.path("runtime.wasm")
    assert first == tmp_path / "runtime.wasm"
    assert fetching.calls == [(tmp_path, "runtime.wasm", f"{RELEASE}/runtime.wasm", "digest of the webassembly", f"onnxruntime-web {VERSION}")]
    assert await runtime.path("runtime.wasm") == first
    assert len(fetching.calls) == 1


async def test_each_file_is_fetched_for_itself(tmp_path: Path) -> None:
    fetching = Fetching()
    await Runtime(tmp_path, PINNED, fetching).ensured()
    assert [name for _, name, *_ in fetching.calls] == ["runtime.mjs", "runtime.wasm"]


@pytest.mark.parametrize("failure", [
    ValueError("runtime.wasm at onnxruntime-web 1.31.1 is aaa, not the bbb pinned"),
    urllib.error.URLError("the network is down"),
    IncompleteRead(b"", 10),
])
async def test_a_fetch_that_fails_is_unserved_with_the_reason_and_the_next_ask_fetches_again(tmp_path: Path, failure: Exception) -> None:
    fetching = Fetching()
    runtime = Runtime(tmp_path, PINNED, fetching)
    fetching.failure = failure
    with pytest.raises(Unserved) as unserved:
        await runtime.path("runtime.wasm")
    assert (f"onnxruntime-web {VERSION}'s runtime.wasm" in str(unserved.value), str(failure) in str(unserved.value)) == (True, True)
    fetching.failure = None
    assert await runtime.path("runtime.wasm") == tmp_path / "runtime.wasm"
    assert len(fetching.calls) == 2


async def test_asks_that_arrive_while_a_file_is_being_fetched_wait_for_that_fetch(tmp_path: Path) -> None:
    fetching = Fetching()
    fetching.release.clear()
    runtime = Runtime(tmp_path, PINNED, fetching)
    first = asyncio.create_task(runtime.path("runtime.wasm"))
    # The fetch is under way, held open by the test; the other asks arrive in it.
    await fetching.begun.wait()
    others = [asyncio.create_task(runtime.path("runtime.wasm")) for _ in range(3)]
    await asyncio.sleep(0)
    assert not any(ask.done() for ask in others)
    fetching.release.set()
    assert set(await asyncio.gather(first, *others)) == {tmp_path / "runtime.wasm"}
    assert len(fetching.calls) == 1


def test_the_runtime_hands_pins_is_the_two_files_the_page_loads_under_a_name_that_carries_its_version() -> None:
    assert (sorted(FILES), PREFIX) == (["ort-wasm-simd-threaded.wasm", "ort.wasm.bundle.min.mjs"], f"/ort/{VERSION}/")
