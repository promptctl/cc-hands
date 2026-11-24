"""The runtime the phone page runs the wake word's models on: the two files hands pins, fetched from the release at their
version by a release of their own (what a release does is `hands.voice.pinnedfile`'s, tested there). The fetch is the
test's: no socket opens and no thread runs."""

from pathlib import Path

from fakefetch import Fetching

from hands.voice.ortweb import FILES, PREFIX, RELEASE, VERSION, File, Runtime

PINNED = {"runtime.mjs": File("digest of the script", "text/javascript"), "runtime.wasm": File("digest of the webassembly", "application/wasm")}


async def test_each_file_is_fetched_from_the_runtimes_release_at_its_digest_and_version(tmp_path: Path) -> None:
    fetching = Fetching()
    await Runtime(tmp_path, PINNED, fetching).ensured()
    assert [(directory, pin.name, pin.url, pin.digest, pin.at) for directory, pin in fetching.calls] == [
        (tmp_path, "runtime.mjs", f"{RELEASE}/runtime.mjs", "digest of the script", f"onnxruntime-web {VERSION}"),
        (tmp_path, "runtime.wasm", f"{RELEASE}/runtime.wasm", "digest of the webassembly", f"onnxruntime-web {VERSION}"),
    ]


def test_the_runtime_hands_pins_is_the_two_files_the_page_loads_each_served_as_what_it_is_under_a_name_that_carries_its_version() -> None:
    assert ({name: file.content_type for name, file in FILES.items()}, PREFIX) == (
        {"ort.wasm.bundle.min.mjs": "text/javascript", "ort-wasm-simd-threaded.wasm": "application/wasm"},
        f"/ort/{VERSION}/",
    )
