"""ONNX Runtime Web, the runtime the phone page runs the wake word's models on (`hands.voice.phonewake`): its WebAssembly
build, which runs on one thread, so the page need not be cross-origin isolated. The build is two files, a script that
carries its own glue and the WebAssembly file it fetches from beside itself.

hands fetches both at VERSION, holds each to the digest it pins, and serves them: nothing the phone page runs comes from
another site.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from hands.voice.pinnedfile import Fetch, Pin, Release, fetched_off_the_loop

VERSION = "1.31.1"
RELEASE = f"https://cdn.jsdelivr.net/npm/onnxruntime-web@{VERSION}/dist"
SCRIPT = "ort.wasm.bundle.min.mjs"
WASM = "ort-wasm-simd-threaded.wasm"


@dataclass(frozen=True)
class File:
    """A file of the runtime: the digest it is held to, and the type it is served as. A browser runs a module script only
    when it is told it is one, and compiles WebAssembly only from the type it is told is WebAssembly."""

    digest: str
    content_type: str


# [LAW:one-source-of-truth] the files the page loads, each by its digest and its type.
FILES: Mapping[str, File] = {
    SCRIPT: File("51bd7168175767324ab0acfcfc4ca94a096a3826e3ebac9ccc4f195eb865f8dd", "text/javascript"),
    WASM: File("4b884c203b0fa48d057be3332c65513ff85aa3c5bddd6946acc0fab7ecda001d", "application/wasm"),
}
# Where the page asks for them. The version is in the name, so a browser keeps the 14.8 MB file for as long as hands runs
# this version, and a page that asks for another version is told there is none.
ROOT = "/ort/"
PREFIX = f"{ROOT}{VERSION}/"


class Runtime(Release):
    """The runtime's files in `directory`, fetched on the first ask for each (`hands.voice.pinnedfile.Release`)."""

    def __init__(self, directory: Path, files: Mapping[str, File] = FILES, fetch: Fetch = fetched_off_the_loop) -> None:
        super().__init__(directory, (Pin(name, f"{RELEASE}/{name}", file.digest, f"onnxruntime-web {VERSION}") for name, file in files.items()), fetch)
        self.files = files
