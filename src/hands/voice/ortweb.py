"""ONNX Runtime Web, the runtime the phone page runs the wake word's models on (`hands.voice.phonewake`): its WebAssembly
build, which runs on one thread, so the page need not be cross-origin isolated. The build is two files, a script that
carries its own glue and the WebAssembly file it fetches from beside itself.

hands fetches both at VERSION, holds each to the digest it pins, and serves them: nothing the phone page runs comes from
another site.
"""

from dataclasses import dataclass

from hands.voice.pinnedfile import Pin

VERSION = "1.31.1"
RELEASE = f"https://cdn.jsdelivr.net/npm/onnxruntime-web@{VERSION}/dist"


@dataclass(frozen=True)
class File:
    """A file of the runtime: the pin it is held to, and the type it is served as. A browser runs a module script only
    when it is told it is one, and compiles WebAssembly only from the type it is told is WebAssembly."""

    pin: Pin
    content_type: str


def _file(name: str, digest: str, content_type: str) -> File:
    return File(Pin(name, f"{RELEASE}/{name}", digest, f"onnxruntime-web {VERSION}"), content_type)


# [LAW:one-source-of-truth] the files the page loads: each is served at one route, by its name.
SCRIPT = _file("ort.wasm.bundle.min.mjs", "51bd7168175767324ab0acfcfc4ca94a096a3826e3ebac9ccc4f195eb865f8dd", "text/javascript")
WASM = _file("ort-wasm-simd-threaded.wasm", "4b884c203b0fa48d057be3332c65513ff85aa3c5bddd6946acc0fab7ecda001d", "application/wasm")
FILES = (SCRIPT, WASM)
# Where the page asks for them. The version is in the name, so a browser keeps the 14.8 MB file for as long as hands runs
# this version, and a page that asks for another version finds none.
PREFIX = f"/ort/{VERSION}/"
