"""The runtime the phone page runs the wake word's models on: the two files hands pins, from the release at their version
(what is done with a pin is `hands.voice.pinnedfile`'s, tested there)."""

from hands.voice.ortweb import FILES, PREFIX, RELEASE, VERSION


def test_the_runtime_hands_pins_is_the_two_files_the_page_loads_each_from_the_release_at_its_version_and_served_as_what_it_is() -> None:
    assert [(file.pin.name, file.pin.url, file.pin.at, len(file.pin.digest), file.content_type) for file in FILES] == [
        ("ort.wasm.bundle.min.mjs", f"{RELEASE}/ort.wasm.bundle.min.mjs", f"onnxruntime-web {VERSION}", 64, "text/javascript"),
        ("ort-wasm-simd-threaded.wasm", f"{RELEASE}/ort-wasm-simd-threaded.wasm", f"onnxruntime-web {VERSION}", 64, "application/wasm"),
    ]


def test_the_files_are_served_under_a_name_that_carries_their_version() -> None:
    assert PREFIX == f"/ort/{VERSION}/"
