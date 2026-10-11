"""hands.app's logic library decides and never acts: it imports nothing, and only each party's adapter reaches it."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
LIBRARY = APP / "HandsLogic" / "Sources"

# Each framework that reaches a party outside hands, and the one adapter allowed to import it.
ADAPTERS = {"AVFoundation": "privacy.swift", "IOKit": "privacy.swift"}


def imported(source: str) -> set[str]:
    """The modules a Swift source imports, `IOKit.hid` as IOKit."""
    return set(re.findall(r"^[ \t]*(?:@\w+[ \t]+)?import[ \t]+(?:\w+[ \t]+)?(\w+)", source, re.MULTILINE))


def test_the_library_imports_nothing() -> None:
    sources = sorted(LIBRARY.rglob("*.swift"))
    assert sources
    assert {source.name: imported(source.read_text()) for source in sources if imported(source.read_text())} == {}


def test_a_party_is_reached_only_through_its_adapter() -> None:
    sources = sorted(APP.glob("*.swift"))
    assert sources
    reached = {
        (framework, source.name)
        for source in sources
        for framework in imported(source.read_text()) & ADAPTERS.keys()
    }
    assert {(framework, name) for framework, name in reached if ADAPTERS[framework] != name} == set()
    assert reached == set(ADAPTERS.items())


def test_the_check_reads_every_form_of_import() -> None:
    source = "import AppKit\nimport IOKit.hid\n@preconcurrency import AVFoundation\n  import struct Foundation.URL\n// import Nothing\n"
    assert imported(source) == {"AppKit", "IOKit", "AVFoundation", "Foundation"}
