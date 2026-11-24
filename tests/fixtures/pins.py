"""The brain's pin as a test fakes it, in its own process or in a hands run of its own: bytes of the test's own as the
release, handed to the start as the pin it runs."""

import hashlib
from pathlib import Path

from hands.brain import pinned


def of(release: bytes) -> pinned.Pin:
    """The pin whose every build is `release`, at a version of its own named by its digest, as a release's version names
    its bytes: no two fake releases, nor a fake and PIN, share a place in a brain."""
    checksum = hashlib.sha256(release).hexdigest()
    return pinned.Pin(version=f"0.0.0-{checksum[:12]}", arm64=checksum, x64=checksum)


_RAN = ".ran"


def ran(script: Path) -> Path:
    """The mark MARKING, run as `script`, leaves beside it."""
    return script.with_name(f"{script.name}{_RAN}")


# What a home laid out holds in its pinned claude's place: run, it leaves a mark beside itself, so a test can tell
# whether anything ran it. The same bytes in every home, so one pin passes them all.
MARKING = f'#!/bin/sh\ntouch "$0{_RAN}"\n'.encode()
LAID_OUT = of(MARKING)
