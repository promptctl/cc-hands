"""The brain's pin as a test fakes it, in its own process or in a hands run of its own: bytes of the test's own as the
release, handed to the start as the pin it runs."""

import dataclasses
import hashlib
from pathlib import Path

from hands.brain import pinned


def of(release: bytes) -> pinned.Pin:
    """The pin whose every build is `release`, at the version hands runs."""
    checksum = hashlib.sha256(release).hexdigest()
    return dataclasses.replace(pinned.PIN, arm64=checksum, x64=checksum)


_RAN = ".ran"


def ran(script: Path) -> Path:
    """The mark MARKING, run as `script`, leaves beside it."""
    return script.with_name(f"{script.name}{_RAN}")


# What a home laid out holds in its pinned claude's place: run, it leaves a mark beside itself, so a test can tell
# whether anything ran it. The same bytes in every home, so one pin passes them all.
MARKING = f'#!/bin/sh\ntouch "$0{_RAN}"\n'.encode()
LAID_OUT = of(MARKING)
