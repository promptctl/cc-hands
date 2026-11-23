"""The brain's pin as a test fakes it, in its own process or in a hands run of its own: bytes of the test's own as the
release, handed to the start as the pin it runs."""

import dataclasses
import hashlib

from hands.brain import pinned


def of(release: bytes) -> pinned.Pin:
    """The pin whose every build is `release`, at the version hands runs, so its claude is where PIN puts the brain's."""
    return dataclasses.replace(pinned.PIN, checksums=dict.fromkeys(pinned.PIN.checksums, hashlib.sha256(release).hexdigest()))


# What a home laid out holds in its pinned claude's place: run, it leaves a mark beside itself, so a test can tell
# whether anything ran it. The same bytes in every home, so one pin passes them all.
MARKING = b'#!/bin/sh\ntouch "$0.ran"\n'
LAID_OUT = of(MARKING)
