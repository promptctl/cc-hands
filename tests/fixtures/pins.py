"""The brain's pin as a test fakes it, in its own process or in a hands run of its own: bytes of the test's own as the release."""

import hashlib

from hands.brain import pinned


def checksums(release: bytes) -> dict[str, str]:
    """pinned.CHECKSUMS for a pin whose every build is `release`."""
    return dict.fromkeys(pinned.CHECKSUMS, hashlib.sha256(release).hexdigest())
