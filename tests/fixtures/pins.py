"""The brain's pin as a test fakes it, in its own process or in a hands run of its own: a script in the pin's place."""

import hashlib
from pathlib import Path

from hands.brain import pinned


def checksums(script: Path) -> dict[str, str]:
    """pinned.CHECKSUMS for a pin that is `script`, as its bytes are now, on every build."""
    with script.open("rb") as held:
        return dict.fromkeys(pinned.CHECKSUMS, hashlib.file_digest(held, "sha256").hexdigest())
