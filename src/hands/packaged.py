"""Where in hands' package each binary it carries is, by the package of fritter/'s module it is built from.

[LAW:one-source-of-truth] hatch_build.py builds each binary to its place here, and hands finds each one there. The
module imports nothing of hands', because the build reads it by its path: hands cannot be imported while it is being
built.
"""

from pathlib import Path, PurePosixPath
from typing import Final

# fritter, which every session is started under.
FRITTER: Final = PurePosixPath("bin/fritter")
# brioche, the brain's HTTPS proxy: a package of its own, so fritter's binary links none of it, and a change to brioche
# restarts no session.
BRIOCHE: Final = PurePosixPath("bin/brioche")
BUILT: Final = {".": FRITTER, "./brioche": BRIOCHE}


def installed(at: PurePosixPath) -> Path:
    """The binary at `at` in hands' package as this hands is installed: a wheel's install and a checkout's editable
    install alike. It changes only when the package is rebuilt."""
    return Path(__file__).resolve().parent / at


class Unpackaged(Exception):
    """hands' package carries no binary where it should: a hands built without it, which only building hands again mends."""


def found(binary: Path) -> Path:
    """`binary`, an installed binary of hands' package, or Unpackaged naming the rebuild.

    [LAW:single-enforcer] the one check that this hands carries a binary of its own, for everything that runs or copies
    one: a hands built without fritter or brioche is refused in the same words wherever either is wanted."""
    if not binary.is_file():
        raise Unpackaged(f"hands' package carries no {binary.name} at {binary}: install hands again, or in a checkout, `uv sync --reinstall-package hands`")
    return binary
