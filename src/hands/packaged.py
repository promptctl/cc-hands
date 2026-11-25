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
