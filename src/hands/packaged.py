"""Where in hands' package the fritter binary it carries is.

[LAW:one-source-of-truth] hatch_build.py builds fritter to its place here, and hands finds it there. The module imports
nothing of hands, because the build reads it by its path: hands cannot be imported while it is being built.
"""

from pathlib import PurePosixPath
from typing import Final

# fritter, which every session is started under.
FRITTER: Final = PurePosixPath("bin/fritter")
