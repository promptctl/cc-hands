"""How a `hands` command ends: 0, or one of the nonzero exits the parser help documents with what the command decided.

[LAW:types-are-the-program] a command returns an Exit, never a bare int, so a failure without its reason, or a failure
that exits 0, cannot be written: the hands.command event that wraps it says why from the value it returns.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Failed:
    """A command that exited nonzero: the code, and why, in the words the command said it."""

    code: Literal[1, 2, 3]
    why: str


Exit = Literal[0] | Failed


def code_of(ended: Exit) -> int:
    """The code the process exits with."""
    match ended:
        case Failed(code=code):
            return code
        case 0:
            return 0
