"""A stand-in a child process puts in place of one of hands' functions, held by the type checks to what it stands in for."""

import sys
from collections.abc import Callable


def replace[**P, R](original: Callable[P, R], stand_in: Callable[P, R]) -> None:
    """Puts `stand_in` where `original` is defined, for every later call through its module. [LAW:types-are-the-program]
    `stand_in` takes `original`'s parameters and gives its result, so a change to either fails make check, not the child."""
    setattr(sys.modules[original.__module__], original.__name__, stand_in)
