"""cli.main as a fixture's stand-in for `hands` runs it, in a process of its own: on this Mac's processes, its indicators
and children spawned, and a Mac with no managed policy (fakemachine.py)."""

from collections.abc import Sequence

from fakemachine import UNMANAGED
from hands.clock import Clock
from hands.daemon import cli


def main_on_the_machine(argv: Sequence[str], clock: Clock) -> int:
    """[LAW:locality-or-seam] the one place a fixture's command meets the root's real parties, as tests/fakeparties.py
    is the one place a test's meets their fakes."""
    return cli.main(argv, clock, processes=cli.KERNEL, indicators=cli.SpawnedIndicators(), machine=UNMANAGED, children_on=cli.SpawnedChildren, pseudoterminals_on=cli.spawned_pseudoterminals)
