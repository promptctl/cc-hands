"""`hands` as a test runs it: cli.main on a fake of every party the root binds, each test naming only those its behavior
rests on."""

from collections.abc import Sequence
from typing import TypedDict, Unpack

from fakeindicators import FakeIndicators
from fakeprocesses import processes
from fixtures.fakemachine import UNMANAGED
from hands.brain import configdir, pinned
from hands.clock import SYSTEM, Clock
from hands.daemon import cli
from hands.daemon.indicatorprocess import Indicators
from hands.sessions.processes import Processes


class Parties(TypedDict, total=False):
    """The parties a test names: the rest are the fakes main_on_fakes gives."""

    processes: Processes
    indicators: Indicators
    machine: configdir.Machine


def main_on_fakes(argv: Sequence[str], clock: Clock = SYSTEM, pin: pinned.Pin = pinned.PIN, release: pinned.Source = pinned.RELEASE, **named: Unpack[Parties]) -> int:
    """[LAW:locality-or-seam] the one place a test's command meets the root's parties, so a party the root gains is one
    edit here: a table of launchd alone, indicators that start nothing, and a Mac with no managed policy, where the test
    names none of its own."""
    return cli.main(
        argv, clock, pin, release,
        processes=named.get("processes", processes()), indicators=named.get("indicators", FakeIndicators()), machine=named.get("machine", UNMANAGED),
    )  # fmt: skip
