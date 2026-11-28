"""`hands` as a test runs it: cli.main on a fake of every party the root binds, each test naming only those its behavior
rests on, on the clock, pin and release it names, the machine's own where it names none."""

from collections.abc import Sequence
from typing import TypedDict, Unpack

from fakechildren import Programs, exited
from fakeindicators import FakeIndicators
from fakeprocesses import processes
from fakepseudoterminals import FakePseudoterminals
from fixtures.fakemachine import UNMANAGED
from hands.brain import configdir, pinned
from hands.clock import SYSTEM, Clock
from hands.daemon import cli
from hands.daemon.indicatorprocess import Indicators
from hands.sessions.child import ChildrenOn
from hands.sessions.processes import Processes
from hands.sessions.pseudoterminal import PseudoterminalsOn
from hands.voice import library

# A keychain that holds no Hugging Face token, as `security` says of an item it has none of.
NO_TOKEN = exited(library.NOT_IN_KEYCHAIN)


class Parties(TypedDict, total=False):
    """The parties a test names: the rest are the fakes main_on_fakes gives."""

    processes: Processes
    indicators: Indicators
    machine: configdir.Machine
    children_on: ChildrenOn
    pseudoterminals_on: PseudoterminalsOn


def main_on_fakes(argv: Sequence[str], clock: Clock = SYSTEM, pin: pinned.Pin = pinned.PIN, release: pinned.Source = pinned.RELEASE, **named: Unpack[Parties]) -> int:
    """[LAW:locality-or-seam] the one place a test's command meets the root's parties, so a party the root gains is one
    edit here: a table of launchd alone, indicators that start nothing, a Mac with no managed policy, and children that
    stand in for `security` alone, a keychain holding no token, and pseudo-terminals that stand in for no program, where
    the test names none of its own."""
    return cli.main(
        argv, clock, pin, release,
        processes=named.get("processes", processes()), indicators=named.get("indicators", FakeIndicators()), machine=named.get("machine", UNMANAGED),
        children_on=named.get("children_on", Programs({"security": NO_TOKEN})), pseudoterminals_on=named.get("pseudoterminals_on", lambda _clock, _emit: FakePseudoterminals()),
    )  # fmt: skip
