"""hands reaches the user's terminals through `hands.sessions.tmux` and the Mac's screen through
`hands.sessions.launchservices`, and only a composition root reaches either, handing it on as `Terminals` or `Screen`: so
every other module, and every test of one, runs on whatever terminals and screen it is given."""

from pathlib import Path

import hands
from test_clock_boundary import ROOTS
from test_processes_boundary import importers

PACKAGE = Path(hands.__file__).parent


def test_only_a_composition_root_reaches_tmux_but_the_brain_on_its_own_pane() -> None:
    # The brain runs tmux on its own pane until its parties are handed to it (promptctl-testing-c3u.fhu).
    assert set(importers(PACKAGE, "hands.sessions.tmux")) - set(ROOTS) == {"brain/process.py"}


def test_only_a_composition_root_reaches_the_screen() -> None:
    assert set(importers(PACKAGE, "hands.sessions.launchservices")) <= set(ROOTS)
