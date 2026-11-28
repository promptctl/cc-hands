"""hands reaches the user's tmux servers through `hands.sessions.tmux`, the fritters that wrapped sessions through
`hands.sessions.typing`, and the Mac's screen through `hands.sessions.launchservices`, and only the composition root that
builds the run reaches any of them, handing it on as `Terminals`, `Fritters` or `Screen`: so every other module, and every
test of one, runs on whatever terminals and screen it is given."""

from pathlib import Path

import hands
from test_processes_boundary import importers

PACKAGE = Path(hands.__file__).parent
# The root that builds the run, and `hands status`'s check of which pane a session's keys reach.
ROOT = "daemon/cli.py"
# The brain runs its own pane through tmux and fritter past both interfaces, marked as the exception it is;
# promptctl-testing-c3u.fhu hands it Terminals and Fritters and deletes it from here.
BRAIN = "brain/process.py"


def test_only_the_root_reaches_tmux_but_the_brain_on_its_own_pane() -> None:
    assert importers(PACKAGE, "hands.sessions.tmux") == sorted([ROOT, BRAIN])


def test_only_the_root_reaches_fritter_but_the_brain_on_its_own_pane() -> None:
    assert importers(PACKAGE, "hands.sessions.typing") == sorted([ROOT, BRAIN])


def test_only_the_root_reaches_the_screen() -> None:
    assert importers(PACKAGE, "hands.sessions.launchservices") == [ROOT]
