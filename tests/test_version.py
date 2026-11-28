"""`hands --version` names the hands installed, whose version is its release's tag."""

from importlib.metadata import version

import pytest

from hands.daemon.cli import main
from fixtures.fakemachine import UNMANAGED
from fakeprocesses import processes


def test_the_version_printed_is_the_installed_package_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exited:
        main(["--version"], machine=UNMANAGED, processes=processes())
    assert exited.value.code == 0
    assert capsys.readouterr().out == f"hands {version('hands')}\n"
