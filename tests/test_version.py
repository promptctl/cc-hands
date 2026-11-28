"""`hands --version` names the hands installed, whose version is its release's tag."""

from importlib.metadata import version

import pytest

from fakeparties import main_on_fakes


def test_the_version_printed_is_the_installed_package_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exited:
        main_on_fakes(["--version"])
    assert exited.value.code == 0
    assert capsys.readouterr().out == f"hands {version('hands')}\n"
