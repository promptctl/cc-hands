"""Which `hands` commands load Pipecat and torch: status, check, log, phone, and login answer without either."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import SYSTEM_PATH


pytestmark = pytest.mark.outside

# Run in a fresh interpreter, since this one has Pipecat loaded by other tests: any import that reaches it, at the
# CLI's top or inside the command, fails here, whichever module it runs through. The modules are written to a file,
# so whatever the command prints is the command's own. The child's PATH holds nothing installed, so what `check` finds
# is the same on every machine.
CHILD = Path(__file__).resolve().parent / "fixtures" / "command_imports_child.py"


# Each command's code in a home with no daemon: status and check say it is not running, log ends at the Ctrl-C its first
# wait is, and phone and login run whole.
@pytest.mark.parametrize(("command", "code"), [("status", 1), ("check", 1), ("log", 0), ("phone", 0), ("login", 0)])
def test_the_command_answers_without_loading_pipecat_or_torch(command: str, code: int, tmp_path: Path) -> None:
    out = tmp_path / "modules.json"
    child = subprocess.run([sys.executable, str(CHILD), str(tmp_path / "home"), command, str(out)], env=os.environ | {"PATH": SYSTEM_PATH}, capture_output=True, text=True)
    assert child.returncode == 0, child.stderr
    assert json.loads(out.read_text()) == {"code": code, "loaded": []}, child.stderr
