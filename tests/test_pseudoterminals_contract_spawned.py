"""The pseudoterminals contract (pseudoterminalcontract.py), kept by SpawnedPseudoterminals on real terminals: outside,
since it opens terminals and starts processes."""

from collections.abc import Mapping
from pathlib import Path

import pytest

from hands.clock import SystemClock
from hands.sessions.child import SpawnedChildren
from hands.sessions.pseudoterminal import SpawnedPseudoterminals
from pseudoterminalcontract import VARIABLE, PseudoterminalsContract

pytestmark = pytest.mark.outside


class SpawnedWorld:
    def __init__(self, root: Path) -> None:
        self.folder = root / "folder"
        self.folder.mkdir()
        self.bin = root / "bin"
        self.bin.mkdir()
        self.pseudoterminals = SpawnedPseudoterminals(SpawnedChildren(SystemClock(), lambda _: None))

    @property
    def environment(self) -> Mapping[str, str]:
        return {"PATH": str(self.bin)}

    def _script(self, name: str, body: str) -> str:
        script = self.bin / name
        script.write_text(f"#!/bin/sh\n{body}")
        script.chmod(0o755)
        return name

    def program(self, name: str, shown: str, code: int) -> str:
        (self.bin / f"{name}.shown").write_text(shown)
        return self._script(name, f"/bin/cat '{self.bin}/{name}.shown'\nexit {code}\n")

    def echoing(self, name: str) -> str:
        return self._script(name, f"/bin/pwd -P\nprintf '%s\\n' \"${VARIABLE}\" \"$TERM\"\n")

    def hung(self, name: str) -> str:
        return self._script(name, "exec /bin/sleep 60\n")

    def unfound(self) -> str:
        return "absent"

    def unrunnable(self, name: str) -> str:
        script = self.bin / name
        script.write_text("#!/bin/sh\nexit 0\n")
        script.chmod(0o644)
        return name


@pytest.fixture
def world(tmp_path: Path) -> SpawnedWorld:
    return SpawnedWorld(tmp_path.resolve())


class TestSpawnedPseudoterminals(PseudoterminalsContract):
    pass
