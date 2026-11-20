"""A page's own script run as the phone's browser runs it (tests/pagescript.mjs), on what hands answered its requests.

The answers are what the test took from hands' routes, given back in the order the page asks; an `Unreachable` answers as
an address nobody listens on does, as while hands restarts. What comes back is what the page showed once it settled
after its start and after each step.
"""

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import hands.voice

SCRIPT = Path(__file__).with_name("pagescript.mjs")
PAGES = Path(hands.voice.__file__).parent


@dataclass(frozen=True)
class Answered:
    method: str
    path: str
    status: int
    body: str
    times: int = 1


@dataclass(frozen=True)
class Unreachable:
    method: str
    path: str
    times: int = 1
    unreachable: bool = True


@dataclass(frozen=True)
class Submit:
    """A form submitted, with what is set in the page's fields by id first."""

    submit: str
    set: dict[str, str]


@dataclass(frozen=True)
class Shown:
    status: str
    failed: bool
    running: str
    controls: int
    disabled: int
    at_ms: float
    asked: list[str]


def run_page(page: str, answers: list[Answered | Unreachable], steps: list[Submit]) -> list[Shown]:
    """What `page` showed after its start and each of `steps`; fails where it asked past the answers or left one unasked."""
    node = shutil.which("node")
    assert node is not None, "a page's script runs in node, which is not on PATH"
    ran = subprocess.run([node, str(SCRIPT)], input=json.dumps({"page": str(PAGES / page), "answers": [asdict(a) for a in answers], "steps": [asdict(s) for s in steps]}), capture_output=True, text=True, timeout=30)
    assert ran.returncode == 0, ran.stderr
    out = json.loads(ran.stdout)
    assert out["unasked"] == [], f"the page never asked {out['unasked']}"
    return [Shown(**shown) for shown in out["shows"]]
