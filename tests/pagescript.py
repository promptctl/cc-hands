"""A page's own script run as the phone's browser runs it (tests/pagescript.mjs), against hands as the test describes it.

hands is a timeline of eras: from each era's moment on, each route answers as that era says, with a reply the test took
from hands' own routes or with nobody answering, as while hands restarts. Each step is something a person does, or only
time passing, and what comes back is what the page showed once it had run to the step's moment.
"""

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import hands.voice

SCRIPT = Path(__file__).with_name("pagescript.mjs")
PAGES = Path(hands.voice.__file__).parent


@dataclass(frozen=True)
class Reply:
    status: int
    body: str


# None: nobody answers on hands' address.
type Answer = Reply | None


@dataclass(frozen=True)
class Era:
    """hands from `from_ms` on, until a later era begins: each route, "METHOD path", and its answer."""

    from_ms: int
    answers: dict[str, Answer]


@dataclass(frozen=True)
class Submit:
    form: str
    until_ms: int
    set: dict[str, str]


@dataclass(frozen=True)
class Click:
    """The button reading `text` inside the element `within` names."""

    within: str
    text: str
    until_ms: int


@dataclass(frozen=True)
class Wait:
    until_ms: int


type Step = Submit | Click | Wait


@dataclass(frozen=True)
class Shown:
    at_ms: int
    reading: str  # how the page's read of hands stands
    status: str  # what came of the last choice
    errors: list[str]  # which of the two lines say a failure
    running: str
    controls: int
    disabled: int
    asked: list[str]


def _answer(answer: Answer) -> dict[str, object] | None:
    return None if answer is None else {"status": answer.status, "body": answer.body}


def _step(step: Step) -> dict[str, object]:
    match step:
        case Submit(form=form, until_ms=until_ms, set=fields):
            return {"kind": "submit", "form": form, "until_ms": until_ms, "set": fields}
        case Click(within=within, text=text, until_ms=until_ms):
            return {"kind": "click", "within": within, "text": text, "until_ms": until_ms, "set": {}}
        case Wait(until_ms=until_ms):
            return {"kind": "wait", "until_ms": until_ms, "set": {}}


def run_page(page: str, hands: list[Era], steps: list[Step]) -> list[Shown]:
    """What `page` showed at the end of each of `steps`; fails where it asked hands something the eras do not answer."""
    node = shutil.which("node")
    assert node is not None, "a page's script runs in node, which is not on PATH"
    given = {
        "page": str(PAGES / page),
        "hands": [{"from_ms": era.from_ms, "answers": {route: _answer(answer) for route, answer in era.answers.items()}} for era in hands],
        "steps": [_step(step) for step in steps],
    }
    ran = subprocess.run([node, str(SCRIPT)], input=json.dumps(given), capture_output=True, text=True, timeout=30)
    assert ran.returncode == 0, ran.stderr
    return [Shown(**shown) for shown in json.loads(ran.stdout)]
