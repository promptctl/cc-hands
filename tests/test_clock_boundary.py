"""Only the Clock adapter reads the machine's time, sleeps on it, schedules on it, or mints ids and randomness from it."""

import ast
from pathlib import Path

import hands

# The real clock and the virtual one: a seeded `random.Random`, and a `uuid.UUID` built from its bits, are no reach.
ADAPTERS = ("clock.py", "virtualclock.py")

# Modules whose facilities are all the machine's time or randomness, so importing one is reaching for it.
MODULES = ("time", "uuid", "secrets", "random")

# Calls that read or wait on real time or randomness, by the module or class they are made on, however it was imported.
CALLS = {
    "datetime.datetime": ("now", "utcnow", "today"),
    "datetime.date": ("today",),
    "asyncio": ("sleep", "timeout", "timeout_at", "wait_for"),
    "os": ("urandom",),
    "threading": ("Timer",),
}

# Waits a thread or a task makes on the machine's seconds, on whatever it waits on.
TIMED = ("get", "join", "wait", "acquire")

# The event loop's timers, on whatever loop they are called on: anything not a clock is the loop the machine's time runs.
LOOP = ("call_later", "call_at", "time")

# The names a module is handed its clock by.
CLOCKS = ("clock", "_clock")

# [LAW:escape-local-minima] exception: the modules not yet moved onto Clock, each moved and struck off by a child of
# promptctl-testing-c3u.eps; the last one deletes this list. A module may leave it and never join it.
UNMOVED = frozenset(
    {
        "brain/asides.py", "brain/context.py", "brain/mcp.py", "brain/process.py", "brain/stage.py",
        "daemon/cli.py", "daemon/config.py", "daemon/menubar.py", "daemon/notify.py", "daemon/run.py", "daemon/smoke.py",
        "daemon/starting.py", "daemon/update.py",
        "sessions/otlp.py", "sessions/shim.py",
        "spotify.py",
        "voice/atdesk.py", "voice/backlogpage.py", "voice/brainpage.py", "voice/cues.py", "voice/devices.py", "voice/engaged.py",
        "voice/floor.py", "voice/holdowner.py", "voice/keyed.py", "voice/keys.py", "voice/latency.py", "voice/library.py",
        "voice/loads.py", "voice/longpoll.py", "voice/microphone.py", "voice/naming.py", "voice/phone.py", "voice/phoneaddress.py",
        "voice/phonepage.py", "voice/pipeline.py", "voice/ptt.py", "voice/recorder.py", "voice/sentences.py", "voice/summarising.py",
        "voice/system.py", "voice/talkkey.py", "voice/tools.py", "voice/turnstart.py", "voice/utterance.py", "voice/vocabulary.py",
        "voice/voiceprints.py", "voice/wake.py",
    }
)  # fmt: skip


def reached(source: str) -> set[str]:
    """What `source` reaches of the machine's time and randomness, by the name it reaches it with."""
    tree = ast.parse(source)
    named = _imported(tree)
    found: set[str] = set()
    for node in ast.walk(tree):
        match node:
            case ast.Import(names=aliases):
                found.update(alias.name for alias in aliases if alias.name.split(".")[0] in MODULES)
            case ast.ImportFrom(module=str(module), level=0) if module.split(".")[0] in MODULES:
                found.add(module)
            # The machine's clock itself, which only a composition root hands on.
            case ast.ImportFrom(module="hands.clock", names=aliases) if any(alias.name == "SYSTEM" for alias in aliases):
                found.add("hands.clock.SYSTEM")
            case ast.Call(func=ast.Attribute(value=owner, attr=attr)) if attr in CALLS.get(_qualified(owner, named), ()):
                found.add(f"{_qualified(owner, named)}.{attr}()")
            case ast.Call(func=ast.Attribute(attr=attr), keywords=keywords) if attr in TIMED and any(
                keyword.arg == "timeout" for keyword in keywords
            ):
                found.add(f".{attr}(timeout=)")
            case ast.Call(func=ast.Attribute(value=receiver, attr=attr)) if attr in LOOP and _last(receiver) not in CLOCKS:
                found.add(f".{attr}()")
            case _:
                pass
    return found


def _imported(tree: ast.Module) -> dict[str, str]:
    """Each name the module binds by an import, to the dotted name it was imported as."""
    named: dict[str, str] = {}
    for node in ast.walk(tree):
        match node:
            case ast.Import(names=aliases):
                named.update({alias.asname: alias.name for alias in aliases if alias.asname})
                named.update({alias.name.split(".")[0]: alias.name.split(".")[0] for alias in aliases if not alias.asname})
            case ast.ImportFrom(module=str(module), names=aliases, level=0):
                named.update({alias.asname or alias.name: f"{module}.{alias.name}" for alias in aliases})
            case _:
                pass
    return named


def _qualified(node: ast.expr, named: dict[str, str]) -> str:
    """The dotted name `node` reads, with the import its first part was bound by written out."""
    match node:
        case ast.Name(id=name):
            return named.get(name, name)
        case ast.Attribute(value=value, attr=attr):
            return f"{_qualified(value, named)}.{attr}"
        case _:
            return ""


def _last(node: ast.expr) -> str:
    match node:
        case ast.Name(id=name) | ast.Attribute(attr=name):
            return name
        case _:
            return ""


def offenders() -> dict[str, set[str]]:
    root = Path(hands.__file__).parent
    modules = sorted(path for path in root.rglob("*.py") if path.relative_to(root).as_posix() not in ADAPTERS)
    assert modules
    reaching = {path.relative_to(root).as_posix(): reached(path.read_text()) for path in modules}
    return {name: found for name, found in reaching.items() if found}


def test_only_the_clock_adapter_reaches_real_time_and_randomness() -> None:
    assert {name: found for name, found in offenders().items() if name not in UNMOVED} == {}


def test_a_module_moved_onto_the_clock_is_struck_off_the_unmoved() -> None:
    assert sorted(UNMOVED - offenders().keys()) == []


def test_the_check_catches_each_way_of_reaching_the_machines_time() -> None:
    source = (
        "import time\nfrom uuid import uuid4\nimport secrets as s\nfrom random import choice\n"
        "import asyncio as aio\nimport datetime as d\nfrom datetime import datetime as dt\nimport os, threading, queue\n"
        "from hands.clock import SYSTEM\n"
        "dt.now(UTC)\nd.datetime.now(d.UTC)\nd.date.today()\naio.sleep(1)\naio.timeout(2)\naio.wait_for(f, 3)\n"
        "aio.wait(fs, timeout=5)\nlane.queue.get(timeout=6)\nthread.join(timeout=7)\nos.urandom(8)\nthreading.Timer(9, f)\n"
        "loop.call_at(4, f)\naio.get_running_loop().time()\nself._clockwork.call_later(1, f)\n"
        "clock.now()\nclock.timeout(2)\nself._clock.call_at(4, f)\naio.wait(fs)\nqueue.get()\n"
    )
    assert reached(source) == {
        "time", "uuid", "secrets", "random", "hands.clock.SYSTEM", "datetime.datetime.now()", "datetime.date.today()",
        "asyncio.sleep()", "asyncio.timeout()", "asyncio.wait_for()", ".wait(timeout=)", ".get(timeout=)", ".join(timeout=)",
        "os.urandom()", "threading.Timer()", ".call_at()", ".time()", ".call_later()",
    }  # fmt: skip
