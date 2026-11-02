"""Only the Clock adapter reads the machine's time, sleeps on it, schedules on it, or mints ids and randomness from it."""

import ast
from pathlib import Path

import hands

ADAPTER = "clock.py"

# Modules whose facilities are all the machine's time or randomness, so importing one is reaching for it.
MODULES = ("time", "uuid", "secrets", "random")

# Calls that read or wait on real time, on the modules that also carry plain types.
CALLS = {
    "datetime": ("now", "utcnow", "today"),
    "date": ("today",),
    "asyncio": ("sleep", "timeout", "timeout_at", "wait_for"),
}

# The event loop's timers, on whatever loop they are called on: anything not a clock is the loop the machine's time runs.
LOOP = ("call_later", "call_at", "time")

# [LAW:escape-local-minima] exception: the modules not yet moved onto Clock, each moved and struck off by a child of
# promptctl-testing-c3u.eps; the last one deletes this list. A module may leave it and never join it.
UNMOVED = frozenset(
    {
        "brain/asides.py", "brain/mcp.py", "brain/process.py", "brain/stage.py",
        "daemon/cli.py", "daemon/config.py", "daemon/menubar.py", "daemon/run.py", "daemon/smoke.py", "daemon/starting.py",
        "daemon/update.py",
        "spotify.py",
        "voice/brainpage.py", "voice/cues.py", "voice/engaged.py", "voice/floor.py", "voice/holdowner.py", "voice/keys.py", "voice/latency.py",
        "voice/library.py", "voice/longpoll.py", "voice/microphone.py", "voice/phone.py", "voice/phoneaddress.py",
        "voice/phonepage.py", "voice/pipeline.py", "voice/system.py", "voice/talkkey.py", "voice/tools.py", "voice/turnstart.py", "voice/utterance.py",
        "voice/vocabulary.py",
    }
)  # fmt: skip


def reached(source: str) -> set[str]:
    """What `source` reaches of the machine's time and randomness, by the name it reaches it with."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        match node:
            case ast.Import(names=aliases):
                found.update(alias.name for alias in aliases if alias.name.split(".")[0] in MODULES)
            case ast.ImportFrom(module=str(module), level=0) if module.split(".")[0] in MODULES:
                found.add(module)
            case ast.Call(func=ast.Attribute(value=ast.Name(id=owner), attr=attr)) if attr in CALLS.get(owner, ()):
                found.add(f"{owner}.{attr}()")
            case ast.Call(func=ast.Attribute(value=ast.Name(id="asyncio"), attr="wait"), keywords=keywords) if any(
                keyword.arg == "timeout" for keyword in keywords
            ):
                found.add("asyncio.wait(timeout=)")
            case ast.Call(func=ast.Attribute(value=receiver, attr=attr)) if attr in LOOP and "clock" not in ast.unparse(receiver):
                found.add(f".{attr}()")
            case _:
                pass
    return found


def offenders() -> dict[str, set[str]]:
    root = Path(hands.__file__).parent
    modules = sorted(path for path in root.rglob("*.py") if path.relative_to(root).as_posix() != ADAPTER)
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
        "datetime.now(UTC)\nasyncio.sleep(1)\nasyncio.timeout(2)\nasyncio.wait_for(f, 3)\nasyncio.wait(fs, timeout=5)\n"
        "loop.call_at(4, f)\nasyncio.get_running_loop().time()\n"
        "clock.now()\nself._clock.sleep(1)\nclock.timeout(2)\nclock.call_at(4, f)\nasyncio.wait(fs)\n"
    )
    assert reached(source) == {
        "time", "uuid", "secrets", "random", "datetime.now()", "asyncio.sleep()", "asyncio.timeout()", "asyncio.wait_for()",
        "asyncio.wait(timeout=)", ".call_at()", ".time()",
    }  # fmt: skip
