"""Only the Clock adapter reads the machine's time, sleeps on it, schedules on it, or mints ids and randomness from it."""

import ast
import re
import textwrap
from functools import cache
from pathlib import Path

import hands

# The real clock and the virtual one: a seeded `random.Random`, and a `uuid.UUID` built from its bits, are no reach.
ADAPTERS = ("clock.py", "virtualclock.py")

# Modules whose facilities are all the machine's time or randomness, so importing one is reaching for it.
MODULES = ("time", "uuid", "secrets", "random")

# Calls that read or wait on real time or randomness, by the name they were imported as, however that was spelled.
CALLS = frozenset(
    {
        "datetime.datetime.now", "datetime.datetime.utcnow", "datetime.datetime.today", "datetime.date.today",
        "asyncio.sleep", "asyncio.timeout", "asyncio.timeout_at", "asyncio.wait_for",
        "os.urandom", "threading.Timer",
    }
)  # fmt: skip

# The machine's clock itself, which only a composition root hands on.
SYSTEM = frozenset({"hands.clock.SYSTEM", "hands.clock.SystemClock"})

# Waits a thread or a task makes on the machine's seconds, on whatever it waits on, told by their
# `timeout=`: told by their arguments alone, a `str.join` or a `dict.get` would be one.
TIMED = ("get", "join", "wait", "acquire", "result")

# The event loop's timers, on whatever loop they are called on: anything not a clock is the loop the machine's time runs.
LOOP = ("call_later", "call_at", "time")

# Names a module times a wait by.
TIMING = re.compile(r"(SECONDS|SECS|TIMEOUT|DEADLINE|PATIENCE|INTERVAL|DELAY|HUNG)$")

# Modules a star import brings every wait of in by its bare name.
STARRED = ("asyncio", "datetime", "threading")

# The names a module is handed its clock by.
CLOCKS = ("clock", "_clock")

# The composition roots, each binding the machine's clock for its process and reaching nothing else of the machine's time:
# every hands command's, and the hook shim's, a process of its own that must not import the CLI in Claude Code's path.
ROOTS = ("daemon/cli.py", "sessions/shim.py")


def reached(source: str, package: str = "hands") -> set[str]:
    """What `source` reaches of the machine's time and randomness, by the name it reaches it with, in code it runs
    and in code it writes out for another interpreter to run."""
    tree = ast.parse(source)
    named = _imported(tree, package)
    found: set[str] = set()
    for node in ast.walk(tree):
        match node:
            # Code written as a string, for pytester or `python -c`, reaches whatever it would reach run.
            case ast.JoinedStr() | ast.Constant(value=str()) if (text := _code(node)) is not None:
                found.update(_written(text))
            case ast.Import(names=aliases):
                found.update(alias.name for alias in aliases if alias.name.split(".")[0] in MODULES)
            case ast.ImportFrom(module=str(module), level=0) if module.split(".")[0] in MODULES:
                found.add(module)
            # Every name a star brings in is the module's, so each of its waits is reached unnamed.
            case ast.ImportFrom(module=str(module), names=[ast.alias(name="*")], level=0) if module in STARRED:
                found.add(f"{module}.*")
            case ast.Name() | ast.Attribute() if _qualified(node, named) in SYSTEM:
                found.add(_qualified(node, named))
            case ast.Call(func=func) if _qualified(func, named) == "asyncio.sleep" and _yields(node):
                pass
            case ast.Call(func=func) if _qualified(func, named) in CALLS:
                found.add(f"{_qualified(func, named)}()")
            # `asyncio.wait`'s timeout is its second argument, and nothing else that function takes.
            case ast.Call(func=func, args=[_, _, *_]) if _qualified(func, named) == "asyncio.wait":
                found.add("asyncio.wait(timeout)")
            case ast.Call(func=ast.Attribute(attr=attr), keywords=keywords) if attr in TIMED and any(
                keyword.arg == "timeout" for keyword in keywords
            ):
                found.add(f".{attr}(timeout=)")
            # A thread's join, an event's wait, and a future's result take their seconds first, as a number no
            # `str.join` or `asyncio.wait` is handed.
            case ast.Call(func=ast.Attribute(attr="join" | "wait" | "result" as attr) as func, args=[first, *_]) if (
                _seconds(first) and _qualified(func, named) != "asyncio.wait"
            ):
                found.add(f".{attr}(seconds)")
            # A lock's acquire takes its seconds second, after whether to block at all.
            case ast.Call(func=ast.Attribute(attr="acquire"), args=[_, second, *_]) if _seconds(second):
                found.add(".acquire(seconds)")
            case ast.Call(func=ast.Attribute(value=receiver, attr=attr)) if attr in LOOP and _last(receiver) not in CLOCKS:
                found.add(f".{attr}()")
            case _:
                pass
    return found


def _code(node: ast.expr) -> str | None:
    """The code a string literal holds, if it reads as code: an f-string's holes read as a second, which is what a test
    fills them with, and never as the zero a yield is."""
    match node:
        case ast.JoinedStr(values=values):
            return "".join(str(v.value) if isinstance(v, ast.Constant) else "1" for v in values)
        case ast.Constant(value=str(text)) if "\n" in text or "import" in text:
            return text
        case _:
            return None


def written(source: str) -> set[str]:
    """What the code `source` writes out for another interpreter, a pytester module or a `python -c`, reaches."""
    return {facility for node in ast.walk(ast.parse(source)) if isinstance(node, ast.expr) and (text := _code(node)) is not None for facility in _written(text)}


def _written(text: str) -> set[str]:
    try:
        return reached(textwrap.dedent(text))
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return set()


def _seconds(node: ast.expr) -> bool:
    """A number of seconds, written out or named as a timing constant."""
    match node:
        case ast.Constant(value=bool()):
            return False
        case ast.Constant(value=int() | float()):
            return True
        case ast.Name(id=name) | ast.Attribute(attr=name):
            return name.isupper() and bool(TIMING.search(name))
        case _:
            return False


def _yields(call: ast.Call) -> bool:
    """A sleep of nothing yields to the loop and waits on no seconds."""
    match call:
        case ast.Call(args=[delay], keywords=[]) | ast.Call(args=[], keywords=[ast.keyword(arg="delay", value=delay)]):
            return isinstance(delay, ast.Constant) and _seconds(delay) and delay.value == 0
        case _:
            return False


def _imported(tree: ast.Module, package: str) -> dict[str, str]:
    """Each name the module binds by an import, to the dotted name it was imported as."""
    named: dict[str, str] = {}
    for node in ast.walk(tree):
        match node:
            case ast.Import(names=aliases):
                named.update({alias.asname: alias.name for alias in aliases if alias.asname})
                named.update({alias.name.split(".")[0]: alias.name.split(".")[0] for alias in aliases if not alias.asname})
            case ast.ImportFrom(module=str(module), names=aliases, level=0):
                named.update({alias.asname or alias.name: f"{module}.{alias.name}" for alias in aliases})
            # A relative import climbs from the package the module is in, one package per dot after the first.
            case ast.ImportFrom(module=module, names=aliases, level=int(level)) if level > 0:
                base = ".".join([*package.split(".")[: len(package.split(".")) - (level - 1)], *([module] if module else [])])
                named.update({alias.asname or alias.name: f"{base}.{alias.name}" for alias in aliases})
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


@cache
def offenders(root: Path = Path(hands.__file__).parent, exempt: tuple[str, ...] = ADAPTERS) -> dict[str, set[str]]:
    modules = sorted(path for path in root.rglob("*.py") if path.relative_to(root).as_posix() not in exempt)
    assert modules
    reaching = {path.relative_to(root).as_posix(): reached(path.read_text(), ".".join([root.name, *path.relative_to(root).parent.parts])) for path in modules}
    return {name: found for name, found in reaching.items() if found}


def test_only_the_clock_adapter_reaches_real_time_and_randomness() -> None:
    assert {name: found for name, found in offenders().items() if name not in ROOTS} == {}


def test_only_a_composition_root_names_the_machines_clock_and_it_reaches_nothing_else() -> None:
    assert {name: found for name, found in offenders().items() if name in ROOTS} == {root: {"hands.clock.SYSTEM"} for root in ROOTS}


def test_each_composition_root_binds_the_machines_clock_once() -> None:
    root = Path(hands.__file__).parent
    uses = {name: sum(isinstance(node, ast.Name) and node.id == "SYSTEM" for node in ast.walk(ast.parse((root / name).read_text()))) for name in ROOTS}
    assert uses == dict.fromkeys(ROOTS, 1)


def test_the_check_catches_each_way_of_reaching_the_machines_time() -> None:
    source = (
        "import time\nfrom uuid import uuid4\nimport secrets as s\nfrom random import choice\n"
        "import asyncio as aio\nimport datetime as d\nfrom datetime import datetime as dt\nimport os, threading, queue\n"
        "from asyncio import sleep, timeout as limit\nfrom hands.clock import SYSTEM\nfrom hands import clock as clocks\n"
        "from hands.clock import SystemClock\n"
        "dt.now(UTC)\nd.datetime.now(d.UTC)\nd.date.today()\naio.sleep(1)\naio.timeout(2)\naio.wait_for(f, 3)\nsleep(1)\n"
        "limit(2)\naio.wait(fs, 5)\nrun(SYSTEM)\nclocks.SYSTEM.now()\nSystemClock()\n"
        "aio.wait(fs, timeout=5)\nlane.queue.get(timeout=6)\nthread.join(timeout=7)\nos.urandom(8)\nthreading.Timer(9, f)\n"
        "loop.call_at(4, f)\naio.get_running_loop().time()\nself._clockwork.call_later(1, f)\n"
        "clock.now()\nclock.timeout(2)\nself._clock.call_at(4, f)\naio.wait(fs)\nqueue.get()\n', '.join(parts)\nd.get(k, 1)\n"
    )
    assert reached(source) == {
        "time", "uuid", "secrets", "random", "hands.clock.SYSTEM", "hands.clock.SystemClock", "datetime.datetime.now()",
        "datetime.date.today()", "asyncio.sleep()", "asyncio.timeout()", "asyncio.wait_for()", "asyncio.wait(timeout)",
        ".wait(timeout=)", ".get(timeout=)", ".join(timeout=)", "os.urandom()", "threading.Timer()", ".call_at()", ".time()",
        ".call_later()",
    }  # fmt: skip


def test_the_check_passes_a_bare_yield_and_catches_a_lock_acquired_on_seconds() -> None:
    assert reached("import asyncio\nasyncio.sleep(0)\nlock.acquire(True, 2)\nlock.acquire(timeout=3)\nlock.acquire()\n") == {".acquire(seconds)", ".acquire(timeout=)"}


def test_the_check_catches_a_wait_by_its_seconds_and_code_written_for_another_interpreter() -> None:
    source = (
        "import asyncio\nfut.result(timeout=1)\nev.wait(1)\nasyncio.wait(fs)\nasyncio.sleep(0.0)\nasyncio.sleep(delay=0)\n"
        "t.join(0.05)\nfut.result(1)\n', '.join(parts)\n"
        'pytester.makepyfile("import time\\ntime.sleep(1.3)\\n")\nrun([sys.executable, "-c", "import secrets"])\n'
        'f"import datetime\\ndatetime.datetime.now({n})"\ndedent("""\\n    import uuid\\n""")\n'
        "from threading import *\n"
    )
    assert reached(source) == {
        ".result(timeout=)", ".wait(seconds)", ".join(seconds)", ".result(seconds)", "time", "secrets",
        "datetime.datetime.now()", "uuid", "threading.*",
    }  # fmt: skip


def test_a_sleep_of_a_false_is_no_yield() -> None:
    assert reached("import asyncio\nasyncio.sleep(False)\n") == {"asyncio.sleep()"}


def test_the_check_reads_a_relative_import_a_named_wait_and_an_f_string_sleep_of_a_hole() -> None:
    source = (
        "from .clock import SYSTEM\nfrom . import clock as c\nc.SYSTEM\nt.join(TIMEOUT)\nev.wait(BOUND_SECONDS)\n"
        'f"import asyncio\\nasyncio.sleep({DELAY})\\n"\n'
    )
    assert reached(source) == {"hands.clock.SYSTEM", ".join(seconds)", ".wait(seconds)", "asyncio.sleep()"}


def test_a_string_too_deep_to_parse_is_no_code() -> None:
    assert _written("import os\n" + "1+" * 100000 + "1") == set()


def test_a_relative_import_climbs_from_the_package_the_module_is_in() -> None:
    assert reached("from ..clock import SYSTEM\nSYSTEM\n", "hands.voice") == {"hands.clock.SYSTEM"}
    assert reached("from .clock import SYSTEM\nSYSTEM\n", "hands.voice") == set()
