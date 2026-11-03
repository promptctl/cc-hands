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

# Names a module times a wait by, as a test would patch one.
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
            # Code written as a string, for pytester or `python -c`, reaches whatever it would reach run; an f-string's
            # holes are read as a second, which is what a test fills them with, and never as the zero a yield is.
            case ast.JoinedStr(values=values):
                found.update(_written("".join(v.value if isinstance(v, ast.Constant) else "1" for v in values)))
            case ast.Constant(value=str(text)) if "\n" in text or "import" in text:
                found.update(_written(text))
            case ast.Assign(targets=targets, value=value) | ast.AnnAssign(target=_ as targets, value=ast.expr() as value):
                patched = [
                    target
                    for target in (targets if isinstance(targets, list) else [targets])
                    if isinstance(target, ast.Attribute) and _timing(target.attr, value)
                ]
                found.update(f"setattr {target.attr}" for target in patched)
            case ast.Call(func=func, args=args, keywords=keywords) if _last(func) in ("setattr", "object", "patch") and (
                name := _patched(args, keywords)
            ):
                found.add(f"setattr {name.rsplit('.', 1)[-1]}")
            case ast.Call(func=ast.Attribute(attr="multiple"), keywords=keywords) if timed := [
                keyword.arg for keyword in keywords if keyword.arg and _timing(keyword.arg, keyword.value)
            ]:
                found.update(f"setattr {name}" for name in timed)
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
            case ast.Call(func=ast.Attribute(value=receiver, attr=attr)) if attr in LOOP and _last(receiver) not in CLOCKS:
                found.add(f".{attr}()")
            case _:
                pass
    return found


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


# A patched timing constant tunes a wait on the machine's seconds instead of advancing a clock: a constant named for
# its seconds, or one set to a fraction of a second or a span, whatever it is named.
def _timing(name: str, value: ast.expr) -> bool:
    constant = name.rsplit(".", 1)[-1]
    match value:
        case ast.Constant(value=float()) | ast.Call(func=ast.Name(id="timedelta") | ast.Attribute(attr="timedelta")):
            return constant.isupper()
        case _:
            return constant.isupper() and bool(TIMING.search(constant))


def _patched(args: list[ast.expr], keywords: list[ast.keyword]) -> str:
    """The constant a `setattr`, a `monkeypatch.setattr` or a `patch.object` call patches to a timing, or nothing."""
    new = next((keyword.value for keyword in keywords if keyword.arg == "new"), None)
    for at, arg in enumerate(args):
        value = args[at + 1] if at + 1 < len(args) else new
        match arg:
            case ast.Constant(value=str(name)) if value is not None and _timing(name, value):
                return name
            case _:
                pass
    return ""


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


# The clock's own tests, which prove the adapter against the machine it reads, and this check's, whose samples reach it: the tests' counterpart of ADAPTERS.
CLOCK_TESTS = ("test_clock.py", "test_clock_boundary.py")

# Test modules whose verdict still waits on the machine's time, each with exactly what it reaches, struck off as each
# moves onto a VirtualClock it advances.
UNMOVED: dict[str, frozenset[str]] = {
        "conftest.py": frozenset({'.time()', 'asyncio.sleep()', 'asyncio.wait_for()', 'datetime.datetime.now()', 'hands.clock.SYSTEM', 'setattr HUNG', 'time'}),
        "fetches.py": frozenset({'time'}),
        "fixtures/standin_daemon.py": frozenset({'datetime.datetime.now()', 'hands.clock.SYSTEM'}),
        "fixtures/watched_run.py": frozenset({'setattr HUNG', 'time'}),
        "test_answer_permission.py": frozenset({'asyncio.sleep()', 'asyncio.timeout()', 'asyncio.wait_for()'}),
        "test_atdesk.py": frozenset({'asyncio.sleep()', 'hands.clock.SYSTEM', 'time'}),
        "test_attention.py": frozenset({'asyncio.sleep()', 'asyncio.timeout()', 'asyncio.wait_for()', 'hands.clock.SYSTEM'}),
        "test_audit.py": frozenset({'asyncio.wait_for()', 'hands.clock.SYSTEM', 'setattr LOG_POLL_SECONDS'}),
        "test_backlogpage.py": frozenset({'asyncio.wait_for()', 'hands.clock.SYSTEM', 'uuid'}),
        "test_brain.py": frozenset({'asyncio.sleep()', 'asyncio.timeout()', 'asyncio.wait_for()', 'hands.clock.SYSTEM', 'hands.clock.SystemClock', 'setattr BRAIN_REPLY_BY_SECONDS', 'setattr DRAIN_SECONDS', 'setattr START_SECONDS', 'setattr STOP_SECONDS', 'setattr TAKE_SECONDS', 'time'}),
        "test_brain_e2e.py": frozenset({'asyncio.sleep()', 'asyncio.timeout()', 'asyncio.wait_for()', 'hands.clock.SYSTEM'}),
        "test_brainpage.py": frozenset({'asyncio.sleep()', 'hands.clock.SYSTEM', 'setattr POLL_SECONDS', 'setattr WAIT_SECONDS'}),
        "test_catch_up.py": frozenset({'hands.clock.SYSTEM'}),
        "test_child.py": frozenset({'asyncio.sleep()', 'hands.clock.SYSTEM', 'time'}),
        "test_closesession.py": frozenset({'.time()', 'setattr END_SECONDS', 'time'}),
        "test_command_event.py": frozenset({'datetime.datetime.now()', 'hands.clock.SYSTEM'}),
        "test_command_imports.py": frozenset({'hands.clock.SYSTEM', 'hands.clock.SystemClock'}),
        "test_context.py": frozenset({'hands.clock.SYSTEM'}),
        "test_conversationpage.py": frozenset({'asyncio.sleep()', 'hands.clock.SYSTEM', 'setattr WAIT_SECONDS'}),
        "test_daemon_config.py": frozenset({'asyncio.sleep()', 'asyncio.wait_for()', 'datetime.datetime.now()', 'hands.clock.SYSTEM'}),
        "test_delta.py": frozenset({'.time()', '.wait(timeout=)', 'asyncio.sleep()', 'asyncio.wait_for()', 'datetime.datetime.now()', 'hands.clock.SYSTEM', 'time'}),
        "test_devices.py": frozenset({'asyncio.sleep()', 'hands.clock.SYSTEM'}),
        "test_engaged.py": frozenset({'asyncio.sleep()', 'asyncio.timeout()', 'hands.clock.SYSTEM'}),
        "test_filing.py": frozenset({'hands.clock.SYSTEM'}),
        "test_focus.py": frozenset({'datetime.datetime.now()', 'hands.clock.SYSTEM'}),
        "test_front.py": frozenset({'time'}),
        "test_gallerypage.py": frozenset({'asyncio.sleep()', 'hands.clock.SYSTEM'}),
        "test_hold.py": frozenset({'.time()', 'asyncio.sleep()', 'asyncio.timeout()', 'asyncio.wait_for()', 'datetime.datetime.now()', 'hands.clock.SYSTEM', 'setattr HOLD_SECONDS', 'setattr TURN_LIMIT_SECONDS'}),
        "test_holdowner.py": frozenset({'asyncio.sleep()', 'asyncio.timeout()', 'hands.clock.SYSTEM', 'setattr HEARING_SECONDS'}),
        "test_input_tools.py": frozenset({'datetime.datetime.now()', 'hands.clock.SYSTEM'}),
        "test_intermediary.py": frozenset({'hands.clock.SYSTEM'}),
        "test_latency.py": frozenset({'hands.clock.SYSTEM'}),
        "test_library.py": frozenset({'.wait(seconds)', 'hands.clock.SYSTEM', 'setattr LISTING_TIMEOUT_SECONDS'}),
        "test_list_sessions.py": frozenset({'hands.clock.SYSTEM'}),
        "test_liveness.py": frozenset({'.time()', 'asyncio.wait_for()', 'time'}),
        "test_loads.py": frozenset({'hands.clock.SYSTEM'}),
        "test_microphone.py": frozenset({'.join(seconds)', 'asyncio.sleep()', 'hands.clock.SYSTEM', 'time'}),
        "test_naming.py": frozenset({'hands.clock.SYSTEM'}),
        "test_narrator.py": frozenset({'asyncio.sleep()', 'asyncio.timeout()', 'asyncio.wait_for()', 'hands.clock.SYSTEM'}),
        "test_occurrences.py": frozenset({'asyncio.sleep()', 'hands.clock.SYSTEM'}),
        "test_otlp.py": frozenset({'datetime.datetime.now()', 'hands.clock.SYSTEM', 'time'}),
        "test_own_model.py": frozenset({'asyncio.wait_for()', 'hands.clock.SYSTEM'}),
        "test_patience.py": frozenset({'.call_later()', 'setattr PATIENCE_SECS', 'time'}),
        "test_phone.py": frozenset({'asyncio.sleep()', 'asyncio.timeout()', 'asyncio.wait_for()', 'hands.clock.SYSTEM', 'setattr QUIET_SECS'}),
        "test_phonepage.py": frozenset({'asyncio.sleep()', 'hands.clock.SYSTEM', 'setattr TAILSCALE_TIMEOUT_SECONDS', 'time'}),
        "test_pinned.py": frozenset({'asyncio.sleep()', 'setattr STALL_SECONDS'}),
        "test_playback.py": frozenset({'asyncio.wait_for()', 'hands.clock.SYSTEM'}),
        "test_progress.py": frozenset({'asyncio.sleep()', 'asyncio.wait_for()', 'hands.clock.SYSTEM'}),
        "test_proxy.py": frozenset({'asyncio.sleep()', 'asyncio.wait_for()', 'hands.clock.SYSTEM'}),
        "test_ptt.py": frozenset({'hands.clock.SYSTEM'}),
        "test_read_session.py": frozenset({'hands.clock.SYSTEM'}),
        "test_readiness.py": frozenset({'datetime.datetime.now()', 'hands.clock.SYSTEM'}),
        "test_recall.py": frozenset({'hands.clock.SYSTEM'}),
        "test_recorder.py": frozenset({'hands.clock.SYSTEM'}),
        "test_restart.py": frozenset({'.wait(timeout=)', 'asyncio.sleep()', 'datetime.datetime.now()', 'hands.clock.SYSTEM', 'time'}),
        "test_run_end.py": frozenset({'hands.clock.SYSTEM'}),
        "test_shim.py": frozenset({'asyncio.wait_for()', 'datetime.datetime.now()'}),
        "test_smoke.py": frozenset({'asyncio.wait_for()', 'datetime.datetime.now()', 'hands.clock.SYSTEM'}),
        "test_spoken.py": frozenset({'hands.clock.SYSTEM'}),
        "test_spotify.py": frozenset({'hands.clock.SYSTEM'}),
        "test_start.py": frozenset({'hands.clock.SYSTEM'}),
        "test_startsession.py": frozenset({'hands.clock.SYSTEM'}),
        "test_status.py": frozenset({'asyncio.sleep()', 'asyncio.wait_for()', 'datetime.datetime.now()', 'hands.clock.SYSTEM', 'time'}),
        "test_streaming.py": frozenset({'.wait(timeout=)', 'asyncio.sleep()', 'asyncio.timeout()', 'asyncio.wait_for()', 'hands.clock.SYSTEM'}),
        "test_summary_store.py": frozenset({'asyncio.wait_for()', 'hands.clock.SYSTEM', 'setattr LIT_TIMEOUT_SECONDS'}),
        "test_system.py": frozenset({'asyncio.sleep()', 'asyncio.timeout()', 'datetime.datetime.now()', 'hands.clock.SYSTEM', 'time'}),
        "test_tail.py": frozenset({'asyncio.wait_for()'}),
        "test_talking_over.py": frozenset({'hands.clock.SYSTEM', 'time'}),
        "test_tap.py": frozenset({'asyncio.wait_for()', 'hands.clock.SYSTEM'}),
        "test_timeouts.py": frozenset({'asyncio.sleep()', 'time'}),
        "test_tool_runs.py": frozenset({'hands.clock.SYSTEM'}),
        "test_typing.py": frozenset({'.join(timeout=)', 'hands.clock.SYSTEM', 'setattr ANSWER_TIMEOUT', 'time'}),
        "test_update.py": frozenset({'.wait(timeout=)', 'datetime.datetime.now()', 'hands.clock.SYSTEM', 'time'}),
        "test_usage.py": frozenset({'hands.clock.SYSTEM'}),
        "test_utterance.py": frozenset({'.wait(seconds)', 'asyncio.sleep()', 'asyncio.timeout()', 'hands.clock.SYSTEM'}),
        "test_vocabulary.py": frozenset({'hands.clock.SYSTEM', 'time'}),
        "test_voice_turns.py": frozenset({'.wait(timeout=)', 'asyncio.sleep()', 'hands.clock.SYSTEM', 'setattr TRANSCRIBING_SECONDS', 'setattr TURN_LIMIT_SECONDS'}),
        "test_voiceprints.py": frozenset({'.wait(timeout=)', 'asyncio.sleep()', 'asyncio.timeout()', 'hands.clock.SYSTEM'}),
        "test_voices.py": frozenset({'asyncio.wait_for()', 'hands.clock.SYSTEM', 'time'}),
        "test_wake.py": frozenset({'asyncio.sleep()', 'asyncio.timeout()', 'hands.clock.SYSTEM'}),
        "test_watchdog.py": frozenset({'datetime.datetime.now()', 'hands.clock.SYSTEM'}),
        "test_whisper.py": frozenset({'hands.clock.SYSTEM'}),
        "test_wide.py": frozenset({'datetime.datetime.now()', 'hands.clock.SYSTEM'}),
        "test_wrapper.py": frozenset({'.wait(timeout=)'}),
        "timeouts.py": frozenset({'time'}),
}  # fmt: skip


def test_no_test_reaches_the_machines_time_but_what_unmoved_records() -> None:
    assert offenders(Path(__file__).parent, CLOCK_TESTS) == UNMOVED


def test_the_check_catches_a_patched_timing_constant_and_passes_a_bare_yield() -> None:
    source = (
        'monkeypatch.setattr(holdowner, "HEARING_SECONDS", 0.1)\nmonkeypatch.setattr("hands.voice.hold.TURN_LIMIT_SECONDS", 1)\n'
        'monkeypatch.setattr(typing, "ANSWER_TIMEOUT", 1)\nmonkeypatch.setattr(pinned, "RELEASES", [])\n'
        "import asyncio\nasyncio.sleep(0)\n"
    )
    assert reached(source) == {"setattr HEARING_SECONDS", "setattr TURN_LIMIT_SECONDS", "setattr ANSWER_TIMEOUT"}


def test_the_check_catches_each_way_of_patching_a_timing_constant() -> None:
    source = (
        'setattr(m, "A_SECONDS", 1)\npatch.object(m, "B_SECONDS", 1)\nm.C_SECONDS = 0.1\nmp.setattr(m, "D_SECONDS", 1, False)\n'
        'mp.setattr(progress, "SETTLE", 0.01)\nmp.setattr(heartbeat, "HEARTBEAT", timedelta(seconds=0))\n'
        'mock.patch("hands.x.E_TIMEOUT", 1)\nmp.setattr(m, "LINES", 3)\nmp.setattr(m, "lowered", 0.5)\n'
        'patch.object(m, "F_SECONDS", new=1)\npatch.multiple(m, G_DELAY=1, OTHER=2)\nprogress.LONGEST = 0.5\n'
        'hands.voice.hold.H_SECONDS = 1\nmp.setattr(m, "stop_delay", fake)\n'
    )
    names = ("A_SECONDS", "B_SECONDS", "C_SECONDS", "D_SECONDS", "SETTLE", "HEARTBEAT", "E_TIMEOUT", "F_SECONDS", "G_DELAY", "LONGEST", "H_SECONDS")
    assert reached(source) == {f"setattr {name}" for name in names}


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
