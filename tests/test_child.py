"""Every child hands starts: run to its end inside its caller's limit, killed with its whole process group past it or
once its caller stops waiting, ended with hands however hands ends, and told as one closed outcome. A start outside the
runner names its owner where it is made."""

import ast
import asyncio
import errno
import io
import json
import os
import signal
import subprocess
import sys
import tokenize
import uuid
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, cast

import pytest

import hands
from conftest import BrokenClock, VirtualClock, events, patiently
from hands.clock import SYSTEM
from hands.sessions import audit, child, warden
from hands.sessions.audit import Entry
from hands.sessions.child import Child, Children, Exited, NeverStarted, Ran, Reaped, Running, Signalled, TimedOut, Warden, WardenGone, fields, summary


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def runner(tmp_path: Path, recorded: list[Entry], clock: child.Clock = SYSTEM) -> Children:
    """A runner whose events are `recorded`, its warden writing to an audit log of the test's own."""
    return Children(clock, recorded.append, tmp_path / "audit")


# ---- How a child ended, as the event's consumer queries it.


@pytest.mark.parametrize(
    ("ending", "facts", "said"),
    [
        (Ran(Exited(3), b"", ""), {"ending": "exited", "returncode": 3}, "exited 3"),
        (Ran(Signalled(signal.SIGHUP), b"", ""), {"ending": "signalled", "signal": "SIGHUP"}, "was ended by SIGHUP"),
        (NeverStarted(FileNotFoundError(errno.ENOENT, "No such file or directory")), {"ending": "never_started", "errno": "ENOENT"}, "could not be started: No such file or directory"),
        (TimedOut(5.0, b"", ""), {"ending": "timed_out", "limit": 5.0}, "ran past 5.0s"),
        (Reaped(Exited(0), killed=False, leftovers=False), {"ending": "exited", "returncode": 0}, "exited 0"),
        # hands' own kill is never read as a signal something else sent.
        (Reaped(Signalled(signal.SIGKILL), killed=True, leftovers=False), {"ending": "killed"}, "was killed by hands"),
    ],
    ids=["exited", "signalled", "never-started", "timed-out", "reaped", "killed"],
)
def test_each_way_a_child_ends_is_its_own_facts_and_its_own_words(ending: child.Outcome | Reaped, facts: dict[str, object], said: str) -> None:
    assert (fields(ending), summary(ending)) == (facts, said)


def test_a_signal_is_a_status_of_its_own_and_an_exit_code_is_its_code() -> None:
    # Popen's convention, a negative return code for the signal that ended it, read once into the status.
    assert (child.status_of(-signal.SIGTERM), child.status_of(0), child.status_of(2)) == (Signalled(signal.SIGTERM), Exited(0), Exited(2))


# ---- The one policy of a wait, driven as each driver drives it: a wait is asked for, and how it went is sent back.


class Kills(Child):
    """A child that is never started: what is asked of it is the policy's alone."""

    def __init__(self) -> None:
        super().__init__(Warden(Path("/unwritten")))
        self.kills = 0

    def kill(self) -> None:
        self.kills += 1


# What a supervisor settles with, as the policy sees it: anything, handed back as it came.
SETTLED = object()


def policy_of(kills: Kills) -> Generator[float | None, object, object]:
    """The policy of a wait of 5 seconds on `kills`."""
    return child._policy(kills, 5.0)  # pyright: ignore[reportPrivateUsage]


def test_a_child_settled_inside_its_limit_is_how_it_ran_and_is_never_killed() -> None:
    kills = Kills()
    policy = policy_of(kills)
    assert next(policy) == 5.0
    with pytest.raises(StopIteration) as done:
        policy.send(SETTLED)
    assert (done.value.value, kills.kills) == (SETTLED, 0)


def test_a_child_past_its_limit_is_killed_and_waited_for_until_its_reap() -> None:
    kills = Kills()
    policy = policy_of(kills)
    next(policy)
    # Time up: killed, then waited for with no limit, until its supervisor has reaped it.
    assert (policy.send(child._TIME_UP), kills.kills) == (None, 1)  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(StopIteration) as done:
        policy.send(SETTLED)
    assert done.value.value is SETTLED


@pytest.mark.parametrize("interruption", [asyncio.CancelledError(), KeyboardInterrupt(), RuntimeError("no timer")], ids=["cancelled", "interrupted", "clock-failed"])
def test_a_wait_interrupted_kills_the_child_waits_for_its_reap_and_raises_the_interruption(interruption: BaseException) -> None:
    kills = Kills()
    policy = policy_of(kills)
    next(policy)
    assert (policy.throw(interruption), kills.kills) == (None, 1)
    with pytest.raises(type(interruption)):
        policy.send(SETTLED)


def test_what_failed_once_the_child_ran_is_raised_as_itself() -> None:
    policy = policy_of(Kills())
    next(policy)
    with pytest.raises(OSError, match="could not be read"):
        policy.send(OSError("the output could not be read"))


class Refusing(Warden):
    """A warden that is ready, and fails each record as a warden that ended and could not be started again does."""

    def ready(self) -> None:
        pass

    def added(self, group: int) -> None:
        raise WardenGone("no warden could be started in place of the one that ended")


@dataclass
class Started:
    """A process as Popen gives one, that was never made: its pid, and a wait that has nothing to wait for."""

    pid: int

    def wait(self) -> int:
        return 0


def test_a_child_the_warden_could_not_record_is_killed_once_and_never_after_its_reap(monkeypatch: pytest.MonkeyPatch) -> None:
    killed: list[int] = []

    def kill(group: int) -> bool:
        killed.append(group)
        return False

    monkeypatch.setattr(child.warden, "killed", kill)
    supervised = Child(Refusing(Path("/unwritten")))
    with pytest.raises(WardenGone):
        supervised.supervise(lambda: cast(subprocess.Popen[bytes], Started(4242)))
    # Reaped as the failure went up: a kill after it, as a stop's would be, signals no group whose pid is free again.
    supervised.kill()
    assert killed == [4242]


def test_output_files_that_cannot_be_made_are_a_child_that_never_started_and_none_is_left_open(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    made: list[IO[bytes]] = []
    real = child.tempfile.TemporaryFile

    def second_refused() -> IO[bytes]:
        if made:
            raise OSError(errno.EMFILE, "Too many open files")
        made.append(file := real())
        return file

    monkeypatch.setattr(child.tempfile, "TemporaryFile", second_refused)
    recorded: list[Entry] = []
    match runner(tmp_path, recorded).run_blocking("true", limit=5.0):
        case NeverStarted(error=error):
            assert error.errno == errno.EMFILE
        case other:
            raise AssertionError(f"a child with nowhere to print was {other}")
    [first] = made
    assert first.closed
    [ran] = events(recorded, "child.run")
    assert (ran.outcome, ran.facts["ending"], ran.facts["errno"]) == ("failed", "never_started", "EMFILE")


# ---- The warden's own reading of what hands writes it.


def test_the_warden_holds_each_group_recorded_until_it_is_taken_back() -> None:
    groups = frozenset[int]()
    for line in [b"+10", b"+11", b"-10", b"+12"]:
        groups = warden.after(groups, line)
    assert groups == {11, 12}


@pytest.mark.parametrize("line", [b"10", b"*10", b"+", b"+ten"])
def test_the_warden_refuses_a_line_hands_never_writes(line: bytes) -> None:
    with pytest.raises(ValueError):
        warden.after(frozenset(), line)


@pytest.mark.parametrize(("raised", "left"), [(ProcessLookupError(errno.ESRCH, "No such process"), False), (PermissionError(errno.EPERM, "Operation not permitted"), False), (None, True)], ids=["gone", "all-exited", "running"])
def test_a_group_gone_or_all_exited_is_nothing_left_to_kill(raised: OSError | None, left: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    def killpg(_group: int, _signal: int) -> None:
        # EPERM is what macOS says to a group whose processes have all exited but are not yet reaped.
        if raised is not None:
            raise raised

    monkeypatch.setattr(warden.os, "killpg", killpg)
    assert warden.killed(7) is left


def test_the_warden_ends_every_group_recorded_around_a_line_it_cannot_read_and_says_it(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    killed: list[int] = []

    def kill(group: int) -> bool:
        killed.append(group)
        return True

    monkeypatch.setattr(warden, "killed", kill)
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(b"+10\n+what\n+11\n-10\n")))
    warden.main(tmp_path / "audit", 1)
    assert killed == [11]
    [said] = wardens(tmp_path / "audit")
    assert (said["outcome"], said["facts"]["recorded"], said["facts"]["killed"]) == ("failed", 1, 1)
    assert "b'+what'" in said["error"]


# ---- [LAW:single-enforcer] every start outside the runner names its owner where it is made.

# The modules whose calls start a process, by what each is called.
STARTING = {
    "subprocess": {"Popen", "run", "call", "check_call", "check_output", "getoutput", "getstatusoutput"},
    "os": {"fork", "forkpty", "posix_spawn", "posix_spawnp", "system", "popen", "spawnl", "spawnle", "spawnlp", "spawnlpe", "spawnv", "spawnve", "spawnvp", "spawnvpe"},
    "asyncio": {"create_subprocess_exec", "create_subprocess_shell"},
    "pty": {"fork", "spawn"},
}
# Where a pool of processes is made: multiprocessing in any of its parts, concurrent.futures by its one process pool.
POOLS = {"multiprocessing": None, "concurrent.futures": "ProcessPoolExecutor"}
MARKER = "[LAW:single-enforcer] exception:"
# The runner, whose starts are the starts every other is measured against.
RUNNER = {Path("sessions/child.py")}


@dataclass(frozen=True)
class Unowned:
    line: int
    what: str


def unowned(source: str) -> list[Unowned]:
    """Each start in `source` whose line names no owner: a call that starts a process, an import of a pool of them, or a
    star import that would hide either from this scan."""
    comments = {token.start[0]: token.string for token in tokenize.generate_tokens(io.StringIO(source).readline) if token.type == tokenize.COMMENT}

    def owned(line: int) -> bool:
        # An owner and a reason after the marker: a bare marker names nobody.
        _, marked, owner = comments.get(line, "").partition(MARKER)
        return bool(marked) and bool(owner.strip())

    # What each name in the source stands for: a module the scan knows, or one of its starting calls.
    modules: dict[str, str] = {}
    calls: dict[str, str] = {}
    found: list[Unowned] = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        match node:
            case ast.Import(names=names):
                for alias in names:
                    if alias.name in STARTING:
                        modules[alias.asname or alias.name] = alias.name
                    if alias.name.split(".")[0] == "multiprocessing":
                        found.append(Unowned(node.lineno, f"import {alias.name}"))
            case ast.ImportFrom(module=str(module), names=names):
                for alias in names:
                    pooled = module.split(".")[0] == "multiprocessing" or (module in POOLS and alias.name in (POOLS[module], "*"))
                    if alias.name == "*" and (module in STARTING or pooled):
                        found.append(Unowned(node.lineno, f"from {module} import *"))
                    elif module in STARTING and alias.name in STARTING[module]:
                        calls[alias.asname or alias.name] = f"{module}.{alias.name}"
                    elif pooled:
                        found.append(Unowned(node.lineno, f"from {module} import {alias.name}"))
            case _:
                pass
    for node in ast.walk(tree):
        match node:
            case ast.Call(func=ast.Attribute(value=ast.Name(id=name), attr=attr)) if name in modules and attr in STARTING[modules[name]]:
                found.append(Unowned(node.lineno, f"{modules[name]}.{attr}()"))
            case ast.Call(func=ast.Name(id=name)) if name in calls:
                found.append(Unowned(node.lineno, f"{calls[name]}()"))
            case _:
                pass
    return sorted((start for start in found if not owned(start.line)), key=lambda start: start.line)


def test_every_start_outside_the_runner_names_its_owner() -> None:
    root = Path(hands.__file__).parent
    found = {str(path.relative_to(root)): unowned(path.read_text()) for path in root.rglob("*.py") if path.relative_to(root) not in RUNNER}
    assert {name: starts for name, starts in found.items() if starts} == {}


def test_the_scan_catches_each_way_of_starting_a_process_and_a_marker_that_names_no_owner() -> None:
    source = (
        "import subprocess\nimport subprocess as sp\nfrom subprocess import run as go\nimport os\nfrom os import posix_spawn\n"
        "import asyncio\nfrom asyncio import *\nimport multiprocessing\nfrom concurrent.futures import ProcessPoolExecutor\n"
        "subprocess.run(['true'])\nsp.Popen(['true'])\ngo(['true'])\nos.fork()\nposix_spawn('/bin/true', ['true'], {})\n"
        "asyncio.create_subprocess_exec('true')\nos.system('true')\n"
        f"subprocess.check_output(['true'])  # {MARKER}\n"
        f"subprocess.call(['true'])  # {MARKER}   \n"
    )
    assert unowned(source) == [
        Unowned(7, "from asyncio import *"), Unowned(8, "import multiprocessing"), Unowned(9, "from concurrent.futures import ProcessPoolExecutor"),
        Unowned(10, "subprocess.run()"), Unowned(11, "subprocess.Popen()"), Unowned(12, "subprocess.run()"), Unowned(13, "os.fork()"),
        Unowned(14, "os.posix_spawn()"), Unowned(15, "asyncio.create_subprocess_exec()"), Unowned(16, "os.system()"),
        Unowned(17, "subprocess.check_output()"), Unowned(18, "subprocess.call()"),
    ]  # fmt: skip


def test_the_scan_passes_a_start_that_names_its_owner_and_what_starts_nothing() -> None:
    source = (
        f"import subprocess, os\nsubprocess.run(['true'])  # {MARKER} the person at this terminal, who answers it\n"
        "subprocess.DEVNULL\nos.getpid()\nos.execv('/bin/true', ['true'])\nprint('subprocess.run(x)')\nfrom concurrent.futures import Future\n"
    )
    assert unowned(source) == []


# ---- Real children, started and ended.


@pytest.mark.outside
async def test_a_child_s_output_and_exit_are_what_it_ran_to(tmp_path: Path) -> None:
    recorded: list[Entry] = []
    assert await runner(tmp_path, recorded).run("sh", "-c", "echo out; echo err >&2; exit 3", limit=5.0) == Ran(Exited(3), b"out\n", "err\n")
    # An exit of any code ran: the caller's to judge, which the event says without judging.
    [ran] = events(recorded, "child.run")
    assert (ran.outcome, ran.facts["program"], ran.facts["ending"], ran.facts["returncode"], ran.facts["leftovers"]) == ("ok", "sh", "exited", 3, False)


@pytest.mark.outside
async def test_a_child_hung_past_its_limit_is_killed_with_its_group_and_reaped(tmp_path: Path) -> None:
    recorded: list[Entry] = []
    pid = tmp_path / "pid"
    outcome = await runner(tmp_path, recorded).run("sh", "-c", f"sleep 600 & echo $! > {pid}; echo started; wait", limit=0.5)
    assert outcome == TimedOut(0.5, b"started\n", "")
    grandchild = int(pid.read_text())
    # Killed, it is reparented to launchd, which reaps it.
    await patiently(lambda: not alive(grandchild), lambda: None)
    [ran] = events(recorded, "child.run")
    assert (ran.outcome, ran.facts["ending"], ran.facts["limit"]) == ("failed", "timed_out", 0.5)


@pytest.mark.outside
async def test_a_child_s_limit_is_measured_on_the_clock_it_was_given(tmp_path: Path) -> None:
    clock = VirtualClock()
    running = asyncio.create_task(runner(tmp_path, [], clock).run("sleep", "600", limit=5.0))
    await clock.sleeping(5.0)
    clock.advance(5)
    assert await running == TimedOut(5.0, b"", "")


@pytest.mark.outside
def test_a_child_waited_on_from_a_blocking_thread_is_held_to_the_same_limit(tmp_path: Path) -> None:
    # A virtual clock's blocking wait passes at once: the limit is up before the sleep can have ended.
    recorded: list[Entry] = []
    assert runner(tmp_path, recorded, VirtualClock()).run_blocking("sleep", "600", limit=5.0) == TimedOut(5.0, b"", "")
    [ran] = events(recorded, "child.run")
    assert ran.facts["ending"] == "timed_out"


class BreaksOnceStarted(VirtualClock):
    """A clock whose sleep fails once the child has written its pid: its kill lands on a child that is running."""

    def __init__(self, pid: Path) -> None:
        super().__init__()
        self._pid = pid

    def sleep(self, seconds: float) -> asyncio.Future[None]:
        return asyncio.ensure_future(self._broken())

    async def _broken(self) -> None:
        await patiently(lambda: self._pid.exists() and bool(self._pid.read_text().strip()), lambda: None)
        raise RuntimeError("no timer")


@pytest.mark.outside
async def test_a_child_whose_clock_fails_is_killed_and_reaped_before_the_failure_leaves(tmp_path: Path) -> None:
    pid = tmp_path / "pid"
    with pytest.raises(RuntimeError, match="no timer"):
        await runner(tmp_path, [], BreaksOnceStarted(pid)).run("sh", "-c", f"echo $$ > {pid}; exec sleep 600", limit=5.0)
    assert not alive(int(pid.read_text()))


@pytest.mark.outside
async def test_a_child_whose_caller_stops_waiting_is_killed_and_reaped_before_the_cancellation_leaves(tmp_path: Path) -> None:
    pid = tmp_path / "pid"
    running = asyncio.create_task(runner(tmp_path, []).run("sh", "-c", f"echo $$ > {pid}; exec sleep 600", limit=600.0))
    await patiently(lambda: pid.exists() and bool(pid.read_text().strip()), lambda: None)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert not alive(int(pid.read_text()))


@pytest.mark.outside
async def test_a_child_that_exits_leaves_nothing_of_its_group_behind(tmp_path: Path) -> None:
    recorded: list[Entry] = []
    pid = tmp_path / "pid"
    assert await runner(tmp_path, recorded).run("sh", "-c", f"sleep 600 & echo $! > {pid}", limit=5.0) == Ran(Exited(0), b"", "")
    await patiently(lambda: not alive(int(pid.read_text())), lambda: None)
    [ran] = events(recorded, "child.run")
    assert ran.facts["leftovers"] is True


@pytest.mark.outside
async def test_a_tmux_server_a_child_started_detaches_and_outlives_it(tmp_path: Path) -> None:
    # tmux's server leaves the group of the command that started it with setsid, as every session hands starts runs in.
    socket = f"hands-test-{uuid.uuid4().hex[:8]}"
    children = runner(tmp_path, [])
    try:
        assert await children.run("tmux", "-L", socket, "-f", "/dev/null", "new-session", "-d", "-s", "detached", "sleep 600", limit=10.0) == Ran(Exited(0), b"", "")
        match await children.run("tmux", "-L", socket, "has-session", limit=10.0):
            case Ran(status=Exited(code=0)):
                pass
            case other:
                raise AssertionError(f"the tmux server did not outlive the command that started it: {other}")
    finally:
        # Its one session gone, the test's own server exits with it.
        await children.run("tmux", "-L", socket, "kill-session", "-t", "=detached", limit=10.0)


@pytest.mark.outside
async def test_a_program_that_is_not_there_never_started_and_the_os_says_why(tmp_path: Path) -> None:
    recorded: list[Entry] = []
    match await runner(tmp_path, recorded).run("/nonexistent/hands-no-such-program", limit=5.0):
        case NeverStarted(error=error):
            assert (error.errno, error.filename) == (errno.ENOENT, "/nonexistent/hands-no-such-program")
        case other:
            raise AssertionError(f"a program that is not there was {other}")
    [ran] = events(recorded, "child.run")
    assert (ran.outcome, ran.facts["ending"], ran.facts["errno"]) == ("failed", "never_started", "ENOENT")
    assert ran.error is not None and "No such file or directory" in ran.error


@pytest.mark.outside
async def test_a_directory_that_is_gone_is_a_child_that_never_started_there(tmp_path: Path) -> None:
    gone = tmp_path / "gone"
    match await runner(tmp_path, []).run("true", limit=5.0, cwd=gone):
        case NeverStarted(error=error):
            assert (error.errno, str(error.filename)) == (errno.ENOENT, str(gone))
        case other:
            raise AssertionError(f"a child in a directory that is gone was {other}")


@pytest.mark.outside
async def test_a_terminal_child_stopped_on_a_clock_that_fails_is_killed_and_reaped_before_the_failure_leaves(tmp_path: Path) -> None:
    # A child that ignores the terminate, so only the kill ends it.
    read, write = os.pipe()

    def popen() -> subprocess.Popen[bytes]:
        try:
            return subprocess.Popen(["sh", "-c", "trap '' TERM; echo ready; exec sleep 600"], stdout=write, start_new_session=True)
        finally:
            os.close(write)

    recorded: list[Entry] = []
    running = await runner(tmp_path, recorded).terminal("stubborn", popen)
    assert isinstance(running, Running), running
    with os.fdopen(read, "rb") as ready:
        assert ready.readline() == b"ready\n"
    with pytest.raises(RuntimeError, match="no timer"):
        await running.stopped(BrokenClock(), 5)
    match running.ended.result():
        case Reaped(killed=killed):
            assert killed
        case other:
            raise AssertionError(f"a terminal child killed at its stop was {other}")
    # [LAW:nothing-unseen] a terminal child is a child.run of its own, its kill hands' and never a signal's.
    [ran] = events(recorded, "child.run")
    assert (ran.outcome, ran.facts["program"], ran.facts["ending"]) == ("failed", "stubborn", "killed")


@pytest.mark.outside
async def test_a_terminal_child_that_could_not_be_started_never_started_and_says_so(tmp_path: Path) -> None:
    recorded: list[Entry] = []

    def popen() -> subprocess.Popen[bytes]:
        return subprocess.Popen(["/nonexistent/hands-no-such-terminal"], start_new_session=True)

    match await runner(tmp_path, recorded).terminal("absent", popen):
        case NeverStarted(error=error):
            assert error.errno == errno.ENOENT
        case other:
            raise AssertionError(f"a terminal program that is not there was {other}")
    [ran] = events(recorded, "child.run")
    assert (ran.outcome, ran.facts["ending"], ran.facts["errno"]) == ("failed", "never_started", "ENOENT")


# ---- No child outlives hands, however hands ends.

# A hands that started a child, then ends as its argument says. Its own process, with nothing of the test's in it.
ENDING = """
import os, signal, sys, threading, time
from pathlib import Path
from hands.clock import SYSTEM
from hands.sessions.child import Children, Exited, Ran

audit, pid, ending = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
children = Children(SYSTEM, lambda _entry: None, audit)
threading.Thread(target=children.run_blocking, args=("sh", "-c", f"sleep 600 & echo $! > {pid}; wait"), kwargs={"limit": 600.0}, daemon=True).start()
while not (pid.exists() and pid.read_text().strip()):
    time.sleep(0.01)
match ending:
    case "exit":
        sys.exit(0)
    case "crash":
        raise RuntimeError("hands crashed")
    case "stop":
        os.kill(os.getpid(), signal.SIGTERM)
    case "kill":
        os.kill(os.getpid(), signal.SIGKILL)
    case "warden-killed":
        first = children._warden._running.process.pid
        os.kill(first, signal.SIGKILL)
        while children._warden._running is None or children._warden._running.process.pid == first:
            time.sleep(0.01)
        os.kill(os.getpid(), signal.SIGKILL)
    case "exec":
        os.execv("/usr/bin/true", ["true"])
    case "failed-exec":
        try:
            os.execv("/nonexistent/hands", ["hands"])
        except OSError:
            pass
        os.kill(int(pid.read_text()), 0)
        assert children.run_blocking("true", limit=30.0) == Ran(Exited(0), b"", ""), "a failed exec left hands unable to start a child"
        print("started another", flush=True)
"""


@pytest.mark.outside
@pytest.mark.parametrize("ending", ["exit", "crash", "stop", "kill", "warden-killed", "exec", "failed-exec"])
async def test_no_child_outlives_hands_however_hands_ends(ending: str, tmp_path: Path) -> None:
    pid = tmp_path / "pid"
    ended = subprocess.run([sys.executable, "-c", ENDING, str(tmp_path / "audit"), str(pid), ending], capture_output=True, text=True, timeout=60)
    # A failed exec leaves its children running, and hands able to start more, until it ends.
    assert ("started another" in ended.stdout) == (ending == "failed-exec"), ended.stderr
    grandchild = int(pid.read_text())
    await patiently(lambda: not alive(grandchild), lambda: None)
    # [LAW:nothing-unseen] the warden says it ended what hands left, once hands had ended.
    await patiently(lambda: bool(wardens(tmp_path / "audit")), lambda: None)
    [said] = wardens(tmp_path / "audit")
    assert (said["facts"]["recorded"], said["facts"]["killed"]) == (1, 1)


def wardens(directory: Path) -> list[dict[str, Any]]:
    if not directory.exists():
        return []
    return [line for line in map(json.loads, audit.tail(directory, 100)[0]) if line.get("event") == "hands.warden"]
