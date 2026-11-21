"""Every child hands starts ends within its caller's time and never outlives hands, and its end is one closed outcome."""

import ast
import asyncio
import errno
import io
import os
import queue
import subprocess
import textwrap
import threading
import time
import tokenize
from pathlib import Path
from typing import IO

import pytest
from loguru import logger

import hands
from conftest import BrokenClock, VirtualClock
from hands.clock import SYSTEM, SystemClock
from hands.sessions import child
from hands.sessions.child import Exited, NeverStarted, TimedOut, run, run_blocking, told
from sourcescan import code, imported, qualified


@pytest.fixture
def started(monkeypatch: pytest.MonkeyPatch) -> list[subprocess.Popen[bytes]]:
    """Every child the runner starts, so a test can ask after it once the runner has let it go."""
    seen: list[subprocess.Popen[bytes]] = []

    class Seen(subprocess.Popen[bytes]):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)  # type: ignore[arg-type]
            seen.append(self)

    monkeypatch.setattr(child.subprocess, "Popen", Seen)
    return seen


@pytest.fixture
def live(monkeypatch: pytest.MonkeyPatch) -> child._Live:  # pyright: ignore[reportPrivateUsage]
    """A set of live children of the test's own, so a sweep ends this test's children and leaves the process's runner
    starting children for the tests after it."""
    fresh = child._Live()  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(child, "_LIVE", fresh)
    return fresh


def reaped(process: subprocess.Popen[bytes]) -> bool:
    return process.returncode is not None


def gone(pid: int) -> bool:
    """Whether `pid` has ended: killed, a grandchild is reparented to launchd, which reaps it in its own time."""
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.02)
    return False


class Interrupted(SystemClock):
    """A clock whose wait is ended by Ctrl-C, as a command's is when the person at its terminal presses it."""

    def take[T](self, source: queue.Queue[T] | queue.SimpleQueue[T], seconds: float) -> T:
        raise KeyboardInterrupt


def test_how_a_run_ended_is_said_without_what_it_wrote() -> None:
    assert [told(ended) for ended in (Exited(3, "token", ""), Exited(-9, "", ""), TimedOut(2.0, "token", ""), NeverStarted(FileNotFoundError(errno.ENOENT, "No such file", "x")))] == [
        "exited 3",
        "was killed by signal 9",
        "ran past 2.0s",
        "never started: [Errno 2] No such file: 'x'",
    ]
    assert "token" not in repr(Exited(0, "token", "token")) + repr(TimedOut(1.0, "token", "token"))


@pytest.mark.outside
async def test_a_child_s_output_and_exit_are_what_it_ran_to(started: list[subprocess.Popen[bytes]]) -> None:
    exits: list[str] = []
    sink = logger.add(lambda message: exits.append(message.record["message"]), level="DEBUG")
    try:
        assert await run("sh", "-c", "echo out; echo err >&2; exit 3", timeout=5, clock=SYSTEM) == Exited(3, "out\n", "err\n")
    finally:
        logger.remove(sink)
    assert any(line.startswith(f"sh ({started[0].pid}) exited 3 in ") for line in exits)


@pytest.mark.outside
async def test_a_child_killed_by_a_signal_exited_with_its_negative() -> None:
    assert await run("sh", "-c", "kill -9 $$", timeout=5, clock=SYSTEM) == Exited(-9, "", "")


@pytest.mark.outside
async def test_output_that_is_not_text_is_read_with_replacements() -> None:
    assert await run("printf", "\\377ok", timeout=5, clock=SYSTEM) == Exited(0, "�ok", "")


@pytest.mark.outside
async def test_a_child_that_runs_past_its_time_is_killed_and_reaped_with_what_it_wrote(started: list[subprocess.Popen[bytes]], logged: list[str]) -> None:
    start = time.monotonic()
    assert await run("sh", "-c", "echo early; exec sleep 30", timeout=0.5, clock=SYSTEM) == TimedOut(0.5, "early\n", "")
    assert time.monotonic() - start < 2
    assert reaped(started[0])
    assert any(f"killed sh ({started[0].pid})" in line and "ran past 0.5s" in line for line in logged)


@pytest.mark.outside
async def test_a_child_s_time_is_up_on_the_clock_it_was_given(started: list[subprocess.Popen[bytes]], logged: list[str]) -> None:
    clock = VirtualClock()
    running = asyncio.create_task(run("sleep", "30", timeout=5, clock=clock))
    await clock.sleeping(5.0)
    clock.advance(5)
    assert await running == TimedOut(5, "", "")
    assert reaped(started[0])
    assert any(f"killed sleep ({started[0].pid})" in line and "ran past 5.0s" in line for line in logged)


@pytest.mark.outside
async def test_a_child_whose_clock_fails_is_killed_and_reaped_before_the_failure_leaves(started: list[subprocess.Popen[bytes]], logged: list[str]) -> None:
    with pytest.raises(RuntimeError, match="no timer"):
        await run("sleep", "30", timeout=5, clock=BrokenClock())
    assert reaped(started[0])
    assert any(f"killed sleep ({started[0].pid})" in line and "its wait failed" in line for line in logged)


@pytest.mark.outside
async def test_what_a_child_started_is_killed_with_it(tmp_path: Path) -> None:
    pid = tmp_path / "pid"
    assert isinstance(await run("sh", "-c", f"sleep 30 & echo $! > {pid}; wait", timeout=0.5, clock=SYSTEM), TimedOut)
    assert gone(int(pid.read_text())), "the sleep its timed-out parent started is still running"


@pytest.mark.outside
async def test_a_child_whose_caller_stops_waiting_is_killed_and_reaped(started: list[subprocess.Popen[bytes]], logged: list[str]) -> None:
    running = asyncio.create_task(run("sleep", "30", timeout=30, clock=SYSTEM))
    await asyncio.sleep(0.1)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert reaped(started[0])
    assert any(f"killed sleep ({started[0].pid})" in line and "its caller stopped waiting" in line for line in logged)


@pytest.mark.outside
async def test_a_child_that_ended_as_it_was_killed_still_times_out(monkeypatch: pytest.MonkeyPatch, logged: list[str]) -> None:
    # What macOS says to a kill of a group whose processes have all exited but are not yet reaped.
    def exited(_pgid: int, _signal: int) -> None:
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(child.os, "killpg", exited)
    began = time.monotonic()
    assert isinstance(await run("sleep", "0.4", timeout=0.1, clock=SYSTEM), TimedOut)
    assert time.monotonic() - began >= 0.3, "it left before the child it could not kill was reaped"
    assert any("had already ended when it ran past 0.1s" in line for line in logged)
    assert not any(line.startswith("killed") for line in logged)


@pytest.mark.outside
async def test_a_failure_reading_what_a_child_wrote_is_hands_own(monkeypatch: pytest.MonkeyPatch, logged: list[str]) -> None:
    def unreadable(started: object) -> object:
        raise OSError("the output could not be read")

    monkeypatch.setattr(child._Started, "read", unreadable)  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(OSError, match="could not be read"):
        await run("true", timeout=5, clock=SYSTEM)
    assert not any("killed" in line or "already ended" in line for line in logged), "a failure of its own was told as a kill"


@pytest.mark.outside
async def test_a_program_that_is_not_there_never_started_and_the_os_says_why() -> None:
    match await run("/nonexistent/hands-no-such-program", timeout=5, clock=SYSTEM):
        case NeverStarted(error=error):
            assert (error.errno, error.filename) == (errno.ENOENT, "/nonexistent/hands-no-such-program")
        case ended:
            raise AssertionError(f"it {told(ended)}")


@pytest.mark.outside
async def test_a_directory_that_is_gone_is_a_child_that_never_started_there(tmp_path: Path) -> None:
    gone = tmp_path / "gone"
    match await run("true", timeout=5, cwd=gone, clock=SYSTEM):
        case NeverStarted(error=error):
            assert (error.errno, str(error.filename)) == (errno.ENOENT, str(gone))
        case ended:
            raise AssertionError(f"it {told(ended)}")


async def test_output_files_that_cannot_be_made_are_a_child_that_never_started_and_none_is_left_open(monkeypatch: pytest.MonkeyPatch) -> None:
    made: list[IO[bytes]] = []

    def second_refused() -> IO[bytes]:
        if made:
            raise OSError(errno.EMFILE, "Too many open files")
        made.append(file := real())
        return file

    real = child.tempfile.TemporaryFile
    monkeypatch.setattr(child.tempfile, "TemporaryFile", second_refused)
    match await run("true", timeout=5, clock=SYSTEM):
        case NeverStarted(error=error):
            assert error.errno == errno.EMFILE
        case ended:
            raise AssertionError(f"it {told(ended)}")
    [first] = made
    assert first.closed


@pytest.mark.outside
async def test_a_reaper_that_cannot_start_leaves_no_child(monkeypatch: pytest.MonkeyPatch, started: list[subprocess.Popen[bytes]], live: child._Live) -> None:  # pyright: ignore[reportPrivateUsage]
    def refused(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(child, "settles_off_loop", refused)
    with pytest.raises(RuntimeError, match="new thread"):
        await run("sleep", "30", timeout=5, clock=SYSTEM)
    assert reaped(started[0])
    assert not live._groups  # pyright: ignore[reportPrivateUsage]


@pytest.mark.outside
async def test_a_child_stopped_on_a_clock_that_fails_is_killed_and_reaped_before_the_failure_leaves(logged: list[str]) -> None:
    # A child that ignores the terminate, so only the kill ends it.
    process = child.start(["sh", "-c", "trap '' TERM; echo ready; sleep 30"], cwd=Path.cwd(), env=os.environ, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert process.stdout is not None and process.stdout.readline() == b"ready\n"
    stopping = child.reaped("stubborn", process, process.wait)
    with pytest.raises(RuntimeError, match="no timer"):
        await stopping.stopped(BrokenClock(), 5)
    assert reaped(process)
    assert any(f"killed stubborn ({process.pid})" in line and "its wait failed" in line for line in logged)


@pytest.mark.outside
def test_a_blocking_run_gives_what_the_child_ran_to() -> None:
    assert run_blocking("sh", "-c", "echo out; echo err >&2; exit 3", timeout=5, clock=SYSTEM) == Exited(3, "out\n", "err\n")


@pytest.mark.outside
def test_a_blocking_run_past_its_time_kills_the_child_and_what_it_started(tmp_path: Path, started: list[subprocess.Popen[bytes]]) -> None:
    pid = tmp_path / "pid"
    assert isinstance(run_blocking("sh", "-c", f"sleep 30 & echo $! > {pid}; wait", timeout=0.5, clock=SYSTEM), TimedOut)
    assert reaped(started[0])
    assert gone(int(pid.read_text())), "the sleep its timed-out parent started is still running"


@pytest.mark.outside
def test_a_blocking_run_on_a_virtual_clock_has_no_real_seconds_to_wait(started: list[subprocess.Popen[bytes]]) -> None:
    # The taker is not the loop that moves a virtual clock, so the wait moves nothing: a child not done at once is past.
    assert isinstance(run_blocking("sleep", "30", timeout=5, clock=VirtualClock()), TimedOut)
    assert reaped(started[0])


@pytest.mark.outside
def test_a_blocking_run_ended_by_ctrl_c_kills_the_child_before_the_interrupt_leaves(started: list[subprocess.Popen[bytes]], logged: list[str]) -> None:
    with pytest.raises(KeyboardInterrupt):
        run_blocking("sleep", "30", timeout=30, clock=Interrupted())
    assert reaped(started[0])
    assert any(f"killed sleep ({started[0].pid})" in line and "its caller stopped waiting" in line for line in logged)


@pytest.mark.outside
def test_hands_exit_kills_the_child_a_thread_it_left_is_waiting_on(monkeypatch: pytest.MonkeyPatch, live: child._Live) -> None:  # pyright: ignore[reportPrivateUsage]
    spawned = threading.Event()

    class Seen(subprocess.Popen[bytes]):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)  # type: ignore[arg-type]
            spawned.set()

    monkeypatch.setattr(child.subprocess, "Popen", Seen)
    ended: list[object] = []
    # A daemon thread, as the daemon's off-loop work is, which hands' exit does not wait for.
    waiting = threading.Thread(target=lambda: ended.append(run_blocking("sleep", "30", timeout=30, clock=SYSTEM)), daemon=True)
    waiting.start()
    assert spawned.wait(5)
    live.sweep()
    waiting.join(5)
    assert ended == [Exited(-9, "", "")]
    # Nothing starts once hands is exiting: a start the sweep raced would outlive it.
    match run_blocking("true", timeout=5, clock=SYSTEM):
        case NeverStarted(error=error):
            assert error.errno == errno.ECANCELED
        case after:
            raise AssertionError(f"a child started after the sweep {told(after)}")


# Every way Python starts a process, by the name it is reached by.
SPAWNS = frozenset(
    {
        "subprocess.Popen", "subprocess.run", "subprocess.call", "subprocess.check_call", "subprocess.check_output",
        "subprocess.getoutput", "subprocess.getstatusoutput", "os.fork", "os.forkpty", "os.posix_spawn", "os.posix_spawnp",
        "os.system", "os.popen", "os.spawnv", "os.spawnve", "os.spawnvp", "os.spawnvpe", "os.spawnl", "os.spawnle",
        "os.spawnlp", "os.spawnlpe", "pty.fork", "pty.spawn", "asyncio.create_subprocess_exec", "asyncio.create_subprocess_shell",
    }
)  # fmt: skip

# A loop's own, on whatever loop it is called on.
LOOP_SPAWNS = ("subprocess_exec", "subprocess_shell")

# Where a start that is not the runner's names who ends what it started.
OWNER = "# owner:"

RUNNER = "sessions/child.py"


def spawns(source: str, package: str = "hands") -> set[int]:
    """The lines of `source` that start a process other than through the runner, without an owner named on the line, in
    code it runs and in code it writes out for another interpreter to run."""
    tree = ast.parse(source)
    named = imported(tree, package)
    owned = {token.start[0] for token in tokenize.generate_tokens(io.StringIO(source).readline) if token.type == tokenize.COMMENT and token.string.startswith(OWNER)}
    found: set[int] = set()
    for node in ast.walk(tree):
        match node:
            case ast.JoinedStr() | ast.Constant(value=str()) if (text := code(node)) is not None and _written(text):
                found.add(node.lineno)
            case ast.Name() | ast.Attribute() if qualified(node, named) in SPAWNS:
                found.add(node.lineno)
            case ast.Attribute(attr=attr) if attr in LOOP_SPAWNS:
                found.add(node.lineno)
            # `from os import fork` binds the start by its own name, which the reference reads as.
            case ast.ImportFrom(module=str(module), names=aliases, level=0) if any(f"{module}.{alias.name}" in SPAWNS for alias in aliases):
                found.add(node.lineno)
            case _:
                pass
    return found - owned


def _written(text: str) -> set[int]:
    try:
        return spawns(textwrap.dedent(text))
    except (SyntaxError, ValueError, RecursionError, MemoryError, tokenize.TokenError):
        return set()


def test_every_start_of_a_process_is_the_runner_s_or_names_its_owner() -> None:
    root = Path(hands.__file__).parent
    modules = sorted(path for path in root.rglob("*.py") if path.relative_to(root).as_posix() != RUNNER)
    assert modules
    found = {name: lines for path in modules if (lines := spawns(path.read_text(), ".".join([root.name, *path.relative_to(root).parent.parts]))) for name in [path.relative_to(root).as_posix()]}
    assert found == {}


def test_the_check_catches_each_way_of_starting_a_process() -> None:
    source = (
        "import subprocess\nimport subprocess as sp\nfrom subprocess import run as r\nimport os, pty, asyncio\n"
        "subprocess.Popen(['a'])\nsp.run(['a'])\nr(['a'])\nos.fork()\nos.posix_spawn(x, [], {})\nos.system('a')\n"
        "pty.fork()\nasyncio.create_subprocess_exec('a')\nloop.subprocess_exec(f, 'a')\npartial(subprocess.check_output, ['a'])\n"
        "run(['python', '-c', 'import subprocess\\nsubprocess.run([\"a\"])'])\n"
        "os.fork()  # owner: the test\nsubprocess.DEVNULL\nos.getpid()\n"
    )
    # Line 3 binds `r`, a start by another name, and the import itself is a start reached.
    assert spawns(source) == {3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
