"""What the pinned Claude Code does as the brain's conversation moves: the order of its SessionStart, its SessionEnd and
the requests of the session it moves to, at its launch, a /clear and a /resume, with a SessionStart command that takes
seconds to return. Each measurement is a line in docs/claude-code-behavior.md; what hands builds on them is the unit
layer's (promptctl-brain-uoz0.mvv)."""

import asyncio
import json
import os
import shlex
import sys
import tempfile
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path

import pytest
from aiohttp import web

from conftest import located
from pinnedbrain import PATIENCE_SECONDS, MockApi, logged_in
from pinnedbrain import claude_code, mock_api, mock_binary  # pyright: ignore[reportUnusedImport]
from hands.brain import pinned
from hands.brain.process import Fresh, InTmux, Station, _typist, in_tmux, settings_mode, slim, workdir  # pyright: ignore[reportPrivateUsage]
from hands.clock import SYSTEM
from hands.core.effects import Command
from hands.core.session import CommandName, PromptText, SessionId
from hands.core.wire import Sent
from hands.sessions.wrapper import packaged
from hands.sessions.proxy import Wire, serve_proxy
from hands.sessions.typing import Typist

pytestmark = pytest.mark.outside

# Long beside anything else Claude Code does as a session starts: a request or an input that does not wait for the start
# is seen ahead of its return.
START_SECONDS = 3
# Each hook is one line of the log, the one posted by HTTP as each run by command: its event, its session, and the source of a start or the reason of an end. A start
# writes a second line, `returned` with its session and source, once it is done.
HOOK = """import json, sys, time
said = json.load(sys.stdin)
log = open(sys.argv[1], "a")
event = said["hook_event_name"]
print(event, said["session_id"], said.get("source") or said.get("reason") or "", file=log, flush=True)
if event == "SessionStart":
    time.sleep(float(sys.argv[2]))
    print("returned", said["session_id"], said["source"], file=log, flush=True)
"""


@dataclass(frozen=True)
class Moves:
    """The pinned Claude Code, typed into by fritter, with every hook it ran in `hooks` and every request it sent paired
    with the hooks that had run as it reached hands' proxy."""

    launched: SessionId
    log: Path
    typist: Typist
    running: InTmux
    seen: list[tuple[Sent, list[str]]]
    # The hooks that had run as Claude Code turned its input on.
    up: list[str]

    def hooks(self) -> list[str]:
        return self.log.read_text().splitlines()

    async def until(self, line: str, ran: int = 0) -> None:
        """Waits for `line` among the hooks run after the first `ran`."""
        try:
            async with asyncio.timeout(PATIENCE_SECONDS):
                while line not in self.hooks()[ran:]:
                    await asyncio.sleep(0.05)
        except TimeoutError:
            raise AssertionError(f"no {line!r} in {self.hooks()}, after {[(sent.session, sent.kind) for sent, _ in self.seen]}; the screen:\n{await self.running.screen()}") from None

    def turn(self, words: str) -> None:
        self.typist.type(PromptText(words), "enter")

    def run(self, name: str, args: str | None = None) -> None:
        """A slash command, typed as hands types one: its name as keys, its arguments pasted, then Return."""
        self.typist.command(Command(CommandName(name), None if args is None else PromptText(args)))


@pytest.fixture
async def moves(tmp_path: Path, mock_api: MockApi, mock_binary: Path, claude_code: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncGenerator[Moves]:
    brain = tmp_path / "brain"
    logged_in(brain, mock_binary)
    monkeypatch.setattr(pinned, "claude", located(claude_code))
    for words in ("one", "two", "three"):
        mock_api.script(f"1-{words}", {"lastUserContains": [words]}, {"type": "text", "text": f"Said {words}."})
    log = tmp_path / "hooks.log"
    log.touch()
    hook = tmp_path / "hook.py"
    hook.write_text(HOOK)
    seen: list[tuple[Sent, list[str]]] = []
    wire = Wire(SYSTEM, lambda observed: seen.append((observed, log.read_text().splitlines())) if isinstance(observed, Sent) else None)
    proxy = await serve_proxy(mock_api.url, wire.observe, wire.route, clock=SYSTEM)
    home = tmp_path / "home"
    home.mkdir()
    environment = {"HOME": str(home), "HTTPS_PROXY": mock_api.url, "HTTP_PROXY": mock_api.url, "NODE_EXTRA_CA_CERTS": str(mock_api.ca), "NO_PROXY": "127.0.0.1,localhost", "CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL": "1"}
    station = Station(brain, workdir(brain), "claude-sonnet-5", proxy.url, os.environ | environment)
    logged = [{"hooks": [{"type": "command", "command": shlex.join([sys.executable, "-I", str(hook), str(log), str(START_SECONDS)])}]}]
    # The mode the brain's settings name said in --settings too, as the brain's command line says it: otherwise Claude Code
    # offers to make it the default once it is idle, and the offer takes the first turn typed.
    # SessionEnd posted as the brain posts it, by HTTP: the pin runs no HTTP hook for SessionStart (docs/claude-code-behavior.md).
    ended = await _ended_at(log)
    hooks = {event: logged for event in ("SessionStart", "UserPromptSubmit", "Stop")} | {"SessionEnd": [{"hooks": [{"type": "http", "url": ended.url}]}]}
    settings = {"hooks": hooks, "permissions": {"defaultMode": settings_mode(brain)}}
    launched = SessionId(str(uuid.uuid4()))
    sockets = Path(tempfile.mkdtemp(prefix="hands-moves-"))
    running: InTmux = await in_tmux(SYSTEM, station, [str(packaged()), "--socket-dir", str(sockets), "--", *slim(pinned.claude(brain), station.model, Fresh(launched, None)), "--settings", json.dumps(settings)], sockets)
    try:
        typist, _wrapper = await _typist(SYSTEM, running, sockets, launched)  # pyright: ignore[reportPrivateUsage]
        yield Moves(launched, log, typist, running, seen, log.read_text().splitlines())
    finally:
        await running.stop()
        await proxy.close()
        await ended.runner.cleanup()


@dataclass(frozen=True)
class Listening:
    runner: web.AppRunner
    url: str


async def _ended_at(log: Path) -> Listening:
    """A listener on loopback that writes each SessionEnd posted to it as the hook script writes a hook."""

    async def ended(request: web.Request) -> web.Response:
        said = await request.json()
        with log.open("a") as written:
            print(said["hook_event_name"], said["session_id"], said["reason"], file=written, flush=True)
        return web.json_response({})

    app = web.Application()
    app.router.add_post("/", ended)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    host, port = runner.addresses[0][:2]
    return Listening(runner, f"http://{host}:{port}/")


async def test_the_launch_start_runs_as_the_input_comes_up_and_a_turn_typed_then_waits_for_it(moves: Moves) -> None:
    claude, launched = moves, moves.launched
    assert claude.up == [f"SessionStart {launched} startup"]
    claude.turn("one")
    await claude.until(f"Stop {launched} ")
    assert _order(claude, 0, launched, launched, "startup") == [f"SessionStart {launched} startup", f"returned {launched} startup"]


async def test_a_clear_and_a_resume_end_the_old_session_and_start_the_new_one_before_its_first_request(moves: Moves) -> None:
    claude, launched = moves, moves.launched
    claude.turn("one")
    await claude.until(f"Stop {launched} ")
    since = len(claude.seen)
    claude.run("clear")
    # The turn typed at once behind the command: whether it waits for the start to return.
    claude.turn("two")
    cleared = await _started(claude, "clear")
    await claude.until(f"Stop {cleared} ")
    assert _order(claude, since, launched, cleared, "clear") == [f"SessionEnd {launched} clear", f"SessionStart {cleared} clear", f"returned {cleared} clear"]
    claude.run("resume", launched)
    # Typed once the resume has begun: a turn typed at once behind it may be taken before it, in the session it leaves.
    await claude.until(f"SessionEnd {cleared} resume")
    since, ran = len(claude.seen), len(claude.hooks())
    claude.turn("three")
    await claude.until(f"Stop {launched} ", ran)
    assert _order(claude, since, cleared, launched, "resume") == [f"SessionEnd {cleared} resume", f"SessionStart {launched} resume", f"returned {launched} resume"]

async def _started(claude: Moves, source: str) -> SessionId:
    async with asyncio.timeout(PATIENCE_SECONDS):
        while True:
            for line in claude.hooks():
                event, session, said = line.split(" ")
                if event == "SessionStart" and said == source:
                    return SessionId(session)
            await asyncio.sleep(0.05)


def _order(claude: Moves, since: int, left: str, entered: str, how: str) -> list[str]:
    """The end of `left`, the start of `entered` and its return, among the hooks that had run as the first request of
    `entered` sent after the move, the `since`th request on, reached the proxy."""
    after = [hooks for sent, hooks in claude.seen[since:] if sent.session == entered]
    assert after, "\n".join([*claude.hooks(), *(f"request {sent.session} {sent.kind}" for sent, _ in claude.seen[since:])])
    return [line for line in after[0] if line in (f"SessionEnd {left} {how}", f"SessionStart {entered} {how}", f"returned {entered} {how}")]
