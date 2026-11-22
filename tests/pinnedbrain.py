"""The brain end to end, for the tests that must have the pinned Claude Code itself: hands' stage, its proxy, and the pinned
Claude Code in a terminal of hands' own, with the model's replies scripted by the mock of Anthropic's API. No login, no
subscription, and no network while a test runs: the mock is the upstream hands' proxy forwards to, and the brain's way to
every other host Claude Code calls, and its OAuth login is one the mock accepts. The OS holds the brain to it: the run, and
every Claude Code hands starts in it, runs in a sandbox whose only connections are to loopback (conftest). The pinned Claude
Code, from npm's registry, and the mock are fetched before the run, into pytest's cache (conftest).

Only what is a claim about the pinned Claude Code, or about the wiring of the assembled system, is proven here: each of
hands' own decisions is asserted in the unit layer, on Claude Code's turns as recorded (tests/turns.py). The fixtures are
imported by the modules that use them."""

import asyncio
import json
import os
import subprocess
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from pipecat.frames.frames import ErrorFrame, Frame, InterruptionFrame, UserStartedSpeakingFrame, UserStoppedSpeakingFrame

from conftest import children_on, located, new_sessions, RUN, running, unread
from spoken import Received, Spoken
from hands.brain import pinned
from hands.brain.mcp import SERVER_NAME
from hands.brain.process import starting_settings, workdir
from hands.clock import SYSTEM
from hands.core.attention import Attention
from hands.core.wire import SIDE_QUESTION_OPENING, Sent
from hands.daemon.run import mind
from hands.sessions.audit import Entry
from hands.sessions.home import Home
from hands.sessions.proxy import Wire, serve_proxy
from hands.sessions.summaries import Summaries
from hands.sessions.tmux import TmuxExit, ran_at
from hands.brain.stage import BrainStage
from hands.voice import voicechoice
from hands.voice.backends import Account, ClaudeCodeBackend
from hands.voice.conversation import UserWords
from hands.voice.pipeline import VoiceConfig
from hands.voice.refocus import Refocus
from hands.voice.summaries import SummaryStore
from hands.voice.tools import stay_silent_tool
from hands.sessions.wide import WideEvent

# A turn of the real Claude Code against the mock takes a few seconds; a minute is a hang. The one wait the pinned Claude
# Code's screen forces: nothing tells a test its screen changed, so it is read until it shows what the test waits for.
PATIENCE_SECONDS = 60.0
ACCOUNT = Account("claude.ai", "user@example.com")
# What only the brain's own turns carry, among every request its Claude Code makes: hands' tools offered to the model.
# A title or a summary Claude Code asks for beside a turn carries the user's words too, but none of hands' tools.
TURN = f"mcp__{SERVER_NAME}__{stay_silent_tool().name}"
# The words a side question's prompt opens with, past the reminder's tag: nothing in them is escaped in a JSON string.
SIDE_QUESTION = SIDE_QUESTION_OPENING.removeprefix("<system-reminder>")
# What only a request carrying a tool's result holds, as Claude Code's compact JSON writes the block.
RESULT = '"type":"tool_result"'
# The outcomes the mock writes its records under that are no fault: a request served, a connection closed, a request
# whose client left before its answer, as Claude Code leaves a call it no longer needs, and the serve command's own end.
FINE = ("SERVED ", "CLOSED ", "ABORTED ", "OK ")


@dataclass(frozen=True)
class MockApi:
    """The mock, listening: its url, the CA its TLS is signed by, and the directory each test scripts its replies in."""

    url: str
    ca: Path
    scenarios: Path

    def script(self, name: str, match: dict[str, list[str]], *content: dict[str, object], stop: str = "end_turn") -> None:
        """The model's reply to every Messages request `match` accepts: `content` blocks, ending on `stop`. Scripts are
        tried in the order of their names, so a narrower match is named ahead of a wider one. Claude Code's own calls
        beside a turn (a title, the auto mode classifier, which allows) are the mock's built-in answers."""
        script = {"match": match, "reply": {"kind": "message", "content": list(content), "stop_reason": stop}}
        (self.scenarios / f"{name}.json").write_text(json.dumps(script))


@pytest.fixture(scope="session")
def mock_binary(pytestconfig: pytest.Config) -> Path:
    """The mock at the release conftest pins, as the run built it into pytest's cache."""
    return pytestconfig.stash[RUN].mock


@pytest.fixture
async def mock_api(tmp_path: Path, mock_binary: Path) -> AsyncGenerator[MockApi]:
    scenarios = tmp_path / "scenarios"
    scenarios.mkdir()
    log = tmp_path / "mock.log"
    with log.open("w") as stderr:
        served = await asyncio.create_subprocess_exec(str(mock_binary), "serve", "--port", "0", "--scenarios", str(scenarios), stdout=asyncio.subprocess.PIPE, stderr=stderr)
    assert served.stdout is not None
    try:
        # [LAW:no-silent-failure] a mock that never says where it listens, or says it unreadably, fails the test with what it
        # logged.
        try:
            line = await asyncio.wait_for(served.stdout.readline(), PATIENCE_SECONDS)
            ready = json.loads(line)
            api = MockApi(f"http://127.0.0.1:{ready['port']}", Path(ready["ca"]), scenarios)
        except (TimeoutError, ValueError, KeyError, TypeError) as error:
            raise AssertionError(f"the mock did not say where it listens ({error!r}): {log.read_text()}") from error
        # Every old tool result is asked as a side question forked from the brain's turn, whose body also holds what the
        # turn's own scripts match: its answer is named ahead of them all.
        api.script("0-aside", {"bodyContains": [SIDE_QUESTION]}, {"type": "text", "text": "It ran."})
        yield api
    finally:
        served.terminate()
        await served.wait()
    # [LAW:no-silent-failure] every other record is a gap in the test or in the mock, never noise: a request it had no
    # answer for, a script or a handler that failed, a connection it refused; Claude Code went on some way the test did not
    # script. An outcome the mock adds later fails here until it is named.
    faults = [line for line in log.read_text().splitlines() if not line.startswith(FINE)]
    assert not faults, "\n".join(faults)


@pytest.fixture(scope="session")
def claude_code(pytestconfig: pytest.Config) -> Path:
    """The brain's pinned Claude Code, as the run fetched it from npm's registry, against its checksum, into pytest's cache."""
    return pytestconfig.stash[RUN].claude_code


def logged_in(brain: Path, mock_binary: Path) -> None:
    """A brain directory logged in as the mock's account by the mock's own login, and the directory it works in trusted."""
    # [LAW:one-source-of-truth] what a login the mock accepts looks like is the mock's to say, never a copy here.
    subprocess.run([str(mock_binary), "login", str(brain), "--trust", str(workdir(brain).resolve())], check=True)
    # [LAW:one-source-of-truth] the settings a brain home starts with are the ones hands writes for every brain.
    wrote = starting_settings(brain)
    assert wrote
    # No command is allowed beyond what the brain's own settings allow: the contract's dialog case is a command they do not
    # (test_terminal_contract_pinned.py), and how hands puts a dialog to the user is the unit layer's.


@dataclass(frozen=True)
class Hands:
    """hands' brain, up against the mock: what its stage handed the speaker, and the pipeline to hand the user's words in."""

    stage: BrainStage
    out: Spoken
    # What hands' voice pipeline is handed, as the user aggregator hands it on.
    queue: Callable[[Frame], Awaitable[None]]
    recorded: list[Entry]
    # Every request the brain sent, as it left hands' proxy for the mock.
    sent: list[Sent]
    # What the run keeps beside the pipeline: the brain itself, its turns, its context.
    watches: list[asyncio.Task[None]]
    # What the pipeline raised, which the system channel would say.
    errors: list[ErrorFrame]
    # The directory the brain works in, where what it makes is made.
    workdir: Path

    async def until(self, what: Callable[[], bool]) -> None:
        patience = asyncio.timeout(PATIENCE_SECONDS)
        try:
            async with patience:
                while not what():
                    # [LAW:no-silent-failure] a watch that ends stops hands' run, and an error the pipeline raises is said to
                    # the user, so here each fails the test with its reason, not a minute later as a timeout.
                    assert not self.errors, self.errors
                    for watch in self.watches:
                        if watch.done():
                            raise AssertionError(f"{watch.get_name()} ended") from (None if watch.cancelled() else watch.exception())
                    await asyncio.sleep(0.05)
        except TimeoutError:
            # Only the wait's own running out is reported as one; any other TimeoutError is its own failure.
            if not patience.expired():
                raise
            # [LAW:no-silent-failure] a wait that runs out says where hands was: what it said, and what the brain was doing.
            raise AssertionError(f"waited {PATIENCE_SECONDS:g}s: hands said {self.out.said()!r}, the brain's work {self.stage.work()!r}, after {len(self.sent)} requests through hands' proxy, the brain's screen:\n{self.shown()}") from None

    async def say(self, words: str) -> None:
        """The user's words, spoken once they have heard all hands handed the speaker, its marks played: behind the cut
        their turn opens with and the turn's stop, as every turn that carries words is cut (test_stage's `barge_in`). A
        question is answered by what is said after it is heard, so words handed on without the cut are a turn of their own."""
        reached = asyncio.Event()
        await self.queue(Received(reached.set))
        await self.until(reached.is_set)
        cuts = self.out.shape().count("InterruptionFrame")
        await self.queue(UserStartedSpeakingFrame())
        await self.queue(InterruptionFrame())
        await self.until(lambda: self.out.shape().count("InterruptionFrame") > cuts)
        stops = self.out.stops
        await self.queue(UserStoppedSpeakingFrame())
        await self.until(lambda: self.out.stops > stops)
        await self.queue(UserWords(words))

    async def connected(self) -> None:
        """Waits until the brain has listed hands' tools over MCP. Claude Code connects to hands' server after it starts, not
        before, so a turn typed sooner goes to the model without them, and whether it does is a race the test must not have."""
        await self.until(lambda: any(isinstance(entry, WideEvent) and entry.event == "mcp.request" and entry.facts.get("method") == "tools/list" for entry in self.recorded))

    def shown(self) -> str:
        """The last lines the brain's terminal showed."""
        return self._claude().shown()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    def held(self) -> tuple[object, ...]:
        """The dialogs the brain's model holds now: what Claude Code has stacked, the first on its screen."""
        session = self.stage._brain._modelled()  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]
        # Before the brain's first hook there is no session, and nothing is held: a wait on this is for that to change.
        return () if session is None else session.dialogs  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    async def at_the_pane(self, *command: str) -> None:
        """A tmux command run against the brain's pane, as a terminal attached to it types: tmux writes the keys to the
        pane, and the pane's fritter forwards them to Claude Code byte for byte."""
        claude = self._claude()
        ran = await ran_at(children_on(SYSTEM), claude._tmux, claude._socket, command)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportPrivateUsage]
        assert isinstance(ran, TmuxExit) and ran.code == 0, ran

    async def keyed(self, *keys: str) -> None:
        """`keys`, as tmux's send-keys names them, typed into the brain's pane."""
        await self.at_the_pane("send-keys", "-t", self._claude()._session, *keys)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportPrivateUsage]

    async def pasted(self, text: str) -> None:
        """`text` pasted into the brain's pane in one bracketed paste by tmux, its newlines kept."""
        await self.at_the_pane("set-buffer", "-b", "pasted", text, ";", "paste-buffer", "-p", "-r", "-d", "-b", "pasted", "-t", self._claude()._session)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportPrivateUsage]

    def _claude(self) -> Any:
        return self.stage._brain._claude  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]


@pytest.fixture
async def hands(tmp_path: Path, mock_api: MockApi, mock_binary: Path, claude_code: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncGenerator[Hands]:
    brain = tmp_path / "brain"
    logged_in(brain, mock_binary)
    # Every start of a claude, the brain's, an aside's, a login's, runs the cached Claude Code, the pin itself, held
    # offline as the run is (conftest): a connection it or anything it starts makes around the mock is refused by the OS,
    # and one through its proxy is in the mock's log.
    monkeypatch.setattr(pinned, "claude", located(claude_code))
    recorded: list[Entry] = []
    sent: list[Sent] = []
    wire = Wire(SYSTEM, lambda observed: sent.append(observed) if isinstance(observed, Sent) else None)
    # [LAW:effects-at-boundaries] the change from the run's own wiring, beside the pinned claude: the upstream hands' proxy
    # forwards to.
    proxy = await serve_proxy(mock_api.url, wire.observe, wire.route, clock=SYSTEM)
    try:
        home = tmp_path / "home"
        home.mkdir()
        # Every other host Claude Code calls is reached through the mock, and nothing reaches past it.
        # Claude Code clones its official plugin marketplace from GitHub at launch, which the mock does not answer (refused, and the
        # test fails on it); the marketplace is no part of what these tests prove, so the switch keeps the clone out of the run.
        environment = os.environ | {"HOME": str(home), "HTTPS_PROXY": mock_api.url, "HTTP_PROXY": mock_api.url, "NODE_EXTRA_CA_CERTS": str(mock_api.ca), "NO_PROXY": "127.0.0.1,localhost", "CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL": "1"}
        refocus = Refocus(new_sessions(clock=SYSTEM, record=recorded.append), Home(tmp_path / "hands"), recorded.append)
        config = VoiceConfig(llm=ClaudeCodeBackend(model="claude-sonnet-5", config_dir=brain, account=ACCOUNT), voice=voicechoice.DEFAULT)
        out = Spoken()
        store = SummaryStore(Summaries(tmp_path / "summaries.db"))
        async with mind(SYSTEM, config, [stay_silent_tool()], [], lambda: "", unread, lambda: "screen", lambda: "shortcut", refocus, proxy.url, wire, store, tmp_path / "audit", "hands", recorded.append, environment, out.next_barge_in, Attention) as minded:
            async with running([minded.llm, out]) as run:
                watches = [asyncio.create_task(watch.run(), name=watch.name) for watch in minded.watches]
                try:
                    yield Hands(minded.llm, out, run.worker.queue_frame, recorded, sent, watches, run.errors, workdir(brain))
                finally:
                    for watch in watches:
                        watch.cancel()
                    # [LAW:no-silent-failure] a watch that failed, or an error raised, after the test's last wait fails it here.
                    failed = [ended for ended in await asyncio.gather(*watches, return_exceptions=True) if isinstance(ended, Exception)]
                    assert not failed, failed
                    assert not run.errors, run.errors
    finally:
        await proxy.close()
        # What the brain sent is kept beside the mock's log, for a failure to be read from.
        (tmp_path / "sent.json").write_text(json.dumps([request.body for request in sent], indent=1))
