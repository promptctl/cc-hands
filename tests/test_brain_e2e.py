"""The brain end to end: hands' stage, its proxy, and the pinned Claude Code in a terminal of hands' own, with the
model's replies scripted by the mock of Anthropic's API. No login, no subscription, no network: the mock is the upstream
hands' proxy forwards to, and the brain's way to every other host Claude Code calls, and its OAuth login is one the mock
accepts. What each test proves is what hands does with the replies it scripts, the same on every run."""

import asyncio
import json
import os
import shlex
import subprocess
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

from conftest import running
from hands.brain.stage import OWED, BrainStage
from hands.core.brainwork import Idle, Running, Stopping, Working
from hands.core.wire import Sent
from hands.core.front import FrontUnread, InFront
from hands.brain import pinned
from hands.brain.process import workdir
from hands.daemon.run import mind
from hands.sessions.audit import Entry
from hands.sessions.home import Home
from hands.sessions.proxy import Wire, serve_proxy
from hands.sessions.registry import Sessions
from hands.voice import voices
from hands.voice.backends import Account, ClaudeCodeBackend
from hands.voice.conversation import UserWords
from hands.voice.pipeline import VoiceConfig
from hands.voice.refocus import Refocus
from hands.sessions.sentences import Sentences
from hands.voice.sentences import SummaryStore
from hands.voice.tools import stay_silent_tool
from test_stage import Spoken

# The canonical mock of Anthropic's API, at the release these tests were written against. Moving it is one commit, and
# CI's cache of it is keyed on it.
MOCK_MODULE = "github.com/promptctl/mock-anthropic"
MOCK_RELEASE = "v0.1.0"
# A turn of the real Claude Code against the mock takes a few seconds; a minute is a hang.
PATIENCE_SECONDS = 60.0
# The account the brain's login holds. `login` and `serve` both read it from a file, so it is the one source of what the
# login says, and ACCOUNT names the same address the audit records.
MOCK_ACCOUNT = {
    "accountUuid": "00000000-0000-4000-8000-0000000000a1",
    "emailAddress": "user@example.com",
    "organizationUuid": "00000000-0000-4000-8000-0000000000a2",
    "organizationName": "Hands Test Organization",
    "subscriptionType": "max",
}
ACCOUNT = Account("claude.ai", MOCK_ACCOUNT["emailAddress"])
# What only the brain's own turns carry, among every request its Claude Code makes: hands' tools offered to the model.
# A title or a summary Claude Code asks for beside a turn carries the user's words too, but none of hands' tools.
TURN = "mcp__hands__stay_silent"
# What only a request carrying a tool's result holds, as Claude Code's compact JSON writes the block. The same words in
# a prompt's text are escaped inside a JSON string, so they never match.
RESULT = '"type":"tool_result"'


@dataclass(frozen=True)
class MockApi:
    """The mock, listening: its url, the CA its TLS is signed by, and the directory each test scripts its replies in."""

    url: str
    port: int
    ca: Path
    scenarios: Path

    def script(self, name: str, match: dict[str, list[str]], *content: dict[str, object], stop: str = "end_turn") -> None:
        """The model's reply to every Messages request `match` accepts: `content` blocks, ending on `stop`. Scripts are
        tried in the order of their names, so a narrower match is named ahead of a wider one. Claude Code's own calls
        beside a turn (a title, the auto mode classifier, which allows) are the mock's built-in answers."""
        script = {"match": match, "reply": {"kind": "message", "content": list(content), "stop_reason": stop}}
        # [LAW:no-silent-failure] the mock reads a script the moment a request needs it, so a script is written under a
        # dot-name the mock ignores and renamed into place: a request never reads it half-written.
        staging = self.scenarios / f".{name}.tmp"
        staging.write_text(json.dumps(script))
        os.replace(staging, self.scenarios / f"{name}.json")


def mock(binary: Path, *argv: str) -> tuple[str, ...]:
    """The mock's command line: its release binary, run with `argv`."""
    return (str(binary), *argv)


@pytest.fixture(scope="session")
def mock_binary(pytestconfig: pytest.Config) -> Path:
    """The mock at MOCK_RELEASE, built once by `go install` into pytest's cache, a directory per release."""
    cache = pytestconfig.cache.mkdir("mock-anthropic") if pytestconfig.cache else pytest.fail("the cache provider is off")
    binary = cache / MOCK_RELEASE / "mock-anthropic"
    if not binary.is_file():
        # [LAW:no-silent-failure] an install that fails stops the session in go's own words. The mock writes its CA to a
        # certs directory beside the binary, so GOBIN is the directory the install puts it in, and it must be writable.
        binary.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["go", "install", f"{MOCK_MODULE}@{MOCK_RELEASE}"], check=True, env={**os.environ, "GOBIN": str(binary.parent)})
    return binary


@pytest.fixture
def account_file(tmp_path: Path) -> Path:
    """The account file `login` and `serve` both read, holding MOCK_ACCOUNT."""
    path = tmp_path / "account.json"
    path.write_text(json.dumps(MOCK_ACCOUNT))
    return path


@pytest.fixture
async def mock_api(tmp_path: Path, mock_binary: Path, account_file: Path) -> AsyncGenerator[MockApi]:
    scenarios = tmp_path / "scenarios"
    scenarios.mkdir()
    served = await asyncio.create_subprocess_exec(
        *mock(mock_binary, "serve"), "--port", "0", "--scenarios", str(scenarios), "--account", str(account_file), stdout=asyncio.subprocess.PIPE, stderr=(tmp_path / "mock.log").open("w")
    )
    assert served.stdout is not None
    try:
        # [LAW:no-silent-failure] a mock that never says where it listens fails the test with what it logged.
        line = await asyncio.wait_for(served.stdout.readline(), PATIENCE_SECONDS)
        if not line:
            raise AssertionError(f"the mock exited before it listened: {(tmp_path / 'mock.log').read_text()}")
        ready = json.loads(line)
        yield MockApi(f"http://127.0.0.1:{ready['port']}", ready["port"], Path(ready["ca"]), scenarios)
    finally:
        served.terminate()
        await served.wait()
    # [LAW:no-silent-failure] a request the mock had no answer for is a gap in the test or in the mock, never noise: Claude
    # Code was told 400 and went on some way the test did not script.
    unanswered = [line for line in (tmp_path / "mock.log").read_text().splitlines() if line.startswith(("UNMATCHED", "REFUSED", "FAILED"))]
    assert not unanswered, "\n".join(unanswered)


@pytest.fixture(scope="session")
async def claude_code(pytestconfig: pytest.Config) -> Path:
    """The directory holding the brain's pinned Claude Code, fetched once, against its checksum, into pytest's cache."""
    cache = pytestconfig.cache.mkdir("claude-code") if pytestconfig.cache else pytest.fail("the cache provider is off")
    await pinned.install(cache)
    return cache / "pinned-claude"


def mock_environment(binary: Path, port: int, ca: Path, config_dir: Path) -> dict[str, str]:
    """The variables that point Claude Code at the mock, as `mock env` names them, read back the way a shell reads them.
    The plain-http proxy variables are left out: hands' MCP server is plain http on loopback, and a plain-http proxy sends
    that call to the mock, which has no answer for it, so the brain would run without hands' tools."""
    printed = subprocess.run(mock(binary, "env", "--port", str(port), "--ca", str(ca), "--config-dir", str(config_dir)), check=True, capture_output=True, text=True).stdout
    variables = dict(shlex.split(line)[0].split("=", 1) for line in printed.splitlines())
    return {name: value for name, value in variables.items() if name not in ("HTTP_PROXY", "http_proxy")}


def logged_in(brain: Path, claude_code: Path, mock_binary: Path, account_file: Path) -> None:
    """A brain directory logged in as the account in `account_file` by the mock's own login, the directory it works in
    trusted, and its pinned Claude Code the cached one."""
    # [LAW:one-source-of-truth] what a login the mock accepts looks like is the mock's to say, never a copy here.
    subprocess.run(mock(mock_binary, "login", str(brain), "--trust", str(workdir(brain).resolve()), "--account", str(account_file)), check=True)
    (brain / "pinned-claude").symlink_to(claude_code)
    (brain / "settings.json").write_text(json.dumps({"syncClaudeAiSkills": False, "syncClaudeAiPlugins": False}))


@dataclass(frozen=True)
class Hands:
    """hands' brain, up against the mock: what its stage handed the speaker, and the pipeline to hand the user's words in."""

    stage: BrainStage
    out: Spoken
    say: Callable[[str], Awaitable[None]]
    recorded: list[Entry]
    # Every request the brain sent, as it left hands' proxy for the mock.
    sent: list[Sent]

    async def until(self, what: Callable[[], bool]) -> None:
        try:
            async with asyncio.timeout(PATIENCE_SECONDS):
                while not what():
                    await asyncio.sleep(0.05)
        except TimeoutError as error:
            # [LAW:nothing-unseen] a wait that gives up says what the audit held meanwhile, the MCP events above all.
            seen = [str(entry) for entry in self.recorded if "mcp" in str(entry).lower() or "brain." in str(entry)]
            requests = [str(request.kind) for request in self.sent]
            raise AssertionError(f"the brain did not get there within {PATIENCE_SECONDS:g}s; its MCP and turn events: {seen}; the requests it sent: {requests}; its screen: {self.stage.screen()!r}") from error


async def unread() -> InFront:
    return FrontUnread("not read in this test")


@pytest.fixture
async def hands(tmp_path: Path, mock_api: MockApi, mock_binary: Path, claude_code: Path, account_file: Path) -> AsyncGenerator[Hands]:
    brain = tmp_path / "brain"
    logged_in(brain, claude_code, mock_binary, account_file)
    recorded: list[Entry] = []
    sent: list[Sent] = []
    wire = Wire(lambda observed: sent.append(observed) if isinstance(observed, Sent) else None)
    # [LAW:effects-at-boundaries] the one change from the run's own wiring: the upstream hands' proxy forwards to.
    proxy = await serve_proxy(mock_api.url, wire.observe, wire.route, clock=time.time)
    home = tmp_path / "home"
    home.mkdir()
    # Every other host Claude Code calls is reached through the mock, and nothing reaches past it: the variables that say so
    # are the mock's to name. HOME is a fresh empty directory, so no real Claude Code profile is read. NO_PROXY is hands' own:
    # its proxy listens on loopback, and the brain reaches it directly.
    environment = {"PATH": os.environ["PATH"], "HOME": str(home), **mock_environment(mock_binary, mock_api.port, mock_api.ca, brain), "NO_PROXY": "127.0.0.1,localhost"}
    refocus = Refocus(Sessions(clock=lambda: 0.0, record=recorded.append), Home(tmp_path / "hands"), recorded.append)
    config = VoiceConfig(llm=ClaudeCodeBackend(model="claude-sonnet-5", config_dir=brain, account=ACCOUNT), voice=voices.DEFAULT)
    out = Spoken()
    store = SummaryStore(Sentences(tmp_path / "sentences.db"))
    try:
        async with mind(config, [stay_silent_tool()], [], lambda: "", unread, lambda: "screen", lambda: "held key", refocus, proxy.url, wire, store, tmp_path / "audit", "hands", recorded.append, environment, out.cut_off) as minded:
            async with running([minded.llm, out]) as run:
                watches = [asyncio.create_task(watch.run()) for watch in minded.watches]
                try:
                    yield Hands(minded.llm, out, lambda words: run.worker.queue_frame(UserWords(words)), recorded, sent)
                finally:
                    for watch in watches:
                        watch.cancel()
    finally:
        await proxy.close()
        # [LAW:nothing-unseen] what the brain sent is kept beside the mock's log, for a failure to be read from.
        (tmp_path / "sent.json").write_text(json.dumps([request.body for request in sent], indent=1))


async def test_a_turn_that_acted_and_closed_silent_is_told_to_say_what_it_did_and_hands_says_it(tmp_path: Path, mock_api: MockApi, hands: Hands) -> None:
    """The brain runs a shell command for the user, then calls stay_silent: hands refuses the silence once, telling it what
    it owes, and the sentence it answers with is said (promptctl-silence-bsog.sfi)."""
    made = tmp_path / "made"
    # Tried in name order: the refusal's answer ahead of the silence, which would match the request carrying it too.
    mock_api.script("1-owed", {"bodyContains": [OWED[:60]]}, {"type": "text", "text": "I made the file."})
    mock_api.script("2-silent", {"bodyContains": [TURN, RESULT]}, {"type": "tool_use", "name": "mcp__hands__stay_silent", "input": {}}, stop="tool_use")
    mock_api.script("3-act", {"lastUserContains": ["make the file"], "bodyContains": [TURN]}, {"type": "tool_use", "name": "Bash", "input": {"command": f"touch {made}"}}, stop="tool_use")
    await hands.say("make the file")
    await hands.until(lambda: "I made the file." in "".join(hands.out.said()))
    assert made.exists()


async def test_the_brain_s_running_call_is_shown_and_its_stop_ends_the_turn_and_the_next_is_answered(mock_api: MockApi, hands: Hands) -> None:
    """While the brain runs a long command, its work shows the call and its input; a stop pressed on that turn ends it, the
    brain is idle, and the turn after it is answered (promptctl-backlog-7qoh.8nx)."""
    mock_api.script("1-wait", {"lastUserContains": ["wait a while"], "bodyContains": [TURN]}, {"type": "tool_use", "name": "Bash", "input": {"command": "sleep 120"}}, stop="tool_use")
    mock_api.script("2-hello", {"lastUserContains": ["say hello"], "bodyContains": [TURN]}, {"type": "text", "text": "Hello."})
    await hands.say("wait a while")
    await hands.until(lambda: isinstance(work := hands.stage.work(), Working) and [call.tool for call in work.calls] == ["Bash"])
    work = hands.stage.work()
    assert isinstance(work, Working)
    [running] = work.calls
    assert isinstance(running, Running) and running.input == {"command": "sleep 120"}
    assert hands.stage.stop(work.turn) == Stopping()
    await hands.until(lambda: hands.stage.work() == Idle())
    await hands.say("say hello")
    await hands.until(lambda: "Hello." in "".join(hands.out.said()))
