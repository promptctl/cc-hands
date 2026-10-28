"""The brain end to end: hands' stage, its proxy, and the pinned Claude Code in a terminal of hands' own, with the
model's replies scripted by the mock of Anthropic's API. No login, no subscription, and no network while a test runs: the
mock is the upstream hands' proxy forwards to, and the brain's way to every other host Claude Code calls, and its OAuth
login is one the mock accepts. The OS holds the brain to it: every Claude Code hands starts, and all it starts in turn,
runs in a sandbox whose only connections are to loopback. A machine's Claude Code policy reaches every Claude Code, so
the tests run on a machine with none, as CI's is. The pinned Claude Code and the mock are fetched once, into pytest's
cache. What each test proves is what hands does with the replies it scripts, the same on every run."""

import asyncio
import json
import os
import pwd
import shlex
import subprocess
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

import pytest
from pipecat.frames.frames import ErrorFrame

from conftest import running, unread
from hands.brain.stage import OWED, BrainStage
from hands.core.brainwork import Idle, Running, Stopping, Working
from hands.core.wire import Sent, tool_answers
from hands.brain import pinned
from hands.brain.process import starting_settings, workdir
from hands.daemon.run import mind
from hands.sessions.audit import Entry
from hands.sessions.wide import WideEvent
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
ACCOUNT = Account("claude.ai", "user@example.com")
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
    """The mock at MOCK_RELEASE, built once by `go install` into pytest's cache, a directory per release."""
    cache = pytestconfig.cache.mkdir("mock-anthropic") if pytestconfig.cache else pytest.fail("the cache provider is off")
    binary = cache / MOCK_RELEASE / "mock-anthropic"
    if not binary.is_file():
        # [LAW:no-silent-failure] an install that fails stops the session in go's own words. The binary is installed into its
        # release's directory, so the cache of one release is never taken for another.
        binary.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["go", "install", f"{MOCK_MODULE}@{MOCK_RELEASE}"], check=True, env={**os.environ, "GOBIN": str(binary.parent)})
    return binary


# The outcomes the mock writes its records under that are no fault: a request served, a connection closed, a request
# whose client left before its answer, as Claude Code leaves a call it no longer needs, and the serve command's own end.
FINE = ("SERVED ", "CLOSED ", "ABORTED ", "OK ")


@pytest.fixture
async def mock_api(tmp_path: Path, mock_binary: Path) -> AsyncGenerator[MockApi]:
    scenarios = tmp_path / "scenarios"
    scenarios.mkdir()
    log = tmp_path / "mock.log"
    with log.open("w") as stderr:
        served = await asyncio.create_subprocess_exec(str(mock_binary), "serve", "--port", "0", "--scenarios", str(scenarios), stdout=asyncio.subprocess.PIPE, stderr=stderr)
    assert served.stdout is not None
    try:
        # [LAW:no-silent-failure] a mock that never says where it listens fails the test with what it logged.
        line = await asyncio.wait_for(served.stdout.readline(), PATIENCE_SECONDS)
        if not line:
            raise AssertionError(f"the mock exited before it listened: {log.read_text()}")
        ready = json.loads(line)
        yield MockApi(f"http://127.0.0.1:{ready['port']}", Path(ready["ca"]), scenarios)
    finally:
        served.terminate()
        await served.wait()
    # [LAW:no-silent-failure] every other record is a gap in the test or in the mock, never noise: a request it had no
    # answer for, a script or a handler that failed, a connection it refused; Claude Code went on some way the test did not
    # script. An outcome the mock adds later fails here until it is named.
    faults = [line for line in log.read_text().splitlines() if not line.startswith(FINE)]
    assert not faults, "\n".join(faults)


@pytest.fixture(scope="session")
async def claude_code(pytestconfig: pytest.Config) -> Path:
    """The directory holding the brain's pinned Claude Code, fetched once, against its checksum, into pytest's cache."""
    cache = pytestconfig.cache.mkdir("claude-code") if pytestconfig.cache else pytest.fail("the cache provider is off")
    await pinned.install(cache)
    return cache / "pinned-claude"


# What a Claude Code hands starts may connect to, as macOS's sandbox holds it and every process it starts: loopback,
# where hands' proxy, its server and the mock listen, and local sockets, but not the system's resolver, which would ask
# the network for a name; `localhost` resolves without it, from /etc/hosts. A proxy setting reaches only the clients that
# honour it, and ssh, among others, does not.
OFFLINE = '(version 1) (allow default) (deny network-outbound) (allow network-outbound (remote ip "localhost:*")) (allow network-outbound (remote unix-socket)) (deny network-outbound (remote unix-socket (path-literal "/private/var/run/mDNSResponder")))'


def drop_ins(directory: Path) -> list[Path]:
    """The files of a policy drop-in directory Claude Code reads: each `*.json` whose name does not start with a dot."""
    try:
        return sorted(path for path in directory.iterdir() if path.name.endswith(".json") and not path.name.startswith("."))
    except FileNotFoundError:
        # No directory is no policy, as Claude Code reads it; one it cannot list fails the collection, loudly.
        return []


# Where the pinned Claude Code (2.1.308 when this was read from it) reads a machine's policy on macOS, which outranks
# every setting of the brain's own and which no process may opt out of: a settings file and its drop-ins, an MCP config
# that takes every MCP server over, and an MDM profile, the account's or the machine's. A release that reads another is
# read here when the pin moves.
MANAGED = Path("/Library/Application Support/ClaudeCode")
PREFERENCES = Path("/Library/Managed Preferences")
POLICY = [
    MANAGED / "managed-settings.json",
    *drop_ins(MANAGED / "managed-settings.d"),
    MANAGED / "managed-mcp.json",
    PREFERENCES / pwd.getpwuid(os.getuid()).pw_name / "com.anthropic.claudecode.plist",
    PREFERENCES / "com.anthropic.claudecode.plist",
]
HELD = [str(path) for path in POLICY if path.is_file()]

# A brain on the machine's policy is not the brain these tests prove: this Mac's, for one, names plugin marketplaces on
# GitHub the brain clones at its start, and runs it in auto mode with the person's hooks. Skipped before any fixture is
# built, and in CI a skip fails the run (conftest).
pytestmark = pytest.mark.skipif(bool(HELD), reason=f"this machine's Claude Code policy reaches every Claude Code, the brain under test among them: {', '.join(HELD)}")


def offline(verified: Callable[[Path], Path], brain: Path) -> Callable[[Path], Path]:
    """pinned.verified, handing back `brain`'s pin to run in the OFFLINE sandbox: a script that execs sandbox-exec, which
    execs the pin in its turn, so the process hands starts is that Claude Code, as without the sandbox. The pin is still
    checked at every start, as hands checks it."""
    script = brain / "offline-claude"
    script.write_text(f'#!/bin/sh\nexec /usr/bin/sandbox-exec -p {shlex.quote(OFFLINE)} {shlex.quote(str(verified(brain)))} "$@"\n')
    script.chmod(0o755)

    def sandboxed(config_dir: Path) -> Path:
        assert config_dir == brain, f"only {brain}'s Claude Code runs in the sandbox, and {config_dir}'s was started"
        verified(config_dir)
        return script

    return sandboxed


def logged_in(brain: Path, claude_code: Path, mock_binary: Path) -> None:
    """A brain directory logged in as the mock's account by the mock's own login, the directory it works in trusted, and
    its pinned Claude Code the cached one."""
    # [LAW:one-source-of-truth] what a login the mock accepts looks like is the mock's to say, never a copy here.
    subprocess.run([str(mock_binary), "login", str(brain), "--trust", str(workdir(brain).resolve())], check=True)
    (brain / "pinned-claude").symlink_to(claude_code)
    # [LAW:one-source-of-truth] the settings a brain home starts with are the ones hands writes for every brain.
    assert starting_settings(brain)
    # The commands the tests script, allowed as a brain's own settings allow a command, so no permission waits on a person:
    # how hands puts one to the user is test_permissions'.
    settings = brain / "settings.json"
    said = json.loads(settings.read_text())
    allowed = [*said["permissions"].get("allow", []), "Bash(touch:*)", "Bash(sleep:*)", "Bash(nc:*)"]
    settings.write_text(json.dumps({**said, "permissions": {**said["permissions"], "allow": allowed}}))


@dataclass(frozen=True)
class Hands:
    """hands' brain, up against the mock: what its stage handed the speaker, and the pipeline to hand the user's words in."""

    stage: BrainStage
    out: Spoken
    say: Callable[[str], Awaitable[None]]
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
            raise AssertionError(f"waited {PATIENCE_SECONDS:g}s: hands said {self.out.said()!r}, the brain's work {self.stage.work()!r}, after {len(self.sent)} requests through hands' proxy") from None


async def connected(hands: Hands) -> None:
    """Waits until the brain has listed hands' tools over MCP. Claude Code connects to hands' server after it starts, not
    before, so a turn typed sooner goes to the model without them, and whether it does is a race the test must not have."""
    await hands.until(lambda: any(isinstance(entry, WideEvent) and entry.event == "mcp.request" and entry.facts.get("method") == "tools/list" for entry in hands.recorded))


@pytest.fixture
async def hands(tmp_path: Path, mock_api: MockApi, mock_binary: Path, claude_code: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncGenerator[Hands]:
    brain = tmp_path / "brain"
    logged_in(brain, claude_code, mock_binary)
    # [LAW:single-enforcer] the one checkpoint for "no network while a test runs" is the Claude Code hands runs, which every
    # start of one, the brain's, an aside's, a login's, takes from pinned.verified: a connection it or anything it starts
    # makes around the mock fails here, at the OS, and one that goes through its proxy is named in the mock's log.
    monkeypatch.setattr(pinned, "verified", offline(pinned.verified, brain))
    recorded: list[Entry] = []
    sent: list[Sent] = []
    wire = Wire(lambda observed: sent.append(observed) if isinstance(observed, Sent) else None)
    # [LAW:effects-at-boundaries] the one change from the run's own wiring: the upstream hands' proxy forwards to.
    proxy = await serve_proxy(mock_api.url, wire.observe, wire.route, clock=time.time)
    home = tmp_path / "home"
    home.mkdir()
    # Every other host Claude Code calls is reached through the mock, and nothing reaches past it.
    # Claude Code clones its official plugin marketplace from GitHub at launch, which the mock does not answer (refused, and the
    # e2e fails on it); the marketplace is no part of what these tests prove, so the switch keeps the clone out of the run.
    environment = {"PATH": os.environ["PATH"], "HOME": str(home), "HTTPS_PROXY": mock_api.url, "HTTP_PROXY": mock_api.url, "NODE_EXTRA_CA_CERTS": str(mock_api.ca), "NO_PROXY": "127.0.0.1,localhost", "CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL": "1"}
    refocus = Refocus(Sessions(clock=lambda: 0.0, record=recorded.append), Home(tmp_path / "hands"), recorded.append)
    config = VoiceConfig(llm=ClaudeCodeBackend(model="claude-sonnet-5", config_dir=brain, account=ACCOUNT), voice=voices.DEFAULT)
    out = Spoken()
    store = SummaryStore(Sentences(tmp_path / "sentences.db"))
    try:
        async with mind(config, [stay_silent_tool()], [], lambda: "", unread, lambda: "screen", lambda: "held key", refocus, proxy.url, wire, store, tmp_path / "audit", "hands", recorded.append, environment, out.cut_off) as minded:
            async with running([minded.llm, out]) as run:
                watches = [asyncio.create_task(watch.run(), name=watch.name) for watch in minded.watches]
                try:
                    yield Hands(minded.llm, out, lambda words: run.worker.queue_frame(UserWords(words)), recorded, sent, watches, run.errors, workdir(brain))
                finally:
                    for watch in watches:
                        watch.cancel()
                    # [LAW:no-silent-failure] a watch that failed, or an error raised, after the test's last wait fails it here.
                    failed = [ended for ended in await asyncio.gather(*watches, return_exceptions=True) if isinstance(ended, Exception)]
                    assert not failed, failed
                    assert not run.errors, run.errors
    finally:
        await proxy.close()
        # [LAW:nothing-unseen] what the brain sent is kept beside the mock's log, for a failure to be read from.
        (tmp_path / "sent.json").write_text(json.dumps([request.body for request in sent], indent=1))


async def test_a_turn_that_acted_and_closed_silent_is_told_to_say_what_it_did_and_hands_says_it(mock_api: MockApi, hands: Hands) -> None:
    """The brain runs a shell command for the user, then calls stay_silent: hands refuses the silence once, telling it what
    it owes, and the sentence it answers with is said (promptctl-silence-bsog.sfi)."""
    # Where the brain works: on CI a command on a path outside it was asked about whatever the brain's settings allowed.
    made = hands.workdir / "made"
    # Tried in name order: the refusal's answer ahead of the silence, which would match the request carrying it too.
    mock_api.script("1-owed", {"bodyContains": [TURN, OWED[:60]]},{"type": "text", "text": "I made the file."})
    mock_api.script("2-silent", {"bodyContains": [TURN, RESULT]}, {"type": "tool_use", "name": "mcp__hands__stay_silent", "input": {}}, stop="tool_use")
    mock_api.script("3-act", {"lastUserContains": ["make the file"], "bodyContains": [TURN]}, {"type": "tool_use", "name": "Bash", "input": {"command": f"touch {made}"}}, stop="tool_use")
    await connected(hands)
    await hands.say("make the file")
    await hands.until(lambda: "I made the file." in "".join(hands.out.said()))
    assert made.exists()


async def test_the_brain_s_running_call_is_shown_and_its_stop_ends_the_turn_and_the_next_is_answered(mock_api: MockApi, hands: Hands) -> None:
    """While the brain runs a long command, its work shows the call and its input; a stop pressed on that turn ends it, the
    brain is idle, and the turn after it is answered (promptctl-backlog-7qoh.8nx)."""
    mock_api.script("1-wait", {"lastUserContains": ["wait a while"], "bodyContains": [TURN]}, {"type": "tool_use", "name": "Bash", "input": {"command": "sleep 120"}}, stop="tool_use")
    mock_api.script("2-hello", {"lastUserContains": ["say hello"], "bodyContains": [TURN]}, {"type": "text", "text": "Hello."})
    await connected(hands)
    await hands.say("wait a while")
    await hands.until(lambda: isinstance(work := hands.stage.work(), Working) and [call.tool for call in work.calls] == ["Bash"])
    work = hands.stage.work()
    assert isinstance(work, Working)
    [call] = work.calls
    assert isinstance(call, Running) and call.input == {"command": "sleep 120"}
    assert hands.stage.stop(work.turn) == Stopping()
    await hands.until(lambda: hands.stage.work() == Idle())
    await hands.say("say hello")
    await hands.until(lambda: "Hello." in "".join(hands.out.said()))


async def test_a_connection_the_brain_makes_past_loopback_is_refused_by_the_os(mock_api: MockApi, hands: Hands) -> None:
    """The brain hands started is the one in the sandbox: a command it runs that connects past loopback is refused by the
    OS before a packet leaves, so a way around the mock never reaches the machine's network."""
    # 192.0.2.1 is TEST-NET-1 (RFC 5737), routed nowhere: outside the sandbox the attempt times out, inside it is refused.
    mock_api.script("1-tried", {"bodyContains": [TURN, RESULT]}, {"type": "text", "text": "Tried."})
    mock_api.script("2-reach", {"lastUserContains": ["reach past loopback"], "bodyContains": [TURN]}, {"type": "tool_use", "name": "Bash", "input": {"command": "nc -zv -w 5 192.0.2.1 443"}}, stop="tool_use")
    await connected(hands)
    await hands.say("reach past loopback")
    await hands.until(lambda: "Tried." in "".join(hands.out.said()))
    # nc's own words, in the result of its call: Claude Code's prompt says "Operation not permitted" too.
    results = [answer.text for request in hands.sent for answer in tool_answers(request.body)]
    assert any("192.0.2.1" in result and "Operation not permitted" in result for result in results), results
