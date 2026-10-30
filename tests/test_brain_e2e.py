"""The brain end to end: hands' stage, its proxy, and the pinned Claude Code in a terminal of hands' own, with the
model's replies scripted by the mock of Anthropic's API. No login, no subscription, and no network while a test runs: the
mock is the upstream hands' proxy forwards to, and the brain's way to every other host Claude Code calls, and its OAuth
login is one the mock accepts. The OS holds the brain to it: the run, and every Claude Code hands starts in it, runs in
a sandbox whose only connections are to loopback (conftest). A machine's Claude Code policy reaches every Claude Code, so
the tests run on a machine with none, as CI's is. The pinned Claude Code, from npm's registry, and the mock are fetched
before the run, into pytest's cache (conftest). What each test proves is what hands does with the replies it scripts, the same on every run."""

import asyncio
import errno
import json

from hands.core.attention import Attention
import os
import pwd
import subprocess
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pytest
from pipecat.frames.frames import ErrorFrame, Frame, InterruptionFrame, UserStartedSpeakingFrame, UserStoppedSpeakingFrame

from conftest import RUN, events, located, new_sessions, running, unread
from hands.brain import pinned
from hands.brain.fork import REMINDER
from hands.brain.stage import ANSWERS, OWED, BrainStage
from hands.core.brainwork import Idle, Running, Stopping, Working
from hands.core.wire import Sent, tool_answers
from hands.brain.process import BROKEN, STOPPED, starting_settings, workdir
from hands.core.effects import Deny
from hands.core.session import DenyAndInterrupt
from hands.daemon.run import mind
from hands.sessions.audit import Entry
from hands.sessions.wide import WideEvent
from hands.sessions.home import Home
from hands.sessions.proxy import Wire, serve_proxy
from hands.voice import voices
from hands.voice.backends import Account, ClaudeCodeBackend
from hands.voice.conversation import UserWords
from hands.voice.pipeline import VoiceConfig
from hands.voice.refocus import Refocus
from hands.sessions.sentences import Sentences
from hands.sessions.child import Ran
from hands.sessions.tmux import ran_at
from hands.voice.sentences import SummaryStore
from hands.voice.tools import stay_silent_tool
from test_stage import Received, Spoken

# A turn of the real Claude Code against the mock takes a few seconds; a minute is a hang.
PATIENCE_SECONDS = 60.0
ACCOUNT = Account("claude.ai", "user@example.com")
# What only the brain's own turns carry, among every request its Claude Code makes: hands' tools offered to the model.
# A title or a summary Claude Code asks for beside a turn carries the user's words too, but none of hands' tools.
TURN = "mcp__hands__stay_silent"
# What only a request carrying a tool's result holds, as Claude Code's compact JSON writes the block. The same words in
# a prompt's text are escaped inside a JSON string, so they never match.
RESULT = '"type":"tool_result"'
# The words a side question's prompt opens with, past the reminder's tag: nothing in them is escaped in a JSON string.
SIDE_QUESTION = REMINDER.removeprefix("<system-reminder>").split(".")[0]


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
        api = MockApi(f"http://127.0.0.1:{ready['port']}", Path(ready["ca"]), scenarios)
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


# The brain's keybindings, as docs/claude-code-screen.md's "Submitting the prompt with a chord no dialog binds" settles
# them: Ctrl-Q submits the prompt, and Return is unbound from submitting it, without which the prompt never takes Ctrl-Q as
# a submit. hands writes no keybindings yet, so only the tests that prove the chord give the brain these.
BRAIN_KEYS = {"bindings": [{"context": "Chat", "bindings": {"ctrl+q": "chat:submit", "enter": None}}]}
# BRAIN_KEYS' submit chord, as tmux names the key.
SUBMIT = "C-q"


def logged_in(brain: Path, mock_binary: Path, keys: Mapping[str, object] | None) -> None:
    """A brain directory logged in as the mock's account by the mock's own login, the directory it works in trusted, and
    its keybindings `keys`, or Claude Code's defaults as hands leaves them when None."""
    # [LAW:one-source-of-truth] what a login the mock accepts looks like is the mock's to say, never a copy here.
    subprocess.run([str(mock_binary), "login", str(brain), "--trust", str(workdir(brain).resolve())], check=True)
    # [LAW:one-source-of-truth] the settings a brain home starts with are the ones hands writes for every brain.
    assert starting_settings(brain)
    # The commands the tests script, allowed as a brain's own settings allow a command, so no permission waits on a person:
    # how hands puts one to the user is test_permissions'.
    settings = brain / "settings.json"
    said = json.loads(settings.read_text())
    allowed = [*said["permissions"].get("allow", []), "Bash(touch:*)", "Bash(sleep:*)", "Bash(nc:*)", "Bash(cat:*)"]
    settings.write_text(json.dumps({**said, "permissions": {**said["permissions"], "allow": allowed}}))
    if keys is not None:
        (brain / "keybindings.json").write_text(json.dumps(keys))


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
            raise AssertionError(f"waited {PATIENCE_SECONDS:g}s: hands said {self.out.said()!r}, the brain's work {self.stage.work()!r}, after {len(self.sent)} requests through hands' proxy") from None

    async def say(self, words: str) -> None:
        """The user's words, spoken once they have heard all hands handed the speaker, its marks played: behind the cut
        their turn opens with and the turn's stop, as every turn that carries words is cut (test_stage's `cut_in`). A
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


async def connected(hands: Hands) -> None:
    """Waits until the brain has listed hands' tools over MCP. Claude Code connects to hands' server after it starts, not
    before, so a turn typed sooner goes to the model without them, and whether it does is a race the test must not have."""
    await hands.until(lambda: any(isinstance(entry, WideEvent) and entry.event == "mcp.request" and entry.facts.get("method") == "tools/list" for entry in hands.recorded))


def gate(hands: Hands) -> Path:
    """A named pipe in the brain's workdir: a command the brain runs that reads it runs until the test opens it, so what
    the test says happens while the command runs happens then, however slow the run is."""
    pipe = hands.workdir / "go"
    os.mkfifo(pipe)
    return pipe


def opened(pipe: Path) -> bool:
    """Whether the gate opened: written and closed once a command has it open to read. It never blocks, so a wait on it is
    `Hands.until`'s, and a command that never came to read it fails the test there with what hands saw, not as a hang."""
    try:
        end = os.open(pipe, os.O_WRONLY | os.O_NONBLOCK)
    except OSError as error:
        # ENXIO is the one answer that means nothing reads it yet; every other error is the test's own fault, and raised.
        if error.errno != errno.ENXIO:
            raise
        return False
    with open(end, "w") as written:
        written.write("go")
    return True


@pytest.fixture
def keys() -> Mapping[str, object] | None:
    """The brain's keybindings.json: none, as hands starts the brain. A test that needs others parametrizes `keys`."""
    return None


@pytest.fixture
async def hands(tmp_path: Path, mock_api: MockApi, mock_binary: Path, claude_code: Path, monkeypatch: pytest.MonkeyPatch, keys: Mapping[str, object] | None) -> AsyncGenerator[Hands]:
    brain = tmp_path / "brain"
    logged_in(brain, mock_binary, keys)
    # Every start of a claude, the brain's, an aside's, a login's, runs the cached Claude Code, the pin itself, held
    # offline as the run is (conftest): a connection it or anything it starts makes around the mock is refused by the OS,
    # and one through its proxy is in the mock's log.
    monkeypatch.setattr(pinned, "claude", located(claude_code))
    recorded: list[Entry] = []
    sent: list[Sent] = []
    wire = Wire(lambda observed: sent.append(observed) if isinstance(observed, Sent) else None)
    # [LAW:effects-at-boundaries] the change from the run's own wiring, beside the pinned claude: the upstream hands' proxy
    # forwards to.
    proxy = await serve_proxy(mock_api.url, wire.observe, wire.route, clock=time.time)
    home = tmp_path / "home"
    home.mkdir()
    # Every other host Claude Code calls is reached through the mock, and nothing reaches past it.
    # Claude Code clones its official plugin marketplace from GitHub at launch, which the mock does not answer (refused, and the
    # e2e fails on it); the marketplace is no part of what these tests prove, so the switch keeps the clone out of the run.
    environment = os.environ | {"HOME": str(home), "HTTPS_PROXY": mock_api.url, "HTTP_PROXY": mock_api.url, "NODE_EXTRA_CA_CERTS": str(mock_api.ca), "NO_PROXY": "127.0.0.1,localhost", "CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL": "1"}
    refocus = Refocus(new_sessions(record=recorded.append), Home(tmp_path / "hands"), recorded.append)
    config = VoiceConfig(llm=ClaudeCodeBackend(model="claude-sonnet-5", config_dir=brain, account=ACCOUNT), voice=voices.DEFAULT)
    out = Spoken()
    store = SummaryStore(Sentences(tmp_path / "sentences.db"))
    try:
        async with mind(config, [stay_silent_tool()], [], lambda: "", unread, lambda: "screen", lambda: "held key", refocus, proxy.url, wire, store, tmp_path / "audit", "hands", recorded.append, environment, out.cut_off, Attention) as minded:
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
    """The brain hands started is held by the run's sandbox: a command it runs that connects past loopback is refused by
    the OS before a packet leaves, so a way around the mock never reaches the machine's network."""
    # 192.0.2.1 is TEST-NET-1 (RFC 5737), routed nowhere: outside the sandbox the attempt times out, inside it is refused.
    mock_api.script("1-tried", {"bodyContains": [TURN, RESULT]}, {"type": "text", "text": "Tried."})
    mock_api.script("2-reach", {"lastUserContains": ["reach past loopback"], "bodyContains": [TURN]}, {"type": "tool_use", "name": "Bash", "input": {"command": "nc -zv -w 5 192.0.2.1 443"}}, stop="tool_use")
    await connected(hands)
    await hands.say("reach past loopback")
    await hands.until(lambda: "Tried." in "".join(hands.out.said()))
    # nc's own words, in the result of its call: Claude Code's prompt says "Operation not permitted" too.
    results = [answer.text for request in hands.sent for answer in tool_answers(request.body)]
    assert any("192.0.2.1" in result and "Operation not permitted" in result for result in results), results



# Two addresses no setting of the brain's allows, so each WebFetch of one is asked about in default mode. WebFetch is
# concurrency-safe, so two in one reply run their permission checks at once and Claude Code stacks their dialogs, the
# later beneath the earlier (cmt-36bc6251, read from 2.1.308, proved here).
ADDRESSES = ("https://example.com/first", "https://example.org/second")


def fetch(url: str) -> dict[str, object]:
    return {"type": "tool_use", "name": "WebFetch", "input": {"url": url, "prompt": "what does it say?"}}


def permissions(recorded: list[Entry]) -> list[WideEvent]:
    return [entry for entry in recorded if isinstance(entry, WideEvent) and entry.event == "brain.permission"]


def overlap(one: WideEvent, other: WideEvent) -> bool:
    """Whether each began before the other ended: held at once."""
    ends = [event.started_at + timedelta(milliseconds=event.duration_ms) for event in (one, other)]
    return one.started_at < ends[1] and other.started_at < ends[0]


def shown(hands: Hands) -> str:
    """The last lines the brain's terminal showed."""
    return hands.stage._brain._claude.shown()  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]


def held(hands: Hands) -> tuple[object, ...]:
    """The dialogs the brain's model holds now: what Claude Code has stacked, the first on its screen."""
    session = hands.stage._brain._modelled()  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]
    return () if session is None else session.dialogs  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]


async def at_the_pane(hands: Hands, *command: str) -> None:
    """A tmux command run against the brain's pane, as a terminal attached to it types: tmux writes the keys to the pane,
    and the pane's fritter forwards them to Claude Code byte for byte."""
    claude = hands.stage._brain._claude  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]
    ran = await ran_at(claude._tmux, claude._socket, command)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    assert isinstance(ran, Ran) and ran.returncode == 0, ran


async def keyed(hands: Hands, *keys: str) -> None:
    """`keys`, as tmux's send-keys names them, typed into the brain's pane."""
    claude = hands.stage._brain._claude  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]
    await at_the_pane(hands, "send-keys", "-t", claude._session, *keys)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]


async def pasted(hands: Hands, text: str) -> None:
    """`text` pasted into the brain's pane in one bracketed paste, its newlines kept, as hands pastes a turn."""
    claude = hands.stage._brain._claude  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]
    await at_the_pane(hands, "set-buffer", "-b", "pasted", text, ";", "paste-buffer", "-p", "-r", "-d", "-b", "pasted", "-t", claude._session)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]


async def test_two_fetches_the_brain_asks_about_at_once_are_held_together_and_each_put_to_the_user_in_the_order_asked(mock_api: MockApi, hands: Hands) -> None:
    """Both hooks are open at once and both dialogs held in the brain's model; each is put to the user in turn, and each
    refusal goes to the dialog it answers."""
    mock_api.script("1-unfetched", {"bodyContains": [TURN, RESULT]}, {"type": "text", "text": "I left them unfetched."})
    mock_api.script("2-fetch", {"lastUserContains": ["fetch both pages"], "bodyContains": [TURN]}, *(fetch(url) for url in ADDRESSES), stop="tool_use")
    await connected(hands)
    await hands.say("fetch both pages")
    await hands.until(lambda: len(held(hands)) == 2)
    # The dialog on the brain's screen is the first held, the one asked first. It is drawn below whatever lists the calls,
    # so the screen's last mention of either domain is the dialog's.
    await hands.until(lambda: (screen := shown(hands)).rfind("example.com") > screen.rfind("example.org"))
    for url in ADDRESSES:
        await hands.until(lambda: f"May I use WebFetch on {url}? Say yes to allow it." in hands.out.said())
        await hands.say("no")
    await hands.until(lambda: "I left them unfetched." in "".join(hands.out.said()))
    one, other = permissions(hands.recorded)
    assert overlap(one, other)
    assert [(event.facts["tool"], type(event.facts["decision"])) for event in (one, other)] == [("WebFetch", Deny)] * 2


@pytest.mark.parametrize("fetches", [1, 2], ids=["one dialog on screen", "one on screen and one stacked beneath it"])
async def test_a_stop_pressed_while_the_brain_holds_dialogs_refuses_each_with_interrupt_and_ends_the_turn(mock_api: MockApi, hands: Hands, fetches: int) -> None:
    """Every refusal of a stopped turn ends it, so no request carrying one reaches the mock: none is scripted, and the
    mock's fault check fails the test on any request it has no answer for. The turn after it is answered."""
    mock_api.script("1-fetch", {"lastUserContains": ["fetch the pages"], "bodyContains": [TURN]}, *(fetch(url) for url in ADDRESSES[:fetches]), stop="tool_use")
    mock_api.script("2-hello", {"lastUserContains": ["say hello"], "bodyContains": [TURN]}, {"type": "text", "text": "Hello."})
    await connected(hands)
    await hands.say("fetch the pages")
    await hands.until(lambda: len(held(hands)) == fetches)
    work = hands.stage.work()
    assert isinstance(work, Working)
    assert hands.stage.stop(work.turn) == Stopping()
    await hands.until(lambda: hands.stage.work() == Idle())
    await hands.until(lambda: len(permissions(hands.recorded)) == fetches)
    assert [event.facts["decision"] for event in permissions(hands.recorded)] == [DenyAndInterrupt(STOPPED)] * fetches
    assert held(hands) == ()
    await hands.say("say hello")
    await hands.until(lambda: "Hello." in "".join(hands.out.said()))


async def test_a_turn_whose_asker_breaks_while_its_dialog_is_held_is_refused_with_interrupt_and_ends(mock_api: MockApi, hands: Hands, monkeypatch: pytest.MonkeyPatch) -> None:
    """A refusal of a turn the brain broke ends it as a stop's does, so no request carrying it reaches the mock."""
    mock_api.script("1-fetch", {"lastUserContains": ["fetch the page"], "bodyContains": [TURN]}, fetch(ADDRESSES[0]), stop="tool_use")
    mock_api.script("2-hello", {"lastUserContains": ["say hello"], "bodyContains": [TURN]}, {"type": "text", "text": "Hello."})

    def broken(*_: object) -> None:
        raise RuntimeError("no voice")

    # What the brain calls to put a dialog to the user, as the stage hands it to each turn. Put back by itself: the
    # fixture's pin of the cached claude is on the same monkeypatch.
    put = hands.stage._put  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(hands.stage, "_put", broken)
    await connected(hands)
    await hands.say("fetch the page")
    # The turn's failure is the run's error, read here rather than failing the waits.
    async with asyncio.timeout(PATIENCE_SECONDS):
        while not (hands.errors and permissions(hands.recorded)):
            await asyncio.sleep(0.05)
    [permission] = permissions(hands.recorded)
    assert permission.facts["decision"] == DenyAndInterrupt(BROKEN)
    assert any("no voice" in str(error.error) for error in hands.errors)
    hands.errors.clear()
    monkeypatch.setattr(hands.stage, "_put", put)
    await hands.until(lambda: hands.stage.work() == Idle())
    assert held(hands) == ()
    await hands.say("say hello")
    await hands.until(lambda: "Hello." in "".join(hands.out.said()))


@pytest.mark.parametrize("keys", [BRAIN_KEYS])
@pytest.mark.parametrize("before", [[], [("Tab",), ("-l", "use a copy instead")]], ids=["Yes focused", "Yes amended, its field holding words"])
async def test_the_brain_s_submit_chord_typed_into_a_held_permission_dialog_answers_nothing(mock_api: MockApi, hands: Hands, before: list[tuple[str, ...]]) -> None:
    """Ctrl-Q, bound to the prompt's submit, typed while a permission dialog is on the brain's screen: the dialog stays up
    and the tool does not run, whether Yes or Yes's own text field has the focus. A Down typed after it moves the focus,
    so once the screen shows it moved, Claude Code has read everything typed before it. That the chord arrives by this
    path at all is the paste test's to show: here a chord lost on the way looks the same (promptctl-brain-z04s.ar0)."""
    made = hands.workdir / "made"
    mock_api.script("1-left", {"bodyContains": [TURN, RESULT]}, {"type": "text", "text": "I left it unmade."})
    mock_api.script("2-link", {"lastUserContains": ["link the file"], "bodyContains": [TURN]}, {"type": "tool_use", "name": "Bash", "input": {"command": f"ln -s target {made}"}}, stop="tool_use")
    await connected(hands)
    await hands.say("link the file")
    await hands.until(lambda: len(held(hands)) == 1 and "Do you want to proceed?" in shown(hands))
    for typed in before:
        await keyed(hands, *typed)
    await keyed(hands, SUBMIT)
    await keyed(hands, "Down")
    # The second option, whatever it says: a Bash dialog in default mode offers "Yes, and don't ask again" there.
    await hands.until(lambda: "❯ 2." in shown(hands))
    assert "Do you want to proceed?" in shown(hands)
    assert len(held(hands)) == 1
    assert not made.is_symlink()
    await hands.until(lambda: any("Say yes to allow it." in said for said in hands.out.said()))
    await hands.say("no")
    await hands.until(lambda: "I left it unmade." in "".join(hands.out.said()))
    assert not made.is_symlink()
    [permission] = permissions(hands.recorded)
    assert isinstance(permission.facts["decision"], Deny)


@pytest.mark.parametrize("keys", [BRAIN_KEYS])
async def test_words_pasted_at_the_brain_s_prompt_are_sent_by_its_submit_chord(mock_api: MockApi, hands: Hands) -> None:
    """A paste long enough that Claude Code folds it into a placeholder, then Ctrl-Q: the whole text is sent as a turn,
    which hands speaks as any turn Claude Code opens (promptctl-brain-z04s.ar0)."""
    words = "\n".join(f"line {line} of a long turn said to the brain" for line in range(1, 30)) + "\nand last, say you read it"
    mock_api.script("1-read", {"lastUserContains": ["line 1 of a long turn", "line 29 of a long turn", "and last, say you read it"], "bodyContains": [TURN]}, {"type": "text", "text": "I read it."})
    await connected(hands)
    await pasted(hands, words)
    await hands.until(lambda: "[Pasted text #1 +29 lines]" in shown(hands))
    await keyed(hands, SUBMIT)
    await hands.until(lambda: "I read it." in "".join(hands.out.said()) and hands.stage.work() == Idle())


async def test_words_said_while_the_brain_works_in_silence_are_answered_at_once_and_are_the_next_turn(mock_api: MockApi, hands: Hands) -> None:
    """docs/barge-in.md's S2, against the real brain: words said while a call runs and everything said has played are not
    a barge-in. hands answers them within about a second, the turn runs on to its end heard, and the words are the next
    turn, with no note (promptctl-barge-in-duao.51y)."""
    pipe = gate(hands)
    # Tried in name order: the second turn, whose history holds the first's result too, ahead of the first's end.
    mock_api.script("1-logs", {"lastUserContains": ["check the logs too"], "bodyContains": [TURN]}, {"type": "text", "text": "The logs are clean."})
    mock_api.script("2-done", {"bodyContains": [TURN, RESULT]}, {"type": "text", "text": "Renamed it."})
    mock_api.script("3-look", {"lastUserContains": ["rename the module"], "bodyContains": [TURN]}, {"type": "text", "text": "Let me look."}, {"type": "tool_use", "name": "Bash", "input": {"command": f"cat {pipe}"}}, stop="tool_use")
    await connected(hands)
    await hands.say("rename the module")
    await hands.until(lambda: isinstance(work := hands.stage.work(), Working) and [call.tool for call in work.calls] == ["Bash"])
    await hands.say("and check the logs too")
    # hands' answer is said while the call still runs.
    await hands.until(lambda: any(answer in "".join(hands.out.said()) for answer in ANSWERS))
    assert "Renamed it." not in "".join(hands.out.said())
    await hands.until(lambda: opened(pipe))
    await hands.until(lambda: "The logs are clean." in "".join(hands.out.said()) and len(events(hands.recorded, "voice.turn")) == 2)
    said = "".join(hands.out.said())
    first, second = events(hands.recorded, "voice.turn")
    match first.facts["asides"]:
        case ((str() as answer, int() | float() as waited_ms),):
            pass
        case asides:
            raise AssertionError(f"the first turn's asides are not one answer: {asides!r}")
    # C1: not a barge-in. Neither turn is interrupted, the first is cut in on once, and nothing of it is spoken over.
    assert (first.facts["interrupted"], first.counts["cut_in"], first.facts["spoken_over"], second.facts["interrupted"]) == (False, 1, "", False)
    # C2: nothing is lost: the turn runs on to its end, heard, its sentence before the call, then the answer, then the rest.
    assert answer in ANSWERS
    assert said.index("Let me look.") < said.index(answer) < said.index("Renamed it.") < said.index("The logs are clean."), said
    # C3: the next turn's asked is the words, with no note of the turn before; only the screen's, which every turn carries.
    assert second.facts["asked"] == "and check the logs too\n\n[hands] The user can see a screen."
    # C5: the answer within about a second of the words reaching hands, and the next turn's waits recorded.
    assert waited_ms < 1000, first.facts["asides"]
    assert isinstance(second.facts["queued_ms"], int | float) and isinstance(second.facts["waited_ms"], int | float), second.facts


async def test_a_background_command_s_turn_ends_while_it_runs_and_its_end_opens_a_turn_hands_speaks(mock_api: MockApi, hands: Hands) -> None:
    """A command the brain starts in the background returns at once, so its turn ends while the command still runs: a turn
    held for a background call ran its whole length (promptctl-latency-3fui.h2r). The command's end is news Claude Code
    opens a turn of its own for, and hands speaks that turn and ends it."""
    pipe = gate(hands)
    # Tried in name order: the notification's turn, whose history holds the launch's result too, ahead of the launch's end.
    # The launch is answered only once Claude Code says it backgrounded the command; a launch it refused or ran in the
    # foreground has no answer, and the mock's fault check fails the test on it.
    mock_api.script("1-finished", {"lastUserContains": ["<task-notification>"], "bodyContains": [TURN]}, {"type": "text", "text": "It finished."})
    mock_api.script("2-started", {"bodyContains": [TURN, "Command running in background with ID"]}, {"type": "text", "text": "Started it."})
    mock_api.script("3-background", {"lastUserContains": ["start it in the background"], "bodyContains": [TURN]}, {"type": "tool_use", "name": "Bash", "input": {"command": f"cat {pipe}", "run_in_background": True}}, stop="tool_use")
    await connected(hands)
    await hands.say("start it in the background")
    # The asked turn is over, spoken and recorded, while the command it started still runs.
    await hands.until(lambda: "Started it." in "".join(hands.out.said()) and hands.stage.work() == Idle() and len(events(hands.recorded, "voice.turn")) == 1)
    assert "It finished." not in "".join(hands.out.said())
    await hands.until(lambda: opened(pipe))
    await hands.until(lambda: "It finished." in "".join(hands.out.said()) and hands.stage.work() == Idle() and len(events(hands.recorded, "voice.turn")) == 2)
    asked, told = events(hands.recorded, "voice.turn")
    assert (asked.facts["origin"], asked.facts["opener"], told.facts["origin"], told.facts["opener"]) == ("awaited", "typed", "unawaited", "notified")
