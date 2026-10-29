"""The brain's process: one long-lived, slim Claude Code on its own login, behind hands' proxy, reaching hands over MCP.

It is Claude Code as anyone runs it: interactive, on a terminal hands holds, under fritter, and asked nothing a person
at its keyboard could not ask. A turn is typed into its input and sent with Return, and Escape stops a prompt that is no
turn's, or a turn hands' own work broke. Nothing else is typed into it: its input is the user's, and what hands asks in
the background is a /btw forked from its newest main turn through the proxy (`hands.brain.asides`). What it says is read from the wire, not from its screen [the design's rule: primary facts
from the wire, derivative ones from the harness], and the harness is heard only through the hooks it posts to a
listener of hands' own: that a typed turn was taken, and that it ended, or that the API failed it.

Its login, settings, and skills live in a directory hands owns, set up once by `hands login`, which runs Claude Code's
own first run there, as any Claude Code is set up, on any login Claude Code takes, and it runs in that directory's empty cwd, never in a project. What it may use, what it may do without asking,
and which MCP servers it has are that directory's to say, as they are for any Claude Code: its settings.json and its
.claude.json. hands adds only its own server, its hooks, and the skills it ships for the brain's own jobs, and keeps out
what the login brings from the account.
"""

import asyncio
import contextlib
import glob
import json
import re
import shutil
import subprocess
import tempfile
import time
from uuid import uuid4
from collections.abc import Awaitable, Callable, Coroutine, Generator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, cast, get_args

from aiohttp import web
from loguru import logger

from hands.brain import pinned
from hands.brain.mcp import SERVER_NAME
from hands.core.effects import Allow, AllowWith, Approve, Deny, Heard, HookReply, SessionGone, Story, Submitted, Summarise, Text, Withdraw
from hands.core.events import Ending, Event, Prompted, Stopped
from hands.core.reducer import UNTOLD
from hands.core.session import DenyAndInterrupt, Permission, PromptId, PromptText, RequestId, Session, SessionId, all_told, holds, ids, pasted
from hands.core.wire import MainTurn, Observed, Sent, tool_names
from hands.core.trace import Span
from hands.core.stopfailure import StopFailed
from hands.sessions.audit import Record
from hands.sessions.registry import Sessions, StopHeld
from hands.sessions.reader import READ_SECONDS, Reader
from hands.sessions.tail import Tails
from hands.sessions.hookconfig import BRAIN_REPLY_BY_SECONDS, TAKE_SECONDS, brain_declared
from hands.sessions.hooks import Hook, PermissionHook, hook_output, parse_own_hook
from hands.sessions.payload import Payload, Rejected
from hands.sessions.pseudoterminal import ClaudeCode, on_terminal
from hands.sessions.typing import Typist, Untyped
from hands.sessions.untap import untapped
from hands.sessions.wide import Begun, annotate, begun, continuing, ended, fail, here, unit, within
from hands.sessions.wrapper import EVERY_RUN, Unpackaged, packaged
from hands.voice.backends import Account

# What --bare would have switched off, switched off one by one so the OAuth login stays on (hands-wire-6ic.8wu).
# LSP needs no switch: it comes only from plugins, the brain's own setup installs none, and hands' own declares none.
SLIM = {
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
    "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "CLAUDE_CODE_ENABLE_PROMPT_SUGGESTION": "false",
    # A scheduled prompt opens a turn of its own the same way when it fires (hands-brain-d8g.9d6 review).
    "CLAUDE_CODE_DISABLE_CRON": "1",
    # The account's claude.ai connectors are Brandon's, never the brain's: loaded, they joined every request after the
    # first (turn 2 grew from 8.7 KB to 112 KB; hands-wire-6ic.eph). The skills and plugins the account syncs
    # are turned off in the brain's settings.json, the only place Claude Code reads that switch from.
    "ENABLE_CLAUDEAI_MCP_SERVERS": "false",
}

# The tool Claude Code offers in place of the MCP tools it defers.
TOOL_SEARCH = "ToolSearch"

# The skills hands gives the brain, as a plugin of its own: shipped with the code that runs the jobs they are for, beside
# the skills of the brain's own setup and never in it.
PLUGIN = Path(__file__).parent / "plugin"
# Each named as the brain calls it, plugin:skill, so it may use them without asking, as it may hands' tools.
PLUGIN_SKILLS = tuple(
    f"Skill({json.loads((PLUGIN / '.claude-plugin' / 'plugin.json').read_text())['name']}:{skill.parent.name})" for skill in sorted((PLUGIN / "skills").glob("*/SKILL.md"))
)

# What Claude Code prefers to its own login: credentials, and the switches that send it to a provider other than
# Anthropic's API. The brain's login is its config directory's, whichever Claude Code takes, a key in its
# settings.json among them; inherited from hands' environment, any of these would put the brain on another account or
# provider without a word, so none is passed on.
FOREIGN_LOGINS = (
    "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN",
    *(f"CLAUDE_CODE_USE_{provider}" for provider in ("BEDROCK", "VERTEX", "FOUNDRY", "ANTHROPIC_AWS", "ANTHROPIC_GOOGLE_CLOUD", "MANTLE", "GATEWAY")),
)

# A dialog nobody can see would hold its turn open forever with no hook to say so. A permission its setup asks
# about is held at its hook while the user is asked by voice, in a turn of theirs; anything else is answered no there,
# and the brain hears why.
NOBODY = "Nobody can be asked: no turn of the user's is in flight to ask in, so what this Claude Code's settings would ask about is refused."
UNREAD = "hands could not read this permission request, so it is refused."
BROKEN = "hands failed while it settled this permission request, so it is refused."
UNVOICED = "Nobody sees this dialog: ask the user in your reply instead, and they will answer in their next turn."
UNANSWERED = "No answer came from the user in the day its hook is held, so it did not run. Tell them aloud that it did not run and what that leaves undone."
SPOKEN_OVER = "The user spoke over this turn before they could be asked, so it did not run."
STOPPED = "hands stopped this turn, so it did not run."
LET_GO = "Claude Code went on past this dialog before anyone answered it, so it is refused."
DECLINED: Mapping[str, object] = {"hookSpecificOutput": {"hookEventName": "Elicitation", "action": "decline"}}

# What the brain posts to hands, each to its own path: a typed turn taken, a turn ended, a turn the API failed, and a
# dialog about to open. Escape ends a turn with none of them, so a turn told to stop is over when
# it is told.
HOOKS = ("UserPromptSubmit", "Stop", "StopFailure", "PermissionRequest", "Elicitation")
# A paste of four lines or more, as a UserPromptSubmit hook's `prompt` holds it: what was pasted, with a line end added
# where it had none, inside a tag pair.
PASTE = re.compile(r'\n\n<pasted_content id="(\w+)">\n(.*)</pasted_content id="\1">\n', re.DOTALL)

# `claude auth status` answers in about a second; one that has not answered in this long is not going to.
AUTH_STATUS_SECONDS = 20.0
# How long fritter has to start the brain and open its socket, and the brain to turn its input on.
START_SECONDS = 10.0
# How far apart two Ctrl-Cs are pressed into the brain: Claude Code exits on a second within 800ms of one that found its
# input empty.
EXIT_SECONDS = 1.0
# How long Claude Code has to set its session idle once hands pressed Escape: it does in milliseconds, and the rest is
# room for a loaded machine.
ESCAPE_SECONDS = 4.0
# How long Claude Code has to interrupt a turn hands pressed Escape for: its idle, then a status read to hear it, then the
# model's wait past an idle for the record of how the turn ended (UNTOLD). [LAW:one-source-of-truth] the model's two
# bounds are the model's, so a change to either moves this one.
INTERRUPT_SECONDS = ESCAPE_SECONDS + READ_SECONDS + UNTOLD / 1000


def kept(prompt: str) -> str:
    """A prompt as Claude Code keeps what was typed: out of a paste's tags, and without the whitespace Claude Code trims
    from what it takes: trailing from every prompt, leading too from one it queued (the brain's transcript 15192e41, 2026-10-19 05:29:11)."""
    paste = PASTE.fullmatch(prompt)
    return (prompt if paste is None else paste[2]).strip()


@dataclass(frozen=True)
class Station:
    """What every slim Claude Code of hands' own runs on."""

    config_dir: Path
    cwd: Path
    model: str
    proxy_url: str
    # hands' own environment, which each is started in less what environment() keeps out; kept out of the repr, as it
    # holds keys, and out of the station's identity, which it is not part of.
    inherited: Mapping[str, str] = field(repr=False, compare=False)


@dataclass(frozen=True)
class Launch:
    """Everything the brain is started with."""

    station: Station
    # The account its config directory is logged in as, read as the run started.
    account: Account
    instruction: str
    mcp_config: str
    conversation: "Conversation"

    @property
    def session(self) -> SessionId:
        return self.conversation.session


@dataclass(frozen=True)
class Fresh:
    """A conversation begun new, under a session chosen by hands; `untranscribed` is the one held before it, when there
    was one, which Claude Code has no transcript of: it never had a turn (Claude Code writes none until one), or its
    transcript was cleaned away."""

    # [LAW:one-source-of-truth] chosen by hands, so the brain's requests are known as its own from the first one on the
    # wire, and its hooks from the first one posted.
    session: SessionId
    untranscribed: SessionId | None


@dataclass(frozen=True)
class Resumed:
    """The conversation the brain held when hands last ran, taken up again under its own session: a restart, an upgrade,
    or a change of model goes on from what was said, compacted as Claude Code compacts any conversation."""

    session: SessionId


Conversation = Fresh | Resumed


def held(config_dir: Path) -> Path:
    """The session of the brain's conversation, written by hands alone as each brain on `config_dir` starts."""
    return config_dir / "conversation"


def conversation(config_dir: Path, new: SessionId) -> Conversation:
    """The conversation a brain on `config_dir` starts in: the one held, while Claude Code still has its transcript, or
    else one begun under `new`."""
    session = _holding(config_dir)
    if session is None:
        return Fresh(new, None)
    # [LAW:one-source-of-truth] Claude Code's own transcript says whether there is a conversation to resume: named by
    # its session, under any project directory, as Claude Code finds the session it resumes.
    if glob.glob(glob.escape(str(config_dir / "projects")) + f"/*/{glob.escape(session)}.jsonl"):
        return Resumed(session)
    return Fresh(new, session)


def hold(config_dir: Path, session: SessionId) -> None:
    """Remember `session` as the brain's conversation, replacing whatever was held whole."""
    path = held(config_dir)
    written = path.with_name(f"{path.name}.writing")
    written.write_text(f"{session}\n")
    written.replace(path)


def let_go(config_dir: Path) -> SessionId | None:
    """Forget the brain's conversation, so the next brain on `config_dir` begins a new one: the session that was held,
    when one was."""
    session = _holding(config_dir)
    held(config_dir).unlink(missing_ok=True)
    return session


def _holding(config_dir: Path) -> SessionId | None:
    try:
        return SessionId(held(config_dir).read_text().strip())
    except FileNotFoundError:
        return None


def slim(claude: Path, model: str, conversation: Conversation) -> list[str]:
    """A slim Claude Code's command line: `claude`, the brain's pinned release, interactive, reading no settings but its config directory's."""
    return [
        str(claude),
        "--model", model,
        *_conversing(conversation),
        # The config directory's settings.json is the only settings file read: none from the working directory.
        "--setting-sources", "user",
    ]


def _conversing(conversation: Conversation) -> tuple[str, str]:
    match conversation:
        case Fresh(session=session):
            return ("--session-id", session)
        case Resumed(session=session):
            # Claude Code keeps the session it resumes unless told to fork it (--fork-session).
            return ("--resume", session)


# The permission modes Claude Code takes as a setting's defaultMode, and the other name it reads as one of them.
type Mode = Literal["acceptEdits", "auto", "bypassPermissions", "default", "dontAsk", "plan"]
MODES: dict[str, Mode] = {**{mode: cast(Mode, mode) for mode in get_args(Mode.__value__)}, "manual": "default"}


def command(launch: Launch, claude: Path, hooks: str, mode: Mode | None) -> list[str]:
    """The brain's command line: a slim Claude Code on its own setup, plus hands' server, instruction, skills, and hooks posted to the listener at `hooks`;
    `mode` is the permission mode the brain's own settings name, if they name one."""
    return [
        *slim(claude, launch.station.model, launch.conversation),
        # The instruction as this hands writes it on every request, a resumed conversation's too: Claude Code otherwise
        # sends the system prompt its conversation first had until it is compacted, even for a resume launched with a new
        # --append-system-prompt. Rendered fresh, it is byte-identical request to request.
        "--system-prompt-snapshot", "off",
        # Beside the MCP servers its own setup names, never in place of them.
        "--mcp-config", launch.mcp_config,
        # After Claude Code's own system prompt, never in place of it: the API checks that it opens as Claude Code's does.
        "--append-system-prompt", launch.instruction,
        # hands' tools are how the brain reaches the sessions at all, and its skills how it does its jobs there; a deny rule
        # in its own setup still outranks this.
        "--allowedTools", f"mcp__{SERVER_NAME}", *PLUGIN_SKILLS,
        "--plugin-dir", str(PLUGIN),
        "--settings", json.dumps({
            # [LAW:single-enforcer] each hook declared as hookconfig declares the brain's: a held permission's lives as long
            # as a working session's, a day, and is denied only as that day ends, since its dialog is one nobody sees; a
            # prompt's and a Stop's as long as the order of the brain's turns rests on them (`ClaudeTurns`).
            "hooks": {event: [{"hooks": [{"type": "http", "url": f"{hooks}/{event}", **brain_declared(event)}]}] for event in HOOKS},
            # Tool search defers MCP tools behind ToolSearch, so hands' tools would reach the model only after a round trip
            # of their own on a spoken turn. Set here, it outranks the brain's settings.json env and hands' environment,
            # which a process environment does not (hands-brain-8gb).
            "env": {"ENABLE_TOOL_SEARCH": "false"},
            # The mode the brain's own settings name, said here too, so it is the same mode: Claude Code offers to make auto
            # mode the default to a session whose mode its user settings alone name, once it is idle, and the first turn
            # typed into the offer answers Yes (promptctl-brain-lq3.mcw).
            **({"permissions": {"defaultMode": mode}} if mode is not None else {}),
        }),
    ]


def settings_mode(config_dir: Path) -> Mode | None:
    """The permission mode the brain's settings.json names, or None when it names none."""
    settings = config_dir / "settings.json"
    try:
        said = Payload.parse(settings.read_bytes())
        named = Payload.of(said.fields.get("permissions", {}), "its permissions").optional_text("defaultMode")
    except FileNotFoundError:
        return None
    except (OSError, Rejected) as error:
        raise Unstartable(f"the brain's settings could not be read ({error}): {settings}") from None
    if named is None:
        return None
    # [LAW:parse-dont-validate] said again in the brain's --settings, beside its hooks: a mode Claude Code does not take
    # would put the settings holding them in doubt, so the brain does not start on one.
    if (mode := MODES.get(named)) is None:
        raise Unstartable(f'the brain\'s settings name a defaultMode Claude Code does not take, "{named}": name one of {", ".join(MODES)} in {settings}')
    return mode


def environment(config_dir: Path, base_url: str, inherited: Mapping[str, str]) -> dict[str, str]:
    """A slim Claude Code's environment: hands' own, less anything that would stand in for the login in `config_dir`, reaching the API at `base_url`."""
    # [LAW:one-source-of-truth] the brain is the same brain wherever hands was started: a daemon started inside a tapped
    # session does not hand the brain that session's tap.
    kept = {name: value for name, value in untapped(inherited).items() if name not in FOREIGN_LOGINS}
    return {**kept, **SLIM, **EVERY_RUN, "CLAUDE_CONFIG_DIR": str(config_dir), "ANTHROPIC_BASE_URL": base_url}


def workdir(config_dir: Path) -> Path:
    """The empty directory a slim Claude Code on the login in `config_dir` runs in, made if it is not there: never a project."""
    cwd = _cwd(config_dir)
    cwd.mkdir(parents=True, exist_ok=True)
    return cwd


def _cwd(config_dir: Path) -> Path:
    return config_dir / "cwd"


async def spawn(station: Station, argv: Sequence[str]) -> ClaudeCode:
    """Run a slim Claude Code's command on a terminal of hands' own, in its own directory and environment."""
    return await on_terminal(argv, station.cwd, environment(station.config_dir, station.proxy_url, station.inherited))


class BrainGone(Exception):
    """The brain ended, or cannot be typed into, while a turn waited on it, or before one was asked."""


class StopUnrecorded(Exception):
    """The brain's Claude Code did not stop the turn hands pressed Escape for, as the model reads it: the stop is no stop."""


class Untaken(Exception):
    """The brain was typed a turn and never took it: it is on a screen that is not its input."""


class NotLoggedIn(Exception):
    """The brain's config directory holds no login whose requests reach hands' proxy, so every turn would fail or go unheard."""


class LoginFailed(Exception):
    """Claude Code's own login did not finish; whatever login the brain held before, it still holds."""


class Unstartable(Exception):
    """A Claude Code of hands' own could not be started: no claude to run, or a fritter that never opened its socket, or a brain that never turned its input on."""


def brain_claude(config_dir: Path) -> Path:
    """The Claude Code hands runs as its own on `config_dir`: hands' copy of the release it is pinned to
    (hands.brain.pinned), never the person's claude, whose release their own install moves under it."""
    try:
        return pinned.verified(config_dir)
    except pinned.Unpinned as error:
        raise Unstartable(str(error)) from None


@dataclass(frozen=True)
class Login:
    """What `hands login` left the brain holding: its account, and why it took Claude Code's first run, if it did."""

    account: Account
    first_run: str | None


def unanswered(config_dir: Path) -> str | None:
    """Why the brain would start on one of Claude Code's first screens, or None when it would not: Claude Code records
    in its own .claude.json that its onboarding finished, that the directory the brain runs in, or one above it, is
    trusted, and its answer on an API key its settings.json sets, by the key's last 20 characters. A config
    directory made by anything else, `claude auth status` among them, has none of them."""
    state = config_dir / ".claude.json"
    settings = config_dir / "settings.json"
    cwd = _cwd(config_dir).resolve()
    try:
        key = _settings_key(settings)
    except Rejected as error:
        return f"{settings} unreadable: {error}"
    try:
        said = Payload.parse(state.read_bytes())
        projects = Payload.of(said.fields.get("projects", {}), "its projects").fields
        trusted = any(Payload.of(projects[place], place).optional_flag("hasTrustDialogAccepted") for place in map(str, (cwd, *cwd.parents)) if place in projects)
        onboarding = said.optional_flag("hasCompletedOnboarding")
        responses = Payload.of(said.fields.get("customApiKeyResponses", {}), "its API key answers")
        keys_answered = {*responses.optional_items("approved"), *responses.optional_items("rejected")}
    except FileNotFoundError:
        return f"no {state}"
    except Rejected as error:
        # Claude Code's own first run is what writes this state, so it is the run that mends one hands cannot read.
        return f"{state} unreadable: {error}"
    if not onboarding:
        return "its onboarding unfinished"
    if not trusted:
        return f"{cwd} untrusted"
    if key is not None and key[-20:] not in keys_answered:
        return f"the API key {settings} sets unanswered"
    return None


def _settings_key(settings: Path) -> str | None:
    """The API key `settings` puts in the brain's environment, where the brain's own is set: hands' never reaches it (FOREIGN_LOGINS)."""
    try:
        raw = settings.read_bytes()
    except FileNotFoundError:
        return None
    return Payload.of(Payload.parse(raw).fields.get("env", {}), "its env").optional_text("ANTHROPIC_API_KEY")


def answered(config_dir: Path) -> None:
    """Raises Unstartable, naming the command that answers them, while the brain would start on Claude Code's first screens."""
    if (why := unanswered(config_dir)) is not None:
        raise Unstartable(f"the brain has not been through Claude Code's first screens ({why}): `hands login` answers them")


# The logins `claude auth login` makes, each named by its own flag: a Claude plan, or an Anthropic Console key.
type Method = Literal["claudeai", "console"]

# [LAW:one-source-of-truth] the authMethod `claude auth status` says of each login once it is made.
AUTH_METHODS: dict[Method, str] = {"claudeai": "claude.ai", "console": "api_key"}


def login(config_dir: Path, base_url: str, inherited: Mapping[str, str], method: Method) -> Login:
    """Log `config_dir` in with Claude Code's own login, at this terminal, in the directory the brain runs in; the
    account it holds after.

    A config directory whose first screens are unanswered gets Claude Code's first run, its screens, its login among
    them, answered once here: `claude auth login` alone leaves them for the brain's first start, where nobody is at its
    keyboard. `method` is the login asked for: `claude auth login` makes it, and on the first run's login screen, which
    offers every login, it is the one to pick. Raises LoginFailed when the brain holds another after."""
    # Before any run of Claude Code on it, which would sync what the brain's settings do not keep out.
    account_kept_out(config_dir)
    first_run = unanswered(config_dir)
    # [LAW:one-source-of-truth] the brain's own claude, environment, and settings sources, so the login lands in its config
    # directory, which the daemon reads, no credential of this shell's stands in for the one being made, and the first
    # run's screens are answered under the settings the brain starts with.
    argv, ran = (["auth", "login", f"--{method}"], "`claude auth login` for the brain") if first_run is None else (["--setting-sources", "user"], "the brain's first run of Claude Code")
    signed = subprocess.run([brain_claude(config_dir), *argv], cwd=workdir(config_dir), env=environment(config_dir, base_url, inherited))
    if signed.returncode != 0:
        raise LoginFailed(f"{ran} exited {signed.returncode}")
    # A first run quit before its last screen exits 0 as one that answered them all.
    answered(config_dir)
    held = logged_in(config_dir, base_url, inherited)
    # [LAW:no-silent-failure] a first run's screen, or a login its settings.json sets ahead of the one made, can leave the
    # brain on another login than the one asked for.
    if held.method != AUTH_METHODS[method]:
        raise LoginFailed(f"the brain holds {held}, not the {AUTH_METHODS[method]} login asked for")
    return Login(held, first_run)


# Each is on unless settings.json says false: they sync the login's account's skills and plugins, which are Brandon's.
KEPT_OUT = ("syncClaudeAiSkills", "syncClaudeAiPlugins")


def starting_settings(config_dir: Path) -> bool:
    """Writes the settings.json a brain home starts with when it has none; whether it wrote one. One that is there is
    left as it is: the brain's settings are its directory's to say."""
    # [LAW:one-source-of-truth] what account_kept_out requires, and the mode the README says the brain runs in.
    starting = {**dict.fromkeys(KEPT_OUT, False), "permissions": {"defaultMode": "default"}}
    config_dir.mkdir(parents=True, exist_ok=True)
    try:
        with (config_dir / "settings.json").open("x") as made:
            json.dump(starting, made, indent=2)
    except FileExistsError:
        return False
    return True


def account_kept_out(config_dir: Path) -> None:
    """Raises Unstartable unless the brain's settings.json keeps out the skills and plugins its login's account syncs.

    They are Brandon's, never the brain's, and settings.json is the only place Claude Code reads either switch from:
    not --settings, and not the environment. Left out, each is on."""
    settings = config_dir / "settings.json"
    fix = f'set "syncClaudeAiSkills": false and "syncClaudeAiPlugins": false in {settings}'
    try:
        said = Payload.parse(settings.read_bytes())
        synced = [switch for switch in KEPT_OUT if said.fields.get(switch) is not False]
    except (OSError, Rejected) as error:
        raise Unstartable(f"the brain's settings could not be read ({error}): {fix}") from None
    if synced:
        raise Unstartable(f"the brain would load its account's {' and '.join(synced)}: {fix}")


def logged_in(config_dir: Path, base_url: str, inherited: Mapping[str, str]) -> Account:
    """The account `config_dir` is logged in as; raises NotLoggedIn, naming the command that makes a login, when it has
    none, or one whose requests do not reach hands' proxy."""
    try:
        # A timed-out child is killed and reaped by run itself.
        asked = subprocess.run(
            [brain_claude(config_dir), "auth", "status"],
            env=environment(config_dir, base_url, inherited),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=AUTH_STATUS_SECONDS,
        )
    except subprocess.TimeoutExpired:
        raise NotLoggedIn(f"`claude auth status` for the brain did not answer in {AUTH_STATUS_SECONDS:.0f}s") from None
    try:
        status = Payload.parse(asked.stdout)
        if not status.flag("loggedIn"):
            raise NotLoggedIn("the brain has no login: `hands login` gives it one")
        # [LAW:no-silent-failure] hands' proxy, which hears what the brain says, forwards to Anthropic's API alone, so a
        # brain on another provider would answer nothing. hands' environment never chooses one (FOREIGN_LOGINS).
        if (provider := status.text("apiProvider")) != "firstParty":
            raise NotLoggedIn(f"the brain reaches Claude through {provider}, not the Anthropic API hands' proxy forwards to: {config_dir / 'settings.json'} or the machine's managed settings choose {provider}")
        return Account(status.text("authMethod"), status.optional_text("email") or status.optional_text("apiKeySource"))
    except Rejected as error:
        raise NotLoggedIn(f"`claude auth status` for the brain answered {asked.stdout[:200]!r} {asked.stderr[:200]!r}, not its status: {error}") from None


@dataclass(frozen=True)
class BrainAnswered:
    """The end of a brain turn by itself: the model told the turn of Claude Code's that answered what it typed, under the
    prompt id of the Stop that ended it, or with what the API failed it with. Its words are on the wire."""

    # The prompt id the turn was told under: its Stop's, which its last transcript records carry too.
    prompt: str | None
    error: StopFailed | None


@dataclass(frozen=True)
class BrainStopped:
    """The end of a brain turn by the Escape hands pressed when it was asked to stop it (`Brain.interrupt`): once the
    model tells the turn of Claude Code's it stopped, or at once when none of it ran. The ids it went by are on its
    event."""


# How a turn of the brain's ended.
BrainEnded = BrainAnswered | BrainStopped


# Compared by identity: each is its own request, however alike two calls are.
@dataclass(frozen=True, eq=False)
class Asked:
    """A permission the brain's own setup asks about, as the brain's model holds it at its hook while the user is asked:
    settled once, by what they answer, by its hook's day running out, or by the turn's end, whichever comes first.

    [LAW:one-source-of-truth] a view of the model's held request, not a second record of it: open while the model holds
    it, settled through the model, and decided as its hook is answered, so the stage, the event and Claude Code read
    one fact."""

    permission: Permission
    request: RequestId
    replying: "asyncio.Future[HookReply]"
    sessions: Sessions
    session: SessionId

    @property
    def open(self) -> bool:
        return holds(self.sessions.live_session(self.session), self.request)

    def settle(self, decision: Allow | Deny) -> None:
        # The first to settle it answers its hook; any later one is NotWaiting, and changes nothing.
        self.sessions.answer(self.request, decision)

    async def decided(self) -> Allow | Deny | DenyAndInterrupt:
        """What its hook was sent, once it was."""
        match refused(await asyncio.shield(self.replying)):
            case Allow() | Deny() | DenyAndInterrupt() as decision:
                return decision
            case other:
                # [LAW:no-silent-failure] the model answers a permission only with these: anything else is a bug.
                raise ValueError(f"permission request {self.request} was answered {other}, which answers no permission")


def _unanswered(event: str) -> Mapping[str, object]:
    """What a hook the brain posted is answered when hands broke handling it: a dialog refused, any other hook let go."""
    match event:
        case "PermissionRequest":
            return hook_output(Deny(BROKEN))
        case "Elicitation":
            return DECLINED
        case _:
            return {}


def refused(reply: HookReply) -> Allow | AllowWith | Approve | Deny | DenyAndInterrupt:
    """What a brain dialog's hook is sent for the model's reply. The model lets a brain dialog go only when Claude Code
    went on past it unanswered (at its prompt, a prompt typed, the session's end): never once its tool ran, since the
    brain posts no PostToolUse or PostToolUseFailure (HOOKS). Left to Claude Code's own dialog, nobody would see it on the
    brain's screen and its turn would wait on it, so it is refused instead."""
    match reply:
        case Withdraw():
            return Deny(LET_GO)
        case Allow() | AllowWith() | Approve() | Deny() | DenyAndInterrupt():
            return reply


@dataclass
class _Turn:
    """A turn asked of the brain: what it typed, whether Claude Code took it, and the facts its event carries. Which of
    Claude Code's turns answers it, and when that one is over, is the model's to say (`Brain._told`)."""

    answered: asyncio.Future[BrainEnded]
    taken: asyncio.Future[None]  # resolved when Claude Code took what the turn typed
    # Resolved once the turn's typing is over, however it went: then it is taken, or never will be.
    tried: asyncio.Future[None]
    # Told of each permission the turn holds at its hook, to put it to the user.
    asks: Callable[[Asked], None]
    # The turn's own unit of work, begun as it was asked: each dialog it holds is a part of it.
    began: Begun
    # How long the turn has to be taken, read once as it is asked: its typing and its stop wait the same time.
    take: float
    # What the turn types into the brain's input.
    typed: PromptText
    # The tools the turn's latest request offered the model: what the brain's own setup gave it, beside hands' tools.
    offered: tuple[str, ...] = ()
    # The prompt id Claude Code took what was typed under, as its UserPromptSubmit names it: a turn's own, the running
    # turn's, or, while the last one's Stop hooks still ran, the last one's.
    taken_under: PromptId | None = None
    # Whether that id was of a turn Claude Code already had, so it queued what was typed behind or into that turn.
    queued: bool = False
    # The turn of Claude Code's that runs what it typed, as `ClaudeTurns` holds it: the one its words opened, or were
    # filed into, or, once its telling found them still waiting, the turn they run as behind it. None until taken.
    running: "ClaudeTurn | None" = None
    # The id the model told the turn that answers it under, stopped or not: its answer, or what it got as far as.
    told: PromptId | None = None
    # Whether hands pressed the Escape that stops it: then it ends BrainStopped once Claude Code holds nothing of it
    # (`Brain._escape`, once `_clear` returns), and its telling, if the model told it, has been heard.
    stopped: bool = False
    # What broke in the brain's own work for the turn: it ends with this once Claude Code's turn is stopped.
    broken: BaseException | None = None
    # Why the order of the brain's turns was lost while this one was in flight, if it was (`Brain._lost`).
    order_lost: str | None = None


class Unran(Exception):
    """Words typed for a turn, waiting to run as the turn behind another, whose turn ended before Claude Code ran them."""


class OrderLost(Exception):
    """Claude Code went on before hands heard which of its turns it ran: a prompt hook answered past its declared
    timeout, or a Stop let go before it was decided. Which turn ran what is not guessed: every turn open then fails."""


# [LAW:one-type-per-behavior] what opened a turn of Claude Code's that hands did not type, as a table: each row a kind and
# the prompt it is known by. A turn of another kind, a scheduled prompt's say, is one more row, and nothing that matches
# a turn to its end reads the kind.
OPENERS: list[tuple[str, re.Pattern[str]]] = [
    # A background task's result: the notification alone, or inside the system reminder Claude Code wraps it in.
    ("notified", re.compile(r"\s*(<system-reminder>\s*)?<task-notification>")),
]
# A turn hands typed, and one of no kind in OPENERS.
TYPED, OTHER = "typed", "other"


def opener(words: str) -> str:
    """The kind of turn a prompt hands did not type opens."""
    return next((kind for kind, prompt in OPENERS if prompt.match(words)), OTHER)


# Compared by identity: each is its own turn of Claude Code's, however alike two are.
@dataclass(eq=False)
class ClaudeTurn:
    """A turn of Claude Code's in the brain, heard opening and not yet told: [LAW:one-source-of-truth] what the brain
    and its stage both read for which of Claude Code's turns runs and when it ends (`ClaudeTurns`)."""

    # The ids it goes by: its prompt hook's; none, for words queued behind a turn, until its telling names it.
    ids: frozenset[PromptId]
    # What opened it: TYPED, or a kind of OPENERS, or OTHER.
    opener: str
    # Resolved by its telling, or failed when the brain exits or the order of its turns is lost.
    ended: "asyncio.Future[BrainAnswered]"
    # Whether a turn hands typed is answered by it: its words opened it, or were filed into it, or wait to run as it.
    awaited: bool = False
    # How its telling was found to be its: by an id they share, or as the turn behind the one told before it.
    matched: Literal["id", "behind"] | None = None


class ClaudeTurns:
    """[LAW:no-ambient-temporal-coupling] the one owner of which of Claude Code's turns runs in the brain and which
    telling ends each, moved only by what the brain's model hands over, in its order (`Brain._took`, `Brain._told`).

    Every turn but one opens under a prompt hook, which holds Claude Code until hands has heard it, so a turn is known by
    its id before its first request leaves. The one no hook opens is words left waiting at a telling, which Claude Code
    runs once that telling's Stop is let go: it is known as the turn behind, named by the first telling no turn held
    goes by. Both rest on the brain's hooks being waited on (hookconfig's brain table); when one is not, every turn
    open is `failed` (`Brain._lost`), and none is guessed."""

    def __init__(self) -> None:
        self._open: list[ClaudeTurn] = []
        # The turn Claude Code runs now: the last one opened, while it is open.
        self._current: ClaudeTurn | None = None

    @property
    def current(self) -> ClaudeTurn | None:
        return self._current

    @property
    def open(self) -> tuple[ClaudeTurn, ...]:
        return tuple(self._open)

    @property
    def opened_itself(self) -> tuple[ClaudeTurn, ...]:
        """The open turns Claude Code opened itself, of a kind OPENERS knows: spoken to their end, never stopped for a
        turn typed. A turn of words it took as none hands typed is no turn's, and is stopped before the next (`_taking`)."""
        kinds = {kind for kind, _ in OPENERS}
        return tuple(running for running in self._open if running.opener in kinds)

    def opened(self, prompt: PromptId, kind: str, awaited: bool) -> ClaudeTurn:
        """A turn its prompt hook opened under `prompt`: Claude Code runs it now, as it runs one prompt at a time."""
        return self._run(ClaudeTurn(frozenset({prompt}), kind, asyncio.get_running_loop().create_future(), awaited))

    def behind(self, kind: str, awaited: bool) -> ClaudeTurn:
        """The turn words left waiting run as, next, which its Stop holds back until now: one turn, however many words
        wait, so words that join one already waiting run in it, and it is awaited once any of them is hands'."""
        running = next((running for running in self._open if not running.ids), None)
        if running is None:
            return self._run(ClaudeTurn(frozenset(), kind, asyncio.get_running_loop().create_future(), awaited))
        running.awaited = running.awaited or awaited
        return running

    def filed(self, prompt: PromptId) -> ClaudeTurn | None:
        """The open turn words filed under `prompt` were filed into."""
        return next((running for running in self._open if prompt in running.ids), None)

    def told(self, told: Summarise) -> ClaudeTurn | None:
        """The open turn `told` ends, ended by it: the one it shares an id with, or else the turn behind no telling has
        named yet, which this one names; None for a telling of no turn heard opening."""
        running = next((running for running in self._open if not running.ids.isdisjoint(told.ids)), None)
        if running is not None:
            running.matched = "id"
        else:
            # A telling that goes by no id is of no turn: it names none, the turn behind included.
            running = next((running for running in self._open if not running.ids), None) if told.ids else None
            if running is None:
                return None
            running.ids, running.matched = told.ids, "behind"
        self._end(running)
        running.ended.set_result(BrainAnswered(told.turn, told.failure))
        return running

    def failed(self, error: BaseException) -> tuple[ClaudeTurn, ...]:
        """Ends every open turn with `error`: what of them Claude Code still runs is no turn's from here on."""
        failed = self.open
        for running in failed:
            self.fail(running, error)
        return failed

    def fail(self, running: ClaudeTurn, error: BaseException) -> None:
        """Ends `running` with `error`, as no telling of it will: words Ctrl-C cleared before they ran as the turn behind."""
        self._end(running)
        running.ended.set_exception(error)
        # Said by whoever waits on it, or by the brain itself: not again by asyncio.
        running.ended.exception()

    def _run(self, running: ClaudeTurn) -> ClaudeTurn:
        self._open.append(running)
        self._current = running
        return running

    def _end(self, running: ClaudeTurn) -> None:
        self._open.remove(running)
        if self._current is running:
            self._current = None


def model(record: Record, told: Callable[[Story], None], took: Callable[[Submitted], None]) -> Sessions:
    """The brain's Claude Code session, as the one model of a Claude Code turn reads every working session, in a registry
    of its own: where each prompt's words went handed to `took`, and each turn it tells to `told`, in the order the model
    decided them.

    [LAW:one-type-per-behavior] the registry working sessions live in, another of it: its effects are the same, and of
    them only the tellings are the brain's to hear. It works in an empty directory, so no repository is read; and what
    its turns say is the brain's stage's to say, from the wire, never what the model makes of its session.
    """
    # [LAW:one-source-of-truth] the brain's Stop held as long as its hook declares it is waited on (hookconfig).
    return Sessions(time.monotonic, record, story_sink=told, heard_sink=_unsaid, submitted_sink=took, stop_hold=TAKE_SECONDS)


def _unsaid(_heard: Heard) -> None:
    """What the brain's session would have said of its progress: the stage says the brain's turn from the wire instead."""


class Brain:
    """A running brain: a turn is asked with `ask`, which returns once the brain has said the turn is over."""

    def __init__(
        self,
        claude: ClaudeCode,
        session: SessionId,
        typist: Typist,
        listener: web.AppRunner,
        sockets: Path,
        config_dir: Path,
        cwd: Path,
        record: Record,
        launched: Span,
        pause: Callable[[float], Awaitable[None]],
    ) -> None:
        self._claude = claude
        self._config_dir = config_dir
        self.session = session
        self._typist = typist
        # The Claude Code fritter wrapped, and where it works: what its own hooks do not say of it.
        self._pid = typist.pid
        self._cwd = cwd
        self._listener = listener
        self._sockets = sockets
        self._record = record
        # What the brain does that no turn asked for is a part of its launch, in the launch's trace.
        self._launched = launched
        # [LAW:no-ambient-temporal-coupling] how the brain's bounds pass is the brain's to be handed, and is that alone: a
        # typing's take and a stop's end wait on one pause, and no other wait of the brain's is one.
        self._pause = pause
        # [LAW:one-source-of-truth] Claude Code's turns in the brain, as the one model of a turn reads them: which runs,
        # which waits in its queue, and which is over.
        self._sessions = model(record, self._told, self._took)
        # [LAW:single-enforcer] the brain's status and transcript, read as every working session's are, on the same period.
        self._reader = Reader(self._sessions, Tails(self._sessions), self._apply)
        # [LAW:single-enforcer] the input is the user's, and only two things are ever typed into it: a turn, and the
        # keys that stop one. A turn keeps it until its hook says Claude Code took the turn, and a stop's keys go in alone.
        self._input = asyncio.Lock()
        # [LAW:no-ambient-temporal-coupling] the turn in flight is the brain's own state, not its asker's: it is over
        # when the model tells the turn that answers it, whether or not anyone still waits on it, and the next is typed only then.
        self._turn: _Turn | None = None
        # [LAW:one-source-of-truth] which of Claude Code's turns runs, and which telling ends each.
        self._claude_turns = ClaudeTurns()
        # [LAW:no-ambient-temporal-coupling] set, and replaced, each time what the model applied has been heard (`_heard`):
        # what a clear waits on to see Claude Code hold nothing of what it stopped, re-read from the model each time it moves.
        self._moved = asyncio.Event()
        self._typing: set[asyncio.Task[None]] = set()
        # When the last stop pressed Ctrl-C, on the event loop's clock.
        self._cleared = float("-inf")
        with continuing(launched):
            self._reading = asyncio.ensure_future(self._reader.keep_reading(READ_SECONDS))
            self._exit = asyncio.ensure_future(self._run_out(begun()))

    @property
    def pid(self) -> int:
        return self._claude.pid

    async def ask(self, text: str, asks: Callable[[Asked], None]) -> BrainEnded:
        """One turn of the brain's, from its typing to its end; `asks` is told of each permission the turn holds for the user."""
        # [LAW:no-ambient-temporal-coupling] one turn in flight at a time: the next is asked once the one before is answered.
        while self._turn is not None:
            await asyncio.wait({self._turn.answered})
        if self._exit.done():
            raise BrainGone(f"the brain had exited ({self._exit.result()}) before it was asked")
        loop = asyncio.get_running_loop()
        # Behind a space, as every prompt hands types: a leading / or ! is then the character it is.
        turn = self._turn = _Turn(loop.create_future(), loop.create_future(), loop.create_future(), asks, begun(), TAKE_SECONDS, Text(pasted(text)).typed)
        # An asker that stops waiting leaves the turn to be typed and to run to its end, which is still the brain's to hear.
        self._keep(self._send(turn), turn)
        return await asyncio.shield(turn.answered)

    def current(self) -> ClaudeTurn | None:
        """The turn of Claude Code's the brain runs now, which its requests on the wire are of: None while it runs none
        it was heard opening."""
        return self._claude_turns.current

    def answering(self) -> ClaudeTurn | None:
        """[LAW:one-source-of-truth] the turn of Claude Code's that answers the turn in flight: None until it is taken,
        and moved, as `_told` moves it, to the turn behind once its words are found still waiting."""
        return None if self._turn is None else self._turn.running

    def interrupt(self, running: ClaudeTurn | None) -> bool:
        """Stops `running`, the turn of Claude Code's whose work was shown, as Escape at the brain's keyboard stops it, and
        says whether there was one to stop: the turn in flight, where `running` answers it, or none Claude Code has yet;
        else one Claude Code opened itself, which no turn in flight waits on, while it is open.

        [LAW:single-enforcer] the one way a turn is stopped, a broken one's included: `_stop`, with the input held, so its
        keys go in alone, and what breaks it is `_broke`'s. The turn ends BrainStopped, unless it ended by itself first.
        """
        turn = self._turn
        if turn is not None and not turn.answered.done() and turn.running is running:
            self._keep(self._stop(turn), turn)
            return True
        if running is not None and running in self._claude_turns.opened_itself:
            self._keep(self._halt(running), None)
            return True
        return False

    async def _halt(self, running: ClaudeTurn) -> None:
        """Stops a turn Claude Code opened itself, with the input held: over once its telling ends it (`_clear`)."""
        async with self._input:
            if running in self._claude_turns.open:
                await self._clear(None)

    def _keep(self, work: Coroutine[None, None, None], turn: _Turn | None) -> None:
        """Runs on in a task of the brain's own, whoever asked for it, for `turn`, the turn in flight it was begun in."""
        task = asyncio.create_task(work)
        self._typing.add(task)
        task.add_done_callback(lambda done: self._kept(done, turn))

    def _kept(self, task: "asyncio.Task[None]", turn: _Turn | None) -> None:
        self._typing.discard(task)
        if not task.cancelled() and (error := task.exception()) is not None:
            self._broke(turn, error)

    def _broke(self, turn: _Turn | None, error: BaseException) -> None:
        """[LAW:single-enforcer] what the brain's own work raised that nothing in it expected: said once, with where it
        came from, and the end of the turn it was done for, stopped as an Escape stops it, so no dialog of it is left open
        for the next turn's Return, its asker is answered, and the turns behind it are typed. A turn already over is ended
        already: what Claude Code still runs of it is no turn's (`_owner`), and is cleared before the next is typed
        (`_taking`), so its break ends nothing the turn in flight runs."""
        logger.opt(exception=error).error(f"the brain's own work failed: {error!r}")
        if turn is not None and turn is self._turn:
            # [LAW:single-enforcer] the turn's dialogs are refused, and it ends, in the step this is learned, whether or not
            # its stop's keys go in: the model keeps whichever ending came first.
            self._sessions.hear(Ending(self.session, DenyAndInterrupt(BROKEN)))
            if turn.broken is None:
                turn.broken = error
                self._keep(self._stop(turn), turn)

    async def _send(self, turn: _Turn) -> None:
        # [LAW:nothing-unseen] the turn is one event of the brain's own, from its typing to its end, however long its asker
        # still waits on it: a turn left to run on is still heard to its end.
        with unit("brain.turn", self._record, began=turn.began):
            try:
                # [LAW:no-ambient-temporal-coupling] the input is the turn's from its first key until Claude Code says it
                # took the turn. Claude Code reads keys that reach it together as one paste, and a Return inside a paste
                # sends nothing: what was typed 10ms behind a turn joined the turn's prompt, or was left in the input with
                # it.
                try:
                    async with self._input:
                        taken = await self._taking(turn)
                finally:
                    turn.tried.set_result(None)
                if not taken:
                    raise Untaken(f"the brain did not take the turn typed into it in {turn.take:g}s; if it is on one of Claude Code's first screens, `hands login` answers them")
            except (BrainGone, Untaken, StopUnrecorded) as error:
                self._over(turn, error)
            await asyncio.wait({turn.answered})
            running = turn.running
            # [LAW:nothing-unseen] which of Claude Code's turns answered it, and how its telling was found to be that one's.
            annotate(
                taken=turn.taken_under, queued=turn.queued, offered=turn.offered, told=turn.told, order_lost=turn.order_lost,
                ids=None if running is None else tuple(sorted(running.ids)), matched=None if running is None else running.matched,
            )
            match turn.answered.exception() or turn.answered.result():
                case BrainAnswered(error=None):
                    pass
                case BrainAnswered(error=error):
                    fail(f"the brain's turn ended in error: {error}")
                case BrainStopped():
                    annotate(stopped=True)
                case failed:
                    fail(f"the brain failed the turn: {failed}")

    async def _taking(self, turn: _Turn) -> bool:
        """Types `turn`, with the input held, behind nothing, and waits out a pause of `turn.take` for it to be taken, or to
        end: says whether it was. [LAW:single-enforcer] what Claude Code runs that the model has not told, or holds in its
        queue to run, is no turn's in flight, for an ask that already failed or one nobody made: it is stopped before the
        turn is typed, so the turn is Claude Code's next and runs alone."""
        # [LAW:no-ambient-temporal-coupling] typed at once, never held behind a turn Claude Code opened itself: Claude Code
        # queues the words behind it, and runs them in it or as the turn behind it (`_took`, `_told`), so the turn's take
        # is bounded from now. Only what is no turn's, and none of Claude Code's own, is stopped.
        if not self._claude_turns.opened_itself and not self._all_told():
            await self._clear(turn)
        await self._type(lambda typist: typist.type(turn.typed))
        await self._waited(turn.take, turn.taken, turn.answered)
        return turn.taken.done() or turn.answered.done()

    async def _waited(self, seconds: float, *ending: "asyncio.Future[Any]") -> None:
        """Until the first of `ending` is done, or a pause of `seconds` on the brain's one pause has passed.

        [LAW:no-silent-failure] a pause that failed, however the wait ended, says so, and is not taken for the time
        running out; a wait that was itself cancelled stays cancelled, as asyncio's every canceller counts on."""
        pausing = asyncio.ensure_future(self._pause(seconds))
        try:
            await asyncio.wait({*ending, pausing}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            pausing.cancel()
            await asyncio.wait({pausing})
        if not pausing.cancelled():
            pausing.result()

    async def _clear(self, turn: _Turn | None) -> None:
        """Stops whatever Claude Code runs and empties its input, with the input held, for `turn`, and returns once the
        model holds nothing of what it stopped. Escape posts no hook for the turn it stops: the model hears it over from
        the transcript and Claude Code's status.

        [LAW:single-enforcer] "a stop has taken" is decided here, for every stop: a turn's (`_escape`) and what is left
        untold before a turn is typed (`_taking`). It raises StopUnrecorded when Claude Code still holds what it stopped
        after INTERRUPT_SECONDS, and BrainGone when the brain exits first."""
        held = self._modelled()
        # [LAW:no-ambient-temporal-coupling] what runs is ended in the model in the step its Escape goes: every dialog it
        # holds is refused, and each asked until a prompt starts new work, with a refusal that ends the turn whether it
        # reaches Claude Code before the Escape or after it.
        self._sessions.hear(Ending(self.session, DenyAndInterrupt(STOPPED)))
        await self._type(lambda typist: typist.press("escape"))
        # Escape puts a prompt stopped before any reply back in the input, and with it what was typed queued behind the
        # prompt (its cancel pops the queue into the input), which the next turn typed would join; Ctrl-C clears
        # it. On an input left empty it arms Claude Code's exit instead, which a second Ctrl-C within 800ms takes,
        # whatever was typed between; so no two are pressed that close,
        # counted from when each went and with room for Claude Code to read the first late.
        loop = asyncio.get_running_loop()
        await asyncio.sleep(self._cleared + EXIT_SECONDS - loop.time())
        await self._type(lambda typist: typist.press("ctrl_c"))
        self._cleared = loop.time()
        if turn is not None and (turn is not self._turn or not turn.stopped):
            logger.warning(f"the brain stopped what Claude Code ran that was no turn's: {None if held is None else held.turn}")
        # Taken once Claude Code holds nothing of what it ran: every turn it ran told, and nothing of it waiting in its
        # queue. [LAW:no-silent-failure] still held past INTERRUPT_SECONDS, it was not stopped, and says what Claude Code
        # showed.
        waiting = asyncio.ensure_future(self._told_out())
        try:
            await self._waited(INTERRUPT_SECONDS, waiting, self._exit)
        finally:
            waiting.cancel()
            await asyncio.wait({waiting})
        if not waiting.cancelled():
            waiting.result()
            # [LAW:single-enforcer] Claude Code holds nothing now, so a turn of `ClaudeTurns` still open is one no telling
            # will end: words Ctrl-C cleared from its queue before they ran. Each ends here, where they were cleared.
            self._claude_turns.failed(Unran("Claude Code was stopped, and its input cleared, before they ran"))
            return
        if self._exit.done():
            raise BrainGone("the brain exited while hands stopped what it ran")
        raise StopUnrecorded(f"the brain's Claude Code did not stop what hands pressed Escape for in {INTERRUPT_SECONDS:g}s: the model holds {self._modelled()}; it showed:\n{self._claude.shown()}")

    async def _told_out(self) -> None:
        """Returns once Claude Code holds nothing of what it ran, as the model reads it.

        [LAW:one-source-of-truth] each read comes straight after the registry is drained, so every telling the model
        decided has reached `_told` first: a stopped turn ends with the id it was told under. The move waited on is the
        one current before the drain, so one heard during the drain or the read wakes the wait at once, and no hook's
        `_heard` waits on a clear."""
        while True:
            moved = self._moved
            await self._sessions.drained(self.session)
            if self._all_told():
                return
            await moved.wait()

    async def _stop(self, turn: _Turn) -> None:
        try:
            await self._escape(turn)
        except Exception as error:
            # A stop that breaks is never pressed again: its keys may have gone, and a second Ctrl-C within Claude Code's
            # exit window ends it. The turn ends with what broke it, and `_broke` says it.
            if turn.broken is None:
                turn.broken = error
            raise
        finally:
            # A broken turn ends with what broke it however its stop went, even unstopped: nothing else is left to end it.
            # What it holds the model refused as it learned the turn broke (`_broke`), so no dialog of it is left open
            # for the next turn's Return.
            if turn.broken is not None and not turn.answered.done():
                self._over(turn, turn.broken)

    async def _escape(self, turn: _Turn) -> None:
        # [LAW:no-ambient-temporal-coupling] Escape goes once Claude Code has taken the turn, never before: until then
        # nothing of the turn's runs for it to stop.
        await asyncio.wait({turn.taken, turn.tried}, return_when=asyncio.FIRST_COMPLETED)
        if self._turn is not turn:
            return
        if not turn.taken.done():
            logger.warning("the brain was not stopped: its turn was not taken")
            return
        try:
            # Held until the stop has taken, so nothing is typed into a Claude Code still running what was stopped.
            async with self._input:
                if self._turn is not turn:
                    return
                # Whatever way Claude Code's turn ends from here, it was stopped: an Escape went for it.
                turn.stopped = True
                await self._clear(turn)
        except (BrainGone, StopUnrecorded) as error:
            self._over(turn, error)
            return
        # Over now Claude Code holds nothing of it: the turn the Escape stopped told, and the words it put back in the
        # input from Claude Code's queue, which Ctrl-C cleared, taken out of it.
        self._over(turn, BrainStopped())

    def hear(self, observed: Observed) -> None:
        """The brain's own requests, read from the wire: the tools its turn offered the model, and whether they reached hands'."""
        match observed:
            case Sent(session=session, kind=MainTurn(), body=body) if session == self.session:
                tools = tool_names(body)
                if self._turn is not None:
                    self._turn.offered = tools
                # [LAW:no-silent-failure] a brain without hands' tools answers every question about the sessions from nothing,
                # and one with them only behind ToolSearch asks the model for them before it can answer.
                if not any(name.startswith(f"mcp__{SERVER_NAME}__") for name in tools):
                    logger.error(
                        f"the brain's turn went to the model without hands' tools and with {TOOL_SEARCH}: tool search is on"
                        f" though hands' --settings turn it off ({tools})"
                        if TOOL_SEARCH in tools
                        else f"the brain's turn went to the model without hands' tools: it did not connect to hands' MCP server ({tools})"
                    )
            case _:
                pass

    async def exited(self) -> int:
        """Waits for the brain to end, and returns its exit code; the line that says it ended is written once, however many
        wait. Raises what stopped the model reading the brain's turns, if that came first: no turn's end would be heard."""
        await asyncio.wait({asyncio.shield(self._exit), self._reading}, return_when=asyncio.FIRST_COMPLETED)
        if self._reading.done() and not self._reading.cancelled() and not self._exit.done():
            self._reading.result()
        return await asyncio.shield(self._exit)

    async def stop(self) -> None:
        """Ends the brain and lets go of what it held, however its end went. What stopped the model reading its turns is
        `exited`'s to raise, to whoever waits on the brain, not the stop's."""
        try:
            await self._claude.stop()
            await asyncio.shield(self._exit)
        finally:
            self._reading.cancel()
            await self._listener.cleanup()
            shutil.rmtree(self._sockets, ignore_errors=True)

    async def _type(self, typing: Callable[[Typist], None]) -> None:
        try:
            await asyncio.to_thread(typing, self._typist)
        except Untyped as error:
            raise BrainGone(f"the brain cannot be typed into: {error}") from error

    async def _run_out(self, up: Begun) -> int:
        # [LAW:nothing-unseen] the brain's run, from its input coming up to its process's end, is one event as it ends:
        # its exit code, and the last of what it showed on its terminal; or that hands stopped waiting for its end.
        with unit("brain.run", self._record, began=up):
            code = await asyncio.shield(self._claude.exit)
            annotate(pid=self.pid, code=code, shown=self._claude.shown())
            # Nothing is left to interrupt, and every dialog it held is answered all the same.
            self._sessions.hear(Ending(self.session, DenyAndInterrupt(f"the brain exited ({code})")))
            # [LAW:no-silent-failure] a turn that can never end is said to have failed, not left waiting.
            gone = f"the brain exited ({code}) before it answered"
            self._claude_turns.failed(BrainGone(gone))
            if self._turn is not None:
                self._over(self._turn, BrainGone(gone))
        return code

    async def posted(self, event: str, body: bytes) -> Mapping[str, object]:
        """A hook the brain posted, to the path of its event, as it was posted, and the body its post is answered with:
        each handled as it arrives, in a handler of its own, as a working session's is (serve_hooks), so none waits
        behind another. What it does is a part of the brain's launch, where no turn it is found a part of says else."""
        with continuing(self._launched):
            try:
                return await self._posted(event, body)
            except Exception as error:
                # [LAW:single-enforcer] whatever breaks, the hook is answered, as one nobody answered: what breaks here is
                # no turn's. A dialog left unanswered is left to Claude Code's own, on a screen nobody sees, and the next
                # turn's Return would answer it yes.
                self._broke(None, error)
                return _unanswered(event)

    async def _posted(self, event: str, body: bytes) -> Mapping[str, object]:
        if event == "Elicitation":
            # A dialog is kept shut, whatever it says.
            self._refused(body)
            return DECLINED
        try:
            hook = parse_own_hook(body, pid=self._pid, cwd=self._cwd, at=self._sessions.now(), heard=self._sessions.stamp(), request=RequestId(uuid4().hex))
        except Rejected as error:
            return self._unread(event, error)
        if hook.session != self.session:
            logger.info(f"the brain's {hook.name} hook came for session {hook.session}, not the brain's")
            return self._foreign(hook)
        match hook:
            case PermissionHook():
                return hook_output(await self._permit(hook))
            case Hook():
                # Answered once it is applied, whatever it says, as a working session's hook is (serve_hooks): Claude Code
                # goes on only once the model has it, so nothing it writes after is read before it.
                loop = asyncio.get_running_loop()
                arrived = loop.time()
                held = await self._reader.hooked(hook)
                await self._heard()
                self._ordered(hook, held, loop.time() - arrived)
                return {}

    def _ordered(self, hook: Hook, held: StopHeld | None, took: float) -> None:
        """[LAW:no-silent-failure] whether Claude Code waited on the hook as the order of the brain's turns needs
        (`ClaudeTurns`): a prompt's answered within its declared timeout, a Stop's decided within its hold."""
        match hook.happened:
            case Prompted(prompt=prompt) | Stopped(prompt=prompt) if took > TAKE_SECONDS:
                self._lost(f"the {hook.name} hook of {prompt} was answered in {took:.1f}s, past the {TAKE_SECONDS:g}s hands has to answer it in")
            case Stopped(prompt=prompt) if held == "let go":
                self._lost(f"the Stop of {prompt} was let go before it was decided")
            case _:
                pass

    def _lost(self, why: str) -> None:
        """The order of the brain's turns lost: every turn of Claude Code's open fails, and the turn in flight with them
        when one of them answers it. What Claude Code still runs of them is no turn's, cleared before the next is typed
        (`_taking`)."""
        logger.error(f"the brain lost the order of its turns: {why}")
        failed = self._claude_turns.failed(OrderLost(why))
        turn = self._turn
        if turn is not None and not turn.answered.done():
            turn.order_lost = why
            if turn.running in failed:
                self._over(turn, OrderLost(why))

    def _unread(self, event: str, error: Rejected) -> Mapping[str, object]:
        match event:
            case "PermissionRequest":
                # [LAW:nothing-unseen] one event, though hands could not read which turn's it is.
                with unit("brain.permission", self._record):
                    fail(f"hands could not read the permission request: {error}")
                return hook_output(Deny(UNREAD))
            case "UserPromptSubmit" | "Stop":
                # [LAW:no-silent-failure] a turn's opening or its Stop the model never heard: which turn a telling ends is
                # no longer known (`ClaudeTurns`), and is not guessed.
                self._lost(f"the brain's {event} hook does not parse: {error}")
                return {}
            case _:
                logger.warning(f"the brain posted a hook that does not parse: {error}")
                return {}

    def _foreign(self, hook: Hook | PermissionHook) -> Mapping[str, object]:
        match hook:
            case PermissionHook():
                reply = Deny(NOBODY)
                # [LAW:nothing-unseen] one event, as every permission the brain posts is, though no turn of the brain's asked it.
                with unit("brain.permission", self._record):
                    annotate(session=hook.session, prompt=hook.prompt, tool=hook.tool, request=hook.asked.request, decision=reply)
                return hook_output(reply)
            case Hook():
                return {}

    async def _permit(self, hook: PermissionHook) -> Allow | AllowWith | Approve | Deny | DenyAndInterrupt:
        """Answers a permission request: held in the model as it arrives, and put to the user while a turn of theirs is
        in flight to ask in.

        [LAW:single-enforcer] it holds the request from its hold to its hook's answer: whatever breaks from the hold on is
        settled through the model before the reply is waited on, so the wait cannot hang on hands' own failure.
        [LAW:nothing-unseen] each is one event, written once its hook's reply is known: a part of the turn it was held
        for, or of the brain's launch where no turn of the user's was in flight to ask in."""
        began = begun()
        turn: _Turn | None = None
        raised: BaseException | None = None
        try:
            replying = self._reader.asked(hook)
        except Exception as error:
            # The model never held it, so it is answered here, beside the model: the one answer that is.
            raised = error
            self._broke(None, error)
            reply: Allow | AllowWith | Approve | Deny | DenyAndInterrupt = Deny(BROKEN)
        else:
            # [LAW:no-ambient-temporal-coupling] its hook's day runs from its hold, as Claude Code's clock on the hook runs
            # from its post, however long the read below takes.
            by = asyncio.get_running_loop().time() + BRAIN_REPLY_BY_SECONDS
            try:
                # [LAW:no-ambient-temporal-coupling] owned by the turn the model says goes by its id once Claude Code is
                # read through now, as a turn's hook is decided (Reader.hooked): a turn dequeued under a new id asks before a
                # period would have read where it opened. A read that breaks is no turn's.
                await self._reader.read()
                turn = self._owner(hook.prompt)
                await self._decide(hook, turn, replying, by)
            except Exception as error:
                raised = error
                # A turn's ends with it, so the model refuses this dialog with the turn's ending; one no turn owns is
                # refused here. NotWaiting once it is.
                self._broke(turn, error)
                self._sessions.answer(hook.asked.request, Deny(BROKEN))
            reply = refused(await asyncio.shield(replying))
        # Timed from its hold, and a part of the turn it was held for, which is known only once Claude Code is read through.
        ended("brain.permission", self._record, replace(began, span=within(self._part_of(turn))), raised, prompt=hook.prompt, tool=hook.tool, request=hook.asked.request, decision=reply)
        return reply

    async def _decide(self, hook: PermissionHook, turn: _Turn | None, replying: "asyncio.Future[HookReply]", by: float) -> None:
        """What hands decides of a held dialog, each settled through the model: put to the user in the turn it is a part
        of, or refused because nobody can be asked, its turn broke, nobody sees a dialog of questions or a plan, or nobody
        answered by `by`, the end of its hook's day on the loop's clock. A dialog the model already refused, of a turn
        hands is ending, is never put to the user."""
        request = hook.asked.request
        match hook.asked.on:
            case Permission() if turn is not None and turn.broken is not None:
                # A turn hands broke is ending, its stop under way, whether or not a prompt cleared the model's ending first.
                self._sessions.answer(request, Deny(BROKEN))
            case Permission() as permission if turn is not None and holds(self._modelled(), request):
                asked = Asked(permission, request, replying, self._sessions, self.session)
                turn.asks(asked)
                await asyncio.wait({replying}, timeout=by - asyncio.get_running_loop().time())
                asked.settle(Deny(UNANSWERED))
            case Permission() if turn is not None:
                # Refused as it was held: its turn is ending.
                pass
            case Permission():
                self._sessions.answer(request, Deny(NOBODY))
            case _:
                # A dialog of questions or a plan is never put to the user by voice: the brain's own words ask them.
                self._sessions.answer(request, Deny(UNVOICED))

    def _part_of(self, owner: _Turn | None) -> Span:
        """The span of what a dialog the brain posted is a part of: its `owner`, the turn in flight it was posted for, or
        else the brain's launch, for one a turn before it left behind or one posted between turns."""
        return self._launched if owner is None else owner.began.span

    @contextlib.contextmanager
    def _dialog(self, event: str, prompt: str | None, owner: _Turn | None) -> Generator[None]:
        """A dialog the brain posted, as one unit of work `event`: a part of its `owner`, the turn in flight it was posted
        for, or else of the brain's launch, for one a turn before it left behind or one posted between turns."""
        with continuing(self._part_of(owner)), unit(event, self._record):
            annotate(prompt=prompt)
            yield

    def _owner(self, prompt: str | None) -> _Turn | None:
        """The turn in flight a dialog posted for `prompt` is a part of: the turn taken, while Claude Code's turn in the
        model goes by that id."""
        turn = self._turn
        if turn is None or turn.answered.done() or not turn.taken.done() or prompt is None:
            return None
        # [LAW:single-enforcer] the model's turn, as its telling is: one a turn before it left behind is nobody's to answer.
        held = self._modelled()
        return turn if held is not None and PromptId(prompt) in ids(held.turn) else None

    def _refused(self, body: bytes) -> None:
        """An MCP server's ask for input, answered no in a turn or between turns, where it has no prompt id: one event, so
        the refusal is not only the brain's to tell, a part of the turn it was posted in or else of the brain's launch."""
        try:
            said = Payload.parse(body)
            prompt, server = said.optional_text("prompt_id"), said.optional_text("mcp_server_name")
        except Rejected as error:
            with unit("brain.elicitation", self._record):
                fail(f"hands could not read the Elicitation hook: {error}")
            return
        with self._dialog("brain.elicitation", prompt, self._owner(prompt)):
            annotate(server=server)

    def _apply(self, event: Event) -> "asyncio.Future[None]":
        """What the brain's hooks, status and transcript say, decided now, as Sessions.apply decides it; done once it is
        applied and heard (`_heard`)."""
        return asyncio.ensure_future(self._applied(self._sessions.apply(event)))

    async def _applied(self, applying: "asyncio.Future[None]") -> None:
        await applying
        await self._heard()

    def _took(self, submitted: Submitted) -> None:
        """Where Claude Code filed the words its prompt hook took, heard in the model's order, so every telling decided
        before it is heard first, and none decided after. Words that are no turn's open one of Claude Code's own, of the
        kind they are; words filed under a turn it already had run in it or behind it.

        [LAW:single-enforcer] a turn is taken by the prompt Claude Code took that is what it typed, by the words, whatever
        turn it was filed under."""
        turn = self._turn
        ours = turn is not None and not turn.answered.done() and not turn.taken.done() and kept(submitted.words) == kept(turn.typed)
        if not submitted.queued:
            running = self._claude_turns.opened(submitted.prompt, TYPED if ours else opener(submitted.words), ours)
        elif ours:
            # Filed under a turn Claude Code still runs, or under one told already, whose Stop hooks still ran: then
            # they run as the turn behind it, which no hook opens.
            running = self._claude_turns.filed(submitted.prompt) or self._claude_turns.behind(TYPED, awaited=True)
            running.awaited = True
        elif self._claude_turns.filed(submitted.prompt) is None:
            # Words of no turn's queued behind one already told, a task's result among them: they run as the turn behind it.
            self._claude_turns.behind(opener(submitted.words), awaited=False)
            return
        else:
            # Filed into a turn Claude Code still runs: they run in it, or behind it if its telling finds them waiting.
            return
        if turn is not None and ours:
            turn.taken_under, turn.queued, turn.running = submitted.prompt, submitted.queued, running
            turn.taken.set_result(None)

    def _told(self, told: Story) -> None:
        """A turn of Claude Code's the model told: the turn of `ClaudeTurns` it ends, and with it the turn in flight that
        turn answers, unless what that one typed still waits in Claude Code's queue as it is told: then it is answered by
        the turn they run as behind it. A turn stopped is over once Claude Code holds nothing of it (`_escape`)."""
        match told:
            case SessionGone():
                logger.warning(f"the brain's session {told.session} is gone, as the model reads it")
                return
            case Summarise():
                pass
        running = self._claude_turns.told(told)
        turn = self._turn
        theirs = turn is not None and running is not None and turn.running is running and not turn.answered.done()
        ours = theirs and turn is not None and any(kept(words) == kept(turn.typed) for words in told.waiting)
        # [LAW:one-source-of-truth] what still waits as a turn is told runs next, as the turn behind it, whoever's it is.
        behind = None if not told.waiting else self._claude_turns.behind(TYPED if ours else opener(told.waiting[0]), awaited=ours)
        if running is None:
            logger.info(f"the brain's Claude Code ended turn {told.turn}, which it was not heard opening")
            return
        if not theirs or turn is None:
            return
        if behind is not None and ours:
            running.awaited = False
            turn.running = behind
            return
        turn.told = told.turn
        if not turn.stopped:
            self._over(turn, BrainAnswered(told.turn, told.failure))

    async def _heard(self) -> None:
        """What the model applied has been heard: every telling it decided has reached `_told`, and a clear waiting on
        Claude Code to hold nothing of what it stopped reads the model again (`_told_out`).

        [LAW:one-source-of-truth] the registry is read once it is drained, so the model and the tellings the brain heard
        are one clock."""
        await self._sessions.drained(self.session)
        moved, self._moved = self._moved, asyncio.Event()
        moved.set()

    def _modelled(self) -> Session | None:
        """The brain's session as the model holds it now, or None before its first hook."""
        return self._sessions.live_session(self.session)

    def _all_told(self) -> bool:
        """Whether every turn Claude Code ran in the brain is told, and nothing waits in its queue: true before any is
        heard, since nothing runs in a Claude Code nobody typed into."""
        held = self._modelled()
        return held is None or all_told(held)

    def _over(self, turn: _Turn, outcome: BrainEnded | BaseException) -> None:
        if self._turn is turn:
            self._turn = None
        if turn.answered.done():
            return
        # Words of its waiting to run as the turn behind, which no telling has named, never run now: Ctrl-C cleared them
        # from the input as its stop went, or nothing is left to run them.
        if (running := turn.running) is not None and not running.ids and running in self._claude_turns.open:
            self._claude_turns.fail(running, Unran(f"the turn they were typed for ended ({outcome!r}) before they ran"))
        # A broken turn ends with what broke it, whatever ended it: its Stop, its stop's Escape, or the brain's exit.
        match outcome if turn.broken is None else turn.broken:
            case BrainAnswered() | BrainStopped() as ended:
                turn.answered.set_result(ended)
            case failed:
                turn.answered.set_exception(failed)
                # Said already, by the turn's own event or by `_broke`: not again by asyncio for an asker that stopped waiting.
                turn.answered.exception()


async def start(launch: Launch, record: Record, pause: Callable[[float], Awaitable[None]] = asyncio.sleep) -> Brain:
    """Start the brain under fritter on a terminal of hands' own, on the login its backend was parsed with; `pause` is how
    the time a typed turn has to be taken passes."""
    station = launch.station
    # [LAW:nothing-unseen] the launch is one event, from the spawn until the brain's input is up, or what failed it.
    with unit("brain.launch", record):
        annotate(session=launch.session, account=launch.account, model=station.model, config_dir=station.config_dir, cwd=station.cwd)
        match launch.conversation:
            case Fresh(untranscribed=untranscribed):
                annotate(conversation="fresh", untranscribed=untranscribed)
            case Resumed():
                annotate(conversation="resumed")
        # [LAW:one-source-of-truth] the fritter built with this hands, never a copy installed apart from it: what hands
        # asks of fritter and what the brain's fritter answers are one build's.
        try:
            fritter = packaged()
        except Unpackaged as error:
            raise Unstartable(str(error)) from error
        annotate(fritter=fritter)
        # Off the loop: a copy not hashed yet this process is read whole, about 230 MB.
        claude = await asyncio.to_thread(brain_claude, station.config_dir)
        mode = settings_mode(station.config_dir)
        annotate(claude=claude, mode=mode)
        # [LAW:no-ambient-temporal-coupling] the listener is up before the brain is spawned, and the brain exists only
        # once its input is up: a hook posted between waits for the brain, behind no other hook. No turn is typed in
        # that window, so only an MCP server's Elicitation can come.
        posting: asyncio.Future[Callable[[str, bytes], Awaitable[Mapping[str, object]]]] = asyncio.get_running_loop().create_future()
        listener, url = await _listen(posting)
        # A unix socket's path is capped near 104 bytes on macOS, so not under the brain's own directory.
        sockets = Path(tempfile.mkdtemp(prefix="hands-brain-"))
        running: ClaudeCode | None = None
        try:
            try:
                running = await spawn(station, [str(fritter), "--socket-dir", str(sockets), "--", *command(launch, claude, url, mode)])
                annotate(pid=running.pid)
                typist = await _typist(running, sockets, launch.session)
            except Unstartable:
                # A conversation the brain could not come up in never fails the starts after it: the next begins anew, and
                # Claude Code keeps its transcript (it exits "No conversation found" on one it cannot read).
                annotate(let_go=let_go(station.config_dir))
                raise
            # Held once the brain is up in it, so the next start resumes what this one says.
            hold(station.config_dir, launch.session)
        except BaseException:
            # [LAW:no-silent-failure] a start that fails or is cancelled leaves nothing running: the brain is in a session of
            # its own, which nothing but hands' own end would hang up.
            if running is not None:
                await running.stop()
            posting.cancel()
            await listener.cleanup()
            shutil.rmtree(sockets, ignore_errors=True)
            raise
        launched = here()
    brain = Brain(running, launch.session, typist, listener, sockets, station.config_dir, station.cwd, record, launched, pause)
    posting.set_result(brain.posted)
    return brain


async def _listen(posting: "asyncio.Future[Callable[[str, bytes], Awaitable[Mapping[str, object]]]]") -> tuple[web.AppRunner, str]:
    """The listener the brain posts its hooks to, on loopback, each answered with what the brain's `posted` settles for
    it, once there is a brain."""

    async def hook(request: web.Request) -> web.Response:
        # The event read off the path, so a body hands cannot read is still answered as its hook asks, by the brain,
        # whose record of the hook says it could not be read.
        event, body = request.match_info["event"], await request.read()
        posted = await asyncio.shield(posting)
        # Shielded: a handler aiohttp cancels leaves the hook to be applied and answered by the brain, which is never
        # refused a hook.
        return web.json_response(await asyncio.shield(asyncio.ensure_future(posted(event, body))))

    app = web.Application()
    app.router.add_post("/{event}", hook)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    host, port = runner.addresses[0][:2]
    return runner, f"http://{host}:{port}"


async def _typist(running: ClaudeCode, sockets: Path, session: SessionId) -> Typist:
    """The brain as fritter types into it, once its input is up: the socket fritter opened under `sockets`, and the claude
    it started.

    [LAW:no-ambient-temporal-coupling] up is the brain's own word, the bracketed paste Claude Code turns on once it reads
    its input, not a guess at how long that takes: a
    turn typed into a terminal not yet raw has its Return made a newline, which Claude Code takes as one more line."""
    deadline = asyncio.get_running_loop().time() + START_SECONDS
    waiting = "fritter had not started the brain and opened its socket"
    while asyncio.get_running_loop().time() < deadline and not running.exit.done():
        found = [path for path in sockets.rglob("*") if path.is_socket()]
        child = await asyncio.to_thread(_child_of, running.pid)
        if found and child is not None:
            typist = Typist(session, found[0], child)
            try:
                await asyncio.to_thread(typist.pasting)
                return typist
            except Untyped as error:
                waiting = f"the brain had not turned its input on ({error})"
        await asyncio.sleep(0.05)
    # What fritter's terminal showed says why: a fritter that could not be run at all fails there, in the shell it was run by.
    if running.exit.done():
        raise Unstartable(f"fritter exited ({running.exit.result()}) while {waiting}; it showed:\n{running.shown()}")
    raise Unstartable(f"{waiting} after {START_SECONDS:.0f}s; it showed:\n{running.shown()}")


def _child_of(pid: int) -> int | None:
    """The process `pid` started: fritter starts one, the claude it wraps."""
    found = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True).stdout.split()
    return int(found[0]) if found else None
