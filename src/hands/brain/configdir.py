"""Claude Code's config directory as the brain uses it, read from its files alone, starting no Claude Code: the pinned
claude, settings.json, the first screens .claude.json records, and the syncs settings.json keeps out; and the machine's
Claude Code policy, which every config directory on it is started under.

[LAW:parse-dont-validate] each file is parsed once into a value whose unreadable fields are variants of it, never an
exception that discards the fields parsed beside it, so every refusal is read off the values and none hides another.
"""

import os
import plistlib
import getpass
from dataclasses import dataclass
from pathlib import Path
from xml.parsers.expat import ExpatError

from hands.brain import pinned
from hands.core.session import PERMISSION_MODES, PermissionMode, PinnedClaude, Runnable, UnknownMode
from hands.sessions.files import Unread
from hands.sessions.payload import Payload, Rejected
from hands.sessions.untap import ROUTING


class Unstartable(Exception):
    """A Claude Code of hands' own could not be started or asked: no claude to run, a config directory it would not start on, an `auth status` that gave no answer, a fritter that never opened its socket, or a brain that never turned its input on."""


# Each is on unless settings.json says false: they sync the login's account's skills and plugins, which are Brandon's.
KEPT_OUT = ("syncClaudeAiSkills", "syncClaudeAiPlugins")

# The permission modes Claude Code takes as a setting's defaultMode: a hook's, and "manual", the other name it reads as one of them.
SETTINGS_MODES: dict[str, PermissionMode] = {**PERMISSION_MODES, "manual": "default"}

SETTINGS = "settings.json"
STATE = ".claude.json"


@dataclass(frozen=True)
class Machine:
    """Where Claude Code reads the machine's policy on macOS (docs/claude-code-behavior.md, machine policy): the folder
    of its managed settings and their drop-ins, and the managed preferences that hold its plists, the machine's and each
    account's; and the account the brain runs as, whose plist it reads beside the machine's, Unread where it cannot be
    told."""

    folder: Path
    preferences: Path
    account: str | Unread


_PLIST = "com.anthropic.claudecode.plist"


def this_machine() -> Machine:
    """[LAW:effects-at-boundaries] the Mac hands runs on, read once by a composition root and handed to every read of a
    config directory: macOS's policy locations, and the account this process runs as, as the brain does."""
    try:
        account: str | Unread = getpass.getuser()
    except (OSError, KeyError) as error:
        account = Unread(f"the account hands runs as cannot be told: {error}")
    return Machine(Path("/Library/Application Support/ClaudeCode"), Path("/Library/Managed Preferences"), account)


@dataclass(frozen=True)
class Env:
    """What the env of Claude Code's settings puts in the brain's environment that hands weighs, each read apart, so
    neither hides the other: the API key, by the last 20 characters Claude Code records its answer on, None for none;
    and each variable of ROUTING it sets, which Claude Code puts over the environment it is started in, so over hands'
    way to the API."""

    key: str | Unread | None
    routing: tuple[str, ...]


@dataclass(frozen=True)
class Settings:
    """The brain's settings.json: the switches of KEPT_OUT it leaves on, the permission mode it names, and its env. None
    is a mode it names none of."""

    synced: tuple[str, ...]
    mode: PermissionMode | UnknownMode | Unread | None
    env: Env | Unread


@dataclass(frozen=True)
class Policy:
    """One file of the machine's Claude Code policy, and its env: Claude Code puts a policy's env over the environment it
    is started in, as it does settings.json's (docs/claude-code-behavior.md, policy env). A policy is enterprise config
    almost no Mac has, and hands' start never rests on one: an env hands cannot read refuses nothing, and is named on the
    start's event (unread_policy)."""

    path: Path
    env: Env | Unread


@dataclass(frozen=True)
class Absent:
    """A file that is not there."""


@dataclass(frozen=True)
class State:
    """What Claude Code's own .claude.json records of its first screens: its onboarding finished, the directory the brain
    runs in, or one above it, trusted, and the API keys it was asked about, by their last 20 characters."""

    onboarded: bool | Unread
    trusted: bool | Unread
    answered: frozenset[str] | Unread


@dataclass(frozen=True)
class ConfigDir:
    """The brain's config directory, every file of it read, and the directory the brain runs in, as Claude Code records
    a directory it trusts: resolved."""

    path: Path
    cwd: Path
    claude: PinnedClaude | pinned.Unpinned
    settings: Settings | Unread
    state: State | Absent | Unread
    # Each file of the machine's policy that is there.
    policy: tuple[Policy, ...]


def cwd(config_dir: Path) -> Path:
    """The directory the brain on `config_dir` runs in: never a project."""
    return config_dir / "cwd"


def read(config_dir: Path, pin: pinned.Pin, machine: Machine) -> ConfigDir:
    """Every file of `config_dir` the brain starts on, read, its claude weighed against `pin`, and `machine`'s policy;
    never raises on what they hold. Hashes a pinned claude this process has not hashed since it changed, about
    230 MB, so it is called off the loop."""
    try:
        claude: PinnedClaude | pinned.Unpinned = pin.verified(config_dir)
    except pinned.Unpinned as error:
        claude = error
    # realpath, not resolve: a link loop is a directory Claude Code trusts none of, never a raise.
    here = Path(os.path.realpath(cwd(config_dir)))
    return ConfigDir(config_dir, here, claude, _settings(config_dir / SETTINGS), _state(config_dir / STATE, here), _policies(machine))


def _settings(path: Path) -> Settings | Unread:
    try:
        said = Payload.parse(path.read_bytes())
    except FileNotFoundError:
        # Left out, each switch is on, and the brain names no mode and sets no env.
        return Settings(KEPT_OUT, None, Env(None, ()))
    except (OSError, Rejected) as error:
        return Unread(str(error))
    return Settings(tuple(switch for switch in KEPT_OUT if said.fields.get(switch) is not False), _mode(said), _env(said))


def _mode(said: Payload) -> PermissionMode | UnknownMode | Unread | None:
    try:
        named = Payload.of(said.fields.get("permissions", {}), "its permissions").optional_text("defaultMode")
    except Rejected as error:
        return Unread(str(error))
    if named is None:
        return None
    return SETTINGS_MODES.get(named) or UnknownMode(named)


def _env(said: Payload) -> Env | Unread:
    """[LAW:single-enforcer] the one reading of the env of Claude Code's settings, settings.json's and each policy's:
    what it sets that hands weighs. Its API key reaches the brain where the brain's own is set: hands' never does
    (hands.brain.process.FOREIGN_LOGINS)."""
    try:
        env = Payload.of(said.fields.get("env", {}), "its env")
    except Rejected as error:
        return Unread(str(error))
    return Env(_key(env), tuple(name for name in ROUTING if name in env.fields))


def _key(env: Payload) -> str | Unread | None:
    try:
        key = env.optional_text("ANTHROPIC_API_KEY")
    except Rejected as error:
        return Unread(str(error))
    # As Claude Code reads it (docs/claude-code-behavior.md): a blank key is no key, and it is answered by its last 20
    # characters once it is trimmed.
    return key.strip()[-20:] if key else None


def _policies(machine: Machine) -> tuple[Policy, ...]:
    # Claude Code reads every one of them, the plist of the account the brain runs as among them; a variable any of them
    # sets is put over hands' way to the API.
    files = (machine.folder / "managed-settings.json", *sorted((machine.folder / "managed-settings.d").glob("*.json")), machine.preferences / _PLIST)
    return (*(policy for file in files for policy in _policy(file)), *_account_policy(machine))


def _account_policy(machine: Machine) -> tuple[Policy, ...]:
    match machine.account:
        case Unread(reason=reason):
            # [LAW:no-silent-failure] an account that cannot be told is a plist that cannot be read.
            return (Policy(machine.preferences, Unread(f"the plist of the account the brain runs as is under it, and {reason}")),)
        case str() as account:
            return _policy(machine.preferences / account / _PLIST)


def _policy(path: Path) -> tuple[Policy, ...]:
    """The policy at `path`, or none where there is no file."""
    try:
        held = path.read_bytes()
    except FileNotFoundError:
        return ()
    except OSError as error:
        return (Policy(path, Unread(str(error))),)
    try:
        said = Payload.of(plistlib.loads(held), "the plist") if path.suffix == ".plist" else Payload.parse(held)
    except (Rejected, ValueError, ExpatError) as error:
        return (Policy(path, Unread(str(error))),)
    return (Policy(path, _env(said)),)


def unread_policy(config: ConfigDir) -> tuple[str, ...]:
    """Each file of the machine's policy whose env hands cannot read, with why: it refuses nothing, so the start's event
    names it, the one trace of a file Claude Code may stop on at a Settings Error dialog (docs/claude-code-behavior.md,
    unreadable machine policy)."""
    return tuple(f"{policy.path} ({policy.env.reason})" for policy in config.policy if isinstance(policy.env, Unread))


def _state(path: Path, cwd: Path) -> State | Absent | Unread:
    try:
        said = Payload.parse(path.read_bytes())
    except FileNotFoundError:
        return Absent()
    except (OSError, Rejected) as error:
        return Unread(str(error))
    return State(_flag(said, "hasCompletedOnboarding"), _trusted(said, cwd), _answered(said))


def _flag(said: Payload, key: str) -> bool | Unread:
    try:
        return said.optional_flag(key)
    except Rejected as error:
        return Unread(str(error))


def _trusted(said: Payload, cwd: Path) -> bool | Unread:
    try:
        projects = Payload.of(said.fields.get("projects", {}), "its projects").fields
        return any(Payload.of(projects[place], place).optional_flag("hasTrustDialogAccepted") for place in map(str, (cwd, *cwd.parents)) if place in projects)
    except Rejected as error:
        return Unread(str(error))


def _answered(said: Payload) -> frozenset[str] | Unread:
    try:
        responses = Payload.of(said.fields.get("customApiKeyResponses", {}), "its API key answers")
        return frozenset(key for key in (*responses.optional_items("approved"), *responses.optional_items("rejected")) if isinstance(key, str))
    except Rejected as error:
        return Unread(str(error))


def runnable(config: ConfigDir) -> Runnable | tuple[str, ...]:
    """What a Claude Code of hands' own is run with on `config`, or every reason none may run on it, each naming its fix:
    no pinned claude, or settings it must not start on, which a run of one, `hands login`'s among them, would load."""
    # [LAW:dataflow-not-control-flow] every file is weighed every time: one refused hides none after it.
    refused = (*_pinned(config.claude), *_kept_out(config))
    match config.claude, config.settings:
        case PinnedClaude() as claude, Settings(mode=str() | None as mode) if not refused:
            return Runnable(claude, mode)
        case _:
            return refused


def around_proxy(config: ConfigDir) -> tuple[str, ...]:
    """Why a brain on `config` started through hands' proxy would reach the API some other way: each variable of ROUTING
    its settings' env or the machine's policy sets, which Claude Code puts over the environment hands starts it in, each
    naming its fix. None for a Claude Code that reaches the API directly, `hands login`'s, whose environment no proxy of
    hands' is in for these to override."""
    match config.settings:
        case Settings(env=env):
            own: tuple[tuple[Env | Unread, Path], ...] = ((env, config.path / SETTINGS),)
        case Unread():
            # Refused by runnable: what settings that cannot be read set is not known.
            own = ()
    return tuple(refusal for env, path in (*own, *((policy.env, policy.path) for policy in config.policy)) for refusal in _routed(env, path))


def _routed(env: Env | Unread, path: Path) -> tuple[str, ...]:
    """[LAW:single-enforcer] the one refusal of an env that routes the brain: settings.json's and each policy's."""
    match env:
        case Env(routing=()) | Unread():
            # settings.json's is refused by runnable; a policy's refuses nothing (Policy).
            return ()
        case Env(routing=routing):
            said = "; ".join(f"{name} {ROUTING[name]}" for name in routing)
            return (f"{path} sets {', '.join(routing)} in its env, which Claude Code puts over the environment hands starts the brain in: {said}: remove them from it",)


def _pinned(claude: PinnedClaude | pinned.Unpinned) -> tuple[str, ...]:
    match claude:
        case pinned.Unpinned():
            return (str(claude),)
        case PinnedClaude():
            return ()


def _kept_out(config: ConfigDir) -> tuple[str, ...]:
    """What settings.json would have the brain start on that it must not: a sync not kept out, a mode Claude Code does
    not take, which would put the settings beside it in doubt, or an env hands cannot read."""
    path = config.path / SETTINGS
    match config.settings:
        case Unread(reason=reason):
            return (f"the brain's settings could not be read ({reason}): correct {path}",)
        case Settings(synced=synced, mode=mode, env=env):
            pass
    kept = " and ".join(f'"{switch}": false' for switch in KEPT_OUT)
    return (
        *((f"the brain would load its account's {' and '.join(synced)}: set {kept} in {path}",) if synced else ()),
        *_moded(mode, path),
        *_unweighed(env, path),
    )


def _unweighed(env: Env | Unread, path: Path) -> tuple[str, ...]:
    """An env of settings.json hands cannot weigh: Claude Code would start on what it sets, unseen, whatever way it reaches
    the API."""
    match env:
        case Unread(reason=reason):
            return (f"hands cannot read the env {path} gives Claude Code ({reason}): correct it",)
        case Env(key=Unread(reason=reason)):
            return (f"hands cannot read the API key {path} sets in its env ({reason}): correct it",)
        case Env():
            return ()


def _moded(mode: PermissionMode | UnknownMode | Unread | None, path: Path) -> tuple[str, ...]:
    named = f"name one of {', '.join(SETTINGS_MODES)} in {path}"
    match mode:
        case UnknownMode(name=name):
            return (f'the brain\'s settings name a defaultMode Claude Code does not take, "{name}": {named}',)
        case Unread(reason=reason):
            return (f"the brain's settings name no defaultMode hands can read ({reason}): {named}",)
        case str() | None:
            return ()


def unanswered(config: ConfigDir) -> tuple[str, ...]:
    """Why the brain would start on one of Claude Code's first screens: none when it would not. A config directory made
    by anything but Claude Code's first run, `claude auth status` among them, records none of them."""
    state = config.path / STATE
    match config.state:
        case Absent():
            return (f"no {state}",)
        case Unread() as unread:
            # Claude Code's own first run is what writes this state, so it is the run that mends one hands cannot read.
            return (_unreadable(state, unread),)
        case State(onboarded=onboarded, trusted=trusted, answered=answered):
            pass
    return (
        *_screen(onboarded, state, "its onboarding unfinished"),
        *_screen(trusted, state, f"{config.cwd} untrusted"),
        *_keyed(answered, config.settings, state, config.path / SETTINGS),
    )


def _unreadable(state: Path, unread: Unread) -> str:
    return f"{state} unreadable: {unread.reason}"


def _screen(passed: bool | Unread, state: Path, why: str) -> tuple[str, ...]:
    match passed:
        case Unread() as unread:
            return (_unreadable(state, unread),)
        case bool():
            return () if passed else (why,)


def _keyed(answered: frozenset[str] | Unread, settings: Settings | Unread, state: Path, path: Path) -> tuple[str, ...]:
    match answered, settings:
        case Unread() as unread, _:
            return (_unreadable(state, unread),)
        case frozenset(), Settings(env=Env(key=str() as key)):
            return () if key in answered else (f"the API key {path} sets unanswered",)
        case _:
            # No key, or settings whose key cannot be read, which is their own refusal, said by runnable: whether it was
            # answered is told once they are mended.
            return ()


def first_screens(config: ConfigDir) -> tuple[str, ...]:
    """Why the brain would not start on `config` past Claude Code's first screens, naming `hands login`, which answers them."""
    why = unanswered(config)
    return (f"the brain has not been through Claude Code's first screens ({'; '.join(why)}): `hands login` answers them",) if why else ()
