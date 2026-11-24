"""Claude Code's config directory as the brain uses it, read from its files alone, starting no Claude Code: the pinned
claude, settings.json, the first screens .claude.json records, and the syncs settings.json keeps out.

[LAW:parse-dont-validate] each file is parsed once into a value whose unreadable fields are variants of it, never an
exception that discards the fields parsed beside it, so every refusal is read off the values and none hides another.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from hands.brain import pinned
from hands.core.session import PERMISSION_MODES, PermissionMode, PinnedClaude, Runnable, UnknownMode
from hands.sessions.files import Unread
from hands.sessions.payload import Payload, Rejected


class Unstartable(Exception):
    """A Claude Code of hands' own could not be started: no claude to run, a config directory it would not start on, a fritter that never opened its socket, or a brain that never turned its input on."""


# Each is on unless settings.json says false: they sync the login's account's skills and plugins, which are Brandon's.
KEPT_OUT = ("syncClaudeAiSkills", "syncClaudeAiPlugins")

# The permission modes Claude Code takes as a setting's defaultMode: a hook's, and "manual", the other name it reads as one of them.
SETTINGS_MODES: dict[str, PermissionMode] = {**PERMISSION_MODES, "manual": "default"}

SETTINGS = "settings.json"
STATE = ".claude.json"


@dataclass(frozen=True)
class Settings:
    """The brain's settings.json: the switches of KEPT_OUT it leaves on, the permission mode it names, and the API key
    it puts in the brain's environment, by the last 20 characters Claude Code records its answer on. None is a mode or
    a key it names none of."""

    synced: tuple[str, ...]
    mode: PermissionMode | UnknownMode | Unread | None
    key: str | Unread | None


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


def cwd(config_dir: Path) -> Path:
    """The directory the brain on `config_dir` runs in: never a project."""
    return config_dir / "cwd"


def read(config_dir: Path, pin: pinned.Pin) -> ConfigDir:
    """Every file of `config_dir` the brain starts on, read, its claude weighed against `pin`; never raises on what they
    hold. Hashes a pinned claude this process has not hashed since it changed, about 230 MB, so it is called off the loop."""
    try:
        claude: PinnedClaude | pinned.Unpinned = pin.verified(config_dir)
    except pinned.Unpinned as error:
        claude = error
    # realpath, not resolve: a link loop is a directory Claude Code trusts none of, never a raise.
    here = Path(os.path.realpath(cwd(config_dir)))
    return ConfigDir(config_dir, here, claude, _settings(config_dir / SETTINGS), _state(config_dir / STATE, here))


def _settings(path: Path) -> Settings | Unread:
    try:
        said = Payload.parse(path.read_bytes())
    except FileNotFoundError:
        # Left out, each switch is on, and the brain names no mode and no key.
        return Settings(KEPT_OUT, None, None)
    except (OSError, Rejected) as error:
        return Unread(str(error))
    return Settings(tuple(switch for switch in KEPT_OUT if said.fields.get(switch) is not False), _mode(said), _key(said))


def _mode(said: Payload) -> PermissionMode | UnknownMode | Unread | None:
    try:
        named = Payload.of(said.fields.get("permissions", {}), "its permissions").optional_text("defaultMode")
    except Rejected as error:
        return Unread(str(error))
    if named is None:
        return None
    return SETTINGS_MODES.get(named) or UnknownMode(named)


def _key(said: Payload) -> str | Unread | None:
    """The API key settings.json puts in the brain's environment, where the brain's own is set: hands' never reaches it
    (hands.brain.process.FOREIGN_LOGINS)."""
    try:
        key = Payload.of(said.fields.get("env", {}), "its env").optional_text("ANTHROPIC_API_KEY")
    except Rejected as error:
        return Unread(str(error))
    # As Claude Code reads it (docs/claude-code-behavior.md): a blank key is no key, and it is answered by its last 20
    # characters once it is trimmed.
    return key.strip()[-20:] if key else None


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
        case Settings(synced=synced, mode=mode, key=key):
            pass
    kept = " and ".join(f'"{switch}": false' for switch in KEPT_OUT)
    return (
        *((f"the brain would load its account's {' and '.join(synced)}: set {kept} in {path}",) if synced else ()),
        *_moded(mode, path),
        *((f"the brain's settings set an env hands cannot read ({key.reason}): correct {path}",) if isinstance(key, Unread) else ()),
    )


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
    key = config.settings.key if isinstance(config.settings, Settings) else config.settings
    return (
        *_screen(onboarded, state, "its onboarding unfinished"),
        *_screen(trusted, state, f"{config.cwd} untrusted"),
        *_keyed(answered, key, state, config.path / SETTINGS),
    )


def _unreadable(state: Path, unread: Unread) -> str:
    return f"{state} unreadable: {unread.reason}"


def _screen(passed: bool | Unread, state: Path, why: str) -> tuple[str, ...]:
    match passed:
        case Unread() as unread:
            return (_unreadable(state, unread),)
        case bool():
            return () if passed else (why,)


def _keyed(answered: frozenset[str] | Unread, key: str | Unread | None, state: Path, settings: Path) -> tuple[str, ...]:
    match answered, key:
        case Unread() as unread, _:
            return (_unreadable(state, unread),)
        case frozenset(), str():
            return () if key in answered else (f"the API key {settings} sets unanswered",)
        case frozenset(), None:
            return ()
        case frozenset(), Unread():
            # A key that cannot be read is the settings' own refusal, said by runnable: whether it was answered is told
            # once they are mended.
            return ()


def first_screens(config: ConfigDir) -> tuple[str, ...]:
    """Why the brain would not start on `config` past Claude Code's first screens, naming `hands login`, which answers them."""
    why = unanswered(config)
    return (f"the brain has not been through Claude Code's first screens ({'; '.join(why)}): `hands login` answers them",) if why else ()
