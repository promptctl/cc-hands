"""Claude Code's config directory as the brain uses it, read from its files alone, starting no Claude Code: the pinned
claude, settings.json, the first screens .claude.json records, and the syncs settings.json keeps out.

[LAW:parse-dont-validate] each file is parsed once into a value whose unreadable fields are variants of it, never an
exception that discards the fields parsed beside it, so every refusal is read off the values and none hides another.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from hands.brain import pinned
from hands.core.session import PERMISSION_MODES, PermissionMode
from hands.sessions.payload import Payload, Rejected


class Unstartable(Exception):
    """A Claude Code of hands' own could not be started: no claude to run, a config directory it would not start on, a fritter that never opened its socket, or a brain that never turned its input on."""


def brain_claude(config_dir: Path) -> Path:
    """The Claude Code hands runs as its own on `config_dir`: hands' copy of the release it is pinned to
    (hands.brain.pinned), never the person's claude, whose release their own install moves under it."""
    try:
        return pinned.verified(config_dir)
    except pinned.Unpinned as error:
        raise Unstartable(str(error)) from None


@dataclass(frozen=True)
class Unread:
    """A file, or a field of one, that could not be read: why."""

    reason: str


# Each is on unless settings.json says false: they sync the login's account's skills and plugins, which are Brandon's.
KEPT_OUT = ("syncClaudeAiSkills", "syncClaudeAiPlugins")

# The permission modes Claude Code takes as a setting's defaultMode: a hook's, and "manual", the other name it reads as one of them.
SETTINGS_MODES: Mapping[str, PermissionMode] = {**PERMISSION_MODES, "manual": "default"}


@dataclass(frozen=True)
class Untaken:
    """A defaultMode Claude Code does not take, by the name settings.json gives it."""

    named: str


@dataclass(frozen=True)
class Settings:
    """The brain's settings.json: the switches of KEPT_OUT it leaves on, the permission mode it names, and the API key
    it puts in the brain's environment, by the last 20 characters Claude Code records its answer on. None is a mode or
    a key it names none of."""

    synced: tuple[str, ...]
    mode: PermissionMode | Untaken | Unread | None
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
    """The brain's config directory, every file of it read."""

    path: Path
    claude: Path | Unread
    settings: Settings | Unread
    state: State | Absent | Unread

    @property
    def settings_path(self) -> Path:
        return self.path / "settings.json"

    @property
    def state_path(self) -> Path:
        return self.path / ".claude.json"


def cwd(config_dir: Path) -> Path:
    """The directory the brain on `config_dir` runs in: never a project."""
    return config_dir / "cwd"


def read(config_dir: Path) -> ConfigDir:
    """Every file of `config_dir` the brain starts on, read; never raises on what they hold. Hashes a pinned claude this
    process has not hashed since it changed, about 230 MB, so it is called off the loop."""
    try:
        claude: Path | Unread = pinned.verified(config_dir)
    except pinned.Unpinned as error:
        claude = Unread(str(error))
    return ConfigDir(config_dir, claude, _settings(config_dir / "settings.json"), _state(config_dir / ".claude.json", cwd(config_dir).resolve()))


def _settings(path: Path) -> Settings | Unread:
    try:
        said = Payload.parse(path.read_bytes())
    except FileNotFoundError:
        # Left out, each switch is on, and the brain names no mode and no key.
        return Settings(KEPT_OUT, None, None)
    except (OSError, Rejected) as error:
        return Unread(str(error))
    return Settings(tuple(switch for switch in KEPT_OUT if said.fields.get(switch) is not False), _mode(said), _key(said))


def _mode(said: Payload) -> PermissionMode | Untaken | Unread | None:
    try:
        named = Payload.of(said.fields.get("permissions", {}), "its permissions").optional_text("defaultMode")
    except Rejected as error:
        return Unread(str(error))
    if named is None:
        return None
    return SETTINGS_MODES.get(named) or Untaken(named)


def _key(said: Payload) -> str | Unread | None:
    """The API key settings.json puts in the brain's environment, where the brain's own is set: hands' never reaches it
    (hands.brain.process.FOREIGN_LOGINS)."""
    try:
        key = Payload.of(said.fields.get("env", {}), "its env").optional_text("ANTHROPIC_API_KEY")
    except Rejected as error:
        return Unread(str(error))
    # A blank key is no key: Claude Code asks of none.
    return key[-20:] if key else None


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


def unanswered(config: ConfigDir) -> tuple[str, ...]:
    """Why the brain would start on one of Claude Code's first screens: none when it would not. A config directory made
    by anything but Claude Code's first run, `claude auth status` among them, records none of them."""
    state = config.state_path
    match config.state:
        case Absent():
            return (f"no {state}",)
        case Unread(reason=reason):
            # Claude Code's own first run is what writes this state, so it is the run that mends one hands cannot read.
            return (f"{state} unreadable: {reason}",)
        case State(onboarded=onboarded, trusted=trusted, answered=answered):
            pass
    # An unreadable key is the settings' refusal, said apart from these.
    key = config.settings.key if isinstance(config.settings, Settings) else None
    return (
        *_screen(onboarded, state, "its onboarding unfinished"),
        *_screen(trusted, state, f"{cwd(config.path).resolve()} untrusted"),
        *_keyed(answered, key, state, config.settings_path),
    )


def _screen(passed: bool | Unread, state: Path, why: str) -> tuple[str, ...]:
    match passed:
        case Unread(reason=reason):
            return (f"{state} unreadable: {reason}",)
        case bool():
            return () if passed else (why,)


def _keyed(answered: frozenset[str] | Unread, key: str | Unread | None, state: Path, settings: Path) -> tuple[str, ...]:
    match answered, key:
        case Unread(reason=reason), str():
            return (f"{state} unreadable: {reason}",)
        case frozenset(), str() if key not in answered:
            return (f"the API key {settings} sets unanswered",)
        case _:
            return ()


def refusals(config: ConfigDir) -> tuple[str, ...]:
    """Every reason the brain would not start on `config`, each naming its fix: none when it would."""
    # [LAW:dataflow-not-control-flow] every file is weighed every time: one refused hides none after it.
    return (*_pinned(config.claude), *kept_out(config), *first_screens(config))


def _pinned(claude: Path | Unread) -> tuple[str, ...]:
    return (claude.reason,) if isinstance(claude, Unread) else ()


def kept_out(config: ConfigDir) -> tuple[str, ...]:
    """What settings.json would have the brain start on that it must not: a sync not kept out, or a mode Claude Code does not take."""
    path = config.settings_path
    match config.settings:
        case Unread(reason=reason):
            return (_unreadable(path, reason),)
        case Settings(synced=synced, mode=mode, key=key):
            pass
    kept = f'set "syncClaudeAiSkills": false and "syncClaudeAiPlugins": false in {path}'
    # [LAW:parse-dont-validate] said again in the brain's --settings, beside its hooks: a mode Claude Code does not take
    # would put the settings holding them in doubt, so the brain does not start on one.
    return (
        *((f"the brain would load its account's {' and '.join(synced)}: {kept}",) if synced else ()),
        *_moded(mode, path),
        *((f"the brain's settings set an env hands cannot read ({key.reason}): correct {path}",) if isinstance(key, Unread) else ()),
    )


def _unreadable(path: Path, reason: str) -> str:
    return f"the brain's settings could not be read ({reason}): correct {path}"


def _moded(mode: PermissionMode | Untaken | Unread | None, path: Path) -> tuple[str, ...]:
    match mode:
        case Untaken(named=named):
            return (f'the brain\'s settings name a defaultMode Claude Code does not take, "{named}": name one of {", ".join(SETTINGS_MODES)} in {path}',)
        case Unread(reason=reason):
            return (f"the brain's settings name no defaultMode hands can read ({reason}): name one of {', '.join(SETTINGS_MODES)} in {path}",)
        case _:
            return ()


def first_screens(config: ConfigDir) -> tuple[str, ...]:
    why = unanswered(config)
    return (f"the brain has not been through Claude Code's first screens ({'; '.join(why)}): `hands login` answers them",) if why else ()


def settings_mode(config_dir: Path) -> PermissionMode | None:
    """The permission mode the brain's settings.json names, None where it names none; raises Unstartable where it
    names one Claude Code does not take, or hands cannot read it."""
    path = config_dir / "settings.json"
    match _settings(path):
        case Unread(reason=reason):
            raise Unstartable(_unreadable(path, reason))
        case Settings(mode=str() | None as mode):
            return mode
        case Settings(mode=mode):
            raise Unstartable("; ".join(_moded(mode, path)))
