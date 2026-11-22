"""The brain's config directory, as Claude Code reads it: the claude hands runs on it, the first screens its .claude.json
records as answered, and the settings its settings.json holds. Each is read from its files alone, so asking whether a
Claude Code could start on it starts none; `starting_settings`, which `hands login` runs, is the one writer here.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from hands.brain import pinned
from hands.core.session import PERMISSION_MODES, PermissionMode
from hands.sessions.files import replace_whole
from hands.sessions.payload import Payload, Rejected


class Unstartable(Exception):
    """A Claude Code of hands' own could not be started: no claude to run, or a fritter that never opened its socket, or a brain that never turned its input on."""


def brain_claude(config_dir: Path) -> Path:
    """The Claude Code hands runs as its own on `config_dir`: hands' copy of the release it is pinned to
    (hands.brain.pinned), never the person's claude, whose release their own install moves under it."""
    try:
        return pinned.verified(config_dir)
    except pinned.Unpinned as error:
        raise Unstartable(str(error)) from None


def run_directory(config_dir: Path) -> Path:
    """The empty directory the brain runs in: never a project."""
    return config_dir / "cwd"


# The permission modes Claude Code takes as a setting's defaultMode: the ones a session's hooks report, and the other
# name it reads as one of them.
MODES: dict[str, PermissionMode] = {**PERMISSION_MODES, "manual": "default"}

# Each is on unless settings.json says false: they sync the login's account's skills and plugins, which are Brandon's.
KEPT_OUT = ("syncClaudeAiSkills", "syncClaudeAiPlugins")


@dataclass(frozen=True)
class BrainSettings:
    """The brain's settings.json, as far as hands reads it."""

    config_dir: Path
    # The permission mode it names, or None when it names none.
    mode: PermissionMode | None
    # The switches of KEPT_OUT it leaves on.
    synced: tuple[str, ...]
    # The last 20 characters of the API key its env puts in the brain's environment, by which Claude Code records its
    # answer on the key, where its env sets one: hands' never reaches it (FOREIGN_LOGINS). Never the key itself.
    key_suffix: str | None

    @property
    def path(self) -> Path:
        return self.config_dir / "settings.json"


def settings(config_dir: Path) -> BrainSettings:
    """The brain's settings.json, read once; raises Unstartable where it cannot be read, or names a mode Claude Code does
    not take. A file left out is Claude Code's defaults, so each switch of KEPT_OUT is on."""
    path = config_dir / "settings.json"
    try:
        said = Payload.parse(path.read_bytes())
        named = Payload.of(said.fields.get("permissions", {}), "its permissions").optional_text("defaultMode")
        api_key = Payload.of(said.fields.get("env", {}), "its env").optional_text("ANTHROPIC_API_KEY")
    except FileNotFoundError:
        return BrainSettings(config_dir, None, KEPT_OUT, None)
    except (OSError, Rejected) as error:
        raise Unstartable(f"the brain's settings could not be read ({error}): correct {path}") from None
    # [LAW:parse-dont-validate] said again in the brain's --settings, beside its hooks: a mode Claude Code does not take
    # would put the settings holding them in doubt, so the brain does not start on one.
    if named is not None and named not in MODES:
        raise Unstartable(f'the brain\'s settings name a defaultMode Claude Code does not take, "{named}": name one of {", ".join(MODES)} in {path}')
    return BrainSettings(config_dir, None if named is None else MODES[named], tuple(switch for switch in KEPT_OUT if said.fields.get(switch) is not False), api_key[-20:] if api_key else None)


def starting_settings(config_dir: Path) -> bool:
    """Writes the settings.json a brain home starts with when it has none; whether it wrote one. One that is there is
    left as it is: the brain's settings are its directory's to say. Run by `hands login`, never by a check."""
    # [LAW:one-source-of-truth] what account_kept_out requires, and the mode the README says the brain runs in.
    mode: PermissionMode = "default"
    starting = {**dict.fromkeys(KEPT_OUT, False), "permissions": {"defaultMode": mode}}
    staged = config_dir / ".settings.json.starting"
    # [LAW:no-ambient-temporal-coupling] written whole beside it, then linked in only where there is none, so a login
    # stopped part way leaves either no settings.json or this one, never half of it.
    replace_whole(staged, json.dumps(starting, indent=2), 0o644)
    try:
        (config_dir / "settings.json").hardlink_to(staged)
    except FileExistsError:
        return False
    finally:
        staged.unlink()
    return True


def account_kept_out(held: BrainSettings) -> None:
    """Raises Unstartable unless the brain's settings keep out the skills and plugins its login's account syncs.

    They are Brandon's, never the brain's, and settings.json is the only place Claude Code reads either switch from:
    not --settings, and not the environment. Left out, each is on."""
    if held.synced:
        raise Unstartable(f'the brain would load its account\'s {" and ".join(held.synced)}: set "syncClaudeAiSkills": false and "syncClaudeAiPlugins": false in {held.path}')


def unanswered(config_dir: Path, key_suffix: str | None) -> tuple[str, ...]:
    """Why the brain would start on one of Claude Code's first screens, every one of them: Claude Code records in its own
    .claude.json that its onboarding finished, that the directory the brain runs in, or one above it, is trusted, and its
    answer on the API key its settings.json sets, by the key's last 20 characters, `key_suffix`. A config directory made
    by anything else, `claude auth status` among them, has none of them. Raises Unstartable where the file is there and
    cannot be read: a fault of the file's own, which no first run mends."""
    state = config_dir / ".claude.json"
    cwd = run_directory(config_dir).resolve()
    try:
        said = Payload.parse(state.read_bytes())
        projects = Payload.of(said.fields.get("projects", {}), "its projects").fields
        trusted = any(Payload.of(projects[place], place).optional_flag("hasTrustDialogAccepted") for place in map(str, (cwd, *cwd.parents)) if place in projects)
        onboarding = said.optional_flag("hasCompletedOnboarding")
        responses = Payload.of(said.fields.get("customApiKeyResponses", {}), "its API key answers")
        keys_answered = {*responses.optional_texts("approved"), *responses.optional_texts("rejected")}
    except FileNotFoundError:
        return (f"no {state}",)
    except OSError as error:
        raise Unstartable(f"{state} cannot be read ({error}): make it readable") from None
    except Rejected as error:
        # Claude Code's own first run is what writes this state, so it is the run that mends one hands cannot parse.
        return (f"{state} unreadable: {error}",)
    return (
        *(() if onboarding else ("its onboarding unfinished",)),
        *(() if trusted else (f"{cwd} untrusted",)),
        *(() if key_suffix is None or key_suffix in keys_answered else (f"the API key {config_dir / 'settings.json'} sets unanswered",)),
    )


def answered(config_dir: Path, key_suffix: str | None) -> None:
    """Raises Unstartable, naming the command that answers them, while the brain would start on Claude Code's first screens."""
    if why := unanswered(config_dir, key_suffix):
        raise Unstartable(f"the brain has not been through Claude Code's first screens ({'; '.join(why)}): `hands login` answers them")


def startable(config_dir: Path) -> tuple[str, ...]:
    """What in `config_dir` a Claude Code of hands' own would not start on, every one of them: its pinned claude, its
    settings, its first screens, and the syncs they keep out; empty where it would start. Settings that cannot be read
    are said, and the first screens are weighed without the API key they would name."""
    pin = _refused(lambda: brain_claude(config_dir))
    try:
        held: BrainSettings | None = settings(config_dir)
        unread: tuple[str, ...] = ()
    except Unstartable as error:
        held, unread = None, (str(error),)
    screens = _refused(lambda: answered(config_dir, None if held is None else held.key_suffix))
    kept_out = () if held is None else _refused(lambda: account_kept_out(held))
    return (*pin, *unread, *screens, *kept_out)


def _refused(check: Callable[[], object]) -> tuple[str, ...]:
    """The Unstartable `check` raises, said; none where it raises none."""
    try:
        check()
    except Unstartable as error:
        return (str(error),)
    return ()
