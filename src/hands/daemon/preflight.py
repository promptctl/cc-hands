"""The start's preflight: what a start of hands refuses on in its home's files and the environment it starts in, read
from them alone, starting no process. The start runs it once, as its own step before the configuration is read, and an edit to the settings and a
model chosen by voice are each weighed by it before hands restarts on them, so an edit the restart would refuse on a
file is said and outlived, never restarted on. The login is the one refusal it leaves out: only a Claude Code can say
it, and a login lost since the start is the brain's own API answers' to say (promptctl-phone-app-9d9y.h4s).

[LAW:single-enforcer] the one list of the start's file refusals, and what a start that passes it is run with.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from hands.brain import configdir
from hands.core.session import Runnable
from hands.sessions import wrapper
from hands.sessions.files import Unread
from hands.sessions.home import Home
from hands.sessions.payload import Rejected
from hands.voice import voicechoice

# What the home's copy of fritter is to this hands, or why that cannot be told.
type FritterState = wrapper.Copy | Unread
# What the home's shim is to this hands: the one it writes, one an older hands wrote, no shim of hands' at all, or one
# that cannot be read.
type ShimState = Literal["current", "stale", "not-a-shim"] | Unread


@dataclass(frozen=True)
class Startable:
    """A home hands starts on: what its brain is run with, and the voice it keeps."""

    brain: Runnable
    voice: voicechoice.Chosen


@dataclass(frozen=True)
class Refused:
    """A home hands would not start on: every refusal, each naming its fix, and what its brain is run with where its own
    config directory refuses no run of Claude Code on it, so its login is asked beside them; None where it does."""

    refusals: tuple[str, ...]
    brain: Runnable | None


@dataclass(frozen=True)
class Preflight:
    """What a start finds in its home's files: the states of its fritter copy and its shim, which the start's event
    records, and whether it starts on them."""

    fritter: FritterState
    shim: ShimState
    verdict: Startable | Refused

    def facts(self) -> dict[str, str]:
        """The states as the start's event records them."""
        return {"fritter": _fact(self.fritter), "shim": _fact(self.shim)}


def _fact(state: FritterState | ShimState) -> str:
    match state:
        case Unread():
            return "unreadable"
        case str():
            return state


def preflight(home: Home, environment: Mapping[str, str]) -> Preflight:
    """Every refusal a start would make on `home`'s files and `environment`. Hashes a pinned claude this process has not hashed since it
    changed, about 230 MB, so it is called off the loop."""
    config = configdir.read(home.brain)
    fritter = _fritter(home)
    shim = _shim(home)
    brain = configdir.runnable(config)
    voice = _voice(home)
    # [LAW:dataflow-not-control-flow] every file is weighed every time: one refused hides none after it.
    refusals = (
        # First: `hands install-fritter`, the fix of the two after it, cannot run until hands is rebuilt.
        *_unpackaged(),
        *_fritter_refusals(home, fritter),
        *_shim_refusals(home, shim),
        *(() if isinstance(brain, Runnable) else brain),
        *configdir.first_screens(config),
        *((voice.reason,) if isinstance(voice, Unread) else ()),
        *_strays(home, environment),
    )
    match brain, voice:
        case Runnable(), (str() | voicechoice.LibraryVoice()) as kept if not refusals:
            return Preflight(fritter, shim, Startable(brain, kept))
        case _:
            return Preflight(fritter, shim, Refused(refusals, brain if isinstance(brain, Runnable) else None))


def beside(home: Home, environment: Mapping[str, str], unread: Rejected) -> tuple[str, ...]:
    """Every refusal of a start whose settings do not parse: theirs, then every one the preflight finds beside it, so a
    settings file that cannot be read hides none of the start's other refusals."""
    match preflight(home, environment).verdict:
        case Refused(refusals=refusals):
            return (str(unread), *refusals)
        case Startable():
            return (str(unread),)


def _strays(home: Home, environment: Mapping[str, str]) -> tuple[str, ...]:
    # [LAW:no-silent-failure] a setting in the environment would be one silently not applied: settings are the home's
    # config.toml, and HANDS_HOME, where that is, is the one variable of hands' own it reads.
    stray = sorted(name for name in environment if name.startswith("HANDS_") and name != "HANDS_HOME")
    return (f"{', '.join(stray)} set, and hands reads no setting from the environment; settings go in {home.config}",) if stray else ()


def _unpackaged() -> tuple[str, ...]:
    try:
        wrapper.packaged()
    except wrapper.Unpackaged as error:
        return (str(error),)
    return ()


def _fritter(home: Home) -> FritterState:
    try:
        return wrapper.copy_of(home)
    except OSError as error:
        return Unread(str(error))


def _fritter_refusals(home: Home, state: FritterState) -> tuple[str, ...]:
    # [LAW:no-silent-failure] every session this home's shim starts runs the home's copy of fritter, which updating hands
    # leaves as the older hands copied it, so this hands may ask it what it cannot do. No copy yet, or a hands that
    # carries none, is the survey's to say.
    match state:
        case Unread(reason=reason):
            return (f"cannot tell whether {home.fritter} is the fritter this hands carries ({reason}): run `hands install-fritter` to copy it again",)
        case "stale":
            return (f"{home.fritter} is not the fritter this hands carries, {wrapper.PACKAGED}, so sessions started under it may not do what this hands asks of them: run `hands install-fritter`",)
        case "current" | "absent" | "unpackaged":
            return ()


def _shim(home: Home) -> ShimState:
    match wrapper.shim_of(home.shim):
        case Unread() as unreadable:
            return unreadable
        case wrapper.Stale():
            return "stale"
        case None:
            return "not-a-shim"
        case wrapper.Shim():
            return "current"


def _shim_refusals(home: Home, state: ShimState) -> tuple[str, ...]:
    # The home's shim is the claude every session hands starts runs, and updating hands leaves it as the older hands wrote
    # it, without what this hands writes into it, its auto-updater switched off among the rest. A PATH whose claude is
    # not this home's shim is the survey's to say.
    match state:
        case Unread(reason=reason):
            return (f"cannot tell whether {home.shim} is the shim this hands writes ({reason}): run `hands install-fritter` to write it again",)
        case "stale":
            return (f"{home.shim} is not the shim this hands writes, so sessions started through it may not do what this hands asks of them: run `hands install-fritter`",)
        case "current" | "not-a-shim":
            return ()


def _voice(home: Home) -> voicechoice.Chosen | Unread:
    """The voice the home keeps, or why a start would refuse it."""
    try:
        return voicechoice.chosen(home)
    except Rejected as error:
        return Unread(str(error))
    except OSError as error:
        return Unread(f"{home.voice} could not be read ({error}): correct it, or remove it to speak in {voicechoice.spoken(voicechoice.DEFAULT)}")
