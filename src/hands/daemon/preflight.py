"""The start's preflight: what a start of hands refuses on in its home's files, read from them alone, starting no
process. The start runs it once, beside the readiness survey, and an edit to the settings and a model chosen by voice
are each weighed by it before hands restarts on them, so an edit the restart would refuse on a file is said and
outlived, never restarted on. The login is the one refusal it leaves out: only a Claude Code can say it.

[LAW:single-enforcer] the one list of the start's file refusals: no caller composes its parts.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from hands.brain import configdir
from hands.brain.configdir import Unread
from hands.sessions import wrapper
from hands.sessions.home import Home
from hands.sessions.payload import Rejected

if TYPE_CHECKING:
    from hands.voice import voicechoice

# What the home's shim is to this hands: the one it writes, one an older hands wrote, or no shim of hands' at all.
type ShimState = Literal["current", "stale", "not-a-shim"]


@dataclass(frozen=True)
class Startable:
    """A home hands starts on: its fritter copy's state and its shim's, for the start's event, and the voice it keeps."""

    fritter: wrapper.Copy | Literal["unreadable"]
    shim: ShimState
    voice: "voicechoice.Chosen"


@dataclass(frozen=True)
class Refused:
    """A home hands would not start on: the same states, and every refusal, each naming its fix."""

    fritter: wrapper.Copy | Literal["unreadable"]
    shim: ShimState
    refusals: tuple[str, ...]


def preflight(home: Home) -> Startable | Refused:
    """Every refusal a start would make on `home`'s files. Hashes a pinned claude this process has not hashed since it
    changed, about 230 MB, so it is called off the loop."""
    # [LAW:dataflow-not-control-flow] every file is weighed every time: one refused hides none after it.
    unpackaged = _unpackaged()
    fritter, stale_fritter = _fritter(home)
    shim, stale_shim = _shim(home)
    voice = _voice(home)
    refusals = (
        # First: `hands install-fritter`, the fix of the two after it, cannot run until hands is rebuilt.
        *unpackaged,
        *stale_fritter,
        *stale_shim,
        *configdir.refusals(configdir.read(home.brain)),
        *((voice.reason,) if isinstance(voice, Unread) else ()),
    )
    match voice:
        case Unread():
            return Refused(fritter, shim, refusals)
        case kept:
            return Refused(fritter, shim, refusals) if refusals else Startable(fritter, shim, kept)


def _unpackaged() -> tuple[str, ...]:
    try:
        wrapper.packaged()
    except wrapper.Unpackaged as error:
        return (str(error),)
    return ()


def _fritter(home: Home) -> tuple[wrapper.Copy | Literal["unreadable"], tuple[str, ...]]:
    # [LAW:no-silent-failure] every session this home's shim starts runs the home's copy of fritter, which updating hands
    # leaves as the older hands copied it, so this hands may ask it what it cannot do. No copy yet, or a hands that
    # carries none, is the survey's to say.
    try:
        copy = wrapper.copy_of(home)
    except OSError as error:
        return "unreadable", (f"cannot tell whether {home.fritter} is the fritter this hands carries: {error}",)
    if copy == "stale":
        return copy, (f"{home.fritter} is not the fritter this hands carries, {wrapper.PACKAGED}, so sessions started under it may not do what this hands asks of them: run `hands install-fritter`",)
    return copy, ()


def _shim(home: Home) -> tuple[ShimState, tuple[str, ...]]:
    # The home's shim is the claude every session hands starts runs, and updating hands leaves it as the older hands wrote
    # it, without what this hands writes into it, its auto-updater switched off among the rest. A PATH whose claude is
    # not this home's shim is the survey's to say.
    match wrapper.shim_of(home.shim):
        case wrapper.Stale():
            return "stale", (f"{home.shim} is not the shim this hands writes, so sessions started through it may not do what this hands asks of them: run `hands install-fritter`",)
        case None:
            return "not-a-shim", ()
        case wrapper.Shim():
            return "current", ()


def _voice(home: Home) -> "voicechoice.Chosen | Unread":
    """The voice the home keeps, or why a start would refuse it."""
    # Imported here, so that a command that reads the settings alone never loads the voices on offer, which are torch's.
    from hands.voice import voicechoice

    try:
        return voicechoice.chosen(home)
    except Rejected as error:
        return Unread(str(error))
    except OSError as error:
        return Unread(f"{home.voice} could not be read ({error}): correct it, or remove it to speak in {voicechoice.spoken(voicechoice.DEFAULT)}")
