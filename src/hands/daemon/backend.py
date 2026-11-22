"""What a start of hands refuses on: its home's files, read by `preflight` with no process started, and the login the
backend config.toml names reaches its model on, in an environment that names no setting.

`preflight` is read by every start at its door, by `hands check`, and by a settings edit or a model chosen by voice before
the run starts again on it, so an edit the start would refuse is said and outlived. The login is a Claude Code of its own
(`hands.brain.process.logged_in`), so it is read by `backend` alone, as a start builds its brain and by `hands check`,
never to weigh an edit: the running brain's login is the account its start read.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from hands.brain.configdir import Unstartable, startable
from hands.core.wire import UPSTREAM
from hands.daemon.config import Claude
from hands.sessions import wrapper
from hands.sessions.child import Children
from hands.sessions.home import Home
from hands.sessions.payload import Rejected
from hands.voice.backends import ClaudeCodeBackend


@dataclass(frozen=True)
class Preflight:
    """What a start reads of its home before anything runs: the home's copy of fritter and its shim, as the start's event
    records them, and why the start would be refused, where it would."""

    fritter: wrapper.Copy | Literal["unreadable"]
    shim: Literal["current", "stale", "not-a-shim"]
    refused: str | None


def preflight(home: Home) -> Preflight:
    """[LAW:single-enforcer] the one list of what a start refuses on in its home's files, read by the start, `hands
    check`, and the weigh of an edit alike, so none of them can refuse what another takes."""
    # [LAW:no-silent-failure] every session this home's shim starts runs the home's copy of fritter, which updating hands
    # leaves as the older hands copied it, so this hands may ask it what it cannot do: a stale copy is refused, naming its
    # fix, before any brain types through it. No copy yet, a hands that carries none, or a PATH whose claude is not this
    # home's shim, is the survey's to say.
    try:
        fritter: wrapper.Copy | Literal["unreadable"] = wrapper.copy_of(home)
        copied = f"{home.fritter} is not the fritter this hands carries, {wrapper.PACKAGED}, so sessions started under it may not do what this hands asks of them: run `hands install-fritter`" if fritter == "stale" else None
    except OSError as error:
        fritter, copied = "unreadable", f"cannot tell whether {home.fritter} is the fritter this hands carries: {error}"
    # The home's shim is the claude every session hands starts runs, and updating hands leaves it as the older hands wrote
    # it, without what this hands writes into it, its auto-updater switched off among the rest: refused the same way.
    shim: Literal["current", "stale", "not-a-shim"]
    match wrapper.shim_of(home.shim):
        case wrapper.Stale():
            shim, shimmed = "stale", f"{home.shim} is not the shim this hands writes, so sessions started through it may not do what this hands asks of them: {_rebuilt()}run `hands install-fritter`"
        case None:
            shim, shimmed = "not-a-shim", None
        case wrapper.Shim():
            shim, shimmed = "current", None
    refusals = (copied, shimmed, _brain(home))
    return Preflight(fritter, shim, next((refusal for refusal in refusals if refusal is not None), None))


def weigh(home: Home) -> None:
    """Raises Rejected naming what in its home's files a start of hands would be refused on: an edit's weigh."""
    if (refused := preflight(home).refused) is not None:
        raise Rejected(refused)


def _brain(home: Home) -> str | None:
    """Why the brain could not start on its home, or None when it could."""
    try:
        startable(home.brain)
        # The brain runs under the fritter hands' package carries: a hands built without it is refused here, not at its start.
        wrapper.packaged()
    except (Unstartable, wrapper.Unpackaged) as error:
        return str(error)
    return None


def _rebuilt() -> str:
    """The rebuild `hands install-fritter` needs first, ahead of it in a fix, where this hands carries no fritter."""
    try:
        wrapper.packaged()
    except wrapper.Unpackaged as error:
        return f"{error}, then "
    return ""


def backend(children: Children, llm: Claude, home: Home, environment: Mapping[str, str]) -> ClaudeCodeBackend:
    """The brain on the model the settings name, given the login it reaches its model with; raises Rejected naming what it cannot have."""
    # [LAW:no-silent-failure] a setting in the environment would be one silently not applied: settings are the home's
    # config.toml, and HANDS_HOME, where that is, is the one variable of hands' own it reads.
    if stray := sorted(name for name in environment if name.startswith("HANDS_") and name != "HANDS_HOME"):
        raise Rejected(f"{', '.join(stray)} set, and hands reads no setting from the environment; settings go in {home.config}")
    # Imported here, so that only a check loads the brain's process, its MCP server, and Pipecat's tools.
    from hands.brain.process import NotLoggedIn, logged_in

    try:
        account = logged_in(children, home.brain, UPSTREAM, environment)
    except (NotLoggedIn, Unstartable) as error:
        raise Rejected(str(error)) from error
    return ClaudeCodeBackend(model=llm.model, config_dir=home.brain, account=account)
