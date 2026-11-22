"""What a start of hands refuses on, each read with no process started but the last: this hands' fritter and the
home's copy of it and its shim (`copies`), which the start's door refuses on; the brain's config directory
(`hands.brain.configdir.startable`), which the start refuses on as it is configured, with every other step of
`hands check` said beside it; and the login the backend config.toml names reaches its model on, in an environment that
names no setting (`backend`).

[LAW:single-enforcer] `weigh` is both reads of the home together, so a settings edit or a model chosen by voice is
weighed on every refusal a start raises on the home's files, and none of them can refuse what a start takes. The login
is a Claude Code of its own (`hands.brain.process.logged_in`), so it is read by `backend` alone, as a start builds its
brain and by `hands check`, never to weigh an edit: a brain logged out since its start passes the weigh and is refused
at the restart (promptctl-phone-app-9d9y.h4s).
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
class Copies:
    """The home's copy of fritter and its shim, as this hands finds them and the start's event records them, and every
    refusal of a start they make."""

    fritter: wrapper.Copy | Literal["unreadable"]
    shim: Literal["current", "stale", "not-a-shim"]
    refusals: tuple[str, ...]


def copies(home: Home) -> Copies:
    """The home's copy of fritter and its shim, as this hands finds them."""
    # [LAW:no-silent-failure] every session this home's shim starts runs the home's copy of fritter, which updating hands
    # leaves as the older hands copied it, so this hands may ask it what it cannot do: a stale copy is refused, naming its
    # fix. No copy yet, or a PATH whose claude is not this home's shim, is the survey's to say. A hands that carries no
    # fritter, which copy_of is the one probe of, is refused naming the rebuild: the brain runs under that fritter, and
    # `hands install-fritter`, every other fix here, copies it.
    try:
        fritter: wrapper.Copy | Literal["unreadable"] = wrapper.copy_of(home)
        copied: tuple[str, ...] = (f"{home.fritter} is not the fritter this hands carries, {wrapper.PACKAGED}, so sessions started under it may not do what this hands asks of them: run `hands install-fritter`",) if fritter == "stale" else ()
    except OSError as error:
        fritter, copied = "unreadable", (f"cannot tell whether {home.fritter} is the fritter this hands carries: {error}",)
    unpackaged = (wrapper.rebuild(),) if fritter == "unpackaged" else ()
    # The home's shim is the claude every session hands starts runs, and updating hands leaves it as the older hands wrote
    # it, without what this hands writes into it, its auto-updater switched off among the rest: refused the same way.
    shim: Literal["current", "stale", "not-a-shim"]
    match wrapper.shim_of(home.shim):
        case wrapper.Stale():
            shim, shimmed = "stale", (f"{home.shim} is not the shim this hands writes, so sessions started through it may not do what this hands asks of them: run `hands install-fritter`",)
        case None:
            shim, shimmed = "not-a-shim", ()
        case wrapper.Shim():
            shim, shimmed = "current", ()
    # The rebuild first: `hands install-fritter` cannot run until it is done.
    return Copies(fritter, shim, (*unpackaged, *copied, *shimmed))


def weigh(home: Home) -> None:
    """Raises Rejected naming every refusal of a start in its home's files: an edit's weigh."""
    if refusals := (*copies(home).refusals, *startable(home.brain)):
        raise Rejected("; ".join(refusals))


def backend(children: Children, llm: Claude, home: Home, environment: Mapping[str, str]) -> ClaudeCodeBackend:
    """The brain on the model the settings name, given a home it would start on and the login it reaches its model with;
    raises Rejected naming what it cannot have."""
    # [LAW:no-silent-failure] a setting in the environment would be one silently not applied: settings are the home's
    # config.toml, and HANDS_HOME, where that is, is the one variable of hands' own it reads.
    if stray := sorted(name for name in environment if name.startswith("HANDS_") and name != "HANDS_HOME"):
        raise Rejected(f"{', '.join(stray)} set, and hands reads no setting from the environment; settings go in {home.config}")
    if refusals := startable(home.brain):
        raise Rejected("; ".join(refusals))
    # Imported here, so that only a check loads the brain's process, its MCP server, and Pipecat's tools.
    from hands.brain.process import NotLoggedIn, logged_in

    try:
        account = logged_in(children, home.brain, UPSTREAM, environment)
    except (NotLoggedIn, Unstartable) as error:
        raise Rejected(str(error)) from error
    return ClaudeCodeBackend(model=llm.model, config_dir=home.brain, account=account)
