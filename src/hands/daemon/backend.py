"""Where a setting meets its login: the backend config.toml names, with the login it reaches its model on.

Read by `hands run` before the voice loads, and by `hands check`, which loads the voice's pipeline for neither, each on
what the start's preflight weighed, and by a start the preflight refused, which asks the login beside its refusals. The
login is the one refusal the preflight leaves out: only a Claude Code can say it.
"""

from collections.abc import Mapping

from hands.core.session import Runnable
from hands.core.wire import Direct
from hands.daemon.config import Claude
from hands.sessions.child import Children
from hands.sessions.home import Home
from hands.sessions.payload import Rejected
from hands.voice.backends import ClaudeCodeBackend


def backend(children: Children, llm: Claude, home: Home, environment: Mapping[str, str], brain: Runnable) -> ClaudeCodeBackend:
    """The brain on the model the settings name, run with what the start's preflight passed, given the login it reaches
    its model with; raises Rejected naming what it cannot have, and Unstartable where its Claude Code could not be asked,
    which says nothing of its login."""
    # [LAW:single-enforcer] where a setting meets its login: the brain's login is checked here, once, before the voice
    # loads, rather than once every turn has failed.
    # Imported here, so that only a check loads the brain's process, its MCP server, and Pipecat's tools.
    from hands.brain.process import NotLoggedIn, logged_in

    try:
        # Direct: asked in hands' environment as it stands, the one `hands check` runs in too, which holds no proxy of
        # hands': only a brain is given that.
        account = logged_in(children, brain.claude.path, home.brain, Direct(), environment)
    except NotLoggedIn as error:
        raise Rejected(str(error)) from error
    return ClaudeCodeBackend(model=llm.model, config_dir=home.brain, account=account, brain=brain)
