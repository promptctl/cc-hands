"""Where a setting meets its login: the backend config.toml names, with the login it reaches its model on, in an
environment that names no setting.

Read by `hands run` before the voice loads, and by `hands check`, which loads the voice's pipeline for neither, each on
what the start's preflight passed. The login is the one refusal the preflight leaves out: only a Claude Code can say it.
"""

from collections.abc import Mapping

from hands.brain.configdir import Unstartable
from hands.core.session import Runnable
from hands.core.wire import UPSTREAM
from hands.daemon.config import Claude
from hands.sessions.child import Children
from hands.sessions.home import Home
from hands.sessions.payload import Rejected
from hands.voice.backends import ClaudeCodeBackend


def backend(children: Children, llm: Claude, home: Home, environment: Mapping[str, str], brain: Runnable) -> ClaudeCodeBackend:
    """The brain on the model the settings name, run with what the start's preflight passed, given the login it reaches
    its model with; raises Rejected naming what it cannot have."""
    # [LAW:single-enforcer] where a setting meets its login: the brain's login is checked here, once, before the voice
    # loads, rather than once every turn has failed.
    # [LAW:no-silent-failure] a setting in the environment would be one silently not applied: settings are the home's
    # config.toml, and HANDS_HOME, where that is, is the one variable of hands' own it reads.
    if stray := sorted(name for name in environment if name.startswith("HANDS_") and name != "HANDS_HOME"):
        raise Rejected(f"{', '.join(stray)} set, and hands reads no setting from the environment; settings go in {home.config}")
    # Imported here, so that only a check loads the brain's process, its MCP server, and Pipecat's tools.
    from hands.brain.process import NotLoggedIn, logged_in

    try:
        account = logged_in(children, brain.claude, home.brain, UPSTREAM, environment)
    except (NotLoggedIn, Unstartable) as error:
        raise Rejected(str(error)) from error
    return ClaudeCodeBackend(model=llm.model, config_dir=home.brain, account=account, brain=brain)
