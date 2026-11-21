"""The screen path for what speech cannot say: a macOS notification, posted through osascript."""

from loguru import logger

from hands.clock import Clock
from hands.sessions.child import Exited, NeverStarted, TimedOut, run

# The text is passed as an argument, never spliced into the script, so nothing in it needs quoting.
_SCRIPT = ("on run argv", 'display notification (item 1 of argv) with title "hands"', "end run")

# osascript posts in a fraction of a second; past this the screen is not going to take it.
POST_TIMEOUT_SECONDS = 10.0


def notification_command(text: str) -> list[str]:
    return ["osascript", *(part for line in _SCRIPT for part in ("-e", line)), "--", text]


async def post_notification(text: str, clock: Clock) -> bool:
    """True when the screen took it, so a caller records as given only what was given."""
    # [LAW:no-silent-failure] when speech and the screen have both failed, the log is the path left — and the
    # refusal is told to the caller as well, because a run started over ssh may have no GUI session to post into.
    match await run(*notification_command(text), timeout=POST_TIMEOUT_SECONDS, clock=clock):
        case Exited(returncode=0):
            logger.info(f"posted a notification: {text}")
            return True
        case Exited(returncode=code, err=err):
            logger.error(f"osascript refused to post {text!r} ({code}): {err.strip()}")
        case TimedOut():
            logger.error(f"osascript did not post {text!r} in {POST_TIMEOUT_SECONDS:.0f}s")
        case NeverStarted(error=error):
            logger.error(f"cannot run osascript to post {text!r}: {error}")
    return False
