"""The Mac's screen: the app in front, and the terminal its front tab shows. One real adapter,
`hands.sessions.launchservices.LaunchServices`, which only a composition root builds; every other module is handed a
`Screen`."""

from dataclasses import dataclass
from typing import Protocol

from hands.core.front import FrontUnread


@dataclass(frozen=True)
class FrontApp:
    """The app in front: the one the user is using, whether or not a window of its is on screen. `tab` is the device of
    the terminal its front tab shows, where the app says which; None for an app that cannot be asked."""

    pid: int
    name: str
    tab: int | None


class Screen(Protocol):
    async def front(self) -> FrontApp | FrontUnread:
        """The app in front now, or why it could not be read."""
        ...
