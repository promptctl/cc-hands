"""The Mac's screen: the app in front, and the terminal its front tab shows. One real adapter,
`hands.sessions.launchservices.LaunchServices`, which only a composition root builds; every other module is handed a
`Screen`."""

from dataclasses import dataclass
from typing import Protocol

from hands.core.front import FrontUnread


@dataclass(frozen=True)
class FrontTab:
    """An app in front that said which of its tabs is in front: its name, and the device of the terminal that tab shows;
    none for a terminal closed since the app named it, which shows nothing."""

    name: str
    terminal: int | None


@dataclass(frozen=True)
class FrontApp:
    """An app in front that cannot be asked which of its tabs is: its pid, under which every terminal it shows runs, and
    its name. The one the user is using, whether or not a window of its is on screen."""

    pid: int
    name: str


class Screen(Protocol):
    async def front(self) -> FrontTab | FrontApp | FrontUnread:
        """The app in front now, or why it could not be read."""
        ...
