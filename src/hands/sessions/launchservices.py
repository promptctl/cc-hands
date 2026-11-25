# pyright: basic, reportAttributeAccessIssue=false
"""The Mac's screen as LaunchServices and the app in front say it: the one real adapter of
`hands.sessions.screen.Screen`. LaunchServices names the app in front; an app that can say which of its tabs is in front
by AppleScript is asked."""

import re
from collections.abc import Mapping
from dataclasses import dataclass

from Foundation import NSAppleScript

from hands.core.front import FrontUnread
from hands.sessions import child
from hands.sessions.child import Children
from hands.sessions.screen import FrontApp, FrontTab
from hands.sessions.terminals import device
from hands.threads import SerialThread

# Each app that says by AppleScript which terminal its front tab shows, by bundle id. Only the app in front is asked:
# telling an app anything launches it. The time limit is the Apple event's own, which otherwise waits two minutes on an
# app that does not answer, or on the user deciding whether hands may ask it at all.
_FRONT_TAB: Mapping[str, str] = {
    bundle: f'with timeout of 2 seconds\ntell application id "{bundle}" to {asked}\nend timeout'
    for bundle, asked in (("com.googlecode.iterm2", "tty of current session of current window"), ("com.apple.Terminal", "tty of selected tab of front window"))
}

_LSAPPINFO = "/usr/bin/lsappinfo"
# lsappinfo info -only names one field a line: "pid"=93900, "CFBundleIdentifier"="com.google.Chrome".
_FIELD = re.compile(r'^"(\w+)"=(?:"(.*)"|(\d+))$', re.MULTILINE)


@dataclass(frozen=True)
class Launched:
    """An app as LaunchServices names it: its pid, its bundle id where it has one, and its name."""

    pid: int
    bundle: str | None
    name: str


def serial_of(front: str) -> str | FrontUnread:
    """The app in front, by the serial number lsappinfo's `front` answer names it with."""
    serial = front.strip()
    return serial if serial.startswith("ASN:") else FrontUnread(f"no app is in front: lsappinfo said {serial!r}")


def launched(serial: str, info: str) -> Launched | FrontUnread:
    """The app lsappinfo's `info -only pid -only bundleid -only name` answer for `serial` names."""
    fields = {match[1]: match[2] if match[3] is None else match[3] for match in _FIELD.finditer(info)}
    match fields:
        case {"pid": pid, "LSDisplayName": name}:
            return Launched(int(pid), fields.get("CFBundleIdentifier"), name)
        case _:
            return FrontUnread(f"lsappinfo did not say which app {serial} is: {info!r}")


class _Unread(Exception):
    pass


def _known[T](read: T | FrontUnread) -> T:
    if isinstance(read, FrontUnread):
        raise _Unread(read.reason)
    return read


class LaunchServices:
    """The screen, read through lsappinfo, run by `children`, and the app in front's AppleScript."""

    def __init__(self, children: Children) -> None:
        self._children = children
        # [LAW:no-shared-mutable-globals] NSAppleScript is not thread-safe: every script is compiled and run on this one
        # thread, which owns the compiled scripts, and which a stopping daemon never waits on. Compiled once, on the first
        # ask: the first costs ~170 ms, the ones after it ~5 ms (measured with iTerm2 3.6).
        self._applescript = SerialThread("AppleScript")
        self._scripts: dict[str, NSAppleScript] = {}

    async def front(self) -> FrontTab | FrontApp | FrontUnread:
        try:
            serial = _known(serial_of(await self._ran(_LSAPPINFO, "front")))
            match _known(launched(serial, await self._ran(_LSAPPINFO, "info", "-only", "pid", "-only", "bundleid", "-only", "name", serial))):
                case Launched(bundle=str(bundle), name=name) if bundle in _FRONT_TAB:
                    tty = await self._applescript.run(lambda: self._asked(name, _FRONT_TAB[bundle]))
                    try:
                        return FrontTab(name, device(tty))
                    except OSError as error:
                        return FrontUnread(f"the terminal {tty} could not be read: {error}")
                case Launched(pid=pid, name=name):
                    return FrontApp(pid, name)
        except _Unread as unread:
            return FrontUnread(str(unread))

    async def _ran(self, *argv: str) -> str:
        ran = await self._children.run(*argv, limit=2)
        match ran:
            case child.Ran(ending=child.Exited(code=0), out=out):
                return out.text()
            case child.Ran() | child.NeverStarted() | child.Unreaped() | child.Lost():
                raise _Unread(f"{' '.join(argv)} {child.said(ran)}")

    def _asked(self, app: str, source: str) -> str:
        if source not in self._scripts:
            self._scripts[source] = NSAppleScript.alloc().initWithSource_(source)
        answer, error = self._scripts[source].executeAndReturnError_(None)
        if answer is None:
            raise _Unread(f"{app} did not say which tab is in front: {error.get('NSAppleScriptErrorMessage', error) if error else 'no answer'}")
        return str(answer.stringValue())
