# pyright: basic, reportAttributeAccessIssue=false
"""The Mac's screen as LaunchServices and the app in front say it: the one real adapter of
`hands.sessions.screen.Screen`. LaunchServices names the app in front; an app that can say which of its tabs is in front
by AppleScript is asked."""

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass

from Foundation import NSAppleScript

from hands.core.front import FrontUnread
from hands.sessions import child
from hands.sessions.child import Children
from hands.sessions.screen import FrontApp
from hands.threads import SerialThread

# Each app that says by AppleScript which terminal its front tab shows, by bundle id. Only the app in front is asked:
# telling an app anything launches it. The time limit is the Apple event's own, which otherwise waits two minutes on an
# app that does not answer, or on the user deciding whether hands may ask it at all.
_FRONT_TAB: Mapping[str, str] = {
    bundle: f'with timeout of 2 seconds\ntell application id "{bundle}" to {asked}\nend timeout'
    for bundle, asked in (("com.googlecode.iterm2", "tty of current session of current window"), ("com.apple.Terminal", "tty of selected tab of front window"))
}
# [LAW:no-shared-mutable-globals] NSAppleScript is not thread-safe: every script is compiled and run on this one thread,
# which owns the compiled scripts, and which a stopping daemon never waits on. Compiled once, on the first ask: the first
# costs ~170 ms, the ones after it ~5 ms (measured with iTerm2 3.6).
_applescript = SerialThread("AppleScript")
_scripts: dict[str, NSAppleScript] = {}

_LSAPPINFO = "/usr/bin/lsappinfo"
# lsappinfo info -only names one field a line: "pid"=93900, "CFBundleIdentifier"="com.google.Chrome".
_FIELD = re.compile(r'^"(\w+)"=(?:"(.*)"|(\d+))$', re.MULTILINE)


class _Unread(Exception):
    pass


@dataclass(frozen=True)
class _App:
    pid: int
    bundle: str | None
    name: str


class LaunchServices:
    """The screen, read through lsappinfo, run by `children`, and the app in front's AppleScript."""

    def __init__(self, children: Children) -> None:
        self._children = children

    async def front(self) -> FrontApp | FrontUnread:
        try:
            app = await self._front_app()
            match _FRONT_TAB.get(app.bundle or ""):
                case str() as source:
                    return FrontApp(app.pid, app.name, _device(await _applescript.run(lambda: _asked(app.name, source))))
                case None:
                    return FrontApp(app.pid, app.name, None)
        except _Unread as unread:
            return FrontUnread(str(unread))

    async def _front_app(self) -> _App:
        """The app LaunchServices says is in front."""
        serial = (await self._ran(_LSAPPINFO, "front")).strip()
        if not serial.startswith("ASN:"):
            raise _Unread(f"no app is in front: lsappinfo said {serial!r}")
        info = await self._ran(_LSAPPINFO, "info", "-only", "pid", "-only", "bundleid", "-only", "name", serial)
        fields = {match[1]: match[2] if match[3] is None else match[3] for match in _FIELD.finditer(info)}
        match fields:
            case {"pid": pid, "LSDisplayName": name}:
                return _App(int(pid), fields.get("CFBundleIdentifier"), name)
            case _:
                raise _Unread(f"lsappinfo did not say which app {serial} is: {info!r}")

    async def _ran(self, *argv: str) -> str:
        ran = await self._children.run(*argv, limit=2)
        match ran:
            case child.Ran(ending=child.Exited(code=0), out=out):
                return out.text()
            case child.Ran() | child.NeverStarted() | child.Unreaped() | child.Lost():
                raise _Unread(f"{' '.join(argv)} {child.said(ran)}")


def _asked(app: str, source: str) -> str:
    script = _scripts.setdefault(source, NSAppleScript.alloc().initWithSource_(source))
    answer, error = script.executeAndReturnError_(None)
    if answer is None:
        raise _Unread(f"{app} did not say which tab is in front: {error.get('NSAppleScriptErrorMessage', error) if error else 'no answer'}")
    return str(answer.stringValue())


def _device(path: str) -> int:
    try:
        return os.stat(path).st_rdev
    except OSError as error:
        raise _Unread(f"the terminal {path} could not be read: {error}") from error
