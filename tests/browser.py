"""A headless Chrome a test drives over the Chrome DevTools Protocol, through aiohttp's websocket client: a page opened
at an address, what it shows read by evaluating JavaScript in it, and a person's clicks and typing sent as input events,
so a page's own handlers run as a phone's browser runs them. A tab whose page threw an exception, or called
console.error, fails the test that opened it, and nothing a test waits on in the browser is waited on past WITHIN_SECONDS.
"""

import asyncio
import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from itertools import count
from pathlib import Path
from typing import Any, cast

import aiohttp
import pytest

# The Chrome CI's macos-15 runner ships, and a Mac with Chrome installed has, at the path its installer gives it.
CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
# How long anything a test waits on in the browser is given: a page draws what hands answers within a second or two.
WITHIN_SECONDS = 10.0


class Tab:
    """One page open in the browser, and what it has thrown."""

    def __init__(self, socket: aiohttp.ClientWebSocketResponse) -> None:
        self._socket = socket
        self._ids = count(1)
        self._answers: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._loaded = asyncio.Event()
        self.thrown: list[str] = []
        self._listening = asyncio.create_task(self._listen())

    async def _listen(self) -> None:
        async for message in self._socket:
            match json.loads(message.data):
                case {"id": int(id)} as said:
                    answered = self._answers.pop(id)
                    # A command whose sender stopped waiting at its deadline is cancelled: its late answer goes to no one.
                    if not answered.cancelled():
                        answered.set_result(said)
                case {"method": "Page.loadEventFired"}:
                    self._loaded.set()
                case {"method": "Runtime.exceptionThrown", "params": {"exceptionDetails": details}}:
                    self.thrown.append(details.get("exception", {}).get("description", details["text"]))
                case {"method": "Runtime.consoleAPICalled", "params": {"type": "error", "args": args}}:
                    self.thrown.append(" ".join(str(arg.get("description", arg.get("value"))) for arg in args))
                case _:
                    # The enabled domains' other events, a frame navigated or a context made, say nothing a test reads.
                    pass
        # [LAW:no-silent-failure] a command still waiting is answered with why no answer will come.
        closed = ConnectionError(f"Chrome closed the tab's socket: {self._socket.close_code} {self._socket.exception()!r}")
        for waiting in self._answers.values():
            if not waiting.cancelled():
                waiting.set_exception(closed)

    async def send(self, method: str, **params: object) -> dict[str, Any]:
        """The result of one protocol command within WITHIN_SECONDS, or a failure naming the protocol's error or the
        command that went unanswered."""
        id = next(self._ids)
        answered = self._answers[id] = asyncio.get_running_loop().create_future()
        await self._socket.send_json({"id": id, "method": method, "params": params})
        try:
            async with asyncio.timeout(WITHIN_SECONDS):
                said = await answered
        except TimeoutError:
            raise AssertionError(f"{method} unanswered after {WITHIN_SECONDS} s") from None
        if "error" in said:
            raise AssertionError(f"{method} failed: {said['error']}")
        return cast(dict[str, Any], said["result"])

    async def evaluate(self, expression: str) -> Any:
        """What `expression` is in the page, awaited if it is a promise; a failure carrying what it threw."""
        result = await self.send("Runtime.evaluate", expression=expression, awaitPromise=True, returnByValue=True)
        if "exceptionDetails" in result:
            details = result["exceptionDetails"]
            raise AssertionError(details.get("exception", {}).get("description", details["text"]))
        return result["result"].get("value")

    async def until(self, expression: str) -> Any:
        """What `expression` is in the page once it is truthy, looked at every 20 ms, or a failure saying what it last was
        when WITHIN_SECONDS ran out, or what it threw."""
        now = None
        try:
            async with asyncio.timeout(WITHIN_SECONDS):
                while not (now := await self.evaluate(expression)):
                    await asyncio.sleep(0.02)
        except TimeoutError:
            raise AssertionError(f"still {now!r} after {WITHIN_SECONDS} s: {expression}") from None
        return now

    async def open(self, url: str) -> None:
        """The page at `url`, loaded: its scripts run, as a module's are before the page's load event."""
        self._loaded.clear()
        await self.send("Page.navigate", url=url)
        async with asyncio.timeout(WITHIN_SECONDS):
            await self._loaded.wait()

    async def click(self, selector: str) -> None:
        """A press and release of the mouse at the middle of the element `selector` finds, scrolled into view first, as a
        person's tap; a failure if the tap would reach something else, as a hidden or covered element's does."""
        x, y = await self.evaluate(f"""(() => {{
          const element = {_found(selector)};
          element.scrollIntoView({{block: "center"}});
          const box = element.getBoundingClientRect();
          const [x, y] = [box.left + box.width / 2, box.top + box.height / 2];
          if (!element.contains(document.elementFromPoint(x, y))) throw new Error({json.dumps(f"a tap at the middle of {selector} reaches something else")});
          return [x, y];
        }})()""")
        for kind in ("mousePressed", "mouseReleased"):
            await self.send("Input.dispatchMouseEvent", type=kind, x=x, y=y, button="left", clickCount=1)

    async def type(self, selector: str, text: str) -> None:
        """`text` typed into the field `selector` finds, as a keyboard would; a failure if the field does not take focus."""
        await self.evaluate(f"""(() => {{
          const element = {_found(selector)};
          element.focus();
          if (document.activeElement !== element) throw new Error({json.dumps(f"{selector} does not take focus")});
        }})()""")
        await self.send("Input.insertText", text=text)

    async def close(self) -> None:
        await self._socket.close()
        await self._listening


def _found(selector: str) -> str:
    """JavaScript for the element `selector` finds in the page, or a throw naming the selector."""
    return f"(document.querySelector({json.dumps(selector)}) ?? (() => {{ throw new Error({json.dumps(f'nothing matches {selector}')}); }})())"


class Browser:
    """A headless Chrome, and the tabs a test opens in it."""

    def __init__(self, session: aiohttp.ClientSession, endpoint: str) -> None:
        self._session = session
        self._endpoint = endpoint

    @asynccontextmanager
    async def tab(self) -> AsyncGenerator[Tab]:
        """A tab, open for the block; a block that ends with the page having thrown, or called console.error, fails."""
        async with self._session.put(f"{self._endpoint}/json/new?about:blank") as made:
            target = await made.json()
        tab = Tab(await self._session.ws_connect(target["webSocketDebuggerUrl"], max_msg_size=0))
        try:
            await tab.send("Runtime.enable")
            await tab.send("Page.enable")
            yield tab
        finally:
            await tab.close()
        # [LAW:single-enforcer] the one check that a page threw nothing, for every tab: reached only by a block that ended
        # without failing on its own.
        if tab.thrown:
            raise AssertionError(f"the page threw: {tab.thrown}")


@asynccontextmanager
async def chrome(profile: Path) -> AsyncGenerator[Browser]:
    """A headless Chrome with `profile` its own, a fake microphone granted to every page that asks, until the block ends."""
    if not CHROME.exists():
        pytest.fail(f"no Chrome at {CHROME}: the phone's pages are run in it")
    # --no-sandbox: the run holds itself offline in macOS's sandbox (tests/offline.py), which refuses the second one Chrome's
    # helpers would put themselves in, and they end as they start. --use-mock-keychain: a test's HOME has no login keychain,
    # and Chrome waits on one for the key its cookies are kept under before it loads a page.
    process = await asyncio.create_subprocess_exec(
        str(CHROME), "--headless=new", "--no-sandbox", "--use-mock-keychain", "--remote-debugging-port=0", f"--user-data-dir={profile}", "--no-first-run",
        "--no-default-browser-check", "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", "about:blank",
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
    )
    listening = asyncio.get_running_loop().create_future()
    # Chrome's stderr is read to its end, so a Chrome that says a lot never stops on a full pipe.
    reading = asyncio.create_task(_said(process, listening))
    try:
        async with asyncio.timeout(WITHIN_SECONDS):
            endpoint = await listening
        async with aiohttp.ClientSession() as session:
            yield Browser(session, endpoint)
    finally:
        # A Chrome that ended on its own, as one that never listened does, is past killing.
        if process.returncode is None:
            process.kill()
        await process.wait()
        await reading


async def _said(process: asyncio.subprocess.Process, listening: asyncio.Future[str]) -> None:
    """Chrome's stderr to its end: `listening` is given the address its DevTools listens on as Chrome says it, or a failure
    carrying everything Chrome said if it ends first."""
    assert process.stderr is not None
    said: list[str] = []
    while line := (await process.stderr.readline()).decode():
        said.append(line)
        if line.startswith("DevTools listening on ws://") and not listening.done():
            # ws://127.0.0.1:<port>/devtools/browser/<id>: the HTTP endpoint is the same host and port.
            listening.set_result("http://" + line.removeprefix("DevTools listening on ws://").split("/", 1)[0])
    if not listening.done():
        listening.set_exception(AssertionError(f"Chrome ended before it listened:\n{''.join(said)}"))
