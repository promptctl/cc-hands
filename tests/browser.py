"""A headless Chrome a test drives over the Chrome DevTools Protocol, through aiohttp's websocket client: a page opened
at an address, what it shows read by evaluating JavaScript in it, and a person's clicks and typing sent as input events,
so a page's own handlers run as a phone's browser runs them. Every exception the page throws, and every console error,
is kept, and a test fails on any.
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
        self.thrown: list[str] = []
        self._ended: BaseException | None = None
        self._listening = asyncio.create_task(self._listen())

    async def _listen(self) -> None:
        try:
            async for message in self._socket:
                said = json.loads(message.data)
                if "id" in said:
                    self._answers.pop(said["id"]).set_result(said)
                elif said.get("method") == "Runtime.exceptionThrown":
                    details = said["params"]["exceptionDetails"]
                    self.thrown.append(details.get("exception", {}).get("description", details["text"]))
                elif said.get("method") == "Runtime.consoleAPICalled" and said["params"]["type"] == "error":
                    self.thrown.append(" ".join(str(arg.get("description", arg.get("value"))) for arg in said["params"]["args"]))
            ended: BaseException = ConnectionError(f"Chrome closed the tab's socket: {self._socket.close_code} {self._socket.exception()!r}")
        except Exception as error:
            ended = error
        # [LAW:no-silent-failure] a command still waiting is answered with why no answer will come.
        for waiting in self._answers.values():
            waiting.set_exception(ended)
        self._answers.clear()
        self._ended = ended

    async def send(self, method: str, **params: object) -> dict[str, Any]:
        """The result of one protocol command, or a failure naming the protocol's error."""
        if self._ended is not None:
            raise self._ended
        id = next(self._ids)
        answered = self._answers[id] = asyncio.get_running_loop().create_future()
        await self._socket.send_json({"id": id, "method": method, "params": params})
        said = await answered
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
        """What `expression` is in the page once it is truthy, looked at every 20 ms, or a failure saying what it was when
        WITHIN_SECONDS ran out. A timer, not a frame: a tab behind another is drawn no frames, and its timers still fire."""
        return await self.evaluate(f"""new Promise((done, failed) => {{
          const by = performance.now() + {WITHIN_SECONDS * 1000};
          const looked = () => {{
            const now = (() => {expression})();
            if (now) done(now);
            else if (performance.now() > by) failed(new Error(`still ${{JSON.stringify(now)}} after {WITHIN_SECONDS} s: {json.dumps(expression)[1:-1]}`));
            else setTimeout(looked, 20);
          }};
          looked();
        }})""")

    async def open(self, url: str) -> None:
        """The page at `url`, loaded."""
        await self.send("Page.navigate", url=url)
        await self.until("document.readyState === 'complete'")

    async def click(self, selector: str) -> None:
        """A press and release of the mouse at the middle of the element `selector` finds, scrolled into view first, as a
        person's tap."""
        x, y = await self.evaluate(f"""(() => {{
          const element = document.querySelector({json.dumps(selector)});
          element.scrollIntoView({{block: "center"}});
          const box = element.getBoundingClientRect();
          return [box.left + box.width / 2, box.top + box.height / 2];
        }})()""")
        for kind in ("mousePressed", "mouseReleased"):
            await self.send("Input.dispatchMouseEvent", type=kind, x=x, y=y, button="left", clickCount=1)

    async def type(self, selector: str, text: str) -> None:
        """`text` typed into the field `selector` finds, as a keyboard would."""
        await self.evaluate(f"document.querySelector({json.dumps(selector)}).focus()")
        await self.send("Input.insertText", text=text)

    async def close(self) -> None:
        await self._socket.close()
        await self._listening


class Browser:
    """A headless Chrome, and the tabs a test opens in it."""

    def __init__(self, session: aiohttp.ClientSession, endpoint: str) -> None:
        self._session = session
        self._endpoint = endpoint

    @asynccontextmanager
    async def tab(self) -> AsyncGenerator[Tab]:
        async with self._session.put(f"{self._endpoint}/json/new?about:blank") as made:
            target = await made.json()
        tab = Tab(await self._session.ws_connect(target["webSocketDebuggerUrl"], max_msg_size=0))
        try:
            await tab.send("Runtime.enable")
            await tab.send("Page.enable")
            yield tab
        finally:
            await tab.close()


@asynccontextmanager
async def chrome(profile: Path) -> AsyncGenerator[Browser]:
    """A headless Chrome with `profile` its own, a fake microphone granted to every page that asks, until the block ends."""
    if not CHROME.exists():
        # On CI a skip fails the run (conftest): the runner has Chrome, so this is a machine that does not.
        pytest.skip(f"no Chrome at {CHROME}")
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
