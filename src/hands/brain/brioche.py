"""brioche, the brain's HTTPS proxy, for as long as the brain runs.

The brain's Claude Code reaches Anthropic's API through brioche, and takes it for that API: brioche answers its TLS for
api.anthropic.com under an authority of its own, which the brain is given to trust, and sends each request on, as it was
sent, to hands' proxy, which holds, rewrites, answers and records every exchange, as it always has. A Claude Code given a
gateway in ANTHROPIC_BASE_URL behaves as a gateway's client; through brioche it behaves first-party.

brioche is fritter/brioche, built into hands' package beside fritter. It prints the variables that point a client at it
and serves until its input ends, so it never outlives hands; each line it says on its stderr is in hands' audit log.
"""

import asyncio
import os
from collections.abc import AsyncGenerator, Mapping
from contextlib import ExitStack, asynccontextmanager
from pathlib import Path
from typing import BinaryIO

from hands import packaged
from hands.core.wire import UPSTREAM
from hands.sessions.audit import BriocheSaid, Record
from hands.sessions.child import Children, Ending, Lost, NeverStarted, Pipes, Running, Unreaped, said
from hands.sessions.untap import TRUST

# [LAW:one-source-of-truth] the brioche hands' own package carries.
BRIOCHE = packaged.installed(packaged.BRIOCHE)
# How long brioche has to stop once it is asked to before its group is killed: it stops at once.
STOP_SECONDS = 5.0


class BriocheFailed(Exception):
    """brioche could not be started, or ended before it said how a client reaches it: what it said is why."""


class Brioche:
    """brioche, running: the variables that point the brain's Claude Code at it, and its end."""

    def __init__(self, running: Running, environment: Mapping[str, str], said: "asyncio.Task[str]") -> None:
        self._running = running
        # [LAW:one-source-of-truth] what brioche printed, as it printed it: the brain's way to Anthropic's API.
        self.environment = environment
        self._said = said

    async def outlived(self) -> None:
        """Ends once brioche does, raising how it ended and the last it said."""
        # [LAW:no-silent-failure] a brioche that ends while hands runs leaves every request of the brain's unreached, and one
        # started again answers under an authority the running brain does not trust: it stops the run, as the brain's end does.
        ended = await self._running.ended
        raise RuntimeError(f"brioche, the brain's HTTPS proxy, ended while hands was running: {_ended(ended, await self._said)}")


@asynccontextmanager
async def proxied(children: Children, to: str, trust: Path, inherited: Mapping[str, str], record: Record) -> AsyncGenerator[Brioche]:
    """brioche, sending each request of the brain's for Anthropic's API to hands' proxy at `to`, run in `inherited`, the
    environment the brain would be given without it: the proxy it reaches the world through, and the certificates it
    trusts, which brioche writes to `trust` beside its own authority. Stopped as the context ends; raises BriocheFailed,
    saying why, when it does not start."""
    loop = asyncio.get_running_loop()
    with ExitStack() as held:
        reads, writes = os.pipe()
        printed, prints = os.pipe()
        complaints, complains = os.pipe()
        # [LAW:effects-at-boundaries] hands' end of brioche's input is held open for as long as brioche is to run: hands
        # ending, however it ends, ends that input, and so brioche. Closing a pipe hands has read already is no close.
        held.callback(os.close, writes)
        printing = held.enter_context(os.fdopen(printed, "rb", 0))
        saying = held.enter_context(os.fdopen(complaints, "rb", 0))
        argv = [str(BRIOCHE), "--intercept", UPSTREAM, "--to", to, "--trust-variable", TRUST, "--trust-file", str(trust)]
        try:
            started = children.start(argv, trust.parent, inherited, Pipes(reads, prints, complains))
        finally:
            # brioche's own ends are its alone once it is started: each pipe ends when brioche's end of it does.
            for theirs in (reads, prints, complains):
                os.close(theirs)
        if isinstance(started, NeverStarted):
            raise BriocheFailed(f"brioche, the brain's HTTPS proxy, {said(started)}")
        # [LAW:nothing-unseen] each line it says is in the audit log as it says it; the last is why it ended.
        last = asyncio.create_task(_said(loop, saying, record), name="brioche's stderr")
        try:
            environment = _environment(await _whole(loop, printing))
            if not environment:
                # It ended its output with nothing printed, as it does only as it ends: how, and what it said.
                raise BriocheFailed(f"brioche, the brain's HTTPS proxy, did not start: {_ended(await started.ended, await last)}")
            yield Brioche(started, environment, last)
        finally:
            await started.stopped(STOP_SECONDS)
            await last


def _ended(ended: Ending | Lost | Unreaped, last: str) -> str:
    """How brioche ended, and the last it said, which says why."""
    return f"{said(ended)}: {last}" if last else said(ended)


def _environment(printed: bytes) -> dict[str, str]:
    """The variables brioche printed, a NAME=VALUE line each. [LAW:parse-dont-validate] a line that is none is a brioche
    that is not this hands' own, and fails the start."""
    environment: dict[str, str] = {}
    for line in printed.decode().splitlines():
        name, equals, value = line.partition("=")
        if not equals or not name:
            raise BriocheFailed(f"brioche, the brain's HTTPS proxy, printed {line!r}, which is no NAME=VALUE line")
        environment[name] = value
    return environment


async def _reader(loop: asyncio.AbstractEventLoop, pipe: BinaryIO) -> tuple[asyncio.StreamReader, asyncio.BaseTransport]:
    """A reader of `pipe`, whose transport closes it."""
    reader = asyncio.StreamReader()
    transport, _ = await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), pipe)
    return reader, transport


async def _whole(loop: asyncio.AbstractEventLoop, pipe: BinaryIO) -> bytes:
    """What is written to `pipe`, to its end."""
    reader, transport = await _reader(loop, pipe)
    try:
        return await reader.read()
    finally:
        transport.close()


async def _said(loop: asyncio.AbstractEventLoop, pipe: BinaryIO, record: Record) -> str:
    """Each line said on `pipe`, to the audit log as it is said; the last, once the pipe ends."""
    reader, transport = await _reader(loop, pipe)
    last = ""
    try:
        async for line in reader:
            last = line.decode(errors="replace").rstrip("\n")
            record(BriocheSaid(last))
    finally:
        transport.close()
    return last
