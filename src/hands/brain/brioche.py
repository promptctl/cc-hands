"""brioche, the brain's HTTPS proxy, for as long as the brain runs.

The brain's Claude Code reaches Anthropic's API through brioche, and takes it for that API: brioche answers its TLS for
api.anthropic.com under an authority of its own, which the brain is given to trust, and sends each request on, as it was
sent, to hands' proxy, which holds, rewrites, answers and records every exchange, as it always has. A Claude Code given a
gateway in ANTHROPIC_BASE_URL behaves as a gateway's client; through brioche it behaves first-party.

brioche is fritter/brioche, built into hands' package beside fritter. It prints the variables that point a client at it
and serves until its input ends, so it never outlives hands; each report it makes on its stderr is in hands' audit log.
"""

import asyncio
import os
from collections.abc import AsyncGenerator, Mapping
from contextlib import ExitStack, asynccontextmanager
from pathlib import Path
from types import MappingProxyType
from typing import BinaryIO

from hands import packaged
from hands.brain.configdir import Unstartable
from hands.brain.process import Through, passed_on
from hands.core.wire import UPSTREAM
from hands.sessions.audit import LEVELS, BriocheSaid, Record
from hands.sessions.child import Children, Ending, Lost, NeverStarted, Pipes, Running, Unreaped, said
from hands.sessions.payload import Payload, Rejected, one_of
from hands.sessions.untap import REPLACED, TRUST
from hands.tasks import cancel_and_wait

# [LAW:one-source-of-truth] the brioche hands' own package carries.
BRIOCHE = packaged.installed(packaged.BRIOCHE)
# How long brioche has to print how a client reaches it: it does once it has written its trust file.
START_SECONDS = 10.0
# How long brioche has to stop once it is asked to before its group is killed: it stops at once, so this bounds only a
# stop that stalls.
STOP_SECONDS = 5.0
# What of its stderr is read at once: a report is as long as brioche makes it, read whole however long.
READ_BYTES = 64 * 1024
# Each report's own fields, beside the facts it holds (fritter/brioche's report).
_OWN = ("kind", "level", "said")


def carried() -> Path:
    """The brioche hands' package carries, or Unpackaged naming the rebuild (hands.packaged.found)."""
    return packaged.found(BRIOCHE)


class Brioche:
    """brioche, running: the brain's way to Anthropic's API through it, and its end."""

    def __init__(self, running: Running, route: Through, heard: "_Heard") -> None:
        self.running = running
        self.route = route
        self._heard = heard

    async def outlived(self) -> None:
        """Ends once brioche does, raising how it ended and the last it said; or raises as soon as its stderr cannot be
        read to the audit log, as a brioche whose reports go unheard is."""
        # [LAW:no-silent-failure] a brioche that ends while hands runs leaves every request of the brain's unreached, and one
        # started again answers under an authority the running brain does not trust: it stops the run, as the brain's end
        # does. Its stderr ends as brioche does; a read of it that fails stops the run too. Shielded, as the run's end
        # cancels this watch and the drain is the context's to end.
        await asyncio.shield(self._heard.drained)
        ended = await self.running.ended
        raise RuntimeError(f"brioche, the brain's HTTPS proxy, ended while hands was running: {_ended(ended, self._heard.last)}")


@asynccontextmanager
async def proxied(children: Children, to: str, trust: Path, environment: Mapping[str, str], record: Record) -> AsyncGenerator[Brioche]:
    """brioche, sending each request of the brain's for Anthropic's API to hands' proxy at `to`, run in hands'
    `environment` as anything hands starts for its brain is given it (`passed_on`): the proxy it reaches the world
    through, and the certificates it trusts, which brioche writes to `trust` beside its own authority. Stopped as the
    context ends; raises Unstartable, saying why, when it does not start."""
    try:
        binary = carried()
    except packaged.Unpackaged as error:
        raise Unstartable(str(error)) from error
    argv = [str(binary), "--intercept", UPSTREAM, "--to", to, "--trust-variable", TRUST, "--trust-file", str(trust)]
    with ExitStack() as held:
        with ExitStack() as theirs:
            # [LAW:no-ambient-temporal-coupling] each end is held by the stack that closes it from the moment it is made:
            # brioche's own ends close once it is started, so each pipe ends when brioche's end of it does; hands' end of
            # its input is held open for as long as brioche is to run, so hands ending, however it ends, ends that input,
            # and so brioche. No other child inherits it: a pipe's ends are made non-inheritable (PEP 446).
            reads, writes = os.pipe()
            theirs.callback(os.close, reads)
            held.callback(os.close, writes)
            printed, prints = os.pipe()
            printing = held.enter_context(os.fdopen(printed, "rb", 0))
            theirs.callback(os.close, prints)
            complaints, complains = os.pipe()
            saying = held.enter_context(os.fdopen(complaints, "rb", 0))
            theirs.callback(os.close, complains)
            started = children.start(argv, trust.parent, passed_on(environment), Pipes(reads, prints, complains))
        if isinstance(started, NeverStarted):
            raise Unstartable(f"brioche, the brain's HTTPS proxy, at {binary}, {said(started)}")
        heard = _Heard(saying, record)
        try:
            try:
                async with children.clock.timeout(START_SECONDS):
                    environment_printed = await _whole(printing)
            except TimeoutError:
                raise Unstartable(f"brioche, the brain's HTTPS proxy, printed no way to reach it within {START_SECONDS:.0f}s: {_last(heard.last)}") from None
            if not environment_printed:
                # It ended its output with nothing printed, as it does only as it ends: how, and what it said.
                ended = await started.ended
                await asyncio.wait([heard.drained])
                raise Unstartable(f"brioche, the brain's HTTPS proxy, did not start: {_ended(ended, heard.last)}")
            yield Brioche(started, _through(environment_printed, trust), heard)
        finally:
            match await started.stopped(STOP_SECONDS):
                case Unreaped():
                    # Killed and never reaped, it may hold its stderr open for ever: what it said is heard to here.
                    await cancel_and_wait(heard.drained)
                case _:
                    # Its stderr ends with it. A read of it that failed is the brioche watch's to raise (`outlived`).
                    await asyncio.wait([heard.drained])


class _Heard:
    """brioche's stderr, read to its end: each report in the audit log as it is made, and the last of them."""

    def __init__(self, pipe: BinaryIO, record: Record) -> None:
        self.last: BriocheSaid | None = None
        self._record = record
        self.drained = asyncio.create_task(self._drain(pipe), name="brioche's stderr")

    async def _drain(self, pipe: BinaryIO) -> None:
        reader, transport = await _reader(pipe)
        try:
            # Split on its own line breaks, as brioche ends each report with one: a StreamReader's lines are refused
            # past a limit, and a report is as long as brioche makes it.
            pending = b""
            while chunk := await reader.read(READ_BYTES):
                *lines, pending = (pending + chunk).split(b"\n")
                for line in lines:
                    self._hear(line)
            if pending:
                self._hear(pending)
        finally:
            transport.close()

    def _hear(self, line: bytes) -> None:
        self.last = _report(line)
        self._record(self.last)


def _report(line: bytes) -> BriocheSaid:
    """A line brioche wrote on its stderr, as the report it is. [LAW:parse-dont-validate] one that is none, as the trace
    of a panic Go's runtime writes there is, is the error of kind "unframed", said as it was written: never lost."""
    try:
        told = Payload.parse(line)
        facts = {name: _fact(name, value) for name, value in told.fields.items() if name not in _OWN}
        return BriocheSaid(told.text("kind"), one_of(told.text("level"), LEVELS, "a report's level is one of"), told.text("said"), facts)
    except Rejected:
        return BriocheSaid("unframed", "error", line.decode(errors="replace"), {})


def _fact(name: str, value: object) -> str | int:
    """A fact a report holds: a string or a whole number, which JSON's true and false are not."""
    if isinstance(value, str | int) and not isinstance(value, bool):
        return value
    raise Rejected(f"{name} is {value!r}, not a fact brioche reports")


def _through(printed: bytes, trust: Path) -> Through:
    """The brain's way to Anthropic's API through brioche, as brioche printed it: a NAME=VALUE line for each variable an
    interceptor sets, each once, its trust variable naming `trust`. [LAW:parse-dont-validate] anything else is a brioche
    that is not this hands' own, or one cut short, and refuses the start."""
    try:
        text = printed.decode()
    except UnicodeDecodeError as error:
        raise Unstartable(f"brioche, the brain's HTTPS proxy, printed {printed!r}, which is no text: {error}") from None
    # Split on brioche's own line breaks alone: a path may hold any other.
    *lines, end = text.split("\n")
    variables: dict[str, str] = {}
    for line in lines:
        name, equals, value = line.partition("=")
        if not equals or name in variables:
            raise Unstartable(f"brioche, the brain's HTTPS proxy, printed {line!r}, which is no NAME=VALUE line of a variable it has not printed")
        variables[name] = value
    if end or set(variables) != set(REPLACED) or variables[TRUST] != os.path.abspath(trust):
        raise Unstartable(f"brioche, the brain's HTTPS proxy, printed {text!r}, not a line each for {', '.join(REPLACED)}, its {TRUST} naming {os.path.abspath(trust)}")
    return Through(MappingProxyType(variables))


def _ended(ended: Ending | Lost | Unreaped, last: BriocheSaid | None) -> str:
    """How brioche ended, and the last it said, which says why."""
    return f"{said(ended)}: {_last(last)}"


def _last(last: BriocheSaid | None) -> str:
    return "it said nothing" if last is None else f"it said {last.said!r}"


async def _reader(pipe: BinaryIO) -> tuple[asyncio.StreamReader, asyncio.BaseTransport]:
    """A reader of `pipe`, whose transport closes it."""
    reader = asyncio.StreamReader()
    transport, _ = await asyncio.get_running_loop().connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), pipe)
    return reader, transport


async def _whole(pipe: BinaryIO) -> bytes:
    """What is written to `pipe`, to its end."""
    reader, transport = await _reader(pipe)
    try:
        return await reader.read()
    finally:
        transport.close()
