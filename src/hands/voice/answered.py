"""A door's answer to a call answered with an `After`, and the effect that follows it, as every door that answers one
keeps them: hands' MCP server (`hands.brain.mcp`) and the phone's gallery page (`hands.voice.gallerypage`)."""

from aiohttp import web

from hands.sessions.audit import said
from hands.sessions.wide import fail
from hands.voice.tool import After, Effect


async def answered_then(request: web.Request, response: web.StreamResponse, after: After, what: str) -> web.StreamResponse:
    """`response`, the answer to `after`, written whole, then its effect run, as part of the unit of work open here: never
    before, since the effect would cut the answer off. Where the answer cannot be written the effect is run or dropped as
    `after.unanswered` says, and what stopped the answer raises here, for the unit to record.

    [LAW:single-enforcer] the one order of an `After`, and the one place its effect's failure is seen, whichever door
    answers it.
    """
    try:
        await response.prepare(request)
        await response.write_eof()
    except Exception:
        match after.unanswered:
            case "run":
                await _run(after.effect, what)
            case "dropped":
                pass
        raise
    await _run(after.effect, what)
    return response


async def _run(effect: Effect, what: str) -> None:
    try:
        await effect()
    except Exception as error:
        # [LAW:no-silent-failure] the caller has had its answer, so what the effect raised ends here, and is said here.
        failed = f"{what} failed once answered: {type(error).__name__}: {error}"
        said(failed)
        fail(failed)
