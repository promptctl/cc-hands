"""The sessions on the phone's page: each live Claude Code session as the brain's listing says it, and a Stop that
interrupts the one it is beside, as Escape at that session's keyboard does.

Served beside the phone's page, on its port and under its key (`hands.voice.phonepage`, `hands.voice.keyed`).

[LAW:one-source-of-truth] a row is `standing`'s, the listing the brain's briefing reads, and a Stop is `interrupt`, what
the brain's interrupt_session returns: the page and the brain list sessions, stop one, and say what came of it one way.
"""

import hashlib
import json

from aiohttp import web

from hands.sessions.audit import Record
from hands.sessions.registry import Sessions
from hands.sessions.wide import annotate, count
from hands.voice.keyed import keyed
from hands.voice.longpoll import waited
from hands.voice.tools import interrupt, standing, state_and_mode


def shown(sessions: Sessions) -> str:
    """What every row shows but its name, as one mark. Read off the registry alone, so a read waiting on it reads no
    transcript; and a session streaming text, which leaves its state line as it was, leaves the mark as it was too."""
    rows = {id: state_and_mode(session) for id, session in sessions.live_sessions().items()}
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()[:16]


def session_routes(sessions: Sessions, key: str, record: Record) -> list[web.RouteDef]:
    """The read of the sessions, and a Stop.

    [LAW:nothing-unseen] each read is one `sessions.read` event: how many sessions it gave, how long it waited for what
    the rows show to change, and how many looks that took; each Stop one `sessions.interrupt`: the session, and what was
    read back or why nothing could be.
    """

    async def look() -> tuple[str, None]:
        return shown(sessions), None

    @keyed("sessions.read", record, key, counts=("sessions",))
    async def read(request: web.Request) -> web.Response:
        mark, _ = await waited(look, request.query.get("seen"))
        # Named once what the rows show has changed: a name is read off its session's transcript.
        listed = standing(sessions)
        count(sessions=len(listed))
        return web.json_response({"seen": mark, "sessions": listed})

    @keyed("sessions.interrupt", record, key)
    async def stop(request: web.Request) -> web.Response:
        session = request.match_info["session"]
        said = await interrupt(sessions, session)
        annotate(session=session, **{name: str(text) for name, text in said.items()})
        return web.json_response(said)

    return [web.get("/sessions", read), web.post("/sessions/{session}/interrupt", stop)]
