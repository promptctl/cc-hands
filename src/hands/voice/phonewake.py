"""What the phone's page needs to hear the wake word on the phone (design-docs/phone-wake-word/design.md): the Mac's own
wake word models and the runtime that runs them in a browser, served by hands, and the scripts that run them.

The models are the Mac's: the same three openWakeWord files `hands.voice.wake` runs, the word's phrase and the threshold
it is held to, so the phrase that wakes the phone is the phrase that wakes the desk. The runtime is `hands.voice.ortweb`'s.
The page and the worker that runs the models ask nothing of any site but hands.
"""

from aiohttp import web

from hands.clock import Clock
from hands.sessions.audit import Record
from hands.sessions.wide import annotate, begun, fail, unit
from hands.voice import ortweb
from hands.voice.clips import KEPT_SECONDS
from hands.voice.keyed import keyed
from hands.voice.ortweb import File
from hands.voice.pagescript import script_route
from hands.voice.pinnedfile import Release, Unserved
from hands.voice.wakemodels import ROLES, WakeModels
from hands.voice.wakeword import THRESHOLD, Word

# The scripts the page runs the detector with.
SCRIPTS = ("wakestream.js", "detector.js", "detector-worker.js")

# A file named by its version is the same file for as long as that version is what hands serves.
KEPT_A_YEAR = "public, max-age=31536000, immutable"

# The three models, each by the role it plays: the word's is one role whichever word it is, so no path the user chose is in
# a URL.
MODEL_PREFIX = "/wake/models/"


def unavailable(error: Unserved) -> web.Response:
    """What cannot be had, said to the page that asked for it in hands' words, and recorded as the unit's failure."""
    fail(str(error))
    return web.Response(status=503, text=str(error))


def wake_routes(clock: Clock, word: Word, models: WakeModels, runtime: Release, key: str, record: Record) -> list[web.RouteDef]:
    """The detector's scripts and the runtime it runs on, open as the phone's page is; and the word's settings and its
    models, under the phone's `key`, because the phone's own models are its owner's.

    [LAW:one-source-of-truth] the word is the one `phone_pages` is given for the gallery, so the phrase the page is told
    and the phrase the gallery shows are one.
    """

    def ran(file: File) -> web.RouteDef:
        async def serve(_request: web.Request) -> web.StreamResponse:
            with unit("phone.wake.runtime", record, begun(clock)):
                try:
                    kept = await runtime.path(file.pin)
                except Unserved as error:
                    return unavailable(error)
                # [LAW:nothing-unseen] a request that made hands fetch the file, or write over one with other bytes, says so.
                annotate(file=file.pin.name, source=kept.how)
                response = web.FileResponse(kept.path, headers={"Cache-Control": KEPT_A_YEAR})
                response.content_type = file.content_type
                return response

        return web.get(f"{ortweb.PREFIX}{file.pin.name}", serve)

    @keyed(clock, "phone.wake.settings", record, key)
    async def settings(_request: web.Request) -> web.StreamResponse:
        try:
            ensured = await models.ensured(word)
        except Unserved as error:
            return unavailable(error)
        annotate(phrase=word.phrase, **ensured.facts())
        # No-store, as the models are: the phone runs the word the Mac names now, never the one a browser kept.
        return web.json_response(
            {
                "phrase": word.phrase,
                "threshold": THRESHOLD,
                "kept_seconds": KEPT_SECONDS,
                "runtime": f"{ortweb.PREFIX}{ortweb.SCRIPT.pin.name}",
                "models": {role: f"{MODEL_PREFIX}{role}" for role in ROLES},
            },
            headers={"Cache-Control": "no-store"},
        )

    def model(role: str) -> web.RouteDef:
        @keyed(clock, "phone.wake.model", record, key)
        async def serve(_request: web.Request) -> web.StreamResponse:
            annotate(model=role)
            try:
                ensured = await models.ensured(word)
            except Unserved as error:
                return unavailable(error)
            annotate(**ensured.facts())
            return web.FileResponse(ensured.files.by_role()[role], headers={"Cache-Control": "no-store"})

        return web.get(f"{MODEL_PREFIX}{role}", serve)

    return [
        *(script_route(name) for name in SCRIPTS),
        *(ran(file) for file in ortweb.FILES),
        web.get("/wake/settings", settings),
        *(model(role) for role in ROLES),
    ]
