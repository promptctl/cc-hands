"""What the phone's page needs to hear the wake word on the phone (design-docs/phone-wake-word/design.md): the Mac's own
wake word models and the runtime that runs them in a browser, served by hands, and the scripts that run them.

The models are the Mac's: the same three openWakeWord files `hands.voice.wake` runs, the word's phrase and the threshold
it is held to, so the phrase that wakes the phone is the phrase that wakes the desk. The runtime is `hands.voice.ortweb`'s.
The page and the worker that runs the models ask nothing of any site but hands.
"""

from importlib import resources
from pathlib import Path

import aiohttp
from aiohttp import web

from hands.clock import Clock
from hands.sessions.audit import Record
from hands.sessions.wide import annotate, begun, fail, unit
from hands.voice import ortweb
from hands.voice.clips import KEPT_SECONDS
from hands.voice.keyed import keyed
from hands.voice.ortweb import Runtime, Unserved
from hands.voice.wake import EMBEDDING, MELSPECTROGRAM, THRESHOLD, Models
from hands.voice.wakeword import Word, model_of

# The scripts the page runs the detector with: each is served as the phone page's script is, kept by no browser.
SCRIPTS = ("wakestream.js", "detector.js", "detector-worker.js")

# A runtime file's type: a browser runs a module script only when it is told it is one, and compiles WebAssembly only
# from the type it is told is WebAssembly.
CONTENT_TYPES = {"ort.wasm.bundle.min.mjs": "text/javascript", "ort-wasm-simd-threaded.wasm": "application/wasm"}

# A file named by its version is the same file for as long as that version is what hands serves.
KEPT_A_YEAR = "public, max-age=31536000, immutable"

# The three models, each by the name the page asks for it: the word's is one name whichever word it is, so no path the
# user chose is in a URL.
MODELS = ("melspectrogram.onnx", "embedding.onnx", "word.onnx")
MODEL_PREFIX = "/wake/models/"
UNMODELLED = "hands has no wake word model for the phone: it could not fetch or find one; the desk's log says which"


def wake_routes(clock: Clock, word: Word, models: Models, runtime: Runtime, key: str, record: Record) -> list[web.RouteDef]:
    """The detector's scripts and the runtime it runs on, open as the phone's page is; and the word's settings and its
    models, under the phone's `key`, because the phone's own models are its owner's.

    [LAW:one-source-of-truth] the word is the one `phone_pages` is given for the gallery, so the phrase the page is told
    and the phrase the gallery shows are one.
    """
    scripts = {name: resources.files("hands.voice").joinpath(name).read_text() for name in SCRIPTS}

    def script_of(name: str) -> web.RouteDef:
        # Never kept by the browser, as the phone page is not: a page and its scripts are always one run's.
        async def scripted(_request: web.Request) -> web.Response:
            return web.Response(text=scripts[name], content_type="text/javascript", headers={"Cache-Control": "no-store"})

        return web.get(f"/{name}", scripted)

    async def ran(request: web.Request) -> web.StreamResponse:
        with unit("phone.wake.runtime", record, begun(clock)):
            name = request.match_info["name"]
            if request.match_info["version"] != ortweb.VERSION or name not in runtime.files:
                fail(f"no such file in onnxruntime-web {ortweb.VERSION}")
                return web.Response(status=404, text=f"hands serves onnxruntime-web {ortweb.VERSION}, and only its {', '.join(runtime.files)}")
            annotate(file=name)
            try:
                path = await runtime.path(name)
            except Unserved as error:
                fail(str(error))
                return web.Response(status=503, text=str(error))
            response = web.FileResponse(path, headers={"Cache-Control": KEPT_A_YEAR})
            response.content_type = CONTENT_TYPES[name]
            return response

    async def found() -> dict[str, Path] | None:
        """Each model file by the name the page asks it by, fetched first if the Mac has not; or None, the unit failed in hands' words, when one cannot be had."""
        directory = models.directory
        try:
            fetched = await models.ensured(word)
        except (aiohttp.ClientError, OSError, TimeoutError) as error:
            fail(f"hands cannot fetch the wake word's models: {error}")
            return None
        annotate(fetched=", ".join(fetched))
        files: dict[str, Path] = dict(zip(MODELS, (directory / MELSPECTROGRAM, directory / EMBEDDING, model_of(word, directory)), strict=True))
        if absent := [path for path in files.values() if not path.is_file()]:
            fail(f"the wake word's model is not there: {absent[0]}")
            return None
        return files

    @keyed(clock, "phone.wake.settings", record, key)
    async def settings(_request: web.Request) -> web.StreamResponse:
        files = await found()
        if files is None:
            return web.Response(status=503, text=UNMODELLED)
        try:
            for name in runtime.files:
                await runtime.path(name)
        except Unserved as error:
            fail(str(error))
            return web.Response(status=503, text=str(error))
        # No-store, as the models are: the phone runs the word the Mac names now, never the one a browser kept.
        return web.json_response(
            {
                "phrase": word.phrase,
                "threshold": THRESHOLD,
                "kept_seconds": KEPT_SECONDS,
                "runtime": ortweb.PREFIX,
                "models": {name: f"{MODEL_PREFIX}{name}" for name in files},
            },
            headers={"Cache-Control": "no-store"},
        )

    @keyed(clock, "phone.wake.model", record, key)
    async def model(request: web.Request) -> web.StreamResponse:
        name = request.match_info["name"]
        if name not in MODELS:
            fail(f"no such model: {name}")
            return web.Response(status=404, text=f"the models are {', '.join(MODELS)}")
        annotate(model=name)
        files = await found()
        if files is None:
            return web.Response(status=503, text=UNMODELLED)
        return web.FileResponse(files[name], headers={"Cache-Control": "no-store"})

    return [
        *(script_of(name) for name in SCRIPTS),
        web.get("/ort/{version}/{name}", ran),
        web.get("/wake/settings", settings),
        web.get(f"{MODEL_PREFIX}{{name}}", model),
    ]
