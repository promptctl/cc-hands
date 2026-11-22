"""The gallery on the phone's page: every voice hands speaks in, each heard, chosen, and given the line it says when heard,
and hands' own options, the
listening level, the modality, and the model it runs on, set from the same page.

Served beside the phone's page, on its port and under its key (`hands.voice.phonepage`, `hands.voice.keyed`). Each choice
is the one its by-voice tool makes: the voices are `hands.voice.voices`', the level is the desk's, the modality is the
push-to-talk key's, and the model is `hands.voice.tools.switch_model`'s, so a choice on the page and one by voice are
the same change.
"""

import json
from dataclasses import dataclass
from importlib import resources
from typing import Protocol, TypeVar, TypedDict, cast, get_args

from aiohttp import web

from hands.clock import Clock
from hands.core.place import Modality
from hands.sessions.audit import Record
from hands.sessions.payload import Rejected
from hands.sessions.wide import annotate, fail
from hands.voice.desk import LEVELS, Level
from hands.voice.keyed import keyed
from hands.voice.longpoll import version_of, waited
from hands.voice.modelswitch import Heard, OwnModel, switch_model
from hands.voice.trigger import described
from hands.voice.library import Found
from hands.voice.voicechoice import VOICES, Chosen, Voice, parse_voice
from hands.voice.wakeword import Word

# [LAW:one-source-of-truth] the modalities are the ones Modality names; the page offers and accepts exactly those.
MODALITIES: tuple[Modality, ...] = get_args(Modality)

T = TypeVar("T", bound=str)


class Voicing(Protocol):
    """The voices as the page chooses them: the one spoken, each heard, a line heard and kept, one kept, and the library
    searched."""

    def speaking_in(self) -> Chosen: ...

    async def said_by(self, voices: tuple[Chosen, ...]) -> tuple[str, ...]: ...

    async def hear(self, voices: tuple[Chosen, ...]) -> Chosen: ...

    async def preview(self, voice: Chosen, line: str) -> Chosen: ...

    async def keep_line(self, voice: Chosen, line: str) -> str: ...

    async def use(self, voice: Chosen) -> None: ...

    async def search(self, query: str, limit: int) -> Found: ...


class Listening(Protocol):
    """The level the desk listens at: its level now, and a choice of another."""

    @property
    def level(self) -> Level: ...

    async def choose(self, level: Level) -> Level: ...


class Desk(Protocol):
    """Where the user is taken to be: the modality, and a switch of it."""

    @property
    def modality(self) -> Modality: ...

    def switch(self, to: Modality) -> None: ...


@dataclass(frozen=True)
class Gallery:
    """What the page reads and sets. `speaker` is what a model switch's line is heard through, and `models` the models
    the page offers, which `own` refuses where they are not hands' to run on."""

    voices: Voicing
    levels: Listening
    desk: Desk
    own: OwnModel
    speaker: Heard
    wake: Word
    models: tuple[str, ...]


class Shown(TypedDict):
    """What the page shows of hands, as the read answers it: what each choice offers and what hands is on now."""

    voices: tuple[Voice, ...]
    speaking_in: str
    levels: tuple[Level, ...]
    level: Level
    modalities: tuple[Modality, ...]
    modality: Modality
    models: tuple[str, ...]
    running_on: str


def parse_field(body: bytes, field: str) -> str:
    """The text a page's request names in `field`, as its body's JSON; raises Rejected naming what is wrong with it."""
    if named := parse_line(body, field):
        return named
    raise Rejected(f"the page sends {{{field}: ...}}")


def parse_line(body: bytes, field: str) -> str:
    """The text a page's request sends in `field`, blank or not, as its body's JSON; raises Rejected naming what is wrong
    with it. A line is blank where the user cleared it, which is a line of its own: hands' default."""
    try:
        sent: object = json.loads(body)
    except ValueError as error:
        # Not JSON, or not text at all: JSONDecodeError and UnicodeDecodeError are both ValueErrors.
        raise Rejected(f"the page sends JSON: {error}") from error
    if isinstance(sent, dict):
        named: object = cast(dict[str, object], sent).get(field)
        if isinstance(named, str):
            return named
    raise Rejected(f"the page sends {{{field}: ...}}")


def parse_level(text: str) -> Level:
    return _one_of(text, LEVELS, "hands listens at")


def parse_modality(text: str) -> Modality:
    return _one_of(text, MODALITIES, "hands takes you as")


def _one_of(text: str, names: tuple[T, ...], offered: str) -> T:
    """The one name in `names` that `text` is, or a refusal naming what is offered."""
    for name in names:
        if text == name:
            return name
    raise Rejected(f"{offered} {', '.join(names)}, not {text!r}")


def _refusal(error: Exception) -> web.Response:
    """A choice the page asked for that hands could not make, as the page shows it: a bad request is a 400, a failure
    of the machine's a 503, and the event says why."""
    fail(str(error))
    return web.Response(status=400 if isinstance(error, Rejected) else 503, text=str(error))


def gallery_routes(clock: Clock, gallery: Gallery, key: str, record: Record) -> list[web.RouteDef]:
    """The page, its read of the voices and options, and one route per choice.

    [LAW:nothing-unseen] the read is one `gallery.read` event: what it showed, what the page had seen, and how long it
    waited for that to change; a voice's line read for its editor is `gallery.line.read`; each choice one event of its
    own, `gallery.hear`, `gallery.preview`, `gallery.line`, `gallery.use`, `gallery.level`, `gallery.modality`, or
    `gallery.model`, naming what was asked and what came of it.
    """
    page = resources.files("hands.voice").joinpath("gallery.html").read_text()

    async def served(_request: web.Request) -> web.Response:
        return web.Response(text=page, content_type="text/html", headers={"Cache-Control": "no-store"})

    async def look() -> tuple[str, Shown]:
        state: Shown = {
            "voices": VOICES, "speaking_in": str(gallery.voices.speaking_in()),
            "levels": LEVELS, "level": gallery.levels.level,
            "modalities": MODALITIES, "modality": gallery.desk.modality,
            "models": gallery.models, "running_on": gallery.own.running(),
        }
        return version_of(state), state

    # [LAW:one-source-of-truth] the page shows what hands answers here and keeps no picture of its own: what hands speaks
    # in, listens at, takes the user as and runs on now, so a choice made by voice, and the run a model switch's restart
    # brings up, are on the page by the next answer.
    @keyed(clock, "gallery.read", record, key)
    async def read_state(request: web.Request) -> web.Response:
        read = await waited(clock, look, request.query.get("seen"))
        state = read["shown"]
        annotate(speaking_in=state["speaking_in"], level=state["level"], modality=state["modality"], running_on=state["running_on"])
        return web.json_response(read)

    @keyed(clock, "gallery.hear", record, key)
    async def hear_voice(request: web.Request) -> web.Response:
        try:
            heard = parse_voice(parse_field(await request.read(), "voice"))
            speaking = await gallery.voices.hear((heard,))
        except (Rejected, OSError) as error:
            return _refusal(error)
        annotate(heard=str(heard), speaking_in=str(speaking))
        return web.json_response({"heard": str(heard), "speaking_in": str(speaking)})

    # A line is read as its editor opens, not carried on every read: it is the file's now, and the read waits on hands.
    @keyed(clock, "gallery.line.read", record, key)
    async def read_line(request: web.Request) -> web.Response:
        try:
            voice = parse_voice(request.query.get("voice", ""))
            [line] = await gallery.voices.said_by((voice,))
        except (Rejected, OSError) as error:
            return _refusal(error)
        annotate(voice=str(voice))
        return web.json_response({"voice": str(voice), "line": line})

    @keyed(clock, "gallery.preview", record, key)
    async def preview_line(request: web.Request) -> web.Response:
        try:
            body = await request.read()
            heard, line = parse_voice(parse_field(body, "voice")), parse_line(body, "line")
            speaking = await gallery.voices.preview(heard, line)
        except (Rejected, OSError) as error:
            return _refusal(error)
        # The line itself is the user's words, so the event says how long it was, not what it said.
        annotate(heard=str(heard), line_chars=len(line), speaking_in=str(speaking))
        return web.json_response({"heard": str(heard), "speaking_in": str(speaking)})

    @keyed(clock, "gallery.line", record, key)
    async def keep_line(request: web.Request) -> web.Response:
        try:
            body = await request.read()
            voice, line = parse_voice(parse_field(body, "voice")), parse_line(body, "line")
            saying = await gallery.voices.keep_line(voice, line)
        except (Rejected, OSError) as error:
            return _refusal(error)
        annotate(voice=str(voice), line_chars=len(line.strip()), default=not line.strip())
        return web.json_response({"voice": str(voice), "line": saying, "default": not line.strip()})

    @keyed(clock, "gallery.use", record, key)
    async def use_voice(request: web.Request) -> web.Response:
        try:
            voice = parse_voice(parse_field(await request.read(), "voice"))
            await gallery.voices.use(voice)
        except (Rejected, OSError) as error:
            return _refusal(error)
        annotate(voice=str(voice))
        return web.json_response({"speaking_in": str(voice)})

    @keyed(clock, "gallery.library", record, key)
    async def find_voices(request: web.Request) -> web.Response:
        query = request.query.get("q", "")
        try:
            limit = int(request.query.get("limit", "25"))
        except ValueError as error:
            return _refusal(Rejected(f"the limit is a number, not {request.query.get('limit')!r}: {error}"))
        try:
            found = await gallery.voices.search(query, limit)
        except OSError as error:
            return _refusal(error)
        annotate(query=query, matches=len(found.names), listed_at=found.listed_at, from_cache=found.from_cache)
        return web.json_response({"voices": list(found.names), "listed_at": found.listed_at, "from_cache": found.from_cache})

    @keyed(clock, "gallery.level", record, key)
    async def choose_level(request: web.Request) -> web.Response:
        try:
            level = parse_level(parse_field(await request.read(), "level"))
        except Rejected as error:
            return _refusal(error)
        was = await gallery.levels.choose(level)
        annotate(level=level, was=was)
        return web.json_response({"level": level, "was": was, "readback": described(level, gallery.wake)})

    @keyed(clock, "gallery.modality", record, key)
    async def choose_modality(request: web.Request) -> web.Response:
        try:
            modality = parse_modality(parse_field(await request.read(), "modality"))
        except Rejected as error:
            return _refusal(error)
        gallery.desk.switch(modality)
        annotate(modality=modality)
        return web.json_response({"modality": modality})

    @keyed(clock, "gallery.model", record, key)
    async def choose_model(request: web.Request) -> web.Response:
        try:
            model = parse_field(await request.read(), "model")
            annotate(model=model, switching=False)
            # [LAW:one-source-of-truth] the same switch the voice tool makes: its order, its line, and its refusals.
            result = await switch_model(gallery.own, gallery.speaker, model)
        except (Rejected, OSError) as error:
            return _refusal(error)
        annotate(switching=True)
        return web.json_response(result)

    return [
        web.get("/gallery", served),
        web.get("/gallery/state", read_state),
        web.get("/gallery/library", find_voices),
        web.get("/gallery/line", read_line),
        web.post("/gallery/hear", hear_voice),
        web.post("/gallery/preview", preview_line),
        web.post("/gallery/line", keep_line),
        web.post("/gallery/use", use_voice),
        web.post("/gallery/level", choose_level),
        web.post("/gallery/modality", choose_modality),
        web.post("/gallery/model", choose_model),
    ]
