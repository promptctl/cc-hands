"""The gallery on the phone's page: every voice hands speaks in, each heard and chosen, and hands' own options, the
listening level, the modality, and the model it runs on, set from the same page.

Served beside the phone's page, on its port and under its key (`hands.voice.phonepage`, `hands.voice.keyed`). Each choice
is the one its by-voice tool makes: the voices are `hands.voice.voices`', the level is the desk's, the modality is the
push-to-talk key's, and the model is `hands.voice.tools.switch_model`'s, so a choice on the page and one by voice are
the same change.
"""

import json
from dataclasses import dataclass
from importlib import resources
from typing import Protocol, TypeVar, cast, get_args

from aiohttp import web

from hands.core.place import Modality
from hands.sessions.audit import Record
from hands.sessions.payload import Rejected
from hands.sessions.wide import annotate, fail
from hands.voice.desk import LEVELS, Level
from hands.voice.keyed import keyed
from hands.voice.modelswitch import Heard, OwnModel, switch_model
from hands.voice.trigger import described
from hands.voice.voices import VOICES, Voice, parse_voice
from hands.voice.wakeword import Word

# [LAW:one-source-of-truth] the modalities are the ones Modality names; the page offers and accepts exactly those.
MODALITIES: tuple[Modality, ...] = get_args(Modality)

T = TypeVar("T", bound=str)


class Voicing(Protocol):
    """The voices as the page chooses them: the one spoken, each heard, and one kept."""

    async def speaking_in(self) -> Voice: ...

    async def hear(self, voices: tuple[Voice, ...]) -> Voice: ...

    async def use(self, voice: Voice) -> None: ...


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


def parse_field(body: bytes, field: str) -> str:
    """The text a page's request names in `field`, as its body's JSON; raises Rejected naming what is wrong with it."""
    try:
        sent: object = json.loads(body)
    except ValueError as error:
        # Not JSON, or not text at all: JSONDecodeError and UnicodeDecodeError are both ValueErrors.
        raise Rejected(f"the page sends JSON: {error}") from error
    if isinstance(sent, dict):
        named: object = cast(dict[str, object], sent).get(field)
        if isinstance(named, str) and named:
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


def gallery_routes(gallery: Gallery, key: str, record: Record) -> list[web.RouteDef]:
    """The page, its read of the voices and options, and one route per choice.

    [LAW:nothing-unseen] the read is one `gallery.read` event; each choice one event of its own, `gallery.hear`,
    `gallery.use`, `gallery.level`, `gallery.modality`, or `gallery.model`, naming what was asked and what came of it.
    """
    page = resources.files("hands.voice").joinpath("gallery.html").read_text()

    async def shown(_request: web.Request) -> web.Response:
        return web.Response(text=page, content_type="text/html", headers={"Cache-Control": "no-store"})

    @keyed("gallery.read", record, key)
    async def read_state(_request: web.Request) -> web.Response:
        try:
            speaking = await gallery.voices.speaking_in()
        except (Rejected, OSError) as error:
            return _refusal(error)
        annotate(speaking_in=speaking, level=gallery.levels.level, modality=gallery.desk.modality)
        return web.json_response({
            "voices": VOICES, "speaking_in": speaking,
            "levels": LEVELS, "level": gallery.levels.level,
            "modalities": MODALITIES, "modality": gallery.desk.modality,
            "models": gallery.models, "running_on": gallery.own.running(),
        })

    @keyed("gallery.hear", record, key)
    async def hear_voice(request: web.Request) -> web.Response:
        try:
            heard = parse_voice(parse_field(await request.read(), "voice"))
            speaking = await gallery.voices.hear((heard,))
        except (Rejected, OSError) as error:
            return _refusal(error)
        annotate(heard=heard, speaking_in=speaking)
        return web.json_response({"heard": heard, "speaking_in": speaking})

    @keyed("gallery.use", record, key)
    async def use_voice(request: web.Request) -> web.Response:
        try:
            voice = parse_voice(parse_field(await request.read(), "voice"))
            await gallery.voices.use(voice)
        except (Rejected, OSError) as error:
            return _refusal(error)
        annotate(voice=voice)
        return web.json_response({"speaking_in": voice})

    @keyed("gallery.level", record, key)
    async def choose_level(request: web.Request) -> web.Response:
        try:
            level = parse_level(parse_field(await request.read(), "level"))
        except Rejected as error:
            return _refusal(error)
        was = await gallery.levels.choose(level)
        annotate(level=level, was=was)
        return web.json_response({"level": level, "was": was, "readback": described(level, gallery.wake)})

    @keyed("gallery.modality", record, key)
    async def choose_modality(request: web.Request) -> web.Response:
        try:
            modality = parse_modality(parse_field(await request.read(), "modality"))
        except Rejected as error:
            return _refusal(error)
        gallery.desk.switch(modality)
        annotate(modality=modality)
        return web.json_response({"modality": modality})

    @keyed("gallery.model", record, key)
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
        web.get("/gallery", shown),
        web.get("/gallery/state", read_state),
        web.post("/gallery/hear", hear_voice),
        web.post("/gallery/use", use_voice),
        web.post("/gallery/level", choose_level),
        web.post("/gallery/modality", choose_modality),
        web.post("/gallery/model", choose_model),
    ]
