"""A tool the model can call: the schema it sees, read once from a plain async body's signature and docstring, and the
body that answers a call. [LAW:decomposition] it is the act alone, the same to any caller: what a call is to a voice
turn is `hands.voice.braindoor`'s, and hands' MCP server is the adapter over tools.
"""

import functools
import inspect
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from typing import Literal, Protocol, cast, get_args, get_origin, get_type_hints, is_typeddict

import docstring_parser

from hands.clock import Clock
from hands.sessions.audit import Record
from hands.sessions.wide import annotate, begun, fail, unit

from hands.core.playback import Labelled


# What the model is handed back from a call: an object, as every tool API carries a result.
Result = Mapping[str, object]
JsonSchema = Mapping[str, object]


class Effect(Protocol):
    """What is done once the caller has the answer, holding nothing until it is run, so an answer dropped unanswered
    leaves nothing held."""

    async def __call__(self) -> None: ...


# What becomes of an effect whose answer cannot be written. "run": someone was already told of it, as the user hears a
# switch said before it is answered, so it is done as they were told. "dropped": only the answer would have told anyone.
Unanswered = Literal["run", "dropped"]


@dataclass(frozen=True)
class After:
    """A result, and an effect that would cut its answer off if it ran first, as keeping the model hands runs on restarts
    hands. [LAW:effects-at-boundaries] a body describes the effect, and the door that answers runs it once the answer is
    written, or as `unanswered` says where it cannot be (`hands.voice.answered`)."""

    result: Result
    effect: Effect
    unanswered: Unanswered


type Answer = Result | After
type Body[A: Answer] = Callable[..., Awaitable[A]]


def result_of(answer: Answer) -> Result:
    """What the caller is handed back for an answer."""
    match answer:
        case After(result=result):
            return result
        case _:
            return answer


def with_result(answer: Answer, result: Callable[[Result], Result]) -> Answer:
    """The answer with its result made by `result` from the one it had, and the effect, if any, kept with it."""
    match answer:
        case After(result=had):
            return replace(answer, result=result(had))
        case _:
            return result(answer)


@dataclass(frozen=True)
class Tool[A: Answer]:
    """One tool: the schema the model sees, and the body that answers a call. Typed by what its body can answer, so one
    built to answer a `Result` alone is known as one by whoever holds it."""

    name: str
    description: str
    # [LAW:one-source-of-truth] each argument's schema and which are required, as listed and as the body is held to them.
    properties: Mapping[str, JsonSchema]
    required: tuple[str, ...]
    body: Body[A]

    @property
    def input_schema(self) -> JsonSchema:
        """The arguments' schema as MCP lists it."""
        return {"type": "object", "properties": self.properties, "required": list(self.required)}


@dataclass(frozen=True)
class Called:
    """What the model gave a tool, and the result it was handed back: None while the body runs, and for a body that raised."""

    arguments: Mapping[str, object]
    result: Result | None


def audited[A: Answer](clock: Clock, tool: Tool[A], record: Record) -> Tool[A]:
    """The tool, each call one wide event: the tool's name, what it was given, and what the model was handed back, ending
    failed where the body raised or the result refuses the call."""

    # [LAW:single-enforcer] one unit on every body, whichever adapter calls it, so no call goes unseen [LAW:nothing-unseen].
    @functools.wraps(tool.body)
    async def call(**arguments: object) -> A:
        with unit("tool.run", record, begun(clock)):
            annotate(tool=tool.name, called=Called(arguments, None))
            answer = await tool.body(**arguments)
            result = result_of(answer)
            annotate(called=Called(arguments, result))
            match result:
                case {"error": refused}:
                    fail(str(refused))
                case _:
                    pass
            return answer

    return replace(tool, body=call)


def read_back(says: str, label: str) -> Result:
    """A result hands says `says` for, as written, and tells the brain of by `label`, never by its text.
    [LAW:single-enforcer] the one shape of such a result, which `said_back` reads."""
    return {"says": says, "as": label}


def said_back(result: Result) -> Labelled | None:
    """The line a result `read_back` made has hands say; None for any other result, which hands says nothing for."""
    match result:
        case {"says": str() as says, "as": str() as label}:
            return Labelled(says, label)
        case _:
            return None


def tool[A: Answer](body: Body[A]) -> Tool[A | Result]:
    """The body as a tool, its schema read once from its signature and docstring: the name, the text, and each argument's type and line."""
    # [LAW:parse-dont-validate] a signature with a type no schema says is refused here, as the daemon builds its tools.
    docstring = docstring_parser.parse(inspect.getdoc(body) or "")
    lines = {param.arg_name: param.description or "" for param in docstring.params}
    hints = get_type_hints(body)
    parameters = inspect.signature(body).parameters.values()
    properties = {parameter.name: {**_schema(hints[parameter.name]), "description": lines.get(parameter.name, "")} for parameter in parameters}
    required = tuple(parameter.name for parameter in parameters if parameter.default is inspect.Parameter.empty)
    return Tool(body.__name__, (docstring.description or "").strip(), properties, required, _closed(body, properties))


def _closed[A: Answer](body: Body[A], properties: Mapping[str, JsonSchema]) -> Body[A | Result]:
    """The body, refusing a call whose arguments do not fit its signature or the schema the model was shown for them,
    so the model is told, as a result, and can call again."""
    # [LAW:single-enforcer] every adapter calls the tool's body, so the schema it advertises is held here, once.
    # [LAW:one-source-of-truth] held against the schema itself, so what is advertised and what is taken cannot differ.
    signature = inspect.signature(body)

    @functools.wraps(body)
    async def call(**arguments: object) -> A | Result:
        # A null is an argument left out, which some models send for one they leave empty: it takes the argument's
        # default, and is refused where there is none.
        sent = {name: value for name, value in arguments.items() if value is not None}
        try:
            signature.bind(**sent)
        except TypeError as error:
            return {"error": f"{body.__name__} was called with the wrong arguments: {error}"}
        refused = [misfit for name, value in sent.items() for misfit in _misfits(properties[name], value, name)]
        return {"error": "; ".join(refused)} if refused else await body(**sent)

    return call


def _misfits(schema: JsonSchema, value: object, what: str) -> list[str]:
    """Each way a value the model sent does not fit a schema `_schema` wrote, in words the model can correct it from.

    A value of the wrong type is named by its type, never echoed: it may be a whole draft.
    """
    got = type(value).__name__
    match schema:
        case {"enum": enum} if isinstance(value, str):
            allowed = cast(list[str], enum)
            return [] if value in allowed else [f"{value!r} is no {what}; it is one of {', '.join(allowed)}"]
        case {"type": "string"}:
            return [] if isinstance(value, str) else [f"{what} should be a string, got {got}"]
        case {"type": "boolean"}:
            return [] if isinstance(value, bool) else [f"{what} should be true or false, got {got}"]
        case {"type": "integer"}:
            # A bool is an int to Python, and no integer to the model.
            return [] if isinstance(value, int) and not isinstance(value, bool) else [f"{what} should be an integer, got {got}"]
        case {"type": "array", "items": items}:
            if not isinstance(value, list):
                return [f"{what} should be a list, got {got}"]
            return [misfit for index, item in enumerate(cast(list[object], value)) for misfit in _misfits(cast(JsonSchema, items), item, f"{what}[{index}]")]
        case {"type": "object", "properties": fields, "required": required}:
            if not isinstance(value, dict):
                return [f"{what} should be an object, got {got}"]
            given = cast(dict[str, object], value)
            return [
                *(f"{what} has no {name}" for name in cast(list[str], required) if name not in given),
                *(misfit for name, field in cast(Mapping[str, JsonSchema], fields).items() if name in given for misfit in _misfits(field, given[name], f"{what}.{name}")),
            ]
        case _:
            raise TypeError(f"a tool argument's schema {schema!r} is none written here")


def _schema(hint: object) -> JsonSchema:
    match hint:
        case type() if hint is str:
            return {"type": "string"}
        case type() if hint is bool:
            return {"type": "boolean"}
        case type() if hint is int:
            return {"type": "integer"}
        case type() if is_typeddict(hint):
            fields = get_type_hints(hint)
            return {"type": "object", "properties": {name: _schema(field) for name, field in fields.items()}, "required": list(fields)}
        case _ if get_origin(hint) is Literal:
            return {"type": "string", "enum": list(get_args(hint))}
        case _ if get_origin(hint) is list:
            [item] = get_args(hint)
            return {"type": "array", "items": _schema(item)}
        case _:
            raise TypeError(f"a tool argument typed {hint!r} has no schema here")
