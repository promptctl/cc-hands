"""Progress while a session works: each tool call as it is made, the text Claude writes as it writes it, and a burst
of them said as one sentence.

A call is said by what it sets out to do, read off its input, and never by its result: the result is the turn's to
tell when it finishes, and a call is worth hearing about while it runs, before there is any result to tell.
"""

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePath
from typing import cast

from hands.core.spoken import spoken_count


@dataclass(frozen=True)
class Work:
    """A kind of work a call does, and how several calls of it are said together.

    [LAW:one-type-per-behavior] the kinds differ only in their words, so they are values of one type, and a new kind
    is a new constant rather than new code.
    """

    # "{count}" stands for how many different things were done: "edit {count} files".
    several: str
    # One call of it whose input does not say what it was done to.
    one: str


EDITING = Work("edit {count} files", "edit a file")
READING = Work("read {count} files", "read a file")
SEARCHING = Work("run {count} searches", "search the code")
LOOKING_UP = Work("make {count} web lookups", "look something up on the web")
RUNNING = Work("run {count} commands", "run a command")
DELEGATING = Work("start {count} subagents", "start a subagent")
PLANNING = Work("update its plan", "update its plan")
USING = Work("use {count} tools", "use a tool")
MESSAGING = Work("send {count} messages", "send a message")
# A call that asks the user: the voice speaks it as its PermissionRequest hook arrives, and the page shows it.
ASKING = Work("ask {count} questions", "ask the user")
# Text Claude wrote, said by a summary of it, which every burst of it has: alone, it is never said as this work's.
EXPLAINING = Work("explain {count} things", "explain something")
# Text Claude wrote that could not be summarised: said to have been written, since what it says is not known, and never
# read out.
WRITING = Work("write {count} things", "write something")


@dataclass(frozen=True)
class Doing:
    """One call as it is made: its kind of work, and what it is said as when it is the only one of its kind; None when
    its input does not say what it was done to, and then no other call is the same thing done."""

    work: Work
    alone: str | None


# What a call is done to, each kind rendered by the medium that tells it: the voice says a file by its name, holds back
# what is code, and leaves out what a listener cannot follow; the page shows it whole.
@dataclass(frozen=True)
class Path:
    """A file: the voice says its name, the page its path with the session's cwd taken off the front."""

    value: str


@dataclass(frozen=True)
class Words:
    """What Claude wrote of the call: a description, a query, a subject, a question. The page adds an `aside`, a subagent's
    type, which the voice leaves out."""

    text: str
    aside: str | None = None


@dataclass(frozen=True)
class Pattern:
    """A search: the voice says it only when it is words, since a regular expression or a glob read out is code; the page
    always, with the path it searched `within`."""

    value: str
    within: str | None


@dataclass(frozen=True)
class Command:
    """A command's first line: the page shows it, and the voice never reads code out."""

    first_line: str


@dataclass(frozen=True)
class Site:
    """A page by its site: a whole address read out is noise."""

    host: str


@dataclass(frozen=True)
class Skill:
    """A skill and its arguments: the page shows it as it was typed, the voice by its name."""

    name: str
    args: str | None

    @property
    def typed(self) -> str:
        """The skill as a user types it, `/name args`. [LAW:one-source-of-truth] the one wording of a skill's invocation: the
        page's line, a forked skill's task and a subagent's job all take it from here."""
        return f"/{self.name} {self.args}" if self.args else f"/{self.name}"


@dataclass(frozen=True)
class Named:
    """A recipient called by a name a listener can follow: a teammate's, or `main`."""

    name: str


@dataclass(frozen=True)
class ById:
    """A recipient called by the id of the agent it is: the page shows it, the voice leaves it out."""

    id: str


Recipient = Named | ById

# An agent's id as Claude Code mints it: `a` and sixteen hex digits.
_AGENT_ID = re.compile(r"a[0-9a-f]{16}")


def recipient(to: str) -> Recipient:
    return ById(to) if _AGENT_ID.fullmatch(to) else Named(to)


@dataclass(frozen=True)
class Message:
    """A message to another agent: the page shows who it is to, the voice says a name and leaves out an agent's id."""

    to: Recipient
    text: str | None


@dataclass(frozen=True)
class ToolInput:
    """A tool no row knows, by its input: the page shows it as JSON, cut to a line, and the voice says only the tool."""

    json: str


Part = Path | Words | Pattern | Command | Site | Skill | Message | ToolInput

# The states a task is in as Claude Code's TaskUpdate names them, and how each is said; a state not here is left unsaid.
_TASK_STATES: Mapping[str, str] = {
    "pending": "pending",
    "in_progress": "in progress",
    "completed": "completed",
    "deleted": "deleted",
}

# How much of one call the page shows on its line.
LINE = 120


@dataclass(frozen=True)
class Intent:
    """What a call sets out to do, as the voice and the page each tell it: the kind of work, the act, and what it is done
    to. [LAW:one-source-of-truth] the one value per call; `heard` and `line` differ only in how a medium renders a part, so
    no tool's act is worded twice. A call whose input says nothing of what it was done to is its act alone."""

    work: Work
    act: str
    of: Part | None

    def heard(self) -> Doing | None:
        """What the voice says of it; None for a call that asks the user, which is spoken as it is asked."""
        if self.work is ASKING:
            return None
        said_of = None if self.of is None else _voiced(self.of)
        return Doing(self.work, None if said_of is None else _joined(self.act, said_of))

    def line(self, cwd: PurePath) -> str:
        """The one line the page shows of it, `cwd` taken off the front of a path under it."""
        return _cut(self.act if self.of is None else _joined(self.act, _shown(self.of, cwd)), LINE)


def intent(tool: str, input: Mapping[str, object]) -> Intent:
    """What a call of `tool` with this input sets out to do, for every tool Claude Code calls."""
    return _SAID.get(tool, _used)(tool, input)


def doing(tool: str, input: Mapping[str, object]) -> Doing | None:
    """What the voice says of a call; None for a call that asks the user, which is spoken as it is asked and needs no
    progress besides."""
    return intent(tool, input).heard()


def said(doings: Sequence[Doing]) -> str:
    """A burst of calls as one spoken clause, each kind of work once, in the order it was first done: "edit ten files,
    then run the test suite". A kind done to one thing only is said as that thing."""
    # Ordered and deduplicated in one step: ten edits of one file are one thing done, and eight commands that say
    # nothing of themselves are eight, each its own by where it stands.
    grouped: dict[Work, dict[str | int, str]] = {}
    for at, each in enumerate(doings):
        grouped.setdefault(each.work, {})[at if each.alone is None else each.alone] = each.work.one if each.alone is None else each.alone
    return ", then ".join(
        next(iter(things.values())) if len(things) == 1 else work.several.format(count=spoken_count(len(things))) for work, things in grouped.items()
    )


def _ran(_tool: str, input: Mapping[str, object]) -> Intent:
    # Claude Code asks for a description of every command, in the imperative ("Run the test suite"), which is what
    # this sentence is made of; the command itself is code, and never read aloud.
    description = _text(input, "description")
    if description is not None:
        return Intent(RUNNING, "", Words(lowered(description)))
    command = _text(input, "command")
    return Intent(RUNNING, RUNNING.one, None) if command is None else Intent(RUNNING, "", Command(command.splitlines()[0]))


def _on_file(work: Work, act: str) -> Callable[[str, Mapping[str, object]], Intent]:
    def row(_tool: str, input: Mapping[str, object]) -> Intent:
        file = _file(input)
        return Intent(work, work.one, None) if file is None else Intent(work, act, Path(file))

    return row


def _searched(_tool: str, input: Mapping[str, object]) -> Intent:
    pattern = _text(input, "pattern")
    return Intent(SEARCHING, SEARCHING.one, None) if pattern is None else Intent(SEARCHING, "search for", Pattern(pattern, _text(input, "path")))


def _looked_up(_tool: str, input: Mapping[str, object]) -> Intent:
    query = _text(input, "query")
    return Intent(LOOKING_UP, LOOKING_UP.one, None) if query is None else Intent(LOOKING_UP, "search the web for", Words(query))


def _fetched(_tool: str, input: Mapping[str, object]) -> Intent:
    host = _HOST.match(_text(input, "url") or "")
    return Intent(LOOKING_UP, LOOKING_UP.one, None) if host is None else Intent(LOOKING_UP, "read a page on", Site(host[1]))


def _dispatched(_tool: str, input: Mapping[str, object]) -> Intent:
    description = _text(input, "description")
    if description is None:
        return Intent(DELEGATING, DELEGATING.one, None)
    return Intent(DELEGATING, "start a subagent to", Words(lowered(description), aside=_text(input, "subagent_type")))


def _skilled(_tool: str, input: Mapping[str, object]) -> Intent:
    name = _text(input, "skill")
    return Intent(USING, USING.one, None) if name is None else Intent(USING, "", Skill(name, _text(input, "args")))


def _messaged(_tool: str, input: Mapping[str, object]) -> Intent:
    # `to` is a name or an agent's id; `summary` is the sender's own line for it, absent on some calls.
    to = _text(input, "to")
    text = _text(input, "summary") or _first_line(_text(input, "message"))
    return Intent(MESSAGING, MESSAGING.one, None) if to is None else Intent(MESSAGING, "send a message", Message(recipient(to), text))


def _tasked(act: str, key: str) -> Callable[[str, Mapping[str, object]], Intent]:
    def row(_tool: str, input: Mapping[str, object]) -> Intent:
        named = _text(input, key)
        if named is None:
            return Intent(PLANNING, PLANNING.one, None)
        state = _TASK_STATES.get(_text(input, "status") or "")
        return Intent(PLANNING, act, Words(named if state is None or key == "subject" else f"{named}: {state}"))

    return row


def _planned(_tool: str, _input: Mapping[str, object]) -> Intent:
    return Intent(PLANNING, PLANNING.one, None)


def _asked(_tool: str, input: Mapping[str, object]) -> Intent:
    questions = input.get("questions")
    first = cast(list[object], questions)[0] if isinstance(questions, list) and questions else None
    question = _text(cast(Mapping[str, object], first), "question") if isinstance(first, dict) else None
    return Intent(ASKING, ASKING.one, None) if question is None else Intent(ASKING, "", Words(question))


def _approval(_tool: str, _input: Mapping[str, object]) -> Intent:
    return Intent(ASKING, "asked to approve its plan", None)


def _used(tool: str, input: Mapping[str, object]) -> Intent:
    # An MCP tool is named mcp__<server>__<tool>: the tool's own name is the part a listener can follow.
    return Intent(USING, f"use {tool.rsplit('__', 1)[-1].replace('_', ' ')}", ToolInput(json.dumps(dict(input), ensure_ascii=False, default=str)))


# [LAW:one-type-per-behavior] one table keyed by tool name, as the step recognisers are; a tool not in it is used.
_SAID: Mapping[str, Callable[[str, Mapping[str, object]], Intent]] = {
    "Bash": _ran,
    "Edit": _on_file(EDITING, "edit"),
    "MultiEdit": _on_file(EDITING, "edit"),
    "NotebookEdit": _on_file(EDITING, "edit"),
    "Write": _on_file(EDITING, "write"),
    "Read": _on_file(READING, "read"),
    "Grep": _searched,
    "Glob": _searched,
    "WebSearch": _looked_up,
    "WebFetch": _fetched,
    "Task": _dispatched,
    "Agent": _dispatched,
    "Skill": _skilled,
    "SendMessage": _messaged,
    "TodoWrite": _planned,
    "TaskCreate": _tasked("add task", "subject"),
    "TaskUpdate": _tasked("update task", "taskId"),
    "AskUserQuestion": _asked,
    "ExitPlanMode": _approval,
}


def _joined(act: str, what: str) -> str:
    return f"{act} {what}".strip()


def _voiced(part: Part) -> str | None:
    """What the voice says of a part, "" where the act says all there is, and None where a listener cannot be told it."""
    match part:
        case Path(value=value):
            return PurePath(value).name
        case Words(text=text):
            return text
        case Pattern(value=value):
            return value if _WORDS.fullmatch(value) else None
        case Command():
            return None
        case Site(host=host):
            return host
        case Skill(name=name):
            return f"use the {name.rsplit(':', 1)[-1].replace('-', ' ').replace('_', ' ')} skill"
        case Message(to=Named(name=name), text=str() as text):
            return f"to {name} saying {text}"
        case Message(to=Named(name=name)):
            return f"to {name}"
        case Message(text=str() as text):
            return f"saying {text}"
        case Message():
            return ""
        case ToolInput():
            return ""


def _shown(part: Part, cwd: PurePath) -> str:
    match part:
        case Path(value=value):
            return _relative(value, cwd)
        case Words(text=text, aside=str() as aside):
            return f"{text} ({aside})"
        case Words(text=text):
            return text
        case Pattern(value=value, within=str() as within):
            return f"{value} in {_relative(within, cwd)}"
        case Pattern(value=value):
            return value
        case Command(first_line=first_line):
            return first_line
        case Site(host=host):
            return host
        case Skill() as skill:
            return skill.typed
        case Message(to=to, text=str() as text):
            return f"to {_called(to)}: {text}"
        case Message(to=to):
            return f"to {_called(to)}"
        case ToolInput(json=json_text):
            return json_text


def _called(to: Recipient) -> str:
    match to:
        case Named(name=name):
            return name
        case ById(id=id):
            return id


def _relative(path: str, cwd: PurePath) -> str:
    """A path as the session sees it: with its cwd taken off the front where it lies under it, whole where it does not."""
    try:
        return str(PurePath(path).relative_to(cwd))
    except ValueError:
        return path


def _cut(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def _first_line(text: str | None) -> str | None:
    return None if text is None else text.splitlines()[0]


# What a listener can follow: words, and the dashes, dots, and underscores names are spelled with.
_WORDS = re.compile(r"[\w][\w .'-]*")

# The host of a URL: what follows the scheme, up to its port, path, query, or fragment.
_HOST = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://(?:[^@/]*@)?([^/:?#]+)")


def _file(input: Mapping[str, object]) -> str | None:
    """The file a call works on, as the call gave it."""
    return _text(input, "file_path") or _text(input, "notebook_path")


def _text(input: Mapping[str, object], key: str) -> str | None:
    value = input.get(key)
    return value.strip() if isinstance(value, str) and value.strip() else None


def explained(summary: str) -> Doing:
    """Text Claude wrote, said as its summary: a phrase in the imperative, as the middle of a sentence."""
    return Doing(EXPLAINING, lowered(summary.strip().rstrip(".!")))


def lowered(phrase: str) -> str:
    """An imperative phrase as the middle of a sentence: "Run the tests" is "run the tests", and "PR" stays "PR"."""
    first, rest = phrase[:1], phrase[1:]
    return first.lower() + rest if rest[:1].islower() or not rest else phrase


# [LAW:no-ambient-temporal-coupling] a burst is over once the session has made no new call for SETTLE seconds, or once
# its first call has waited LONGEST, as in a debounce with a maximum wait, so a session that never pauses is still heard.
# Both are read on the reducer's own clock, at its tick, which is the one owner of when gathered progress is told.
SETTLE = 2.0
LONGEST = 8.0


@dataclass(frozen=True)
class Gathering:
    """Calls a running turn made and text it wrote that nobody has been told of yet, and when the first and the last of
    them came."""

    doings: tuple[Doing, ...]
    # Every line written since the burst began, as Claude Code displayed it; empty when it wrote none.
    written: str
    first: float
    last: float

    def joined(self, doings: Sequence[Doing], at: float) -> "Gathering":
        return Gathering((*self.doings, *doings), self.written, self.first, at)

    def wrote(self, text: str, at: float) -> "Gathering":
        return Gathering(self.doings, self.written + text, self.first, at)

    def due(self) -> float:
        """When the burst is told: once it has settled, and no later than its first call's longest wait."""
        return min(self.last + SETTLE, self.first + LONGEST)
