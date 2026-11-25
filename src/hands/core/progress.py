"""Progress while a session works: each tool call as it is made, the text Claude writes as it writes it, and a burst
of them said as one sentence.

A call is said by what it sets out to do, read off its input, and never by its result: the result is the turn's to
tell when it finishes, and a call is worth hearing about while it runs, before there is any result to tell.
"""

import json
import re
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import PurePath
from typing import cast
from urllib.parse import urlsplit

from hands.core.skill import Skill
from hands.core.spoken import spoken_count
from hands.core.turn import AGENT_ID, AgentId


@dataclass(frozen=True)
class Work:
    """A kind of work a call does, how several calls of it are said together, and how one is said of what it was done to.

    [LAW:one-type-per-behavior] the kinds differ only in their words, so they are values of one type, and a new kind
    is a new constant rather than new code. A kind has one act, so what a call is done to is all that tells two of it apart.
    """

    # "{count}" stands for how many different things were done: "edit {count} files".
    several: str
    # One call of it whose input does not say what it was done to.
    one: str
    # What is done to a thing: "edit" of a file. Empty where the thing says all there is, as a command's own words do.
    act: str = ""
    # Its calls are spoken as they arrive by something other than progress, which leaves them out.
    hooked: bool = False


EDITING = Work("edit {count} files", "edit a file", "edit")
WRITING_FILES = Work("write {count} files", "write a file", "write")
READING = Work("read {count} files", "read a file", "read")
SEARCHING = Work("run {count} searches", "search the code", "search for")
LOOKING_UP = Work("make {count} web lookups", "look something up on the web", "search the web for")
FETCHING = Work("read {count} pages", "read a page", "read a page on")
RUNNING = Work("run {count} commands", "run a command")
DELEGATING = Work("start {count} subagents", "start a subagent", "start a subagent to")
PLANNING = Work("update its plan", "update its plan")
ADDING_TASKS = Work("add {count} tasks", "add a task", "add task")
UPDATING_TASKS = Work("update {count} tasks", "update a task", "update task")
SKILLS = Work("use {count} skills", "use a skill")
USING = Work("use {count} tools", "use a tool", "use")
MESSAGING = Work("send {count} messages", "send a message", "send a message")
# Calls that ask the user: the voice speaks each as its PermissionRequest hook arrives, and the page shows it.
ASKING = Work("ask {count} questions", "ask the user", hooked=True)
APPROVING = Work("ask for {count} approvals", "asked to approve its plan", hooked=True)
# Text Claude wrote, said by a summary of it, which every burst of it has: alone, it is never said as this work's.
EXPLAINING = Work("explain {count} things", "explain something")
# Text Claude wrote that could not be summarised: said to have been written, since what it says is not known, and never
# read out.
WRITING = Work("write {count} things", "write something")


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
class Named:
    """A recipient called by a name a listener can follow: a teammate's, or `main`."""

    name: str


@dataclass(frozen=True)
class ById:
    """A recipient called by the id of the agent it is: the page shows it, the voice leaves it out."""

    id: AgentId


Recipient = Named | ById


def recipient(to: str) -> Recipient:
    return ById(AgentId(to)) if AGENT_ID.fullmatch(to) else Named(to)


@dataclass(frozen=True)
class Message:
    """A message to another agent: the page shows who it is to, the voice says a name and leaves out an agent's id."""

    to: Recipient
    text: str | None


@dataclass(frozen=True)
class ToolInput:
    """A tool no row knows, and its input as JSON, kept to what the page shows of it: the voice says only the tool. Two
    calls of one tool are one thing to a listener, whatever their inputs, so the input is no part of what makes them equal."""

    tool: str
    json: str = field(compare=False)


Part = Path | Words | Pattern | Command | Site | Skill | Message | ToolInput


@dataclass(frozen=True)
class Doing:
    """One call as it is made: its kind of work, what it is said as when it is the only one of its kind, and what it was
    done to; `alone` is None where the voice has nothing to say of it, and `of` where its input does not say.

    `of` is what makes two calls one thing done: an edit of one file twice is that file, and two files of one name in two
    directories are two. Where it is None, `alone` stands for it, and a call that has neither is its own thing."""

    work: Work
    alone: str | None
    of: Part | None = None

    def same(self, at: int) -> Hashable:
        """What this call is the same thing done as, `at` being where it stands among the calls said together."""
        return at if self.of is None and self.alone is None else self.alone if self.of is None else self.of


# How much of one call the page shows on its line.
LINE = 120


@dataclass(frozen=True)
class Intent:
    """What a call sets out to do, as the voice and the page each tell it: its kind of work, which holds its act, and what
    it is done to. [LAW:one-source-of-truth] the one value per call; `heard` and `line` word it from the same two things,
    each rendering a part as its medium needs and the voice leaving out the parts a listener cannot follow. A call whose
    input says nothing of what it was done to is its kind of work, one."""

    work: Work
    of: Part | None

    def heard(self) -> Doing | None:
        """What the voice says of it; None for a call that asks the user, which is spoken as it is asked."""
        if self.work.hooked:
            return None
        match self.of:
            case None:
                return Doing(self.work, None)
            case part:
                said_of = _voiced(part)
                return Doing(self.work, None if said_of is None else _joined(self.work.act, said_of), part)

    def line(self, cwd: PurePath) -> str:
        """The one line the page shows of it, `cwd` taken off the front of a path under it, and cut to LINE."""
        return _cut(_first_line(self.work.one if self.of is None else _joined(self.work.act, _shown(self.of, cwd))), LINE)


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
    # Ordered and deduplicated in one step: ten edits of one file are one thing done, and eight calls that say nothing of
    # what they were done to are eight, each its own by where it stands.
    grouped: dict[Work, dict[Hashable, str]] = {}
    for at, each in enumerate(doings):
        grouped.setdefault(each.work, {})[each.same(at)] = each.work.one if each.alone is None else each.alone
    return ", then ".join(
        next(iter(things.values())) if len(things) == 1 else work.several.format(count=spoken_count(len(things))) for work, things in grouped.items()
    )


def _ran(_tool: str, input: Mapping[str, object]) -> Intent:
    # Claude Code asks for a description of every command, in the imperative ("Run the test suite"), which is what
    # this sentence is made of; the command itself is code, and never read aloud.
    description = _text(input, "description")
    if description is not None:
        return Intent(RUNNING, Words(lowered(description)))
    command = _text(input, "command")
    return Intent(RUNNING, None if command is None else Command(_first_line(command)))


def _on_file(work: Work) -> Callable[[str, Mapping[str, object]], Intent]:
    def row(_tool: str, input: Mapping[str, object]) -> Intent:
        file = _file(input)
        return Intent(work, None if file is None else Path(file))

    return row


def _searched(_tool: str, input: Mapping[str, object]) -> Intent:
    pattern = _text(input, "pattern")
    return Intent(SEARCHING, None if pattern is None else Pattern(pattern, _text(input, "path")))


def _looked_up(_tool: str, input: Mapping[str, object]) -> Intent:
    query = _text(input, "query")
    return Intent(LOOKING_UP, None if query is None else Words(query))


def _fetched(_tool: str, input: Mapping[str, object]) -> Intent:
    host = _host(_text(input, "url"))
    return Intent(FETCHING, None if host is None else Site(host))


def _dispatched(_tool: str, input: Mapping[str, object]) -> Intent:
    description = _text(input, "description")
    return Intent(DELEGATING, None if description is None else Words(lowered(description), aside=_text(input, "subagent_type")))


def _skilled(_tool: str, input: Mapping[str, object]) -> Intent:
    name = _text(input, "skill")
    return Intent(SKILLS, None if name is None else Skill.of(name, input.get("args")))


def _messaged(_tool: str, input: Mapping[str, object]) -> Intent:
    # `to` is a name or an agent's id; `summary` is the sender's own line for it, absent on some calls.
    to = _text(input, "to")
    message = _text(input, "message")
    text = _text(input, "summary") or (None if message is None else _first_line(message))
    return Intent(MESSAGING, None if to is None else Message(recipient(to), text))


def _task_added(_tool: str, input: Mapping[str, object]) -> Intent:
    subject = _text(input, "subject")
    return Intent(ADDING_TASKS, None if subject is None else Words(subject))


def _task_updated(_tool: str, input: Mapping[str, object]) -> Intent:
    task = _text(input, "taskId")
    # Its status as Claude Code names it, "in_progress" said as words.
    status = _text(input, "status")
    return Intent(UPDATING_TASKS, None if task is None else Words(task if status is None else f"{task}: {status.replace('_', ' ')}"))


def _planned(_tool: str, _input: Mapping[str, object]) -> Intent:
    return Intent(PLANNING, None)


def _asked(_tool: str, input: Mapping[str, object]) -> Intent:
    # The first question that has its text, as the turn's record of the dialog reads them.
    questions = input.get("questions")
    blocks = cast(list[object], questions) if isinstance(questions, list) else []
    asked = (_text(cast(Mapping[str, object], each), "question") for each in blocks if isinstance(each, dict))
    question = next((text for text in asked if text is not None), None)
    return Intent(ASKING, None if question is None else Words(question))


def _approval(_tool: str, _input: Mapping[str, object]) -> Intent:
    return Intent(APPROVING, None)


def _used(tool: str, input: Mapping[str, object]) -> Intent:
    # An MCP tool is named mcp__<server>__<tool>: the tool's own name is the part a listener can follow. The page cuts
    # its input to a line, so no more of it is kept.
    return Intent(USING, ToolInput(tool.rsplit("__", 1)[-1].replace("_", " "), _cut(json.dumps(dict(input), ensure_ascii=False), LINE)))


# [LAW:one-type-per-behavior] one table keyed by tool name, as the step recognisers are; a tool not in it is used.
_SAID: Mapping[str, Callable[[str, Mapping[str, object]], Intent]] = {
    "Bash": _ran,
    "Edit": _on_file(EDITING),
    "MultiEdit": _on_file(EDITING),
    "NotebookEdit": _on_file(EDITING),
    "Write": _on_file(WRITING_FILES),
    "Read": _on_file(READING),
    "Grep": _searched,
    "Glob": _searched,
    "WebSearch": _looked_up,
    "WebFetch": _fetched,
    "Task": _dispatched,
    "Agent": _dispatched,
    "Skill": _skilled,
    "SendMessage": _messaged,
    "TodoWrite": _planned,
    "TaskCreate": _task_added,
    "TaskUpdate": _task_updated,
    "AskUserQuestion": _asked,
    "ExitPlanMode": _approval,
}


def _joined(act: str, what: str) -> str:
    return " ".join(part for part in (act, what) if part)


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
        case ToolInput(tool=tool):
            return tool


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
        case ToolInput(tool=tool, json=json_text):
            return f"{tool} {json_text}"


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


def _first_line(text: str) -> str:
    return text.splitlines()[0]


# What a listener can follow: words, and the dashes, dots, and underscores names are spelled with.
_WORDS = re.compile(r"[\w][\w .'-]*")


def _host(url: str | None) -> str | None:
    """The host of a URL, None where it names none."""
    try:
        return None if url is None else urlsplit(url).hostname
    except ValueError:
        return None


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
