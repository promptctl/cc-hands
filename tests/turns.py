"""Claude Code's turns as recorded on the brain's pin by scripts/record_turns.py: one value per turn, holding all Claude
Code said of it on each channel hands reads, in the order it said it. The stand-in Claude Code plays them to the brain
(conftest.fake_claude), and the reducer's tests read them, so both meet the protocol as Claude Code speaks it.
[LAW:one-source-of-truth] the files are the one copy; nothing here or in a test restates a turn.

A recording's times are seconds from its first line: a line's `at`, which is all the time a status holds, and a
record's timestamp in its own format (1970-01-01T00:00:14.045Z is 14.045 s in).
"""

import datetime
import json
import re
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

# A body as json.loads gives it.
Json = Mapping[str, Any]

RECORDED = Path(__file__).resolve().parent / "fixtures" / "recorded_turns"


@dataclass(frozen=True)
class Posted:
    """A hook body Claude Code posted, without the session's own fields (session_id, transcript_path, cwd), which the
    one playing it adds."""

    at: float
    body: Json


@dataclass(frozen=True)
class Written:
    """A record Claude Code appended to the session's transcript."""

    at: float
    record: Json


@dataclass(frozen=True)
class StatusSet:
    """A status Claude Code set in the session's file, sessions/<pid>.json, without the file's pid and sessionId."""

    at: float
    status: Json


Line = Posted | Written | StatusSet
Recording = tuple[Line, ...]


def _line(raw: str) -> Line:
    item = cast(dict[str, Any], json.loads(raw))
    at = float(item["at"])
    if "posted" in item:
        return Posted(at, item["posted"])
    if "written" in item:
        return Written(at, item["written"])
    if "status" in item:
        return StatusSet(at, item["status"])
    raise ValueError(f"not a line of a recorded turn: {raw!r}")


def load(path: Path) -> Recording:
    return tuple(_line(raw) for raw in path.read_text().splitlines())


# Each recorded turn by its name, the file's.
RECORDINGS: Mapping[str, Recording] = {path.stem: load(path) for path in sorted(RECORDED.glob("*.jsonl"))}


def played(recording: Recording, *, subscribed: Collection[str], session: str, first: int = 1, prompt: str | None = None) -> Recording:
    """`recording` as a session `session` plays it: only the hooks its settings subscribe to, as Claude Code posts only
    those; its prompt pK as p{first + K - 1}, so it follows the prompts the session has run; and, if `prompt` is given,
    the opening prompt as `prompt` on both channels that carry it, its UserPromptSubmit and its user record."""
    lines = [line for line in recording if not isinstance(line, Posted) or line.body["hook_event_name"] in subscribed]
    renumbered = tuple(_mapped(line, lambda text: _renumbered(text.replace("SESSION", session), first)) for line in lines)
    return renumbered if prompt is None else _prompted(renumbered, prompt)


def prompts(recording: Recording) -> int:
    """How many prompts `recording` runs: the K of its last pK."""
    return max(int(found) for line in recording for found in re.findall(r'"(?:prompt_id|promptId)": "p(\d+)"', json.dumps(_body(line))))


def _body(line: Line) -> Json:
    match line:
        case Posted(body=body):
            return body
        case Written(record=record):
            return record
        case StatusSet(status=status):
            return status


def transcribed(record: Json) -> str:
    """`record` as Claude Code appends it to the transcript: one line, compact, as the tail knows a turn's records by
    their bytes (sessions/transcript.py), and with what is not ASCII as itself."""
    return json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n"


def status_file(line: StatusSet, *, pid: int, session: str) -> str:
    """The session file, sessions/<pid>.json, as Claude Code writes it with `line`'s status set: whose it is, the
    status, and when it was set."""
    return json.dumps({"pid": pid, "sessionId": session, **line.status, "statusUpdatedAt": round(line.at * 1000)})


def shifted(line: Line, by: float) -> Line:
    """`line` `by` seconds later, every time its body holds with it."""
    match line:
        case Posted(at, body):
            return Posted(at + by, body)
        case Written(at, record):
            return Written(at + by, _later(record, by))
        case StatusSet(at, status):
            return StatusSet(at + by, status)


def _mapped(line: Line, edit: Callable[[str], str]) -> Line:
    match line:
        case Posted(at, body):
            return Posted(at, json.loads(edit(json.dumps(body))))
        case Written(at, record):
            return Written(at, json.loads(edit(json.dumps(record))))
        case StatusSet():
            return line


def _renumbered(text: str, first: int) -> str:
    return re.sub(r'("(?:prompt_id|promptId)": ")p(\d+)"', lambda found: f'{found[1]}p{first + int(found[2]) - 1}"', text)


def _prompted(lines: Recording, prompt: str) -> Recording:
    """`lines` with the opening prompt, the first UserPromptSubmit's and the first user record's, as `prompt`."""
    opening = next(line for line in lines if isinstance(line, Posted) and line.body["hook_event_name"] == "UserPromptSubmit")
    typed = next(line for line in lines if isinstance(line, Written) and line.record.get("type") == "user")
    message = cast(Json, typed.record["message"])
    assert isinstance(message["content"], str), f"the opening user record holds no typed prompt: {typed.record!r}"

    def edited(line: Line) -> Line:
        match line:
            case Posted(at, body) if line is opening:
                return Posted(at, {**body, "prompt": prompt})
            case Written(at, record) if line is typed:
                return Written(at, {**record, "message": {**message, "content": prompt}})
            case _:
                return line

    return tuple(edited(line) for line in lines)


def _later(value: Json, by: float) -> Json:
    return {key: _moved(key, inner, by) for key, inner in value.items()}


def _moved(key: str, value: Any, by: float) -> Any:
    if key == "timestamp" and isinstance(value, str):
        moved = datetime.datetime.fromisoformat(value.replace("Z", "+00:00")) + datetime.timedelta(seconds=by)
        return moved.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    if isinstance(value, dict):
        return _later(cast(Json, value), by)
    if isinstance(value, list):
        return [_moved("", inner, by) for inner in cast(list[Any], value)]
    return value
