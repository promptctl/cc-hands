"""The one model of a Claude Code turn, read off each turn Claude Code was recorded running (tests/turns.py): what it
posted to its hooks through parse_hook, what it wrote to the transcript through the tail, and what it set in its
status file through that file's reader, each reduced as hands reduces it.

The tail polls, so a record can be read as soon as it is written or only at the next hook or status: both are run.
"""

import json
from collections.abc import Collection
from pathlib import Path

import pytest

from hands.core.effects import Snapshot, Summarise
from hands.core.events import Event, StatusReported
from hands.core.reducer import reduce
from hands.core.session import Instant, Membership, Registry, RequestId, Session, SessionId, Unreported, status_stamp
from hands.core.status import Stamp
from hands.sessions.home import Home
from hands.sessions.hookconfig import SUBSCRIBED
from hands.sessions.hooks import parse_hook
from hands.sessions.membership import write_membership
from hands.sessions.statusfile import parse_report
from hands.sessions.tail import Tails
from turns import RECORDINGS, Posted, StatusSet, Written, played, shifted, status_file, transcribed

SESSION, PID = SessionId("s1"), 4242
# Where on the clock a recording is played from.
BASE = 1_800_000_000.0
SCHEDULES = ("records as written", "records at the next hook or status")

# What each recorded turn gives, in order: S a turn marked as it begins (Snapshot), Tn the turn pn names told
# (Summarise). Every Claude Code turn is marked once as it begins, and told once as it ends, under the id its Stop names;
# words and a notification that joined a turn in flight are a part of it, not a turn of their own; and a turn that went
# on under the id Claude Code gave the words sent into it is one turn, told by that id.
TOLD = {
    "backgrounded_then_notified": ("S", "T1", "S", "T2"),
    "joined_as_next_prompt": ("S", "T1", "S", "T2"),
    "joined_at_boundary": ("S", "T1"),
    "notified_mid_turn": ("S", "T1"),
    "sent_now": ("S", "T1", "S", "T2"),
    "sent_now_mcp": ("S", "T1", "S", "T2"),
    # p1 goes on as p2, which only its records say; the call moved to the background then opens p3 for its notification.
    "sent_now_mid_batch": ("S", "T2", "S", "T3"),
}
# The model does not mark these turns right yet: it reads a UserPromptSubmit under the running turn's id as a message
# queued behind it. Claude Code writes a prompt's record and sets busy before it posts the prompt's own UserPromptSubmit,
# so a turn the tail opened from its record is left unmarked, and its own hook then marks a turn after it; words or a
# notification taken into the turn mark a turn after it too, and none comes. Claude Code's queue records say what was
# queued, and what was absorbed into the turn (promptctl-brain-vu2t.bv3).
MISMARKED = set(TOLD)


def _case(name: str, schedule: str, unmet: Collection[str] = ()) -> object:
    marks = [pytest.mark.xfail(strict=True, raises=AssertionError, reason="promptctl-brain-vu2t.bv3")] if name in unmet else []
    return pytest.param(name, schedule, marks=marks, id=f"{name}-{schedule.replace(' ', '_')}")


async def _told(name: str, schedule: str, tmp: Path) -> tuple[str, ...]:
    transcript = tmp / "transcript.jsonl"
    transcript.write_text("")
    home = Home(tmp / "home")
    member = Membership(SESSION, pid=PID, cwd=Path("/brain/cwd"), transcript=transcript)
    write_membership(home, member)
    registry = Registry(sessions={SESSION: Session(member, Unreported(), mode=None)}, drafts={})
    clock: list[Instant] = [BASE]

    class Known:
        def live_members(self) -> list[Membership]:
            return [member]

        def status_read(self, session: SessionId) -> bool:
            known = registry.sessions[SESSION]
            return isinstance(known, Session) and status_stamp(known.state) is not None

        def now(self) -> Instant:
            return clock[0]

        def stamp(self) -> Stamp:
            return Stamp(round(clock[0] * 1000))

        def membership(self, session: SessionId) -> Membership | None:
            return member

    tails = Tails(Known())
    told: list[str] = []

    def apply(event: Event) -> None:
        nonlocal registry
        registry, effects = reduce(registry, event)
        for effect in effects:
            match effect:
                case Snapshot():
                    told.append("S")
                case Summarise(turn=turn):
                    told.append(f"T{str(turn).removeprefix('p')}")
                case _:
                    pass

    async def read() -> None:
        for event in await tails.catch_up():
            apply(event)

    for i, line in enumerate(played(RECORDINGS[name], subscribed=SUBSCRIBED, session=SESSION)):
        line = shifted(line, BASE)
        clock[0] = line.at
        match line:
            case Posted(body=body):
                raw = json.dumps({"session_id": SESSION, "transcript_path": str(transcript), "cwd": str(member.cwd), **body}).encode()
                apply(parse_hook(raw, home=home, at=line.at, heard=Stamp(round(line.at * 1000)), request=RequestId(f"r{i}")).happened)
                await read()
            case Written(record=record):
                with transcript.open("a") as written:
                    written.write(transcribed(record))
                if schedule == "records as written":
                    await read()
            case StatusSet(status=status):
                report = parse_report(member, status_file(line, pid=PID, session=SESSION).encode())
                if report is None:
                    # Not an AssertionError: a recording the player cannot read must never pass for an expected failure.
                    raise ValueError(f"{name}: a status line sets no status: {status!r}")
                apply(StatusReported(SESSION, report, line.at))
                await read()
    return tuple(told)


@pytest.mark.parametrize(("name", "schedule"), [_case(name, schedule) for name in RECORDINGS for schedule in SCHEDULES])
async def test_each_claude_code_turn_recorded_is_told_once_under_the_id_its_stop_names(name: str, schedule: str, tmp_path: Path) -> None:
    assert set(RECORDINGS) == set(TOLD)
    told = await _told(name, schedule, tmp_path)
    assert tuple(step for step in told if step.startswith("T")) == tuple(step for step in TOLD[name] if step.startswith("T"))


@pytest.mark.parametrize(("name", "schedule"), [_case(name, schedule, MISMARKED) for name in RECORDINGS for schedule in SCHEDULES])
async def test_each_claude_code_turn_recorded_is_marked_once_as_it_begins_and_nothing_else_is(name: str, schedule: str, tmp_path: Path) -> None:
    assert await _told(name, schedule, tmp_path) == TOLD[name]
