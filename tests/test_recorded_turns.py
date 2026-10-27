"""The one model of a Claude Code turn, read off each turn Claude Code was recorded running (tests/turns.py): what it
posted to its hooks through parse_hook, what it wrote to the transcript through the tail, and what it set in its
status file through that file's reader, each reduced as hands reduces it, from the session's start hook on.

The tail polls, so a record can be read as soon as it is written or only at the next hook or status: both are run.

The contract, which each recording's own Stops spell out: every Claude Code turn ends at a Stop, and is marked once as
it begins (S, a Snapshot) and told once as it ends (Tn, a Summarise naming pn), under the id that Stop names. Words or
a notification joined to a turn in flight are a part of it, not a turn of their own, and a turn that went on under the
id Claude Code gave the words sent into it is one turn, told by the id its Stop names.
"""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest

from hands.core.effects import Snapshot, Summarise
from hands.core.events import Continued, Event, StatusReported, Taken
from hands.core.reducer import reduce
from hands.core.session import Instant, Membership, Registry, RequestId, Session, SessionId, status_stamp
from hands.core.status import Stamp
from hands.sessions.home import Home
from hands.brain import pinned
from hands.sessions.hookconfig import plugin_hooks
from hands.sessions.hooks import parse_display, parse_hook
from hands.sessions.membership import write_membership
from hands.sessions.statusfile import parse_report
from hands.sessions.tail import Tails
from turns import RECORDINGS, Posted, StatusSet, Written, played, shifted, status_file, stops, transcribed

SESSION, PID = SessionId("s1"), 4242
# Where on the clock a recording is played from.
BASE = 1_800_000_000.0
SCHEDULES = ("records as written", "records at the next hook or status")
# [LAW:one-source-of-truth] the hooks a wrapped session posts: hands' plugin's own.
POSTED = tuple(cast(Mapping[str, object], plugin_hooks()["hooks"]))
# The recorded turns the model does not read right yet (promptctl-brain-vu2t.bv3), both from one gap: it does not read
# Claude Code's queue records. Words or a notification absorbed into a turn (queue-operation remove, absorbed_mid_turn)
# are taken for a message queued behind it, so a turn after it is marked and none comes. And in sent_now_mid_batch,
# Claude Code sets shell for a few ms as send now moves the running call to the background, which ends the turn for the
# model, so it is told by the id it began under at the first read past its Stop hold: here the MessageDisplay that comes
# before Stop(p2), and with no read between, p2 would never be told. The records that say it went on under the words'
# id (a dequeue, then the later results and the words under that id) are read as a turn of their own, never marked,
# which its Stop tells again: one Claude Code turn told twice.
MISMARKED = {"joined_at_boundary", "notified_mid_turn", "sent_now", "sent_now_mcp", "sent_now_mid_batch"}
MISTOLD = {"sent_now_mid_batch"}


def _contract(name: str) -> tuple[str, ...]:
    """What the model must give of the recording: each turn marked as it begins, and told under the id its Stop names."""
    return tuple(step for stop in stops(RECORDINGS[name]) for step in ("S", f"T{stop.removeprefix('p')}"))


def _case(name: str, schedule: str, unmet: set[str]) -> object:
    marks = [pytest.mark.xfail(strict=True, raises=AssertionError, reason="promptctl-brain-vu2t.bv3")] if name in unmet else []
    return pytest.param(name, schedule, marks=marks, id=f"{name}-{schedule.replace(' ', '_')}")


@dataclass(frozen=True)
class Reading:
    """What the model made of a recording."""

    # Its marks (S) and tellings (Tn), in order.
    told: tuple[str, ...]
    # The prompt ids it heard from the transcript.
    heard: frozenset[str]
    # How many of the recording's hooks of each name reached it as an event.
    reached: Mapping[str, int]


async def _told(name: str, schedule: str, tmp: Path) -> Reading:
    """What the model makes of the recording, read in the schedule given."""
    transcript = tmp / "transcript.jsonl"
    transcript.write_text("")
    home = Home(tmp / "home")
    member = Membership(SESSION, pid=PID, cwd=Path("/brain/cwd"), transcript=transcript)
    # As the shim writes it before the session's first hook posts.
    write_membership(home, member)
    registry = Registry(sessions={}, drafts={})
    clock: list[Instant] = [BASE]

    class Known:
        def live_members(self) -> list[Membership]:
            return [session.membership for session in registry.sessions.values() if isinstance(session, Session)]

        def status_read(self, session: SessionId) -> bool:
            known = registry.sessions.get(session)
            return isinstance(known, Session) and status_stamp(known.state) is not None

        def now(self) -> Instant:
            return clock[0]

        def stamp(self) -> Stamp:
            return Stamp(round(clock[0] * 1000))

        def membership(self, session: SessionId) -> Membership | None:
            return member

    tails = Tails(Known())
    told: list[str] = []
    heard: set[str] = set()
    reached: Counter[str] = Counter()

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
            match event:
                case Taken(prompt=prompt) | Continued(now=prompt):
                    heard.add(prompt)
                case _:
                    pass
            apply(event)

    for i, line in enumerate(played(RECORDINGS[name], subscribed=POSTED, session=SESSION)):
        line = shifted(line, BASE)
        clock[0] = line.at
        match line:
            case Posted(body=body) if body["hook_event_name"] == "MessageDisplay":
                # Posted to the daemon's display route, which takes MessageDisplay alone.
                apply(parse_display(transcribed({"session_id": SESSION, **body}).encode(), at=line.at))
                reached["MessageDisplay"] += 1
                await read()
            case Posted(body=body):
                raw = transcribed({"session_id": SESSION, "transcript_path": str(transcript), "cwd": str(member.cwd), **body}).encode()
                hook = parse_hook(raw, home=home, at=line.at, heard=Stamp(round(line.at * 1000)), request=RequestId(f"r{i}"))
                # As the server applies a hook: where its session is first, then what it says.
                if hook.joining is not None:
                    apply(hook.joining)
                apply(hook.happened)
                reached[body["hook_event_name"]] += 1
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
    # What was written after the last hook and status is read too, as the tail's next poll would.
    await read()
    return Reading(tuple(told), frozenset(heard), reached)


def test_every_recorded_turn_ends_at_a_stop() -> None:
    assert RECORDINGS and all(stops(recording) for recording in RECORDINGS.values())


def test_every_recorded_turn_begins_at_its_sessions_start() -> None:
    assert all(next(line for line in recording if isinstance(line, Posted)).body["hook_event_name"] == "SessionStart" for recording in RECORDINGS.values())


def test_every_recorded_turn_is_of_the_brains_pin() -> None:
    # The records name the release they were written by: once the pin moves, the turns are recorded again on it.
    versions = {line.record["version"] for recording in RECORDINGS.values() for line in recording if isinstance(line, Written) and "version" in line.record}
    assert versions == {pinned.VERSION}


@pytest.mark.parametrize(("name", "schedule"), [_case(name, schedule, set()) for name in RECORDINGS for schedule in SCHEDULES])
async def test_every_prompt_a_recorded_turns_records_carry_reaches_the_model_from_the_transcript(name: str, schedule: str, tmp_path: Path) -> None:
    carried = {id for line in RECORDINGS[name] if isinstance(line, Written) and line.record.get("type") in ("user", "assistant") for id in [line.record.get("promptId")] if isinstance(id, str)}
    assert carried and carried <= (await _told(name, schedule, tmp_path)).heard


@pytest.mark.parametrize("name", RECORDINGS)
async def test_every_hook_a_recorded_turn_posts_that_hands_subscribes_to_reaches_the_model(name: str, tmp_path: Path) -> None:
    # [LAW:verifiable-goals] against what hands' plugin subscribes to, not the hooks the player was handed, so a hook the
    # player or its parsers leave unread fails here, whichever it is.
    subscribed = cast(Mapping[str, object], plugin_hooks()["hooks"])
    posted = Counter(line.body["hook_event_name"] for line in RECORDINGS[name] if isinstance(line, Posted) and line.body["hook_event_name"] in subscribed)
    assert {"SessionStart", "UserPromptSubmit", "MessageDisplay", "Stop"} <= posted.keys()
    assert (await _told(name, SCHEDULES[0], tmp_path)).reached == posted


@pytest.mark.parametrize(("name", "schedule"), [_case(name, schedule, MISTOLD) for name in RECORDINGS for schedule in SCHEDULES])
async def test_each_claude_code_turn_recorded_is_told_once_under_the_id_its_stop_names(name: str, schedule: str, tmp_path: Path) -> None:
    told = (await _told(name, schedule, tmp_path)).told
    assert tuple(step for step in told if step.startswith("T")) == tuple(step for step in _contract(name) if step.startswith("T"))


@pytest.mark.parametrize(("name", "schedule"), [_case(name, schedule, MISMARKED) for name in RECORDINGS for schedule in SCHEDULES])
async def test_each_claude_code_turn_recorded_is_marked_once_as_it_begins_and_nothing_else_is(name: str, schedule: str, tmp_path: Path) -> None:
    assert (await _told(name, schedule, tmp_path)).told == _contract(name)
