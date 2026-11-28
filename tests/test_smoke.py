"""`hands smoke`: each stage read off the record the pipeline writes as it reaches it, the session started as a terminal
outside any session starts one, and a run that stops short naming the stage on its command's event.

The whole run, against a running hands and a working session, is what the command is for: it is run by
hand on an installed Mac, not here."""

import asyncio
import json
import os
from aiortc import RTCPeerConnection
from aiortc.exceptions import InvalidStateError
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fakeparties import main_on_fakes
from fakepseudoterminals import FakePseudoterminals, Shows
from hands.clock import SYSTEM
from hands.core.effects import Fritter, Summarise, Text, Type, ToldByStop
from hands.core.session import PromptId, PromptText, SessionId
from hands.core.trace import Span
from hands.sessions.child import Children
from hands.sessions.startsession import joined
from hands.sessions.liveness import live
from hands.daemon.smoke import FOLDER, QUIET_SECS, WORDS, Caller, Ear, Line, NotReached, parsed, proof
from hands.sessions import heartbeat
from hands.sessions.audit import ClipHeard, Typing, TypingConfirmed, TypingFailed, TypingUnconfirmed, Unsaid, encoded, segment
from hands.sessions.home import Home
from hands.voice import transcription
from hands.voice.mlxthread import OnMLX
from hands.sessions.wide import Fact, Outcome, WideEvent
from fakeprocesses import FakeProcesses, Running, processes

SESSION = SessionId("5086f176-e3cf-4222-bdb9-d7993c507d33")
OTHER = SessionId("33a1f45e-fc95-41ca-98d0-a9310ab422fe")
SPAN = "e5fe41d042769d03"
# The running hands a heartbeat here names, and a claude left open as a session of it.
HANDS, CLAUDE = 4242, 4343


def up(now: datetime) -> FakeProcesses:
    """The Mac's processes with HANDS and CLAUDE among them, both started an hour before `now`."""
    started = (now - timedelta(hours=1)).timestamp()
    return processes(Running(HANDS, start=started), Running(CLAUDE, start=started))


def as_logged(record: object) -> Line:
    """`record` as the audit log holds it: the daemon's own encoding, so a field renamed there is renamed here."""
    return encoded(record)


def sent(session: SessionId) -> Type[Text]:
    return Type(session, Fritter(Path("/tmp/f/session.sock"), 7), Text(PromptText("Reply with the file.")))


def typing(session: SessionId, span: str = SPAN) -> Line:
    return as_logged(Typing(sent(session), Span("t", span, None)))


def confirmed(session: SessionId) -> Line:
    return as_logged(TypingConfirmed(sent(session)))


def event(name: str, span: str = "s", outcome: Outcome = "ok", error: str | None = None, **facts: Fact) -> Line:
    return as_logged(WideEvent(name, "t", span, None, datetime.now(UTC), 1.0, outcome, error, (), {}, facts))


def ended(span: str = SPAN) -> Line:
    return event("tool.run", span)


def hook(name: str, session: SessionId) -> Line:
    return event("hook", hook=name, session=session)


def stop(session: SessionId) -> Line:
    return hook("Stop", session)


def test_a_clip_is_heard_once_whisper_took_words_from_it() -> None:
    assert proof("heard", [], SESSION) is None
    assert proof("heard", [as_logged(ClipHeard(2, "tell the smoke session", ()))], SESSION) == "heard 'tell the smoke session'"


def test_a_clip_whisper_took_no_words_from_is_never_heard_and_says_so() -> None:
    nothing = as_logged(ClipHeard(1, None, (Unsaid("Thank you.", 2.9, -1.2, 0.02),)))
    with pytest.raises(NotReached) as raised:
        proof("heard", [nothing], SESSION)
    assert (raised.value.stage, raised.value.why) == ("heard", "Whisper took no words from the clip; it dropped 1 segment(s)")


def utterance(session: SessionId, fate: str, outcome: Outcome = "ok", error: str | None = None) -> Line:
    heard = Summarise(session, PromptId("8dc206d5-ceb8-4cf1-887e-e00c5c57a55c"), ToldByStop("falcon.txt"), ids=frozenset({PromptId("8dc206d5-ceb8-4cf1-887e-e00c5c57a55c")}))
    return event("utterance", outcome=outcome, error=error, session=session, heard=heard, fate=fate)


def test_the_session_s_turn_is_settled_once_its_utterance_is_held_or_played_to_its_end() -> None:
    assert proof("told", [stop(SESSION), utterance(OTHER, "played")], SESSION) is None
    assert proof("told", [utterance(SESSION, "noted")], SESSION) == f"hands' telling of session {SESSION}'s turn: noted"
    assert proof("told", [utterance(SESSION, "played")], SESSION) == f"hands' telling of session {SESSION}'s turn: played"


def test_a_telling_cut_off_is_never_settled_and_says_so() -> None:
    with pytest.raises(NotReached) as raised:
        proof("told", [utterance(SESSION, "barged_in")], SESSION)
    assert (raised.value.stage, raised.value.why) == ("told", f"hands' telling of session {SESSION}'s turn was cut off before the question was asked")


def test_a_turn_hands_could_not_read_is_never_told_and_says_why() -> None:
    failed = utterance(SESSION, "played", "failed", "the turn could not be read: OSError: gone")
    with pytest.raises(NotReached) as raised:
        proof("told", [failed], SESSION)
    assert (raised.value.stage, raised.value.why) == ("told", f"hands could not tell session {SESSION}'s turn: the turn could not be read: OSError: gone")


def test_a_session_that_asks_permission_never_finishes_and_says_so() -> None:
    assert proof("finished", [hook("PermissionRequest", OTHER)], SESSION) is None
    with pytest.raises(NotReached) as raised:
        proof("finished", [hook("PermissionRequest", SESSION)], SESSION)
    assert raised.value.stage == "finished" and raised.value.why.startswith(f"session {SESSION} asked permission to use a tool")


def test_a_send_is_typed_only_once_its_typist_said_it_typed_it() -> None:
    # The Typing line is written before the typing: alone, it says the send is under way, not that it landed. Nor does
    # the unit of work that typed it ending, which it does however its send ended.
    assert proof("typed", [typing(SESSION), ended()], SESSION) is None
    assert proof("typed", [typing(SESSION), confirmed(SESSION), ended()], SESSION) == f"typed 'Reply with the file.' into session {SESSION}"


def test_a_send_that_failed_is_never_typed_and_says_why() -> None:
    failed = as_logged(TypingFailed(sent(SESSION), "this socket types into process 45443"))
    with pytest.raises(NotReached) as raised:
        proof("typed", [typing(SESSION), failed, ended()], SESSION)
    assert (raised.value.stage, raised.value.why) == ("typed", f"hands could not type into session {SESSION}: this socket types into process 45443")


def test_a_send_never_confirmed_is_never_typed_and_says_why() -> None:
    unconfirmed = as_logged(TypingUnconfirmed(sent(SESSION), "the fritter never answered"))
    with pytest.raises(NotReached) as raised:
        proof("typed", [typing(SESSION), unconfirmed, ended()], SESSION)
    assert (raised.value.stage, raised.value.why) == ("typed", f"hands was never told whether what it typed into session {SESSION} arrived: the fritter never answered")


def test_only_the_smoke_session_counts_for_its_send_and_its_finish() -> None:
    lines = [typing(OTHER), confirmed(OTHER), ended(), stop(OTHER)]
    assert proof("typed", lines, SESSION) is None
    assert proof("finished", lines, SESSION) is None
    assert proof("finished", [*lines, stop(SESSION)], SESSION) == f"session {SESSION} finished its turn"


def test_a_torn_line_is_no_evidence_and_breaks_none_of_the_rest() -> None:
    assert parsed(['{"type": "ClipHe', json.dumps({"type": "ClipHeard", "said": "hi"}), "[1]"]) == [{"type": "ClipHeard", "said": "hi"}]


def frame(level: int) -> bytes:
    return np.full(320, level, dtype=np.int16).tobytes()


def test_hands_has_finished_speaking_once_it_spoke_after_the_release_and_went_quiet() -> None:
    ear = Ear()
    ear.heard(frame(3000), 1.0)
    # Spoke, but before the release: nothing said back yet.
    assert ear.finished_speaking(2.0, 10.0) is None
    ear.heard(frame(0), 2.5)
    ear.heard(frame(3000), 3.0)
    assert ear.finished_speaking(2.0, 3.0 + QUIET_SECS / 2) is None
    assert ear.finished_speaking(2.0, 3.0 + QUIET_SECS) == 3.0
    assert ear.since(2.0) == frame(0) + frame(3000)


class Dropped:
    """A call's channel once the call has dropped, which refuses what is sent on it."""

    def send(self, data: bytes | str) -> None:
        raise InvalidStateError("RTCDataChannel is not open")


async def test_a_call_that_drops_stops_the_stage_being_said_and_says_why() -> None:
    peer = RTCPeerConnection()
    caller = Caller(peer, Dropped(), SYSTEM)  # pyright: ignore[reportArgumentType]  (the channel's send is all the caller uses)
    caller.tasks.append(asyncio.create_task(caller.keep_sending(), name="the smoke test's voice"))
    try:
        with pytest.raises(NotReached) as raised:
            await asyncio.wait_for(caller.say("typed", frame(3000)), 5)
        assert (raised.value.stage, raised.value.why) == ("typed", "the smoke test's voice stopped: InvalidStateError('RTCDataChannel is not open')")
    finally:
        await peer.close()


def test_the_smoke_session_is_the_one_that_joined_under_its_process(tmp_path: Path) -> None:
    home = Home(tmp_path / "home")
    home.memberships.mkdir(parents=True)
    # The `claude` the smoke run started, and another process's.
    started, other = 4242, 4343
    table = processes(Running(started), Running(other))

    def member(session: SessionId, pid: int) -> None:
        record = {"pid": pid, "cwd": str(tmp_path), "transcript_path": f"/t/{session}.jsonl", "fritter_socket": "/tmp/f/session.sock"}
        home.membership(session).write_text(json.dumps(record))

    # A session of another process's in the same folder, and one being written.
    member(OTHER, other)
    home.membership(SessionId("0998fa11-3723-4c9a-bb7a-b454ae05dfca")).write_text('{"pid": ')
    assert joined(live(table, home)[0], started, table.table()) is None
    member(SESSION, started)
    found = joined(live(table, home)[0], started, table.table())
    assert found is not None and (found.id, found.fritter) == (SESSION, Path("/tmp/f/session.sock"))


def events(home: Home) -> list[dict[str, Any]]:
    return [line for line in map(json.loads, segment(home.audit, 0).read_text().splitlines()) if line["type"] == "WideEvent"]


def test_with_hands_not_running_the_run_stops_at_up_and_its_event_says_so(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    home = Home(tmp_path / "home")
    assert main_on_fakes(["--home", str(home.root), "smoke"]) == 1
    assert capsys.readouterr().out == f"FAILED up: hands has not run: there is no heartbeat at {home.status}; start it with `hands run`\n"
    [command] = events(home)
    assert (command["event"], command["outcome"], command["facts"]["command"], command["facts"]["failed_at"]) == ("hands.command", "failed", "smoke", "up")
    # The event says why, as the run said it: the stage it stopped at, and what it found there.
    assert command["error"] == f"exited 1: up: hands has not run: there is no heartbeat at {home.status}; start it with `hands run`"
    assert "reached" not in command["facts"] and "model" not in command["facts"]


def test_a_run_that_stops_after_up_says_on_its_event_what_it_reached_and_why_it_stopped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    home = Home(tmp_path / "home")
    now = datetime.now(UTC)
    home.root.mkdir()
    heartbeat.write(home.status, heartbeat.Status(HANDS, now, now, heartbeat.HEARTBEAT, "running", None, 0, False, (), None))
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert main_on_fakes(["--home", str(home.root), "smoke"], processes=up(now)) == 1
    assert capsys.readouterr().out.splitlines()[-1] == "FAILED joined: there is no `claude` on PATH"
    [command] = events(home)
    facts = command["facts"]
    assert facts["word"] in WORDS and facts["folder"] == str((home.root / FOLDER).resolve())
    assert facts["model"] == transcription.MODEL
    assert (facts["reached"], facts["failed_at"], facts["daemon_errors"]) == ("up", "joined", [])
    assert command["error"] == "exited 1: joined: there is no `claude` on PATH"
    assert isinstance(facts["up_ms"], int) and facts["up"].startswith(f"hands is up: pid {HANDS}") and "joined_ms" not in facts


def stand_in_run(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, cwd: Path, written_at: float) -> tuple[str, list[dict[str, str]]]:
    """A smoke run against a hands that is up, whose claude is a stand-in that stops it at joined, with CLAUDE a session
    of hands in `cwd`, its membership file written at `written_at`. The run's last line, and the
    environment of each claude it started."""
    from hands.daemon import smoke

    home = Home(tmp_path / "home")
    now = datetime.now(UTC)
    home.memberships.mkdir(parents=True)
    heartbeat.write(home.status, heartbeat.Status(HANDS, now, now, heartbeat.HEARTBEAT, "running", None, 0, False, (), None))
    record = {"pid": CLAUDE, "cwd": str(cwd), "transcript_path": f"/t/{OTHER}.jsonl", "fritter_socket": "/tmp/f/session.sock"}
    home.membership(OTHER).write_text(json.dumps(record))
    os.utime(home.membership(OTHER), (written_at, written_at))
    real = tmp_path / "real" / "claude"
    real.parent.mkdir()
    real.write_text("#!/bin/sh\n")
    real.chmod(0o755)
    monkeypatch.setenv("PATH", str(real.parent))
    def unloaded(_on: OnMLX, _weights: Path) -> None:
        pass

    monkeypatch.setattr(transcription, "load", unloaded)

    async def synthesized(_children: Children, _text: str) -> bytes:
        return b""

    # The stand-in claude exits at once, so the run stops at joined.
    terminals = FakePseudoterminals({str(real): Shows(lambda _asked: "a stand-in", 0)})
    monkeypatch.setattr(smoke, "_synthesized", synthesized)
    assert main_on_fakes(["--home", str(home.root), "smoke"], processes=up(now), pseudoterminals_on=lambda _clock, _emit: terminals) == 1
    [failed] = [line for line in capsys.readouterr().out.splitlines() if line.startswith("FAILED")]
    return failed, [dict(asked.environment) for asked in terminals.asked]


def stopped_by_the_stand_in(tmp_path: Path) -> str:
    """The line a run stand_in_run made fails with, once it has started its claude."""
    return f"FAILED joined: `{tmp_path / 'real' / 'claude'}` started in {(tmp_path / 'home' / FOLDER).resolve()} exited 0 before it joined hands; it showed:"


def test_the_claude_the_smoke_run_starts_runs_with_its_auto_updater_off(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    # The claude on PATH here is no shim, so nothing but the smoke run itself switches its auto-updater off.
    monkeypatch.setenv("DISABLE_AUTOUPDATER", "0")
    last, started = stand_in_run(tmp_path, capsys, monkeypatch, tmp_path / "elsewhere", datetime.now(UTC).timestamp())
    assert last == stopped_by_the_stand_in(tmp_path)
    [environment] = started
    assert environment["DISABLE_AUTOUPDATER"] == "1"


@pytest.mark.parametrize("folder", ["home/smoke", "code/Smoke"])
def test_a_live_session_in_a_folder_hands_also_calls_smoke_stops_the_run_before_it_empties_its_folder_and_is_named(
    folder: str, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # CLAUDE is a claude left open in the smoke folder, or in any folder hands speaks of as smoke.
    cwd = tmp_path / folder
    smoke_folder = tmp_path / "home" / FOLDER
    smoke_folder.mkdir(parents=True)
    (smoke_folder / "kept.txt").write_text("")
    last, started = stand_in_run(tmp_path, capsys, monkeypatch, cwd, datetime.now(UTC).timestamp())
    assert last == (
        f"FAILED joined: already open: session {OTHER} (pid {CLAUDE}) in {cwd}, which hands calls {cwd.name!r}; the brain"
        f" could take one for the session the test starts, which hands calls 'smoke': close each, or end them with `kill {CLAUDE}`,"
        " then run `hands smoke` again"
    )
    # Refused before the smoke folder was emptied and before a claude was started.
    assert (smoke_folder / "kept.txt").exists() and started == []


@pytest.mark.parametrize(("folder", "ago"), [("home/smoke", 1e9), ("code/smokey", 0.0)], ids=["exited", "named-otherwise"])
def test_a_session_hands_does_not_call_smoke_does_not_stop_the_run(
    folder: str, ago: float, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # A file written before CLAUDE started names a claude that has exited and whose pid was handed out
    # again; a live session in a folder of another name is called by that name.
    last, _ = stand_in_run(tmp_path, capsys, monkeypatch, tmp_path / folder, datetime.now(UTC).timestamp() - ago)
    assert last == stopped_by_the_stand_in(tmp_path)
