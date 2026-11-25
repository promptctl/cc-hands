"""How a live session stands, and the one sentence that says it."""

from pathlib import Path

import pytest

from hands.core.progress import RUNNING, Doing
from hands.core.session import AskedQuestion, Held, Idle as Parked, Membership, Opened, Option, Permission, Plan, PromptId, Question, RequestId, Running, Session, SessionId, Told, Turn, Unreported
from hands.core.status import Busy, Idle, Shell, Stamp, Unknown, UnknownReason, Waiting
from hands.core.turn import AgentId
from hands.sessions.registry import Listing
from hands.core.standing import AtDialog, AtRest, AtUnheldDialog, Delegating, NotReported, ShellRunning, Standing, UnknownStatus, Working, on_a_turn, spoken_state, standing_of, turn_over
from hands.voice.tools import describe_listing

MEMBER = Membership(SessionId("s"), pid=4242, cwd=Path("/code/cc-hands"), transcript=Path("/code/cc-hands/t.jsonl"))
TURN = PromptId("p1")
RUN_TESTS = Doing(RUNNING, "run the test suite")


def held(on: Permission | Question | Plan) -> tuple[Held, ...]:
    return (Held(on, RequestId("r"), asked=1.0),)


def going(status: Busy | Waiting | Unknown, *, turn: Turn = Told(), dialogs: tuple[Held, ...] = (), background: frozenset[AgentId] = frozenset()) -> Session:
    return Session(MEMBER, Running(status, Stamp(1000), None), mode=None, turn=turn, dialogs=dialogs, background=background)


@pytest.mark.parametrize(
    ("session", "says"),
    [
        pytest.param(going(Waiting("permission prompt"), dialogs=held(Permission("Bash", {"command": "ls"}))), "waiting for permission to use Bash", id="a permission dialog hands holds"),
        pytest.param(going(Waiting("input needed"), dialogs=held(Question((AskedQuestion("Which?", (Option("a", None),), several=False),), {}))), "waiting for the user to answer its question", id="a question hands holds"),
        pytest.param(going(Waiting("permission prompt"), dialogs=held(Plan("1. do it"))), "waiting for the user to approve its plan", id="a plan hands holds"),
        pytest.param(Session(MEMBER, Unreported(), mode=None, dialogs=held(Permission("Edit", {}))), "waiting for permission to use Edit", id="a dialog held before its status is read"),
        pytest.param(going(Waiting("permission prompt")), "waiting at a dialog: permission prompt", id="a dialog Claude Code says is up and no hook holds"),
        pytest.param(going(Busy(), turn=Opened(TURN, latest=RUN_TESTS)), "working; the last thing it set out to do: run the test suite", id="working, with a call made"),
        pytest.param(going(Busy(), turn=Opened(TURN)), "working", id="working, before its first call"),
        pytest.param(going(Busy(), background=frozenset({AgentId("a1"), AgentId("a2")})), "idle, with two subagents it started in the background still working", id="at its prompt with subagents out"),
        pytest.param(Session(MEMBER, Parked(Shell(), Stamp(1000), None), mode=None), "idle, with a shell command it started in the background still running", id="at its prompt with a background shell"),
        pytest.param(Session(MEMBER, Unreported(), mode=None), "not reported yet", id="no status read yet"),
        pytest.param(Session(MEMBER, Parked(Idle(), Stamp(1000), None), mode=None), "idle", id="at its prompt"),
        pytest.param(Session(MEMBER, Parked(Idle(), Stamp(1000), None), mode=None, turn=Opened(TURN)), "working", id="a turn opened at its prompt before busy is read"),
        pytest.param(Session(MEMBER, Parked(Shell(), Stamp(1000), None), mode=None, turn=Opened(TURN)), "working", id="a turn opened at its prompt before busy is read, a shell still running"),
        pytest.param(going(Waiting(UnknownReason("elicitation"))), "waiting at a dialog: elicitation", id="a dialog Claude Code names and hands does not know"),
        pytest.param(going(Unknown("compacting")), "in a state hands does not know: compacting", id="a status hands does not know"),
    ],
)
def test_a_session_stands_in_one_sentence(session: Session, says: str) -> None:
    assert spoken_state(session) == says


def test_only_a_session_at_its_prompt_with_nothing_out_is_at_rest() -> None:
    assert standing_of(Session(MEMBER, Parked(Idle(), Stamp(1000), None), mode=None)) == AtRest()
    assert standing_of(Session(MEMBER, Parked(Idle(), Stamp(1000), None), mode=None, turn=Opened(TURN))) != AtRest()
    assert standing_of(Session(MEMBER, Parked(Shell(), Stamp(1000), None), mode=None)) != AtRest()


def test_the_listing_says_a_session_as_the_one_sentence_does() -> None:
    session = going(Busy(), turn=Opened(TURN, latest=RUN_TESTS))
    assert describe_listing(Listing(session, "naming fix")) == {"id": "s", "name": "cc-hands, naming fix", "state": spoken_state(session), "mode": "not reported yet"}


@pytest.mark.parametrize(
    ("standing", "over", "on_a_turn_"),
    [
        pytest.param(AtRest(), True, False, id="at rest"),
        pytest.param(ShellRunning(), True, False, id="a shell out"),
        pytest.param(Delegating(2), True, False, id="subagents out"),
        pytest.param(Working(None), False, True, id="working"),
        pytest.param(AtUnheldDialog("permission prompt"), False, True, id="at a dialog hands does not hold"),
        pytest.param(UnknownStatus("compacting"), False, True, id="a status hands does not know"),
        pytest.param(NotReported(), False, False, id="no status read yet: it may be mid-turn, and nothing says it is"),
    ],
)
def test_what_a_standing_says_of_the_turn(standing: Standing, over: bool, on_a_turn_: bool) -> None:
    assert (turn_over(standing), on_a_turn(standing)) == (over, on_a_turn_)


def test_a_session_held_at_its_dialog_is_mid_turn_and_not_over() -> None:
    at = AtDialog(held(Permission("Bash", {}))[0])
    assert (turn_over(at), on_a_turn(at)) == (False, True)
