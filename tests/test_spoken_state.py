"""How a live session stands, and the one sentence that says it."""

from pathlib import Path

import pytest

from hands.core.progress import RUNNING, Doing
from hands.core.session import AskedQuestion, Held, Idle as Parked, Membership, Opened, Option, Permission, Plan, PromptId, Question, RequestId, Running, Session, SessionId, Told, Turn, Unreported
from hands.core.standing import AtPrompt, standing
from hands.core.status import Busy, Idle, Shell, Stamp, Unknown, UnknownReason, Waiting
from hands.core.turn import AgentId
from hands.sessions.registry import Listing
from hands.core.standing import spoken_state
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
        pytest.param(going(Waiting(UnknownReason("elicitation"))), "waiting at a dialog: elicitation", id="a dialog Claude Code names and hands does not know"),
        pytest.param(going(Unknown("compacting")), "in a state hands does not know: compacting", id="a status hands does not know"),
    ],
)
def test_a_session_stands_in_one_sentence(session: Session, says: str) -> None:
    assert spoken_state(session) == says


def test_only_a_session_at_its_prompt_with_nothing_out_is_at_rest() -> None:
    assert standing(Session(MEMBER, Parked(Idle(), Stamp(1000), None), mode=None)) == AtPrompt()
    assert standing(Session(MEMBER, Parked(Idle(), Stamp(1000), None), mode=None, turn=Opened(TURN))) != AtPrompt()
    assert standing(Session(MEMBER, Parked(Shell(), Stamp(1000), None), mode=None)) != AtPrompt()


def test_the_listing_says_a_session_as_the_one_sentence_does() -> None:
    session = going(Busy(), turn=Opened(TURN, latest=RUN_TESTS))
    assert describe_listing(Listing(session, "naming fix")) == {"id": "s", "name": "cc-hands, naming fix", "state": spoken_state(session), "mode": "not reported yet"}
