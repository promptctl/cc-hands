"""How a live session stands, in the one sentence the brain's listing says and a card on the phone reads."""

from pathlib import Path

import pytest

from hands.core.progress import RUNNING, Doing
from hands.core.session import AskedQuestion, Held, Idle as Parked, Membership, Opened, Option, Permission, Plan, PromptId, Question, RequestId, Running, Session, SessionId, Unreported
from hands.core.status import Busy, Idle, Shell, Stamp, Waiting
from hands.core.turn import AgentId
from hands.sessions.registry import Listing
from hands.voice.readback import describe_listing, spoken_state

MEMBER = Membership(SessionId("s"), pid=4242, cwd=Path("/code/cc-hands"), transcript=Path("/code/cc-hands/t.jsonl"))
TURN = PromptId("p1")
RUN_TESTS = Doing(RUNNING, "run the test suite")


def held(on: Permission | Question | Plan) -> tuple[Held, ...]:
    return (Held(on, RequestId("r"), asked=1.0),)


def going(status: Busy | Waiting, **session: object) -> Session:
    return Session(MEMBER, Running(status, Stamp(1000), None), mode=None, **session)  # pyright: ignore[reportArgumentType]


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
    ],
)
def test_a_session_stands_in_one_sentence(session: Session, says: str) -> None:
    assert spoken_state(session) == says


def test_the_listing_says_each_session_as_a_card_does() -> None:
    session = going(Busy(), turn=Opened(TURN, latest=RUN_TESTS))
    assert describe_listing(Listing(session, "naming fix")) == {"id": "s", "name": "cc-hands, naming fix", "state": spoken_state(session), "mode": "not reported yet"}
