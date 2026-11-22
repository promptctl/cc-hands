"""What hands relies on of lit's writes and its export, as one contract held to two parties: the fake lit the unit layer
plays (FakeLit in tests/test_backlogpage.py), in `make check`, and lit itself, in a workspace of its own, in the outside
tier. The cases are written once, here, through hands' own adapter (hands.sessions.backlog), and each holder runs all of
them against its own `project`, through its own `children`, so each asserts the same of both: that is what holds the fake to lit. A claim lit no
longer keeps fails the run against lit, and the fake is changed to match.

The claims: an epic is made at the bottom of its frame and found by its frame, title, and topic; a ticket filed is at the
top of its epic; a rank puts a ticket beside its target; a move keeps the ticket's rank, and the same move again is code 6,
state_already_holds; a comment is held signed by its signer; each write's first word is the ticket it wrote."""

import os
from pathlib import Path

import pytest

from hands.sessions.backlog import Backlog, LitFailed, Move, NewEpic, Person, Place, adopt, comment, file_ticket, make_epic, rank, read_backlog
from hands.sessions.child import Children

SIGNER = Person("hands-contract", os.environ)


async def read(children: Children, project: Path) -> Backlog:
    backlog = await read_backlog(children, project)
    assert isinstance(backlog, Backlog), backlog
    return backlog


async def made(children: Children, project: Path, title: str) -> str:
    """The id of the epic lit made titled `title`, among the roots, the first word it said."""
    said = await make_epic(children, project, NewEpic("wake-word", title, None, Place("above", "unused")), SIGNER)
    return said.split()[0]


async def filed(children: Children, project: Path, epic: str, title: str) -> str:
    return await file_ticket(children, project, epic, "wake-word", title, f"The words of {title}.", "contract")


class LitContract:
    """The claims, one method each. A holder subclasses it and gives a `project` fixture and the `children` that run its lit: a directory lit holds an empty
    backlog for, as the party it holds runs it."""

    async def test_an_epic_made_is_at_the_bottom_of_its_frame_and_found_by_its_title_and_topic(self, project: Path, children: Children) -> None:
        train = await made(children, project, "Train a new wake word")
        high = await made(children, project, "wake word - high")
        backlog = await read(children, project)
        assert backlog.roots() == (train, high)
        assert backlog.level_epics("wake-word", None) == {"high": high}

    async def test_a_ticket_filed_is_at_the_top_of_its_epic(self, project: Path, children: Children) -> None:
        epic = await made(children, project, "Train a new wake word")
        first = await filed(children, project, epic, "Hey hands")
        second = await filed(children, project, epic, "Many wake words")
        backlog = await read(children, project)
        assert backlog.children[epic] == (second, first)
        assert (backlog.tickets[first].title, backlog.parent[first]) == ("Hey hands", epic)

    async def test_a_rank_puts_the_ticket_directly_beside_its_target(self, project: Path, children: Children) -> None:
        epic = await made(children, project, "Train a new wake word")
        first = await filed(children, project, epic, "Hey hands")
        second = await filed(children, project, epic, "Many wake words")
        move = Move(second, Place("below", first))
        said = await rank(children, project, move, SIGNER)
        assert said.split()[0] == second
        backlog = await read(children, project)
        assert backlog.landed(move) and backlog.children[epic] == (first, second)

    async def test_a_move_keeps_the_tickets_rank_and_the_same_move_again_already_holds(self, project: Path, children: Children) -> None:
        epic = await made(children, project, "Train a new wake word")
        ticket = await filed(children, project, epic, "Hey hands")
        high = await made(children, project, "wake word - high")
        order = tuple((await read(children, project)).tickets)
        said = await adopt(children, project, ticket, high, SIGNER)
        assert said.split()[0] == ticket
        backlog = await read(children, project)
        assert backlog.parent[ticket] == high and tuple(backlog.tickets) == order
        with pytest.raises(LitFailed, match="exited 6") as again:
            await adopt(children, project, ticket, high, SIGNER)
        assert again.value.reason == "state_already_holds"

    async def test_a_comment_is_held_on_its_ticket_signed_by_its_signer(self, project: Path, children: Children) -> None:
        epic = await made(children, project, "Train a new wake word")
        ticket = await filed(children, project, epic, "Hey hands")
        said = await comment(children, project, ticket, "Split it", SIGNER)
        assert said.split()[0] == ticket
        [note] = (await read(children, project)).comments[ticket]
        assert (note.by, note.body) == (SIGNER.name, "Split it")
