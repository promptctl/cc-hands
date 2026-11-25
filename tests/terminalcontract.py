"""What hands relies on of the brain's terminal, as one contract held to two parties: the fake Claude Code the unit layer plays
(tests/claudescreen.py), in `make check` (test_terminal_contract.py), and the pinned Claude Code itself, in a terminal of
hands' own against the mock of Anthropic's API, in `make check-outside` (test_terminal_contract_pinned.py). The cases are
written once, here, and each of those modules runs all of them against its own `terminal`, so each asserts the same of both:
that is what holds the fake to the pin. A claim the pin no longer keeps fails the pinned run, and the fake is changed to
match.

The claims: the submit chord sends words pasted at the prompt, and the same chord typed into a held permission dialog answers
nothing, whether Yes or Yes's own text field has the focus (promptctl-brain-z04s.ar0)."""

from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol

import pytest

from hands.brain.process import SUBMIT
from hands.sessions.tmux import tmux_key

# [LAW:one-source-of-truth] the brain's submit chord, as tmux names the key: the one hands types, whose Claude Code name
# hands binds in the brain's keybindings.json at every start.
CHORD = tmux_key(SUBMIT)
# What the permission dialog of a command Claude Code will not run unless allowed asks: the claim both parties are held to.
QUESTION = "Do you want to proceed?"


class Terminal(Protocol):
    """The brain's terminal as the contract drives it, keys named as tmux's send-keys names them."""

    async def at_dialog(self) -> None:
        """Claude Code stops at a permission dialog for a command it will not run unless it is allowed."""
        ...

    async def paste(self, text: str) -> None: ...

    async def type(self, text: str) -> None: ...

    async def key(self, name: str) -> None: ...

    async def shows(self, text: str) -> str:
        """The screen, once it shows `text`."""
        ...

    async def took(self, words: Sequence[str]) -> None:
        """Returns once Claude Code has taken a turn holding all of `words`."""
        ...

    async def ran(self) -> bool:
        """Whether the command the permission dialog asked about has run."""
        ...


# What is done at the dialog before the chord, and what the screen shows once Claude Code has read it, so the chord arrives as a
# read of its own, never in one with the words before it.
_Step = Callable[[Terminal], Awaitable[None]]


async def _tab(terminal: Terminal) -> None:
    await terminal.key("Tab")


async def _amend(terminal: Terminal) -> None:
    await terminal.type("use a copy instead")


class TerminalContract:
    """The claims, one method each. A holder subclasses it and gives a `terminal` fixture, so every claim added here is held
    to both parties without either holder naming it."""

    @pytest.mark.parametrize(("before", "shows"), [([], QUESTION), ([_tab, _amend], "use a copy instead")], ids=["Yes focused", "Yes amended, its field holding words"])
    async def test_the_submit_chord_typed_into_a_held_permission_dialog_answers_nothing(self, terminal: Terminal, before: list[_Step], shows: str) -> None:
        """The chord, bound to the prompt's submit, typed while a permission dialog is on the screen: the dialog stays up and
        the command does not run, whether Yes or Yes's own text field has the focus. A Down typed after it moves the focus,
        so once the screen shows it moved, Claude Code has read everything typed before it."""
        await terminal.at_dialog()
        for step in before:
            await step(terminal)
        await terminal.shows(shows)
        await terminal.key(CHORD)
        await terminal.key("Down")
        # The second option, whatever it says: a command's dialog in default mode offers "Yes, and don't ask again" there.
        after = await terminal.shows("❯ 2.")
        assert QUESTION in after
        assert not await terminal.ran()

    async def test_words_pasted_at_the_prompt_are_sent_by_the_submit_chord(self, terminal: Terminal) -> None:
        """A paste long enough that Claude Code folds it into a placeholder, then the chord: the whole text is sent as a turn."""
        first, last, closing = "line 1 of a long turn", "line 29 of a long turn", "and last, say you read it"
        words = "\n".join(f"line {line} of a long turn said to the brain" for line in range(1, 30)) + f"\n{closing}"
        await terminal.paste(words)
        await terminal.shows("[Pasted text #1 +29 lines]")
        await terminal.key(CHORD)
        await terminal.took([first, last, closing])
