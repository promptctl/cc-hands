"""Claude Code's terminal, as the unit layer plays it: what its screen shows and takes of the keys hands sends, for the claims
tests/test_terminal_contract.py holds to the pinned Claude Code itself.

[LAW:one-source-of-truth] what each screen text and each key does here is the contract's to say: the contract holds the pinned
Claude Code to the submit chord's two claims, and everything else here is played from docs/claude-code-screen.md and docs/
claude-code-behavior.md, a guess until a case of the contract holds it. A pin bump that changes a held claim fails the
contract's real case, and this is changed to match in the same commit.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

from terminalcontract import QUESTION

# What the permission dialog of a command Claude Code runs shows beside the contract's question: the choices, the first focused.
CHOICES = ("Yes", "Yes, and don't ask again", "No")
# A paste of this many lines or more is folded into a placeholder in the draft, and kept whole for the prompt it makes.
FOLDED = 4


@dataclass
class _Dialog:
    focus: int = 1
    # What Yes's own text field holds once Tab put the focus in it: None while it is not open.
    amended: str | None = None


@dataclass
class ClaudeScreen:
    """A Claude Code at its prompt, or at a permission dialog, as the keys hands sends reach it. Keys are named as tmux's
    send-keys names them, the submit chord among them."""

    chord: str
    # The draft, one entry per paste or run of typed text: a paste kept whole beside the placeholder it shows as.
    draft: list[tuple[str, str]] = field(default_factory=lambda: [])
    dialog: _Dialog | None = None
    # The turns Claude Code took, in order, as the words its prompt hook says.
    turns: list[str] = field(default_factory=lambda: [])
    pastes: int = 0
    # Whether Claude Code ran the command its dialog asked about: only Yes does that.
    ran: bool = False

    def at_dialog(self) -> None:
        self.dialog = _Dialog()

    def paste(self, text: str) -> None:
        # A permission dialog with no field open swallows a paste, and one with Yes's field open takes it there (docs/
        # claude-code-screen.md).
        match self.dialog:
            case _Dialog(amended=str()):
                self.type(text)
                return
            case _Dialog():
                return
            case None:
                pass
        if text.count("\n") + 1 < FOLDED:
            self.draft.append((text, text))
            return
        self.pastes += 1
        self.draft.append((f"[Pasted text #{self.pastes} +{text.count(chr(10))} lines]", text))

    def type(self, text: str) -> None:
        match self.dialog:
            case None:
                self.draft.append((text, text))
            case _Dialog(amended=str() as amended) as dialog:
                dialog.amended = amended + text
            case _:
                raise AssertionError(f"the fake Claude Code models no text typed at a dialog with no field open: {text!r}")

    def key(self, name: str) -> None:
        match (self.dialog, name):
            case (None, chord) if chord == self.chord and self.draft:
                # The chord at the prompt submits the draft, each paste whole.
                self.turns.append("".join(whole for _, whole in self.draft))
                self.draft = []
            case (None, chord) if chord == self.chord:
                # Nothing drafted, nothing to submit.
                pass
            case (None, "C-u" | "C-c"):
                # C-u kills the row to its start, and a draft of one row is all of it; C-c clears the input
                # (docs/claude-code-behavior.md, app:interrupt).
                self.draft = []
            case (None, "C-e" | "C-k" | "C-h" | "Escape"):
                # At the end of a row with nothing after it and no row before it, or with nothing running, these change nothing.
                pass
            case (_Dialog(focus=1, amended=None) as dialog, "Tab"):
                dialog.amended = ""
            case (_Dialog(focus=1), "Enter"):
                # Return is the yes key of every dialog (docs/claude-code-screen.md).
                self.ran, self.dialog = True, None
            case (_Dialog(), "Escape"):
                self.dialog = None
            case (_Dialog() as dialog, "Down"):
                dialog.focus = min(dialog.focus + 1, len(CHOICES))
            case (_Dialog(), chord) if chord == self.chord:
                # The chord at a dialog answers nothing: the contract's claim.
                pass
            case _:
                raise AssertionError(f"the fake Claude Code models no {name!r} at {'a dialog' if self.dialog else 'the prompt'}")

    def shown(self) -> str:
        match self.dialog:
            case None:
                return "\n".join(["─" * 40, "❯\u00a0" + "".join(shown for shown, _ in self.draft), "─" * 40])
            case _Dialog(focus=focus, amended=amended):
                rows = [f"{'❯' if at == focus else ' '} {at}. {label}{'' if at != 1 or amended is None else ': ' + amended}" for at, label in enumerate(CHOICES, start=1)]
                return "\n".join(["Bash command", QUESTION, *rows])

    def took(self, words: Sequence[str]) -> bool:
        return any(all(word in turn for word in words) for turn in self.turns)
