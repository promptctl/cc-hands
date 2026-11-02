"""Where a turn typed into Claude Code now goes, read off one `tmux capture-pane -p -e` capture of its screen, its
attributes kept as SGR: conditions 1 to 3 of the rule in docs/claude-code-screen.md ("The rule"), and whether the input
holds words. Condition 4, a settled screen, is two captures whose input boxes agree, and so belongs to whoever takes them.

Pure: the capture is the input, and nothing here reads the screen.
"""

import re
from dataclasses import dataclass

# The prompt input's first row: `❯`, or `!` in bash mode, then U+00A0. tmux returns U+00A0 as its own character, and
# Python's `strip` takes it for a space, so a row is matched before anything is stripped, and only ASCII spaces are.
_PROMPT = re.compile("^ *([❯!])\u00a0")
# What, below the prompt row, holds the keys instead of it: a select's cursor (`❯` and an ordinary space), numbered or
# not, or the autocomplete list's; and the inline history search.
_TAKEN = re.compile("^ *❯ |^ *(search prompts:|no matching prompt:)")
# An SGR sequence, the only escape `capture-pane -e` writes: its parameters, `;` or `:` separated.
_SGR = re.compile("\x1b\\[([0-9;:]*)m")


@dataclass(frozen=True)
class Box:
    """Claude Code's input box as the screen shows it: its rows as text, from its top rule down, and whether its input
    holds words. [LAW:types-are-the-program] `drafted` is read off the attributes, which `rows` has none of: the empty
    input shows its placeholder, text that is not words."""

    rows: str
    drafted: bool


def plain(shown: str) -> str:
    """The capture `shown` as text, its attributes dropped."""
    return _SGR.sub("", shown)


def _drafted(row: str) -> bool:
    """Whether the input row `row`, its prefix gone, shows words. Claude Code draws its placeholder only while the input is
    empty, dim, its first character inverse as the cursor when the input has focus (docs/claude-code-behavior.md); so a
    non-blank character drawn neither dim nor as an inverse cursor ahead of dim text is a word."""
    cells: list[tuple[str, bool, bool]] = []
    dim = inverse = False
    at = 0
    for sgr in _SGR.finditer(row):
        cells += [(char, dim, inverse) for char in row[at:sgr.start()]]
        at = sgr.end()
        dim, inverse = _styled(sgr[1], dim, inverse)
    cells += [(char, dim, inverse) for char in row[at:]]
    return any(
        not char.isspace() and not dim and not (inverse and at + 1 < len(cells) and cells[at + 1][1])
        for at, (char, dim, inverse) in enumerate(cells)
    )


def _styled(parameters: str, dim: bool, inverse: bool) -> tuple[bool, bool]:
    """Dim and inverse once the SGR with `parameters` is applied: a colour's own parameters (38, 48 and 58, `;5;n` or
    `;2;r;g;b`, or `:`-joined) are skipped, so none of their numbers reads as an attribute."""
    codes = [code.split(":")[0] for code in parameters.split(";")]
    at = 0
    while at < len(codes):
        match codes[at]:
            case "" | "0":
                dim = inverse = False
            case "2":
                dim = True
            case "22":
                dim = False
            case "7":
                inverse = True
            case "27":
                inverse = False
            case "38" | "48" | "58" if at + 1 < len(codes):
                at += {"5": 2, "2": 4}.get(codes[at + 1], 0)
            case _:
                pass
        at += 1
    return dim, inverse


def typable(shown: str) -> Box | None:
    """The input box of the screen `shown`, a `-e` capture, from its top rule down, when a turn pasted now goes into Claude Code's prompt
    input: a prompt row is drawn as the box's first row, under its top rule, nothing below it holds the keys, and the
    prompt is not in bash mode, where the turn would run as a shell command. None otherwise.

    Only the box: what is drawn above it, a turn's spinner and streamed output, changes every frame while the prompt
    takes keys all along."""
    styled = shown.split("\n")
    rows = [plain(row) for row in styled]
    # Transcript text above the box can show a prompt row too; Claude Code draws the input box last.
    found = [(at, prompt[1]) for at, row in enumerate(rows) if (prompt := _PROMPT.match(row))]
    if not found:
        return None
    at, pointer = found[-1]
    if pointer != "❯" or at == 0 or not rows[at - 1].lstrip(" ").startswith("─") or any(_TAKEN.match(row) for row in rows[at + 1:]):
        return None
    # The input's rows run from the prompt row to the box's closing rule.
    end = next((below for below in range(at + 1, len(rows)) if rows[below].lstrip(" ").startswith("─")), len(rows))
    # The prompt's own `❯` and U+00A0 are blanked where they stand, so every attribute set before them still holds.
    drafted = any(_drafted(row) for row in [styled[at].replace("❯\u00a0", "  ", 1), *styled[at + 1:end]])
    return Box("\n".join(rows[at - 1:]), drafted)
