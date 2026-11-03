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
# The escapes `capture-pane -e` writes: an SGR, its parameters `;` or `:` separated, and an OSC 8 hyperlink, which
# draws nothing.
_ESCAPE = re.compile("\x1b\\[([0-9;:]*)m|\x1b\\]8;[^\x07\x1b]*(?:\x07|\x1b\\\\)")
# A cell as drawn: its character, whether dim, whether inverse.
_Cell = tuple[str, bool, bool]


@dataclass(frozen=True)
class Box:
    """Claude Code's input box as the screen shows it: its rows as text, from its top rule down, how many of them the
    input fills, and whether it holds words. [LAW:types-are-the-program] `drafted` is read off the attributes, which
    `rows` has none of: the empty input shows its placeholder, text that is not words."""

    rows: str
    input: int
    drafted: bool


def plain(shown: str) -> str:
    """The capture `shown` as text, its escapes dropped."""
    return _ESCAPE.sub("", shown)


def _cells(styled: list[str]) -> list[list[_Cell]]:
    """The rows `styled` as cells. tmux writes an SGR only where the attributes change, and carries them across a line
    break, so each row starts in the attributes the row above ended in."""
    rows: list[list[_Cell]] = []
    dim = inverse = False
    for row in styled:
        cells: list[_Cell] = []
        drawn = 0
        for escape in _ESCAPE.finditer(row):
            cells += [(char, dim, inverse) for char in row[drawn:escape.start()]]
            drawn = escape.end()
            if escape[1] is not None:
                dim, inverse = _styled(escape[1], dim, inverse)
        cells += [(char, dim, inverse) for char in row[drawn:]]
        rows.append(cells)
    return rows


def _drafted(cells: list[_Cell]) -> bool:
    """Whether the input cells `cells`, the prompt's own gone, show words. Claude Code draws its placeholder only while
    the input is empty, dim, its first character inverse as the cursor when the input has focus
    (docs/claude-code-behavior.md); so a non-blank character drawn neither dim nor as an inverse cursor ahead of dim text
    is a word."""
    for column, (char, dim, inverse) in enumerate(cells):
        cursor_on_placeholder = inverse and column + 1 < len(cells) and cells[column + 1][1]
        if not char.isspace() and not dim and not cursor_on_placeholder:
            return True
    return False


def _styled(parameters: str, dim: bool, inverse: bool) -> tuple[bool, bool]:
    """Dim and inverse once the SGR with `parameters` is applied: a colour's own parameters (38, 48 and 58, `;5;n` or
    `;2;r;g;b`, or `:`-joined) are skipped, so none of their numbers reads as an attribute. A `:`-joined colour carries
    its own parameters, so the `;`-separated one after it is the next attribute."""
    codes = parameters.split(";")
    at = 0
    while at < len(codes):
        match codes[at].split(":")[0]:
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
            case "38" | "48" | "58" if ":" not in codes[at] and at + 1 < len(codes):
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
    # The input's rows run from the prompt row to the box's closing rule, drawn in the prompt's column, where a draft's
    # own rows are indented past the prompt's `❯` and so never end it, whatever they start with.
    column = rows[at].index(pointer)
    end = next((below for below in range(at + 1, len(rows)) if rows[below][column:].startswith("─") and not rows[below][:column].strip(" ")), len(rows))
    # The prompt's own `❯` and U+00A0 are dropped by their columns, wherever an SGR falls among them.
    cells = _cells(styled)
    drafted = any(_drafted(row) for row in [cells[at][column + 2:], *cells[at + 1:end]])
    return Box("\n".join(rows[at - 1:]), end - at, drafted)
