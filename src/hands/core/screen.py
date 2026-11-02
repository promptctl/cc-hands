"""Where a turn typed into Claude Code now goes, read off one `tmux capture-pane -p` capture of its screen: conditions 1
to 3 of the rule in docs/claude-code-screen.md ("The rule"). Condition 4, a settled screen, is two captures whose input
boxes agree, and so belongs to whoever takes them.

Pure: the capture is the input, and nothing here reads the screen.
"""

import re

# The prompt input's first row: `❯`, or `!` in bash mode, then U+00A0. tmux returns U+00A0 as its own character, and
# Python's `strip` takes it for a space, so a row is matched before anything is stripped, and only ASCII spaces are.
_PROMPT = re.compile("^ *([❯!])\u00a0")
# What, below the prompt row, holds the keys instead of it: a select's cursor (`❯` and an ordinary space), numbered or
# not, or the autocomplete list's; and the inline history search.
_TAKEN = re.compile("^ *❯ |^ *(search prompts:|no matching prompt:)")


def typable(shown: str) -> str | None:
    """The input box of the screen `shown`, from its top rule down, when a turn pasted now goes into Claude Code's prompt
    input: a prompt row is drawn as the box's first row, under its top rule, nothing below it holds the keys, and the
    prompt is not in bash mode, where the turn would run as a shell command. None otherwise.

    Only the box: what is drawn above it, a turn's spinner and streamed output, changes every frame while the prompt
    takes keys all along."""
    rows = shown.split("\n")
    # Transcript text above the box can show a prompt row too; Claude Code draws the input box last.
    found = [(at, prompt[1]) for at, row in enumerate(rows) if (prompt := _PROMPT.match(row))]
    if not found:
        return None
    at, pointer = found[-1]
    if pointer != "❯" or at == 0 or not rows[at - 1].lstrip(" ").startswith("─") or any(_TAKEN.match(row) for row in rows[at + 1:]):
        return None
    return "\n".join(rows[at - 1:])
