"""Whether a key typed into Claude Code now reaches its prompt input, read off one `tmux capture-pane -p` capture of its
screen: conditions 1 to 3 of the rule in docs/claude-code-screen.md ("The rule"). Condition 4, a settled screen, is two
equal captures, and so belongs to whoever takes them.

Pure: the capture is the input, and nothing here reads the screen.
"""

import re

# The prompt input's first row: `❯`, or `!` in bash mode, then U+00A0. tmux returns U+00A0 as its own character, and
# Python's `strip` takes it for a space, so a row is matched before anything is stripped, and only ASCII spaces are.
_PROMPT = re.compile("^ *[❯!] ")
# What, below the prompt row, holds the keys instead of it: a select's cursor (`❯` and an ordinary space), numbered or
# not, or the autocomplete list's; and the inline history search.
_TAKEN = re.compile("^ *❯ |^ *(search prompts:|no matching prompt:)")


def reaches(shown: str) -> bool:
    """Whether a key typed into the screen `shown` reaches Claude Code's prompt input: a prompt row is drawn as the first
    row of the input box, under the box's top rule, and nothing below it holds the keys."""
    rows = shown.split("\n")
    # Transcript text above the box can show a prompt row too; Claude Code draws the input box last.
    found = [at for at, row in enumerate(rows) if _PROMPT.match(row)]
    if not found:
        return False
    at = found[-1]
    return at > 0 and rows[at - 1].lstrip(" ").startswith("─") and not any(_TAKEN.match(row) for row in rows[at + 1:])
