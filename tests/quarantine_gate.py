"""[LAW:single-enforcer] the one owner of how long a quarantined flake may stay quarantined: QUARANTINE_DAYS, and the check that
fails CI once a quarantine is older. A quarantine starts when git first records its marker: the earliest commit, repo-wide,
that changes how many times `quarantined("<ticket id>")` occurs. A rename does not reset it, a reindent does not, and a
rebase keeps the author date. A quarantine the history cannot date stops the gate, loudly.

    uv run --no-sync python tests/quarantine_gate.py

Standard library only, so CI runs it with no test environment, after the suite, from a checkout with its full history.
"""

import ast
import subprocess
import sys
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path

# A quarantine may last this many days from its start before CI fails on it.
QUARANTINE_DAYS = 7
ROOT = Path(__file__).parents[1]


def markers(tests: Path) -> list[str]:
    """One entry per `quarantined("<ticket id>")` marker in the source under `tests`, read from the syntax tree, so a
    marker's text in a string or a comment is not one. Matches `pytest.mark.quarantined` and `mark.quarantined` alike, the
    two spellings conftest's marker lookup honours, and refuses a marker that does not name its ticket as a string."""
    tickets: list[str] = []
    for source in tests.rglob("*.py"):
        nodes = list(ast.walk(ast.parse(source.read_text(), filename=str(source))))
        called = {id(node.func) for node in nodes if isinstance(node, ast.Call)}
        for node in nodes:
            if isinstance(node, ast.Attribute) and ast.unparse(node).endswith("mark.quarantined") and id(node) not in called:
                raise ValueError(f'{source}:{node.lineno}: quarantined needs its ticket as a string: quarantined("<ticket id>")')
            if isinstance(node, ast.Call) and ast.unparse(node.func).endswith("mark.quarantined"):
                ticket = node.args[0] if node.args else None
                if not (isinstance(ticket, ast.Constant) and isinstance(ticket.value, str) and ticket.value):
                    raise ValueError(f'{source}:{node.lineno}: quarantined needs its ticket as a string: quarantined("<ticket id>")')
                tickets.append(ticket.value)
    return tickets


def started(root: Path, ticket: str) -> datetime:
    """When the history first recorded this ticket's marker: the author date of the earliest commit, repo-wide, that
    changes how many times the marker's text occurs. No pathspec, so a file moved keeps its clock."""
    dates = subprocess.run(
        ["git", "log", "-S", f'quarantined("{ticket}")', "--reverse", "--format=%aI"],
        cwd=root, capture_output=True, text=True, check=True,
    ).stdout.split()
    if not dates:
        raise ValueError(f"no commit in this history changes the quarantine of {ticket}: its marker is not committed here")
    return datetime.fromisoformat(dates[0])


def overdue(starts: Mapping[str, datetime], today: date) -> list[str]:
    """Each quarantine whose start is more than QUARANTINE_DAYS before `today`, with its age, sorted by ticket."""
    late: list[str] = []
    for ticket, start in sorted(starts.items()):
        age = (today - start.date()).days
        if age > QUARANTINE_DAYS:
            late.append(f"{ticket}: quarantined for {age} days, past its {QUARANTINE_DAYS}")
    return late


def main(root: Path, today: date) -> int:
    found = markers(root / "tests")
    # [LAW:no-silent-failure] a quarantine's clock is its ticket's, so two tests under one ticket would share it, and a test
    # added later would inherit the older test's start and fail at once. One ticket, one test: refused, with the fix named.
    shared = sorted({ticket for ticket in found if found.count(ticket) > 1})
    for ticket in shared:
        print(f"{ticket}: quarantines more than one test; a quarantine is one test, so give each its own ticket")
    if shared:
        return 1
    late = overdue({ticket: started(root, ticket) for ticket in found}, today)
    for line in late:
        print(line)
    return 1 if late else 0


if __name__ == "__main__":
    sys.exit(main(ROOT, date.today()))
