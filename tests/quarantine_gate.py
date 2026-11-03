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
from datetime import UTC, date, datetime
from pathlib import Path

# A quarantine may last this many days from its start before CI fails on it.
QUARANTINE_DAYS = 7
ROOT = Path(__file__).parents[1]


def markers(tests: Path) -> list[str]:
    """One entry per quarantine marker in the source under `tests`, read from the syntax tree, so a marker's text in a string
    or a comment is not one. Every `quarantined` attribute or name is a marker, and a reference this scan cannot read stops the
    gate: a bare reference that is not a call, a call without its ticket as the one string, and `getattr(..., "quarantined")`.
    A spelling built some other way (a computed name, a string passed through a helper) is not seen; conftest would honour it,
    so it is not covered here."""
    tickets: list[str] = []
    for source in tests.rglob("*.py"):
        text = source.read_text()
        nodes = [node for node in ast.walk(ast.parse(text, filename=str(source))) if isinstance(node, ast.expr)]
        called = {id(node.func) for node in nodes if isinstance(node, ast.Call)}
        for node in nodes:
            # [LAW:no-silent-failure] getattr(mark, "quarantined") reaches the same marker with no attribute to read, so it stops the gate too.
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr" and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and node.args[1].value == "quarantined":
                raise ValueError(f'{source}:{node.lineno}: quarantined is looked up with getattr, so its ticket cannot be read here: write pytest.mark.quarantined("<ticket id>")')
            if isinstance(node, ast.Call) and _names_quarantined(node.func):
                ticket = node.args[0] if len(node.args) == 1 and not node.keywords else None
                if not (isinstance(ticket, ast.Constant) and isinstance(ticket.value, str) and ticket.value):
                    raise ValueError(f'{source}:{node.lineno}: quarantined needs its ticket as the one string: quarantined("<ticket id>")')
                # [LAW:one-source-of-truth] the start is searched for in the double-quoted spelling alone, so a ticket written any
                # other way would have its clock started at the respelling; it is refused instead.
                if not (ast.get_source_segment(text, ticket) or "").startswith('"'):
                    raise ValueError(f'{source}:{node.lineno}: quarantined spells its ticket with double quotes, so its start can be found: quarantined("<ticket id>")')
                tickets.append(ticket.value)
            elif _names_quarantined(node) and id(node) not in called:
                raise ValueError(f'{source}:{node.lineno}: quarantined must be called with its ticket: quarantined("<ticket id>")')
    return tickets


def _names_quarantined(node: ast.expr) -> bool:
    """Whether `node` is a `quarantined` attribute or name, under whatever module alias the file gives pytest's mark."""
    return (isinstance(node, ast.Attribute) and node.attr == "quarantined") or (isinstance(node, ast.Name) and node.id == "quarantined")


def started(root: Path, ticket: str) -> datetime:
    """When the history first recorded this ticket's marker: the earliest author date of any commit, repo-wide, that changes
    how many times the marker's text occurs. The minimum over every listed date, not the first line, because git lists by
    commit order and a commit can be listed first with a later author date. No pathspec, so a file moved keeps its clock."""
    dates = subprocess.run(
        ["git", "log", "-S", f'quarantined("{ticket}")', "--format=%aI"],
        cwd=root, capture_output=True, text=True, check=True,
    ).stdout.split()
    if not dates:
        raise ValueError(f"no commit in this history changes the quarantine of {ticket}: its marker is not committed here")
    return min(datetime.fromisoformat(listed) for listed in dates)


def overdue(starts: Mapping[str, datetime], today: date) -> list[str]:
    """Each quarantine whose start is more than QUARANTINE_DAYS before `today`, with its age, sorted by ticket. Each start is
    taken in UTC, so the boundary does not move with the author's offset."""
    late: list[str] = []
    for ticket, start in sorted(starts.items()):
        # [LAW:no-silent-failure] a start after today is a clock that is wrong, and a negative age would pass the gate forever.
        age = (today - start.astimezone(UTC).date()).days
        if age < 0:
            raise ValueError(f"{ticket}: its quarantine starts {-age} days after today, so the history's dates are wrong")
        if age > QUARANTINE_DAYS:
            late.append(f"{ticket}: quarantined for {age} days, past its {QUARANTINE_DAYS}")
    return late


def main(root: Path, today: date) -> int:
    # [LAW:no-silent-failure] a shallow history shows a commit's tree as added, so the tip would date every quarantine as fresh.
    shallow = subprocess.run(
        ["git", "rev-parse", "--is-shallow-repository"], cwd=root, capture_output=True, text=True, check=True,
    ).stdout.strip()
    if shallow == "true":
        raise ValueError("this checkout is shallow, so its history cannot date a quarantine: fetch the full history")
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
    sys.exit(main(ROOT, datetime.now(UTC).date()))
