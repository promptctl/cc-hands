"""lit's line in its events file for a run it refused, as lit writes it (links docs/events.md): written for every fake of
lit, the one in this process and a stub script alike, and held to lit by the contract in tests/litcontract.py.

    python litevents.py <reason> <command>    # a stub script's refusal, in the environment and directory it runs in
"""

import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path

from hands.sessions.backlog import events_file, rotated as rotation

# [LAW:one-source-of-truth] the code lit gives each reason a fake of lit refuses on (docs/cli-reference.md in links,
# "Every reason"): each of the first two may follow a commit, and code 6 is a write whose asked state already holds.
CODES = {"command_failed": 1, "workspace_busy": 1, "workspace_not_initialized": 3, "outside_git_workspace": 3, "state_already_holds": 6}


def refused(environment: Mapping[str, str], cwd: Path, command: str, reason: str, *, worker: str | None = None, rotated: bool = False) -> int:
    """Append lit's line for a run of `command` in `cwd` refused on `reason`, under the trace `environment` hands it, to
    the events file lit keeps there (`backlog.events_file`), or to the `.1` it rotated it to, and none where lit keeps
    none; the code lit exits on. With `worker`, the line is instead that of a worker the run spawned, under that span."""
    _, trace, span, _ = environment["TRACEPARENT"].split("-")
    match events_file(environment, cwd):
        case Path() as events:
            events = rotation(events) if rotated else events
            events.parent.mkdir(parents=True, exist_ok=True)
            line = {"event": command, "trace_id": trace, "parent_span_id": worker or span, "exit_code": CODES[reason], "outcome": "error", "reason": reason}
            with events.open("a") as lines:
                lines.write(json.dumps(line) + "\n")
        case _:
            pass
    return CODES[reason]


if __name__ == "__main__":
    reason, command = sys.argv[1:]
    refused(os.environ, Path.cwd(), command, reason)
