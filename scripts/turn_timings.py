"""The timings docs/claude-code-behavior.md states of the recorded turns, computed from tests/fixtures/recorded_turns:
run it again whenever the turns are recorded again (scripts/record_turns.py), and the doc's lines follow it.

    uv run python scripts/turn_timings.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from turns import RECORDINGS, Posted, Recording, StatusSet, Written  # noqa: E402  [LAW:one-source-of-truth] the recordings' reader


def _hooks(recording: Recording, event: str) -> list[Posted]:
    return [line for line in recording if isinstance(line, Posted) and line.body["hook_event_name"] == event]


def timings(recording: Recording) -> dict[str, list[float]]:
    """Each gap, in seconds, for every turn of `recording` it is measured on."""
    gaps: dict[str, list[float]] = {"record": [], "busy": [], "stop": [], "display": []}
    busy = [line.at for line in recording if isinstance(line, StatusSet) and line.status["status"] == "busy"]
    opened: set[str] = set()
    for hook in _hooks(recording, "UserPromptSubmit"):
        prompt = hook.body["prompt_id"]
        if prompt in opened:
            # Words sent into a running turn: that turn's, not one it opens.
            continue
        opened.add(prompt)
        gaps["record"] += [next(line.at for line in recording if isinstance(line, Written) and line.record.get("type") == "user" and line.record.get("promptId") == prompt) - hook.at]
        gaps["busy"] += [min((at - hook.at for at in busy), key=abs)]
    for stop in _hooks(recording, "Stop"):
        prompt = stop.body["prompt_id"]
        gaps["stop"] += [next(line.at for line in recording if isinstance(line, Written) and line.at >= stop.at and line.record.get("subtype") == "stop_hook_summary") - stop.at]
        gaps["display"] += [max(line.at for line in _hooks(recording, "MessageDisplay") if line.body["prompt_id"] == prompt) - stop.at]
    return gaps


SAID = {
    "record": "a turn's opening prompt's user record, after its UserPromptSubmit",
    "busy": "the busy nearest a turn's opening UserPromptSubmit, after it (before it if negative)",
    "stop": "a Stop's stop_hook_summary record, after the Stop",
    "display": "a turn's last MessageDisplay, after its Stop",
}

if __name__ == "__main__":
    for kind, said in SAID.items():
        gaps = sorted(gap for recording in RECORDINGS.values() for gap in timings(recording)[kind])
        print(f"{said}: {len(gaps)} turns, {gaps[0]:.3f} to {gaps[-1]:.3f} s, {sum(gap < 0 for gap in gaps)} negative")
