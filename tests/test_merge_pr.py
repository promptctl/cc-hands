"""scripts/merge-pr.sh merges a PR through master's merge queue and returns only once GitHub says it merged or will not.
Each test puts a gh of its own first on a PATH of the system's directories, so no gh that can reach GitHub is found:
it answers each of the script's questions, in turn, with a pull request as GitHub's GraphQL API gives one, through the
script's own jq expression, and records what it was asked to do."""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from conftest import SYSTEM_PATH, executable

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "merge-pr.sh"
PR = "7"
ENABLED = "2026-10-29T10:00:00Z"

# gh as the script calls it: `pr merge`, and `pr merge --disable-auto`, exit as the test says, and each `api graphql` answers with the next of the
# test's pull requests through the --jq expression it was given, as gh does; a question past the last fails.
GH = """#!/bin/sh
here=$(dirname "$0")
case "$1 $2" in
  "pr merge") echo "$*" >> "$here/calls"; case $3 in --disable-auto) exit "$(cat "$here/disable-exit")" ;; esac
    exit "$(cat "$here/merge-exit")" ;;
  "api graphql")
    asked=$(($(cat "$here/asked") + 1)); echo "$asked" > "$here/asked"
    for arg in "$@"; do case $arg in number=*) echo "api graphql $arg" >> "$here/calls" ;; esac; done
    test -f "$here/$asked.json" || { echo "gh: no answer for question $asked" >&2; exit 1; }
    while [ "$1" != --jq ]; do shift; done
    exec jq -r "$2" "$here/$asked.json" ;;
esac
echo "gh: not a call merge-pr.sh makes: $*" >&2; exit 64
"""


def check(name: str, conclusion: str | None, required: bool = True, run: int = 1) -> dict[str, Any]:
    """A check run, `run` its databaseId: a later run of a check has a greater one."""
    return {"name": name, "conclusion": conclusion, "isRequired": required, "databaseId": run}


def status(context: str, state: str, required: bool = True) -> dict[str, Any]:
    return {"context": context, "state": state, "isRequired": required}


def pull(
    state: str = "OPEN",
    *,
    position: int | None = None,
    auto: bool = False,
    mergeable: str = "MERGEABLE",
    checks: tuple[dict[str, Any], ...] = (),
    removed: tuple[str, str] | None = None,
) -> dict[str, Any]:
    """A pull request as the script's query gets it: in the queue at `position`, its auto-merge enabled at ENABLED,
    the `checks` of its head, and its last removal from the queue, at a time and for a reason."""
    return {"data": {"repository": {"pullRequest": {
        "state": state,
        "mergeable": mergeable,
        "isInMergeQueue": position is not None,
        "mergeQueueEntry": None if position is None else {"position": position},
        "autoMergeRequest": {"enabledAt": ENABLED} if auto else None,
        "commits": {"nodes": [{"commit": {"statusCheckRollup": {"contexts": {"nodes": list(checks)}}}}]},
        "timelineItems": {"nodes": [] if removed is None else [{"createdAt": removed[0], "reason": removed[1]}]},
    }}}}  # fmt: skip


@dataclass(frozen=True)
class Merged:
    code: int
    said: list[str]
    calls: list[str]


def merge(tmp_path: Path, *answers: dict[str, Any], merge_exit: int = 0, disable_exit: int = 0, args: tuple[str, ...] = (PR, "0")) -> Merged:
    """The script run on `args` against a gh that answers with `answers` in turn; what it said is its stderr's lines."""
    gh = executable(tmp_path / "gh" / "gh", GH).parent
    (gh / "merge-exit").write_text(str(merge_exit))
    (gh / "disable-exit").write_text(str(disable_exit))
    (gh / "asked").write_text("0")
    (gh / "calls").touch()
    for asked, answer in enumerate(answers, 1):
        (gh / f"{asked}.json").write_text(json.dumps(answer))
    ran = subprocess.run((str(SCRIPT), *args), env={"PATH": f"{gh}:{SYSTEM_PATH}"}, capture_output=True, text=True)
    return Merged(ran.returncode, ran.stderr.splitlines(), (gh / "calls").read_text().splitlines())


def test_a_pr_that_merges_through_the_queue_exits_0_having_said_each_place_it_was(tmp_path: Path) -> None:
    pending = (check("test", None),)
    passed = (check("test", "SUCCESS"),)
    ran = merge(tmp_path, pull(), pull(auto=True, checks=pending), pull(auto=True, checks=pending), pull(position=2, checks=passed), pull(position=1, checks=passed), pull("MERGED"))
    assert ran.code == 0, ran.said
    # A place it is still in is said once.
    assert ran.said == [
        "merge-pr: #7 WAITING for its checks before it is queued",
        "merge-pr: #7 QUEUED at position 2",
        "merge-pr: #7 QUEUED at position 1",
        "merge-pr: #7 MERGED",
    ]
    assert ran.calls == [f"api graphql number={PR}", f"pr merge {PR}"] + [f"api graphql number={PR}"] * 5


@pytest.mark.parametrize(
    "checks",
    [(check("test", "FAILURE"),), (check("lint", "SUCCESS"), check("test", "TIMED_OUT")), (status("test", "ERROR"),)],
    ids=["its check run failed", "one of its check runs timed out", "its commit status errored"],
)
def test_a_required_check_that_fails_before_the_queue_ends_the_wait_and_the_auto_merge(tmp_path: Path, checks: tuple[dict[str, Any], ...]) -> None:
    """GitHub keeps waiting to queue it, and nothing would ever end the wait but this."""
    ran = merge(tmp_path, pull(), pull(auto=True, checks=checks))
    assert ran.code == 1
    assert ran.said == ["merge-pr: #7 WILL NOT MERGE: its required test failed"]
    assert ran.calls[-1] == f"pr merge --disable-auto {PR}"


def test_a_check_that_is_not_required_failing_does_not_end_the_wait(tmp_path: Path) -> None:
    checks = (check("lint", "FAILURE", required=False), check("test", "SUCCESS"))
    ran = merge(tmp_path, pull(), pull(auto=True, checks=checks), pull("MERGED"))
    assert ran.code == 0, ran.said
    assert ran.said[0] == "merge-pr: #7 WAITING for its checks before it is queued"


@pytest.mark.parametrize(
    ("checks", "said"),
    [
        ((check("test", "CANCELLED", run=1), check("test", None, run=2)), "WAITING for its checks before it is queued"),
        ((check("test", "SUCCESS", run=1), check("test", "FAILURE", run=2)), "WILL NOT MERGE: its required test failed"),
    ],
    ids=["a run a later one replaced", "a run that a later one failed after"],
)
def test_a_check_is_its_latest_run(tmp_path: Path, checks: tuple[dict[str, Any], ...], said: str) -> None:
    """As the gate reads it: a cancelled run that a rerun on the same commit replaced is not the check's result."""
    ran = merge(tmp_path, pull(), pull(auto=True, checks=checks), pull("MERGED"))
    assert ran.said[0] == f"merge-pr: #7 {said}"


def test_a_pr_that_conflicts_with_master_while_it_waits_ends_the_wait_and_the_auto_merge(tmp_path: Path) -> None:
    ran = merge(tmp_path, pull(), pull(auto=True, mergeable="CONFLICTING"))
    assert ran.code == 1
    assert ran.said == ["merge-pr: #7 WILL NOT MERGE: it conflicts with master"]
    assert ran.calls[-1] == f"pr merge --disable-auto {PR}"


def test_a_pr_the_queue_drops_that_github_leaves_set_to_merge_ends_the_wait_and_the_auto_merge(tmp_path: Path) -> None:
    ran = merge(tmp_path, pull(), pull(position=1), pull(auto=True, removed=("2026-10-29T10:20:00Z", "FAILED_CHECKS")))
    assert ran.code == 1
    assert ran.said[-1] == "merge-pr: #7 WILL NOT MERGE: removed from the merge queue: FAILED_CHECKS"
    assert ran.calls[-1] == f"pr merge --disable-auto {PR}"


def test_a_removal_before_this_auto_merge_was_enabled_does_not_end_the_wait(tmp_path: Path) -> None:
    """A PR dropped once and merged again with this script is waiting on its new auto-merge, not dropped."""
    removed = ("2026-10-29T09:00:00Z", "FAILED_CHECKS")
    ran = merge(tmp_path, pull(removed=removed), pull(auto=True, removed=removed), pull("MERGED"))
    assert ran.code == 0, ran.said
    assert f"pr merge {PR}" in ran.calls


@pytest.mark.parametrize(
    ("answer", "said"),
    [
        (pull(removed=("2026-10-29T10:20:00Z", "FAILED_CHECKS")), "NOT QUEUED: removed from the merge queue: FAILED_CHECKS"),
        (pull(), "NOT QUEUED: never in the merge queue"),
        (pull("CLOSED"), "CLOSED"),
    ],
    ids=["dropped by the queue", "never queued", "closed"],
)
def test_a_pr_nothing_will_merge_exits_1_saying_why(tmp_path: Path, answer: dict[str, Any], said: str) -> None:
    ran = merge(tmp_path, pull(), pull(position=1), answer)
    assert ran.code == 1
    assert ran.said[-1] == f"merge-pr: #7 {said}"
    assert f"pr merge --disable-auto {PR}" not in ran.calls


def test_a_pr_gh_cannot_merge_fails_without_waiting(tmp_path: Path) -> None:
    ran = merge(tmp_path, pull(), merge_exit=1)
    assert ran.code == 2
    assert ran.calls == [f"api graphql number={PR}", f"pr merge {PR}"]


@pytest.mark.parametrize(
    ("first", "said"),
    [(pull(position=3), "QUEUED at position 3"), (pull(auto=True), "WAITING for its checks before it is queued")],
    ids=["queued", "set to merge"],
)
def test_a_pr_already_being_merged_is_waited_on_not_merged_again(tmp_path: Path, first: dict[str, Any], said: str) -> None:
    """So a wait that exits 2 when GitHub cannot be asked is taken up again by running the script again."""
    ran = merge(tmp_path, first, pull("MERGED"))
    assert ran.code == 0, ran.said
    assert ran.said == [f"merge-pr: #7 {said}", "merge-pr: #7 MERGED"]
    assert f"pr merge {PR}" not in ran.calls


def test_a_pr_that_already_merged_exits_0_without_merging(tmp_path: Path) -> None:
    ran = merge(tmp_path, pull("MERGED"))
    assert ran.code == 0, ran.said
    assert ran.calls == [f"api graphql number={PR}"]


def test_a_pr_the_queue_dropped_is_merged_again(tmp_path: Path) -> None:
    ran = merge(tmp_path, pull(removed=("2026-10-29T09:00:00Z", "FAILED_CHECKS")), pull(position=1), pull("MERGED"))
    assert ran.code == 0, ran.said
    assert f"pr merge {PR}" in ran.calls


def test_a_question_github_cannot_answer_fails(tmp_path: Path) -> None:
    ran = merge(tmp_path, pull(), pull(position=1))
    assert ran.code == 2
    assert ran.said[-1] == "gh: no answer for question 3"


def test_an_auto_merge_gh_cannot_disable_fails(tmp_path: Path) -> None:
    ran = merge(tmp_path, pull(), pull(auto=True, mergeable="CONFLICTING"), disable_exit=1)
    assert ran.code == 2
    assert ran.calls[-1] == f"pr merge --disable-auto {PR}"


@pytest.mark.parametrize(
    "args",
    [(), (PR, "0", "extra"), ("#7",), ("https://github.com/o/r/pull/7",), ("",), (PR, "30s")],
    ids=["no PR", "too many", "a PR with its #", "a PR's URL", "an empty PR", "an interval with its unit"],
)
def test_arguments_that_are_not_a_pr_number_and_interval_fail_before_asking_anything(tmp_path: Path, args: tuple[str, ...]) -> None:
    ran = merge(tmp_path, args=args)
    assert ran.code == 2
    assert "merge-pr: usage: merge-pr.sh PR [INTERVAL]" in ran.said[0]
    assert ran.calls == []
