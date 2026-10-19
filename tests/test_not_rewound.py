"""scripts/not-rewound.sh refuses a commit built on the line master was rewound away from on 2026-10-15. Each test
builds its own history in a scratch repository, with a first commit of its own standing in for the line's, and hands
that commit to the script; nothing reads this repository's history or the network."""

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "not-rewound.sh"
# The real line's first commit, ab3ad39, as git stores it: parent 50557d9, subject (#270). Its hash is the script's
# default, so the default is pinned without reading this repository's history or the network.
REAL_FIRST = Path(__file__).resolve().parent / "fixtures" / "rewound-line-first-commit"
ON_THE_LINE = "is built on the line master was rewound away from on 2026-10-15"
OFF_THE_LINE = "is not built on the line master was rewound away from on 2026-10-15"
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"


def git_env() -> dict[str, str]:
    """git with no config but the repository's own, so a signing or hook setting on the machine cannot reach a scratch
    commit, and no GIT_ variable from outside can point it at another repository."""
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    return env | {
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "t@example.com",
    }


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(("git", *args), cwd=cwd, env=git_env(), capture_output=True, text=True, check=True).stdout.strip()


def commit(repo: Path, message: str, *parents: str) -> str:
    """A new commit of the empty tree on `parents`; the message tells apart commits that share their parents."""
    return git(repo, "commit-tree", EMPTY_TREE, *(arg for parent in parents for arg in ("-p", parent)), "-m", message)


@dataclass(frozen=True)
class History:
    """put_back, where master was put back; first, the rewound line's first commit, on put_back; master, master's next
    commit after the rewind, also on put_back; rewound, work two commits on top of first. Branch `master` holds master
    and branch `rewound` holds rewound."""

    repo: Path
    put_back: str
    first: str
    master: str
    rewound: str


@pytest.fixture
def history(tmp_path: Path) -> History:
    repo = tmp_path / "repo"
    git(tmp_path, "init", "--quiet", "--initial-branch", "master", str(repo))
    put_back = commit(repo, "put back")
    first = commit(repo, "first", put_back)
    master = commit(repo, "master", put_back)
    rewound = commit(repo, "rewound", commit(repo, "on first", first))
    git(repo, "update-ref", "refs/heads/master", master)
    git(repo, "update-ref", "refs/heads/rewound", rewound)
    return History(repo, put_back, first, master, rewound)


def checked(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run((str(SCRIPT), *args), cwd=cwd, env=git_env(), capture_output=True, text=True)


def clone(history: History, branch: str, dest: Path, *options: str) -> Path:
    """A clone of `branch` alone, as CI makes one, over a transport, so it holds only what that branch reaches."""
    git(dest.parent, "clone", "--quiet", "--single-branch", "--branch", branch, *options, history.repo.as_uri(), str(dest))
    return dest


def test_the_rewound_lines_first_commit_is_refused(history: History) -> None:
    result = checked(history.repo, history.first, history.first)
    assert result.returncode == 1
    assert f"{history.first} ({history.first}) {ON_THE_LINE}, which begins at {history.first}" in result.stderr
    assert "rebuilt on master from its own commits alone" in result.stderr


def test_new_work_on_top_of_the_rewound_line_is_refused(history: History) -> None:
    result = checked(history.repo, history.rewound, history.first)
    assert result.returncode == 1
    assert f"{history.rewound} ({history.rewound}) {ON_THE_LINE}" in result.stderr


def test_the_merge_ci_tests_of_a_pr_built_on_the_rewound_line_is_refused(history: History) -> None:
    """A pull request's check runs on master merged with the PR's head; the line comes in through the second parent."""
    merge = commit(history.repo, "merge", history.master, history.rewound)
    assert checked(history.repo, merge, history.first).returncode == 1


@pytest.mark.parametrize("name", ["put_back", "master"], ids=["where master was put back", "master after the rewind"])
def test_a_commit_off_the_rewound_line_is_accepted(history: History, name: str) -> None:
    off = getattr(history, name)
    result = checked(history.repo, off, history.first)
    assert result.returncode == 0, result.stderr
    assert f"{off} ({off}) {OFF_THE_LINE}" in result.stdout


def test_new_work_on_master_is_accepted(history: History) -> None:
    assert checked(history.repo, commit(history.repo, "work", history.master), history.first).returncode == 0


def test_a_whole_clone_without_the_lines_first_commit_accepts(history: History, tmp_path: Path) -> None:
    """Once no ref holds the line, a clone of master lacks its first commit, and that absence proves master is off it."""
    repo = clone(history, "master", tmp_path / "master")
    result = checked(repo, "HEAD", history.first)
    assert result.returncode == 0, result.stderr
    assert f"{OFF_THE_LINE}; this whole clone does not hold its first commit {history.first}" in result.stdout


def test_a_whole_clone_of_work_on_the_line_brings_the_first_commit_and_refuses(history: History, tmp_path: Path) -> None:
    """The same clone of a branch built on the line holds its first commit, since it is that branch's ancestor."""
    repo = clone(history, "rewound", tmp_path / "rewound")
    assert checked(repo, "HEAD", history.first).returncode == 1


def test_the_default_is_the_real_lines_first_commit_and_work_on_it_is_refused(history: History) -> None:
    """The real first commit's object goes into the scratch repository with its parent grafted away, since 50557d9 is
    not there; a wrong default would be absent, and the script would accept."""
    first = git(history.repo, "hash-object", "-t", "commit", "-w", str(REAL_FIRST))
    git(history.repo, "replace", "--graft", first)
    work = commit(history.repo, "work", first)
    result = checked(history.repo, work)
    assert result.returncode == 1, result.stdout
    assert f"{work} ({work}) {ON_THE_LINE}, which begins at {first}" in result.stderr


def test_the_real_first_commit_is_the_default_and_its_absence_accepts(history: History) -> None:
    """With no FIRST given, the script asks about the real line's first commit, which this scratch repository lacks."""
    result = checked(history.repo, "master")
    assert result.returncode == 0, result.stderr
    assert "this whole clone does not hold its first commit" in result.stdout


def test_a_shallow_clone_cannot_tell_and_fails(history: History, tmp_path: Path) -> None:
    """Depth 1 of a branch on the line lacks the first commit too, so absence proves nothing there and the check fails."""
    repo = clone(history, "rewound", tmp_path / "shallow", "--depth", "1")
    result = checked(repo, "HEAD", history.first)
    assert result.returncode == 2
    assert "this clone is shallow" in result.stderr


@pytest.mark.parametrize("first", ["not-an-object", ""], ids=["not an object", "empty"])
def test_a_first_commit_git_cannot_look_up_fails_rather_than_accepting(history: History, first: str) -> None:
    result = checked(history.repo, "master", first)
    assert result.returncode == 2
    assert f"git could not tell whether this clone holds {first}" in result.stderr


def test_a_commit_this_clone_does_not_hold_fails(history: History) -> None:
    result = checked(history.repo, "no-such-ref", history.first)
    assert result.returncode == 2
    assert "no-such-ref is not a commit in this clone" in result.stderr


@pytest.mark.parametrize("args", [(), ("a", "b", "c")], ids=["no commit", "too many"])
def test_a_wrong_number_of_arguments_fails(history: History, args: tuple[str, ...]) -> None:
    result = checked(history.repo, *args)
    assert result.returncode == 2
    assert "usage: not-rewound.sh COMMIT [FIRST]" in result.stderr
