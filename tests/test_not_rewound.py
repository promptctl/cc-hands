"""scripts/not-rewound.sh refuses a commit built on the line master was rewound away from on 2026-10-15, read from
this repository's own history: the line's first commit ab3ad39 (#270), and 50557d9, where master was put back."""

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "not-rewound.sh"
FIRST = "ab3ad395640441644c48b37dfea8884330f1ede7"
REWOUND_TO = "50557d9141a68a871e44917be9d4b414304bd932"


def git(cwd: Path, env: dict[str, str], *args: str) -> str:
    return subprocess.run(("git", *args), cwd=cwd, env=env, capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture(scope="session")
def scratch(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    """An environment in which git reads this repository's objects and writes new ones to a scratch directory, so a
    test commits on top of the real line without adding a thing to the repository, and in which git reads no
    configuration but the repository's, so the developer's signing or hooks never touch a test's commits."""
    objects = subprocess.run(
        ("git", "rev-parse", "--path-format=absolute", "--git-path", "objects"), cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    env = os.environ | {
        "GIT_OBJECT_DIRECTORY": str(tmp_path_factory.mktemp("objects")),
        "GIT_ALTERNATE_OBJECT_DIRECTORIES": objects,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "t@example.com",
    }
    # No branch need hold the line once its branches are deleted, so a whole clone may lack its first commit; it is
    # fetched as the script would fetch it, into the scratch directory, and with no FETCH_HEAD naming an object the
    # repository does not hold once the scratch directory is gone.
    if subprocess.run(("git", "cat-file", "-e", f"{FIRST}^{{commit}}"), cwd=ROOT, env=env, capture_output=True).returncode != 0:
        git(ROOT, env, "fetch", "--quiet", "--no-tags", "--no-write-fetch-head", "origin", FIRST)
    return env


def on_top(env: dict[str, str], *parents: str) -> str:
    """A new commit whose parents are `parents`, holding the first parent's tree."""
    return git(ROOT, env, "commit-tree", f"{parents[0]}^{{tree}}", *(arg for parent in parents for arg in ("-p", f"{parent}^{{commit}}")), "-m", "work")


def checked(env: dict[str, str], commit: str, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run((str(SCRIPT), commit), cwd=cwd, env=env, capture_output=True, text=True)


def test_the_rewound_lines_first_commit_is_refused(scratch: dict[str, str]) -> None:
    result = checked(scratch, FIRST)
    assert result.returncode == 1
    assert f"{FIRST} ({FIRST}) is built on the line master was rewound away from on 2026-10-15" in result.stderr
    assert "the work must be rebuilt on master" in result.stderr


def test_new_work_on_top_of_the_rewound_line_is_refused(scratch: dict[str, str]) -> None:
    work = on_top(scratch, on_top(scratch, FIRST))
    result = checked(scratch, work)
    assert result.returncode == 1
    assert f"{work} ({work}) is built on the line master was rewound away from on 2026-10-15" in result.stderr


def test_the_merge_ci_tests_of_a_pr_built_on_the_rewound_line_is_refused(scratch: dict[str, str]) -> None:
    """A pull request's check runs on master merged with the PR's head; the line comes in through the second parent."""
    merge = on_top(scratch, "v0.2.0", on_top(scratch, FIRST))
    assert checked(scratch, merge).returncode == 1


@pytest.mark.parametrize("commit", [REWOUND_TO, "v0.2.0"], ids=["where master was put back", "a release"])
def test_a_commit_off_the_rewound_line_is_accepted(scratch: dict[str, str], commit: str) -> None:
    result = checked(scratch, commit)
    assert result.returncode == 0, result.stderr
    assert "is not built on the line master was rewound away from on 2026-10-15" in result.stdout


def test_new_work_on_master_is_accepted(scratch: dict[str, str]) -> None:
    work = on_top(scratch, "v0.2.0")
    assert checked(scratch, work).returncode == 0


def test_a_shallow_clone_cannot_tell_and_fails(tmp_path: Path) -> None:
    clone = tmp_path / "shallow"
    git(tmp_path, os.environ.copy(), "clone", "--quiet", "--depth", "1", "--no-checkout", ROOT.as_uri(), str(clone))
    result = checked(os.environ.copy(), "HEAD", cwd=clone)
    assert result.returncode == 2
    assert "this clone is shallow" in result.stderr


def test_a_clone_without_the_lines_first_commit_fails_rather_than_accepting(scratch: dict[str, str], tmp_path: Path) -> None:
    """With no origin to fetch ab3ad39 from, the ancestry test cannot run, and the check fails instead of passing."""
    repo = tmp_path / "elsewhere"
    env = {key: value for key, value in scratch.items() if key not in ("GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES")}
    git(tmp_path, env, "init", "--quiet", str(repo))
    git(repo, env, "commit", "--quiet", "--allow-empty", "-m", "first")
    result = checked(env, "HEAD", cwd=repo)
    assert result.returncode == 2
    assert f"{FIRST}, the rewound line's first commit, is not in this clone and could not be fetched from origin" in result.stderr
