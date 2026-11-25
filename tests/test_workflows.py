"""What .github/workflows promises about every run: each job ends inside the 5 minutes a person waiting on it will wait, and
the check a workflow is named for says whether all of its jobs passed."""

from pathlib import Path
from typing import Any

import pytest
import yaml

WORKFLOWS = sorted((Path(__file__).parents[1] / ".github" / "workflows").glob("*.yml"))
# GitHub's own limit on a job without `timeout-minutes`, which is what a job that goes without one is held to.
UNBOUNDED = 360


def read(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda path: path.name)
def test_every_job_is_held_to_five_minutes(path: Path) -> None:
    jobs: dict[str, dict[str, Any]] = read(path)["jobs"]
    assert [name for name, job in jobs.items() if job.get("timeout-minutes", UNBOUNDED) > 5] == []


def shard_lists(path: Path) -> dict[str, list[int]]:
    """The shard list of each job of a workflow that has one."""
    return {name: job["strategy"]["matrix"]["shard"] for name, job in read(path)["jobs"].items() if "shard" in job.get("strategy", {}).get("matrix", {})}


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda path: path.name)
def test_a_shard_list_is_every_number_from_one_to_its_length_once(path: Path) -> None:
    # A job takes its K/N from the list's position and length (`strategy.job-total`): a number twice leaves another unrun.
    assert {name: shards for name, shards in shard_lists(path).items() if shards != list(range(1, len(shards) + 1))} == {}


def test_a_prs_test_run_has_no_more_macos_jobs_than_a_free_plan_runs_at_once() -> None:
    # Five at once, shared by every PR: a sixth waits for a first to end, and the run is two waves long.
    jobs: dict[str, dict[str, Any]] = read(Path(__file__).parents[1] / ".github" / "workflows" / "test.yml")["jobs"]
    macos = [len(job.get("strategy", {}).get("matrix", {}).get("shard", [0])) for job in jobs.values() if job["runs-on"] == "macos-15"]
    assert sum(macos) <= 5


@pytest.mark.parametrize("path", [path for path in WORKFLOWS if read(path)["name"] in read(path)["jobs"]], ids=lambda path: path.name)
def test_the_check_a_workflow_is_named_for_waits_on_every_other_job_of_it(path: Path) -> None:
    workflow = read(path)
    verdict = workflow["name"]
    assert sorted(workflow["jobs"][verdict]["needs"]) == sorted(name for name in workflow["jobs"] if name != verdict)
