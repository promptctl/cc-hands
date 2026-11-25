"""What the workflows say of themselves in their headers, held by the files: a PR's run is jobs of 5 minutes at most, that fit the
Free plan's 5 macOS runners at once, whose shards are every number of their list, and whose verdict waits for all of them."""

from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
# A Free plan's concurrent macOS jobs, which every PR, the merge queue and master's pushes share.
MACOS_RUNNERS = 5


def jobs_of(name: str) -> dict[str, dict[str, Any]]:
    return yaml.safe_load((WORKFLOWS / name).read_text())["jobs"]


def macos(job: dict[str, Any]) -> bool:
    return str(job["runs-on"]).startswith("macos")


def legs(job: dict[str, Any]) -> int:
    """How many jobs a job is: the product of the lists in its matrix."""
    count = 1
    for values in job.get("strategy", {}).get("matrix", {}).values():
        count *= len(values)
    return count


@pytest.mark.parametrize("workflow, job", [(file, job) for file in ("test.yml", "release.yml") for job in jobs_of(file)])
def test_every_job_a_prs_or_a_releases_run_has_is_bounded_at_5_minutes(workflow: str, job: str) -> None:
    assert jobs_of(workflow)[job]["timeout-minutes"] == 5


def test_a_prs_test_run_has_no_more_macos_jobs_than_a_free_plan_runs_at_once() -> None:
    assert sum(legs(job) for job in jobs_of("test.yml").values() if macos(job)) <= MACOS_RUNNERS


@pytest.mark.parametrize("workflow", ["test.yml"])
def test_a_shard_list_is_every_number_from_one_to_its_length_once(workflow: str) -> None:
    sharded = [job for job in jobs_of(workflow).values() if "shard" in job.get("strategy", {}).get("matrix", {})]
    assert sharded
    for job in sharded:
        shards = job["strategy"]["matrix"]["shard"]
        assert shards == list(range(1, len(shards) + 1))


@pytest.mark.parametrize("workflow, verdict", [("test.yml", "test"), ("outside.yml", "outside")])
def test_the_verdict_job_waits_for_every_other_job(workflow: str, verdict: str) -> None:
    jobs = jobs_of(workflow)
    needs = jobs[verdict]["needs"]
    assert ({needs} if isinstance(needs, str) else set(needs)) == set(jobs) - {verdict}
    assert jobs[verdict]["if"] == "${{ !cancelled() }}"


def test_each_unit_job_runs_the_shard_its_matrix_names_of_as_many_as_the_list_holds() -> None:
    steps = jobs_of("test.yml")["unit"]["steps"]
    assert "make check-unit SHARD=${{ matrix.shard }}/${{ strategy.job-total }}" in [step.get("run") for step in steps]
