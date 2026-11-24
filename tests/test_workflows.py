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


@pytest.mark.parametrize("path", [path for path in WORKFLOWS if read(path)["name"] in read(path)["jobs"]], ids=lambda path: path.name)
def test_the_check_a_workflow_is_named_for_waits_on_every_other_job_of_it(path: Path) -> None:
    workflow = read(path)
    verdict = workflow["name"]
    assert sorted(workflow["jobs"][verdict]["needs"]) == sorted(name for name in workflow["jobs"] if name != verdict)
