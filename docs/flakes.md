# Flaky tests

A flaky test is quarantined, not retried. Retrying until green hides the failure the test was written to catch.

## Reproducing a flake

Drive the event that breaks it: a `SIGSTOP` at the moment that matters, a stalled stream, a second process started
mid-call. Run one case at a time. Running many copies at once loads the machine and does not reproduce reliably
(promptctl-testing-px1f.1rp). Keep what the thing under test said in the failure message; a cause was found once only
because a file happened to hold it.

## Quarantining

```python
@pytest.mark.quarantined("promptctl-<ticket id>", until="YYYY-MM-DD")
def test_the_flaky_one() -> None: ...
```

- The ticket is the one that will fix or delete the test; the date is the day its week ends, no more than
  `QUARANTINE_DAYS` (tests/conftest.py) from the day it is written. Both are required, and the run stops before any test
  if either is missing, malformed, or too far off.
- A quarantined test still runs and still reports. It is marked as pytest's own expected failure, so a failure in setup,
  call, or teardown leaves the run green. The summary lists it as XFAIL with the ticket and the date, and as XPASS when it
  passed, which is the sign its flake may be fixed and the marker can go. What it said as it failed is printed under
  "quarantined failure" in the summary, so a CI run keeps the text a flake's cause is found from.
- The day after `until`, the run stops before any test and names the quarantine: fix the test or delete it. Moving the
  date is an edit a reviewer sees in the diff, beside the ticket that says why. The date is the machine's, in UTC.
- The marker works on a function, a class, or a module's `pytestmark`; each test under it is quarantined, and every marker
  that applies to a test is checked, the nearer one not hiding an expired one above it.
- Do not retry a quarantined test to make it green.

Layers and where a new test belongs: `laws:testing`.
