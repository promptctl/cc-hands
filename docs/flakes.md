# Flaky tests

A flaky test is quarantined, not retried. Retrying until green hides the failure the test was written to catch.

## Reproducing a flake

Drive the event that breaks it: a `SIGSTOP` at the moment that matters, a stalled stream, a second process started
mid-call. Run one case at a time. Running many copies at once loads the machine and does not reproduce reliably
(promptctl-testing-px1f.1rp).

## Quarantining

```python
@pytest.mark.quarantined("promptctl-<ticket id>")
def test_the_flaky_one() -> None: ...
```

- The marker names the ticket that will fix or delete the test. It is required.
- A quarantined test still runs and still reports. A failure in setup, call, or teardown is recorded as an expected
  failure, with the ticket as its reason and the original traceback kept, so the run stays green.
- The quarantine's start is the commit that first added the marker. `tests/quarantine_gate.py` fails CI once a
  quarantine is older than `QUARANTINE_DAYS`, which is the one place that number is set.
- Do not retry a quarantined test to make it green. Fix it, or delete it.

Layers and where a new test belongs: `laws:testing`.
