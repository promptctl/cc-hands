# Testing

Where a test belongs, and why, is `laws:testing`: the layers, and the seams that let a test sit at the lowest one. This is
how they fall here.

## The unit layer

A test is in the unit layer when it passes with everything outside the process refused by the OS. `pytest --layer=unit`
runs every test not marked `@pytest.mark.outside` inside a macOS sandbox (tests/offline.py, tests/unit.sb) that refuses a
process start, a socket bound or connected (loopback and local ones too), and a read of the speech models' weights the
run fetched. A test that depends on one fails with the OS's `PermissionError`, naming the path or the address, and says
what to do: mark it `outside`, or take the reach out. A fixture that does it fails every test that uses it.

What code under test makes of a refusal is its own. `PermissionError` is an `OSError`, so a function that answers "absent"
or "unknown" on one answers it here, and a test whose verdict holds on that answer does not need the outside: it is a unit
test. The hold is what makes an unmarked test unit, and it is checked on every `make check`.

Everything else is marked `outside` (a module's `pytestmark` or a test's decorator). The mark names what the test needs,
not a layer: whether an outside test proves an adapter, a contract or the assembled system is not this mark's to say. CI's
tiers (below) are cut by what a run costs, not by those layers.

A wait the OS cannot refuse, a `time.sleep` or a timer the loop runs out, stays in the unit layer. `tests/clockwatch.py`
owns it: its `UNMOVED` lists the modules that still reach the machine's time, and a module comes off it by moving onto a
`VirtualClock`.

## The real pair, and its recordings

`Stream` reads the streaming pair only through `streaming.Session`, so its logic runs in the unit layer on a recording of
what the pair decoded (tests/recordings.py, tests/fixtures/decodes). Each such test takes the `session` fixture and runs
twice with the same assertions: `recorded`, in `make check`, and `real`, the pair, `outside`. The two-people case adds
to `real` the check of when each word starts, which a recording does not keep. The `real` case is what holds a recording
to the pair: one the pair no longer decodes as fails there, and one fed other audio (checked at each close) or in other
feeds than it was made from fails `recorded`, naming it. A recording is made one way: `pytest tests/test_streaming.py -k
real --record-decodes` writes the recording of each run a passing `real` case made. Run it after a pin bump or an edit to the
pair's code, and commit what it changes; the diff is what the pair now decodes differently.

## The commands

- `make check`: the unit layer, `pyright`, and wake-training's `pyright`. Run it on every turn.
- `make check-pair`: the real pair's cases, the `outside` tests in tests/test_streaming.py, with no hold. They load the speech
  models.
- `make check-rest`: every other `outside` test with no hold, then fritter's Go tests and wake-training's suite.
- `make audit`: the `outside` tests under the same hold, each expected to be refused. One that is not does not need its
  mark; run it when a test moves down.

## CI's tiers

`.github/workflows/test.yml` runs `make check`, `make check-pair` and `make check-rest` as the matrix `unit`, `pair` and `rest`, in
parallel, and the check `test` passes when all three did. Together they are the whole suite: `check` takes the tests not marked
`outside`, and the other two the ones that are, split by module. The tiers are cut by what a run costs, not by laws:testing's
layers: the pair has one of its own because it is proven in full every run and is the longest. Green on `make check` is not green
on CI, which also runs the `outside` tests, fritter's Go tests and wake-training's suite.

## Placing a new test

Write it as a unit test: in process, on fakes, on a virtual clock. If `make check` says the unit layer refused it, ask
whether the behavior can be proven without the process, the socket or the weights; usually it can, behind a port you own.
When it cannot, because the test is about the process, the socket or the models, mark it `outside`.
