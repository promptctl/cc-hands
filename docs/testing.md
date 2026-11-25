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
not a layer: whether an outside test proves an adapter, a contract or the assembled system is not this mark's to say. Where
CI runs the suite (below) is cut by what a run costs, not by those layers.

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

## The brain's terminal, and its one smoke test

hands' decisions about what Claude Code does (a dialog held, a stop, a turn opened by a background task) are asserted in the
unit layer: a `Brain` built of fakes and `BrainStage` on a `VirtualClock`, Claude Code's turns played from tests/turns.py. What hands
relies on of Claude Code's own terminal is a contract, tests/terminalcontract.py, written once and run by two modules: against
the fake Claude Code (tests/claudescreen.py) in `make check` (test_terminal_contract.py), and against the pinned Claude Code in
a terminal of hands' own, with the mock of Anthropic's API as its model, in `make check-rest` (test_terminal_contract_pinned.py).
A claim the pin stops keeping fails the second, and the fake is changed to match. tests/test_pinned_brain.py is the brain's one
smoke test, the happy path through hands' proxy, and tests/test_session_contract_pinned.py pins the order of a /clear's and a
/resume's end, start and first request that the brain follows. The rig the three pinned modules share is tests/pinnedbrain.py.

## The phone's pages

tests/test_pages.py runs each phone page's own code in the installed Google Chrome, driven by Playwright, against
`phonepage.phone_pages`, the routes hands serves the phone, on the route tests' fakes. Chrome must be installed, as it is
on GitHub's macos-15 runner image, which CI's `rest` jobs run on: one that is missing fails the test. Every page test ends by failing on any uncaught error or `console.error`
its pages made, whether the test passed or not.

A page test that needs speech as the microphone is tests/test_phonewakepage.py's kind. The page's `getUserMedia` is replaced by an
init script that hands it a Web Audio destination, and the test plays a recording of speech into it (`tests/speech.py`'s
`spoken(text)`, 16 kHz, the same synthesis the wake word's other tests use) when it means the page to hear it, and silence
otherwise. What the page sends is written down where it leaves, on the call's data channel. The Mac's half is the real clip owner
and `LatencyObserver` fed from the phone's audio, so what the page is told of a turn is what hands says of it. A page test that
holds a fetch, or fails one, does it with Playwright's `route`, not a fake in the page.

## The commands

- `make check`: the unit layer (`make check-unit`), `pyright`, and wake-training's `pyright` (`make check-types`). Run it on every turn.
- `make check-pair`: the real pair's cases, the `outside` tests in tests/test_streaming.py, with no hold. They load the speech
  models.
- `make check-rest`: every other `outside` test with no hold (`make check-outside`), then fritter's Go tests (`make check-fritter`)
  and wake-training's suite (`make check-wake-training`).
- `SHARD=K/N` on `check-unit`, `check-pair` and `check-outside` runs the Kth of N shards of what the target selects, as `pytest
  --shard=K/N` does (tests/shards.py). Every shard of one commit takes the same partition, packed by tests/shard-costs.json, so
  the shards together run each test once. A module is kept whole in a shard, its fixtures set up once, unless it is more than a
  shard's share. The costs come from the cost reports CI prints (tests/costs.py): after a change that moves a lot of test time,
  `gh run view <run> --log` of the shards' jobs into a file each and `uv run --no-sync python tests/shards.py FILE... >
  tests/shard-costs.json` regenerates them, from a job cancelled at its limit too: a test its run stopped inside is marked in the
  report, is named when the file is regenerated, and costs what the run spent in it until a run finishes it. A test costs the
  most any of the logs says it did, so shards hold on a slow runner. A shard says at its start what it was packed at, and at its end what it cost and
  which tests cost 0.5 s or more and are not in the file, so a file gone stale shows in the log; a job cancelled at its limit
  still has the first. `make costs-check-unit` (and `-check-pair`, `-check-outside`) prints the cost report of that target's last
  run, which is what CI's jobs put in their logs.
- `make audit`: the `outside` tests under the same hold, each expected to be refused. One that is not does not need its
  mark; run it when a test moves down.

## Where CI runs the suite

Every job in `.github/workflows` has `timeout-minutes: 5`: a workflow's run is held to the 5 minutes a person waiting on it
will wait, and a job that goes past it is a shard too big for it. The suite is about 40 minutes of test time on a macOS runner,
with a Free plan's 5 of them at once for every PR, so it cannot all run in 5 minutes, and is cut by where it runs:

- `.github/workflows/test.yml`, on every PR and in the merge queue: `make check-unit` in shards (the `unit` jobs) and `make
  check-types` (`types`), and the check `test` passes when they all did. The unit layer is every test not marked `outside`.
- `.github/workflows/outside.yml`, on master every day, on request, and on a PR that changes how it is cut or run: the tests
  marked `outside`, in shards (`pair`: tests/test_streaming.py, `rest`: the others), fritter's Go tests and wake-training's
  suite, as jobs of their own, and the check `outside` passes when they all did. It is more jobs than the runners hold at once,
  so the run takes several waves of them; no job in it takes more than 5 minutes.

Together they are the whole suite. Green on `make check` is not green on `outside`, which runs what `make check` holds
nothing of: the real models, the pinned Claude Code, the Go tests. A shard's count is its matrix's list in the workflow; add a
shard to the list when a job nears its limit, and refresh tests/shard-costs.json when the tests have changed a lot (the
commands above).

## Placing a new test

Write it as a unit test: in process, on fakes, on a virtual clock. If `make check` says the unit layer refused it, ask
whether the behavior can be proven without the process, the socket or the weights; usually it can, behind a port you own.
When it cannot, because the test is about the process, the socket or the models, mark it `outside`.
