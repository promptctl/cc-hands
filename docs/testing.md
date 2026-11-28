# Testing

Where a test belongs, and why, is `laws:testing`: the layers, and the seams that let a test sit at the lowest one. This is
how they fall here.

## The unit layer

A test is in the unit layer when it passes with everything outside the process refused by the OS. `pytest --layer=unit`
runs every test not marked `@pytest.mark.outside` inside a macOS sandbox (tests/offline.py, tests/unit.sb) that refuses a
process start, a socket bound or connected (loopback and local ones too), a read of any other process, and a read of the
speech models' weights the run fetched. A test that depends on one fails with the OS's `PermissionError`, naming the path or the address, and says
what to do: mark it `outside`, or take the reach out. A fixture that does it fails every test that uses it.

What code under test makes of a refusal is its own. `PermissionError` is an `OSError`, so a function that answers "absent"
or "unknown" on one answers it here, and a test whose verdict holds on that answer does not need the outside: it is a unit
test. The hold is what makes an unmarked test unit, and it is checked on every `make check`.

Everything else is marked `outside` (a module's `pytestmark` or a test's decorator). The mark names what the test needs,
not a layer: whether an outside test proves an adapter, a contract or the assembled system is not this mark's to say. CI's
workflows (below) are cut by what a run costs, not by those layers.

A wait the OS cannot refuse, a `time.sleep` or a timer the loop runs out, stays in the unit layer. `tests/clockwatch.py`
owns it: its `UNMOVED` lists the modules that still reach the machine's time. Those in `TO_MOVE` come off it by moving
onto a `VirtualClock`. Those in `PERMANENT` hold hands to a real party that runs on the machine's time, each with its
reason, and mark all their tests `outside`.

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

## The Mac's processes

The daemon and its sessions read and signal other processes through `Processes` (hands.sessions.processes), whose one
real adapter, hands.sessions.kernel's `KERNEL`, only a composition root imports (tests/test_processes_boundary.py); the
brain's reads of its own pane's processes are the exception left, each marked, until promptctl-testing-c3u.fhu. A test of
what hands makes of the process table gives its code a fake table, tests/fakeprocesses.py, whose rows hold the facts the
kernel answers from: which file each process has open as its stdin and stdout, and which call the kernel refuses about it.
The `hands check` diagnostic uses this host's terminal processes to find Claude runs that may have missed registration.
Registered sessions are identified by the session ID reported by their hooks. In `tests/test_process_discovery.py` the
process host supplies the executable for a relative invocation, and a process can exit between listing and inspection.
Session readiness, pane routing, liveness and close-session tests cover how those callers use process information.

## The user's terminals and the Mac's screen

The daemon reads and types into the user's tmux panes through `Terminals`, types into fritters through `Fritters` (both
hands.sessions.terminals), and reads which app is in front through `Screen` (hands.sessions.screen). Their real
adapters, hands.sessions.tmux's `TmuxTerminals`, hands.sessions.typing's `FritterSockets` and
hands.sessions.launchservices' `LaunchServices`, only a composition root imports (tests/test_terminals_boundary.py); the
brain's tmux and fritter on its own pane are the exception left, until promptctl-testing-c3u.fhu.
Which servers may hold a session's pane, and which pane it is, is hands' own logic over Terminals and Processes
(hands.sessions.panes), tested on tests/faketerminals.py: tmux servers, their panes and clients as values, and what a
session's program read of what was typed into it. What hands relies on of tmux and fritter is a contract,
tests/terminalscontract.py, run against the fake in `make check` (test_terminals_contract.py) and against a tmux server of
the test's own and the fritter this checkout carries, `outside` (test_terminals_contract_tmux.py).

## The phone's pages

tests/test_pages.py runs each phone page's own code in the installed Google Chrome, driven by Playwright, against
`phonepage.phone_pages`, the routes hands serves the phone, on the route tests' fakes. Chrome must be installed, as it is
on GitHub's macos-15 runner image, which outside.yml's `rest` job runs on: one that is missing fails the test. Every page test ends by failing on any uncaught error or `console.error`
its pages made, whether the test passed or not.

A page test that needs speech as the microphone is tests/test_phonewakepage.py's kind. The page's `getUserMedia` is replaced by an
init script that hands it a Web Audio destination, and the test plays a recording of speech into it (`tests/speech.py`'s
`spoken(text)`, 16 kHz, the same synthesis the wake word's other tests use) when it means the page to hear it, and silence
otherwise. What the page sends is written down where it leaves, on the call's data channel. The Mac's half is the real clip owner
and `LatencyObserver` fed from the phone's audio, so what the page is told of a turn is what hands says of it. A page test that
holds a fetch, or fails one, does it with Playwright's `route`, not a fake in the page.

## The commands

- `make check`: the unit layer (`make check-unit`) and the type checks, `pyright` and wake-training's (`make check-types`). Run it on every turn.
- `make check-pair`: the real pair's cases, the `outside` tests in tests/test_streaming.py, with no hold. They load the speech
  models.
- `make check-rest`: every other `outside` test with no hold, then fritter's Go tests and wake-training's suite.
- `make audit`: the `outside` tests under the same hold, each expected to be refused. One that is not does not need its
  mark; run it when a test moves down.

## Where CI runs the suite

A job's 5 minutes hold about 100 s of tests: a macOS job spends 95 s before its first test, on the runner, the tools, the
locked environment and pytest's collection, and a job that loads the speech models 50 to 100 s more restoring their cache. A
Free plan runs 5 macOS jobs at once, shared by every PR, the merge queue and master. So the suite is on two paths.

`.github/workflows/test.yml`, every PR and the merge queue, ends inside 5 minutes: `make check-unit` as the shards of its
`unit` matrix, and `make check-types`, which tests/test_workflows.py holds to the macOS runners one wave of a Free plan has.
Each shard is `pytest --shard=K/N` (tests/shards.py): the tests the Kth of N holds, packed to equal seconds from what each test cost on CI (tests/shard-costs.json), the same cut in every job. The
cost file is made from the jobs' logs, which print each test's cost (tests/costs.py); tests/shards.py says how. A stale file
loses no test: it unbalances the shards, and each prints what it was packed at beside what it cost, and the slow tests and
setups the file does not name. Each also records a digest of the test ids it collected, and the check `test` passes when all
of the jobs did and the shards' digests are one: a test is in exactly one shard when they collected the same.

`.github/workflows/outside.yml`, daily and on request, runs what `test` leaves out: `make check-pair` and `make check-rest`,
the tests marked `outside`, fritter's Go tests and wake-training's suite. They are about 1,800 s of tests and each job spends
150 s before its first, so held to 5 minutes they would be 60 jobs and more against the 5 runners; they are not on the path a
merge waits on. The check `outside` passes when both did.

## Placing a new test

Write it as a unit test: in process, on fakes, on a virtual clock. If `make check` says the unit layer refused it, ask
whether the behavior can be proven without the process, the socket or the weights; usually it can, behind a port you own.
When it cannot, because the test is about the process, the socket or the models, mark it `outside`.
