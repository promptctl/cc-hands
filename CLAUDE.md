## HARD RULES

- NO WORK PROCEEDS WITHOUT A LIT TICKET.  ALL TICKETS MUST BE IN AN EPIC
- NO WORK PROCEEDS WHEN THERE ARE FAILING TESTS.   Create a ticket and fix the tests first.
- A test that fails and then passes on a re-run is a flake: it is never re-run until green. Quarantine it as docs/flakes.md says.
- No work proceeds except on a fresh work tree.  the work tree is cleaned up when the ticket is finished.
- master merges through a merge queue: merge a PR with `scripts/merge-pr.sh <pr>`, and its ticket closes only on that command's exit 0.
- No test anywhere may require a human to log into Anthropic as part of an automated test suite.  Tests use appropriate mocks and APIs.
- HANDS NEVER STARTS A CLAUDE CODE PROCESS TO GET WHAT THE RUNNING BRAIN ALREADY GIVES IT.  hands is a realtime voice application, and every Claude Code started on the way to an answer is latency the user hears.  What the brain's Claude Code can do, hands gets without starting another: a request of its own through the proxy, completely consistent with the requests Claude Code itself sends (a shorthand for running the command inside Claude Code, the same in principle); a change to the brain's request that piggybacks on it; an API response the brain receives, written or rewritten by hands; a command sent to the brain's stdin; or some other way.  hands may write and rewrite the brain's requests and the API responses the brain receives.  A separate Claude Code started for each side question is this mistake.  One-shot `claude` CLI checks at startup are fine; repeated `claude` CLI commands for anything that can be done another way are not.  Effort spent on a design that breaks this is wasted, and someone has to clean it up later.

## Tests

`make check` runs the unit layer, the tests that pass with everything outside the process refused by the OS, plus the type checks: it is the run to make on every turn, and it takes about a minute. Green on it is not green on CI, which runs it as the `unit` tier beside `pair` (`make check-pair`) and `rest` (`make check-rest`: the other tests that need the outside, fritter's Go tests and wake-training's suite among them). A test that needs a process, a socket or the speech models is marked `@pytest.mark.outside`; docs/testing.md has the layers and where a new test belongs.

<!-- BEGIN LIT INTEGRATION -->
## lit Agent-Native Workflow

This repository uses `lit` for agent-native issue tracking.

Start by running `lit quickstart` to load the workflow instructions. It prints how tickets are found, created, updated, and closed here, so running it first means the rest of your work follows the conventions this repo expects. It's a quick, read-only command — no need to check in before running it.

As you work a ticket, record its phase at each move: `lit phase <id> --to impl` when you start writing code, `lit phase <id> --to review` when its PR opens, `lit phase <id> --to merge` when review is done and the merge is under way. `lit start` puts it in `plan`, and `lit done <id>` ends it.

<!-- END LIT INTEGRATION -->
