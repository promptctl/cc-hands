# AGENTS

## HARD RULES

- NO WORK PROCEEDS WITHOUT A LIT TICKET.  ALL TICKETS MUST BE IN AN EPIC
- NO WORK PROCEEDS WHEN THERE ARE FAILING TESTS.   Create a ticket and fix the tests first.
- No work proceeds except on a fresh work tree.  the work tree is cleaned up when the ticket is finished.
- HANDS NEVER STARTS A CLAUDE CODE PROCESS TO GET WHAT THE RUNNING BRAIN ALREADY GIVES IT.  hands is a realtime voice application, and every Claude Code started on the way to an answer is latency the user hears.  What the brain's Claude Code can do, hands gets from the brain: a direct API call, a change to the brain's request that piggybacks on it, a command sent to its stdin, or some other way.  A separate Claude Code started for each side question was this mistake.  Allowed: hands rewriting the brain's requests; hands sending its own requests through the proxy, as long as each is completely consistent with the requests Claude Code itself sends (a shorthand for running the command inside Claude Code, the same in principle); a one-shot `claude` CLI check at startup.  Not allowed: repeated `claude` CLI commands for anything that can be done another way.  Effort spent on a design that breaks this is wasted, and someone has to clean it up later.

<!-- BEGIN LIT INTEGRATION -->
## lit Agent-Native Workflow

This repository uses `lit` for agent-native issue tracking.

Start by running `lit quickstart` to load the workflow instructions. It prints how tickets are found, created, updated, and closed here, so running it first means the rest of your work follows the conventions this repo expects. It's a quick, read-only command — no need to check in before running it.

<!-- END LIT INTEGRATION -->
