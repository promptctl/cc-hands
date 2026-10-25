## HARD RULES

- NO WORK PROCEEDS WITHOUT A LIT TICKET.  ALL TICKETS MUST BE IN AN EPIC
- NO WORK PROCEEDS WHEN THERE ARE FAILING TESTS.   Create a ticket and fix the tests first.
- No work proceeds except on a fresh work tree.  the work tree is cleaned up when the ticket is finished.
- HANDS NEVER STARTS A CLAUDE CODE PROCESS OF ITS OWN, OR ANYTHING LIKE ONE.  The brain is the one Claude Code hands runs.  Brandon: "HANDS CANNOT START ADDITIONAL CLAUDE CODE PROCESSES AT ANY TIME."  "Use subagents or /btw."  "Hands can rewrite requests: approved."  hands is a realtime voice application, and a proxy already sits between Claude Code and Anthropic: whatever hands needs from the model goes through the brain (a subagent, /btw) or through requests at that proxy, never through a process started for it.  The per-question Claude Code in `src/hands/brain/asides.py` is the mistake this rule exists for (its removal: promptctl-brain-lq3.3ck); do not build on it, copy it, cite it as a pattern, or propose a design shaped like it (a second responder, a helper session, a model process per task).  Effort spent on that is not just wasted: someone has to clean it up later.

<!-- BEGIN LIT INTEGRATION -->
## lit Agent-Native Workflow

This repository uses `lit` for agent-native issue tracking.

Start by running `lit quickstart` to load the workflow instructions. It prints how tickets are found, created, updated, and closed here, so running it first means the rest of your work follows the conventions this repo expects. It's a quick, read-only command — no need to check in before running it.

<!-- END LIT INTEGRATION -->
