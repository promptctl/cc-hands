# Claude Code behavior hands depends on

Every fact here was seen in a Claude Code release, named in its line as `(measured on X.Y.Z)`. This is the one place in the repo that names those releases. Code comments describe the behavior and link here instead, so a new release changes this file and nothing else.

When a fact is re-measured on a newer release, add its release here rather than editing a code comment. A fact whose release is a range (`from X to Y`) was seen across those releases. Each line keeps the area it came from in bold.

## Status and busy/idle

- **status:** Claude Code keeps ~/.claude/sessions/<pid>.json for every interactive session and rewrites its status as the session moves (measured on 2.1.280).
- **status:** Claude Code keeps ~/.claude/sessions/<pid>.json for every interactive session and rewrites its status as the session moves (measured on 2.1.289).
- **status:** the waitingFor reasons Claude Code writes are "permission prompt" and "input needed" (measured on 2.1.282).
- **status:** a background subagent counts as running from its launch until the turn that reports it back ends, though every turn between has stopped (measured on 2.1.289).
- **status:** Claude Code writes shell instead of idle while any of its background bash tasks has not ended (measured on 2.1.289).
- **status:** Claude Code sets idle for about 3 ms between a turn and the message queued behind it (measured on 2.1.282).
- **status:** an Escape and the next prompt set idle and busy again 93 ms apart (measured on 2.1.283).
- **statusfile:** Claude Code keeps a session's status at sessions/<pid>.json in its config directory, which holds the transcript at projects/<project>/<session>.jsonl (measured on 2.1.282)
- **statusfile:** Claude Code writes the status file as the session starts and sets a status in it a moment later (measured on 2.1.299)
- **statusfile:** Claude Code set a status 91 ms after writing the file; an earlier release took 793 ms, a figure whose release was not recorded (measured on 2.1.299)
- **statusfile:** Claude Code runs the end hook 330 to 640 ms after it removes the status file as its process exits (measured on 2.1.283)
- **queued message status:** a message queued behind a turn has no prompt of its own, so the earlier turn's Stop marks it, and its hook holds Claude Code (measured on 2.1.282)
- **background subagent busy state:** after a turn that starts a subagent in the background, Claude Code is busy from that turn on (measured on 2.1.289)
- **cleared session with background shell:** a session cleared while a background poll it started still runs is reported as running a shell, though it sits at its prompt doing nothing (measured on 2.1.289)
- **background subagent busy state until report:** Claude Code stays busy from a background subagent's launch until the turn reporting it back ends, after the launching turn has stopped (measured on 2.1.289)
- **status file:** Claude Code's status of a session is in the bytes of the files it writes, and is heard each time it is set (measured on 2.1.282)
- **status file at start:** a status file as first written at session start has no status set (measured on 2.1.299)
- **busy at prompt with background shell:** at its prompt with a background shell command running, Claude Code runs no turn (measured on 2.1.289)
- **busy before prompt hooks:** Claude Code sets busy before a prompt's hooks run, so the status can be read before the hook lands (measured on 2.1.282)
- **busy between blocked stop and next stop:** the status file says busy between a Stop blocked by a Stop hook and the Stop after it [ref: hands-status-bpp.55k] (measured on 2.1.282)
- **escape mid-tool idle timing:** Claude Code sets idle about 100 ms before it writes the interrupt record (measured on 2.1.282)
- **flush sets busy:** the flush sets busy again, not idle, and Claude answers under the flushed id (measured on 2.1.282)
- **idle between turns:** Claude Code sets idle for about 3 ms between a turn and the one queued behind it (measured on 2.1.282)
- **interrupt records:** an interrupt fires no hook; Claude Code writes interrupt records instead (measured on 2.1.281)
- **status:** Claude Code's busy status also covers a subagent working in the background, so busy with no turn open and a subagent out means delegating (measured on 2.1.289)
- **interrupt:** no hook fires when the user interrupts a turn with Escape or Ctrl-C (measured on 2.1.281)
- **interrupt:** Claude Code sets the session idle 37 ms before it writes the interrupt record (measured on 2.1.283)
- **status:** Claude Code can set idle and set it again inside one status read, 93 ms apart (measured on 2.1.283)
- **status:** a single Escape that flushes a queued message sets busy again, not idle (measured on 2.1.282)
- **status:** Claude Code sets busy before a prompt's hooks run (measured on 2.1.282; attribution inferred, the source says "All three were seen live" without naming them)
- **status-file:** Claude Code keeps sessions/<pid>.json in the config directory for every interactive session, with status idle, busy, waiting, or shell (from 2.1.280 to 2.1.289)
- **status-file:** shell is written in place of idle while a background bash task the session started still runs (measured on 2.1.289)

## Subagents

- **subagents:** a tool call's result marks a background launch with isAsync or background set to true, for an async agent or a skill forked to run in the background (measured on 2.1.289).
- **subagents:** a subagent that hands its report back through a tool (SubagentHandback) has no closing text on its Stop hook (measured on 2.1.289).
- **subagents:** a subagent's requests carry its parent's session header and shape; only the cc_is_subagent billing field tells them apart (measured on 2.1.285).
- **subagent that reports through a tool:** a subagent that hands its report back through a tool sends SubagentStop with no last_assistant_message (measured on 2.1.289)
- **background subagent through stop:** Claude Code stays busy from the launch of a background subagent until the turn reporting it back ends, through the Stop of the turn that launched it [ref: hands-session-mgmt-a7t.9qr] (measured on 2.1.289)
- **background subagent records:** a background subagent's launch is the call's result, and its report is a notification: a user record at the prompt, an attachment mid-turn (measured on 2.1.289)

## Turns, queueing, and prompt ids

- **turns:** a flush's prompt id is taken seconds before Claude answers under it, and the turn moves to it; a message queued in between carries it (measured on 2.1.281).
- **turns:** an Escape carries a message queued behind the turn on in the same turn, under the message's own id, and the interrupt record carries that id first (measured on 2.1.281).
- **turns:** a queued message taken mid-turn, or flushed by an Escape, carries its own new prompt id from then on, and no hook names that id (measured on 2.1.281).
- **turns:** a message the user sends while a turn runs is run once that turn's Stop hook returns, under an id no hook names (measured on 2.1.282).
- **turns:** a queued message, a `!` command, and a command such as /compact are each taken with no hook opening a turn for them (measured on 2.1.282).
- **turns:** a message submitted while a turn runs is filed under the running turn's id, and the status is left as it was (measured on 2.1.283).
- **turns:** a command sent to a working session is queued and runs as a command once its turn ends; one sent at its prompt runs at once (measured on 2.1.283).
- **turns:** a draft typed into a working session is queued until its turn ends (measured on 2.1.270).
- **turns:** Escape at a dialog closes it and stops the turn (measured on 2.1.283).
- **turns:** to an empty reply Claude Code answers "[Your previous response had no visible output. Please continue ...]" and asks again; its history keeps the tool results with no reply after them (measured on 2.1.285).
- **turning:** a prompt a command such as /goal hands Claude is written in the same append as what the command printed (measured on 2.1.289)
- **queued prompt during stop hooks:** a prompt sent while the last prompt's other Stop hooks still run gets its UserPromptSubmit under the last prompt's id, trimmed of leading whitespace, then runs under its own id, which its Stop names [ref: brain transcript 15192e41, 2026-10-19] (measured on 2.1.289)
- **input bursts:** Claude Code reads input that arrives close together as one burst (measured on 2.1.286)
- **pasted return:** a Return with more input behind it in the same burst is read as pasted and sends nothing (measured on 2.1.286)
- **typed /compact:** a typed `/compact` shows up as the words, then the compaction, then its command record and output, under a prompt id of their own, naming none of the words (measured on 2.1.286)
- **slash-prefixed prompts:** a prompt Claude was sent that opens with a slash and a name is a prompt, not a command: `/help ...` typed is answered, and a queued `/tmp ...` names a directory (measured on 2.1.283)
- **stop_hook_active:** true on the Stop of a turn Claude goes on in after a Stop hook blocked it (measured on 2.1.282)
- **stop_hook_active presence:** every Stop carries stop_hook_active (measured on 2.1.282)
- **reply display after stop:** Claude Code keeps displaying a reply for seconds after its turn's Stop (measured on 2.1.280)
- **end_turn with no text:** Claude Code asks again after an end_turn with no text, and the turn goes on (measured on 2.1.285)
- **queued inside a turn:** a message sent while a turn runs is queued behind it (measured on 2.1.282)
- **escape during userpromptsubmit:** Escape puts the prompt back in the box and sets idle about 70 ms later, with no Stop and no record (measured on 2.1.282)
- **escape then new prompt:** Claude Code sets idle at the Escape and busy for the next prompt 93 ms later, so no status read sees the idle [ref: hands-status-tlo.tmp] (measured on 2.1.283)
- **queued message after a flush:** a message queued after a flush carries the flush's new id (measured on 2.1.281)
- **queued prompt during a tool:** a prompt queued while a tool runs fires UserPromptSubmit at once, under the running turn's prompt id; Escape then flushes it, and the interrupt record names the queued prompt's new id (measured on 2.1.281)
- **queued message under running turn:** the queued message's hook carries the running turn's id, and once that turn's Stop lands Claude Code runs it under a new id no hook names [ref: hands-status-bpp.44l] (measured on 2.1.282)
- **bang command:** Claude Code is busy while a bang command runs, writes its record under a new id stamped before that busy, Claude answers it, and a Stop names it [ref: 2.1.283] (measured on 2.1.282)
- **slash command at prompt:** /compact is busy for its whole run and writes records under a new id, with no Stop (measured on 2.1.282)
- **message taken into a running turn:** a message taken into the running turn is in that turn and waits behind nothing (measured on 2.1.281)
- **blocked stop goes on:** a blocked Stop's turn goes on under the same prompt id, and its Stop fires again [ref: /tmp/stopprobe] (measured on 2.1.285)
- **own loop and compaction markers:** the request shapes Claude Code sends through the proxy mark the last message with its own loop and the one before with its compaction [ref: 2026-09-29] (measured on 2.1.284)
- **compact on a brain mid-history:** as sent through the proxy, the prompt is merged into the message answering the last reply's call, with the marker on that reply, and a message of Claude Code's own after [ref: hands-wire-6ic.6tr in vivo] (measured on 2.1.285)
- **tools in cache-marked turns:** a working session offers no tools in the body of the turns whose replies call them (measured on 2.1.296)
- **flush records:** a message queued while the loop ran and flushed by Escape: the cancelled call's result and the interrupt carry the queued message's own new id, and so does everything Claude answers after it (measured on 2.1.281)
- **escape during userpromptsubmit:** writes nothing (measured on 2.1.281)
- **flush order:** in a flush, the queued message is written straight after the call it cut off (measured on 2.1.281)
- **turn end records:** as a turn ends Claude Code writes its Stop hook summary, then the turn's duration (measured on 2.1.289)
- **stop hook feedback:** a Stop hook that sends Claude on writes a meta user record, "Stop hook feedback", before Claude answers (measured on 2.1.289)
- **command output records:** a command Claude Code carries out itself writes its output as a user record under the command's prompt id; /goal hands Claude a prompt of its own in the same append as what it printed (measured on 2.1.289)
- **queue:** text pasted into a working session and submitted lands in Claude Code's message queue and runs when the turn ends (measured on 2.1.270)
- **queue:** a message submitted while a turn runs fires UserPromptSubmit with the running turn's prompt_id and leaves the status as it was (measured on 2.1.283)
- **mid-turn words at a boundary:** words sent while a call runs reach the model in the request after that call ends, appended to its tool_result in a system-reminder ("The user sent a new message while you were working: ..."), and the turn goes on under the same prompt id; Claude Code writes a queue-operation enqueue as they are sent and a remove with reason absorbed_mid_turn as they are delivered [ref: promptctl-brain-vu2t.0kr cmt-e0277f83] (from 2.1.289 to 2.1.308)
- **mid-turn words with no boundary left:** words sent while Claude writes, with no call left in the turn, run as the next prompt once its Stop lands, under a new prompt id that posts no UserPromptSubmit, only its Stop [ref: promptctl-brain-vu2t.0kr cmt-e0277f83] (from 2.1.289 to 2.1.308)
- **Ctrl+B:** moves to the background the Bash call running at the press (its tool_response has backgroundTaskId and backgroundedByUser: true) [ref: promptctl-brain-vu2t.0kr cmt-ee913ac6] (from 2.1.289 to 2.1.308)
- **Ctrl+B and other calls:** it never moves an MCP tool's call; the Bash calls of one message run one at a time, so a batch of them needs a press each [ref: promptctl-brain-vu2t.0kr cmt-ee913ac6] (measured on 2.1.289)
- **send now:** Ctrl+X Ctrl+S, with a message typed mid-turn, moves every running call to the background, Bash (backgroundTaskId, backgroundedToDeliverMessage: true) and MCP alike, none interrupted, and the message reaches the model in the next request, under the same prompt id [ref: promptctl-brain-vu2t.0kr cmt-e0277f83] (from 2.1.289 to 2.1.308)
- **send now in a batch:** sent while the first of three Bash calls of one message runs, the first is moved to the background; the second, which may start once the first is moved, is moved too if it has started (it posts PreToolUse and no PostToolUse), and is refused otherwise, as the third always is ("The user doesn't want to take this action right now ..."); the turn goes on under a new prompt id that posts no UserPromptSubmit, which the words' record and the later results' carry [ref: promptctl-brain-vu2t.0kr cmt-e0277f83] (from 2.1.289 to 2.1.308)
- **send now batch, call by call:** in each of three recordings the first call was moved as send now was pressed and its result written under the running prompt id; the second started then, posted no PostToolUse, and was moved about 4 s after its PreToolUse (4.06 s in tests/fixtures/recorded_turns/sent_now_mid_batch.jsonl), before the dequeue that opens the new id (measured on 2.1.308)
- **notification turns:** a background task that ends after its turn's Stop opens a turn of Claude Code's own, its UserPromptSubmit under a new prompt id with a `<task-notification>` prompt; one that ends mid-turn posts its UserPromptSubmit under the running turn's id (from 2.1.289 to 2.1.308)
- **notification queue records:** a background task's notification that opens a turn after a Stop is written as a queue-operation enqueue then dequeue (from 2.1.289 to 2.1.308)
- **send now status:** as send now moves a batch's running call to the background, Claude Code sets shell for a few ms, then busy again, and the turn goes on (measured on 2.1.308)
- **records behind hooks:** Claude Code appends transcript records in batches, behind the hooks: in the recorded turns, a turn's opening prompt reached the transcript 0.07 to 0.62 s after its UserPromptSubmit arrived, over 12 turns, and a Stop's stop_hook_summary 0.11 to 0.29 s after the Stop, over 13 Stops (scripts/turn_timings.py); two earlier recordings, not kept, reached 4.0 s and 2.6 s (measured on 2.1.308)
- **busy around a prompt's hook:** in the recorded turns, the busy nearest a turn's opening UserPromptSubmit was set within 25 ms of it, before it for 1 of 12 (scripts/turn_timings.py); an earlier recording, not kept, had one 0.48 s before and one 0.38 s after (measured on 2.1.308)
- **message-display with a fast endpoint:** with the hook endpoint answering at once, a turn's last MessageDisplay arrived within 2 ms of its Stop, over 13 Stops (scripts/turn_timings.py) (measured on 2.1.308)
- **recorded turns:** tests/fixtures/recorded_turns hold seven turns as the brain's pin ran them, every hook, transcript record and status in the order Claude Code said them, made by scripts/record_turns.py (measured on 2.1.308)

## Transcript records

- **transcript:** a turn's records reach the transcript up to about 40 ms after its Stop hook fires (measured on 2.1.285).
- **transcript:** an interrupt's record is written 37 ms after Claude Code sets idle (measured on 2.1.283).
- **transcript:** every record carries a time, on the clock Claude Code stamps statuses with (measured on 2.1.282).
- **transcript:** the transcript record of a prompt keeps the leading space the prompt was typed with (measured on 2.1.283).
- **transcript:** a turn is matched by the id of the prompt that opened it and by any prompt it went on under after a flush; a Stop or an interrupt may name either (measured on 2.1.281)
- **transcript:** Claude Code writes stop_hook_summary and turn_duration records as a turn ends, also for a turn that launched a background subagent, which keeps the session busy until its report is answered (measured on 2.1.289)
- **transcript:** turn_duration is missing from about one stop in five (measured on 2.1.289)
- **transcript:** a notification handed to a session mid-turn is written as a queued_command attachment with its markup as the prompt; one handed to a waiting session is a user record that opens a turn (measured on 2.1.288)
- **transcript:** a turn the user stops at the keyboard ends with "[Request interrupted by user]", or "[Request interrupted by user for tool use]" when a tool was running; the same two texts in every release checked (measured on 2.1.201 to 2.1.281)
- **transcript:** records of commands the user ran carry command-name, command-message or bash-input markup and output tags (local-command-stdout, bash-stdout, and their stderr forms) (measured on 2.1.226 to 2.1.286)
- **transcript:** every prompt of a main session carries a promptSource (typed or queued), while the words a command was typed as carry none (measured on 2.1.197 to 2.1.286)
- **unreadable transcript:** Claude Code prints "No conversation found with session ID" for a transcript it cannot read (measured on 2.1.289)
- **worktree move of transcript:** entering a worktree renames the session's transcript, with its subagents' folder, into the worktree's project, and hooks name the new path from then on (measured on 2.1.299)
- **user-run records:** Claude Code's records for what the user ran are a caveat ahead of a command it carries out itself, the command, and its output (naming the command as parent, often a system record); a skill command's body is a meta record; a ! command has its output (measured on 2.1.285)
- **transcript:** Claude Code renames the transcript, subagents and all, into the worktree's project as the session enters one (EnterWorktree) (measured on 2.1.299)

## Hooks and dialogs

- **hooks:** SessionStart, idle_prompt, and SessionEnd carry no permission_mode (verified live) (measured on 2.1.281).
- **hooks:** the Prompted, Stopped, PermissionRequested, and ToolFinished hooks always carry permission_mode (measured on 2.1.281).
- **hooks:** a SessionEnd reason is one of clear, resume, logout, prompt_input_exit, bypass_permissions_disabled, or other; a closed terminal reports other (measured on 2.1.270).
- **hooks:** every hook of a background subagent carries the prompt_id of the turn that started it, even after that turn stopped and another opened (measured on 2.1.281).
- **hooks:** a turn opened by a background task's notification has its own prompt_id, which its UserPromptSubmit carries too (measured on 2.1.281).
- **hooks:** when another Stop hook blocks, Claude goes on under the same prompt id and the next Stop carries stop_hook_active (verified live) (measured on 2.1.282).
- **hooks:** Claude Code runs no hook, SessionStart included, at a startup dialog (workspace trust, a project's new MCP servers) until it is answered (measured on 2.1.283).
- **hooks:** an HTTP hook for SessionStart never posted, in seven recorded sessions, where a command hook for it did; the pin's code skips HTTP hooks for SessionStart and Setup ("HTTP hooks are not supported for ...") (measured on 2.1.308)
- **hooks:** Claude Code reads the plan file into an ExitPlanMode call's input before any hook sees it (measured on 2.1.281)
- **hooks:** the input of an approved ExitPlanMode call is what the approval sent, empty unless the plan was edited at the dialog (measured on 2.1.281)
- **hooks:** the plan dialog's own yeses send an empty updatedInput, so the plan is read from its file as the user left it, with the mode to leave plan mode for (measured on 2.1.281)
- **hooks:** Claude Code ignores an allow with no updatedInput for a tool that asks the user something, and shows its dialog instead (measured on 2.1.281)
- **hooks:** every hook carries the session's permission_mode (measured on 2.1.281)
- **hooks:** a hook fired inside a subagent carries the subagent's own permission mode, not the session's (measured on 2.1.281)
- **hooks:** every prompt and Stop hook carries the prompt_id that names its turn (measured on 2.1.281)
- **hooks:** a UserPromptSubmit hook sets the session title by printing hookSpecificOutput.sessionTitle, which Claude Code shows on its terminal (measured on 2.1.286)
- **hooks:** the reply shape Claude Code parses from a PermissionRequest hook's stdout (measured on 2.1.270)
- **hooks:** Claude Code dispatches MessageDisplay synchronously for every batch of lines it displays (measured on 2.1.270)
- **hooks:** Claude Code waits on each displayed batch in the agent's path, so a slow, down or hung display daemon costs lines of narration and never the agent's speed (measured on 2.1.280)
- **side questions:** Claude Code 2.1.289 reads its side questions as Auxiliary (measured on 2.1.289)
- **display payload:** the payload Claude Code posts for display, with paths shortened, has the shape in the test's DISPLAYED dict (measured on 2.1.288)
- **questions payload:** the questions Claude Code posts, captured live, have the shape in the test's QUESTIONS dict (measured on 2.1.280)
- **plan input:** the plan input Claude Code sends carries the plan file already read in, as plan and planFilePath (measured on 2.1.281)
- **hook payload fields:** the fields the hook tests parse were captured from Claude Code (measured on 2.1.270)
- **reply broken past its retry:** when the API breaks a reply past its retry, the turn ends with the block open (measured on 2.1.285)
- **side question shapes:** Claude Code sends a side question through the proxy in three shapes that differ in everything but the words wrapping the question [ref: hands-wire-6ic.l2o] (measured on 2.1.285)
- **side question name:** Claude Code 2.1.289 names a side question auxiliary (measured on 2.1.289)
- **permission-dialog:** a permission dialog swallows pasted text and takes the Enter as "Yes" (measured on 2.1.270)
- **trust-dialog:** the workspace-trust dialog swallows a pasted text the same way as a permission dialog (measured on 2.1.278)
- **startup-dialog:** Claude Code runs no hook until a startup dialog is answered, so a session at one has no membership (measured on 2.1.283)
- **hook-payload:** SessionStart, Notification, and SessionEnd do not carry permission_mode, while the other subscribed events do (measured on 2.1.281)
- **hook-payload:** the event-specific hook payload fields were read out of the Claude Code bundle (measured on 2.1.263)
- **hook-payload:** payloads captured on 2026-09-14 carry no session title (measured on 2.1.270)
- **session-title:** a UserPromptSubmit hook reply's hookSpecificOutput.sessionTitle is written by Claude Code as a custom-title record and shown on the terminal tab, exactly as /rename does (measured on 2.1.286)
- **hook-payload:** SubagentStop's last_assistant_message is absent for a report handed back through a tool (measured on 2.1.289)
- **permission-reply:** the PermissionRequest reply shape was read out of the bundle; an allow runs the tool, a deny's message reaches the agent as the tool's error, and empty output decides nothing (measured on 2.1.270)
- **question:** an AskUserQuestion is answered by allowing it with an answers object; the shape was read out of the bundle and the agent went on with answers given by voice (measured on 2.1.280)
- **plan:** an ExitPlanMode approval with an empty updatedInput and updatedPermissions reads the plan from its file; a keep-planning deny's message reaches the agent as feedback; the shape was read out of the bundle (measured on 2.1.281)
- **http-hooks:** Claude Code supports HTTP hooks (measured on 2.1.270)
- **message-display:** an interactive session fired 35 to 40 MessageDisplay batches for a 40-line reply, one per finished line or two, 0.1 to 0.5 s apart; claude -p fired one final batch holding the whole text (measured on 2.1.280)
- **message-display:** at most one MessageDisplay batch is in flight; against an endpoint that took 1 s, 16 batches arrived and the last came 4.8 s after the turn ended (measured on 2.1.280)
- **message-display:** the turn took 11.3 to 12.6 s against a fast endpoint, a 1-second one, a refused port, and a port that accepts and never answers (measured on 2.1.280)
- **hook-url:** Claude Code puts no variable in a hook's URL (measured on 2.1.288)
- **dialog-race:** Claude Code shows its permission dialog at once and runs the PermissionRequest hook beside it, and whichever answers first decides (measured on 2.1.270)
- **dialog-race:** answering No or Esc at the dialog fires no post-tool hook and no Stop, but Claude Code kills the waiting hook, and the closed connection releases the session (measured on 2.1.270)
- **session-end:** Claude Code ends a session on SIGTERM as on a closed terminal, firing SessionEnd (measured on 2.1.289)
- **session-end:** /exit and a double Ctrl-C report prompt_input_exit, /clear reports clear, and a closed terminal reports other (measured on 2.1.270)

## Wire (requests the proxy sees)

- **wire:** Claude Code names the prompt of a session's main request only when ANTHROPIC_BASE_URL is Anthropic's API; behind a gateway it names none (measured on 2.1.285).
- **wire:** the request class header is derived from querySource in Claude Code's services/api/client.ts (measured on 2.1.299).
- **wire:** Claude Code names fork requests in the class header too, not only auxiliary requests (measured on 2.1.289).
- **wire:** a side question asked before any turn has finished carries no cache marker at all (measured on 2.1.285; ticket hands-wire-6ic.l2o).
- **wire:** Claude Code's list of request classes lives in its own code (measured on 2.1.299).
- **wire:** Claude Code's es() names every request that is neither its loop's nor a subagent's auxiliary, side questions and prompt suggestions among them (measured on 2.1.289).
- **wire:** Claude Code merges adjacent user messages, so a prompt of its own can follow a prompt just typed or a tool's result, and a message of its own can follow it (measured on 2.1.285).
- **wire:** a working session marks an earlier message of a long history as well as the newest one, with markers on messages 5 and 7 of 8 (measured on 2.1.285).
- **wire:** a working session's request can offer no tools in its body (measured on 2.1.296).
- **wire:** Claude Code's main turns ask for streamed replies, except the request it asks again whole after a stream breaks before any block finishes (measured on 2.1.289).
- **wire:** a spent subscription usage limit is refused with status and reset headers and no limit in the body; Claude Code reads the same reset (hands-wire-6ic.gfq) (measured on 2.1.285)
- **wire:** Claude Code sends the x-claude-code-request-class header only to a gateway it is told to send it to, or to its own API when a flag of its own is on (measured on 2.1.299)
- **wire:** Claude Code asks again after an overload whatever the x-should-retry header says, and waits out a spent usage limit to continue the task at its reset (autoContinueAtUsageLimit) (measured on 2.1.286)
- **wire:** Claude Code's decision whether to ask again after an answer is its function lMo (measured on 2.1.286)
- **wire:** Claude Code's hook URL takes no variables (measured on 2.1.288)
- **mcp tool search:** under tool search Claude Code offers MCP tools deferred, as a placeholder tool with defer_loading set, so the server may be connected and not blamed (measured on 2.1.288)
- **worktree move, posttooluse order:** the EnterWorktree call's own PostToolUse is the first hook to name the transcript's new path; the cwd stays the start's (measured on 2.1.299)
- **message after tool results:** Claude Code sends a message of its own after the tool results (measured on 2.1.285)
- **api failures in claude code:** a spent limit is asked once; an API the proxy cannot reach is asked eleven times; what failed the turn is Claude Code's latest answer [ref: hands-wire-6ic.gfq] (measured on 2.1.285)
- **mid-stream api break:** when the API breaks a reply mid-stream, Claude Code makes no retry and no fallback request, only StopFailure [ref: hands-wire-6ic.6dz] (measured on 2.1.285)
- **retry without stream:** the request Claude Code asks again with, without a stream, is answered whole after the blocks ahead of it (measured on 2.1.289)
- **billing line:** a working session's system prompt opens toward Anthropic's API with an attribution line, cc_version and related fields; a subagent's adds cc_is_subagent=true (measured on 2.1.285)
- **long-history cache markers:** a working session marks an earlier message of a long history as well as its last (measured on 2.1.285)

## Brain

- **brain:** the id of the tool_use block a call runs is in the tools/call _meta under claudecode/toolUseId (measured on 2.1.288)
- **brain:** Claude Code asks for server/discover first and, refused, opens with initialize (measured on 2.1.284)
- **brain:** --bare switches off the OAuth login too, so the slim switches are set one by one instead (measured on 2.1.284)
- **brain:** a scheduled prompt opens a turn of its own when it fires, as a background task's notification does (measured on 2.1.288)
- **brain:** loaded claude.ai connectors joined every request after the first: turn 2 grew from 8.7 KB to 112 KB (hands-wire-6ic.eph) (measured on 2.1.284)
- **brain:** settings.json is the only place Claude Code reads syncClaudeAiSkills and syncClaudeAiPlugins from, not --settings and not the environment (measured on 2.1.288)
- **brain:** Claude Code prefers the credentials and provider switches in FOREIGN_LOGINS to its own login (measured on 2.1.289)
- **brain:** a permission dialog nobody can see holds its turn open forever, with no hook to say so (measured on 2.1.288)
- **brain:** Escape ends a turn posting none of the brain's hooks (measured on 2.1.285)
- **brain:** a paste of four lines or more is in a UserPromptSubmit prompt inside a pasted_content tag pair, with a line end added where it had none (measured on 2.1.289, 2026-10-04)
- **brain:** Claude Code trims trailing whitespace from every prompt it keeps (measured on 2.1.289, 2026-10-04)
- **brain:** Claude Code trims leading whitespace from a prompt it queued (measured on 2.1.289; brain transcript 15192e41, 2026-10-19 05:29:11)
- **brain:** Claude Code finds the session it resumes under any project directory in its config directory (measured on 2.1.289)
- **brain:** Claude Code keeps the session it resumes unless --fork-session is given (measured on 2.1.289)
- **brain:** a resumed conversation keeps the system prompt it first had until compacted, even when launched with a new --append-system-prompt (measured on 2.1.289)
- **brain:** a --settings env value outranks the brain's settings.json env and the process environment (measured behind a localhost base URL, hands-brain-8gb) (measured on 2.1.288)
- **brain:** Claude Code's answer on an API key its settings.json sets is by the key's last 20 characters (measured on 2.1.289)
- **brain:** `claude auth status` reports authMethod claude.ai for a claude.ai login and api_key for a Console key (measured on 2.1.289)
- **brain:** what is typed while a permission dialog is held goes to the dialog, whose Return says yes (spike hands-brain-d8g.aur) (measured on 2.1.288)
- **brain:** Escape refuses a permission held at its dialog and leaves its hook unanswered for good (spike hands-brain-d8g.aur) (measured on 2.1.288)
- **brain:** Escape on a prompt stopped before any reply puts it back in the input, and its cancel pops the queued input into it (measured on 2.1.289)
- **brain:** Ctrl-C on an empty input arms Claude Code's exit, which a second Ctrl-C within 800 ms takes, whatever was typed between (read from its source on 2026-09-30, not measured) (measured on 2.1.285)
- **brain:** a key typed 10 ms behind a turn joined that turn's prompt or was left in the input with it; none of 8 typed once the turn was taken (measured on 2.1.286, 2026-09-30, 2 of 4 in the first trial)
- **brain:** Claude Code exits with "No conversation found" on a transcript it cannot read (measured on 2.1.289)
- **brain:** Claude Code turns on bracketed paste once it reads its input, 0.19 s in, with its terminal raw from 0.02 s (measured on 2.1.289, 2026-10-04)
- **asides:** the third /btw typed into one process carried the first two and their answers; x over an answer cleared all but that answer, and /clear cleared none (measured on 2.1.286, 2026-09-30)
- **asides:** a side question started for one question leaves no transcript, line of history or session directory in the brain's config directory (measured on 2.1.286, 2026-09-30, over 12 questions)
- **stage:** Claude Code asks again a reply the API broke before any block of it finished, whole, and without a stream (measured on 2.1.289)
- **stage:** Claude Code keeps a reply the API broke off after a block finished out of the brain's history, and the turn ends in StopFailure (hands-wire-6ic.6dz) (measured on 2.1.285)
- **fake claude login:** the fake records login, onboarding, and API-key approval for claude.ai and api_key auth the way Claude Code writes them (measured on 2.1.289)
- **brain skills:** the brain's skills are installed, listed, changed, and removed in its own config directory, not the user's [ref: hands-brain-d8g.33b] (measured on 2.1.288)

## Fritter (terminal input)

- **fritter:** a bracketed paste and the Return behind it, read together, are sent, a short text and a 3 KB one alike; a Return read apart from the paste sends nothing (measured on 2.1.283).
- **fritter:** text ending on an @ or # token, read together with its Return, is sent as it is, and no completion list opens (measured on 2.1.283).
- **fritter:** text ending on a backslash is not sent; the Return becomes a newline and the next text typed is joined onto it (measured on 2.1.283).
- **fritter:** text behind a leading space that starts with /, @, or ! is sent as plain text, and the transcript records it with the space (measured on 2.1.283).
- **fritter:** the Claude Code binary asks for bracketed paste (measured on 2.1.283).
- **fritter:** killing the tmux session a wrapped session runs in, or sending fritter SIGHUP or SIGTERM, ends fritter, claude, and the processes claude started within a second, and removes the socket directory (measured on 2.1.283).
- **fritter:** an ESC followed within 40 ms by another byte is read as one chord, and an ESC alone from 45 ms on (measured on 2.1.285; measured 2026-09-30 under ticket hands-wire-6ic.ulk).
- **fritter:** text wrapped in bracketed paste arrives as one multi-line message (measured on 2.1.278).
- **fritter:** a two-line prompt sent with the display asleep arrived as one message and was answered (measured on 2.1.278).
- **fritter:** a process the child spawns sees FRITTER_SOCKET (measured on 2.1.278).
- **fritter:** a permission dialog and the workspace-trust dialog each swallow a paste and take the Return as their answer; a session at a dialog is not one to type text into (measured on 2.1.278).
- **tmux:** fritter's loneEscape holds the session's input for 0.1 s after an Escape (fritter/control.go) (measured on 2.1.285)
- **fake claude terminal input:** Claude Code keeps what is typed while it is still starting, including keys typed 0.05s in [ref: 2026-10-04] (measured on 2.1.289)

## Sessions and startup

- **sessions:** `claude update` still updates with DISABLE_AUTOUPDATER set; only DISABLE_UPDATES stops it (measured on 2.1.299)
- **sessions:** Claude Code names itself as the parent of each process it runs, so a claude started under a session writes its turns into that session's transcript and makes none of its own (measured on 2.1.288)
- **sessions:** a project's new MCP servers raise a dialog that holds a session before its first hook, unless enableAllProjectMcpServers is set, which removes the dialog; a server in disabledMcpjsonServers is still not run with it set (measured on 2.1.299)
- **sessions:** SIGTERM ends a Claude Code session: it fires SessionEnd and exits within two seconds (measured on 2.1.289)

## Other

- **events:** Claude Code displays text as it streams it, in one batch of finished lines, or every line since the last batch when the one before was slow to be taken (measured on 2.1.280).
- **events:** Claude Code displays text for seconds after its turn's Stop (measured on 2.1.280).
- **events:** a closed terminal's reason is other (measured on 2.1.270).
- **voice:** an oauth_token-style token names no holder (measured on 2.1.289).
- **voice:** the session footer shows the default permission mode as "manual mode on" (measured on 2.1.281).
- **userpromptsubmit prompt text:** a paste of four or more lines is kept in its tags with a line end added where it had none; other prompts have trailing whitespace trimmed (measured on 2.1.289)
- **spent subscription limit:** the API refuses a spent subscription limit with HTTP 429 and unified rate-limit headers, and Claude Code reads that refusal as a usage limit [ref: hands-wire-6ic.gfq] (measured on 2.1.285)
- **prompt suggestion:** a prompt suggestion is a fork that skips the cache write, which Claude Code 2.1.289 names auxiliary (measured on 2.1.289)
- **features:** "/compact" sent as a command reached a live session as a command, "slash compact" as text, and "stop it" sent Escape, which ended a working turn and closed a question dialog (measured on 2.1.283)
- **features:** approving a plan left plan mode for the mode meant, and a plan sent back stayed in plan mode with the feedback reaching the agent (measured on 2.1.281)
- **features:** an Escape or Ctrl-C mid-turn, at each of mid-tool, mid-text, a permission dialog, and by Ctrl-C, left the session idle and told (measured on 2.1.281)
- **control-point:** the newest recording's system prompt is three blocks of 81, 57, and 27,574 characters, the last two with 1h cache breakpoints, and its body carries thinking adaptive, context_management clearing old thinking, output_config, and metadata.user_id with device, account, and session ids (measured on 2.1.111)
- **control-point:** the newest recording's first user message is 509 characters typed plus 47,889 characters injected as system-reminder blocks: skills list 15K, CLAUDE.md and context 30K, deferred tools 2K, SessionStart hook output 1K (measured on 2.1.111)
- **control-point:** later turns inject 0 to 1.1K characters on a typical turn and 45K on a turn carrying a Read result with plan-mode reminders; fifty-seven tool results over the session totalled 171K characters (measured on 2.1.111)
- **control-point:** the ~300 ms requests have the shape of count_tokens calls (no max_tokens, a file's contents as the one message, input_tokens back), sent in bursts of four (measured on 2.1.111)
