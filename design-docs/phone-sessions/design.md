# The sessions hands sends off, on the phone page

Written 2026-11-23, revised 2026-11-24, from a conversation between Brandon and Claude. Ticket `promptctl-phone-app-9d9y.68y`.

## The idea in one paragraph

The phone gains a page of the sessions hands knows, one card for each, grouped by the project each works in. A card says what the session is doing in hands' own words: `working; the last thing it set out to do: run the test suite`, `waiting for permission to use Bash`, `idle`. Sessions waiting on Brandon are never folded; idle and ended ones fold into one line per group until tapped. Tapping a card opens the session's timeline: every tool call it made, read from Claude Code's own transcript, newest first, each line the time, the tool, one line of what it set out to do, and a mark for one that failed or is still running, grouped under the prompt that started each turn. The page reads back as far as Brandon scrolls, and the calls from while he watched and from while he did not appear the same way, because both are the file's. Sessions that ended keep their timeline for as long as this run of hands remembers them, because the transcript outlives the session. A button shows the session's terminal as text, once per tap. Nothing on the page changes a session. The existing way to show a tmux session in a browser, `tmux-control-mode-js`, was weighed and is not used; "Why this and not the alternatives" says why.

## The picture

```
  the phone page                                           hands, on the Mac
 ┌────────────────────────────────────────────┐           ┌──────────────────────────────────────────┐
 │  the sessions page                         │  GET      │                                          │
 │                                            │  /sessions│  the registry: every session this run    │
 │  phone-wake-bw7 ───────────────────────    │  (waits   │  heard of, live or Gone, its state, its  │
 │   ┌ naming fix ───────────────────────┐    │  for a    │  dialogs, its project, its transcript    │
 │   │ waiting for permission: Bash      │◄───┼──change)──┤            │                             │
 │   │ git push … asked 2 min ago        │    │           │            ▼                             │
 │   └───────────────────────────────────┘    │           │  cards, grouped by cwd, waiting first,   │
 │   ┌ wake word page ───────────────────┐    │           │  idle and ended folded                   │
 │   │ working: run the test suite       │    │           │                                          │
 │   └───────────────────────────────────┘    │           │                                          │
 │   ▸ 1 idle · 4 ended                       │           │                                          │
 │  cc-hands ─────────────────────────────    │           │                                          │
 │   ▸ 3 idle                                 │  GET      │  the transcript, folded as the tail      │
 │          │ tap a card                      │  /sessions│  folds it, held while a page watches:    │
 │          ▼                                 │  /<id>    │  newest page waits for a change,         │
 │  the timeline, newest first ◄──────────────┼──/calls───┤  ?before=<record> answers at once        │
 │   10:34 Bash  run the tests          ✗     │           │            │                             │
 │   10:33 Edit  voice/tools.py               │           │            ▼                             │
 │   ─ 10:31 asked: fix the status line ─     │  GET      │  the subagent's own transcript, beside   │
 │          │ "Show screen"                   │  …/agents │  the session's                           │
 │          ▼                                 │  /<agent> │                                          │
 │  the terminal as text ◄────────────────────┼──/screen──┤  tmux capture-pane, once                 │
 └────────────────────────────────────────────┘           └──────────────────────────────────────────┘
```

## What Brandon does with it, step by step

1. Brandon tells hands to start a session in the `phone-wake-bw7` worktree and have it finish the wake word page. hands starts one (`start_session`), tells it the work, and he leaves the desk for two hours.
2. He opens the page, at the address `hands phone` prints with `sessions` on its path, as the conversation and backlog pages are opened. The page shows one group per project. Under `phone-wake-bw7` one card reads `waiting for permission to use Bash`, with `git push origin …` under it and how long it has waited. That is the first thing the page is for: a session stopped on him, found without asking out loud.
3. Under the same heading, a folded line reads `▸ 4 ended`. The session he started two hours ago is not the one waiting: it wrote a hand-off and ended, and the next in the chain started, twice over. He taps the line and the four cards unfold, newest first, each `ended`, with when it last wrote.
4. He taps the waiting card. Its timeline opens at the top: the prompt that started its turn as a heading, and under it the calls newest first, the first of them `10:34 Bash run the tests ✗`, then `10:33 Edit voice/tools.py`, and so on down. He scrolls; the page reads older calls as he nears the bottom, back through the turns before, each under its prompt, until the session's first.
5. One line reads `10:12 Agent review PR 557 against the laws · 48 calls`. He taps it and the subagent's own calls unfold beneath it, read from its own transcript.
6. He opens an ended card. Its timeline reads the same way: the calls it made, the hand-off it wrote, the prompt that started it. Nothing about it says it ended but its card.
7. He leaves the waiting card's page open. When he allows the push by voice, the card turns to `working`, and the new calls appear at the top of the timeline, under the same heading, while the lines he was reading stay where they are.

## Where this design departs from the words of the request

Each of these is Claude's, and Brandon's to overrule.

**Every session hands knows is shown, not only the ones hands sent off.** hands keeps no record of who started a session or who sent it a turn: `Session` (`src/hands/core/session.py`) has none. A filter would be a second record of a fact the registry never held, and a session Brandon started at the desk is as worth a glance from the phone. Waiting first, and folding, keep the sessions hands sent off from being buried.

**Ended sessions are the ones this run of hands remembers.** The registry keeps a session that ended as `Gone`, with its membership and so its transcript path, for the rest of the run (`src/hands/core/reducer.py`, `src/hands/sessions/registry.py`); nothing prunes it. The page lists those. A restart of hands forgets them, and a chain's earlier links go from the page with it; their transcripts stay on disk. Brandon said the history he wants is the calls, not what comes after a session ends, so nothing is built to find a transcript the run never knew: not the audit log, whose `transcript` facts last only as long as its segments, which on this machine was 6h22m, and not a listing of Claude Code's project directories, which would show every session ever run there. "Why this and not the alternatives" names what would change this.

**Subagent calls are folded under the call that started them, not interleaved.** By the audit's count, session 4d1d1d78 made 2998 tool calls where its own transcript holds 699: three calls in four were subagents'. Interleaved by time they would bury the session's own work. Each subagent is one line, with how many calls it made, and opens on a tap.

**Idle folds every session at its prompt, including ones that worked.** The request says idle sessions are folded away. Of the 42 sessions in the measured window, 30 never ran a prompt and 12 did; a session that ran a turn an hour ago and sits at its prompt is as quiet as one that never did. The fold line counts them, and one tap opens them, so nothing is hidden, only quiet.

**The page changes nothing.** Typing into a session is hands' to decide: `core.reach.writer` and the refusals in "Typing into a session" (`docs/architecture.md`) keep a paste from landing in a permission dialog or the wrong program. A page with a keyboard would need that whole rule or bypass it. This page has no button that reaches a session.

## The timeline

**The source is the session's transcript, Claude Code's own JSONL, and nothing else.** On this machine every `tool_use` in a transcript has its `tool_result` (699 of 699, 728 of 728, 258 of 258, 583 of 583 in the four files checked), a failed call is a result block with `is_error: true` (20, 28, 7 and 25 in those files), every record carries a `timestamp`, every Bash call carries a `description` (423 of 423, 407 of 407, 150 of 150, 352 of 352), and the file persists after the session ends (found for all 7 checked, 5 of them ended). The audit log has none of this: its `hook` events carry the hook's name and the session id, no tool name and no input, and it rolled in 6h22m under heavy sessions. hands already folds the transcript into calls and their results in one place, `Turning` (`src/hands/sessions/turning.py`), which the live tail and `read_session` both run, so the timeline is a third reader of that fold and a second reader of nothing.

**One line is one call.** It carries:

- the time Claude Code wrote the call's record, shown as `HH:MM` on the phone's clock, with the day shown where it changes: the one file of session 4d1d1d78 runs from 11-23 05:08 to 11-24 10:34;
- the tool, by the name Claude Code gave it, an MCP tool by its own name after the server's;
- one line of what it set out to do, from the call's input alone, never its result: the result decides only the mark. Bash is 55 to 60% of calls, and its `description` is what Claude Code asked for in the imperative, so `run the tests` is the line and the command itself is not;
- the mark: nothing for a call that came back without error; `✗` for `is_error`; `…` for a call with no result yet in a live session; `cut off` for a call with no result that the file shows the session moved past (a reply, a prompt, or an interruption after it, as `_settled` in `src/hands/sessions/backfill.py` already judges), and for every unanswered call of an ended session.

How each tool's line is made, from the one table that already turns a call's input into words for the voice, `doing` (`src/hands/core/progress.py`), which gains the rows it lacks:

| tool | the line |
|---|---|
| `Bash` | its `description`; the command's first line where there is none |
| `Edit`, `MultiEdit`, `NotebookEdit` | `edit` and the `file_path`, with the session's `cwd` taken off the front |
| `Write` | `write` and the `file_path`, the same way |
| `Read` | `read` and the `file_path` |
| `Grep`, `Glob` | `search for` the `pattern`, and the `path` where one is given |
| `WebFetch`, `WebSearch` | the host of the `url`; the `query` |
| `Agent` | its `description`, and the `subagent_type` where one is named |
| `Skill` | the `skill` with its `args`, as `/laws:law-review 557 1` |
| `SendMessage` | who its input names as the recipient, and the first line of the message |
| `AskUserQuestion` | the first question's text |
| `ExitPlanMode` | `asked to approve its plan` |
| `TaskCreate`, `TaskUpdate` | the task's subject, or its id |
| any other, MCP tools included | the tool's own name, and its input as JSON cut to the line |

The voice leaves the two dialog tools out, since each is spoken as its hook arrives; that exclusion stays the voice's, applied where `doing` is called for progress, and the table itself answers for every tool. The field names of `SendMessage`'s input were read from no transcript here and are the builder's to confirm.

**Calls group under prompts, newest first throughout.** Each turn's opening is a heading above its calls: what was asked, a notification, a command the user ran, or a `!` command, as `Opening` (`src/hands/core/turn.py`) already tells them apart, with its time and its first line cut to the width of the phone. The newest turn is at the top, its heading first, then its calls newest first, so the first line under the first heading is always what the session did last, which is what the page is opened to see. Under a heading the lines run back to the turn's first call; below them the heading of the turn before. A transcript read from part way through a turn, as `turns` in `src/hands/core/turn.py` allows, puts those calls under a heading that says the turn began before the file.

**Paging goes by record, not by byte.** A page is 80 lines, headings included, a value found by changing it. The page's read for older lines names the record of the oldest line it holds, by the uuid Claude Code writes on it, and is answered with the 80 before it, as `read_turn` is already paged with `since` (`src/hands/voice/tools.py`). A uuid names a record whatever is appended after it, and whatever directory the file is renamed into: Claude Code renames a transcript into the worktree's project as a session enters one, which the tail reads straight through by inode (`Position` in `src/hands/sessions/tail.py`). A byte offset would be as stable under appends, but the fold that makes a line needs the result record that comes after the call, so a cut by bytes would cut calls from their results; and nothing reads back by bytes today, where `read_turn` reads by record. A record that carries no uuid rides with its neighbour, as `_page` in `src/hands/voice/tools.py` already keeps one record's happenings together. A cursor naming a record the file no longer holds, as after a file written anew, is refused in words, and the page reads again from the newest.

**The live end grows without moving what is read.** The newest page is a read that waits, as every phone page's reads do (`hands.voice.longpoll`): the page says the version it has, and is answered when the newest 80 lines differ, which a new call, a result landing, or a new turn each does. The page draws the new lines above the ones it holds. While the reader is at the top, the view stays at the top and the lines show at once; while the reader has scrolled down, the page holds them and shows `3 new calls` at the top until the reader returns there, and nothing below moves. A result that lands for a call whose line is on the newest page changes that line's mark in place; one that lands for a call already paged back, which a command that runs for minutes while the session makes 80 more calls can do, is seen when the page is opened again.

**Subagent calls are one line each, and open on a tap.** A session's own transcript holds none of its subagents' calls (no `isSidechain` record in the four files). Claude Code keeps each subagent's transcript beside the session's, in a directory named for the session, `<session>/subagents/agent-<id>.jsonl`, with a `.meta.json` naming its job; hands reads them there today (`src/hands/sessions/subagents.py`, `_subagent_progress` in `src/hands/sessions/tail.py`), and the directory the measurement saw beside each transcript and did not list is that one. The line for an `Agent` or `Skill` call carries the subagent's id where its result names one (`_agent` in `src/hands/core/steps.py`), the count of calls in its transcript, and whether it has reported. A tap reads the subagent's transcript through the same fold, paged the same way, and shows its calls indented under the line. A subagent's own subagents are its lines, one deeper.

**An ended session shows the same timeline.** Its card says `ended`, with when its transcript last changed, from the file's modification time: `Gone` holds no stamp, and the file is the one thing of it that is still there. Every call without a result is `cut off`, since nothing will answer it.

**A chain of sessions in one worktree reads as one group, newest first.** Five of the seven sessions that ended in the measured window ran in `phone-wake-bw7`, each starting as the one before ended, making 108 to 258 calls over 18 to 73 minutes. The group's heading is the worktree's name; under it the live session's card, then the fold line, and open, the ended cards newest first. Each timeline ends, at its top, on the hand-off the session wrote, and the next card's timeline starts, at its bottom, on the prompt that read it.

**What leaves no line.** A compaction happens inside the same file (`compact_boundary` and the summary record are Claude Code's own, and `turn_record` in `src/hands/sessions/transcript.py` opens no turn on them), so the calls on both sides of it run on without a mark. Claude's text between calls is not a line: the timeline is the calls. Each measured file carries two distinct `sessionId` values; what the second means was not examined, and the timeline shows every record of the file the membership names, whichever id a record carries, so no call is dropped by the answer. A `/clear` or a resume in the same process is a new session to hands, with a new id and file (`MovedOn`, `src/hands/core/events.py`), so it is a new card and the old one is `ended`.

## The cards, the fold and the groups

A card is one session's state in the words `list_sessions` gives the brain (`describe_listing`, `_spoken_state` in `src/hands/voice/tools.py`), so what the phone says and what hands says aloud are one sentence, written once. Its name is the session's, as `identifier` speaks it, the project left to the group heading. What it adds under the sentence comes from the same `Session`, never from the sentence:

| the session is | the card says | under it |
|---|---|---|
| at a dialog hands holds | `waiting for permission to use Bash`, `waiting for the user to answer its question`, `waiting for the user to approve its plan` | the tool's input, the question and its options, or the plan's first lines, from the `Held`; how long it has waited, by the Mac's clock the read carries, never the phone's |
| Claude Code says `waiting`, no hook held | `waiting at a dialog: permission prompt` | nothing hands can read: `Show screen` is the way to it |
| working, with a call made | `working; the last thing it set out to do: run the test suite` | how long the turn has run |
| working, before its first call | `working` | how long the turn has run |
| at its prompt with subagents out | `idle, with 2 subagents it started in the background still working` | |
| at its prompt with a background shell | `idle, with a shell command it started in the background still running` | |
| no status read yet | `not reported yet` | |
| idle, and folded | `idle` | when Claude Code set it idle |
| ended, and folded | `ended` | when its transcript last changed |

**Grouping is by `Membership.cwd`**, the directory the session started in, which a `cd` and a worktree entered later do not move (`docs/architecture.md`, "Hooks carry the moment"), labelled by its last path element, as `identifier` names a session when it speaks (`src/hands/core/session.py`). One notion of a session's project for the voice and the page. The trap: a session started at a repository's root that then entered a worktree groups under the root; its name still says its work.

**The fold.** A session folds when Claude Code says it is at its prompt (`Idle` with status `Idle`), it holds no dialog, no subagent of its started in the background is still out, and no words wait in its queue; and every ended session folds. Everything else is never folded: a session at a dialog hands holds, however it is asked; one Claude Code says is `waiting`, whether or not hands holds its hook; one working; one whose only work is subagents in the background; one with a background shell still running; one whose status has not been read yet; one with words queued behind its last turn. Each is read off `Session` as the registry holds it, by the rules `_spoken_state` and `delegating` already apply. The fold line counts what it holds, `1 idle · 4 ended`, and a tap opens it. On the measured day that leaves 12 cards open across the groups and folds 30, without hiding any.

**Order.** Groups by the state of their most pressing card: a group with a waiting session first, then working, then the rest by the newest activity in them. Within a group, waiting first, then working, then the fold; inside the fold, idle by when Claude Code set it idle (`Idle.stamp`), newest first, then ended by the transcript's modification time, newest first.

### The screen

`Show screen` asks hands to read the pane the session runs in, as `read_screen` does (`tmux.shown`: `capture-pane -p`, the blank rows below the last drawn line left off). One read per tap, never a loop. The text is shown with no markup, in a block that scrolls on its own and wraps nowhere. A session in no tmux pane, and an ended one, has no screen hands can read, and the button is replaced by the sentence `read_screen` returns for it. Sessions hands starts are always in a pane (`start_session`).

## Between the page and hands

Every request carries the phone's key through `keyed` (`src/hands/voice/keyed.py`), and each is a unit of work with its own event. Every read that waits is a `Read` as every page's are (`hands.voice.longpoll.Read`): the version, how long a read may wait, the Mac's clock, and what is shown. The page asks through `hands.js`, the one script every phone page asks hands through.

| route | answers | |
|---|---|---|
| `GET /sessions` | the groups, each its cards and its fold, with the version of the list | waits for a change: the page says the version it has, and hands answers when the list differs or after 25 s |
| `GET /sessions/<id>/calls` | the newest 80 lines of the session's timeline, headings included | the same wait |
| `GET /sessions/<id>/calls?before=<record>` | the 80 lines before that record | answers at once; a record the file no longer holds is refused in words |
| `GET /sessions/<id>/agents/<agent>/calls`, with or without `before` | the subagent's timeline, the same two ways | the same |
| `GET /sessions/<id>/screen` | the pane's text, or why there is none | answers at once |

What one page of the timeline carries:

| field | of | |
|---|---|---|
| `at` | a line, a heading | when Claude Code wrote the record, as its `timestamp`; the page draws the hour and the day |
| `record` | a line, a heading | the record's uuid, which `before` names |
| `tool` | a line | the tool's name as Claude Code gave it |
| `what` | a line | the one line from the table above |
| `outcome` | a line | `ok`, `failed`, `running`, or `cut off` |
| `agent`, `calls`, `reported` | a line that started a subagent | its id, how many calls its transcript holds, whether it has reported |
| `kind` | a heading | `asked`, `notified`, `command`, or `shell`, as `Opening`'s variants |
| `text` | a heading | the opening's first line, cut to fit |
| `before` | the page | whether lines lie before its oldest, so the page knows when it has read to the session's first |
| `ended` | the page | whether the session has ended, so no line of it is drawn `running` |

## On the Mac

- **A route module of its own, as the brain's panel is.** `hands.voice.sessionspage` takes a protocol over what it reads, so the page and the tests see the sessions through one interface and nothing in it reaches a session. It is added to `phone_pages` (`src/hands/voice/phonepage.py`) beside the others, under the same key, its page served `no-store` like theirs.
- **One sentence for the state, written once.** `_spoken_state` becomes the function the brain's listing and the page both call. The page takes the dialog's content from the `Held` it describes, and does not parse the sentence.
- **One fold, held while a page watches.** The backfill's `read_transcript` (`src/hands/sessions/backfill.py`) folds a file whole; its own comment puts the largest file on this machine, 30 MB, at 0.19 s. A read that waits looks every 0.25 s (`POLL_SECONDS`), so a whole fold at every look would cost most of a core for one open page. The fold gains the form the tail already has: a held `Turning` and its openings, fed only the bytes appended since, by `_appended` and `Position` from `src/hands/sessions/tail.py`, and read again from the start when the file is another or shorter, by the tail's own `Restarted` rule. The watcher keeps one such fold per transcript a page has read in the last minute, and a look costs a `stat` while nothing was appended. It is a reading of the file, let go when nobody watches, never written anywhere: the file stays the one source. `read_session` and `read_turn` read through the same fold, so the voice and the page can never disagree about what a call was.
- **Ended sessions come from the registry.** `Sessions.membership` answers for any session the run heard of, ended or not; the page lists `Gone` beside `Session`, as `listing` does.
- **The lines come from one table.** `doing`'s table in `src/hands/core/progress.py` gains `Skill`, `SendMessage`, `AskUserQuestion`, `ExitPlanMode`, and the path with the `cwd` taken off, and the page's line and the voice's clause are read off the same row.
- **The screen is `read_screen`'s.** The pane lookup and `tmux.shown` are what the tool calls; the page's route calls the same two, and the tool's sentences are the page's.
- **Every read is an event.** `sessions.read`, with how many groups and cards, how many folded, which versions the page held and was given, and whether it waited; `sessions.calls`, with the session, whether it was a page back or the newest, how many lines, the bytes folded for it; `sessions.screen`, with the session, the pane, how many lines, or why it could not.

## What can go wrong

- **The screen shows what is on the terminal, secrets included, and the timeline shows every command a session ran.** The page is behind the phone's key, as the conversation page is, and that page already shows everything hands said and every tool it called. The key travels in a URL's fragment, so a browser does not send it with the page request. Anyone who has it already reads the conversation and sets hands' voices and options.
- **A session that does not join hands is not on the page.** The registry is membership by files (`docs/architecture.md`, "Sessions"). A Claude Code held at the workspace-trust dialog has no membership and no card. It has a pane, and the user sees it at the desk.
- **A restart of hands forgets ended sessions.** Their transcripts are still on disk and their cards are gone. Said above, and Brandon's to overrule.
- **A state hands has no words for.** A dialog it let go of, or one newer than hands' model of the screen, shows as `working` or `idle` while the terminal holds a question. `Show screen` is the way out: it is why that button exists. The timeline helps too: a session stuck at such a dialog has a last line that is minutes old.
- **A file written anew under the same id.** The fold is read again from its start, the page's cursor is refused, and the page reads from the newest. Nothing is shown twice.
- **A busy list.** The version is the hash of what the page draws, not of the registry, so a change the page does not draw wakes nothing. A card's `asked 2 min ago` is timed on the page from the stamp hands sent, not sent again each minute.
- **A phone behind a tailnet certificate that lapsed.** As for every phone page: hands serves its own certificate under a LAN address, and the phone is asked once to accept it.

## Why this and not the alternatives

- **The full view on `tmux-control-mode-js`.** Its read-only pane mirror lives in `examples/web-multiplexer/`, not the published package; it needs a Node process beside hands' Python one, a control client attached to the user's tmux server, and a second place that decides who may look at a terminal. On a phone 390 CSS pixels wide, a 148-column pane is 2.6 pixels a column. And now that the requirement is history, the terminal is the wrong source: it shows what is on screen, and the calls from while Brandon was away are in the transcript, not the scrollback. The mirror would answer "what is it drawing now" and nothing of "what did it do". Where the whole terminal is wanted on the phone, an SSH app and `tmux attach` are that, untried here.
- **History from the audit log.** Its hook events name no tool and no input, count subagent calls with the session's (2998 against 699), and it rolled in 6h22m on this machine. It is not a source of tool-call history.
- **A summary sentence per finished turn, with steps only for the running turn**, the first draft's detail. That is `read_session`'s shape, made for a voice that cannot read 200 lines aloud. A phone can, and Brandon asked for the calls. Summaries are not on the page.
- **A live stream of the screen text.** `capture-pane` once a second is the full view's cost without its fidelity. A snapshot on a tap is the cheapest read that answers "what is on the terminal".
- **A byte offset as the cursor.** Answered under "The timeline": the fold needs the result after the call, and `read_turn` already pages by record.
- **Subagent calls interleaved by time.** Three calls in four would be subagents'; the session's own work would be the quarter between them.
- **An index or a database of calls.** The transcript is the record; a second copy is one the session writing the file can make wrong, which is why `read_transcript` folds again each time rather than keeping one (`src/hands/sessions/backfill.py`). The watcher's held fold is let go the moment nobody reads it and is never written.
- **Finding ended sessions the run never knew.** The audit's `transcript` facts last as long as its segments, 6h22m here, which is no better than the run's memory. A listing of the project directory the live session's transcript sits in would find a chain's predecessors across a restart, keyed on Claude Code's layout, `~/.claude/projects/<slug>/<id>.jsonl`, which the measurement saw and hands reads nowhere today. If a restart losing a chain's earlier cards turns out to matter, that listing is the piece to add, bounded to the newest few files of each group, and nothing else in this design moves.
- **Approve and deny on the card.** Answering a dialog is a write into a session, which hands' typing path decides. It is not part of seeing.
- **Add sessions to the brain's panel on the phone page.** The brain's panel shows only calls in flight (`src/hands/voice/brainpage.py`, `src/hands/core/brainwork.py`: `calls` is empty while idle), so a call appears and vanishes as it ends; that is the "things appear, disappear suddenly" Brandon sees, read off the code, not tested. The sessions are a page of their own, so the call's page does not grow with every session.

## What the measurements settled

1. **The transcript is the right and sufficient source of call-level history, live and ended.** Confirmed: every call has its result and its time, failure is marked, the file persists past the end, and the audit log has none of it. One change: a session's subagent calls are not in its file, and come from the subagent transcripts beside it, which hands already reads.
2. **A page can page backwards cheaply with a stable cursor while the end grows.** Confirmed, with the cursor a record uuid rather than a byte offset, and with the fold held while a page watches rather than run whole at each look: 0.19 s for 30 MB, by the backfill's own comment, is too much at four looks a second.
3. **The lighter view holds.** Confirmed, and the history requirement decides it harder than the draft did: a terminal mirror has no history.
4. **Folding idle sessions and grouping by worktree work on this data.** Confirmed: 30 of 42 fold, the five-session chain in `phone-wake-bw7` is one group. One change: ended cards last as long as the run, so a restart cuts the chain's earlier links from the page.
5. **Brandon, back from the desk, learns what the sessions did call by call and what is stuck on them.** Yes: the waiting cards are first and never folded, each saying what it asks and for how long; the first line under the first heading is each session's last act, and the lines run back as far as he scrolls, from while he watched and while he did not alike. What falls short, said plainly: subagents' calls are behind a tap; an ended session is remembered only until hands restarts; a session that never joined hands has no card.

## How it is tested

No rig, and nothing measured here beyond what the brief holds.

- **The lines** are built in process from calls as `Turning` folds them (`tests/test_steps.py`, `tests/test_progress.py` build calls today): one test row for each row of the table, the four outcomes, a prompt heading of each kind, and the `cwd` taken off a path.
- **The fold** runs on a transcript written to a temporary file and appended to between reads, beside `tests/test_backfill.py` and `tests/test_tail.py`: appended bytes fold on; a file replaced by a shorter one is read again from its start; a page before a record holds the 80 before it and its calls carry their results; a cursor the file lacks is refused.
- **The cards and the fold** are built in process from sessions in each state, on the types the registry holds (`tests/test_list_sessions.py` builds sessions for `describe_listing`): one row for each never-folded state, the idle and ended folds, the grouping by `cwd`, and the order.
- **The routes** run on aiohttp's test client under the phone's key, with a fake registry and a fake tmux, beside `tests/test_brainpage.py`: a request without the key is refused, the newest read waits and is answered on an appended call and on the 25 s bound, a page back answers at once, and a change the page does not draw wakes nothing. The wait is forced with a gate in the fake, never timed.
- **The page** runs in Chrome under Playwright against the routes hands serves, on fakes (`tests/test_pages.py`): a fold opens on a tap, a card turns from `working` to waiting with no tap, the timeline pages back on scroll, a new call appears at the top while a reader scrolled down sees a count and nothing moves, a subagent opens under its line, `Show screen` shows the text.
- **Brandon's phone** is the one check no test here can make, and is its own ticket.

## Where the build lands

| file | what changes |
|---|---|
| `src/hands/voice/sessionspage.py` (new) | the routes, the groups and cards, the fold, the timeline pages, the waits, the events |
| `src/hands/voice/sessions.html` (new) | the page: the groups, the fold, the timeline, the new-calls count, the subagent unfold, the screen |
| `src/hands/voice/hands.js` | nothing new if its one loop serves every read; a change here is a finding |
| `src/hands/voice/phonepage.py` | the page and its routes added to `phone_pages` |
| `src/hands/voice/tools.py` | `_spoken_state` and the screen read move to where the tools and the page both call them; `read_session` and `read_turn` read through the held fold |
| `src/hands/sessions/backfill.py` | the fold folds appended records onto a held reading, and is read whole only the first time |
| a new module beside `src/hands/sessions/backfill.py` | the watcher: one held fold per transcript read lately, let go when nobody reads it |
| `src/hands/core/progress.py` | the table gains `Skill`, `SendMessage`, the dialogs, and the path with the `cwd` taken off |
| `src/hands/daemon/cli.py` (`hands phone`) | the address of the page is printed beside the others |
| `docs/architecture.md` ("Transcripts: the live tail, and backfill", "The audio side"), `README.md`, `docs/testing.md` | the page, the held fold, the lines, that the page changes nothing |

## The tickets

Seven tickets under epic `promptctl-phone-app-9d9y`, in one lane, `phone-sessions`, behind this design's own ticket. Each is built and merged before the next starts.

1. `promptctl-phone-app-9d9y.0q9`: one function says a session's state, for the brain's listing and for the page: "The cards, the fold and the groups", "On the Mac".
2. `(to file)`: one table says what a call sets out to do, for the voice and the page, for every tool Claude Code calls, dialogs and skills included: "The timeline", the table.
3. `promptctl-phone-app-9d9y.tpc`: the sessions page lists every session this run of hands knows, grouped by project, the ones waiting on Brandon first and never folded, idle and ended ones folded into a line that opens on a tap, and each card turns as its session does: "The cards, the fold and the groups", "Between the page and hands".
4. `(to file)`: a card opens on the session's timeline, every tool call read from its transcript, newest first under the prompt that started its turn, paged back as far as Brandon scrolls, new calls arriving at the top without moving what he reads, the same for a live session and an ended one: "The timeline", "On the Mac".
5. `(to file)`: a call that started a subagent opens on that subagent's own calls, read from its own transcript: "The timeline".
6. `promptctl-phone-app-9d9y.vkc`: a card's `Show screen` shows the session's terminal as text, once for each tap: "The screen".
7. `promptctl-phone-app-9d9y.bqh`: Brandon's phone shows him what a session he sent off did while he was away, call by call, and what it is waiting on: carries the `external` label, since its event is outside the repository, so nothing waits on it.

Between them they build every row of "Where the build lands".

## Sources

- Measurements on Brandon's machine, 2026-11-24, as given to this revision: the audit log's size, span and event shapes; the 42 sessions of the window and what each did; the membership files; seven transcripts' sizes, `tool_use` and `tool_result` counts, `is_error` counts, Bash `description` counts, `isSidechain` and `compact_boundary` counts, record types and `sessionId` values. Not repeated here.
- `src/hands/sessions/turning.py`, `backfill.py`, `tail.py`, `transcript.py`, `subagents.py`, `registry.py`, `membership.py`, `liveness.py`; `src/hands/core/session.py`, `turn.py`, `steps.py`, `progress.py`, `reducer.py`, `events.py`; `src/hands/voice/tools.py`, `longpoll.py`, `brainpage.py`, `conversationpage.py`, `phonepage.py`, `keyed.py`, `hands.js`; `docs/architecture.md` ("Hooks carry the moment; the transcript carries the record", "Transcripts: the live tail, and backfill", "Sessions: membership from files, state from events, liveness from the OS")
- `~/code/tmux-control-mode-js`, read for the first draft at its HEAD of 2026-11-16: `README.md`, `examples/SHOWCASE.md`, `examples/web-multiplexer/server/mirror-registry.ts`, `src/connectors/websocket/server.ts`
- `design-docs/phone-wake-word/design.md` (PR 556), for how a phone page is served and what it may keep
