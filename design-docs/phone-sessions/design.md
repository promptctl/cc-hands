# The sessions hands sends off, on the phone page

Written 2026-11-23, revised 2026-11-24, from a conversation between Brandon and Claude. Ticket `promptctl-phone-app-9d9y.68y`.

## The idea in one paragraph

The phone gains a page of the sessions hands knows, one card for each, grouped by the project each works in. A card says what the session is doing in hands' own words: `working; the last thing it set out to do: run the test suite`, `waiting for permission to use Bash`, `idle`. Sessions waiting on Brandon are never folded; idle and ended ones fold into one line per group until tapped. Tapping a card opens the session's timeline: every tool call it made, read from Claude Code's own transcript, newest first, each line the time, the tool, one line of what it set out to do, and a mark for one that failed or is still running, grouped under the prompt that started each turn. The page reads back as far as Brandon scrolls, and the calls from while he watched and from while he did not appear the same way, because both are the file's. Sessions that ended keep their timeline for as long as this run of hands remembers them, because the transcript outlives the session. A card's `Show screen` opens the session's terminal, live: its tmux window drawn in the page by `tmux-control-mode-js`'s window terminal on xterm.js, sized to the phone for as long as the phone has it open, and typed into from the phone's keyboard as from the desk's. That is the page's second way into a session. The talk button and the conversation page's box already send Brandon's words to the brain's Claude Code; the terminal reaches every session the page lists, its dialogs included.

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
 │  the terminal, live, at the phone's size   │  wss      │  the bridge: one Node process, a tmux -C │
 │  keys typed ─────────────────────────────► ┼─…/terminal┤  client on the session's tmux server,    │
 │  the window drawn ◄────────────────────────┼───────────┤  reached over a socket in hands' home    │
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
8. Another card reads `waiting at a dialog: permission prompt`, with nothing under it: Claude Code's status says it waits, and hands holds no hook for it. He taps `Show screen`. The terminal opens over the page, the dialog drawn as it is on his desk and reflowed to the phone's width, and his desk's view of that window shrinks to the same size. He taps the pane, the keyboard comes up, he types `2`; the session goes on and the card reads `working`. He taps `Close`, the desk's window is its own size again, and the card's timeline is where he left it.

## Where this design departs from the words of the request

Each of these is Claude's, and Brandon's to overrule.

**Every session hands knows is shown, not only the ones hands sent off.** hands keeps no record of who started a session or who sent it a turn: `Session` (`src/hands/core/session.py`) has none. A filter would be a second record of a fact the registry never held, and a session Brandon started at the desk is as worth a glance from the phone. Waiting first, and folding, keep the sessions hands sent off from being buried.

**Ended sessions are the ones this run of hands remembers.** The registry keeps a session that ended as `Gone`, with its membership and so its transcript path, for the rest of the run (`src/hands/core/reducer.py`, `src/hands/sessions/registry.py`); nothing prunes it. The page lists those. A restart of hands forgets them, and a chain's earlier links go from the page with it; their transcripts stay on disk. Brandon said the history he wants is the calls, not what comes after a session ends, so nothing is built to find a transcript the run never knew: not the audit log, whose `transcript` facts last only as long as its segments, which on this machine was 6h22m, and not a listing of Claude Code's project directories, which would show every session ever run there. "Why this and not the alternatives" names what would change this.

**Subagent calls are folded under the call that started them, not interleaved.** By the audit's count, session 4d1d1d78 made 2998 tool calls where its own transcript holds 699: three calls in four were subagents'. Interleaved by time they would bury the session's own work. Each subagent is one line, with how many calls it made, and opens on a tap.

**Idle folds every session at its prompt, including ones that worked.** The request says idle sessions are folded away. Of the 42 sessions in the measured window, 30 never ran a prompt and 12 did; a session that ran a turn an hour ago and sits at its prompt is as quiet as one that never did. The fold line counts them, and one tap opens them, so nothing is hidden, only quiet.

**The terminal's keys go to the pane as a keyboard's do, not through hands' send.** hands types into a session by one write, the text pasted and Return, and refuses the write while the session is at a dialog (`core.reach.writer`; "Typing into a session", `docs/architecture.md`), because a paste landing in a permission prompt is taken as "Yes" by a sender who cannot see the prompt. A person at the terminal sees it, and answering it is what the terminal is for. So the terminal sends each key as it is typed, through the tmux control client, the way the desk's keyboard reaches the pane, and none of hands' send rules apply to it; those rules stay for the sends that cannot see the screen. Brandon is at one keyboard at a time, the desk's or the phone's, so no rule is built for two.

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

### The terminal

`Show screen` on a live card opens the session's terminal: the tmux window the session runs in, drawn live in the page by `mountWindowTerminal` from `@promptctl/tmux-control-mode-js/window-terminal` (0.4.0) on xterm.js 6, every pane of the window in tmux's layout, the window's scrollback (`history`, 2000 lines by default) copied from tmux and the bytes after it streamed, and what is typed in a pane sent to that pane. It is the library's terminal as its README describes it ("A window in the browser"); the page adds nothing of its own to drawing, sizing, scrollback or keys.

**What Brandon sees and does.** The terminal covers the page: a bar with the session's name, the card's state sentence and `Close`, and the window below it, filling the rest of the phone's visual viewport. A tap on a pane puts the phone's keyboard in it (a click, or `terminal.focus(paneId)`, picks the pane the page's keys go to; a window hands started has one pane). What he types goes to that pane as keys, each sent as it is typed; Claude Code's dialogs are answered with the keys they ask for, as at the desk. `Close` disposes the terminal and returns him to the card's timeline, which went on reading behind it.

**Sizing is the library's, at its default: the phone's box declares the window's size.** `sizing: { window: "box", font }`, the default, has the element's box declare this client's size for the window (`refresh-client -C CxR` and `-C @W:CxR`), and tmux resizes the window to it. The page gives the element a definite size, the visual viewport's less the bar, so the box the library observes (a `ResizeObserver` on the element, `src/window-terminal/render/window-renderer.ts`) is the phone's screen, and while the keyboard is up, the screen above the keyboard. At 14 px, the default font, a 390 px phone gives about 46 columns, and in portrait about 45 rows with the keyboard down and 20 with it up: arithmetic on a 14 px monospace cell of about 8.4 by 17 px, measured on no phone. Claude Code reflows to that width as it does to any terminal's. What the library does at each moment, read from `src/window-terminal/terminal.ts`, `sizing/solve.ts` and `sizing/client-size.ts`, and measured here on tmux 3.8a, Brandon's version, under `window-size latest`, his setting, on a throwaway tmux server with a 148-column client attached as his desk is:

- *On attach.* The terminal declares once the window is `live`, never from a stale picture, and only once the box has an area (a box of nothing is `hold`, not a claim). The window takes the phone's size at once, for every client: the desk shows it at the phone's size for as long as the phone has it open. Measured: the desk's keystrokes in that window, and selecting it, leave the phone's size in place; the claim holds until withdrawn. Measured too: at the moment of the declaration the desk's current window, whichever it is, takes the phone's overall size as well, and returns to the desk's size at the desk's next keystroke in it. The library declares the overall size beside the window's on purpose (`client-size.ts`: without it, the session's other windows would read a control client's 80x24).
- *On rotation, and as the keyboard opens and closes.* The box changes, the observer fires, the solver runs again, and the declarer sends the new grid: one in flight at a time, the latest winning, a grid equal to what tmux holds not sent again. Measured: a second declaration moves the window to the new size at once. That the visual viewport shrinks for the soft keyboard is the browser's doing, and is not checked here on a phone.
- *On `Close`, and on the page going away.* `dispose()` clears this window's size (`refresh-client -C @W:`). tmux has no command that withdraws a client's overall size, so the bridge takes the shared control client out of sizing (`ignore-size`) once no page on it declares one (README, "Things to know"; `src/client-size.ts`), which with one phone is at once. A page that dies without `dispose()`, a phone that slept or left the tailnet, is a connection closed at the bridge, which withdraws its claims the same way; the bridge's heartbeat, a ping every 30 s answered within 10 s (`src/connectors/websocket/server.ts`), is how long the window stays phone-sized after the phone goes dark. Measured: the window the phone left returns to the desk's size at once when the desk is viewing it or has typed in it, and otherwise when the desk next selects it. Brandon confirmed the desk's resize on detach is the behaviour he wants.

**What the bar says in each of the library's phases** (`terminal.onState`; `src/window-terminal/engine/window-model.ts`): `connecting` while the socket and the control client come up; nothing while `live`; `stale`, with since when and the library's reason, while the connection is down and the picture is the last drawn, which the library redraws whole as it reconnects (`reconnect: { maxAttempts: Infinity }`); `gone` when tmux no longer has the window, which an ended session's is once its `claude` exits. A card for a session in no tmux pane has no button, and in its place the sentence `read_screen` gives such a session (`src/hands/voice/tools.py`); a `Gone` card has none either, since the pane in front of its pid, if the process still runs, is another session's now (a `/clear` is a new session in the same pane). Sessions hands starts are always in a pane (`start_session`).

**One person, one keyboard.** Brandon is at the desk or at the phone, never both; the terminal is his one terminal shown in a second place, and no rule is built for two viewers or two typists.

## Between the page and hands

Every request carries the phone's key through `keyed` (`src/hands/voice/keyed.py`), and each is a unit of work with its own event. Every read that waits is a `Read` as every page's are (`hands.voice.longpoll.Read`): the version, how long a read may wait, the Mac's clock, and what is shown. The page asks through `hands.js`, the one script every phone page asks hands through; the terminal's socket is the one request a page makes that is not `hands.js`'s, since what travels on it is the library's bridge protocol, spoken by its `WebSocketTmuxClient`.

| route | answers | |
|---|---|---|
| `GET /sessions` | the groups, each its cards and its fold, with the version of the list | waits for a change: the page says the version it has, and hands answers when the list differs or after 25 s |
| `GET /sessions/<id>/calls` | the newest 80 lines of the session's timeline, headings included | the same wait |
| `GET /sessions/<id>/calls?before=<record>` | the 80 lines before that record | answers at once; a record the file no longer holds is refused in words |
| `GET /sessions/<id>/agents/<agent>/calls`, with or without `before` | the subagent's timeline, the same two ways | the same |
| `GET /sessions/<id>/terminal` | a WebSocket, carrying the library's bridge protocol between the page's `WebSocketTmuxClient` and the bridge's connection for this session's window | the key travels in the `Sec-WebSocket-Protocol` header, the one header a browser's WebSocket can set, and `keyed` refuses without it before upgrading; a session in no pane, and a Mac with no `node`, are refused in words |
| `GET /terminal/<version>/<file>` | the library's and xterm.js's files, as published | public, and kept by the browser under names that carry the version, as the wake word design serves its runtime |

A card carries `window`, the id of the tmux window its process runs in, or why there is none, read with the listing as `list_sessions` reads it (`tmux.panes`, `_in_pane` in `src/hands/voice/tools.py`). The pane lookup gains the window's id, `#{window_id}`, beside the index it lists today (`src/hands/sessions/tmux.py`; `Pane.window` in `src/hands/core/tmux.py`), since `mountWindowTerminal` names a window by its id.

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

- **A route module of its own, as the brain's panel is.** `hands.voice.sessionspage` takes a protocol over what it reads, so the page and the tests see the sessions through one interface; the one thing in it that reaches a session is the terminal's socket, and it reaches one only through the bridge. It is added to `phone_pages` (`src/hands/voice/phonepage.py`) beside the others, under the same key, its page served `no-store` like theirs.
- **One sentence for the state, written once.** `_spoken_state` becomes the function the brain's listing and the page both call. The page takes the dialog's content from the `Held` it describes, and does not parse the sentence.
- **One fold, held while a page watches.** The backfill's `read_transcript` (`src/hands/sessions/backfill.py`) folds a file whole; its own comment puts the largest file on this machine, 30 MB, at 0.19 s. A read that waits looks every 0.25 s (`POLL_SECONDS`), so a whole fold at every look would cost most of a core for one open page. The fold gains the form the tail already has: a held `Turning` and its openings, fed only the bytes appended since, by `_appended` and `Position` from `src/hands/sessions/tail.py`, and read again from the start when the file is another or shorter, by the tail's own `Restarted` rule. The watcher keeps one such fold per transcript a page has read in the last minute, and a look costs a `stat` while nothing was appended. It is a reading of the file, let go when nobody watches, never written anywhere: the file stays the one source. `read_session` and `read_turn` read through the same fold, so the voice and the page can never disagree about what a call was.
- **Ended sessions come from the registry.** `Sessions.membership` answers for any session the run heard of, ended or not; the page lists `Gone` beside `Session`, as `listing` does.
- **The lines come from one table.** `doing`'s table in `src/hands/core/progress.py` gains `Skill`, `SendMessage`, `AskUserQuestion`, `ExitPlanMode`, and the path with the `cwd` taken off, and the page's line and the voice's clause are read off the same row.
- **The bridge: one Node process, hands' child, started for the first terminal and ended a minute after the last closes.** A script hands ships, `src/hands/voice/terminalbridge.mjs`, runs `createWebSocketBridge` from `@promptctl/tmux-control-mode-js/websocket/server` and listens on a Unix socket in hands' home, `<home>/terminal/bridge.sock`, readable by the user alone. Each connection hands opens to it is one browser's terminal: hands sends one line first, the tmux server's socket and the window, then pumps the WebSocket's frames both ways, length-prefixed, and the script wraps the stream in the bridge's `ServerWebSocketLike` (`src/connectors/websocket/types.ts`), a structural interface of which the `ws` package is one implementation, so the bridge needs no `ws` and listens on no port. `createClient` keeps one `TmuxClient(spawnTmux(["-S", socket, "attach-session"]))` per tmux server, shared by every page on it, as the example's `DedicatedControlClient` keeps one and replaces it once its tmux has gone (`examples/web-multiplexer/server/control-client.ts`); `authenticate` accepts, since hands checked the key; `authorize` is left at its default, allow all, because the key already types into the brain. It is started outside `Children`'s waits, as the brain's process is, and named as `Children`'s one other exception (`src/hands/sessions/child.py`); the library's `engines` ask for Node 20 or later, and `node` is looked for on the PATH hands was started with, as tmux is. hands needs no Node today, and this is the one feature that does: a `Show screen` on a Mac with none is refused in words, `the terminal needs node on the PATH, and none is`.
- **The two packages are fetched once, pinned and held to digests, and served from hands' home**, as the wake word design serves its runtime and `dtln.py` its models: `@promptctl/tmux-control-mode-js` 0.4.0 (1.9 MB, no runtime dependencies of its own) and `@xterm/xterm` 6.0.0 (`lib/xterm.mjs` and `css/xterm.css`), each the npm registry's tarball, unpacked under `<home>/terminal/<name>@<version>/`. Node imports the first from there; the page imports `window-terminal` and `websocket` from the same files served at `/terminal/<version>/…`, with an import map that points `@xterm/xterm`, the library's one bare specifier, at the served `xterm.mjs`. No `npm` runs. `hands phone` fetches them as it makes the key, so a phone never waits on a download. Unverified: the published build's import form, since the checkout read here has no built `dist/window-terminal`; the import map is right for a bare specifier and needless for a relative one.
- **The upgrade route is the one key check.** `keyed` gains the one more place a key can arrive, the `Sec-WebSocket-Protocol` header of an upgrade request (the key is hex, `clock.token`, so a valid subprotocol token), and stays the one place a page's route checks a key (`src/hands/voice/keyed.py`); the bridge's `authenticate` trusts what hands connected. hands looks the pane up again at the upgrade, so the window is the one the process is in now.
- **Every read is an event.** `sessions.read`, with how many groups and cards, how many folded, which versions the page held and was given, and whether it waited; `sessions.calls`, with the session, whether it was a page back or the newest, how many lines, the bytes folded for it; `sessions.terminal`, with the session, its tmux socket and window, how long it was open, the bytes each way, and how it ended: closed, the phone gone, the bridge gone, or refused and why; `terminal.bridge`, each start and end of the Node process, and why.

## What can go wrong

- **The terminal shows what is on the terminal, secrets included, and types into it; the timeline shows every command a session ran.** The page is behind the phone's key, as the conversation page is, and that page already shows everything hands said and every tool it called, and sends what is typed in its box to the brain. The key travels in a URL's fragment, so a browser does not send it with the page request. Anyone who has it already types into the brain's Claude Code, stops its turn, and sets hands' voices and options; the terminal adds the sessions beside the brain, under the same one key.
- **The desk's window is the phone's size while the phone has it open.** Designed, and measured in "The terminal": the desk's current window shrinks too at the moment the phone attaches, until the next key in it, and the window the phone left returns to the desk's size when the desk is in it or next selects it. Brandon has accepted the cost, and is at one of the two at a time.
- **The phone goes dark with the terminal open.** The bridge's heartbeat ends the connection within 40 s and withdraws the page's claim, and the window returns to the desk's size then. The page brought back opens a new terminal.
- **The bridge or its tmux client goes away.** The page shows `stale`, and `gone` once the library gives the window up; hands' `terminal.bridge` event says what ended and why. A Mac with no `node` refuses the tap in words, and the cards and the timeline are untouched by it.
- **A key typed by accident.** The keyboard is up only after a tap on the pane, and a key typed is a key in the session, as at the desk. Nothing undoes it; the timeline shows what the session did with it.
- **A session that does not join hands is not on the page.** The registry is membership by files (`docs/architecture.md`, "Sessions"). A Claude Code held at the workspace-trust dialog has no membership and no card. It has a pane, and the user sees it at the desk.
- **A restart of hands forgets ended sessions.** Their transcripts are still on disk and their cards are gone. Said above, and Brandon's to overrule.
- **A state hands has no words for.** A dialog it let go of, or one newer than hands' model of the screen, shows as `working` or `idle` while the terminal holds a question. `Show screen` is the way out, and it is why that button exists: the question is on the terminal, and the keys that answer it are the phone's. The timeline helps too: a session stuck at such a dialog has a last line that is minutes old.
- **A file written anew under the same id.** The fold is read again from its start, the page's cursor is refused, and the page reads from the newest. Nothing is shown twice.
- **A busy list.** The version is the hash of what the page draws, not of the registry, so a change the page does not draw wakes nothing. A card's `asked 2 min ago` is timed on the page from the stamp hands sent, not sent again each minute.
- **A phone behind a tailnet certificate that lapsed.** As for every phone page: hands serves its own certificate under a LAN address, and the phone is asked once to accept it.

## Why this and not the alternatives

- **A text snapshot of the pane, `capture-pane -p` once for each tap**, this document's design before this revision. Brandon: any `Show screen` shows a live terminal. A snapshot of a dialog is a dialog that cannot be answered from the phone, and the state hands has no words for is a dialog. The snapshot stays the brain's, as `read_screen`.
- **A read-only terminal.** Brandon: "fairly useless and not needed". The library has no read-only form either: xterm's `disableStdin` hides the keyboard, and nothing refuses keys.
- **A terminal that follows tmux's size** (`sizing: { window: "tmux" }`), leaving the desk's window as it is. On a phone 390 CSS pixels wide a 148-column pane is 2.6 pixels a column, and pinch-zoom makes it a pan over a wide picture. Brandon: a terminal not resized to the phone is useless to him. The desk's window at the phone's size is the designed behaviour, and it returns when the phone leaves.
- **The bridge on a port of its own, as the example runs it** (`ws://localhost:8080`, `examples/web-multiplexer/server/bridge.ts`). A second door into the user's terminals, with a second check of who may open it. The socket in hands' home has one door, hands' upgrade route, and the file's mode keeps every other user out.
- **The terminal's keys through fritter, as hands' sends go.** fritter types a paste and a Return; a terminal is keys, Escape and the arrows and `2` among them, and the refusals around a send are for a sender who cannot see the dialog.
- **A control-mode client in Python, inside hands.** The protocol is the library's, and the terminal in the browser speaks the library's bridge protocol; a Python bridge would be a second implementation of both, drifting from the one the browser side is tested against.
- **An SSH app and `tmux attach`.** A second login to the Mac from the phone, outside the phone's key, attaching to a whole tmux session rather than one session's window, and no card to open it from.
- **History from the audit log.** Its hook events name no tool and no input, count subagent calls with the session's (2998 against 699), and it rolled in 6h22m on this machine. It is not a source of tool-call history.
- **History from the terminal.** The scrollback is what was drawn, and the calls from while Brandon was away are the transcript's; the terminal answers "what is it drawing now", the timeline "what did it do", and the page has both.
- **A summary sentence per finished turn, with steps only for the running turn**, the first draft's detail. That is `read_session`'s shape, made for a voice that cannot read 200 lines aloud. A phone can, and Brandon asked for the calls. Summaries are not on the page.
- **A byte offset as the cursor.** Answered under "The timeline": the fold needs the result after the call, and `read_turn` already pages by record.
- **Subagent calls interleaved by time.** Three calls in four would be subagents'; the session's own work would be the quarter between them.
- **An index or a database of calls.** The transcript is the record; a second copy is one the session writing the file can make wrong, which is why `read_transcript` folds again each time rather than keeping one (`src/hands/sessions/backfill.py`). The watcher's held fold is let go the moment nobody reads it and is never written.
- **Finding ended sessions the run never knew.** The audit's `transcript` facts last as long as its segments, 6h22m here, which is no better than the run's memory. A listing of the project directory the live session's transcript sits in would find a chain's predecessors across a restart, keyed on Claude Code's layout, `~/.claude/projects/<slug>/<id>.jsonl`, which the measurement saw and hands reads nowhere today. If a restart losing a chain's earlier cards turns out to matter, that listing is the piece to add, bounded to the newest few files of each group, and nothing else in this design moves.
- **Approve and deny on the card.** Answering a dialog is a key in the terminal, one tap away, and Claude Code says there which keys it takes. Buttons on the card would be a second way to answer, with hands deciding what each key means, kept right against every dialog Claude Code adds.
- **Add sessions to the brain's panel on the phone page.** The brain's panel shows only calls in flight (`src/hands/voice/brainpage.py`, `src/hands/core/brainwork.py`: `calls` is empty while idle), so a call appears and vanishes as it ends; that is the "things appear, disappear suddenly" Brandon sees, read off the code, not tested. The sessions are a page of their own, so the call's page does not grow with every session.

## What the measurements settled

1. **The transcript is the right and sufficient source of call-level history, live and ended.** Confirmed: every call has its result and its time, failure is marked, the file persists past the end, and the audit log has none of it. One change: a session's subagent calls are not in its file, and come from the subagent transcripts beside it, which hands already reads.
2. **A page can page backwards cheaply with a stable cursor while the end grows.** Confirmed, with the cursor a record uuid rather than a byte offset, and with the fold held while a page watches rather than run whole at each look: 0.19 s for 30 MB, by the backfill's own comment, is too much at four looks a second.
3. **The lighter view holds, for history.** Confirmed for the timeline: a terminal has no history, and the calls from while Brandon was away are the transcript's. Changed for now: the terminal is on the page too, live and typed into, because a dialog is answered on a terminal and nowhere else; and tmux's sizing, measured here, gives the phone a window at its own size through the library's default, with no sizing of hands' own.
4. **Folding idle sessions and grouping by worktree work on this data.** Confirmed: 30 of 42 fold, the five-session chain in `phone-wake-bw7` is one group. One change: ended cards last as long as the run, so a restart cuts the chain's earlier links from the page.
5. **Brandon, back from the desk, learns what the sessions did call by call and what is stuck on them.** Yes: the waiting cards are first and never folded, each saying what it asks and for how long; the first line under the first heading is each session's last act, and the lines run back as far as he scrolls, from while he watched and while he did not alike. What falls short, said plainly: subagents' calls are behind a tap; an ended session is remembered only until hands restarts; a session that never joined hands has no card.

## How it is tested

No rig, and nothing measured here beyond what the brief holds.

- **The lines** are built in process from calls as `Turning` folds them (`tests/test_steps.py`, `tests/test_progress.py` build calls today): one test row for each row of the table, the four outcomes, a prompt heading of each kind, and the `cwd` taken off a path.
- **The fold** runs on a transcript written to a temporary file and appended to between reads, beside `tests/test_backfill.py` and `tests/test_tail.py`: appended bytes fold on; a file replaced by a shorter one is read again from its start; a page before a record holds the 80 before it and its calls carry their results; a cursor the file lacks is refused.
- **The cards and the fold** are built in process from sessions in each state, on the types the registry holds (`tests/test_list_sessions.py` builds sessions for `describe_listing`): one row for each never-folded state, the idle and ended folds, the grouping by `cwd`, and the order.
- **The routes** run on aiohttp's test client under the phone's key, with a fake registry and a fake tmux, beside `tests/test_brainpage.py`: a request without the key is refused, the newest read waits and is answered on an appended call and on the 25 s bound, a page back answers at once, and a change the page does not draw wakes nothing. The wait is forced with a gate in the fake, never timed.
- **The upgrade route** runs on the same test client with a fake bridge socket: an upgrade without the key in its subprotocol is refused with 401 before it is upgraded; one with it is upgraded, and its frames reach the fake and come back; a session in no pane, and no `node` on the PATH, are refused in words.
- **The bridge script** runs under `node` against the library's mock tmux client (`@promptctl/tmux-control-mode-js/mock`): a connection's first line picks the tmux socket and the window, frames pass both ways, a window tmux refuses ends the connection refused and the client does not retry, and the process exits a minute after its last connection closes, the minute forced by a clock the script takes, never waited. Marked `outside`, since it needs `node`.
- **The page** runs in Chrome under Playwright against the routes hands serves, on fakes (`tests/test_pages.py`): a fold opens on a tap, a card turns from `working` to waiting with no tap, the timeline pages back on scroll, a new call appears at the top while a reader scrolled down sees a count and nothing moves, a subagent opens under its line. The terminal's page test runs against a tmux server made for the test on a socket of its own, with a second client attached at 148 columns and the real bridge: `Show screen` draws the pane's text, a key typed on the page is in the pane (`capture-pane`), the window is the browser's size while the terminal is open and the other client's once it is closed, as measured here.
- **Brandon's phone** is the one check no test here can make, and is its own ticket.

## Where the build lands

| file | what changes |
|---|---|
| `src/hands/voice/sessionspage.py` (new) | the routes, the groups and cards, the fold, the timeline pages, the waits, the terminal's upgrade and its pump to the bridge, the events |
| `src/hands/voice/sessions.html` (new) | the page: the groups, the fold, the timeline, the new-calls count, the subagent unfold, the terminal mounted and disposed, the import map |
| `src/hands/voice/terminalbridge.mjs` (new) | the bridge: the socket in hands' home, the stream as the bridge's socket, one control client per tmux server, the minute's idle exit |
| a new module beside `src/hands/voice/dtln.py` | the two packages, fetched at their versions and held to their digests, served from hands' home; the bridge's life |
| `src/hands/voice/hands.js` | nothing new if its one loop serves every read; a change here is a finding |
| `src/hands/voice/phonepage.py` | the page and its routes added to `phone_pages`; the packages' routes |
| `src/hands/voice/keyed.py` | the key read from an upgrade's `Sec-WebSocket-Protocol` |
| `src/hands/voice/tools.py` | `_spoken_state` moves to where the tools and the page both call it; `read_session` and `read_turn` read through the held fold |
| `src/hands/sessions/tmux.py`, `src/hands/core/tmux.py` | the pane lookup carries the window's id |
| `src/hands/sessions/backfill.py` | the fold folds appended records onto a held reading, and is read whole only the first time |
| a new module beside `src/hands/sessions/backfill.py` | the watcher: one held fold per transcript read lately, let go when nobody reads it |
| `src/hands/core/progress.py` | the table gains `Skill`, `SendMessage`, the dialogs, and the path with the `cwd` taken off |
| `src/hands/daemon/cli.py` (`hands phone`) | the address of the page is printed beside the others, and the terminal's packages are fetched with the key |
| `docs/architecture.md` ("Transcripts: the live tail, and backfill", "The audio side", "Typing into a session"), `README.md`, `docs/testing.md` | the page, the held fold, the lines, the terminal and its bridge, that the terminal's keys are a keyboard's and not a send |

## The tickets

Eight tickets under epic `promptctl-phone-app-9d9y`, in one lane, `phone-sessions`, behind this design's own ticket. Each is built and merged before the next starts.

1. `promptctl-phone-app-9d9y.0q9`: one function says a session's state, for the brain's listing and for the page: "The cards, the fold and the groups", "On the Mac".
2. `(to file)`: one table says what a call sets out to do, for the voice and the page, for every tool Claude Code calls, dialogs and skills included: "The timeline", the table.
3. `promptctl-phone-app-9d9y.tpc`: the sessions page lists every session this run of hands knows, grouped by project, the ones waiting on Brandon first and never folded, idle and ended ones folded into a line that opens on a tap, and each card turns as its session does: "The cards, the fold and the groups", "Between the page and hands".
4. `(to file)`: a card opens on the session's timeline, every tool call read from its transcript, newest first under the prompt that started its turn, paged back as far as Brandon scrolls, new calls arriving at the top without moving what he reads, the same for a live session and an ended one: "The timeline", "On the Mac".
5. `(to file)`: a call that started a subagent opens on that subagent's own calls, read from its own transcript: "The timeline".
6. `(to file)`: hands holds the terminal's two packages, serves them to the page, and runs the bridge: fetched at their versions and held to their digests, the Node process started for the first terminal and ended a minute after the last, reached over a socket in hands' home: "The terminal", "On the Mac".
7. `promptctl-phone-app-9d9y.vkc`: a card's `Show screen` shows the session's terminal live, at the phone's size, typed into from the phone, under the phone's key: "The terminal", "Between the page and hands".
8. `promptctl-phone-app-9d9y.bqh`: Brandon's phone shows him what a session he sent off did while he was away, call by call, and what it is waiting on: carries the `external` label, since its event is outside the repository, so nothing waits on it.

Between them they build every row of "Where the build lands".

## Sources

- Measurements on Brandon's machine, 2026-11-24, as given to this revision: the audit log's size, span and event shapes; the 42 sessions of the window and what each did; the membership files; seven transcripts' sizes, `tool_use` and `tool_result` counts, `is_error` counts, Bash `description` counts, `isSidechain` and `compact_boundary` counts, record types and `sessionId` values. Not repeated here.
- `src/hands/sessions/turning.py`, `backfill.py`, `tail.py`, `transcript.py`, `subagents.py`, `registry.py`, `membership.py`, `liveness.py`; `src/hands/core/session.py`, `turn.py`, `steps.py`, `progress.py`, `reducer.py`, `events.py`; `src/hands/voice/tools.py`, `longpoll.py`, `brainpage.py`, `conversationpage.py`, `phonepage.py`, `keyed.py`, `hands.js`; `docs/architecture.md` ("Hooks carry the moment; the transcript carries the record", "Transcripts: the live tail, and backfill", "Sessions: membership from files, state from events, liveness from the OS")
- `~/code/tmux-control-mode-js` at HEAD 7b603df, read 2026-11-24 for this revision (the first draft read its HEAD of 2026-11-16, 21 commits earlier, before the window terminal was published): `README.md` ("A window in the browser", "Things to know"); `package.json` (its exports, `engines`, no runtime dependencies, `@xterm/xterm` ^6 as a peer); `src/window-terminal/terminal.ts`, `sizing/solve.ts`, `sizing/client-size.ts`, `sizing/cell-metrics.ts`, `render/window-renderer.ts`, `engine/window-model.ts`, `ids.ts`; `src/client-size.ts`; `src/connectors/websocket/server.ts`, `types.ts`, `client.ts`, `handshake.ts`; `examples/web-multiplexer/server/bridge.ts`, `control-client.ts`, `tmux-target.ts`, `main.ts`
- tmux 3.8a on Brandon's machine: its man page (`window-size`, `refresh-client -C`, the `ignore-size` client flag), `show -g window-size` (`latest`) and `aggressive-resize` (`off`), `list-clients` (his desk clients at 148x39); and six probes, 2026-11-24, on throwaway tmux servers (`tmux -L`) with a 148x39 client attached on a pty and a `tmux -C attach` control client declaring `refresh-client -C CxR` and `-C @W:CxR`, then redeclaring, clearing, setting `ignore-size`, and dying, with the desk client typing and selecting windows between: the measurements in "The terminal". Not repeated beyond that section.
- `src/hands/voice/phone.html` (the talk button, the brain's Stop), `src/hands/voice/conversationpage.py` (words typed reach the brain as a clip, `hands.voice.turnstop.Typed`), `src/hands/voice/keyed.py`, `src/hands/core/bearer.py`, `src/hands/voice/phoneaddress.py` (`phone_key`, `clock.token`), `src/hands/sessions/startsession.py`, `src/hands/sessions/tmux.py`, `src/hands/sessions/child.py`, `src/hands/brain/process.py`; `docs/architecture.md` ("Typing into a session"); `CLAUDE.md`, its hard rules
- `design-docs/phone-wake-word/design.md` (PR 556), for how a phone page is served and what it may keep
