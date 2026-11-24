# The sessions hands sends off, on the phone page

Written 2026-11-23 from a conversation between Brandon and Claude. Ticket `promptctl-phone-app-9d9y.68y`.

## The idea in one paragraph

The phone gains a page of the sessions hands knows, one card for each. A card says what the session is doing in hands' own words: `working, running the test suite`, `waiting on you: Bash wants to run git push`, `idle, finished 4 minutes ago`. Tapping a card opens what the session did, a summary for each turn it finished and the steps of the one it is on, and a button that shows its terminal as it looks at that moment, as text. Nothing on the page is live in the sense of a stream: the page reads hands the way the brain's panel does, and a change comes back as it happens. Nothing on the page changes a session: no typing, no answering a dialog. The page is the lighter of the two views Brandon named. The existing way to show a tmux session in a browser, `tmux-control-mode-js`, was weighed against it and is not used. "Why this and not the alternatives" says why, and what would change the answer.

## The picture

```
  the phone page                                          hands, on the Mac
 ┌───────────────────────────────────────────┐           ┌────────────────────────────────────────┐
 │  the sessions page                        │  GET      │                                        │
 │                                           │  /sessions│  the registry         the audit log    │
 │   ┌─ cc-hands, naming fix ──────────────┐ │  (waits   │  state, dialogs,      what each turn   │
 │   │ waiting on you                      │ │  for a    │  latest Doing,        did, summaries   │
 │   │ Bash wants to run: git push         │◄┼──change)──┤  subagents                 │           │
 │   │ asked 2 min ago                     │ │           │       │                    │           │
 │   └─────────────────────────────────────┘ │           │       ▼                    ▼           │
 │   ┌─ lit, status line ──────────────────┐ │           │   one card for each live session       │
 │   │ working: run the test suite         │ │           │                                        │
 │   └─────────────────────────────────────┘ │           │                                        │
 │          │ tap                            │  GET      │                                        │
 │          ▼                                │  /sessions│  its turns, newest first               │
 │   what it did, turn by turn ◄─────────────┼──/<id>────┤  (read_session's reading)              │
 │          │ "Show screen"                  │           │                                        │
 │          ▼                                │  GET      │  tmux capture-pane, once, in the pane  │
 │   the terminal as text ◄──────────────────┼──/screen──┤  it runs in; none for a session in     │
 │                                           │           │  no pane                               │
 └───────────────────────────────────────────┘           └────────────────────────────────────────┘
```

## What Brandon does with it, step by step

1. He tells hands "start a session in lit and have it fix the status line". hands starts one (`start_session`), tells it the work, and answers by voice. He puts the phone in his pocket and walks away from the desk.
2. Ten minutes later he opens the page, the address `hands phone` prints with `sessions` on its path, as the conversation and backlog pages are opened. Every live session is a card, the ones that need him first.
3. The `lit, status line` card reads `working: edit 3 files, then run the test suite`. It is working, and he can tell what at.
4. A second card, `cc-hands, naming fix`, reads `waiting on you: Bash wants to run git push`, with how long it has waited. That is what the page is for: a session stopped on him, found without asking out loud.
5. He taps `lit, status line`. The page lists its turns, newest first, a sentence for each finished turn and, for the one running, the steps so far: `edited statusline.py`, `ran the tests: 2 failed`, `read lit.go`.
6. Something in a step looks wrong. He taps `Show screen` and reads what the terminal shows now, in a monospaced block that scrolls inside the page. He taps it again for a newer one.
7. The page keeps itself current while it is open: a session that finishes turns its card to `idle, finished just now`, with no tap.

## Where this design departs from the words of the request

Each of these is Claude's, and Brandon's to overrule.

**"See that session" is read as: see what it is doing and what it needs of him. Not: watch its terminal draw.** The request offers both and asks for a recommendation. The reasons are in "Why this and not the alternatives". The terminal is still reachable, as a snapshot on a tap, so a session in a state hands has no words for can be looked at.

**Every live session is shown, not only the ones hands sent off.** hands keeps no record of who started a session or who sent it a turn. `Session` (`src/hands/core/session.py`) has none, and the audit log holds hands' tool calls, which are not a state to filter by. A filter would be a second record of a fact the registry never held, and a session Brandon started at the desk is as worth a glance from the phone. The page orders by what needs him, so the sessions hands sent off are not buried.

**The page changes nothing.** Typing into a session is hands' to decide: `core.reach.writer` and the refusals in "Typing into a session" (`docs/architecture.md`) keep a paste from landing in a permission dialog or in the wrong program. A page with a keyboard in it would need that whole rule, or would bypass it. The brain's panel has one button, Stop, and it stops the brain, never a session (`hands.voice.brainpage`). This page has none.

## The two views, and the choice

Both candidates answer "see the session from the phone". They differ in what the phone is given.

**The full view** is `tmux-control-mode-js` (`~/code/tmux-control-mode-js`, `@promptctl/tmux-control-mode-js` 0.4.0): a Node 20 client of tmux's control mode, with a WebSocket bridge and a window terminal that draws tmux panes with xterm.js 6. Its showcase has a read-only pane mirror: a control client taps a pane with `pipe-pane`, seeds a late viewer with `capture-pane -e -p`, and fans the bytes to browsers whose protocol has no message that can write.

**The lighter view** is what this document describes: hands' own reading of each session, sent as JSON, and the screen as one text snapshot.

| | the full view | the lighter view |
|---|---|---|
| what the phone gets | the terminal, byte for byte, live | hands' words for the state, the dialog's content, the turns and steps; the screen as text on a tap |
| fits a phone | no, see below | yes: a card is a few lines |
| a session in no tmux pane | nothing | everything but the snapshot |
| a session on another host (`promptctl-devices-so9`) | a bridge on each host, and a tmux there for it to attach to | what the hub reads of it, as the hub learns to read it |
| what hands has to run that it does not | a Node process beside the Python one, a tmux control client attached to the user's server, a JavaScript bundle | none: routes on the page's own aiohttp app |
| who checks the phone's key | the page's route, and then the bridge behind it must be told to trust that, or check again | the page's route, `keyed`, as for every page |
| what a connected phone can reach | whatever the bridge permits: the library's bridge routes tmux commands from the browser and is restrained only by `authorize(method, args)` | what hands' route returns |
| cost while nobody watches | a control client attached to the tmux server, and a tap per pane watched | none |
| exists today | the example's mirror, not published; the published parts are the window terminal and the bridge | the registry, `read_session`, `read_screen` |

The two numbers that decide how much the full view can mean on a phone:

- A phone is about 390 CSS pixels wide. A terminal of the width tmux clients have on this Mac now, 148 columns (`tmux list-clients`), leaves 2.6 pixels a column, a font of about 4.4 pixels at the usual 0.6 em per cell. The brain's pane is 200 columns by 50 rows. A readable 10-pixel font fits about 65 columns.
- So the full view either shows a desk's terminal too small to read, scrolls sideways across it, or makes the pane phone-sized. The library's default window sizing does the last: the page declares its size to tmux (`refresh-client -C`) and tmux sizes the window by its `window-size` option, which is `latest` on this Mac. Whether a phone's declaration moves the desk's own window was not tried here, and is the kind of thing nobody should find out by opening the page at the desk.

What xterm.js does under a finger is reported by others, not tried here: it has no touch handling of its own, text selection on iOS fights scrolling, and a focused field under 16 pixels makes iOS zoom the page (Sources). A read-only view avoids the keyboard and not the scrolling.

**The recommendation is the lighter view.** The question Brandon asks of the phone when a session is off working is "is it going, is it stuck on me, what has it done". hands already holds every part of that answer, as state (`Running`, `Idle`, `Held` dialogs with the tool and its input, subagents), as `Doing`, and as the turns `read_session` reads. The terminal holds the same facts drawn for a desk, and the phone would be asked to read them back out. The full view's costs are all new: a runtime hands does not need today, a second place that decides who may look at a terminal, and a size the phone cannot hold. Where he does want the whole terminal on the phone, a session is in a tmux session he can `tmux attach` to from any terminal, as the README says of the brain's; an SSH app on the phone is such a terminal. That was not tried here.

## What a card says

A card is one session's state in the words `list_sessions` gives the brain (`describe_listing`, `_spoken_state` in `src/hands/voice/tools.py`), so what the phone says and what hands says aloud are one sentence, written once.

| the session is | the card says |
|---|---|
| at a dialog hands holds | `waiting on you:` what it asks, from the `Held` dialog: the tool and its input for a permission, the question and its options, the plan's first lines; and how long |
| running, with a call in progress | `working:` the latest `Doing`, said as hands says it; how long the turn has run |
| idle, with subagents out | `idle, with 2 subagents still working` |
| idle | `idle`, and when its last turn ended |
| a shell command it started still running | as `list_sessions` says it |
| not reported yet | `not reported yet` |

Order: waiting on him first, then working, then idle by how recently each finished. A session that ends leaves the page at the next read, as it leaves `list_sessions`.

The detail is `read_session`'s reading: a summary for each finished turn, newest first, and the steps of the running one (`Said`, `Edited`, `Ran`, `Tested`, `Looked`, and the rest of `Step`). A turn's summary may not exist yet; the page says the turn is being summarised, as `read_session` does, and does not wait for it.

### The screen

`Show screen` asks hands to read the pane the session runs in, as `read_screen` does (`tmux.shown`: `capture-pane -p`, the blank rows below the last drawn line left off). It is one read per tap, not a loop: a page that asked once a second would be a stream, with a stream's costs. The text is shown with no markup, in a block that scrolls on its own, and wraps nowhere, so a line of a terminal is a line on the page and scrolls sideways when the phone is narrower than it.

A session in no tmux pane has no screen hands can read, and the button is replaced by the sentence `read_screen` returns for it. Sessions hands starts are always in a pane (`start_session`).

## Between the page and hands

Three reads, each under the phone's key, each a unit of work with its own event.

| route | answers | |
|---|---|---|
| `GET /sessions` | every live session as a card, with the version of the list | waits for a change, as the brain's panel does (`hands.voice.longpoll`): the page says the version it has, and hands answers when it differs or after 25 s |
| `GET /sessions/<id>` | the session's turns and the running turn's steps, with their version | the same wait |
| `GET /sessions/<id>/screen` | the pane's text, or why there is none | answers at once |

All three are `Read`s as every page's are (`hands.voice.longpoll.Read`): the version, how long a read may wait, the Mac's clock, and what is shown, so the page times `waiting 2 min` by the Mac's clock and not its own. The page asks through `hands.js`, the one script every phone page asks hands through.

## On the Mac

- **A route module of its own, as the brain's panel is.** `hands.voice.sessionspage` takes a `Watched`-style protocol over what it reads, so the page and the tests see the sessions through one interface, and nothing in it reaches a session. It is added to `phone_pages` (`src/hands/voice/phonepage.py`) beside the others, under the same key, and its page is served `no-store` like theirs.
- **One sentence for the state, written once.** `_spoken_state` becomes the shared function the brain's listing and the page both call. The page takes the dialog's content from the `Held` it describes, as the listing's text does, and does not parse the sentence.
- **The reading is `read_session`'s.** `_session_reading` and the steps behind it move to where the page can call them without the tool's wrapper, so the tool and the page read a session through one function.
- **The screen is `read_screen`'s.** The pane lookup and `tmux.shown` are what the tool calls; the page's route calls the same two, and the tool's sentences for a session in no pane, or a pane that could not be read, are the page's.
- **Every read is an event.** `sessions.read`, with how many cards, which versions the page held and was given, and whether it waited; `sessions.screen`, with the session, the pane, how many lines it read, or why it could not.

## What can go wrong

- **The screen shows what is on the terminal, secrets included.** A terminal shows file contents, diffs and command output. The page is behind the phone's key, as the conversation page is, and that page already shows everything hands said and every tool it called. The route is a new door onto the Mac's screens, and the key is its only lock, as it is for the others. The key travels in a URL's fragment, so a browser does not send it with the page request. Anyone who has it already reads the conversation and sets hands' voices and options.
- **A session that does not join hands is not on the page.** The registry is membership by files (`docs/architecture.md`, "Sessions"). A Claude Code held at the workspace-trust dialog has no membership and no card. It has a pane, and the user sees it at the desk. Whether the page should say that a start is waiting is not decided here.
- **A state hands has no words for.** A dialog it let go of, or one newer than hands' model of the screen, shows as `working` or `idle` while the terminal holds a question. `Show screen` is the way out: it is why that button exists.
- **A busy list.** A long-poll answered on every change in a list that changes every few seconds becomes a poll. The version is the hash of what the page shows, not of the registry, so a change the page does not draw (a status read again, a timestamp the card does not show) wakes nothing.
- **A phone behind a tailnet certificate that lapsed.** As for every phone page: hands serves its own certificate under a LAN address, and the phone is asked once to accept it.

## Why this and not the alternatives

- **The full view on `tmux-control-mode-js`.** The compared costs are in the table. The library is good at what it is for: its read-only mirror cannot write, by the shape of its protocol, and it seeds a late viewer without disturbing the pane. It is the right way to show a terminal in a browser. It is the wrong thing to put on a phone to answer "what is it doing", because the phone gets the terminal at the terminal's size, and hands already holds the answer in words. It also asks hands to run and trust a second process that holds a connection to the user's whole tmux server, where the lighter view adds routes to the server hands already runs.
- **What would change this.** If the wish becomes to watch Claude Code draw, or to type into a session from the phone, the full view is the way, and the design for it has these pieces: the mirror lifted out of the example into a package, a bridge that hands starts as a child and talks to only over loopback, hands' route as the only door and the only place the key is checked, a size the desk's windows never take from the phone, and hands' own typing path for anything written. None of that is built or decided here.
- **A live stream of the screen text.** `capture-pane` once a second is the full view's cost, a read of the pane for each phone for as long as it is open, without its fidelity. A snapshot on a tap is the cheapest read that answers "what is on the terminal".
- **Only sessions hands sent a turn to.** Needs a record that does not exist; see the departures.
- **Approve and deny on the card.** Answering a dialog is a write into a session, which hands' typing path decides. It is not part of seeing.
- **Add sessions to the brain's panel on the phone page.** The phone page is for the call. The sessions are a page of their own, with an address as the conversation and backlog pages have, so the call's page does not grow with every session.

## How it is tested

No rig, and nothing measured.

- **The cards** are built in process from sessions in each state, on the types the registry holds (`tests/test_mode.py` builds sessions for `describe_listing`): one test row for each line of "What a card says", and the order.
- **The routes** run on aiohttp's test client under the phone's key, with a fake `Sessions` and a fake tmux, beside `tests/test_phone.py`: a request without the key is refused, the read waits and is answered on a change and on the 25 s bound, and a version a change does not move does not wake it. The wait is forced with a gate in the fake, never timed.
- **The screen** runs against a fake `tmux.shown`: the text, a session in no pane, a pane that could not be read.
- **The page** runs in Chrome under Playwright against the routes hands serves, on fakes (`tests/test_pages.py`): a card turns from `working` to `waiting on you` with no tap, a tap opens the turns, `Show screen` shows the text and a second tap a newer one.
- **Brandon's phone** is the one check no test here can make, and is its own ticket.

## Where the build lands

| file | what changes |
|---|---|
| `src/hands/voice/sessionspage.py` (new) | the three routes, the cards, the wait, the events |
| `src/hands/voice/sessions.html` (new) | the page: the list, the detail, the screen |
| `src/hands/voice/hands.js` | nothing new if its one loop serves all three reads; a change here is a finding |
| `src/hands/voice/phonepage.py` | the page and its routes added to `phone_pages` |
| `src/hands/voice/tools.py` | `_spoken_state`, `_session_reading` and the screen read move to where the tools and the page both call them |
| `src/hands/daemon/cli.py` (`hands phone`) | the address of the page is printed beside the others |
| `docs/architecture.md` ("The audio side"), `README.md`, `docs/testing.md` | the page, what it shows, that it changes nothing |

## The tickets

Four tickets under epic `promptctl-phone-app-9d9y`, in one lane, `phone-sessions`, behind this design's own ticket. Each is built and merged before the next starts.

1. `promptctl-phone-app-9d9y.0q9`: one function says a session's state, for the brain's listing and for the page: "What a card says", "On the Mac".
2. `promptctl-phone-app-9d9y.tpc`: the sessions page lists every live session as a card, and each card turns as its session does, with the detail of its turns and steps: "What a card says", "Between the page and hands".
3. `promptctl-phone-app-9d9y.vkc`: a card's `Show screen` shows the session's terminal as text, once for each tap: "The screen".
4. `promptctl-phone-app-9d9y.bqh`: a phone shows Brandon the page and the screen of a session he sent off to work: carries the `external` label, since its event is outside the repository, so nothing waits on it.

Between them they build every row of "Where the build lands".

## Sources

- `~/code/tmux-control-mode-js`, read at its HEAD of 2026-11-16: `README.md` ("A window in the browser", "Watching a whole server", the sizing notes), `examples/SHOWCASE.md` and `examples/web-multiplexer/server/mirror-registry.ts` (the read-only pane mirror, and that it lives in the examples and not the package), `src/connectors/websocket/server.ts` and `types.ts` (`authenticate`, `authorize(method, args)`)
- `design-docs/phone-wake-word/design.md` (PR 556), for how a phone page is served and what it may keep
- xterm.js on a phone, as others report it: [xterm.js issue 5377, touch handling](https://www.goodfirstissue.org/xtermjs/xterm.js/issues/5377), [xterm.js discussion 4780, selection on touch](https://github-redirect.dependabot.com/xtermjs/xterm.js/discussions/4780), [agentboard's iOS notes](https://deepwiki.com/gbasin/agentboard/7.3-ios-specific-features) (scrolling, the 16 px zoom, the keyboard over the viewport). Read as search results, not opened one by one here
