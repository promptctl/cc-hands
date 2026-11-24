# hands as a remote MCP server for the Claude app

Written 2026-11-20. Epic `promptctl-remote-mcp-dvzu`, design ticket `promptctl-remote-mcp-dvzu.4l5`.

## What it is for

Brandon, away from the desk or just typing, opens the Claude app (web, desktop, iOS) and does through it what he
does by talking to hands: lists his sessions, reads what one did, stages and sends a prompt, answers a permission,
starts or closes a session, and reaches the brain's other servers (vhid, today) the same way. The Claude app's model
stands where the brain stands; hands is the one server it connects to. Nobody without his Cloudflare Access login can
call anything.

Asked as: "an HTTP MCP server that exposes the tools the hands brain has access to. It should bundle hands' own MCP
tools plus the other MCP servers the brain is given, proxied behind the one server. The client would be the Claude
app, with Cloudflare Access doing OAuth in front of it."

## What already exists

hands already is an HTTP MCP server: `src/hands/brain/mcp.py` serves the tools of `hands.voice.tools` over streamable
HTTP to the brain, POST-only, JSON answers, on `127.0.0.1:<free port>`, admitting only a bearer token that the brain's
`--mcp-config` holds. That transport is exactly what the Claude app's custom connectors speak (a server that opens no
GET stream is allowed by the spec), at the protocol versions it speaks, 2025-06-18 and 2025-11-25. So this is not a
new server. It is a **second door** on the one hands has.

The brain is given three kinds of tools; only two are MCP and in scope:

| Brain gets | From | Remote door serves |
| --- | --- | --- |
| hands' tools (`mcp__hands__*`) | `--mcp-config`, from `serve_mcp` | yes, each that means anything outside a voice turn |
| its other MCP servers (`mcp__vhid__*` today) | `mcpServers` in `<hands home>/brain/.claude.json` | yes, forwarded |
| Claude Code's built-ins (Bash, Read, …) | Claude Code | no: not MCP, and not asked for |

## Shape

```
 Claude app ──HTTPS──► Cloudflare edge ──────────► cloudflared ──► 127.0.0.1:<remote port>/mcp ─┐
 (Anthropic cloud)     Access: Managed OAuth        (launchd,       remote door:                 │
                       policy: Brandon only         outbound)       Cf-Access-Jwt-Assertion     │
                       adds Cf-Access-Jwt-Assertion                  checked                     │
                                                                                                 ▼
 brain (Claude Code) ─────────────────────────────► 127.0.0.1:<free port>/mcp ──►  one JSON-RPC handler ──► hands' tools
                                                    brain door: bearer token        (mcp.py)        └─────► forwarded servers
                                                                                                           (stdio children,
                                                                                                            http clients)
```

One handler, two doors. A door is a value of its own kind, carrying exactly what that door has, and each fact about
a call lives on the caller that has it [LAW:types-are-the-program]. The handler matches the door where doors differ
(who is admitted, what tools/list says, what initialize says) and nowhere else; no field of either is ever absent.

```python
Door = BrainDoor | RemoteDoor
@dataclass(frozen=True)
class BrainDoor:
    token: str            # admits a Brain: the bearer token the brain's --mcp-config holds
    acts: Catalogue       # what both doors serve
    voice: Voice          # the voice's own tools, and what a call is to a voice turn (below)
@dataclass(frozen=True)
class RemoteDoor:
    access: Access        # admits a Remote: the Access team, the application's AUD tag, the team's key set
    acts: Catalogue
    instructions: str     # initialize's instructions, for a model with no system prompt of hands'

Caller = Brain | Remote   # a BrainDoor admits only a Brain, a RemoteDoor only a Remote [LAW:single-enforcer]
@dataclass(frozen=True)
class Brain:  turn: ToolUse | Unjoined    # claudecode/toolUseId from the call's _meta; Unjoined is flagged on its event
@dataclass(frozen=True)
class Remote: email: str; tasks: bool     # from the Access JWT, which is refused without an email; whether the call's
                                          # _meta declared io.modelcontextprotocol/tasks
```

## Decisions

**A tool's act and its place in a voice turn are two values.** Today a `Tool` carries both: the act (its name,
description, schema and body) and the voice turn's protocol (`then`, the cue, sentences in its description that tell
a speaking model how to end its reply, the focus it defaults to and moves, a line it waits to have heard). The remote
door serves the act; the voice part is the brain door's `Voice`, and the act cannot hold any of it
[LAW:types-are-the-program]:

- **The act is built with no player and no refocus**, so its body cannot speak, wait on a heard line, or move the
  focus. The `Voice` wraps the acts it serves the brain, and does those around the act's call: `tell_turn`'s act
  returns the session it told, and the wrap moves the focus to it; `use_model`'s wrap says the switch and waits for it
  to be heard before the call is answered, a barge-in answering it with the refusal the brain is told to ask again on.
  The one effect no body performs is keeping a model, which restarts hands: `use_model`'s result carries it as
  `Keep(model)`, and the handler performs it once the response is written, on both doors, the one place it is
  ordered after the write [LAW:single-enforcer] [LAW:no-ambient-temporal-coupling]. A wrap whose heard line was barged
  in drops the `Keep` with the result it replaces. On the remote door the Claude app's approval prompt stands where
  the heard line stands.
- **The voice's own tools are the `Voice`'s**, not acts: `stay_silent`, `resume`, `skip`, `repeat` (the reply hands is
  speaking) and `focus_session` mean nothing outside a voice turn, so the brain door serves the acts and these, and
  the remote door serves the acts, with no list of names to keep beside them.
- **What a call is to a voice turn** is the `Voice`'s, per tool: `then`; whether it plays the working cue (today a
  tool's place in `intermediary_tools`' `acts` list and two `cued` calls in `run.py`); and whether a call naming no
  session acts on the focus (`defaulting_to_focus`).
- **A description says what is true for any caller**, and the `Voice` adds the sentences true only in a voice turn
  ("Calling it is the whole reply: add no words of your own", "Hands says it is switching, and switches once the user
  has heard that", "The session told becomes the focus"). "Tell the user" is true for both; `catch_up`'s window "since
  they last spoke to you" becomes "since the user last spoke to hands", which it is.
- **The focus is the voice conversation's**, so a remote call names its session and moves no focus, and what the app
  does never changes which session a spoken "send it" lands on.

A `read_back` result is spoken only because the brain's stage reads it off the brain's wire, so a remote call gets it
as data and the room hears nothing. Everything else is served as it is, the room's acts and the voice's settings
included: a remote `set_listening_level`, `set_modality`, attention change or Spotify call does to the desk what the
spoken one does, which is what someone away from the desk asks for.

**The set is the brain's setup, checked against what the brain is offered.** The remote door serves hands' own tools
and the tools of each server named in the brain's `.claude.json`. That file is read by `hands.brain.configdir`, which
already parses it; its `State` gains the servers, from the top level and from `projects[<the brain's cwd>]`, a name in
both being the project's, as Claude Code resolves it. The set is not taken from the brain's requests because those
exist only once the brain has had a turn since hands started, and the remote door is for when nobody is talking to it.
The requests are the check instead: hands already reads the tools each main turn offers the model (`hear` in
`src/hands/brain/process.py`), and each time the `mcp__` tools on one side and not the other change, one event names
them: a server from a scope hands does not read (a plugin's, a project `.mcp.json`), or a tool a deny rule withholds
from the brain.
Deny rules are not applied to the remote door: they bound what the brain's model does unasked, and the remote caller
is Brandon behind the Claude app's own per-tool approvals.

**The other servers are forwarded by hands as their own MCP client.** Rejected alternative: making the brain execute
the call (hands rewriting a response to inject a `tool_use`). That needs a main turn of the brain, which would occupy
it, collide with spoken turns, and write into its transcript. Running `claude mcp serve` is rejected by the hard rule
against starting Claude Code for what hands can do itself.

- Each entry parses into `Stdio(command, args, env) | Http(url, headers)` [LAW:parse-dont-validate]. An entry hands
  cannot run (an `sse` server, one needing OAuth tokens that live in Claude Code's keychain, a `${VAR}` that is
  unset) is left out with one event saying which and why, never half-started.
- A stdio server is started through `hands.sessions.child.Children`, as a child of the daemon for the life of the
  remote door, so it is reaped and stopped as hands' other children are, and the `mcp` SDK's `ClientSession` speaks
  over its pipes (stdio framing, request correlation, server-to-client pings). `Children` hands over no pipes today
  (`Stdio = Null | Terminal`, and `Running` exposes no streams), so a `Pipes` variant, with `Running` exposing its
  streams, is the first step of the ticket that forwards. The SDK's own `stdio_client`, which spawns the child
  itself, is not used.
- Its environment is the brain's (`environment()` in `src/hands/brain/process.py`, which keeps hands' own logins out)
  plus the entry's `env`, and `${VAR}` expands against that same environment. The daemon composes it and hands it to
  the forwarder as a value, so the forwarder imports nothing of `process.py`, which imports `mcp.py` [LAW:one-way-deps].
- It is hands' own instance, not the brain's, so state a server keeps per connection (chrome-devtools' selected page,
  say) is not shared with the brain's. Sharing it would need the brain's Claude Code to make the call, rejected above.
- The catalogue is keyed by tool: hands' own by their bare names, a forwarded one by `(server, tool)`. A server name
  is parsed to hold no `__`, and an entry whose name does is left out with its event, so the wire name
  `<server>__<tool>`, mirroring Claude Code's `mcp__<server>__<tool>`, names one tool and is built in one place beside
  `wire_name`.
- A `Tool` stores the `inputSchema` it is listed with, where today it derives it from `properties` and `required`.
  `tool()` still derives a hands tool's from its body's signature and closes the body against it; a forwarded tool's
  is the upstream's own, as given (`oneOf`, `$defs` and all), and the upstream that owns it checks the arguments. So
  what is listed and what is taken cannot differ for either.
- A forwarded tool is a `Tool` whose body calls the upstream, so every call, hands' own or forwarded, is one `tool.run`
  event from the one `audited` [LAW:single-enforcer], naming its server. A body's result is `Result | Upstream`:
  hands' own JSON object, typed as JSON values so encoding one cannot fail, which the handler sends as today's one
  text block; or the upstream's result as given, its content blocks and `isError` included. `audited` reads failure
  from either (an `error` key, or `isError`), and moves into `hands.voice.tool`, which loads no Pipecat, so the
  forwarder and `mcp.py` can reach it.
- A server that dies is an event, its tools leave `tools/list` (answered fresh on every request), and a call to one
  answers an error naming it.
- hands' server side stays hand-written: it works, it is tested, and the SDK would change nothing a client sees.
  `mcp` and `PyJWT` become dependencies, and `cryptography`, today only pulled in by others, a direct one.

**Access checks the person; hands checks Access.** Cloudflare Access with Managed OAuth is the OAuth authorization
server: it answers the Claude app's unauthenticated request with a 401 and `WWW-Authenticate`, serves the RFC 8414 /
RFC 9728 metadata, runs dynamic client registration and the browser login, and issues opaque tokens. On every
admitted request it adds `Cf-Access-Jwt-Assertion`. The remote door admits a request only when that JWT verifies:
RS256 against the team's keys at `https://<team>.cloudflareaccess.com/cdn-cgi/access/certs`, `iss` the team domain,
`aud` the application's AUD tag, not expired, an `email` claim present. This makes the door safe even to a local
process that reaches its port directly, and the email is who the audit names.

- The key set is fetched off the loop when the door opens and again for a `kid` it does not hold, at most once a
  minute, and held in memory, so a rotation costs one fetch and never stalls the loop the voice runs on. Admission is
  async for that reason. The key set is a port, so the unit tests sign with a key of their own and nothing reaches
  Cloudflare.
- Every refused request is one event saying why: no header, a bad signature, the wrong `iss` or `aud`, expired, a
  `kid` the team does not publish, no email, keys that could not be fetched. The brain door's refusals, today a log
  line, become the same event, so a re-issued AUD tag reads as refusals, never as silence.

**The door stays on loopback; cloudflared is not hands'.** The remote door binds `127.0.0.1` on the port its setting
names, the same across restarts, since a tunnel needs a stable origin. `cloudflared` runs as its own launchd service
(`cloudflared service install <token>`), like any tunnel on the machine; hands neither starts nor supervises it.
Tailscale, which already serves the phone page, cannot do this job: the Claude app connects from Anthropic's cloud,
not from Brandon's devices.

**Configuration.** One `[remote]` table in hands' `config.toml`, parsed by `hands.daemon.config` as every table is,
and read only by the daemon:

```toml
[remote]
port = 8765                                   # the tunnel's origin: http://127.0.0.1:8765
team = "https://<team>.cloudflareaccess.com"  # the JWT's issuer, and where its keys are
aud = "<the Access application's AUD tag>"
```

No table, no remote door; hands' start event says which, with the port. A table that does not parse, or a port that
will not bind, stops hands' start, saying why, as every other setting does. `port` and `aud` are copies of what the
tunnel and the Access application hold, which hands cannot read: an `aud` that disagrees shows as refusals, and a
`port` the tunnel does not point at as no request at all, which the tunnel's own log names.

**The Claude app gets instructions.** The brain learns hands from its appended system prompt and its skills; the
Claude app's model gets only what the server sends. The remote door answers `initialize` with `instructions`: a short
text, written for a model reading rather than one speaking, of what hands is, what a session is, and the order its
tools are used in (list, read, stage, send, then `turn`).
It lives beside the brain's instruction and is written with laws:prompt.

**A call of any length gets its own result.** Cloudflare drops a proxied request whose origin sends nothing for 100
seconds (a 524). hands' own calls answer in seconds (`start_session`, the slowest, in about 35), but a forwarded call
lasts as long as its upstream takes. Streamable HTTP lets a server answer a POST with `text/event-stream`, and every
client must accept that, so every `tools/call` is answered as a stream on both doors: the headers at once, an SSE
comment every 30 seconds while the call runs, and the result as the stream's one message. The bytes never stop for
100 seconds, so the result reaches the client however long the call takes, and no call is answered before it ends.
The act is not cancelled when a client hangs up, as today: a half-done act is worse than an unread result. Whether
the tunnel passes a stream through unbuffered is proven by the live connect.

## Long-running work: a send is the job

The Claude app's guidance for a long job is two tools: one starts it and returns a job id, the other takes the id
and, once the job is done, returns its result. MCP's Tasks extension (SEP-1686, now `io.modelcontextprotocol/tasks`)
is the same pattern in the protocol: a `tools/call` answered with `resultType: "task"` and a `taskId`, polled with
`tasks/get` until it is `completed` (the result rides on it), `failed` or `cancelled`, with `input_required` for a
job that needs an answer, given through `tasks/update`.

The one long job in hands is what a Claude Code session does with a prompt it was sent. Today that job has half the
pattern and a defect: `send_draft` starts it and returns only a readback, and `tell_turn(session)` tells the
session's *last finished* turn. A client that sends and then asks is told the turn before its own as though it were
the answer. The voice brain never meets this, because hands narrates each turn as it finishes. The Claude app has no
such channel: it only learns something by asking.

**Every tool that types a prompt returns a handle naming that send** (`send_draft`, and `send_command` for a command
Claude Code runs as a turn), as soon as it is typed. Nothing waits on the session: what became of the words is read
when it is asked for. **One tool, `turn(handle)`, says where the send stands**, read from the reducer and from the one
store of tellings, never from a job table of its own [LAW:one-source-of-truth]:

- **The reducer carries the send.** Today it files a prompt's words under the turn that takes them by their text
  alone (`Opened.waiting` is `tuple[str, ...]`, matched by `_unqueued`). hands' typing of a send becomes an event the
  reducer applies, so each waiting entry is its words and the send that typed them, or none for words hands did not
  type, and the session keeps, for the life of the session as it keeps `earlier`, the turn each send's words opened
  or were taken into. Equal words are told apart by order, oldest first, as Claude Code's queue takes them and as the
  reducer already matches its queue records; words typed at the keyboard between two equal sends are the one case
  order alone decides.
- **`Recounts` holds tellings by turn**, each session's 16 latest, where today it holds the newest alone. `tell_turn`
  and `expand` read the newest, `turn` reads the send's, and `open` and `unread` change the one telling both read.
- **A handle names the run of hands that issued it**, so a handle from before a restart (and `use_model` restarts
  hands) is told apart from one never issued.

| The send | `turn` answers |
| --- | --- |
| typed, and no turn has its words, or they wait behind the running turn | `queued`, with how the session stands now |
| its words are in a turn that is open: their own, or one they were delivered into | `working`, with what the session is doing now |
| that turn stopped on a dialog | `waiting`, with the permission, question, or plan, answered by the permission tools as any dialog is |
| that turn ended, as the reducer says (`Untold` or `Told`) | `done`, with its `telling`: the turn as `tell_turn` tells it, or `untold` where narration has not told it, or `released` where it is past the 16 |
| its session is gone before that turn ended | `gone`, with how the session ended |
| issued by an earlier run of hands | an error saying so and naming `read_session` |

Tellings are written by narration, which runs once the voice is up, so a turn the reducer says ended is `done` with
`untold` while the voice loads or is down, never a turn that did nothing. `tell_turn` has that gap today
(`Recounts.of` returning None reads as "no turn has finished since hands started"), and reads whether a turn ended
from the reducer too. "What did it just do" is a different question from "how is the prompt I sent going", so
`tell_turn` stays. `turn` moves no focus. Both doors get `turn`, so the voice brain can answer "is it done yet?" by
the prompt it sent, rather than by whichever turn happens to be newest.

**Tasks are not built now.** Anthropic shipped Tasks in the Claude Code CLI on 2026-10-23 (public beta, v2.1.304)
and said other Claude products would follow. Nothing found shows the Claude app declaring the extension yet, and a
server must never return a task to a client that did not declare it in the request's `_meta`. So a `Remote` caller
carries whether its call declared it, and the call's event records it. The ticket that answers with tasks is in the
order below, pulled once a remote call's event shows the Claude app declaring the extension: a `send_draft` from a
`Remote` caller that declared it is answered with a task whose `taskId` is the send's handle, and whose `tasks/get`
reads `turn`. `queued` and `working` map to `working`, `waiting` to `input_required`, `done` to `completed`. That is a
second way of writing out the same job, with no second store behind it. A `Brain` caller carries no such field, so it
gets no task even if its Claude Code declares the extension: hands tells the brain each turn as it finishes, and a
task would tell it twice.

## Cloudflare and the Claude app, set up once

1. A tunnel for the machine, its public hostname (say `hands.<domain>`) pointing at `http://127.0.0.1:<port>`.
2. An Access application for that hostname, an Allow policy naming Brandon's email, and Managed OAuth on, with
   `https://claude.ai/api/mcp/auth_callback` among the allowed redirect URIs for dynamically registered clients.
   Short access-token lifetime, long grant session (the docs' advice), so a revoked login ends within minutes.
3. If Bot Fight Mode or a WAF rule covers the zone, a Skip rule for the hostname: Claude's backend arrives as
   `Claude-User` from US addresses and is otherwise blocked before Access sees it.
4. In the Claude app: Settings → Connectors → Add custom connector, URL `https://hands.<domain>/mcp`, OAuth client
   "Register automatically".

The live connect moves these to `docs/` as it proves them, and this section then points there.

## The risk, and the order that retires it first

Claude's web and mobile connectors have failed against Access Managed OAuth while Claude Code connected to the same
URL (anthropics/claude-ai-mcp#410, June 2026, closed as not planned). The causes found there were the RFC 8707
`resource` parameter, which Access requires on `/authorize`, and protected-resource metadata that offered
`cloudflared` login ahead of `oauth`: the first places to look if the live connect fails. Whether today's Claude app
and today's Access get along is not known from here.

Nothing in hands depends on the answer: whichever Access front works, the origin contract is one JWT header. So the
live connect is its own ticket, taken as soon as there is a door to connect to, and no code waits on it.

## Tickets, in order

1. **A tool's voice part is a value of its own.** `then`, the cue, the focus, the voice's own tools, the
   description's voice sentences, and what a voice turn does around a call (refocus, a heard line) move off the act,
   which is built with no player and no refocus, into the brain door's `Voice`; a model's keep becomes the `Keep` the
   handler performs after the write. Done when the brain is
   offered the same tools, with the same descriptions but `catch_up`'s reworded window, and the same calls cue, speak
   and move the focus as before, by the existing tests, which are the brain's contract here.
2. **Walking skeleton: the remote door, admitting only Access.** `Door` and `Caller` in place of `used`, the remote
   door serving hands' own acts with JWT admission and the `[remote]` setting, a refusal event on both doors, and
   every `tools/call` answered as a stream. Done
   when, in unit tests with a test-signed key set, a request with a valid JWT lists and calls hands' tools with no
   voice-only tool among them, plays no cue and moves no focus, and each bad JWT is refused with its event.
3. **The Claude app connects through Access.** The setup above, done live and moved to `docs/`. Done when
   `list_sessions` answers from claude.ai and from the iOS app, a call held open past 100 seconds reaches the app with
   its result, and a request without Access's header is refused, with the events as the evidence. A live run on a ticket of its own, so no code waits on it.
4. **The brain's other MCP servers, forwarded.** `Children`'s `Pipes`, the servers in `configdir`'s `State`, the
   stdio and http clients behind a port of hands' own, the forwarded tools and their names, `audited` reading
   `isError`, the events for a server left out or dead, and the check against the brain's offered tools. Done when
   unit tests over a fake upstream show the names, the upstream's schema listed as given, results passed through with their content and `isError`, a dead
   server's tools leaving the list, and each change in the difference from the offered list as an event; and an
   outside-marked test runs a tiny stdio MCP server through the remote door.
5. **A send is a job: its handle, and `turn` to ask after it.** The reducer carrying sends, `Recounts` by turn, and
   `tell_turn` reading a turn's end from the reducer. Done when a client that sends a prompt and asks after it is
   told that send's state, through `queued` when the session was busy, and, once its turn ends, that turn and no
   other, with `untold` while narration is not running.
6. **Instructions for the Claude app**, written for a model that sends a prompt and then asks `turn` after it. Done
   when an eval of the Claude app's model, given the remote door's tools and instructions, lists, stages, sends, and
   asks `turn` in that order and shows the user what a readback says.
7. **Tasks for a client that declares them**, pulled once a remote call's event shows the Claude app declaring the
   extension. Done when such a client's `send_draft` is answered with a task that `tasks/get` follows to `completed`.
