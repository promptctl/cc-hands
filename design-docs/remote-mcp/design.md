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
GET stream is allowed by the spec). So this is not a new server. It is a **second door** on the one hands has.

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

One handler, two doors. A door says who it admits and what it serves; a call says which turn it belongs to.

```python
Caller = Brain | Remote          # admitted by the door, once per request [LAW:single-enforcer]
@dataclass(frozen=True)
class Brain: ...                 # bore the brain's bearer token
@dataclass(frozen=True)
class Remote: email: str         # bore an Access JWT that verifies and names an email; one naming none is refused

@dataclass(frozen=True)
class Call:
    caller: Caller
    turn: ToolUse | Unjoined | None   # the brain's: claudecode/toolUseId from the call's _meta, or Unjoined, flagged on
                                      # the call's event; None for a remote call, which is a trace of its own
    tasks: bool                       # the client declared io.modelcontextprotocol/tasks in the call's _meta
```

Only the remote door answers `initialize` with `instructions`; the brain learns hands from its appended system prompt.

## Decisions

**A tool's act and its place in a voice turn are two values.** Today a `Tool` carries both: the act (its name,
description, schema and body) and the voice turn's protocol (`then`, the cue, the sentences in its description that
tell a speaking model how to end its reply, the focus it defaults to and moves). The remote door serves the act, and
the voice part becomes a value of its own that only the brain door applies [LAW:types-are-the-program]:

- **What a call is to a voice turn**: `then`; whether it plays the working cue, a property of the tool where today it
  is a tool's place in `intermediary_tools`' `acts` list and two `cued` calls in `run.py`; and whether the tool means
  anything outside a voice turn at all. `stay_silent`, `resume`, `skip`, `repeat` (the reply hands is speaking) and
  `focus_session` mean nothing outside one, and the remote door does not list them.
- **The description's voice sentences** ("Calling it is the whole reply: add no words of your own") are written by the
  brain door from `then`, not kept in each docstring. The rest of a description is the act's, served to both; the
  remote door's instructions say once that a description's "say", "tell" and "spoken form" mean writing to the user.
- **The focus is the voice conversation's.** A remote call names its session (`defaulting_to_focus` wraps the brain
  door's tools only) and moves no focus (`tell_turn`'s refocus is the brain door's), so what the app does never changes
  which session a spoken "send it" lands on.
- **Speech is the brain door's.** A `read_back` result is spoken only because the brain's stage reads it off the
  brain's wire, so a remote call already gets it as data and the room hears nothing. `use_model` is the one served tool
  whose body speaks: it says the switch and keeps the model only once the line is heard, since keeping it restarts
  hands. That wait is its voice part; a remote `use_model` keeps the model once its answer has gone out, the Claude
  app's approval prompt standing where the heard line stands.

Everything else is served as it is, the room's acts included: a remote `set_listening_level` or Spotify call does to
the desk what the spoken one does.

**The set is the brain's setup, checked against what the brain is offered.** The remote door serves hands' own tools
and the tools of each server named in the brain's `.claude.json`. That file is read by `hands.brain.configdir`, which
already parses it; its `State` gains the servers, from the top level and from `projects[<the brain's cwd>]`, a name in
both being the project's, as Claude Code resolves it. The set is not taken from the brain's requests because those
exist only once the brain has had a turn since hands started, and the remote door is for when nobody is talking to it.
The requests are the check instead: hands already reads the tools each main turn offers the model (`hear` in
`src/hands/brain/process.py`), and every `mcp__` tool on one side and not the other is one event naming it: a server
from a scope hands does not read (a plugin's, a project `.mcp.json`), or a tool a deny rule withholds from the brain.
Deny rules are not applied to the remote door: they bound what the brain's model does unasked, and the remote caller
is Brandon behind the Claude app's own per-tool approvals.

**The other servers are forwarded by hands as their own MCP client.** Rejected alternative: making the brain execute
the call (hands rewriting a response to inject a `tool_use`). That needs a main turn of the brain, which would occupy
it, collide with spoken turns, and write into its transcript. Running `claude mcp serve` is rejected by the hard rule
against starting Claude Code for what hands can do itself.

- Each entry parses into `Stdio(command, args, env) | Http(url, headers)` [LAW:parse-dont-validate]. An entry hands
  cannot run (an `sse` server, one needing OAuth tokens that live in Claude Code's keychain, a `${VAR}` that is
  unset) is left out with one event saying which and why, never half-started.
- A stdio server is started through `hands.sessions.child.Children`, the one place hands starts a child, as a child
  of the daemon for the life of the remote door, and the `mcp` SDK's `ClientSession` speaks over its pipes (stdio
  framing, request correlation, server-to-client pings). Its environment is the brain's (`environment()` in
  `src/hands/brain/process.py`, which keeps hands' own logins out) plus the entry's `env`, and `${VAR}` expands
  against that same environment.
- It is hands' own instance, not the brain's, so state a server keeps per connection (chrome-devtools' selected page,
  say) is not shared with the brain's. Sharing it would need the brain's Claude Code to make the call, rejected above.
- The catalogue is `Own(Tool) | Forwarded(server, definition)`. A forwarded tool is named `<server>__<tool>`,
  mirroring Claude Code's `mcp__<server>__<tool>`, and listed with the upstream's own `inputSchema` and description;
  its call returns the upstream's result as given, its content blocks and `isError` included. hands' own tools keep
  their bare names and today's one-JSON-text-block result, and `Result`'s values are typed as JSON so encoding one
  cannot fail. A name served twice (two servers' names joining to one, an entry named `hands`) leaves both out with
  an event.
- A forwarded call is a `tool.run` event like hands' own, naming its server. A server that dies is an event, its
  tools leave `tools/list` (answered fresh on every request), and a call to one answers an error naming it.
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
tunnel and the Access application hold, which hands cannot read; the refusal events are what say they disagree.

**The Claude app gets instructions.** The brain learns hands from its appended system prompt and its skills; the
Claude app's model gets only what the server sends. The remote door answers `initialize` with `instructions`: a short
text, written for a model reading rather than one speaking, of what hands is, what a session is, how to read a tool
description written for a speaking model, and the order its tools are used in (list, read, stage, send, then `turn`).
It lives beside the brain's instruction and is written with laws:prompt.

**Calls answer inside Cloudflare's window.** Cloudflare drops a proxied request whose origin sends nothing for 100
seconds (a 524). hands' own calls answer in seconds (`start_session`, the slowest, in about 35), but a forwarded call
lasts as long as its upstream takes. The remote door answers every call by 90 seconds: one still running then is
answered with an error saying it is still running and its result will not reach the client, and its event says so.
The act itself is not cancelled, as the brain door cancels none: a half-done act is worse than an unread result.

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
when it is asked for. **One tool, `turn(handle)`, says where the send stands**, read from what hands already keeps,
never from a job table of its own [LAW:one-source-of-truth]. The reducer already files each prompt's words under the
turn that takes them, by their text (`Prompted` and `Submitted` carry the words; `Opened.waiting` holds words filed
under a running turn; `Taken` names the turn Claude Code runs for words it had queued, or for a command), so hands
keeps, for each send, the turn its words were filed under:

| The send | `turn` answers |
| --- | --- |
| typed, and no turn has its words, or they wait behind the running turn | `queued`, with how the session stands now |
| its words are in a turn that is open: their own, or one they were delivered into | `working`, with what the session is doing now |
| that turn stopped on a dialog | `waiting`, with the permission, question, or plan, answered by the permission tools as any dialog is |
| that turn ended (`Untold` or `Told`) | `done`, with the turn told as `tell_turn` tells it, or, until narration has told it, saying it is not told yet and that `read_session` reads it |
| not held by this hands (sent before a restart, or past the bound) | an error naming `read_session` |

hands holds each session's 16 latest sends and the tellings of the turns they name, and lets them go when the session
is gone [LAW:carrying-cost]. Tellings are written by narration, which runs once the voice is up; until it is, or if
it is down, a finished turn's answer is the `done` that says it is not told yet, never a turn that did nothing.
`tell_turn` and `expand` keep reading the session's newest turn: "what did it just do" is a different question from
"how is the prompt I sent going". `turn` moves no focus. Both doors get `turn`, so the voice brain can answer "is it
done yet?" by the prompt it sent, rather than by whichever turn happens to be newest.

**Tasks are not built now.** Anthropic shipped Tasks in the Claude Code CLI on 2026-10-23 (public beta, v2.1.304)
and said other Claude products would follow. Nothing found shows the Claude app declaring the extension yet, and a
server must never return a task to a client that did not declare it in the request's `_meta`. So each call carries
whether its client declared it (`Call.tasks`), and its event records it. The ticket that answers with tasks is in the
order below, pulled once a remote call's event shows the Claude app declaring the extension: a `send_draft` from a
`Remote` caller that declared it is answered with a task whose `taskId` is the send's handle, and whose `tasks/get`
reads `turn`. `queued` and `working` map to `working`, `waiting` to `input_required`, `done` to `completed`. That is a
second way of writing out the same job, with no second store behind it. A `Brain` caller gets no task even if its
Claude Code declares the extension: hands tells the brain each turn as it finishes, and a task would tell it twice.

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
live connect comes straight after the first door that can be connected to, before anything else is built on it.

## Tickets, in order

1. **A tool's voice part is a value of its own.** `then`, the cue, voice-turn-only, and the description's voice
   sentences move off the act; the brain door applies them. Done when the brain is offered the same tools, with the
   same descriptions, and the same calls cue and speak as before, by the existing tests, which are the brain's
   contract here.
2. **Walking skeleton: the remote door, admitting only Access.** `Caller` and `Call` in place of `used`, the remote
   door serving hands' own acts with JWT admission and the `[remote]` setting, a refusal event on both doors. Done
   when, in unit tests with a test-signed key set, a request with a valid JWT lists and calls hands' tools with no
   voice-only tool among them, plays no cue and moves no focus, and each bad JWT is refused with its event.
3. **The Claude app connects through Access.** The setup above, done live and moved to `docs/`. Done when
   `list_sessions` answers from claude.ai and from the iOS app, and a request without Access's header is refused, with
   the events as the evidence. A live run on a ticket of its own, so no code waits on it.
4. **The brain's other MCP servers, forwarded.** The servers in `configdir`'s `State`, the stdio and http clients
   behind a port of hands' own, the catalogue's `Forwarded` entries, the events for a server left out or dead, and the
   check against the brain's offered tools. Done when unit tests over a fake upstream show the names, results passed
   through with their content and `isError`, a dead server's tools leaving the list, and each difference from the
   offered list as an event; and an outside-marked test runs a tiny stdio MCP server through the remote door.
5. **A send is a job: its handle, and `turn` to ask after it.** Done when a client that sends a prompt and asks after
   it is told that send's state, through `queued` when the session was busy, and, once its turn ends, that turn and no
   other.
6. **Instructions for the Claude app**, written for a model that reads hands' tool descriptions and sends a prompt,
   then asks `turn` after it. Done when an eval of the Claude app's model, given the remote door's tools and
   instructions, lists, stages, sends, and asks `turn` in that order and shows the user what a readback says.
7. **Tasks for a client that declares them**, pulled once a remote call's event shows the Claude app declaring the
   extension. Done when such a client's `send_draft` is answered with a task that `tasks/get` follows to `completed`.
