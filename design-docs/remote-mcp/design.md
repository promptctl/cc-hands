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
| hands' tools (`mcp__hands__*`) | `--mcp-config`, from `serve_mcp` | yes, as is |
| its other MCP servers (`mcp__vhid__*` today) | `mcpServers` in `<hands home>/brain/.claude.json`, which `hands login` set up | yes, proxied |
| Claude Code's built-ins (Bash, Read, …) | Claude Code | no: not MCP, and not asked for |

## Shape

```
 Claude app ──HTTPS──► Cloudflare edge ──────────► cloudflared ──► 127.0.0.1:<remote port>/mcp ─┐
 (Anthropic cloud)     Access: Managed OAuth        (launchd,       remote door:                 │
                       policy: Brandon only         outbound)       Cf-Access-Jwt-Assertion     │
                       adds Cf-Access-Jwt-Assertion                  checked                     │
                                                                                                 ▼
 brain (Claude Code) ─────────────────────────────► 127.0.0.1:<free port>/mcp ──►  one JSON-RPC handler ──► hands' tools
                                                    brain door: bearer token        (mcp.py)        └─────► proxied servers
                                                                                                           (stdio children,
                                                                                                            http clients)
```

One handler, two doors. A door is a value: how it admits a request, which tools it lists, and who its callers are.

```python
@dataclass(frozen=True)
class Door:
    admit: Callable[[web.Request], Caller | Refusal]   # the one check this door makes [LAW:single-enforcer]
    tools: Sequence[Tool]                               # what tools/list says and tools/call can reach
    instructions: str | None                            # initialize's `instructions`, for a client that has no other

Caller = Brain | Remote
@dataclass(frozen=True)
class Brain:  tool_use: str | None       # Claude Code's claudecode/toolUseId: the call's span is the turn's child
@dataclass(frozen=True)
class Remote: email: str                 # from the Access JWT: the call is a trace of its own, audited with who
```

`Caller` replaces today's `used: str | None`. The brain door's warning for a call naming no tool_use stays the brain
door's; a remote call is never "cut off from its turn", because it has none.

## Decisions

**The remote door serves the brain's tool set, defined by the brain's own setup.** "The tools the brain has access
to" has one source of truth: hands' tool list as `serve_mcp` is handed it, plus the brain's other servers, minus the
brain's `permissions.deny` rules in `<hands home>/brain/settings.json` (today `mcp__hands__stay_silent`, the one tool
with no meaning outside a voice turn). hands already reads that file (`settings_mode`), so a deny added for the brain
holds on both doors with no second list. The Claude app's own per-tool approval prompts stand where the brain's
`auto` mode stands.

**Tools that end in speech keep working unchanged.** A `then="silence"` tool (stage_draft, use_model, playback)
returns `read_back`'s `{"says", "as"}`; hands speaks it only because the brain's stage reads it off the brain's wire.
A remote call never crosses that wire, so the Claude app gets the text and the room hears nothing. No branch is
needed for that.

**A remote call owes no cue.** `cued` plays hands' "working" sound for the voice turn it acts in. Today it is baked
into the tool list before `serve_mcp` sees it, so the composition moves: run.py builds the audited tools once, the
brain door gets them cued, the remote door gets them uncued.

**Both doors drive one hands.** Focus, drafts, overlays, and sessions are shared state, so a draft staged from the
phone is the draft the voice brain sends, and a focus moved from the app is the focus a spoken "send it" lands on.
That is the point of a single hands, not a race to guard against.

**The other servers are proxied by hands as their own MCP client, configured from the brain's `.claude.json`.**
Rejected alternative: making the brain execute the call (hands rewriting a response to inject a `tool_use`). That
needs a main turn of the brain, which would occupy it, collide with spoken turns, and write into its transcript.
Running `claude mcp serve` is rejected by the hard rule against starting Claude Code for what hands can do itself.

- Read `mcpServers` from the top level and from `projects[<the brain's cwd>]` of the brain's `.claude.json`, skipping
  `hands`. Parse each into `Stdio(command, args, env) | Http(url, headers)` [LAW:parse-dont-validate]. An entry hands
  cannot run (an `sse` server, one needing OAuth tokens that live in Claude Code's keychain, a `${VAR}` that is
  unset) is left out with one event saying which and why, never half-started.
- hands starts each stdio server once, as a child of the daemon, for the life of the remote door: its own instance,
  not the brain's. State that a server keeps per connection (chrome-devtools' selected page, say) is therefore not
  shared with the brain's. That is accepted.
- Proxied tools are named `<server>__<tool>`, mirroring Claude Code's `mcp__<server>__<tool>`; hands' own keep
  their bare names. A call is forwarded with its arguments and its result returned as the server gave it, `isError`
  included. A server that has died answers its calls with an error naming it.
- Use the official `mcp` Python SDK for the client side (stdio framing, request correlation, server-to-client pings).
  hands' server side stays hand-written: it works, it is tested, and the SDK would change nothing a client sees.

**Access checks the person; hands checks Access.** Cloudflare Access with Managed OAuth is the OAuth authorization
server: it answers the Claude app's unauthenticated request with a 401 and `WWW-Authenticate`, serves the RFC 8414 /
RFC 9728 metadata, runs dynamic client registration and the browser login, and issues opaque tokens. On every
admitted request it adds `Cf-Access-Jwt-Assertion`. The remote door admits a request only when that JWT verifies:
RS256 against the team's keys at `https://<team>.cloudflareaccess.com/cdn-cgi/access/certs`, `iss` the team domain,
`aud` the application's AUD tag, not expired. This makes the door safe even to a local process that reaches its port
directly, and the caller's `email` claim is who the audit names. PyJWT (`cryptography` is already in the lock) does
the verifying; the key set is a port, so the unit tests sign with a key of their own and nothing reaches Cloudflare.

**The door stays on loopback; cloudflared is not hands'.** The remote door binds `127.0.0.1` on a fixed port, since a
tunnel needs a stable origin. `cloudflared` runs as its own launchd service (`cloudflared service install <token>`),
like any tunnel on the machine; hands neither starts nor supervises it. Tailscale, which already serves the phone page,
cannot do this job: the Claude app connects from Anthropic's cloud, not from Brandon's devices.

**Configuration.** One `[remote]` table in hands' `config.toml`, read only by the daemon:

```toml
[remote]
port = 8765                                   # the tunnel's origin: http://127.0.0.1:8765
team = "https://<team>.cloudflareaccess.com"  # the JWT's issuer, and where its keys are
aud = "<the Access application's AUD tag>"
```

No table, no remote door. A table that does not parse stops hands' start, saying why, as every other setting does.

**The Claude app gets instructions.** The brain learns hands from its appended system prompt and its skills; the
Claude app's model gets only what the server sends. The remote door answers `initialize` with `instructions`: a short
text, written for a model reading rather than one speaking, of what hands is, what a session is, and the order its
tools are used in (list, focus, read, stage, send). It lives beside the brain's instruction and is written with
laws:prompt.

**Calls answer inside Cloudflare's window.** Cloudflare drops a proxied request whose origin sends nothing for 100
seconds (a 524). The tools' own events already time every call, so the audit says whether any comes near it. One
that does is answered as an SSE stream for that call, which the transport allows; nothing is built for it until one
does.

## Cloudflare and the Claude app, set up once

1. A tunnel for the machine, its public hostname (say `hands.<domain>`) pointing at `http://127.0.0.1:<port>`.
2. An Access application for that hostname, an Allow policy naming Brandon's email, and Managed OAuth on, with
   `https://claude.ai/api/mcp/auth_callback` among the allowed redirect URIs for dynamically registered clients.
   Short access-token lifetime, long grant session (the docs' advice), so a revoked login ends within minutes.
3. If Bot Fight Mode or a WAF rule covers the zone, a Skip rule for the hostname: Claude's backend arrives as
   `Claude-User` from US addresses and is otherwise blocked before Access sees it.
4. In the Claude app: Settings → Connectors → Add custom connector, URL `https://hands.<domain>/mcp`, OAuth client
   "Register automatically".

These go in `docs/` once they are proven, not before.

## The risk, and the order that retires it first

Claude's web and mobile connectors have failed against Access Managed OAuth while Claude Code connected to the same
URL (anthropics/claude-ai-mcp#410, June 2026, closed as not planned). The causes found there were the RFC 8707
`resource` parameter, which Access requires on `/authorize`, and protected-resource metadata that offered
`cloudflared` login ahead of `oauth`. Whether today's Claude app and today's Access get along is not known from
here.

Nothing in hands depends on the answer: whichever Access front works, the origin contract is one JWT header. So the
first ticket proves the path end to end before anything else is built on it.

## Tickets, in order

1. **Walking skeleton: the Claude app lists Brandon's sessions through Access.** The `Door` split, the remote door
   with hands' own tools and JWT admission, the `[remote]` setting, and the setup above, proven by calling
   `list_sessions` from claude.ai and from the iOS app, and by a request without a valid JWT being refused. Unit
   tests for admission with a test-signed key set; the live connect is its own ticket with its evidence pointer, per
   the standing rule that a live run is no prerequisite for code.
2. **A remote call is its own trace, names its caller, and owes no cue.** `Caller` in place of `used`; the cue moved
   to the brain door; the deny rules read from the brain's settings and applied to both doors.
3. **The brain's other MCP servers, proxied.** Parsing the brain's `.claude.json`, the stdio and http clients, the
   `<server>__<tool>` names, the event for a server left out. An outside-marked test runs a tiny stdio MCP server
   through the remote door.
4. **Instructions for the Claude app.**
