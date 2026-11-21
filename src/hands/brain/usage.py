"""What the brain's conversation has cost in tokens, read off the wire: the usage the API reports on each reply.

[the design's rule: primary facts from the wire] Claude Code's own figures for it are derivative, and the one its system
reminders show stays fixed for a whole conversation (15000000, hands-misc-itx.xak). Each reply's usage is read from its
message_start and message_delta frames, which the proxy hands its listeners before Claude Code has them, so a tool the brain
calls from its main conversation reads a figure that already holds the request it called the tool in.
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace

from hands.brain.holder import SessionHolder, is_held
from hands.core.session import SessionId
from hands.core.wire import Exchanged, Heard, MainTurn, MessageDelta, MessageStarted, Observed, Sent, usage_after

# The three parts of what a request put into the model's context, as the API counts them.
INPUT = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
OUTPUT = "output_tokens"


@dataclass(frozen=True)
class Spent:
    """Tokens over some requests: what they put in, what the model wrote back, and how many replies said so."""

    input_tokens: int = 0
    output_tokens: int = 0
    replies: int = 0

    def __add__(self, other: "Spent") -> "Spent":
        return Spent(self.input_tokens + other.input_tokens, self.output_tokens + other.output_tokens, self.replies + other.replies)


@dataclass(frozen=True)
class Context:
    """The brain's main conversation as its latest main-turn reply left it: the model, and what it holds now, the latest
    main turn's request and what the model wrote back to it, which the next request carries."""

    model: str
    tokens: int


@dataclass(frozen=True)
class Tally:
    """The conversation the brain holds now, and what the brain has spent since hands started, in every conversation it
    held: a resumed conversation's earlier runs were not heard."""

    # None until a main turn of the conversation has a reply begun: one just begun, by a /clear, has asked the model nothing.
    context: Context | None
    spent: Spent


@dataclass(frozen=True)
class _Reply:
    """A reply's usage as its frames have said it so far."""

    exchange: str
    model: str
    usage: Mapping[str, object]

    @property
    def spent(self) -> Spent:
        return Spent(sum(_count(self.usage.get(name)) for name in INPUT), _count(self.usage.get(OUTPUT)), 1)


def _count(value: object) -> int:
    # A part the API leaves out put nothing in.
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


class Usage:
    """Hears the wire for the brain's replies: `reading` says what the conversation it holds now holds, and what it has spent."""

    def __init__(self, brain: SessionHolder) -> None:
        self._brain = brain
        # The brain's exchanges sent and not yet whole, and of them its main turns, by the session each was sent in.
        self._sent: set[str] = set()
        self._main: dict[str, SessionId] = {}
        self._replies: dict[str, _Reply] = {}
        # [LAW:one-source-of-truth] what whole exchanges spent, folded in once each is whole, and each conversation's
        # latest main turn, kept under its session, so a /clear or a /resume reads the conversation it moved to.
        self._whole = Spent()
        self._latest: dict[SessionId, _Reply] = {}

    def hear(self, observed: Observed) -> None:
        match observed:
            case Sent(exchange=exchange, session=session, kind=kind) if is_held(session, self._brain):
                self._sent.add(exchange)
                if isinstance(kind, MainTurn):
                    self._main[exchange] = session
            case Heard(exchange=exchange, event=MessageStarted(model=model, usage=usage)) if exchange in self._sent:
                reply = self._replies[exchange] = _Reply(exchange, model, usage)
                if exchange in self._main:
                    self._latest[self._main[exchange]] = reply
            case Heard(exchange=exchange, event=MessageDelta(usage=usage)) if exchange in self._replies:
                # [LAW:one-source-of-truth] the wire's own fold, so a reply is counted as the wire assembles it.
                reply = self._replies[exchange] = replace(self._replies[exchange], usage=usage_after(self._replies[exchange].usage, usage))
                # Only the main turn that began last is its conversation as it stands.
                self._latest = {session: reply if latest.exchange == exchange else latest for session, latest in self._latest.items()}
            case Exchanged(exchange=exchange) if exchange in self._sent:
                self._sent.discard(exchange)
                self._main.pop(exchange, None)
                if (reply := self._replies.pop(exchange, None)) is not None:
                    self._whole += reply.spent
            case _:
                pass

    def reading(self) -> Tally:
        spent = sum((reply.spent for reply in self._replies.values()), self._whole)
        session = self._brain.session
        latest = None if session is None else self._latest.get(session)
        context = None if latest is None else Context(latest.model, latest.spent.input_tokens + latest.spent.output_tokens)
        return Tally(context, spent)
