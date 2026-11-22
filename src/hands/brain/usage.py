"""What the brain's conversation has cost in tokens, read off the wire: the usage the API reports on each reply.

[the design's rule: primary facts from the wire] Claude Code's own figures for it are derivative, and the one its system
reminders show stays fixed for a whole conversation (15000000, hands-misc-itx.xak). Each reply's usage is read from its
message_start and message_delta frames, which the proxy hands its listeners before Claude Code has them, so a tool the brain
calls from its main conversation reads a figure that already holds the request it called the tool in.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

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
class InContext:
    """The brain's main conversation as its latest main-turn reply left it."""

    model: str
    # What the conversation holds now: the latest main turn's request, and what the model wrote back to it, which the
    # next request carries.
    tokens: int


@dataclass(frozen=True)
class Tally:
    """What the conversation the brain is in holds, None until a main turn's reply of it has begun, as a new one after a
    clear has had none; and what the brain has spent since hands started, in every conversation: a resumed
    conversation's earlier runs were not heard."""

    in_context: InContext | None
    spent: Spent


@dataclass(frozen=True)
class _Reply:
    """A reply's usage as its frames have said it so far."""

    exchange: str
    model: str
    usage: Mapping[str, object]
    # The conversation its request was of.
    session: SessionId | None

    @property
    def spent(self) -> Spent:
        return Spent(sum(_count(self.usage.get(name)) for name in INPUT), _count(self.usage.get(OUTPUT)), 1)


def _count(value: object) -> int:
    # A part the API leaves out put nothing in.
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


class Usage:
    """Hears the brain's replies off the wire, of every conversation it is in or was (Kept): `reading` says what the
    conversation it is in now holds, and what the brain has spent."""

    def __init__(self, session: Callable[[], SessionId]) -> None:
        # Which conversation the brain is in now.
        self._session = session
        # The exchanges sent and not yet whole.
        self._sent: dict[str, Sent] = {}
        self._replies: dict[str, _Reply] = {}
        # [LAW:one-source-of-truth] what whole exchanges spent, folded in once each is whole, and the latest main turn of
        # each conversation, so a /resume back reads the one it returns to.
        self._whole = Spent()
        self._latest: dict[SessionId | None, _Reply] = {}

    def hear(self, observed: Observed) -> None:
        match observed:
            case Sent(exchange=exchange) as sent:
                self._sent[exchange] = sent
            case Heard(exchange=exchange, event=MessageStarted(model=model, usage=usage)) if exchange in self._sent:
                sent = self._sent[exchange]
                reply = self._replies[exchange] = _Reply(exchange, model, usage, sent.session)
                if isinstance(sent.kind, MainTurn):
                    self._latest[reply.session] = reply
            case Heard(exchange=exchange, event=MessageDelta(usage=usage)) if exchange in self._replies:
                # [LAW:one-source-of-truth] the wire's own fold, so a reply is counted as the wire assembles it.
                reply = self._replies[exchange] = replace(self._replies[exchange], usage=usage_after(self._replies[exchange].usage, usage))
                # Only the main turn that began last is its conversation as it stands.
                if (latest := self._latest.get(reply.session)) is not None and latest.exchange == exchange:
                    self._latest[reply.session] = reply
            case Exchanged(exchange=exchange) if exchange in self._sent:
                del self._sent[exchange]
                if (reply := self._replies.pop(exchange, None)) is not None:
                    self._whole += reply.spent
            case _:
                pass

    def reading(self) -> Tally:
        latest = self._latest.get(self._session())
        spent = sum((reply.spent for reply in self._replies.values()), self._whole)
        return Tally(None if latest is None else InContext(latest.model, latest.spent.input_tokens + latest.spent.output_tokens), spent)
