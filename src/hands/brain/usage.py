"""What the brain's conversation has cost in tokens, read off the wire: the usage the API reports on each reply.

[the design's rule: primary facts from the wire] Claude Code's own figures for it are derivative, and the one its system
reminders show stays fixed for a whole conversation (15000000, hands-misc-itx.xak). Each reply's usage is read from its
message_start and message_delta frames, which the proxy hands its listeners before Claude Code has them, so a tool the brain
calls from its main conversation reads a figure that already holds the request it called the tool in.
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace

from hands.brain.conversation import CurrentSession
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
class Tally:
    """The brain's main conversation as its latest main-turn reply left it, and what it has spent since hands started: a
    resumed conversation's earlier runs were not heard."""

    model: str
    # What the conversation holds now: the latest main turn's request, and what the model wrote back to it, which the
    # next request carries.
    in_context: int
    spent: Spent


@dataclass(frozen=True)
class _Reply:
    """A reply's usage as its frames have said it so far."""

    # The conversation it is a reply in.
    session: SessionId
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
    """Hears the wire for the brain's replies: `reading` says what the conversation it holds has in context, and what
    it has spent."""

    def __init__(self, current: CurrentSession) -> None:
        self._current = current
        # The brain's exchanges sent and not yet whole: the conversation each is of, and whether it is a main turn.
        self._sent: dict[str, tuple[SessionId, bool]] = {}
        self._replies: dict[str, _Reply] = {}
        # [LAW:one-source-of-truth] what whole exchanges spent, folded in once each is whole, and the latest main turn.
        self._whole = Spent()
        self._latest: _Reply | None = None

    def hear(self, observed: Observed) -> None:
        match observed:
            case Sent(exchange=exchange, session=session, kind=kind) if session == self._current.session:
                self._sent[exchange] = (self._current.session, isinstance(kind, MainTurn))
            case Heard(exchange=exchange, event=MessageStarted(model=model, usage=usage)) if exchange in self._sent:
                conversation, main = self._sent[exchange]
                reply = self._replies[exchange] = _Reply(conversation, exchange, model, usage)
                if main:
                    self._latest = reply
            case Heard(exchange=exchange, event=MessageDelta(usage=usage)) if exchange in self._replies:
                # [LAW:one-source-of-truth] the wire's own fold, so a reply is counted as the wire assembles it.
                reply = self._replies[exchange] = replace(self._replies[exchange], usage=usage_after(self._replies[exchange].usage, usage))
                # Only the main turn that began last is the conversation as it stands.
                if self._latest is not None and self._latest.exchange == exchange:
                    self._latest = reply
            case Exchanged(exchange=exchange) if exchange in self._sent:
                del self._sent[exchange]
                if (reply := self._replies.pop(exchange, None)) is not None:
                    self._whole += reply.spent
            case _:
                pass

    def reading(self) -> Tally | None:
        """None until a main turn's reply has begun in the conversation the brain holds: until then it has asked the
        model nothing, whatever a conversation a /clear left had."""
        if self._latest is None or self._latest.session != self._current.session:
            return None
        spent = sum((reply.spent for reply in self._replies.values()), self._whole)
        return Tally(self._latest.model, self._latest.spent.input_tokens + self._latest.spent.output_tokens, spent)
