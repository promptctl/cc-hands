"""Why a call to the language model failed, as a fact a listener can act on: read off the brain's wire, and said by the
system channel."""

from dataclasses import dataclass

from pipecat.utils.errors import ErrorCategory

from hands.core.wire import UsageLimitReached


@dataclass(frozen=True)
class ModelUnreachable:
    pass


@dataclass(frozen=True)
class ModelFailed:
    category: ErrorCategory


@dataclass(frozen=True)
class ModelReplyEmpty:
    """The model answered with nothing: no words, and no call that would say or do something for it."""


@dataclass(frozen=True)
class ModelActedUntold:
    """The model acted on the user's words and ended its turn with nothing said after it: not even once told to say how it
    went. Whether what it did worked is the model's to know, never this fact's."""


ModelFact = ModelUnreachable | UsageLimitReached | ModelFailed | ModelReplyEmpty | ModelActedUntold


class ModelFault(Exception):
    """A failure of the model already read as the fact it is, as the brain's stage reads it off the wire: carried on the
    stage's error frame so the channel says that fact."""

    def __init__(self, fact: ModelFact) -> None:
        super().__init__(fact)
        self.fact = fact
