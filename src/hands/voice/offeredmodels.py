"""The models hands runs on itself: the one list config.toml, `hands run --model`, the brain's tool and the phone's page
offer, and the brain is launched on. Its own module, importing nothing of the switch, so the file's parser, the brain's
launch and the switch each read it without loading the others."""

from typing import Literal, get_args

from hands.sessions.payload import one_of

# [LAW:types-are-the-program] the models hands runs on, as one type: the file, --model, the brain's tool and the phone's
# page each take only these, the tool's schema shows the brain them, and the brain is launched on one.
OfferedModel = Literal["claude-sonnet-5-5", "claude-opus-5-5", "claude-fable-5-1"]
OFFERED_MODELS: tuple[OfferedModel, ...] = get_args(OfferedModel)


def parse_model(text: str) -> OfferedModel:
    """[LAW:single-enforcer] the model hands runs on that `text` names, blanks around it aside, however it was given;
    raises Rejected naming the ones it runs on."""
    return one_of(text.strip(), OFFERED_MODELS, "hands runs Claude on")
