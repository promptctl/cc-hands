"""The model hands runs on and the context the brain's Claude Code says its session has used: reported by hands' plugin in
the brain as its session starts (`hands/brain/plugin/hooks/report.js`), parsed by the brain's process, and said as hands
starts. [LAW:one-way-deps] here in core, which loads no Pipecat, so the brain's process and the voice both read it."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RunningOn:
    """The model the brain runs on, and the tokens of context its session has used as Claude Code counts them: the input of
    its last reply, cached or not. None where Claude Code reported none, as it reports none before a reply has used any, so
    no figure is ever said that it did not report: never the window's size in its place (docs/claude-code-behavior.md)."""

    model: str
    context_used: int | None
