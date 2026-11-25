"""A skill Claude Code was asked to run, as the model's call to it names it."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Skill:
    """A skill and its arguments, made by `of`, so that however a call carried its arguments, a Skill holds them trimmed or
    holds none."""

    name: str
    args: str | None

    @classmethod
    def of(cls, name: str, args: object) -> "Skill":
        # [LAW:parse-dont-validate] a call's `args` is a string, padded or blank, or absent; every reader of a Skill is past
        # asking which.
        text = args.strip() if isinstance(args, str) else ""
        return cls(name.strip(), text or None)

    @property
    def typed(self) -> str:
        """The skill as a user types it, `/name args`. [LAW:one-source-of-truth] the one wording of a skill's invocation: the
        page's line, a forked skill's task and a subagent's job all take it from here."""
        return f"/{self.name} {self.args}" if self.args else f"/{self.name}"
