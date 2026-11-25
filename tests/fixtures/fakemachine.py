"""Macs whose Claude Code policy a test chooses, handed to every read of a config directory in place of the one hands
runs on, so no test reads the machine's own /Library."""

from pathlib import Path

from hands.brain import configdir

# A Mac with no Claude Code policy: its policy locations hold nothing, and its account is one.
UNMANAGED = configdir.Machine(Path(__file__).parent / "unmanaged" / "ClaudeCode", Path(__file__).parent / "unmanaged" / "Managed Preferences", "tester")


def managed(root: Path, account: str = "tester") -> configdir.Machine:
    """A Mac whose policy is what a test writes under `root`, the brain running as `account`."""
    return configdir.Machine(root / "ClaudeCode", root / "Managed Preferences", account)
