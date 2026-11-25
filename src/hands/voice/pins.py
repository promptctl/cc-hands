"""The weights hands loads from the Hugging Face hub, each at the revision measured for it, and the groups a test run
fetches them in.

[LAW:one-source-of-truth] the one home of every pin: the voice modules read their model's repository and revision from
here, tests/fetches.py fetches what is listed here, and CI keys the cache of each group on it. Stdlib only, so CI reads it
(`python3 -I src/hands/voice/pins.py github-output`) without installing the voice stack, which takes longer than the read.
"""

import hashlib
import sys
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Pin:
    """A repository on the hub at one commit, a commit naming its files for good: `file` of it alone, or all of it."""

    repo: str
    revision: str
    file: str | None = None

    @property
    def what(self) -> str:
        """What is fetched, as a run says it and a cache key digests it: a weight moved to another repository, revision or
        file is a new key."""
        return f"{self.repo} at {self.revision}" if self.file is None else f"{self.file} from {self.repo} at {self.revision}"

    @property
    def directory(self) -> str:
        """Where the hub's cache keeps the repository: its own blobs and the snapshots linking into them, so a repository's
        directory is cached and restored whole, apart from every other's."""
        return "models--" + self.repo.replace("/", "--")


STREAMING = Pin("nvidia/nemotron-speech-streaming-en-0.6b", "ebe59e5a817142986528bbbee5dba8db7b38ed50")
DIARIZATION = Pin("mlx-community/Nemotron-3-Diarization", "59ed2dbfc1346dcea9d423c71306a3a2499c568f")
SMART_TURN = Pin("mlx-community/smart-turn-v3", "74af1538ebace37b0fcc667a6a8a3d902d73eee0")
WHISPER = Pin("mlx-community/whisper-large-v3-turbo", "a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb")
# The voiceprint model is one file of a repository of many, so only it is fetched.
VOICEPRINT_FILE = "3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx"
VOICEPRINT = Pin("csukuangfj/speaker-embedding-models", "0743f301363dec56491a490f6d6cbc9d67f9a3bf", VOICEPRINT_FILE)

# What a test run fetches together, and CI caches together: a test needs whole groups. The streaming pair is one group, since
# neither of its models is loaded without the other.
GROUPS: Mapping[str, tuple[Pin, ...]] = {
    "streaming": (STREAMING, DIARIZATION),
    "smart_turn": (SMART_TURN,),
    "whisper": (WHISPER,),
    "voiceprint": (VOICEPRINT,),
}


def fingerprint(held: tuple[Pin, ...]) -> str:
    """A digest of what of each pin is fetched: what CI keys the cache of a group on."""
    return hashlib.sha256("\n".join(pin.what for pin in held).encode()).hexdigest()[:16]


def key(group: str) -> str:
    return fingerprint(GROUPS[group])


def github_output() -> str:
    """Each group's cache key and the paths of its hub directories, in the format a step writes to $GITHUB_OUTPUT, which
    names `<group>_key` and `<group>_paths` to the steps after it."""
    lines: list[str] = []
    for group, pins in GROUPS.items():
        lines.append(f"{group}_key={key(group)}")
        lines.append(f"{group}_paths<<EOF")
        lines.extend(f"~/.cache/huggingface/hub/{pin.directory}" for pin in pins)
        lines.append("EOF")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    if sys.argv[1:] != ["github-output"]:
        sys.exit("usage: pins.py github-output")
    sys.stdout.write(github_output())
