"""What every wake word is trained on, the same for each word: each asset pinned to a revision, and where it lives
under the corpus directory. Fetching them is wake_training.fetch's.

[LAW:one-source-of-truth] an asset's revision is in its path, so a changed pin is a different asset, fetched fresh,
and an asset on disk is always the one its pin names.
"""

from dataclasses import dataclass
from pathlib import Path

_HUGGING_FACE = "https://huggingface.co/datasets"


@dataclass(frozen=True)
class Download:
    """One file at `url`."""

    url: str


@dataclass(frozen=True)
class Source:
    """A gzipped tarball of a source tree at `url`, unpacked without its top directory, with `added` files (a path in
    the tree, and the url it is fetched from) put into it."""

    url: str
    added: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class Folder:
    """Every file in `folder` of the Hugging Face dataset `dataset` at `revision`."""

    dataset: str
    revision: str
    folder: str


@dataclass(frozen=True)
class Clips:
    """The clips in parquet `shards` of a Hugging Face dataset's audio column, decoded into 16 kHz mono 16-bit WAVs:
    the form the trainer mixes in as background."""

    dataset: str
    revision: str
    shards: tuple[str, ...]


# [LAW:one-type-per-behavior] each kind of fetch is a value; fetch.fetch is the one place that performs each kind.
type Fetch = Download | Source | Folder | Clips


@dataclass(frozen=True)
class Asset:
    """`fetch`, which lands at `path` under the corpus directory."""

    path: str
    fetch: Fetch


_FEATURES = "985bf1b47e7f19c07741af82bfe32d5a9dc56096"
_IMPULSES = "b824a1ef2821f112fda0b9cb26e4278c62b425bb"
_AUDIOSET = "0c609e8302cf139307f639c57652032af0a88041"
_PIPER = "2971426a55072f7d22fec416ca7800df8bd23207"
GENERATOR = "en_US-libritts_r-medium.pt"
_VOICES = "v2.0.0"  # the piper-sample-generator release GENERATOR is fetched from

# About 2,000 hours of speech, music and noise from ACAV100M, already openWakeWord features: the negatives that teach
# a model what is not the wake word.
TRAINING_FEATURES = Asset(
    f"features-{_FEATURES[:8]}/openwakeword_features_ACAV100M_2000_hrs_16bit.npy",
    Download(f"{_HUGGING_FACE}/davidscripka/openwakeword_features/resolve/{_FEATURES}/openwakeword_features_ACAV100M_2000_hrs_16bit.npy"),
)
# About 11 hours of speech, music and noise, as features: what the trainer counts false wakes per hour on.
VALIDATION_FEATURES = Asset(
    f"features-{_FEATURES[:8]}/validation_set_features.npy",
    Download(f"{_HUGGING_FACE}/davidscripka/openwakeword_features/resolve/{_FEATURES}/validation_set_features.npy"),
)
# MIT's room impulse responses, which make a clean synthetic clip sound said in a real room.
IMPULSES = Asset(f"impulses-{_IMPULSES[:8]}", Folder("davidscripka/MIT_environmental_impulse_responses", _IMPULSES, "16khz"))
# Four of AudioSet's balanced training shards, about 2,000 ten-second clips of everyday sound, mixed under the clips.
BACKGROUND = Asset(
    f"audioset-{_AUDIOSET[:8]}",
    Clips("agkphysics/AudioSet", _AUDIOSET, tuple(f"data/bal_train/{n:02d}.parquet" for n in range(4))),
)
# piper-sample-generator's source, which carries `piper_train` (its wheel does not, and the generator needs it to
# load), and the LibriTTS-R generator, 904 voices the synthetic clips are said in.
PIPER = Asset(
    f"piper-sample-generator-{_PIPER[:8]}-{_VOICES}-{GENERATOR.removesuffix('.pt')}",
    Source(
        f"https://codeload.github.com/rhasspy/piper-sample-generator/tar.gz/{_PIPER}",
        ((f"models/{GENERATOR}", f"https://github.com/rhasspy/piper-sample-generator/releases/download/{_VOICES}/{GENERATOR}"),),
    ),
)

ASSETS = (TRAINING_FEATURES, VALIDATION_FEATURES, IMPULSES, BACKGROUND, PIPER)


@dataclass(frozen=True)
class Corpus:
    """The corpus directory, where every asset lands."""

    root: Path

    def at(self, asset: Asset) -> Path:
        return self.root / asset.path

    @property
    def generator(self) -> Path:
        return self.at(PIPER) / "models" / GENERATOR
