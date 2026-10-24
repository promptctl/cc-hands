"""Fetch the corpus (wake_training.corpus). Each asset is made under a `.partial` name and renamed into place once
whole, so an asset's path existing is the proof that it is complete; a fetch cut short resumes (a download) or
starts over (anything unpacked or decoded) on the next run.
"""

import io
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import cast

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from numpy.typing import NDArray

from wake_training import audio
from wake_training.corpus import Asset, Clips, Corpus, Download, Folder, Source


def fetch(corpus: Corpus, asset: Asset) -> bool:
    """Fetch `asset` into `corpus` unless it is there already; whether it was fetched."""
    target = corpus.at(asset)
    if target.exists():
        return False
    partial = target.with_name(target.name + ".partial")
    target.parent.mkdir(parents=True, exist_ok=True)
    match asset.fetch:
        case Download(url=url):
            _curl([(url, partial)])
        case Source(url=url, added=added):
            _fresh(partial)
            tarball = partial / ".source.tar.gz"
            _curl([(url, tarball)])
            subprocess.run(["tar", "-xzf", str(tarball), "-C", str(partial), "--strip-components", "1"], check=True)
            tarball.unlink()
            _curl([(source, partial / path) for path, source in added])
        case Folder(dataset=dataset, revision=revision, folder=folder):
            _fresh(partial)
            files = _listing(dataset, revision, folder)
            _curl([(f"https://huggingface.co/datasets/{dataset}/resolve/{revision}/{path}", partial / Path(path).name) for path in files])
        case Clips(dataset=dataset, revision=revision, shards=shards):
            _fresh(partial)
            for shard in shards:
                parquet = partial / ".shard.parquet"
                _curl([(f"https://huggingface.co/datasets/{dataset}/resolve/{revision}/{shard}", parquet)])
                _decode(parquet, partial)
                parquet.unlink()
    partial.rename(target)
    return True


def _fresh(directory: Path) -> None:
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True)


def _curl(files: list[tuple[str, Path]]) -> None:
    """Download each url to its path, all at once; a download cut short earlier continues where it stopped."""
    pairs = [arg for url, path in files for arg in ("--output", str(path), url)]
    subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error", "--create-dirs", "--continue-at", "-", "--parallel", *pairs], check=True)


def _listing(dataset: str, revision: str, folder: str) -> list[str]:
    """The paths of the files in `folder` of a Hugging Face dataset at `revision`."""
    with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{dataset}/tree/{revision}/{folder}") as response:
        # The listing is one page; a folder of more would be fetched as complete with only its first.
        if response.headers.get("Link"):
            raise RuntimeError(f"{dataset}@{revision} lists {folder} in more than one page")
        entries = cast(list[dict[str, str]], json.load(response))
    files = [entry["path"] for entry in entries if entry["type"] == "file"]
    if not files:
        raise RuntimeError(f"{dataset}@{revision} has no files in {folder}")
    return files


def _decode(parquet: Path, into: Path) -> None:
    """Each clip in the parquet's `audio` column, as a 16 kHz mono 16-bit WAV named for its video."""
    table = pq.read_table(parquet, columns=["video_id", "audio"])  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  (untyped in pyarrow)
    rows = cast(list[dict[str, object]], table.to_pylist())  # pyright: ignore[reportUnknownMemberType]  (untyped in pyarrow)
    for row in rows:
        encoded = cast(dict[str, bytes], row["audio"])
        samples, rate = cast(tuple[NDArray[np.float64], int], sf.read(io.BytesIO(encoded["bytes"]), always_2d=True))  # pyright: ignore[reportUnknownMemberType]  (untyped in soundfile)
        audio.write(into / f"{row['video_id']}.wav", samples.mean(axis=1), rate)
