import io
import tarfile
import wave
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

from wake_training import fetch
from wake_training.corpus import Asset, Corpus, Download, Source


def test_an_asset_is_fetched_once_and_lands_whole(tmp_path: Path) -> None:
    served = tmp_path / "served.bin"
    served.write_bytes(b"features")
    corpus = Corpus(tmp_path / "corpus")
    asset = Asset("features.bin", Download(served.as_uri()))

    assert fetch.fetch(corpus, asset) is True
    assert corpus.at(asset).read_bytes() == b"features"
    assert fetch.fetch(corpus, asset) is False
    assert [path.name for path in corpus.root.iterdir()] == ["features.bin"]


def test_a_source_is_unpacked_without_its_top_directory_with_its_added_files(tmp_path: Path) -> None:
    tree = tmp_path / "piper-2971426a"
    (tree / "piper_sample_generator").mkdir(parents=True)
    (tree / "piper_sample_generator" / "__main__.py").write_text("said = True\n")
    tarball = tmp_path / "source.tar.gz"
    with tarfile.open(tarball, "w:gz") as packed:
        packed.add(tree, arcname=tree.name)
    model = tmp_path / "voices.pt"
    model.write_bytes(b"voices")
    corpus = Corpus(tmp_path / "corpus")
    asset = Asset("piper", Source(tarball.as_uri(), (("models/voices.pt", model.as_uri()),)))

    assert fetch.fetch(corpus, asset) is True
    assert sorted(str(path.relative_to(corpus.at(asset))) for path in corpus.at(asset).rglob("*") if path.is_file()) == [
        "models/voices.pt",
        "piper_sample_generator/__main__.py",
    ]


def test_a_shard_of_clips_is_decoded_to_16_khz_mono_wavs_named_for_their_videos(tmp_path: Path) -> None:
    stereo = io.BytesIO()
    sf.write(stereo, np.full((44_100, 2), 0.25), 44_100, format="FLAC")  # pyright: ignore[reportUnknownMemberType]
    shard = tmp_path / "shard.parquet"
    pq.write_table(pa.table({"video_id": ["abc"], "audio": [{"bytes": stereo.getvalue()}]}), shard)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  (untyped in pyarrow)

    fetch._decode(shard, tmp_path)  # pyright: ignore[reportPrivateUsage]

    with wave.open(str(tmp_path / "abc.wav")) as clip:
        assert (clip.getnchannels(), clip.getframerate(), clip.getnframes()) == (1, 16_000, 16_000)
        samples = np.frombuffer(clip.readframes(clip.getnframes()), dtype=np.int16)
    assert np.median(samples) == 0x2000
