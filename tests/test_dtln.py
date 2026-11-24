"""DTLN-aec's streaming around the model (hands.voice.dtln): what reaches each part, and where the sound it gives comes
out, run on parts that are arithmetic, so what is tested is the plumbing and not the weights; the weights are held to
the harness's numbers in tests/test_echoscore.py."""

from pathlib import Path

import numpy as np
import pytest
from fakefetch import Network

from hands.voice.dtln import BLOCK, MAX_DELAY, SHIFT, Dtln, Stages, fetched
from hands.voice.pinnedfile import Mismatch
from hands.voice.dtln import _EARLY, _REPLAYED, _SETTLED  # pyright: ignore[reportPrivateUsage]  (the constants the plumbing is held to)

BINS = BLOCK // 2 + 1


def passing(seen: list[np.ndarray] | None = None) -> Stages:
    """Parts that leave the microphone as it is (a mask of ones, and a refinement taking a quarter of each block, as four
    blocks overlap each sample), and say each state they were given."""

    def mask(microphone: np.ndarray, reference: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if seen is not None:
            seen.append(state.copy())
        return np.ones(BINS, np.float32), state + 1

    def refine(masked: np.ndarray, reference: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return masked / 4, state

    return Stages(mask, refine, (1,), (1,))


def referencing() -> Stages:
    """Parts that give the reference they were given, so the sound out is the reference read as the model reads it."""

    def mask(microphone: np.ndarray, reference: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(BINS, np.float32), state

    def refine(masked: np.ndarray, reference: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return reference / 4, state

    return Stages(mask, refine, (1,), (1,))


def streamed(dtln: Dtln, microphone: np.ndarray, reference: np.ndarray, delay: int | None, frame: int) -> np.ndarray:
    return np.concatenate([dtln.cancel(microphone[at : at + frame], reference[at : at + frame], delay) for at in range(0, len(microphone), frame)])


@pytest.mark.parametrize("frame", [160, 128, 100, 640, 1])
def test_the_sound_comes_out_as_it_went_in_the_latency_later_in_frames_of_any_length(frame: int) -> None:
    microphone = np.random.default_rng(1).normal(0, 0.1, 6400).astype(np.float32)
    out = streamed(Dtln(passing()), microphone, np.zeros_like(microphone), 0, frame)
    assert len(out) == len(microphone)
    assert np.allclose(out[Dtln.LATENCY :], microphone[: len(microphone) - Dtln.LATENCY], atol=1e-5)
    assert np.allclose(out[: Dtln.LATENCY], 0, atol=1e-6)


@pytest.mark.parametrize("frame", [160, 100])
def test_the_reference_is_read_the_delay_back_less_the_early_it_is_read_by(frame: int) -> None:
    reference = np.zeros(8000, np.float32)
    reference[2000] = 1.0
    out = streamed(Dtln(referencing()), np.zeros_like(reference), reference, 1000, frame)
    assert np.flatnonzero(np.abs(out) > 0.5).tolist() == [2000 + 1000 - _EARLY + Dtln.LATENCY]


def test_a_delay_shorter_than_the_early_reads_the_reference_as_it_came() -> None:
    reference = np.zeros(4000, np.float32)
    reference[1000] = 1.0
    out = streamed(Dtln(referencing()), np.zeros_like(reference), reference, 0, 160)
    assert np.flatnonzero(np.abs(out) > 0.5).tolist() == [1000 + Dtln.LATENCY]


def test_with_no_delay_the_microphone_is_heard_as_it_was_and_the_model_is_not_run() -> None:
    def refused(*_: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise AssertionError("the model was run with no echo path to align the reference by")

    microphone = np.random.default_rng(2).normal(0, 0.1, 3200).astype(np.float32)
    out = streamed(Dtln(Stages(refused, refused, (1,), (1,))), microphone, microphone, None, 160)
    assert np.allclose(out[Dtln.LATENCY :], microphone[: len(microphone) - Dtln.LATENCY], atol=1e-6)


def test_the_model_begins_afresh_where_the_delay_moves_further_than_it_settles_and_not_before() -> None:
    seen: list[np.ndarray] = []
    dtln = Dtln(passing(seen))
    quiet = np.zeros(SHIFT * 4, np.float32)
    began: list[int] = []
    for delay in (1000, 1000 + _SETTLED, 1000 + _SETTLED + 1, 1000 + _SETTLED + 1 + _SETTLED + 1):
        before = len(seen)
        dtln.cancel(quiet, quiet, delay)
        began.append(int(seen[before][0]))
    # The state each call began from: counted up from the last call's, or zero where the delay moved too far.
    assert began == [0, 4, 0, 0]


def test_the_model_begun_afresh_reads_the_last_of_the_microphone_again_before_the_frame_it_is_given() -> None:
    calls: list[int] = []

    def counting(microphone: np.ndarray, reference: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        calls.append(0)
        return np.ones(BINS, np.float32), state

    def refine(masked: np.ndarray, reference: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return masked / 4, state

    dtln = Dtln(Stages(counting, refine, (1,), (1,)))
    quiet = np.zeros(SHIFT * 4, np.float32)
    for _ in range(40):
        dtln.cancel(quiet, quiet, None)
    assert not calls
    dtln.cancel(quiet, quiet, 1000)
    assert len(calls) == _REPLAYED + 4
    dtln.cancel(quiet, quiet, 1000 + _SETTLED)
    assert len(calls) == _REPLAYED + 4 + 4


def test_the_model_begun_afresh_reads_the_microphone_again_from_silence_not_from_the_end_of_the_stream_it_left() -> None:
    first: list[float] = []

    def mask(microphone: np.ndarray, reference: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        first.append(float(microphone[0]))  # the DC bin: the sum of the block's samples
        return np.ones(BINS, np.float32), state

    def refine(masked: np.ndarray, reference: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return masked / 4, state

    dtln = Dtln(Stages(mask, refine, (1,), (1,)))
    loud = np.ones(SHIFT * 4, np.float32)
    for _ in range(40):
        dtln.cancel(loud, loud, None)
    dtln.cancel(loud, loud, 1000)
    # The first hop replayed is one hop of sound after three of silence, whatever the stream was before it.
    assert first[0] == pytest.approx(SHIFT)


def test_a_delay_reaching_further_back_than_the_echo_path_looks_is_refused() -> None:
    with pytest.raises(ValueError, match="not within"):
        Dtln(passing()).cancel(np.zeros(160, np.float32), np.zeros(160, np.float32), MAX_DELAY + 1)


def test_a_frame_of_microphone_and_a_frame_of_reference_of_different_lengths_is_refused() -> None:
    with pytest.raises(ValueError, match="of reference"):
        Dtln(passing()).cancel(np.zeros(160, np.float32), np.zeros(159, np.float32), 0)


def test_a_download_that_is_not_the_pinned_file_is_refused_by_name_and_leaves_no_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    Network(monkeypatch, b"not the model")
    with pytest.raises(Mismatch, match="dtln_aec_512_1.tflite"):
        fetched(tmp_path)
    assert list(tmp_path.iterdir()) == []
