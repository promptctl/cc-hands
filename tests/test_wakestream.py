"""openWakeWord's streaming steps over the three models (src/hands/voice/wakestream.js), run in QuickJS over stand-ins that
answer as the models do -- their shapes, and a value that says which audio an answer came from -- so what is proven is the
steps: how audio is chunked, what each model is given, what a score is made of, and what forgetting leaves. Whether the
real models score the real recordings is the Chrome test's (tests/test_pages.py).
"""

import json
from importlib import resources
from typing import cast

import numpy as np
import pytest
import quickjs

FRAMES = 16
THRESHOLD = 0.5

STANDINS = """
var calls = {mel: [], embed: [], word: []};
var melAnswer = null, wordScore = 0.9, number = 0;
const random = () => (number++ % 7) / 7;
const rowsFor = n => n < 512 ? 0 : Math.floor((n - 512) / 160) + 1;
const summary = samples => {
  let low = Infinity, high = -Infinity;
  for (const sample of samples) { low = Math.min(low, sample); high = Math.max(high, sample); }
  return {n: samples.length, first: samples[0], last: samples[samples.length - 1], low, high};
};
const models = {
  // A row's first band is the first sample of its hop; the other bands are zero.
  melspectrogram: async samples => {
    calls.mel.push(summary(samples));
    if (melAnswer) return melAnswer(samples);
    const rows = rowsFor(samples.length), answer = new Float32Array(rows * 32);
    for (let row = 0; row < rows; row++) answer[row * 32] = samples[row * 160];
    return answer;
  },
  // A vector whose first number is its window's last row's first band, and whose second is its first row's.
  embedding: async window => {
    calls.embed.push({n: window.length, last: window[75 * 32], first: window[0]});
    const vector = new Float32Array(96);
    vector[0] = window[75 * 32];
    vector[1] = window[0];
    return vector;
  },
  word: async features => {
    calls.word.push({n: features.length, newest: features[features.length - 96]});
    return wordScore;
  },
};
var stream;
function start(threshold) { wakeStream({...models, frames: FRAMES, threshold, random}).then(made => { stream = made; }); }
"""


class Stream:
    """The module run in QuickJS, a stream of it over the stand-in models, and what the models were asked."""

    def __init__(self, threshold: float = THRESHOLD) -> None:
        self._js = quickjs.Context()
        # QuickJS here loads no module a module imports, so what the module exports is handed to the test as a page's
        # import would be.
        module = resources.files("hands.voice").joinpath("wakestream.js").read_text()
        self._js.module(f"{module}\nObject.assign(globalThis, {{wakeStream, resampler, CHUNK}});")
        self._js.eval(STANDINS.replace("FRAMES", str(FRAMES)))
        self.run(f"start({threshold})")

    def run(self, code: str) -> None:
        self._js.eval(code)
        while self._js.execute_pending_job():
            pass

    def value(self, expression: str) -> object:
        return json.loads(cast(str, self._js.eval(f"JSON.stringify({expression})")))

    def fed(self, samples: int, *, start: int = 0) -> dict[str, object]:
        """What the stream answers `samples` samples of a rising count from `start`, at a level the models can tell apart."""
        self.run(f"var answer; stream.feed(Float32Array.from({{length: {samples}}}, (_, at) => {start} + at)).then(said => {{ answer = said; }}, why => {{ answer = {{error: why.message}}; }})")
        return cast(dict[str, object], self.value("answer"))

    def calls(self, model: str) -> list[dict[str, float]]:
        return cast(list[dict[str, float]], self.value(f"calls.{model}"))


@pytest.fixture
def stream() -> Stream:
    return Stream()


def test_the_stream_is_filled_with_noise_before_any_audio_as_openwakeword_fills_it(stream: Stream) -> None:
    [noise] = stream.calls("mel")
    # Four seconds of 16-bit integers under a thousand, heard whole, and 41 windows of 76 rows each embedded.
    assert noise["n"] == 64000
    assert -1000 <= noise["low"] and noise["high"] < 1000
    assert (len(stream.calls("embed")), {each["n"] for each in stream.calls("embed")}) == (41, {76 * 32})
    assert stream.calls("word") == []


def test_audio_is_heard_a_chunk_of_80_ms_at_a_time_with_the_30_ms_before_it_once_there_is_some(stream: Stream) -> None:
    assert stream.fed(1000) == {"scores": [], "heard": None}
    # 1000 and 1000 make one chunk of 1280 and leave 720; no audio came before it, so no context goes with it.
    assert stream.fed(1000, start=1000)["scores"] != []
    assert [each["n"] for each in stream.calls("mel")[1:]] == [1280]
    # The chunk's first sample is the 1281st heard, so the next call is 1280 + 480 with the context first.
    stream.fed(560, start=2000)
    assert [(each["n"], each["first"], each["last"]) for each in stream.calls("mel")[1:]] == [(1280, 0, 1279), (1760, 800, 2559)]


def test_what_the_melspectrogram_model_answers_is_scaled_and_the_last_76_rows_go_to_the_embedding_model(stream: Stream) -> None:
    stream.run("melAnswer = samples => new Float32Array(rowsFor(samples.length) * 32)")
    stream.fed(1280)
    # The first chunk's 5 rows of zeros, scaled to 2, are the last of 76 rows; the 71 before them are the ones the buffer starts with.
    [*_, window] = stream.calls("embed")
    assert (window["n"], window["first"], window["last"]) == (76 * 32, 1, 2)


def test_the_word_model_hears_the_newest_frames_vectors_and_its_first_five_scores_are_zero() -> None:
    # Not heard: the threshold is above what the word model answers.
    quiet = Stream(threshold=0.95)
    scores = quiet.fed(8 * 1280)["scores"]
    assert scores == [0, 0, 0, 0, 0, 0.9, 0.9, 0.9]
    assert {each["n"] for each in quiet.calls("word")} == {FRAMES * 96}
    # The newest vector is the last row of the newest window: the chunk's last hop, a row of 8 for a chunk after the first.
    [*_, last] = quiet.calls("word")
    [*_, window] = quiet.calls("embed")
    assert last["newest"] == window["last"]


def test_a_score_at_the_threshold_is_a_hearing_that_forgets_everything_and_drops_the_audio_after_it(stream: Stream) -> None:
    said = stream.fed(8 * 1280)
    # The sixth chunk is the first to score, and a score of 0.9 is over 0.5; the two chunks after it are not heard.
    assert said == {"scores": [0, 0, 0, 0, 0, 0.9], "heard": 0.9}
    embedded, mel = len(stream.calls("embed")), stream.calls("mel")
    # Forgetting heard four seconds of noise again, and the next audio is heard with nothing before it and five zero scores.
    assert (mel[-1]["n"], embedded - 41 - 6) == (64000, 41)
    assert stream.fed(1280)["scores"] == [0]
    assert stream.calls("mel")[-1]["n"] == 1280


@pytest.mark.parametrize(("score", "heard"), [(THRESHOLD, THRESHOLD), (0.4999, None)])
def test_a_score_at_the_threshold_is_a_hit_and_one_under_it_is_not(stream: Stream, score: float, heard: float | None) -> None:
    stream.run(f"wordScore = {score}")
    assert stream.fed(8 * 1280)["heard"] == heard


def test_audio_not_yet_a_whole_chunk_is_forgotten_with_everything_else(stream: Stream) -> None:
    stream.fed(6 * 1280 + 700)
    stream.fed(1280, start=100000)
    # The 700 samples left over from before the hearing are not heard with the next chunk: it is heard alone.
    assert stream.calls("mel")[-1]["first"] == 100000


def test_an_answer_that_is_not_whole_rows_is_thrown(stream: Stream) -> None:
    stream.run("melAnswer = () => new Float32Array(33)")
    assert stream.fed(1280) == {"error": "a model answered 33 numbers, which are not rows of 32"}


def tone(stream: Stream, rate: int, hertz: float, seconds: float = 1.0, level: float = 0.5, pieces: list[int] | None = None) -> np.ndarray:
    """What a resampler from `rate` to 16 kHz answers a tone of `hertz` heard at `rate`, pushed in `pieces` samples at a time, or all at once."""
    samples = int(rate * seconds)
    stream.run(f"var input = Float32Array.from({{length: {samples}}}, (_, at) => {level} * Math.sin(2 * Math.PI * {hertz} * at / {rate})); var made = resampler({rate}), pushed = []")
    cuts = pieces or [samples]
    start = 0
    for size in [*cuts, samples - sum(cuts)]:
        stream.run(f"pushed.push(...made.push(input.subarray({start}, {start + size})))")
        start += size
    return np.array(stream.value("pushed"), dtype=np.float64)


@pytest.mark.parametrize("rate", [48000, 44100, 16000])
def test_a_tone_comes_out_at_16_khz_at_its_level_and_its_phase(stream: Stream, rate: int) -> None:
    heard = tone(stream, rate, 1000)
    # A second of audio is 16000 samples, less the few at the end that wait for the input after them.
    assert 15980 < len(heard) <= 16000
    expected = 0.5 * 32768 * np.sin(2 * np.pi * 1000 * np.arange(len(heard)) / 16000)
    # Past the first few samples, where the window has no input before it, within 1% of the level.
    assert np.max(np.abs(heard[40:] - expected[40:])) < 0.01 * 0.5 * 32768


def test_a_tone_above_half_the_output_rate_does_not_fold_into_what_is_heard(stream: Stream) -> None:
    # 10 kHz at 48 kHz folds to 6 kHz if nothing stops it; it is well under a twentieth of its level here.
    heard = tone(stream, 48000, 10000)
    assert np.max(np.abs(heard[40:])) < 0.05 * 0.5 * 32768


def test_the_same_samples_in_any_split_give_the_same_output(stream: Stream) -> None:
    whole = tone(stream, 44100, 700)
    split = tone(stream, 44100, 700, pieces=[1, 127, 128, 1000, 7, 4096])
    assert np.array_equal(whole, split)


def test_output_is_16_bit_integers_clamped_where_a_microphone_clips(stream: Stream) -> None:
    heard = tone(stream, 48000, 300, level=1.5)
    assert (np.max(heard), np.min(heard), np.all(heard == np.round(heard))) == (32767, -32768, True)


@pytest.mark.parametrize("rate", ["0", "-48000", "NaN", "Infinity", "undefined"])
def test_a_rate_that_is_not_a_number_of_samples_a_second_above_zero_is_refused_where_the_stream_is_made(stream: Stream, rate: str) -> None:
    stream.run(f"var refused; try {{ resampler({rate}) }} catch (why) {{ refused = why.message }}")
    assert stream.value("refused") == f"a stream's rate is a number of samples a second above zero, not {rate}"


def test_a_word_model_that_hears_more_vectors_than_a_stream_starts_with_is_refused_before_any_audio(stream: Stream) -> None:
    stream.run("var refused; wakeStream({...models, frames: 42, threshold: 0.5, random}).catch(why => { refused = why.message; })")
    assert stream.value("refused") == "the wake word's model hears 42 vectors at a time, and a stream starts with 41"
    # The most it can hear is what the noise makes.
    stream.run("var made = null; wakeStream({...models, frames: 41, threshold: 0.5, random}).then(each => { made = each; })")
    assert stream.value("made !== null") is True
