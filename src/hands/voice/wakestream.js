// openWakeWord's streaming steps (hands.voice.wake runs the same three models through openwakeword's Python), as one pure
// module: audio at 16 kHz in, the wake word's score out. The three models are what it is given, as functions, so the same
// steps run in a worker over ONNX Runtime Web and in QuickJS in a test over a stand-in that answers as a model does.
//
// openWakeWord's own steps, one 80 ms chunk at a time: the melspectrogram model hears the chunk and the 30 ms before it and
// answers 8 rows of 32 bands; every row is scaled as openWakeWord scales it; the last 76 rows go to the embedding model,
// which answers a vector of 96; and the word's model hears the last few of those vectors and answers a score. Its first
// five scores are zero, as openWakeWord's are, while the stream fills.

export const RATE = 16000;
export const CHUNK = 1280;
export const BANDS = 32, MEL_WINDOW = 76, EMBEDDING_SIZE = 96;
const CONTEXT = 480, ZERO_SCORES = 5;
// What forgetting fills the stream with in place of audio: four seconds of noise, as openWakeWord does, so the first
// seconds of a stream are heard against noise, never against a silence the model was not trained on.
const NOISE_SAMPLES = 4 * RATE, NOISE_AMPLITUDE = 1000, EMBEDDING_STRIDE = 8;

const concatenated = (first, second) => {
  const both = new Float32Array(first.length + second.length);
  both.set(first);
  both.set(second, first.length);
  return both;
};

const flattened = rows => {
  const flat = new Float32Array(rows.length * rows[0].length);
  rows.forEach((row, at) => flat.set(row, at * row.length));
  return flat;
};

// [LAW:parse-dont-validate] a model's answer as the rows it is: one that is not whole rows is thrown, never run on.
function rowsOf(answer, width) {
  if (answer.length === 0 || answer.length % width !== 0) throw new Error(`a model answered ${answer.length} numbers, which are not rows of ${width}`);
  return Array.from({ length: answer.length / width }, (_, at) => answer.slice(at * width, (at + 1) * width));
}

// A wake word's stream over `models` -- { melspectrogram(samples), embedding(window), word(features) }, each a function
// that answers a promise of what the model answers, flat -- whose word's model hears `frames` vectors at a time, and
// which hears the word at a score of `threshold` or over. `random` is where the noise comes from. A word's model that
// hears more vectors than the noise a stream starts with makes is thrown: its first chunk could not be heard.
//
// feed(samples) takes samples at RATE as 16-bit integers held as numbers, of any count, and answers { scores, heard }:
// the score of each whole chunk it completed, up to and including the first at the threshold, and that score or null.
// A hearing forgets everything, as the desk's does, so one saying wakes once; the rest of the samples are dropped.
export async function wakeStream({ melspectrogram, embedding, word, frames, threshold, random = Math.random }) {
  let raw, pending, mel, features, scored;

  const melRows = async samples => rowsOf(await melspectrogram(samples), BANDS).map(row => row.map(band => band / 10 + 2));

  async function forget() {
    raw = new Float32Array(0);
    pending = new Float32Array(0);
    // As openWakeWord's AudioFeatures starts its buffer (openwakeword/utils.py): rows of ones.
    mel = Array.from({ length: MEL_WINDOW }, () => new Float32Array(BANDS).fill(1));
    scored = 0;
    const noise = Float32Array.from({ length: NOISE_SAMPLES }, () => Math.floor(random() * 2 * NOISE_AMPLITUDE) - NOISE_AMPLITUDE);
    const rows = await melRows(noise);
    features = [];
    for (let start = 0; start + MEL_WINDOW <= rows.length; start += EMBEDDING_STRIDE) {
      features.push(await embedding(flattened(rows.slice(start, start + MEL_WINDOW))));
    }
  }

  async function heard(chunk) {
    raw = concatenated(raw, chunk).slice(-(CHUNK + CONTEXT));
    // Only what a model reads is kept: the last MEL_WINDOW rows, and the last `frames` vectors.
    mel = [...mel, ...(await melRows(raw))].slice(-MEL_WINDOW);
    features = [...features, await embedding(flattened(mel))].slice(-frames);
    const score = await word(flattened(features));
    // openWakeWord's own (openwakeword/model.py): the first five scores of a stream are zero while its buffers fill.
    return scored++ < ZERO_SCORES ? 0 : score;
  }

  await forget();
  if (frames > features.length) throw new Error(`the wake word's model hears ${frames} vectors at a time, and a stream starts with ${features.length}`);
  return {
    async feed(samples) {
      const input = concatenated(pending, samples);
      const scores = [];
      let at = 0;
      for (; input.length - at >= CHUNK; at += CHUNK) {
        const score = await heard(input.subarray(at, at + CHUNK));
        scores.push(score);
        if (score >= threshold) {
          await forget();
          return { scores, heard: score };
        }
      }
      pending = input.slice(at);
      return { scores, heard: null };
    },
  };
}

// Hann-windowed sinc, the half of its width in output samples; wide enough that what the lowpass lets through is flat to
// the ear and what it stops above half the output rate is gone.
const HALF_WIDTH = 8;
const sinc = x => x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x);
const hann = x => 0.5 + 0.5 * Math.cos(Math.PI * x / HALF_WIDTH);

// A resampler from `rate` to RATE, for a stream of samples as a page's microphone gives them, -1 to 1 as floats:
// push(samples) answers the output samples the input so far completes, as 16-bit integers held as numbers, which is what
// the models hear. The same samples in any split give the same output. Each output sample is the windowed sinc of the
// input around its moment, scaled to sum to one, so a tone keeps its level; a rate above RATE widens the window to
// lowpass at half of RATE, so what is above it does not fold into what is heard.
export function resampler(rate) {
  // [LAW:parse-dont-validate] a rate is the stream's, said once: one that is not a count of samples a second would never advance the stream, or advance it on NaN.
  if (!(Number.isFinite(rate) && rate > 0)) throw new Error(`a stream's rate is a number of samples a second above zero, not ${rate}`);
  const ratio = rate / RATE;
  const width = Math.max(ratio, 1);
  const reach = Math.ceil(HALF_WIDTH * width);
  // `held` is the input from index `first` on; `next` is the index of the next output sample.
  let held = new Float32Array(0), first = 0, next = 0;
  return {
    push(samples) {
      held = concatenated(held, samples);
      const out = [];
      for (;; next++) {
        const centre = next * ratio;
        if (Math.floor(centre) + reach >= first + held.length) break;
        let sum = 0, weights = 0;
        for (let at = Math.max(first, Math.ceil(centre - reach)); at <= Math.floor(centre) + reach; at++) {
          const distance = (at - centre) / width;
          const weight = Math.abs(distance) < HALF_WIDTH ? sinc(distance) * hann(distance) : 0;
          sum += weight * held[at - first];
          weights += weight;
        }
        out.push(Math.max(-32768, Math.min(32767, Math.round(sum / weights * 32768))));
      }
      const drop = Math.max(0, Math.floor(next * ratio) - reach - first);
      held = held.slice(drop);
      first += drop;
      return Float32Array.from(out);
    },
  };
}
