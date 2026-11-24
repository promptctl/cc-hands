// The wake word's models, run on the phone in a worker (hands.voice.phonewake serves it at /detector-worker.js): the Mac's own
// three, on ONNX Runtime Web's WebAssembly build, one thread. detector.js starts it; it answers `ready` with the word it
// listens for, `heard` with a score each time it hears it and how many seconds of audio it had been given by then, and
// `failed` with why when it cannot listen. Nothing it loads comes from any site but hands.
//
// The phone's key reaches it in its address's fragment, as it reaches a page, so it asks hands through hands.js as a page does.

import { asked, askedForBytes, unanswered } from "/hands.js";
import { RATE, resampler, wakeStream } from "/wakestream.js";

const MODELS = ["melspectrogram.onnx", "embedding.onnx", "word.onnx"];

// [LAW:parse-dont-validate] hands' answer as the settings it is, or thrown.
function settingsOf(body) {
  const { phrase, threshold, kept_seconds, runtime, models } = body ?? {};
  const ok = typeof phrase === "string" && typeof threshold === "number" && typeof kept_seconds === "number"
    && typeof runtime === "string" && runtime.endsWith("/") && typeof models === "object" && models !== null
    && MODELS.every(name => typeof models[name] === "string");
  if (!ok) throw new Error("hands' answer is not the wake word's settings");
  return { phrase, threshold, keptSeconds: kept_seconds, runtime, models };
}

async function fileOf(path) {
  const outcome = await askedForBytes(path);
  if (outcome.kind !== "answered") throw new Error(unanswered(outcome));
  return outcome.body;
}

async function load() {
  const outcome = await asked("GET", "/wake/settings");
  if (outcome.kind !== "answered") throw new Error(unanswered(outcome));
  const settings = settingsOf(outcome.body);
  const ort = await import(settings.runtime + "ort.wasm.bundle.min.mjs");
  // One thread: the page is not cross-origin isolated. The bundle carries its own glue and asks for its WebAssembly beside
  // itself, which is where hands serves it, so no path is set: one set would have it ask for a glue file hands does not have.
  ort.env.wasm.numThreads = 1;
  const session = async name => ort.InferenceSession.create(await fileOf(settings.models[name]), { executionProviders: ["wasm"] });
  // One after another: sessions created together share the one thread's WebAssembly heap, and ORT Web rejects the later
  // of two as it reads the earlier's half-built graph ("invalid data type string 'uint8'").
  const melspectrogram = await session("melspectrogram.onnx");
  const embedding = await session("embedding.onnx");
  const word = await session("word.onnx");
  // How many of the embedding model's vectors the word's model hears is what the word's model says it takes.
  const frames = word.inputMetadata[0].shape[1];
  if (!Number.isInteger(frames) || frames < 1) throw new Error(`the wake word's model does not say how many frames it hears: ${JSON.stringify(word.inputMetadata[0].shape)}`);
  const run = async (model, data, dims) => (await model.run({ [model.inputNames[0]]: new ort.Tensor("float32", data, dims) }))[model.outputNames[0]].data;
  const stream = await wakeStream({
    melspectrogram: samples => run(melspectrogram, samples, [1, samples.length]),
    embedding: window => run(embedding, window, [1, 76, 32, 1]),
    word: async features => (await run(word, features, [1, frames, 96]))[0],
    frames,
    threshold: settings.threshold,
  });
  return { settings, stream };
}

// What audio does: nothing until the models are loaded, and nothing once they cannot be run.
let listen = async () => {};
// Audio is heard in the order it came, one message at a time.
let queue = Promise.resolve();
const failed = error => {
  listen = async () => {};
  postMessage({ kind: "failed", why: error.message });
};

load().then(({ settings, stream }) => {
  let resample = null, rate = 0, samples = 0;
  listen = async ({ audio, rate: at }) => {
    if (at !== rate) {
      resample = resampler(at);
      rate = at;
    }
    const heard16k = resample.push(audio);
    samples += heard16k.length;
    const { heard } = await stream.feed(heard16k);
    if (heard !== null) postMessage({ kind: "heard", score: heard, at: samples / RATE });
  };
  postMessage({ kind: "ready", phrase: settings.phrase, threshold: settings.threshold, keptSeconds: settings.keptSeconds });
}, failed);

onmessage = event => {
  queue = queue.then(() => listen(event.data)).catch(failed);
};
