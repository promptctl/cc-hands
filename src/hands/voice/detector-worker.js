// The wake word's models, run on the phone in a worker (hands.voice.phonewake serves it at /detector-worker.js): the Mac's own
// three, on ONNX Runtime Web's WebAssembly build, one thread. detector.js starts it with the rate of the audio it will be
// given; it answers `ready` with the word it listens for, `heard` with a score each time it hears it and how many seconds of
// audio it had been given by then, and `failed` with why when it cannot listen. Nothing it loads comes from any site but hands.
//
// The phone's key reaches it in its address's fragment, as it reaches a page, so it asks hands through hands.js as a page does.

import { asked, askedForBytes, unanswered } from "/hands.js";
import { BANDS, EMBEDDING_SIZE, MEL_WINDOW, RATE, resampler, wakeStream } from "/wakestream.js";

// The models by the role each plays in wakeStream's steps, which is the name hands serves each by.
const ROLES = ["melspectrogram", "embedding", "word"];

// A path on hands itself, as the page's own address is: never another site's, so nothing this loads is.
const onHands = value => typeof value === "string" && /^\/(?![/\\])/.test(value);

// [LAW:parse-dont-validate] hands' answer as the settings it is, or thrown.
function settingsOf(body) {
  const { phrase, threshold, kept_seconds, runtime, models } = body ?? {};
  const ok = typeof phrase === "string" && typeof threshold === "number" && typeof kept_seconds === "number"
    && onHands(runtime) && typeof models === "object" && models !== null && ROLES.every(role => onHands(models[role]));
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
  const ort = await import(settings.runtime);
  // One thread: the page is not cross-origin isolated. The bundle carries its own glue and asks for its WebAssembly beside
  // itself, which is where hands serves it, so no path is set: one set would have it ask for a glue file hands does not have.
  ort.env.wasm.numThreads = 1;
  const session = async role => ort.InferenceSession.create(await fileOf(settings.models[role]), { executionProviders: ["wasm"] });
  // [LAW:no-ambient-temporal-coupling] one after another: sessions created together share the one thread's WebAssembly heap,
  // and ORT Web rejects the later of two as it reads the earlier's half-built graph ("invalid data type string 'uint8'").
  const melspectrogram = await session("melspectrogram");
  const embedding = await session("embedding");
  const word = await session("word");
  // The word's model hears `frames` of the embedding model's vectors, and says how many it takes.
  const [, frames, width] = word.inputMetadata[0].shape;
  if (!Number.isInteger(frames) || frames < 1 || width !== EMBEDDING_SIZE) {
    throw new Error(`the wake word's model does not take vectors of ${EMBEDDING_SIZE}, some number at a time: ${JSON.stringify(word.inputMetadata[0].shape)}`);
  }
  const run = async (model, data, dims) => (await model.run({ [model.inputNames[0]]: new ort.Tensor("float32", data, dims) }))[model.outputNames[0]].data;
  const stream = await wakeStream({
    melspectrogram: samples => run(melspectrogram, samples, [1, samples.length]),
    embedding: window => run(embedding, window, [1, MEL_WINDOW, BANDS, 1]),
    word: async features => (await run(word, features, [1, frames, EMBEDDING_SIZE]))[0],
    frames,
    threshold: settings.threshold,
  });
  return { settings, stream };
}

// What the page sends, in the order it sent it, each message held until it is read: audio sent while the models load is
// heard once they have, none of it dropped.
function inbox() {
  const held = [];
  let wake = () => {};
  onmessage = ({ data }) => {
    held.push(data);
    wake();
  };
  return (async function* () {
    for (;;) {
      while (held.length > 0) yield held.shift();
      await new Promise(resolve => { wake = resolve; });
    }
  })();
}

// The detector's whole life: the rate of the audio, which is the page's first message; loading; then listening to each
// message of audio in turn. Whatever ends it ends it said, once.
async function run() {
  const messages = inbox();
  const { value: { rate } } = await messages.next();
  const resample = resampler(rate);
  const { settings, stream } = await load();
  postMessage({ kind: "ready", phrase: settings.phrase, threshold: settings.threshold, keptSeconds: settings.keptSeconds });
  let samples = 0;
  for await (const { audio } of messages) {
    const heard16k = resample.push(audio);
    samples += heard16k.length;
    const { heard } = await stream.feed(heard16k);
    if (heard !== null) postMessage({ kind: "heard", score: heard, at: samples / RATE });
  }
}

run().catch(error => {
  onmessage = null;
  postMessage({ kind: "failed", why: error.message });
});
