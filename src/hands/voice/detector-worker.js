// The wake word's models, run on the phone in a worker (hands.voice.phonewake serves it at /detector-worker.js): the Mac's own
// three, on ONNX Runtime Web's WebAssembly build, one thread. detector.js starts it with the rate of the audio it will be
// given; it answers `ready` with the word it listens for, `heard` with a score each time it hears it and how many seconds of
// audio it had been given by then, `waiting` with why each time hands gave it no answer and it asks again (and with none once hands answers), and `failed` with
// why when it cannot listen. Nothing it loads comes from any site but hands.
//
// The phone's key reaches it in its address's fragment, as it reaches a page, so it asks hands through hands.js as a page does.

import { AGAIN_MS, PAGE_OPEN, asked, askedForBytes, pause, unanswered, wording } from "/hands.js";
import { BANDS, EMBEDDING_SIZE, MEL_WINDOW, RATE, resampler, wakeStream } from "/wakestream.js";

// The models by the role each plays in wakeStream's steps, which is the name hands serves each by.
const ROLES = ["melspectrogram", "embedding", "word"];

// A path on hands itself, as the page's own address is: never another site's, so nothing this loads is.
const onHands = value => typeof value === "string" && /^\/(?![/\\])/.test(value);

// [LAW:parse-dont-validate] hands' answer as the settings it is, or thrown.
function settingsOf(body) {
  const { phrase, threshold, kept_seconds, runtime, models } = body ?? {};
  const ok = typeof phrase === "string" && typeof threshold === "number" && typeof kept_seconds === "number" && onHands(runtime);
  if (!ok) throw new Error("hands' answer is not the wake word's settings");
  for (const role of ROLES) {
    if (!onHands(models?.[role])) throw new Error(`hands' answer for the wake word's settings names no model for the ${role}`);
  }
  return { phrase, threshold, keptSeconds: kept_seconds, runtime, models };
}

// hands that did not answer, or could not be reached, is waited for and never given up on: on a cold first run `/wake/settings`
// waits on the models' fetch for longer than any one request is given, and a `failed` is final, so a page that gave up there
// could only be recovered by reloading it. A refusal is hands' word and fails as it is: hands refuses a model it cannot fetch,
// and says why, which asking again would not change. A wait is over when hands answers, said as a `waiting` with no `why`.
async function patiently(ask) {
  let waited = false;
  for (;;) {
    const outcome = await ask();
    if (outcome.kind !== "lost") {
      if (waited) postMessage({ kind: "waiting", why: null });
      return outcome;
    }
    waited = true;
    postMessage({ kind: "waiting", why: outcome.why });
    await pause(AGAIN_MS, PAGE_OPEN);
  }
}

async function fileOf(path) {
  const outcome = await patiently(() => askedForBytes(PAGE_OPEN, path));
  if (outcome.kind !== "answered") throw new Error(unanswered(outcome));
  return outcome.body;
}

async function load() {
  const outcome = await patiently(() => asked(PAGE_OPEN, "GET", "/wake/settings"));
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
  // [LAW:parse-dont-validate] a model's answer is the size it is heard as, or thrown where it was given: the desk refuses a
  // model that scores more than one word as this does (hands.voice.wake), and a wrong-width vector is not carried on to fail
  // in the word model's words.
  const sized = (model, answer, size) => {
    if (answer.length !== size) throw new Error(`the ${model} model answered ${answer.length} numbers, and ${size} are heard`);
    return answer;
  };
  const scored = async features => {
    const answer = await run(word, features, [1, frames, EMBEDDING_SIZE]);
    if (answer.length !== 1) throw new Error(`the wake word model scores ${answer.length} words, and a wake word is one`);
    return answer[0];
  };
  // Asked once, before it is heard with, so a model that scores several words is refused here and not at its first hearing.
  await scored(new Float32Array(frames * EMBEDDING_SIZE));
  const stream = await wakeStream({
    melspectrogram: samples => run(melspectrogram, samples, [1, samples.length]),
    embedding: async window => sized("embedding", await run(embedding, window, [1, MEL_WINDOW, BANDS, 1]), EMBEDDING_SIZE),
    word: scored,
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
  postMessage({ kind: "failed", why: wording(error) });
});
