// The wake word on the phone, as a page uses it (hands.voice.phonewake serves it at /detector.js): start the detector, give
// it the microphone's audio as it comes, and be told when it is ready, when it hears the word, and when it cannot listen.
// The models run in a worker (detector-worker.js), so listening never holds up the page; nothing it hears leaves the phone.

// startDetector({ onReady, onFailed, onHeard }) answers { feed, stop }. `onReady` is told the phrase, the threshold and how
// many seconds the page keeps before a hearing; `onHeard` the score of each hearing and the seconds of audio the detector
// had been given by it; `onFailed` why the word cannot be
// heard, in words. feed(samples, rate) hears `samples`, floats of -1 to 1 at `rate`, and takes the buffer they are in: the
// caller does not use it after.
export function startDetector({ onReady, onFailed, onHeard }) {
  // The phone's key goes in the worker's address as it is in the page's, where hands.js reads it.
  const worker = new Worker("/detector-worker.js" + location.hash, { type: "module" });
  worker.onmessage = ({ data }) => {
    switch (data.kind) {
      case "ready": return onReady({ phrase: data.phrase, threshold: data.threshold, keptSeconds: data.keptSeconds });
      case "heard": return onHeard(data.score, data.at);
      case "failed": return onFailed(data.why);
    }
  };
  // A worker that does not start at all, as one whose script is refused or does not parse, says only that it did not.
  worker.onerror = event => onFailed(`the wake word detector did not start: ${event.message || "its script did not load"}`);
  return {
    feed: (samples, rate) => worker.postMessage({ audio: samples, rate }, [samples.buffer]),
    stop: () => worker.terminate(),
  };
}
