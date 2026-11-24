// The wake word on the phone, as a page uses it (hands.voice.phonewake serves it at /detector.js): start the detector, give
// it the microphone's audio as it comes, and be told when it is ready, when it hears the word, and when it cannot listen.
// The models run in a worker (detector-worker.js), so listening never holds up the page; nothing it hears leaves the phone.

// The last `length` samples of the microphone made while no turn was open, in the blocks the tap gave them: the second a wake
// sends from before the word. A block that goes over is trimmed at its start, so what it holds is exactly the newest samples.
export class Ring {
  constructor(length) { this.length = length; this.blocks = []; this.held = 0; }
  push(block) {
    this.blocks.push(block); this.held += block.length;
    for (let over = this.held - this.length; over > 0; over = this.held - this.length) {
      const oldest = this.blocks[0];
      if (oldest.length <= over) { this.blocks.shift(); this.held -= oldest.length; }
      else { this.blocks[0] = oldest.subarray(over); this.held -= over; }
    }
  }
  // What it holds, in order, and nothing after.
  empty() { const blocks = this.blocks; this.blocks = []; this.held = 0; return blocks; }
}

// startDetector({ rate, onReady, onFailed, onHeard, onWaiting }) answers { feed, stop }. `rate` is the sample rate of all the
// audio the page will feed. `onReady` is told the phrase, the threshold and how many seconds the page keeps before a hearing;
// `onHeard` the score of each hearing and the seconds of audio the detector had been given by it, counted from the first the
// page fed, whether or not the models had loaded; `onWaiting` why, each time hands gave no answer and the detector asks it
// again, which goes on until hands answers, and null once it has; `onFailed` why the word cannot be heard, in words. feed(samples) hears `samples`,
// floats of -1 to 1 at `rate`, in the order they are fed; the detector takes a copy, so the caller's buffer stays its own.
export function startDetector({ rate, onReady, onFailed, onHeard, onWaiting }) {
  // The phone's key goes in the worker's address as it is in the page's, where hands.js reads it.
  const worker = new Worker("/detector-worker.js" + location.hash, { type: "module" });
  worker.onmessage = ({ data }) => {
    switch (data.kind) {
      case "ready": return onReady({ phrase: data.phrase, threshold: data.threshold, keptSeconds: data.keptSeconds });
      case "heard": return onHeard(data.score, data.at);
      case "waiting": return onWaiting(data.why);
      case "failed": return onFailed(data.why);
      default: return onFailed(`the wake word detector said something this page does not read: ${JSON.stringify(data)}`);
    }
  };
  // A worker that fails outside its own words, as one whose script is refused or does not parse, has only the browser's.
  worker.onerror = event => onFailed(`the wake word detector stopped: ${event.message || "its script could not be loaded"}`);
  worker.postMessage({ rate });
  return {
    feed: samples => {
      // A transferred buffer is detached whole, and `samples` may be a view on a buffer of the caller's: it sends its own.
      const own = samples.slice();
      worker.postMessage({ audio: own }, [own.buffer]);
    },
    stop: () => worker.terminate(),
  };
}
