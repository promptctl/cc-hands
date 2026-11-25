// The Call view: what the phone page shows under the call's strip, the transcript and its toggle and the brain's panel.
// The call itself is call.js's and is not here: the view only shows what the call draws, and reads the brain while shown.
import { readBrain } from "/call.js";

// The view's elements are the document's from the start, since the call draws on them as hands tells it things whether
// the view is shown or not; the document holds them in `#held` while another view is shown.
const content = document.getElementById("call-view");
const held = document.getElementById("held");
const transcript = document.getElementById("transcript");

export function mount(root, _route, signal) {
  root.append(content);
  // A node that moved has lost its scroll: the transcript is read from its end.
  transcript.scrollTop = transcript.scrollHeight;
  signal.addEventListener("abort", () => held.append(content), { once: true });
  readBrain(signal);
}
