// The Call view: what the phone shows under the call's strip, the transcript with its toggle and the brain's panel. The call
// itself is call.js's: it writes the transcript whether this view is shown or not, so the view's elements are the document's
// from the start and the view shows them, hiding them as it is left; the brain's panel is the view's own, read while shown.
import {PAGE_OPEN, asked, handsNow, reading, unanswered} from "/hands.js";

const content = document.getElementById("call-view");
const toggle = document.getElementById("shown");
const transcript = document.getElementById("transcript");
const doing = document.getElementById("doing");
const calls = document.getElementById("calls");
const stopButton = document.getElementById("stop");
const brainSaid = document.getElementById("brain-said");
// hands answers a Stop at once: one unanswered for this long went to an address the phone can no longer reach. What a Stop
// said is said of that tap alone, so it is shown for a while and then taken away: the line above says how the brain stands
// from then on.
const STOP_MS = 10000, SAID_MS = 30000;

// The transcript is hidden until asked for, and stays as this phone last left it.
function showTranscript(open) {
  transcript.hidden = !open;
  toggle.setAttribute("aria-expanded", String(open));
  toggle.textContent = open ? "Hide transcript" : "Show transcript";
  if (open) transcript.scrollTop = transcript.scrollHeight;
}
try { showTranscript(localStorage.getItem("transcript") === "shown"); } catch { showTranscript(false); }
toggle.addEventListener("click", () => {
  showTranscript(transcript.hidden);
  try { localStorage.setItem("transcript", transcript.hidden ? "hidden" : "shown"); } catch { /* kept for this page only */ }
});

// The brain's work as hands last said it, its calls timed by hands' clock; whether a Stop is still being answered; what
// the last Stop said, until SAID_MS have passed; and why the brain could not be read, until it is read again.
const IDLE = {doing: "idle", calls: []};
let work = IDLE;
let stopping = false;
let said = null;
let unread = null;

function sayOfBrain(text, bad = false) { brainSaid.textContent = text; brainSaid.classList.toggle("bad", bad); }

// The line that says what the brain is doing, and for how long: redrawn each second, as the time is.
function drawDoing() {
  const now = handsNow();
  const [first, ...rest] = work.calls;
  const [what, since] = work.doing !== "working" ? ["Idle", null]
    : first === undefined ? ["Thinking", work.since]
    : [rest.length ? `Running ${rest.length + 1} calls` : `Running ${first.tool}`, first.since];
  doing.textContent = what;
  if (since !== null) doing.append(Object.assign(document.createElement("span"), {className: "for", textContent: ` · ${Math.max(0, Math.round(now - since))} s`}));
}

// A call's input as a glance reads it: its values, each a line, as the call gave them.
function inputOf(call) {
  return "unreadable" in call ? `input could not be read: ${call.unreadable}`
    : Object.values(call.input).map((value) => typeof value === "string" ? value : JSON.stringify(value)).join("\n");
}

// [LAW:dataflow-not-control-flow] the whole section is drawn the same way whatever the brain is doing: its calls are
// whatever it is running, none at all while it is idle or thinking.
function drawBrain() {
  drawDoing();
  calls.replaceChildren(...work.calls.map((call) => {
    const item = document.createElement("li");
    item.append(
      Object.assign(document.createElement("p"), {className: "tool", textContent: call.tool}),
      Object.assign(document.createElement("p"), {className: `input${"unreadable" in call ? " bad" : ""}`, textContent: inputOf(call)}),
    );
    return item;
  }));
  stopButton.disabled = stopping || work.doing !== "working";
  const told = said ?? unread ?? {text: "", bad: false};
  sayOfBrain(told.text, told.bad);
}

// Stop ends the turn the page shows, and that turn alone: one the brain went on to after it is never stopped by it. It is a
// deed, done whichever view is shown, so leaving the view does not take it back.
stopButton.addEventListener("click", async () => {
  stopping = true;
  drawBrain();
  try {
    const outcome = await asked(PAGE_OPEN, "POST", "/brain/stop", {turn: work.turn}, STOP_MS);
    sayOfStop(outcome.kind === "answered" ? {text: outcome.body.readback, bad: false} : {text: `Could not stop it: ${unanswered(outcome)}`, bad: true});
  } catch (error) {
    sayOfStop({text: `Could not stop it: ${error.message}`, bad: true});
  } finally {
    stopping = false;
    drawBrain();
  }
});

// What a Stop said, taken away SAID_MS later unless a later Stop has said something since.
function sayOfStop(stopSaid) {
  said = stopSaid;
  setTimeout(() => { if (said === stopSaid) { said = null; drawBrain(); } }, SAID_MS);
}

export function mount(_root, _route, signal) {
  content.hidden = false;
  // Lines were written while the view was away, and a transcript is read from its end.
  transcript.scrollTop = transcript.scrollHeight;
  const redrawing = setInterval(drawDoing, 1000);
  signal.addEventListener("abort", () => { clearInterval(redrawing); content.hidden = true; }, { once: true });
  // What the brain was when this view was last left is not what it is: it is drawn as at a load, until the first read answers.
  work = IDLE;
  unread = null;
  drawBrain();
  // The brain is read for as long as the view is shown: its signal ends the read, the one in flight at hands included.
  reading(signal, "/brain", shown => {
    work = shown;
    unread = null;
    drawBrain();
  }, why => { unread = {text: `Could not read the brain: ${why}`, bad: true}; drawBrain(); });
}
