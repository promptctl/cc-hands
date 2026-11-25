// The Call view: what the phone shows under the call's hero, the transcript with its toggle and the brain's panel, and the scroll
// of the document that melts the hero's round button into a bar. The call itself is call.js's: it writes the transcript whether
// this view is shown or not, so the transcript and its toggle are the document's from the start and the view shows them, hiding
// them as it is left. The brain's panel is the view's own: drawn into `root` from its template as the view mounts, read while the
// view is shown, and gone with it.
import {PAGE_OPEN, asked, handsNow, reading, unanswered} from "/hands.js";
import {reducedMotion, shape} from "/hero.js";

const content = document.getElementById("call-view");
const toggle = document.getElementById("shown");
const transcript = document.getElementById("transcript");
const panel = document.getElementById("brain-panel");
// How far the document scrolls while the button melts from the circle into the bar: a starting value, to be tuned on the phone.
const MELT = 160;
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

// The button's shape is the scroll position's alone: how far the document has scrolled, as a part of MELT (hero.js draws it).
const drawMelt = () => shape(scrollY / MELT);

const IDLE = {doing: "idle", calls: []};

// What the last Stop said, of that tap alone, until SAID_MS have passed. It is a deed's word, said whichever view is shown
// when it comes, so it outlives the view that tapped: a view mounted within SAID_MS shows it.
let stopSaid = null;

// A call's input as a glance reads it: its values, each a line, as the call gave them.
function inputOf(call) {
  return "unreadable" in call ? `input could not be read: ${call.unreadable}`
    : Object.values(call.input).map((value) => typeof value === "string" ? value : JSON.stringify(value)).join("\n");
}

export function mount(root, _route, signal) {
  // Taken first, so a mount that fails partway still ends cleanly when the view is left. The shell gives the next view the bar.
  let frame = 0;
  let redrawing = 0;
  signal.addEventListener("abort", () => {
    clearInterval(redrawing);
    cancelAnimationFrame(frame);
    content.hidden = true;
  }, { once: true });
  content.hidden = false;
  // The circle is the first thing on the view, whatever scroll the browser or the last visit left.
  scrollTo(0, 0);
  drawMelt();
  // The scroll event comes as often as the scroll moves; the shape is drawn once a frame, as far as the scroll has gone by then.
  document.addEventListener("scroll", () => {
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(drawMelt);
  }, {passive: true, signal});
  // A change of the preference redraws the shape it gives, whether or not the scroll moves.
  reducedMotion.addEventListener("change", drawMelt, {signal});
  // Lines were written while the view was away, and a transcript is read from its end.
  transcript.scrollTop = transcript.scrollHeight;
  root.append(panel.content.cloneNode(true));
  const within = id => root.querySelector(`#${id}`);
  const doing = within("doing"), calls = within("calls"), stopButton = within("stop"), brainSaid = within("brain-said");

  // The brain's work as hands last said it, its calls timed by hands' clock; whether a Stop is still being answered; and why
  // the brain could not be read, until it is read again. They are this mount's, so a view shown again draws what a load
  // draws until its first read answers.
  let work = IDLE;
  let stopping = false;
  let unread = null;

  // The line under the brain: what the last Stop said while it is still said, else why the brain could not be read.
  function drawSaid() {
    if (stopSaid !== null && Date.now() >= stopSaid.until) stopSaid = null;
    const told = stopSaid ?? unread ?? {text: "", bad: false};
    brainSaid.textContent = told.text;
    brainSaid.classList.toggle("bad", told.bad);
  }

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
    drawSaid();
  }

  const sayOfStop = (text, bad) => { stopSaid = {text, bad, until: Date.now() + SAID_MS}; };

  // Stop ends the turn the page shows, and that turn alone: one the brain went on to after it is never stopped by it. It is a
  // deed, done whichever view is shown, so leaving the view does not take it back.
  stopButton.addEventListener("click", async () => {
    stopping = true;
    drawBrain();
    try {
      const outcome = await asked(PAGE_OPEN, "POST", "/brain/stop", {turn: work.turn}, STOP_MS);
      outcome.kind === "answered" ? sayOfStop(outcome.body.readback, false) : sayOfStop(`Could not stop it: ${unanswered(outcome)}`, true);
    } catch (error) {
      sayOfStop(`Could not stop it: ${error.message}`, true);
    } finally {
      stopping = false;
      drawBrain();
    }
  });

  redrawing = setInterval(() => { drawDoing(); drawSaid(); }, 1000);
  drawBrain();
  // The brain is read for as long as the view is shown: its signal ends the read, the one in flight at hands included.
  reading(signal, "/brain", shown => {
    work = shown;
    unread = null;
    drawBrain();
  }, why => { unread = {text: `Could not read the brain: ${why}`, bad: true}; drawBrain(); });
}
