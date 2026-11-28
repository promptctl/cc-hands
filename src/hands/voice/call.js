// The call: the shell's own (app.html), run once as the document loads and never mounted or unmounted, so a move between
// the app's views is no move of the call's. It takes the elements of the shell's strip by id, and the transcript the Call view
// (views/call.js) shows: the call writes it whether that view is shown or not.
import {PAGE_OPEN, asked, carriesKey, wording} from "/hands.js";

// This page's own name, made as it opened: a call it makes again after a drop is one hands knows for this page's.
const PAGE = crypto.randomUUID();
// hands sends its voice every 20 ms, silence and all, for as long as the call is up: a call it has sent nothing over
// for this long is gone, as hands takes a page gone that sends nothing for its QUIET_SECS.
const QUIET_MS = 3000;
// The waits between calls made again, doubling from the first to the last, which repeats while hands stays out of reach.
const FIRST_RETRY_MS = 1000, LAST_RETRY_MS = 8000;
// hands answers an offer within a second or two: one unanswered for this long went to an address the phone can no longer
// reach, as when it moves off the LAN, and is a call dropped like any other.
const OFFER_MS = 10000;
const talk = document.getElementById("talk");
const state = document.getElementById("state");
const wakeButton = document.getElementById("wake");
const wakeNote = document.getElementById("wake-note");
const count = document.getElementById("count");
const out = document.getElementById("out");
const transcript = document.getElementById("transcript");

// Every render quantum of the microphone goes to the page as it is captured; whatever is sent in comes back out after
// every block captured before it, which is how a release is sent only once the words said before it have been.
const TAP = `
registerProcessor("tap", class extends AudioWorkletProcessor {
  constructor() { super(); this.port.onmessage = ({data}) => this.port.postMessage(data); }
  process(inputs) { const ch = inputs[0][0]; if (ch) this.port.postMessage(ch.slice(0)); return true; }
});`;

// The microphone, opened at the first Connect pressed and kept across every call made again after it, which no press
// starts: a browser opens the microphone for a press, not for a page calling on its own.
let media = null;  // {stream, audio, tap}
let call = null;  // {pc, channel, ended, presses, tell, turn, listening (the detector's phase), listen, retire, unlisten, press, release, close}
let held = false;
// Whether the page means to be on a call: from its first call's opening after Connect is pressed until hands gives the
// phone to another page, the page cannot call at all, or it goes. While it does, a call that drops is made again; a
// press whose call never opens is said, for the user there to press again.
let wanted = false;
let retry = null, tries = 0;
// Each call's own number for what it sends through the tap to come back, so what a dropped call sent is never taken
// for the next call's.
let tags = 0;

// Whether the line is the one that says how the call stands between turns: it is said again as that changes, and no other is.
let idle = false;

function say(text, bad = false) { state.textContent = text; state.classList.toggle("bad", bad); idle = false; }

// Whether the wake word may open a turn: the user's choice, kept in the browser, and on until they turn it off. A browser that
// will not keep it is said beside the button, so a choice that is gone on the next load is not taken for one that was kept.
let wakeOn = true;
let keptWhy = null;
try { wakeOn = localStorage.getItem("wake word") !== "off"; } catch (error) { keptWhy = wording(error); }

// What the page says of the clip it shows, by the last milestone of it. Three milestones are the page's own: `held`, the button
// down, `woken`, the detector hearing the word, and `sent`, the button let go with hands yet to say it took the clip. A
// milestone that `counts` is waited on: the time since the button was let go counts beside its line. A clip a wake opened
// carries the phrase that woke it, which its lines say, so they are said as they were whatever the detector has become since.
const nothing = (clip) => clip.wake ? `Heard nothing after "${clip.wake}".` : "Heard nothing from that press.";
const listeningLine = {text: () => "Listening…"};
const OWN = {
  "held": listeningLine,
  "woken": listeningLine,
  "sent": {text: () => "Sent. Waiting for hands…", counts: true},
  // Not told as milestones: each entered on hands' `unopened` for the press, which opened no clip, so nothing is coming to wait
  // for. By its reason: the turn was over before hands heard any of it, hands was at the desk, which opens no clip for a wake
  // and only the button brings it, or hands refused it in its own words, which the clip carries.
  "unopened": {text: nothing, bad: true},
  "desk": {text: () => "hands is at the desk. Hold the button to bring it here.", bad: true},
  "refused": {text: (clip) => `hands opened no turn for that: ${clip.why}.`, bad: true},
};
// What an end of the clip may come after: the button down, a wake heard, or the button let go.
const OPEN = ["held", "woken", "sent"];
// The milestones hands tells (hands.voice.milestone), each naming the milestones it may come after. The release names the
// clip it took, so the page takes only its own press's: let go, or still held as a hand at the desk took hands there and
// sent it, which the page says, since hands hears the desk from then on. The milestones after it are of the turn and carry no clip, so what places them is where they fall: one that
// cannot come after the shown clip's last is of a clip the page has let go of, and until hands says it took this clip,
// nothing else told is of it.
const TOLD = {
  "released": {after: OPEN, text: () => held ? "Sent: hands went to the desk, and hears you there now." : "Received. Transcribing…", counts: true},
  // The clip was thrown away, so nothing of it was sent: it ends as a release ends it, and no reply is coming.
  "discarded": {after: OPEN, text: nothing, bad: true},
  "transcript": {after: ["released"], text: () => "Heard. Waiting for a reply…", counts: true},
  "no words": {after: ["released"], text: (clip) => `No words heard, after ${secs(clip.at - clip.released)}.`, bad: true},
  "failed": {after: ["released", "transcript", "first LLM token"], text: (clip) => `hands failed, after ${secs(clip.at - clip.released)}.`, bad: true},
  "first LLM token": {after: ["released", "transcript"], text: () => "Replying…", counts: true},
  // The first sound after the button was let go, whatever hands said with it.
  "first audio": {after: ["released", "transcript", "first LLM token"], text: (clip) => `hands spoke after ${secs(clip.at - clip.released)}${clip.heard === null ? "" : `, your words heard at ${secs(clip.heard - clip.released)}`}.`},
};
// The clip shown: the call's number for its press, hands' number for it, its last milestone and when that came, when it was released, and when its words
// were heard.
let clip = null;
// The one timer the count runs on, while the clip shown is waited on.
let ticking = null;

function secs(ms) { return `${(ms / 1000).toFixed(1)} s`; }

function tick() { count.textContent = secs(performance.now() - clip.released); }

// Shows `next` as the clip: its line said once, and its count running while it is waited on.
function show(next) {
  letGo();
  clip = next;
  const line = OWN[clip.milestone] ?? TOLD[clip.milestone];
  say(line.text(clip), line.bad);
  if (line.counts) { ticking = setInterval(tick, 100); tick(); }
}

// Every time is the page's own, from the button let go to the milestone arriving: the wait as it is felt at the phone.
function milestone(name, number) {
  // [LAW:no-silent-failure] the page and hands disagree about what hands tells: said on its face, not guessed at.
  if (!Object.hasOwn(TOLD, name)) { letGo(); say(`hands said "${name}", which this page does not know.`, true); return; }
  if (clip === null || !TOLD[name].after.includes(clip.milestone)) return;
  // A milestone that names its clip is of that clip alone.
  if (number !== undefined && number !== clip.number) return;
  const at = performance.now();
  // A clip hands took while its button was down was released then, not when the button comes up.
  show({...clip, milestone: name, at, released: clip.released ?? at, heard: name === "transcript" ? at : clip.heard});
}

// The clip a press of the page's opened, by the press's number: hands bound the two as the press opened it, so however late
// either arrives, it is the shown press's own or none of it.
function opened(message) {
  if (clip?.press !== message.press) return;
  clip = {...clip, number: message.clip};
}

// hands' word that a press of the page's opened no clip, by the press's number: the shown press's own or none of it.
const UNOPENED = {"elsewhere": "desk", "unheard": "unopened", "deaf": "refused", "busy": "refused", "unreadable": "refused"};
function unopened(message) {
  if (clip?.press !== message.press) return;
  // [LAW:no-silent-failure] a reason this page does not know is said on its face, not worded as another.
  if (!Object.hasOwn(UNOPENED, message.reason)) { letGo(); say(`hands said it opened nothing for "${message.reason}", which this page does not know.`, true); return; }
  show({...clip, milestone: UNOPENED[message.reason], why: message.why, at: performance.now()});
}

// Lets go of the clip shown: no milestone told from here is of it, and its count stops.
function letGo() { clearInterval(ticking); ticking = null; clip = null; count.textContent = ""; }

// The sentence of hands' playing: its words, where each ends in its text, when it started, and the rate hands speaks
// at, in characters a millisecond; null between sentences.
let saying = null;  // {words, ends, started, rate, frame}

// Writes a line at the transcript's end, keeping its end in view if it was.
function write(line) {
  const atEnd = transcript.scrollHeight - transcript.scrollTop - transcript.clientHeight < 24;
  transcript.append(line);
  if (atEnd) transcript.scrollTop = transcript.scrollHeight;
}

// [LAW:one-source-of-truth] pocket-tts gives no word timings, so a word is reached when as many characters of its
// sentence have been said, at the rate hands measured, as come before it; the last waits there until hands says it is
// done.
function sweep() {
  const reached = (performance.now() - saying.started) * saying.rate;
  const now = Math.min(saying.ends.filter((end) => end <= reached).length, saying.words.length - 1);
  saying.words.forEach((word, i) => { word.className = i < now ? "" : i === now ? "now" : "ahead"; });
  saying.frame = requestAnimationFrame(sweep);
}

// Ends the sentence playing, each word as `left` would have it: said, or cut off.
function stop(left) {
  if (saying === null) return;
  cancelAnimationFrame(saying.frame);
  saying.words.forEach((word) => { word.className = left(word); });
  saying = null;
}

// What of a sentence stopped short was not heard: the words not yet reached.
const bargedIn = (word) => word.className === "ahead" ? "unheard" : "";

// What each line of the transcript does to the page (hands.voice.transcript).
const LINES = {
  "heard": ({text}) => {
    const line = document.createElement("p");
    line.className = "user";
    line.textContent = text;
    write(line);
  },
  "saying": ({text, chars_per_sec}) => {
    stop(() => "");
    const line = document.createElement("p");
    let end = 0;
    const words = [], ends = [];
    for (const part of text.split(/(\s+)/)) {
      if (part === "") continue;
      if (/^\s+$/.test(part)) { line.append(part); end += part.length; continue; }
      const word = document.createElement("span");
      word.textContent = part;
      line.append(word);
      words.push(word); end += part.length; ends.push(end);
    }
    write(line);
    saying = {words, ends, started: performance.now(), rate: chars_per_sec / 1000, frame: null};
    sweep();
  },
  "spoken": () => stop(() => ""),
  "bargein": () => stop(bargedIn),
};

// [LAW:single-enforcer] the wake's turn is open until hands says it is over, and the page never guesses: its press `unopened`, or
// its clip `released` or `discarded`. Nothing else closes it, and a turn the button took over has no word of the wake's to wait on.
function turnTold(made, message) {
  const turn = made.turn;
  switch (turn.kind) {
    // The wake is heard and not yet answered: with its clip, or with none.
    case "waking":
      if (message.kind === "opened" && message.press === turn.press) made.turn = {kind: "wake", press: turn.press, clip: message.clip};
      else if (message.kind === "unopened" && message.press === turn.press) made.close();
      return;
    case "wake":
      if (message.kind === "milestone" && (message.milestone === "released" || message.milestone === "discarded") && message.clip === turn.clip) made.close();
      return;
  }
}

// Everything hands tells the page over `made`'s channel, by its kind: a clip opened, a milestone of the clip shown, or a line
// of the transcript.
function told(made, message) {
  turnTold(made, message);
  if (message.kind === "milestone") { milestone(message.milestone, message.clip); return; }
  if (message.kind === "opened") { opened(message); return; }
  if (message.kind === "unopened") { unopened(message); return; }
  // [LAW:no-silent-failure] the page and hands disagree about what hands tells: said on its face, not guessed at.
  if (!Object.hasOwn(LINES, message.kind)) { say(`hands told a "${message.kind}", which this page does not know.`, true); return; }
  LINES[message.kind](message);
}

function pcm(blocks) {
  const length = blocks.reduce((n, b) => n + b.length, 0);
  const samples = new Int16Array(length);
  let i = 0;
  for (const block of blocks) for (const s of block) samples[i++] = Math.max(-32768, Math.min(32767, Math.round(s * 32767)));
  return samples.buffer;
}

// `blocks` as messages of audio of `per` samples each, the last what is left.
function cut(blocks, per) {
  const all = new Float32Array(blocks.reduce((n, b) => n + b.length, 0));
  let at = 0;
  for (const block of blocks) { all.set(block, at); at += block.length; }
  const messages = [];
  for (let from = 0; from < all.length; from += per) messages.push(pcm([all.subarray(from, from + per)]));
  return messages;
}

// The browser the page is in, as it says it: what tells an iPhone's Safari from a desktop's Chrome in hands' events.
const BROWSER = navigator.userAgent;

// The line that says how the call stands between turns: said again as that changes, and no other line is.
function sayIdle() {
  const listening = call?.listening;
  say(listening?.phase === "ready" ? `Connected. Say "${listening.phrase}", or hold the button.` : "Connected. Hold the button and speak.");
  idle = true;
}

// The wake word's own state, which no turn's line says: why it is not listening, or that hands has not answered it, why the
// screen may go to sleep, and that the choice is not kept.
let lockWhy = null;
function drawNote() {
  const listening = call?.listening;
  const notes = [];
  if (listening?.phase === "failed") notes.push(`The wake word is not listening: ${listening.why}.`);
  else if (listening?.phase === "loading" && listening.waiting !== null) notes.push(`The wake word is waiting on hands: ${listening.waiting}.`);
  if (lockWhy !== null) notes.push(`The screen may go to sleep, which stops the wake word: ${lockWhy}.`);
  if (keptWhy !== null) notes.push(`This browser will not keep the wake word's choice: ${keptWhy}.`);
  wakeNote.textContent = notes.join(" ");
  wakeNote.classList.toggle("bad", listening?.phase === "failed");
}

function refresh() {
  if (idle) sayIdle();
  drawNote();
}

// The screen stays on while the wake word is on and the call is up, which a browser ends when the page leaves the front: the
// lock is asked for again as it comes back, and what came of each ask, and a lock the browser takes back while the page stays in
// front, is told to hands. One change at a time, so a lock asked for as it is no longer wanted is let go.
let lock = null;
let locking = Promise.resolve();
function syncLock() {
  locking = locking.then(async () => {
    const mine = call;
    const wanted = wakeOn && mine?.channel.readyState === "open" && !document.hidden;
    try {
      if (wanted && lock === null) {
        if (navigator.wakeLock === undefined) throw new Error("this browser has no screen wake lock");
        const got = await navigator.wakeLock.request("screen");
        // A lock this page let go is not `lock` by then, so only one the browser took back is told, and is asked for again at
        // the next change of the page's front, as one the page left is.
        got.addEventListener("release", () => {
          if (lock !== got) return;
          lock = null;
          // A page that left the front loses its lock as the browser's rule, which is no taking back: it is asked for again as
          // the page returns.
          if (document.hidden) return;
          lockWhy = "the browser let go of it";
          mine.tell({kind: "screen", state: "unkept", why: lockWhy});
          drawNote();
        });
        lock = got;
        lockWhy = null;
        mine.tell({kind: "screen", state: "kept"});
      } else if (!wanted) {
        lockWhy = null;
        if (lock !== null) {
          const held = lock;
          lock = null;
          await held.release();
        }
      }
    } catch (error) {
      // A lock that was asked for and not had is the wake word's trouble, told on its call; one that would not let go is
      // nothing the wake word is waiting on, and the browser lets go of it as the page leaves.
      if (wanted) {
        lockWhy = wording(error);
        mine.tell({kind: "screen", state: "unkept", why: lockWhy});
      } else {
        console.error(error);
      }
    }
    drawNote();
  });
}

function drawWake() { wakeButton.setAttribute("aria-pressed", String(wakeOn)); }

// The user's choice, kept, and acted on at once by a call that is up: before Connect it changes nothing but what the next call does.
function setWake(on) {
  wakeOn = on;
  try { localStorage.setItem("wake word", on ? "on" : "off"); keptWhy = null; } catch (error) { keptWhy = wording(error); }
  drawWake();
  if (call?.channel.readyState === "open") { if (on) call.listen(); else call.unlisten(); }
  refresh();
  syncLock();
}

async function open() {
  const stream = await navigator.mediaDevices.getUserMedia({audio: {echoCancellation: true, noiseSuppression: true, autoGainControl: true}});
  const audio = new AudioContext();
  await audio.audioWorklet.addModule(URL.createObjectURL(new Blob([TAP], {type: "text/javascript"})));
  const tap = new AudioWorkletNode(audio, "tap");
  const mute = audio.createGain();
  mute.gain.value = 0;
  // Connected through to the output, silent, so the browser keeps pulling the microphone through the tap.
  audio.createMediaStreamSource(stream).connect(tap).connect(mute).connect(audio.destination);
  // A microphone taken away (unplugged, or by another app) sends nothing again: no call made over it would be heard.
  stream.getAudioTracks().forEach((track) => { track.onended = () => end("The microphone was taken away. Press Connect to open it again."); });
  return {stream, audio, tap};
}

async function connect(claim) {
  talk.disabled = true;
  if (!media) media = await open();
  const {audio, tap} = media;
  // A page brought back from the background may find its microphone asleep: a call over it would send nothing.
  await audio.resume();
  const pc = new RTCPeerConnection();
  pc.addTransceiver("audio", {direction: "recvonly"});
  pc.ontrack = (event) => { out.srcObject = new MediaStream([event.track]); out.play(); };
  const channel = pc.createDataChannel("talk", {ordered: true});
  channel.binaryType = "arraybuffer";
  // Twenty milliseconds of the microphone a message.
  const per = Math.round(audio.sampleRate / 50);
  let pending = [], count = 0, marked = null;
  // No sample leaves the phone outside a turn. Inside one, what is gathered goes as audio; outside one, only how many
  // samples were gathered goes, as a `quiet`, which hands hears as that much silence: a message every 20 ms either way is
  // what hands' clock and its watch on this call run on (hands.voice.phone). The turn's edges end the message being
  // gathered, in the order the button's messages go out, so no sample of one side rides in a message of the other.
  const tag = ++tags;
  tap.port.onmessage = ({data}) => {
    if (typeof data === "number") { if (data === tag) { flush(); marked(); } return; }
    // The detector hears the microphone for as long as it is on, turns included, so the word said as a turn ends is heard. It
    // is fed from ready on: the models may load for minutes on a cold first run, and audio fed meanwhile is held in the worker
    // for as long as that takes, while the ring a wake sends from exists only once the detector has said how long it is.
    const listening = made.listening;
    if (listening?.phase === "ready") {
      listening.started.feed(data);
      if (made.turn.kind === "waiting") listening.ring.push(data);
    }
    pending.push(data); count += data.length;
    if (count >= per) flush();
  };
  // [LAW:single-enforcer] everything the page sends goes out here: a channel no longer open is a call that is ending, which
  // its own close says, so what was to go over it has nowhere to go.
  function send(data) { if (channel.readyState === "open") channel.send(data); }
  function tell(message) { send(JSON.stringify(message)); }
  function flush() {
    if (count) send(made.turn.kind === "waiting" ? JSON.stringify({kind: "quiet", samples: count}) : pcm(pending));
    pending = []; count = 0;
  }
  // Each edge of a turn goes out after the one before it has: a press made while a release still waits on its mark is sent
  // after that release, so hands hears them in the order the button made them, and one mark is out at a time.
  let sent = Promise.resolve();
  // The edges made and not yet gone out. A press is numbered as the button goes down and a wake as it is sent, and hands
  // numbers both as it hears them, so a word heard with an edge ahead of it would be numbered after a press that is already
  // numbered: it is not sent, and is told to hands as heard.
  let edges = 0;
  // An edge of a turn waits for the mark to come back through the tap, so every block the tap had when the turn moved has
  // been gathered, and it is flushed as the side of the edge it was gathered on: a count before a press, audio before a
  // release or before hands' word that a wake's turn is over. Then `act` moves the turn.
  const edge = (act) => { edges += 1; sent = sent.then(() => new Promise((done) => {
    marked = () => { try { act(); } finally { edges -= 1; done(); } };
    tap.port.postMessage(tag);
  })); };
  // The turn closes: the ring is emptied, so it holds only what the microphone made after it, as the Mac's clip owner clears
  // what it kept as a clip ends.
  const waiting = () => {
    made.turn = {kind: "waiting"};
    const listening = made.listening;
    if (listening?.phase === "ready") listening.ring.empty();
  };
  // The word the detector heard, at `score`. If no turn is open the page sends the second kept before it, the wake, then the
  // microphone as it comes; else it tells hands it heard the word and sent none, so every hearing is in hands' record while the
  // page's own turn state alone decides what leaves the phone.
  function heardWord(score) {
    if (made.turn.kind !== "waiting" || edges > 0) { tell({kind: "ignored", score}); return; }
    const listening = made.listening;
    // What was gathered since the last message is in the ring, so it goes ahead of the wake and is not sent again.
    for (const kept of cut(listening.ring.empty(), per)) send(kept);
    pending = []; count = 0;
    tell({kind: "wake", score});
    // A wake is a press hands numbers, in the order it hears them.
    const press = ++made.presses;
    made.turn = {kind: "waking", press};
    show({press, number: null, milestone: "woken", at: null, released: null, heard: null, wake: listening.phrase});
  }
  // [LAW:types-are-the-program] The detector of this call is what is true of it and nothing else: `importing`, its script on its
  // way; `loading`, its worker up and its models not, `waiting` on why hands has not answered it; `ready`, listening, with the
  // ring it keeps the second before a word in and the phrase it listens for; `failed`, and why. Null is the wake word off. Every
  // phase stops by `stop()`, so a call that ends, a detector that fails and the button turned off end it one way, and a worker
  // is never left running that no phase holds. A worker stopped is heard from no more, so no message here is of a detector
  // that is no longer the call's; only the script's arrival, which has no worker to stop, asks.
  async function listen() {
    const importing = made.listening = {phase: "importing", stop() {}};
    let detector;
    try {
      detector = await import("/detector.js");
    } catch (error) {
      if (made.listening === importing) failed(`its script could not be loaded: ${wording(error)}`);
      return;
    }
    if (made.listening !== importing) return;
    const started = detector.startDetector({
      rate: audio.sampleRate,
      onReady: (word) => ready(started, word, detector.Ring),
      onFailed: failed,
      onHeard: heardWord,
      onWaiting: waitingOn,
    });
    made.listening = {phase: "loading", stop: started.stop, waiting: null};
  }
  function ready(started, word, Ring) {
    made.listening = {phase: "ready", stop: started.stop, started, ring: new Ring(Math.round(word.keptSeconds * audio.sampleRate)), phrase: word.phrase};
    tell({kind: "detector", state: "loaded", browser: BROWSER});
    tell({kind: "level", level: "wake word"});
    refresh();
  }
  // hands is told as a wait begins and as it ends, not each time it asks again: a null `why` is hands having answered, after
  // which the models load, which is no wait on hands.
  function waitingOn(why) {
    const loading = made.listening;
    if (why === null) {
      loading.waiting = null;
      tell({kind: "detector", state: "answered", browser: BROWSER});
    } else {
      if (loading.waiting === null) tell({kind: "detector", state: "waiting", browser: BROWSER, why});
      loading.waiting = why;
    }
    drawNote();
  }
  // [LAW:single-enforcer] the one way out of a detector's life while the call goes on: it is stopped, and a detector that was
  // listening tells hands the phone no longer is, for hands holds the level the page last told it for as long as the call lasts.
  // A call that ends is retired instead, for hands is at the desk again as that call goes.
  function leave(next) {
    const was = made.listening;
    was.stop();
    made.listening = next;
    if (was.phase === "ready") tell({kind: "level", level: "off"});
    refresh();
  }
  function failed(why) {
    tell({kind: "detector", state: "failed", browser: BROWSER, why});
    leave({phase: "failed", stop() {}, why});
  }
  // Each press is numbered in the call, as hands numbers the presses it hears over the ordered channel.
  const made = call = {pc, channel, ended: false, presses: 0, tell,
    // What the microphone is going to: `waiting` for a turn, `waking` and `wake` for the wake's, or the button's.
    turn: {kind: "waiting"},
    listening: null,
    listen,
    // The detector ended with the call, which says nothing: hands is at the desk again as the call goes.
    retire() { this.listening?.stop(); this.listening = null; },
    // The wake word turned off: no detector, while the call goes on.
    unlisten() { if (this.listening !== null) leave(null); },
    press() { edge(() => { send("press"); made.turn = {kind: "button"}; }); return ++this.presses; },
    release() { edge(() => { send("release"); waiting(); }); },
    // The wake's turn is over on hands' word, unless a press has taken it over in the meantime.
    close() { const turn = this.turn; edge(() => { if (made.turn === turn) waiting(); }); },
  };
  channel.onopen = () => {
    wanted = true; tries = 0;
    talk.disabled = false; talk.textContent = "Hold to talk"; sayIdle();
    watch(made);
    syncLock();
    if (wakeOn) made.listen();
  };
  channel.onmessage = ({data}) => told(made, JSON.parse(data));
  channel.onclose = () => dropped(made, "hands hung up.");
  pc.onconnectionstatechange = () => { if (pc.connectionState === "failed") dropped(made, "The connection to hands failed."); };

  // From here on, what fails is this call, settled against it: never another call made since, as this one ended.
  try {
    await pc.setLocalDescription(await pc.createOffer());
    // Every candidate is gathered before the offer goes: hands answers once, with all of its own.
    await new Promise((done) => {
      if (pc.iceGatheringState === "complete") return done();
      pc.addEventListener("icegatheringstatechange", () => pc.iceGatheringState === "complete" && done());
    });
    const outcome = await asked(PAGE_OPEN, "POST", "/offer", {sdp: pc.localDescription.sdp, type: "offer", rate: audio.sampleRate, page: PAGE, claim}, OFFER_MS);
    if (made.ended) return;
    // hands declines a call made again while another page has the phone: that page is the one in the user's hand. What may
    // pass (`transient`), as hands is restarted, is dialled again; a refusal of the page's key or its offer never passes.
    switch (outcome.kind) {
      case "answered": await pc.setRemoteDescription(outcome.body); break;
      case "refused":
        if (outcome.status === 409) end("Another page has hands' phone. Press Connect to take it here.");
        else if (outcome.transient) dropped(made, `hands could not answer: ${outcome.text}.`);
        else end(`Could not connect: ${outcome.text}`);
        break;
      case "lost":
        if (outcome.transient) dropped(made, `Could not reach hands: ${outcome.why}.`);
        else end(`Could not connect: ${outcome.why}`);
        break;
    }
  } catch (error) {
    dropped(made, `Could not reach hands: ${error.message}.`);
  }
}

// A call is gone once hands has sent nothing over it for QUIET_MS: a page woken from sleep finds its call still
// "connected", hands having hung it up while the phone could not hear it say so.
async function watch(made) {
  let received = null, heard = performance.now();
  while (!made.ended) {
    await new Promise((done) => setTimeout(done, 1000));
    if (made.ended) return;
    let now = 0, stats;
    // Closed while the wait ran: the call is over, and so is its watch.
    try { stats = await made.pc.getStats(); } catch { return; }
    stats.forEach((report) => { if (report.type === "inbound-rtp" && report.kind === "audio") now = report.packetsReceived; });
    if (now !== received) { received = now; heard = performance.now(); }
    else if (performance.now() - heard > QUIET_MS) dropped(made, "hands went quiet.");
  }
}

// What a call that ended leaves behind: the microphone, kept for the next, and nothing else.
function hangUp() {
  clearTimeout(retry); retry = null;
  if (call) { call.ended = true; call.retire(); call.channel.close(); call.pc.close(); call = null; }
  held = false;
  letGo();
  drawNote();
  syncLock();
  // The rest of what was playing never reaches this page.
  stop(bargedIn);
  talk.classList.remove("held");
}

// The call `made` dropped: made again, after a wait that doubles with every try, while the page means to be on one.
function dropped(made, why) {
  if (made !== call || made.ended) return;
  if (!wanted) { end(`Could not connect: ${why}`); return; }
  hangUp();
  const wait = Math.min(FIRST_RETRY_MS * 2 ** tries, LAST_RETRY_MS);
  tries += 1;
  talk.disabled = true; talk.textContent = "Reconnecting";
  say(`${why} Calling again in ${Math.round(wait / 1000)} s…`, true);
  retry = setTimeout(() => { retry = null; place("resume"); }, wait);
}

// The page is done calling until Connect is pressed again: the microphone let go, and why said.
function end(why) {
  wanted = false; tries = 0;
  hangUp();
  if (media) { media.stream.getTracks().forEach((t) => t.stop()); media.audio.close(); media = null; }
  talk.disabled = false; talk.textContent = "Connect";
  say(why, true);
}

function place(claim) {
  say(claim === "take" ? "Connecting…" : "Calling hands again…");
  // What fails here failed before there was a call to make again: the microphone could not be opened or woken.
  connect(claim).catch((error) => end(`Could not connect: ${error.message}`));
}

function down(event) {
  event.preventDefault();
  // A page whose address carries no key never takes the microphone for a call hands would refuse; whether a key it
  // carries is the phone's is hands' to say.
  if (!carriesKey) { say("Open this page from the address `hands phone` prints.", true); return; }
  if (!call && !retry) { place("take"); return; }
  if (!call || call.channel.readyState !== "open" || held) return;
  talk.setPointerCapture(event.pointerId);
  held = true;
  talk.classList.add("held");
  // A new clip takes the place of the one shown: what is shown from here is this clip's.
  show({press: call.press(), number: null, milestone: "held", at: null, released: null, heard: null});
}

function up() {
  if (!held) return;
  held = false;
  talk.classList.remove("held");
  if (clip?.milestone === "held") show({...clip, milestone: "sent", released: performance.now()});
  call.release();
}

talk.addEventListener("pointerdown", down);
talk.addEventListener("pointerup", up);
talk.addEventListener("pointercancel", up);
talk.addEventListener("contextmenu", (event) => event.preventDefault());
// A page put away mid-press lets go of the button; brought back, a call that dropped while it was away is made again at
// once rather than after the wait.
document.addEventListener("visibilitychange", () => {
  syncLock();
  if (document.hidden) { up(); return; }
  if (retry) { clearTimeout(retry); retry = null; place("resume"); }
});
wakeButton.addEventListener("click", () => setWake(!wakeOn));
drawWake();
drawNote();
// One closed hangs up.
addEventListener("pagehide", () => end("This page hung up."));
