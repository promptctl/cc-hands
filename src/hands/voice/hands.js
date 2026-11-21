"use strict";
// How every phone page asks hands (hands.voice.phonepage serves it at /hands.js): one request, whose outcome is hands'
// answer, hands' refusal in its own words, or no answer and why; and a read of what a page shows, for as long as the page
// is open. Each page draws what it is answered and words what came of its own choices; nothing else is a page's own.

// The phone's key travels in the fragment, which the browser never sends with the page's own request.
const key = decodeURIComponent(location.hash.slice(1));

// How long any request is given to be answered once hands has it: a minute, past anything hands does before it answers. A
// read waits at hands first, for as long as hands said its reads wait.
const ANSWER_MS = 60000;

// [LAW:one-source-of-truth] every request a page makes is this one, and what came of it is one of three outcomes.
async function asked(method, path, body, within = ANSWER_MS) {
  const init = { method, headers: { "Authorization": `Bearer ${key}` }, signal: AbortSignal.timeout(within) };
  if (body !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(body);
  }
  // Each step of the crossing fails its own way: never reached, an answer cut off on its way back, or one that is not JSON.
  let response;
  try {
    response = await fetch(path, init);
  } catch (error) {
    return { kind: "lost", why: error.name === "TimeoutError" ? `hands did not answer within ${Math.round(within / 1000)} s` : "hands is out of reach" };
  }
  let text;
  try {
    text = await response.text();
  } catch (error) {
    return { kind: "lost", why: `hands' answer was cut off: ${error.message}` };
  }
  if (!response.ok) return { kind: "refused", status: response.status, text: text || `hands refused with ${response.status}` };
  // A choice hands took with nothing to say back answers with no body at all.
  if (text === "") return { kind: "answered", body: null };
  try {
    return { kind: "answered", body: JSON.parse(text) };
  } catch (error) {
    return { kind: "lost", why: `hands' answer is not one this page reads: ${error.message}` };
  }
}

// What a request hands did not answer says: its refusal, or why no answer came.
const unanswered = outcome => outcome.kind === "refused" ? outcome.text : outcome.why;

// How long the page waits to read hands again after a read that failed.
const AGAIN_MS = 2000;

// Read `path` for as long as the page is open: each read waits at hands until what it shows changes from what was seen
// (hands.voice.longpoll). `draw` is given what hands shows each time it changes, and `unread` why the page cannot show it,
// until it is drawn again. A read refused for the page's key ends the reading, as every read after it would be refused the
// same way; any other failure reads again after a pause, saying nothing was seen, so hands answers it at once and the page
// is drawn as soon as hands is back, whether a new run answers or the one that was there all along.
// [LAW:no-silent-failure] a read the page cannot draw ends the reading too, and says why: the same answer read again would
// fail the same way.
function reading(path, draw, unread) {
  return readOn(path, draw, unread).catch(error => unread(`This page stopped reading hands: ${error.message}. Reload it to read again.`));
}

// A page that has seen nothing is answered at once, so its read waits at hands not at all.
const NOTHING_SEEN = { seen: null, waits: 0 };

async function readOn(path, draw, unread) {
  let last = NOTHING_SEEN;
  for (;;) {
    const asking = last.seen === null ? path : `${path}${path.includes("?") ? "&" : "?"}seen=${encodeURIComponent(last.seen)}`;
    // [LAW:one-source-of-truth] the read's bound is the wait hands says its reads have, never a copy of it kept here.
    const outcome = await asked("GET", asking, undefined, last.waits * 1000 + ANSWER_MS);
    if (outcome.kind === "answered") {
      const answer = outcome.body;
      // An answer that does not say what it shows would be asked for again at once, and again, never waiting at hands.
      if (answer === null || answer.seen === undefined || typeof answer.waits !== "number") throw new Error("hands' answer is not a read");
      // A read that answers at the end of its wait with nothing changed has nothing new to draw.
      if (answer.seen !== last.seen) draw(answer.shown);
      last = answer;
    } else if (outcome.kind === "refused" && outcome.status === 401) {
      unread(outcome.text);
      return;
    } else {
      unread(`${unanswered(outcome)}; reading it again…`);
      last = NOTHING_SEEN;
      await new Promise(done => setTimeout(done, AGAIN_MS));
    }
  }
}
