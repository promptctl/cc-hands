// How every phone page asks hands (hands.voice.phonepage serves it at /hands.js), a module each page imports: one request,
// whose outcome is hands' answer, hands' refusal in its own words, or no answer and why; a read of what a page shows, for
// as long as the page is open; and hands' clock, as its reads last said it. Each page draws what it is answered and words
// what came of its own choices; nothing else is a page's own.

// The phone's key travels in the fragment, which the browser never sends with the page's own request. A page opened
// without it is refused by hands, in hands' words [LAW:single-enforcer].
const key = decodeURIComponent(location.hash.slice(1));

// How long any request is given to be answered once hands has it: a minute, past anything hands does before it answers. A
// read waits at hands first, for as long as hands said its reads wait.
const ANSWER_MS = 60000;

// [LAW:one-source-of-truth] every request a page makes is this one, and what came of it is one of three outcomes.
export async function asked(method, path, body, within = ANSWER_MS) {
  const init = { method, headers: { "Authorization": `Bearer ${key}` }, signal: AbortSignal.timeout(within) };
  if (body !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(body);
  }
  // Each step of the crossing fails its own way: never reached, an answer cut off on its way back, or one that is not JSON.
  // A bound that runs out fails the fetch with the TimeoutError AbortSignal.timeout names it by.
  let response;
  try {
    response = await fetch(path, init);
  } catch (error) {
    return { kind: "lost", why: error.name === "TimeoutError" ? `hands did not answer within ${Math.round(within / 1000)} s` : `hands is out of reach: ${error.message}` };
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
export const unanswered = outcome => outcome.kind === "refused" ? outcome.text : outcome.why;

// [LAW:parse-dont-validate] a read's answer (hands.voice.longpoll.Read) as the page reads it, or thrown: one that does not
// say what it shows would be asked for again at once, and again, never waiting at hands.
function readOf(body) {
  const { seen, waits, now, shown } = body ?? {};
  const read = (typeof seen === "string" || typeof seen === "number") && typeof waits === "number" && typeof now === "number" && typeof shown === "object" && shown !== null;
  if (!read) throw new Error("hands' answer is not a read");
  return { seen, waits, now, shown };
}

// How far hands' clock stood ahead of this page's as its last read answered, so what hands times is timed by its clock,
// never by a phone's that runs fast or slow.
let ahead = 0;

// The time on hands' clock now, in seconds, as hands' reads last said it.
export const handsNow = () => Date.now() / 1000 + ahead;

// How long the page waits to read hands again after a read that failed.
const AGAIN_MS = 2000;

// Read `path` for as long as the page is open: each read says what it last saw and waits at hands until what it shows
// changes from it (hands.voice.longpoll), bounded by the wait hands said; a read that has seen nothing is answered at once.
// `draw` is given what hands shows each time it changes, and `unread` why the page cannot show it, until it is drawn again.
// A read hands refuses as the page's own mistake (a 4xx: its key, its path) ends the reading, as every read after it would
// be refused the same way; one hands could not answer, or that never reached it, reads again after a pause, saying nothing
// was seen, so hands answers it at once and the page is drawn as soon as hands is back, whether a new run answers or the
// one that was there all along.
export async function reading(path, draw, unread) {
  try {
    let last = null;
    for (;;) {
      const outcome = await (last === null ? asked("GET", path)
        : asked("GET", `${path}?seen=${encodeURIComponent(last.seen)}`, undefined, last.waits * 1000 + ANSWER_MS));
      if (outcome.kind === "answered") {
        const read = readOf(outcome.body);
        ahead = read.now - Date.now() / 1000;
        // A read that answers at the end of its wait with nothing changed has nothing new to draw.
        if (read.seen !== last?.seen) draw(read.shown);
        last = read;
      } else if (outcome.kind === "refused" && outcome.status < 500) {
        unread(outcome.text);
        return;
      } else {
        unread(`${unanswered(outcome)}; reading it again…`);
        last = null;
        await new Promise(done => setTimeout(done, AGAIN_MS));
      }
    }
  } catch (error) {
    // [LAW:no-silent-failure] an answer the page cannot read, or cannot draw, ends the reading and says why: the same
    // answer read again would fail the same way. The console keeps where it failed.
    console.error(error);
    unread(`This page stopped reading hands: ${error.message}. Reload it to read again.`);
  }
}
