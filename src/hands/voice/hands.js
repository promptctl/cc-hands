// How every phone page asks hands (hands.voice.phonepage serves it at /hands.js), a module each page imports: one request,
// whose outcome is hands' answer, hands' refusal in its own words, or no answer and why; a read of what a page shows, for
// as long as the page is open; and hands' clock, as its reads last said it. Each page draws what it is answered and words
// what came of its own choices; nothing else is a page's own.

// The phone's key travels in the fragment, which the browser never sends with the page's own request. Whether it is the
// phone's key is hands' to say, and a page opened without one is refused by hands in hands' words [LAW:single-enforcer]:
// a page asks whether its address carries one only before it takes an effect of its own, such as the microphone.
const key = decodeURIComponent(location.hash.slice(1));
export const carriesKey = key !== "";

// How long any request is given to be answered once hands has it: a minute, past anything hands does before it answers.
// A read that has seen something waits at hands, for as long as hands said its reads wait, and is then answered at
// once: it is given that wait and the time an answer takes to cross back, so a phone that lost its way to hands says
// so soon after.
const ANSWER_MS = 60000, CROSSING_MS = 15000;

// [LAW:one-source-of-truth] every request a page makes of hands' routes crosses here, and what came of it is one of three
// outcomes. The browser's own loads, a script's import and the runtime's fetch of its WebAssembly among them, carry no
// key and fail in the browser's words. `readBody` reads an answer's body as the request wants it: its text, or its bytes.
async function crossed(method, path, body, within, readBody) {
  const init = { method, headers: { "Authorization": `Bearer ${key}` }, signal: AbortSignal.timeout(within) };
  if (body !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(body);
  }
  // Each step of the crossing fails its own way: never reached, or an answer cut off on its way back.
  // A bound that runs out fails the fetch with the TimeoutError AbortSignal.timeout names it by.
  let response;
  try {
    response = await fetch(path, init);
  } catch (error) {
    return { kind: "lost", why: error.name === "TimeoutError" ? `hands did not answer within ${Math.round(within / 1000)} s` : `hands is out of reach: ${error.message}` };
  }
  // A refusal is always words, whatever the request wanted.
  let answer;
  try {
    answer = response.ok ? await readBody(response) : await response.text();
  } catch (error) {
    return { kind: "lost", why: `hands' answer was cut off: ${error.message}` };
  }
  if (!response.ok) return { kind: "refused", status: response.status, text: answer || `hands refused with ${response.status}` };
  return { kind: "answered", body: answer };
}

// A request whose answer is JSON, or nothing.
export async function asked(method, path, body, within = ANSWER_MS) {
  const outcome = await crossed(method, path, body, within, response => response.text());
  if (outcome.kind !== "answered") return outcome;
  // A choice hands took with nothing to say back answers with no body at all.
  if (outcome.body === "") return { kind: "answered", body: null };
  try {
    return { kind: "answered", body: JSON.parse(outcome.body) };
  } catch (error) {
    return { kind: "lost", why: `hands' answer is not one this page reads: ${error.message}` };
  }
}

// A request whose answer is a file: its bytes, as a Uint8Array.
export const askedForBytes = (path, within = ANSWER_MS) => crossed("GET", path, undefined, within, async response => new Uint8Array(await response.arrayBuffer()));

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
// never by a phone's that runs fast or slow. Every read writes it and only a read does, each a reading of the one Mac's
// clock, so the newest is the truest; it is 0 only until hands first answers, before which nothing hands times is drawn.
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
        : asked("GET", `${path}?seen=${encodeURIComponent(last.seen)}`, undefined, last.waits * 1000 + CROSSING_MS));
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
