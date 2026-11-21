// A page's own script, run in node as the phone's browser runs it, against hands as a test describes it: the test that
// runs it (tests/pagescript.py) gives, on stdin, the page, hands' answers to each route through time, and the steps a
// person takes; it prints what the page shows after each.
//
// The page sees a DOM of plain objects, fetch answered from what hands answers at the moment it asks, and a virtual
// clock: a timer runs out once the page has nothing else to do, the clock moved on to it, so a page that waits a minute
// waits none of it.
import { readFileSync } from "node:fs";
import vm from "node:vm";

const { page, hands, steps } = JSON.parse(readFileSync(0, "utf8"));
const html = readFileSync(page, "utf8");
const [, script] = html.match(/<script>([\s\S]*)<\/script>/);

// What a run of the page cannot go on from: the test is told why, and nothing is printed as shown.
function fails(why) {
  process.stderr.write(`${why}\n`);
  process.exit(2);
}

const elements = [];
const named = new Map();

class Element {
  constructor(tag, id = "") {
    Object.assign(this, { tagName: tag.toUpperCase(), id, textContent: "", className: "", disabled: false, value: "", selected: false });
    this.children = [];
    this.parent = null;
    this.listeners = {};
    elements.push(this);
  }
  append(...children) {
    for (const child of children) child.parent = this;
    this.children.push(...children);
  }
  replaceChildren(...children) {
    for (const child of this.children) child.parent = null;
    this.children = [];
    this.append(...children);
    // A select's value is the option marked selected, as a browser's is.
    if (this.tagName === "SELECT") this.value = (children.find(option => option.selected) ?? children[0])?.value ?? "";
  }
  addEventListener(type, listener) { (this.listeners[type] ??= []).push(listener); }
  setAttribute() {}
  // On the page: named in its markup, or inside an element that is.
  get shown() { return named.get(this.id) === this || (this.parent?.shown ?? false); }
  within(id) { return this.id === id || (this.parent?.within(id) ?? false); }
}

// Every element the markup names by id, and every control it writes, which a page reaches through querySelectorAll.
for (const [, tag, id] of html.matchAll(/<(\w+)[^>]*\bid="([^"]+)"/g)) named.set(id, new Element(tag, id));
for (const [at, [, tag]] of [...html.matchAll(/<(button|select)(?![^>]*\bid=)[^>]*>/g)].entries()) named.set(`markup-${tag}-${at}`, new Element(tag, `markup-${tag}-${at}`));

const document = {
  getElementById: id => named.get(id) ?? null,
  createElement: tag => new Element(tag),
  querySelectorAll: selector => {
    const tags = selector.split(",").map(tag => tag.trim());
    if (!tags.every(tag => /^[a-z]+$/.test(tag))) fails(`the page asked for ${selector}, and this DOM finds elements by tag alone`);
    return elements.filter(element => tags.includes(element.tagName.toLowerCase()) && element.shown);
  },
};

// The virtual clock, and the timers waiting on it.
let now = 0;
const timers = [];
const setTimeout = (run, ms = 0) => { timers.push({ at: now + ms, run }); timers.sort((a, b) => a.at - b.at); };

// hands at a moment: the last of its eras begun by then answers. An answer is a reply, or null where nobody answers.
const asked = [];
async function fetch(path, init = {}) {
  const route = `${init.method ?? "GET"} ${path}`;
  asked.push(route);
  const era = hands.filter(each => each.from_ms <= now).at(-1);
  if (!era || !(route in era.answers)) fails(`at ${now} ms the page asked ${route}, which hands as the test gave it does not answer`);
  const answer = era.answers[route];
  // What a browser throws for an address nobody answers on.
  if (answer === null) throw new TypeError("Failed to fetch");
  return { ok: answer.status < 400, status: answer.status, text: async () => answer.body };
}

const context = vm.createContext({
  document, fetch, setTimeout, console, JSON, Promise, Error, TypeError,
  location: { hash: `#${encodeURIComponent("the-phones-key")}` },
  // A request's own bound is the page's, never one this run waits out.
  AbortSignal: { timeout: () => new AbortController().signal },
});

// Run the page until `until_ms`, or until it waits on nothing: each timer runs once the page is idle.
async function runUntil(until) {
  for (;;) {
    await new Promise(done => setImmediate(done));
    if (!timers.length || timers[0].at > until) break;
    const timer = timers.shift();
    now = timer.at;
    timer.run();
  }
  now = Math.max(now, until);
  await new Promise(done => setImmediate(done));
}

function shown() {
  const controls = document.querySelectorAll("button, select");
  return {
    at_ms: now,
    reading: named.get("reading").textContent,
    status: named.get("status").textContent,
    errors: ["reading", "status"].filter(id => named.get(id).className === "error"),
    running: named.get("running").textContent,
    controls: controls.length,
    disabled: controls.filter(control => control.disabled).length,
    asked: asked.splice(0),
  };
}

function take(step) {
  for (const [id, value] of Object.entries(step.set)) named.get(id).value = value;
  if (step.kind === "submit") {
    for (const listener of named.get(step.form).listeners.submit ?? []) listener({ preventDefault() {} });
    return;
  }
  const control = elements.find(element => element.shown && element.tagName === "BUTTON" && element.within(step.within) && element.textContent === step.text);
  if (!control) fails(`no button ${step.text} in ${step.within}`);
  for (const listener of control.listeners.click ?? []) listener();
}

const shows = [];
try {
  vm.runInContext(script, context);
  for (const step of steps) {
    if (step.kind !== "wait") take(step);
    await runUntil(step.until_ms);
    shows.push(shown());
  }
} catch (error) {
  fails(error.stack);
}
process.stdout.write(JSON.stringify(shows));
