// A page's own script, run in node as the phone's browser runs it, against what hands answered: the test that runs it
// (tests/pagescript.py) hands it, on stdin, the page, the answers hands gives the page's requests in the order it asks
// them, and the steps a person takes on the page; it prints what the page shows once it has settled after each step.
//
// The page sees a DOM of plain objects, fetch answered from the answers, and a virtual clock: a timer runs out once the
// page has nothing else to do, the clock moved on to it, so a page that waits 90 s waits none.
import { readFileSync } from "node:fs";
import vm from "node:vm";

const { page, answers, steps } = JSON.parse(readFileSync(0, "utf8"));
const html = readFileSync(page, "utf8");
const [, script] = html.match(/<script>([\s\S]*)<\/script>/);

const elements = [];
const roots = new Map();

class Element {
  constructor(tag, id = "") {
    Object.assign(this, { tagName: tag.toUpperCase(), id, textContent: "", className: "", disabled: false, value: "", selected: false });
    this.children = [];
    this.parent = null;
    this.listeners = {};
    this.attributes = {};
    elements.push(this);
  }
  append(...children) {
    for (const child of children) {
      if (child instanceof Element) child.parent = this;
      this.children.push(child);
    }
  }
  replaceChildren(...children) {
    for (const child of this.children) if (child instanceof Element) child.parent = null;
    this.children = [];
    this.append(...children);
    // A select's value is the option marked selected, as a browser's is.
    if (this.tagName === "SELECT") this.value = (children.find(option => option.selected) ?? children[0])?.value ?? "";
  }
  addEventListener(type, listener) { (this.listeners[type] ??= []).push(listener); }
  setAttribute(name, value) { this.attributes[name] = value; }
  get attached() { return roots.has(this.id) && roots.get(this.id) === this || (this.parent?.attached ?? false); }
  get text() { return this.children.length ? this.children.map(child => child instanceof Element ? child.text : child).join(" ") : this.textContent; }
}

// Every element the markup names by id is a root of the page; its unnamed controls are roots too.
for (const [, tag, id] of html.matchAll(/<(\w+)[^>]*\bid="([^"]+)"/g)) roots.set(id, new Element(tag, id));
[...html.matchAll(/<(button|select)(?![^>]*\bid=)[^>]*>/g)].forEach(([, tag], at) => roots.set(`unnamed-${at}`, new Element(tag, `unnamed-${at}`)));

const document = {
  getElementById: id => roots.get(id) ?? null,
  createElement: tag => new Element(tag),
  querySelectorAll: selector => {
    const tags = selector.split(",").map(tag => tag.trim().toUpperCase());
    return elements.filter(element => tags.includes(element.tagName) && element.attached);
  },
};

// The virtual clock, and the timers waiting on it.
let now = 0;
const timers = [];
const setTimeout = (run, ms = 0) => { timers.push({ at: now + ms, run }); timers.sort((a, b) => a.at - b.at); };

// A request the test gave no answer for ends the run: the page would take it for hands failing and read again forever.
function unanswered(why) {
  process.stderr.write(`${why}\n`);
  process.exit(2);
}

const asked = [];
async function fetch(path, init = {}) {
  const method = init.method ?? "GET";
  asked.push(`${method} ${path}`);
  const answer = answers[0];
  if (!answer) unanswered(`the page asked ${method} ${path}, past every answer the test gave`);
  if (answer.method !== method || answer.path !== path) unanswered(`the page asked ${method} ${path}, where the test answers ${answer.method} ${answer.path}`);
  if (!(answer.times > 1)) answers.shift(); else answer.times -= 1;
  // What a browser throws for an address nobody answers on.
  if (answer.unreachable) throw new TypeError("Failed to fetch");
  return { ok: answer.status < 400, status: answer.status, text: async () => answer.body };
}

const context = vm.createContext({
  document, fetch, setTimeout, console, JSON, Promise, Error, TypeError, Date, Set, encodeURIComponent, decodeURIComponent,
  performance: { now: () => now },
  location: { hash: `#${encodeURIComponent("the-phones-key")}`, search: "" },
  URLSearchParams,
  // A request's own bound is the page's, never one this run waits out.
  AbortSignal: { timeout: () => new AbortController().signal },
});

// Run the page until it has nothing left to do but wait on a timer it is not owed: a timer is run once the page is idle.
async function settled() {
  for (;;) {
    await new Promise(done => setImmediate(done));
    if (!timers.length) return;
    const timer = timers.shift();
    now = timer.at;
    timer.run();
  }
}

function shown() {
  const controls = document.querySelectorAll("button, select");
  return {
    status: roots.get("status").textContent,
    failed: roots.get("status").className === "error",
    running: roots.get("running").textContent,
    controls: controls.length,
    disabled: controls.filter(control => control.disabled).length,
    at_ms: now,
    asked: asked.splice(0),
  };
}

const shows = [];
try {
  vm.runInContext(script, context);
  await settled();
  shows.push(shown());
  for (const step of steps) {
    const target = roots.get(step.submit);
    for (const [id, value] of Object.entries(step.set ?? {})) roots.get(id).value = value;
    for (const listener of target.listeners.submit ?? []) listener({ preventDefault() {} });
    await settled();
    shows.push(shown());
  }
} catch (error) {
  process.stderr.write(`${error.stack}\n`);
  process.exit(1);
}
process.stdout.write(JSON.stringify({ shows, unasked: answers }));
