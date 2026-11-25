// The phone's one app (design-docs/phone-sessions/design.md, "The one app"): the document holds the call and the strip it
// is drawn in, and this module mounts the views under them. A tap on a view, or the back gesture, changes the view and
// never the document, so the call is never made again by either; only a load of the document, a reload, ends it.
//
// A view is `mount(root, route, signal)`: it draws into `root` for `route` (the path, and the query as URLSearchParams),
// and everything it starts, the reads of hands among them, ends when `signal` aborts, as the view is left.
import { mount as call } from "/views/call.js";
import { mount as sessions } from "/views/sessions.js";

// [LAW:one-type-per-behavior] the row of views, in the order it is drawn: an entry with a `view` is one of the app's, shown
// in the document at its path; one without is a page not yet moved into the app, opened as it always was, in a tab of its
// own. A page's move into the app is its entry gaining a `view`.
const ROW = [
  { label: "Call", path: "/", view: call },
  { label: "Conversation", path: "/conversation" },
  { label: "Sessions", path: "/sessions", view: sessions },
  { label: "Voices and options", path: "/gallery" },
];

const row = document.getElementById("views");
const root = document.getElementById("view");

const links = new Map(ROW.map(entry => {
  const link = Object.assign(document.createElement("a"), { textContent: entry.label, href: `${entry.path}${location.hash}` });
  if (entry.view === undefined) {
    link.target = "_blank";
    link.rel = "noopener";
  } else {
    link.addEventListener("click", event => {
      // A tap, not a request for the link in a tab of its own.
      if (event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      go(entry.path);
    });
  }
  return [entry, link];
}));
// The controller whose signal the view shown was mounted with; the first `show` ends this one, which no view holds.
let leaving = new AbortController();

function show() {
  leaving.abort();
  root.replaceChildren();
  const entry = ROW.find(each => each.view !== undefined && each.path === location.pathname);
  // [LAW:no-silent-failure] hands serves the shell at the paths of its views alone, so a path with no view is a mistake
  // of the two, said on its face.
  if (entry === undefined) throw new Error(`the app has no view at ${location.pathname}`);
  leaving = new AbortController();
  for (const [each, link] of links) link.ariaCurrent = each === entry ? "page" : null;
  entry.view(root, { path: location.pathname, query: new URLSearchParams(location.search) }, leaving.signal);
}

// A move to a view of the app: the path changes and the fragment, which holds the phone's key, stays as it is.
function go(path) {
  if (location.pathname === path) return;
  history.pushState(null, "", `${path}${location.hash}`);
  show();
}

row.append(...links.values());

addEventListener("popstate", show);
show();
