// The phone's one app (design-docs/phone-sessions/design.md, "The one app"): the document holds the call and the hero it
// is drawn in, and this module mounts the views under them. A tap on a view, or the back gesture, changes the view and
// never the document, so the call is never made again by either; only a load of the document, a reload, ends it.
//
// A view is `mount(root, route, signal)`: it draws into `root` for `route` (the path, and the query as URLSearchParams),
// and everything it starts, the reads of hands among them, ends when `signal` aborts, as the view is left.
import "/call.js";

// The Call view mounts at the top of its scroll, every time, so the browser does not put a reload or a back part-way down it.
history.scrollRestoration = "manual";

const row = document.getElementById("views");
const hero = document.getElementById("hero");
const root = document.getElementById("view");

// [LAW:one-source-of-truth] hands draws the row (hands.voice.phonepage), and this reads it: a link with a `data-view` is a
// view of the app, mounted from the module of that name; one without is a page opened in a tab of its own.
const links = [...row.querySelectorAll("a")];
let mounts;
try {
  mounts = new Map(await Promise.all(links.filter(link => link.dataset.view !== undefined)
    .map(async link => [link.pathname, (await import(`/views/${link.dataset.view}.js`)).mount])));
} catch (error) {
  // [LAW:no-silent-failure] the row stays hidden, so no tap unloads the call for a page that fails the same way, and the
  // app says on its face that it has no views.
  root.textContent = `The app could not load its views: ${error.message}`;
  throw error;
}

// A path of the app, carrying the phone's key, which is in the fragment.
const keyed = path => `${path}${location.hash}`;

// The controller whose signal the view shown was mounted with; null before the first.
let leaving = null;

function show() {
  const mount = mounts.get(location.pathname);
  // [LAW:no-silent-failure] hands serves the document at the paths of the app's views alone, so a path with no view is a
  // mistake, said on its face before the view shown is touched.
  if (mount === undefined) throw new Error(`the app has no view at ${location.pathname}`);
  leaving?.abort();
  leaving = new AbortController();
  root.replaceChildren();
  for (const link of links) link.ariaCurrent = link.pathname === location.pathname ? "page" : null;
  try {
    mount(root, { path: location.pathname, query: new URLSearchParams(location.search) }, leaving.signal);
  } catch (error) {
    // [LAW:no-silent-failure] a view that fails to mount says so where it was to be, as one that fails to load does.
    root.textContent = `The app could not show ${location.pathname}: ${error.message}`;
    throw error;
  }
}

// A move to a view of the app: the path changes and the key stays.
function go(path) {
  if (location.pathname === path) return;
  history.pushState(null, "", keyed(path));
  show();
}

for (const link of links) {
  link.href = keyed(link.getAttribute("href"));
  if (!mounts.has(link.pathname)) continue;
  link.addEventListener("click", event => {
    // A tap, not a request for the link in a tab of its own.
    if (event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    go(link.pathname);
  });
}

addEventListener("popstate", show);
show();
row.dataset.ready = "";
// The call is drawn as the view shown draws it, never as the document's markup has it for a frame.
hero.dataset.ready = "";
