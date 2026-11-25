// The Sessions view, as far as the app has it: the place on the row where the sessions hands sends off to work are shown,
// with nothing of them drawn yet.
export function mount(root, _route, _signal) {
  root.append(Object.assign(document.createElement("p"), { id: "sessions-note", textContent: "The sessions hands has sent off to work are not shown here yet." }));
}
