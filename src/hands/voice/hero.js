// The hero's shape: the one writer of how far the call's big round button has melted into the bar. `reach` is the position
// along the way, 0 the circle and 1 the bar; app.js sets it to 1 as each view is shown, and the Call view sets it from the
// scroll. app.html's stylesheet draws every length of the button, and the two lines beside it, from the `--melt` it writes.
export const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
const hero = document.getElementById("hero");

// The class is the one change of layout between the shapes. It is made at half, where the lines beside the button have faded
// out under it and fade in beside it (app.html), so no one sees it made. Under reduced motion the shape is the circle or the
// bar and no between, switching at the same half.
export function shape(reach) {
  const clamped = Math.min(1, Math.max(0, reach));
  const melt = reducedMotion.matches ? Math.round(clamped) : clamped;
  hero.style.setProperty("--melt", String(melt));
  hero.classList.toggle("bar", melt >= 0.5);
}
