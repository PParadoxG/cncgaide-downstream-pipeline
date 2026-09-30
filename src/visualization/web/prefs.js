// Review preferences that outlive a single page.
//
// Only one so far: whether to show the verdicts that could not be decided
// because an input outside this pipeline is missing (a tool magazine, design
// intent).  Those are not gaps in what the pipeline owes, and on a corpus
// where the magazine will never be connected they only crowd out the
// verdicts the geometry actually produced — so they stay out of sight until
// the reviewer asks for them, and that choice is kept across both pages.

const KEY = 'gatherSeedShowExternalGaps';
const listeners = new Set();

// Another tab flipping the same preference should not leave this one showing
// something different.  Installed once, for the module: it used to be
// installed inside onExternalGapsChange, so every subscriber added a window
// listener that the returned disposer - which only touches the Set - could
// never take off again.
window.addEventListener('storage', (event) => {
  if (event.key !== KEY) return;
  const shown = externalGapsShown();
  for (const listener of listeners) listener(shown);
});

/** Default off: the external inputs are somebody else's to connect. */
export function externalGapsShown() {
  try {
    return localStorage.getItem(KEY) === '1';
  } catch (error) {
    return false;
  }
}

export function setExternalGapsShown(shown) {
  try {
    localStorage.setItem(KEY, shown ? '1' : '0');
  } catch (error) {
    // A blocked storage must not break the page; the choice just will not
    // survive a reload.
  }
  for (const listener of listeners) listener(shown);
}

/** Subscribe; the returned function really does unsubscribe. */
export function onExternalGapsChange(listener) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}
