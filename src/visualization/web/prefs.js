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

export function onExternalGapsChange(listener) {
  listeners.add(listener);
  // Another tab flipping the same preference should not leave this one
  // showing something different.
  window.addEventListener('storage', (event) => {
    if (event.key === KEY) listener(externalGapsShown());
  });
  return () => listeners.delete(listener);
}
