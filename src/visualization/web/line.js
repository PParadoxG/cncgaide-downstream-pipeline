// Which kernel line the pages show, shared by every page.
//
// Two lines fill the run (contract/registry.py): `occ`, the open-source line
// (OCC geometry, AAGNet recognition), and `nx`, the NX line (NX geometry, NX
// feature-based recognition).  Each part carries one folder per line under
// the same file names, so choosing a line is a data-source selection plus a
// theme-token switch — nothing downstream branches on a line id to change
// logic.  It persists like the theme does, and the <html> `data-line`
// attribute is what style.css keys the accent tokens on.

const KEY = 'gatherSeedLine';
const DEFAULT_LINE = 'occ';

const LINES = [
  { id: 'occ', label: 'Open source' },
  { id: 'nx', label: 'NX FBM' },
];

export function currentLine() {
  try {
    const stored = localStorage.getItem(KEY);
    return LINES.some((line) => line.id === stored) ? stored : DEFAULT_LINE;
  } catch (error) {
    return DEFAULT_LINE;
  }
}

export function lineLabel(id) {
  const found = LINES.find((line) => line.id === id);
  return found ? found.label : id;
}

/**
 * Follow a line switch made in another tab, for a page that has no switch of
 * its own to keep in sync.
 *
 * The compare page is that page: it reads the stored choice once, before
 * first paint, and used to stay on the old line for the rest of its life
 * while the launcher next to it showed the other one (node 27G 6 #3).
 * Returns the remover.
 */
export function watchLineChange(onChange) {
  const handler = (event) => {
    if (event.key !== KEY) return;
    onChange(currentLine());
  };
  window.addEventListener('storage', handler);
  return () => window.removeEventListener('storage', handler);
}

let fadeTimer = null;

/** Reflect the line on <html> so the accent tokens follow.

    The `line-fade` class is held only for the moment of the switch: it is
    what lets the wider chrome (primary button, brand icon) cross-fade with
    the thumb while keeping their hover feedback instant otherwise. */
export function applyLineTheme(id) {
  const html = document.documentElement;
  if (html.dataset.line && html.dataset.line !== id) {
    html.classList.add('line-fade');
    clearTimeout(fadeTimer);
    fadeTimer = setTimeout(() => html.classList.remove('line-fade'), 600);
  }
  html.dataset.line = id;
}

/**
 * Build the two-segment switch into *container* (a `.segmented` shell with a
 * `.seg-thumb` already inside) and keep it, the stored choice and the theme
 * attribute in sync.  `onChange(id)` fires on every user switch and on a
 * change made in another tab.
 */
export function setupLineSwitch(container, onChange) {
  if (!container) return null;
  const buttons = LINES.map((line) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.dataset.line = line.id;
    button.textContent = line.label;
    button.title = `Show the ${line.label} line's results`;
    button.addEventListener('click', () => choose(line.id, true));
    container.append(button);
    return button;
  });

  function paint(id) {
    buttons.forEach((button, position) => {
      const on = button.dataset.line === id;
      button.setAttribute('aria-selected', String(on));
      if (on) container.dataset.active = String(position);
    });
    applyLineTheme(id);
  }

  function choose(id, notify) {
    try { localStorage.setItem(KEY, id); } catch (error) { /* choice just won't survive a reload */ }
    paint(id);
    if (notify && onChange) onChange(id);
  }

  // Another tab flipping the line should not leave this one showing a
  // different line under a different colour.
  watchLineChange((id) => {
    paint(id);
    if (onChange) onChange(id);
  });

  paint(currentLine());
  return { paint };
}
