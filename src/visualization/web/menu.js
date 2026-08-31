// Popover menu on a native <details>: the summary is the trigger, the list is
// the floating panel (styles: .menu / .menu-list / .menu-item).  One
// document-level listener closes any open menu on an outside pointer-down or
// Escape; choosing an item runs its handler and closes its own menu.  Items
// with a boolean `checked` draw as check items and stay open when chosen,
// so a menu of switches can be flipped without reopening it each time.

let installed = false;

/** Close-on-outside-click / Escape for every details.menu on the page. */
export function installMenus() {
  if (installed) return;
  installed = true;
  document.addEventListener('pointerdown', (event) => {
    for (const menu of document.querySelectorAll('details.menu[open]')) {
      if (!menu.contains(event.target)) menu.removeAttribute('open');
    }
  });
  document.addEventListener('keydown', (event) => {
    if (event.key !== 'Escape') return;
    for (const menu of document.querySelectorAll('details.menu[open]')) {
      menu.removeAttribute('open');
      const summary = menu.querySelector(':scope > summary');
      if (summary) summary.focus();
    }
  });
}

/** Kebab glyph for a "more actions" trigger; inline SVG so it centres. */
export const DOTS = '<svg viewBox="0 0 16 16" aria-hidden="true" width="15" height="15">'
  + '<circle cx="3" cy="8" r="1.4" fill="currentColor"/>'
  + '<circle cx="8" cy="8" r="1.4" fill="currentColor"/>'
  + '<circle cx="13" cy="8" r="1.4" fill="currentColor"/></svg>';

/**
 * createMenu(trigger, items, options) -> { element, update(items) }
 *
 * trigger  a string label or an element placed inside the <summary>
 * items    array of { label, title, onSelect, checked?, disabled? } or 'sep'
 * options  { title, summaryClass ('btn-text' | 'icon-btn' | …), align ('right' | 'left'), className }
 */
export function createMenu(trigger, items, options = {}) {
  installMenus();
  const details = document.createElement('details');
  details.className = 'menu' + (options.className ? ' ' + options.className : '');

  const summary = document.createElement('summary');
  summary.className = options.summaryClass || 'btn-text';
  if (typeof trigger === 'string') summary.textContent = trigger;
  else summary.append(trigger);
  if (options.title) summary.title = options.title;
  summary.setAttribute('aria-haspopup', 'menu');

  const list = document.createElement('div');
  list.className = 'menu-list' + (options.align === 'left' ? ' left' : '');
  list.setAttribute('role', 'menu');
  details.append(summary, list);

  function update(next) {
    list.innerHTML = '';
    for (const item of next) {
      if (item === 'sep') {
        const sep = document.createElement('div');
        sep.className = 'menu-sep';
        list.append(sep);
        continue;
      }
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'menu-item';
      button.setAttribute('role', typeof item.checked === 'boolean' ? 'menuitemcheckbox' : 'menuitem');
      if (typeof item.checked === 'boolean') button.setAttribute('aria-checked', String(item.checked));
      button.textContent = item.label;
      if (item.title) button.title = item.title;
      button.disabled = Boolean(item.disabled);
      button.addEventListener('click', () => {
        if (typeof item.checked !== 'boolean') details.removeAttribute('open');
        if (item.onSelect) item.onSelect();
      });
      list.append(button);
    }
  }
  update(items);
  return { element: details, update };
}
