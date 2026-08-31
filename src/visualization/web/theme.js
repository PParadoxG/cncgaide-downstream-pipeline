// Shared theme control: an inline-SVG contrast icon plus the persisted choice.

const KEY = 'tempVisualTheme';

const ICON = `<svg viewBox="0 0 16 16" aria-hidden="true">
  <circle cx="8" cy="8" r="6"/>
  <path d="M8 2a6 6 0 0 0 0 12z" fill="currentColor" stroke="none"/>
</svg>`;

export function setupTheme(button, onChange) {
  button.innerHTML = ICON;

  function paint() {
    const light = document.documentElement.dataset.theme === 'light';
    button.title = light ? 'Switch to dark' : 'Switch to light';
    button.setAttribute('aria-label', button.title);
  }

  button.addEventListener('click', () => {
    const theme = document.documentElement.dataset.theme === 'light' ? 'dark' : 'light';
    document.documentElement.dataset.theme = theme;
    localStorage.setItem(KEY, theme);
    paint();
    if (onChange) onChange(theme);
  });

  paint();
}
