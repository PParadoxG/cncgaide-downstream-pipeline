// Part launcher: a searchable, sortable, filterable list — and the store's
// front door.
//
// Built for a corpus that will grow well past the current seven parts, so the
// layout is one row per part rather than a tile grid — the row stays scannable
// at 200 parts, and search/filter/sort do the narrowing.  Each row also shows
// what the store knows about the part (fresh / stale / queued / building /
// failed / source missing) and offers the two per-part actions; the header
// offers add / refresh stale / refresh all.  When a sync finishes, the rows
// are updated in place from the new index — the page never greys out.

import { setupTheme } from './theme.js';
import { installMenus } from './menu.js';
import { fetchParts, postRefresh, setupRefresh } from './refresh.js';
import { createStats } from './stats.js';
import { createSimilarity } from './similarity.js';
import { currentLine, lineLabel, setupLineSwitch } from './line.js';
import { reloadPage, renderRunBanner } from './runwatch.js';
import {
  actionsCell,
  addFiles,
  chipState,
  metaCell,
  removePartInteractive,
  renderJobBar,
  rerunPart,
  statusChip,
} from './parts.js';

const $ = (id) => document.getElementById(id);
const VIEWS = ['parts', 'heuristics', 'similarity'];
const BACKGROUND_KEY = 'gatherSeedShowBackgroundTypes';

const state = {
  parts: [],
  palette: {},
  search: '',
  types: new Set(),      // a part must carry every selected type
  sortKey: 'partId',
  sortAsc: true,
  typeMenu: null,        // the "Filter by type" popover, built once
  backgroundTypes: new Set(),  // feature types the viewer hides by default (from the index)
  featureLabels: {},     // class -> the words a human reads (from the index)
  showBackground: localStorage.getItem(BACKGROUND_KEY) === '1',
  view: 'parts',
  stats: null,
  statsRoot: null,       // stats mount inside #view-heuristics, under the line notice
  similarity: null,
  runId: null,
  partCount: 0,
  indexNotices: [],      // what the run said about itself (schema, self-checks)
  line: currentLine(),
  // store side
  store: new Map(),      // partId -> api/parts row
  status: null,          // last api/status payload
  writable: true,        // false under --no-refresh-api
  note: null,            // {text, error} shown in the job bar for a while
  noteTimer: null,
  lastStoreKey: '',
  refresher: null,
};

boot();

async function boot() {
  bindChrome();
  bindControls();
  bindStatsHost();
  state.similarity = createSimilarity($('view-similarity'), { line: () => state.line });
  bindViewTabs();
  bindStoreControls();
  bindProviderSwitch();
  let runId = null;
  try {
    runId = await loadIndex();
  } catch (error) {
    // 404 is the cold start — no refresh has ever finished here — and the
    // answer to it is on the page, not in a terminal.  Anything else means
    // the index exists but could not be read, which deserves its own words.
    $('run-line').textContent = error.status === 404
      ? 'no run built yet — add parts, then Refresh.'
      : `data/index.json not readable (${error.message}) — a refresh may be running.`;
  }
  await refreshStore();
  renderTypeFilter();
  render();
  state.refresher = setupRefresh($('refresh-btn'), $('refresh-label'), runId, {
    onStatus,
    onDataChanged,
  });

  // Audit hook, the same shape the part and compare pages already publish:
  // read-only handles, no hidden behaviour.  The similarity panel is reached
  // through it so its "a superseded ranking must not land" guard can be
  // provoked from a script instead of hoped for (node 27G 6 #7).
  window.tempLauncher = {
    state,
    get similarity() { return state.similarity; },
    get line() { return state.line; },
  };
}

/** (Re)load the viewer index; returns its run id. Safe to call repeatedly. */
async function loadIndex() {
  const response = await fetch('data/index.json', { cache: 'no-store' });
  if (!response.ok) {
    const error = new Error('HTTP ' + response.status);
    error.status = response.status;
    throw error;
  }
  const index = await response.json();

  state.parts = index.parts;
  state.palette = (index.palette && index.palette.features) || {};
  state.backgroundTypes = new Set(index.defaultHiddenFeatureTypes || []);
  // `{class: the words a human reads}`, folded across the lines that built
  // this run.  Absent for a line whose class names already are the words, in
  // which case `typeLabel` gives the class name back unchanged.
  state.featureLabels = index.featureLabels || {};
  state.runId = index.runId;
  state.partCount = index.parts.length;
  state.indexNotices = indexNotices(index);
  renderFooter();
  // loadIndex() runs again after every sync that changes the run id, so the
  // statistics view is updated in place rather than replaced: a second
  // instance over the same mount leaks the first one and its whole index.
  if (state.stats) state.stats.setIndex(index);
  else state.stats = createStats(state.statsRoot, index);
  if (state.view === 'heuristics') state.stats.render();
  state.similarity.setParts(index.parts);
  return index.runId;
}

// The prefix of every index schema this page knows how to read.  Not a
// version comparison: the page renders whatever it is given and says so when
// it does not recognise the shape, because refusing to draw a run is worse
// than drawing it with a caveat.
const INDEX_SCHEMA = 'v8.viewer_index.';

/**
 * What the run says about itself, in the run's own words.
 *
 * Both of these are written into the index and, until node 27, read by
 * nobody: `checkFailures` went to the terminal only, and the daily way in is
 * a browser, so a run whose own consistency check failed looked perfect on
 * screen; `schemaVersion` was never compared, so an index written by an older
 * pipeline rendered as if it were current, with missing fields quietly
 * falling back to empty.
 */
function indexNotices(index) {
  const notices = [];
  const schema = index.schemaVersion;
  if (!schema || !String(schema).startsWith(INDEX_SCHEMA)) {
    notices.push({
      text: 'unknown index format',
      title: `this page reads ${INDEX_SCHEMA}*, the run says "${schema || 'nothing'}"`
        + '\nshown anyway — anything this page cannot find is left empty',
    });
  }
  const failures = index.checkFailures || [];
  if (failures.length) {
    notices.push({
      text: `${failures.length} check failure(s)`,
      title: failures.map(String).join('\n'),
    });
  }
  return notices;
}

/**
 * Run id, part count, the line whose results the page is showing, and
 * anything the run said about itself.
 *
 * No path.  The footer used to carry the run's source directory in its
 * tooltip, taken from the index's `generatedFrom` - a directory on the
 * machine that produced the artifacts, which after a release export is not
 * even the same computer as the one reading the page.  Whatever the index
 * carries, none of it is printed here: the run id names the run, and the
 * directory serving it is the one in the address bar.
 */
function renderFooter() {
  const foot = $('run-line');
  if (state.runId) {
    foot.textContent = `${state.runId} · ${state.partCount} parts · ${lineLabel(state.line)}`;
  }
  for (const notice of state.indexNotices) {
    const chip = document.createElement('span');
    chip.className = 'foot-note';
    chip.textContent = notice.text;
    chip.title = notice.title;
    foot.append(chip);
  }
}

/**
 * What the shown line knows about one part — read from that line's own block
 * and from nowhere else.  The index also states the open-source line's numbers
 * at the entry's top level, and reading those for the line called `occ` is how
 * this used to work: a branch on the line's name to pick which field holds the
 * number, which is the one thing display code may not do.  A third line would
 * have walked silently into the other branch.
 *
 * `null` counts render as an em dash — "no number from this line" is not zero.
 * `unclaimedRawCadFaceCount` is one today for every line: it is a per-line
 * audit (which of the source STEP's faces no feature of *this* line claimed)
 * and the index carries it only at the entry's top level, where it is the
 * open-source line's answer with nothing in the data saying so.  Guessing
 * which line it belongs to is worse than not printing it; the index carrying
 * it per line is the fix, and that is a producer-side change.
 */
function lineCounts(part) {
  const info = (part.lines && part.lines[state.line]) || null;
  const built = info && info.status === 'PASS';
  return {
    featureCounts: (built && info.featureCounts) || {},
    featureNodeCount: built ? info.nodeCount : null,
    relationEdgeCount: built ? info.edgeCount : null,
    unclaimedRawCadFaceCount: built ? info.unclaimedRawCadFaceCount ?? null : null,
    producedAt: info ? info.producedAt : null,
  };
}

// The statistics view folds the shown line's own verdicts (node 20'b); it
// used to state that only the open-source corpus was folded, which is no
// longer true of either line.
function bindStatsHost() {
  const host = $('view-heuristics');
  const root = document.createElement('div');
  root.className = 'stats-root';
  host.append(root);
  state.statsRoot = root;
}

function bindProviderSwitch() {
  setupLineSwitch($('line-seg'), (id) => {
    state.line = id;
    onProviderChanged();
  });
  onProviderChanged();
}

function onProviderChanged() {
  // Both lines have a corpus since node 21', each with its own
  // representation, its own cost tables and its own cache; the tab is live on
  // either.  What does *not* carry across is a computed ranking - the two
  // scales are not comparable - so the view drops its result and waits for a
  // new Query rather than relabelling numbers from the other line.
  if (state.similarity && state.similarity.setLine) {
    state.similarity.setLine().catch((error) => console.error(error));
  }
  // The statistics view reads the shown line's fold, so switching lines
  // re-renders it - the two dialects' verdicts are never mixed or summed.
  if (state.view === 'heuristics' && state.stats) state.stats.render();
  renderFooter();
  renderTypeFilter();
  render();
}

/** Pull api/parts and re-render the rows in place. */
async function refreshStore() {
  const rows = await fetchParts();
  if (rows === null) return;
  state.store = new Map(rows.map((row) => [row.partId, row]));
  render();
}

// ------------------------------------------------------------ store events

function onStatus(status) {
  state.status = status;
  state.writable = status.state !== 'disabled';
  renderJobBar($('job-bar'), status, state.note);
  for (const id of ['add-parts-btn', 'refresh-all-btn']) {
    const control = $(id);
    if (control) control.classList.toggle('disabled', !state.writable);
  }
  // Fetch the rows while something is moving, and once more when it stops;
  // idle polls with an unchanged store summary cost nothing.
  const key = JSON.stringify([status.state, status.store, (status.queue || []).length]);
  if (status.state === 'running' || key !== state.lastStoreKey) {
    state.lastStoreKey = key;
    refreshStore();
  }
}

async function onDataChanged() {
  // The part list and the statistics are re-read here, so they are current
  // again.  The similarity view is not: a ranking or a pair on screen was
  // computed against the build that has just been replaced, and recomputing
  // it under the reviewer would silently change an answer they are reading.
  // The banner says which build the numbers came from and offers the reload
  // (node 26 G-03; the same sentence the part and compare pages use).
  const shownRun = state.runId;
  try {
    await loadIndex();
  } catch (error) {
    console.error(error);
    return;
  }
  if (shownRun && state.runId && shownRun !== state.runId) {
    renderRunBanner($('run-banner'), {
      shownRun, currentRun: state.runId, onReload: reloadPage,
    });
  }
  await refreshStore();
  renderTypeFilter();
  render();
}

function note(text, error = false) {
  state.note = { text, error };
  clearTimeout(state.noteTimer);
  state.noteTimer = setTimeout(() => { state.note = null; renderJobBar($('job-bar'), state.status, null); }, 8000);
  renderJobBar($('job-bar'), state.status, state.note);
}

const rowHandlers = {
  similar(partId) {
    state.similarity.setQuery(partId, true);
    selectView('similarity', true);
  },
  async rerun(partId) {
    try {
      const job = await rerunPart(partId);
      note(`${partId}: queued as ${job.jobId}`);
    } catch (error) {
      note(`${partId}: ${error.message}`, true);
    }
    if (state.refresher) state.refresher.poke();
  },
  async remove(partId) {
    try {
      const job = await removePartInteractive(partId);
      if (job) note(`${partId}: removal queued as ${job.jobId}`);
    } catch (error) {
      note(`${partId}: ${error.message}`, true);
    }
    if (state.refresher) state.refresher.poke();
  },
};

function bindStoreControls() {
  $('add-files').addEventListener('change', async (event) => {
    const files = [...event.target.files];
    event.target.value = '';
    if (!files.length || !state.writable) return;
    await addFiles(files, note);
    if (state.refresher) state.refresher.poke();
  });
  $('refresh-all-btn').addEventListener('click', async () => {
    if (!state.writable) return;
    if (!window.confirm('Reprocess every part? (STEP parsing, AAGNet, features, relations, heuristics)')) return;
    try {
      const job = await postRefresh({ mode: 'all' });
      note(`refresh all queued as ${job.jobId}`);
    } catch (error) {
      note(error.message, true);
    }
    if (state.refresher) state.refresher.poke();
  });
}

// --------------------------------------------------------------- filtering

/** Index parts plus store-only rows (not built yet, failed, untracked). */
function allParts() {
  const known = new Set(state.parts.map((part) => part.partId));
  const extra = [];
  for (const [partId, row] of state.store) {
    if (known.has(partId)) continue;
    // No `lines` block: every per-line count reads as "no number from this
    // line", which is what a part the run has never indexed actually is.
    extra.push({
      partId,
      rawCadFaceCount: null,
      _storeOnly: true,
      _row: row,
    });
  }
  return [...state.parts, ...extra];
}

function sortValue(part, key) {
  if (key === 'producedAt') {
    // "When was this part built" is a per-line question, so the answer comes
    // from the line's own block; parts the line has no block for sort as the
    // empty string and group together.
    return lineCounts(part).producedAt || '';
  }
  if (key === 'storeState') return chipState(state.store.get(part.partId)) || '';
  if (key === 'featureNodeCount' || key === 'relationEdgeCount' || key === 'unclaimedRawCadFaceCount') {
    return lineCounts(part)[key];
  }
  return part[key];
}

/** *all* is the list from allParts(), passed in so one render builds it once. */
function visibleParts(all) {
  const needle = state.search.trim().toLowerCase();

  const matched = all.filter((part) => {
    const featureCounts = lineCounts(part).featureCounts;
    for (const type of state.types) {
      if (!featureCounts[type]) return false;
    }
    if (!needle) return true;
    if (part.partId.toLowerCase().includes(needle)) return true;
    const chip = chipState(state.store.get(part.partId)) || '';
    if (chip.replace('_', ' ').includes(needle)) return true;
    // Both spellings match: someone reading "Through Hole" on screen should
    // be able to type it, and someone who knows the key should too.
    return Object.keys(featureCounts).some(
      (type) => type.toLowerCase().includes(needle)
        || typeLabel(type).toLowerCase().includes(needle));
  });

  const key = state.sortKey;
  const direction = state.sortAsc ? 1 : -1;
  return matched.sort((a, b) => {
    const left = sortValue(a, key);
    const right = sortValue(b, key);
    if (typeof left === 'string' || typeof right === 'string') {
      return String(left || '').localeCompare(String(right || '')) * direction;
    }
    return ((left || 0) - (right || 0)) * direction;
  });
}

// ---------------------------------------------------------------- rendering

function render() {
  // Everything below reads the same two derived things - the whole list, and
  // each row's counts for the shown line - and used to re-derive them: two
  // allParts() calls building two Maps, and lineCounts() once per part per
  // summary figure.  At nine parts that is nothing; this list is built for a
  // corpus well past two hundred, where it is five passes over the corpus for
  // one render.
  const all = allParts();
  const parts = visibleParts(all);
  const countsOf = new Map(parts.map((part) => [part.partId, lineCounts(part)]));
  const rows = $('rows');
  rows.innerHTML = '';
  const total = all.length;

  for (const part of parts) {
    const row = document.createElement('div');
    row.className = 'part-row' + (part._storeOnly ? ' store-only' : '');
    row.dataset.partId = part.partId;
    const storeRow = state.store.get(part.partId) || null;
    const openable = !part._storeOnly;
    const href = 'viewer.html?part=' + encodeURIComponent(part.partId);
    const counts = countsOf.get(part.partId);

    const name = document.createElement(openable ? 'a' : 'span');
    name.className = 'col-name name';
    name.textContent = part.partId;
    if (openable) name.href = href;

    row.append(
      name,
      statusChip(storeRow, state.line),
      metaCell(storeRow, counts.producedAt),
      cell('col-num', numberOrDash(part.rawCadFaceCount)),
      cell('col-num', numberOrDash(counts.featureNodeCount)),
      cell('col-num', numberOrDash(counts.relationEdgeCount)),
      cell('col-num', numberOrDash(counts.unclaimedRawCadFaceCount), counts.unclaimedRawCadFaceCount ? 'warn' : 'muted'),
      typeStrip(counts.featureCounts),
      actionsCell(storeRow, part.partId, rowHandlers, state.writable),
    );
    if (openable) {
      row.addEventListener('click', (event) => {
        if (event.target.closest('a, button')) return;
        if (event.ctrlKey || event.metaKey) window.open(href, '_blank');
        else location.href = href;
      });
    }
    rows.append(row);
  }

  // When every part is up to date no row draws a status chip, and a column
  // heading over nothing is just a label for empty space.  The heading comes
  // back the moment one part has something to say.
  const anyStatus = rows.querySelector('.col-status > *') !== null;
  $('part-list').classList.toggle('no-status', !anyStatus);

  // What the rows above add up to.  The list no longer stretches to the foot
  // of the window, so this sits at the end of the table rather than floating
  // in the space the table used to occupy.
  const summary = $('list-summary');
  summary.hidden = parts.length === 0;
  if (parts.length) {
    const sum = (key) => parts.reduce(
      (total_, part) => total_ + (countsOf.get(part.partId)[key] || 0), 0);
    const types = new Set();
    for (const part of parts) {
      for (const type of Object.keys(countsOf.get(part.partId).featureCounts)) types.add(type);
    }
    summary.innerHTML = '';
    for (const [value, label] of [
      [parts.length, parts.length === 1 ? 'part' : 'parts'],
      [sum('featureNodeCount'), 'features'],
      [types.size, 'feature types'],
      [sum('relationEdgeCount'), 'relations'],
      [parts.reduce((total_, part) => total_ + (part.rawCadFaceCount || 0), 0), 'CAD faces'],
    ]) {
      const item = document.createElement('span');
      item.innerHTML = `<b>${value}</b>${label}`;
      summary.append(item);
    }
  }

  const empty = $('empty-state');
  empty.hidden = parts.length > 0;
  // An empty store is not a filter that matched nothing: say what to do.
  empty.textContent = total === 0
    ? 'no parts yet — Add parts to build the first one'
    : 'no part matches the current filter';
  $('reset-filters').hidden = !state.search && state.types.size === 0;

  for (const header of document.querySelectorAll('[data-sort]')) {
    const active = header.dataset.sort === state.sortKey;
    header.classList.toggle('sorted', active);
    header.dataset.direction = active ? (state.sortAsc ? 'asc' : 'desc') : '';
  }
}

function numberOrDash(value) {
  return value === null || value === undefined ? '—' : value;
}

function cell(className, text, variant) {
  const span = document.createElement('span');
  span.className = className + (variant ? ' ' + variant : '');
  span.textContent = text;
  return span;
}

/**
 * One bar per row: a segment per feature type, sized by count, coloured by
 * the palette.  Hovering a segment names it and lists the whole breakdown; an
 * active type filter keeps the matched segments and fades the rest.  The
 * types the index names as hidden by default — NX's own plain-face classes,
 * and the notch classes this corpus shows to be mostly noise — stay out of
 * the bar unless the switch beside the filter is on.
 */
function typeStrip(featureCounts) {
  const wrap = document.createElement('span');
  wrap.className = 'col-types';
  const all = Object.entries(featureCounts)
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  const entries = state.showBackground ? all : all.filter(([type]) => !state.backgroundTypes.has(type));
  const left = all.filter(([type]) => !entries.some(([shown]) => shown === type));
  if (!entries.length) return wrap;

  const total = entries.reduce((sum, [, count]) => sum + count, 0);
  const breakdown = entries.map(([type, count]) => `${typeLabel(type)} ${count}`).join(' · ')
    + (left.length ? `\nnot shown: ${left.map(([type, count]) => `${typeLabel(type)} ${count}`).join(' · ')}` : '');
  const bar = document.createElement('span');
  bar.className = 'type-bar';
  for (const [type, count] of entries) {
    const segment = document.createElement('i');
    segment.className = 'seg'
      + (state.types.has(type) ? ' matched' : '')
      + (state.types.size && !state.types.has(type) ? ' dim' : '');
    segment.style.setProperty('--chip', state.palette[type] || '#888');
    segment.style.flexGrow = String(count);
    segment.title = `${typeLabel(type)} × ${count} of ${total} features\n${breakdown}`;
    bar.append(segment);
  }
  wrap.append(bar);
  return wrap;
}

/** The words a human reads for one class; the class name when it has none. */
function typeLabel(type) {
  return state.featureLabels[type] || type;
}

function renderTypeFilter() {
  const counts = new Map();
  for (const part of state.parts) {
    for (const type of Object.keys(lineCounts(part).featureCounts)) {
      counts.set(type, (counts.get(type) || 0) + 1);
    }
  }
  const ordered = [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  const menu = typeMenu();

  menu.label.textContent = state.types.size ? `Filter by type · ${state.types.size}` : 'Filter by type';
  menu.details.classList.toggle('active', state.types.size > 0);

  // The switch only exists when the index names something hidden by default.
  const background = [...state.backgroundTypes].sort();
  menu.background.hidden = background.length === 0;
  menu.background.setAttribute('aria-checked', String(state.showBackground));
  // "Background" is one of two reasons a type starts hidden - NX's own plain
  // faces are, the notch classes are noise we chose to fold away - so the
  // switch names what it does rather than why, and lists the types.
  menu.background.title = (state.showBackground ? 'hide ' : 'include ')
    + `the types hidden by default in the type bars: ${background.map(typeLabel).join(', ')}`;

  menu.panel.innerHTML = '';
  for (const [type, partCount] of ordered) {
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'type-chip' + (state.types.has(type) ? ' on' : '');
    chip.style.setProperty('--chip', state.palette[type] || '#888');
    chip.title = `${partCount} part(s) contain ${type} — filters combine with AND`;
    const swatch = document.createElement('span');
    swatch.className = 'swatch';
    const count = document.createElement('span');
    count.className = 'count';
    count.textContent = String(partCount);
    chip.append(swatch, document.createTextNode(`${typeLabel(type)} `), count);
    chip.addEventListener('click', () => toggleType(type));
    menu.panel.append(chip);
  }
  if (!ordered.length) {
    const empty = document.createElement('span');
    empty.className = 'insp-empty';
    empty.textContent = 'no feature type in this run';
    menu.panel.append(empty);
  }

  // The chosen types stay visible next to the trigger, each removable.
  menu.tokens.innerHTML = '';
  for (const type of [...state.types].sort()) {
    const token = document.createElement('span');
    token.className = 'chip-tint filter-token';
    token.style.setProperty('--chip', state.palette[type] || '#888');
    token.append(document.createTextNode(typeLabel(type)));
    token.title = type;
    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'x';
    remove.title = `stop filtering by ${type}`;
    remove.setAttribute('aria-label', `remove ${type} filter`);
    remove.textContent = '×';
    remove.addEventListener('click', () => toggleType(type));
    token.append(remove);
    menu.tokens.append(token);
  }
}

function toggleType(type) {
  if (state.types.has(type)) state.types.delete(type); else state.types.add(type);
  renderTypeFilter();
  render();
}

/** The popover is built once so it stays open while chips are toggled. */
function typeMenu() {
  if (state.typeMenu) return state.typeMenu;
  const host = $('type-filter');
  const details = document.createElement('details');
  details.className = 'menu type-menu';
  const summary = document.createElement('summary');
  summary.className = 'btn';
  summary.title = 'show only the parts that carry every chosen feature type';
  const label = document.createElement('span');
  const caret = document.createElement('span');
  caret.className = 'caret';
  caret.textContent = '▾';
  summary.append(label, caret);
  const panel = document.createElement('div');
  panel.className = 'menu-list left type-pop';
  details.append(summary, panel);
  const tokens = document.createElement('div');
  tokens.className = 'filter-tokens';
  const background = document.createElement('button');
  background.type = 'button';
  background.className = 'switch compact';
  background.setAttribute('role', 'switch');
  background.innerHTML = '<span class="switch-label">hidden types</span>'
    + '<span class="switch-track"><span class="switch-thumb"></span></span>';
  background.addEventListener('click', () => {
    state.showBackground = !state.showBackground;
    try { localStorage.setItem(BACKGROUND_KEY, state.showBackground ? '1' : '0'); } catch (error) { /* no storage: the choice just does not survive a reload */ }
    renderTypeFilter();
    render();
  });
  host.append(details, tokens, background);
  state.typeMenu = { details, label, panel, tokens, background };
  return state.typeMenu;
}

// ------------------------------------------------------------------ chrome

function bindControls() {
  installMenus();
  $('search').addEventListener('input', (event) => {
    state.search = event.target.value;
    render();
  });

  $('list-head').addEventListener('click', (event) => {
    const header = event.target.closest('[data-sort]');
    if (!header) return;
    const key = header.dataset.sort;
    if (state.sortKey === key) state.sortAsc = !state.sortAsc;
    else { state.sortKey = key; state.sortAsc = key === 'partId' || key === 'storeState'; }
    render();
  });

  $('reset-filters').addEventListener('click', () => {
    state.search = '';
    state.types.clear();
    $('search').value = '';
    renderTypeFilter();
    render();
  });

  window.addEventListener('keydown', (event) => {
    if (event.key === '/' && document.activeElement !== $('search')) {
      event.preventDefault();
      $('search').focus();
    } else if (event.key === 'Escape' && document.activeElement === $('search')) {
      $('search').value = '';
      state.search = '';
      render();
    }
  });
}

function bindChrome() {
  setupTheme($('theme-toggle'));
}

// Three views of one run: the part list, the heuristic statistics and the
// similarity query.  The choice lives in the URL so a link can point straight
// at either; `?view=similarity&query=<id>` (or the older `#similarity?query=`)
// preselects the query part, `&run=1` also sends the query.
function bindViewTabs() {
  const tabs = $('view-tabs');
  const params = new URLSearchParams(location.search);
  let requested = params.get('view');
  let query = params.get('query');
  let run = params.get('run') === '1';
  const hash = location.hash.replace(/^#/, '');
  if (hash.startsWith('similarity')) {
    requested = 'similarity';
    const hashParams = new URLSearchParams(hash.split('?')[1] || '');
    query = hashParams.get('query') || query;
    run = run || hashParams.get('run') === '1';
  }
  if (query || params.get('weights') || params.get('config') || params.get('advanced') || params.get('candidates')) {
    state.similarity.setQuery(query, run, {
      configName: params.get('config'),
      weights: params.get('weights'),
      advanced: params.has('advanced') ? params.get('advanced') === '1' : undefined,
      candidates: (params.get('candidates') || '').split(',').map((id) => id.trim()).filter(Boolean),
    });
  }
  tabs.addEventListener('click', (event) => {
    const button = event.target.closest('button[data-view]');
    if (button) selectView(button.dataset.view, true);
  });
  selectView(VIEWS.includes(requested) ? requested : 'parts', false);
}

function selectView(view, pushUrl) {
  state.view = view;
  const tabs = $('view-tabs');
  const buttons = [...tabs.querySelectorAll('button[data-view]')];
  buttons.forEach((button, position) => {
    const on = button.dataset.view === view;
    button.setAttribute('aria-selected', String(on));
    if (on) tabs.dataset.active = String(position);
  });

  $('view-parts').hidden = view !== 'parts';
  $('view-heuristics').hidden = view !== 'heuristics';
  $('view-similarity').hidden = view !== 'similarity';
  // One primary action per view: adding parts is it on the Parts view, the
  // query button on Similarity; elsewhere the add button steps back.
  const add = $('add-parts-btn');
  if (add) {
    add.classList.toggle('btn-primary', view === 'parts');
    add.classList.toggle('btn', view !== 'parts');
  }
  if (view === 'heuristics' && state.stats) state.stats.render();
  if (view === 'similarity' && state.similarity) state.similarity.render().catch((error) => console.error(error));

  if (pushUrl) {
    const url = new URL(location.href);
    if (view === 'parts') url.searchParams.delete('view');
    else url.searchParams.set('view', view);
    history.replaceState(null, '', url);
  }
}
