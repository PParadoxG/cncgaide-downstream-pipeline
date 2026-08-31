// Compare page: two parts side by side in 3D, the GED score list and the edit
// path rendered as data.
//
// The page never interprets an edit: entries are grouped by the section of
// the edit path they come from (nodes / edges / fixed_nodes / scale) and by
// their `operation` string, and every entry is drawn from its own fields —
// ids and types on the two sides, the cost, and whatever else the solver
// wrote (attribute rows, details, base weight…).  A new operation, channel
// or attribute shows up without any change here.  Feature identity for the
// highlight comes from each part's view.json (built from id_map.json), the
// same way the part page selects a feature.

import { loadIndex, loadPart } from './data.js';
import { applyLineTheme, currentLine } from './line.js';
import { createStore, initialState, resolveSelection } from './store.js';
import { createViewer3D } from './viewer3d.js';
import { setupTheme } from './theme.js';

const $ = (id) => document.getElementById(id);
const DEFAULT_MICRO = 1e-3;              // variants.MICRO_EDIT_COST
const SECTION_ORDER = ['nodes', 'edges', 'fixed_nodes', 'scale'];

const sides = {
  a: { key: 'a', store: null, viewer: null, model: null, partId: null },
  b: { key: 'b', store: null, viewer: null, model: null, partId: null },
};

const state = {
  index: null,
  // Which kernel line both sides are read from.  A comparison is always
  // within one line: the two ontologies do not share a scale, so there is no
  // meaning to a distance across them (node 21').  The page follows the
  // launcher's stored choice and paints the matching accent.
  line: currentLine(),
  configs: null,            // /api/similarity/configs payload
  configName: null,         // preset name used for the request
  weightsText: '',          // K=V,… overrides carried in the URL
  payload: null,            // /api/similarity/compare payload (oriented a -> b)
  direction: 'a_to_b',
  hideMicro: true,
  micro: DEFAULT_MICRO,
  collapsed: new Set(),     // group keys folded by the reviewer
  entries: [],              // flat list of visible, clickable entries in display order
  selected: null,           // index into entries
  linkCameras: false,
  error: null,
};

// ------------------------------------------------------------------- boot

/** The kernel line, on every similarity request this page makes. */
function lineParams(params) {
  const out = params || new URLSearchParams();
  if (state.line) out.set('line', state.line);
  return out;
}

async function boot() {
  const params = new URLSearchParams(location.search);
  applyLineTheme(state.line);
  state.configName = params.get('config') || null;
  state.weightsText = params.get('weights') || '';
  if (params.get('micro')) state.micro = Number(params.get('micro')) || DEFAULT_MICRO;

  setupTheme($('theme-toggle'), (theme) => {
    for (const side of Object.values(sides)) side.store.update({ theme });
  });

  for (const side of Object.values(sides)) {
    side.store = createStore(initialState());
    side.viewer = createViewer3D($(`viewport-${side.key}`), side.store, viewerCallbacks(side));
  }

  state.index = await loadIndex();
  const ids = state.index.parts.map((part) => part.partId);
  for (const side of Object.values(sides)) {
    const select = $(`part-${side.key}`);
    for (const id of ids) {
      const option = document.createElement('option');
      option.value = id;
      option.textContent = id;
      select.append(option);
    }
    select.addEventListener('change', () => {
      side.partId = select.value;
      reload({ parts: [side.key] });
    });
  }
  const wanted = { a: params.get('a'), b: params.get('b') };
  sides.a.partId = ids.includes(wanted.a) ? wanted.a : ids[0];
  sides.b.partId = ids.includes(wanted.b) ? wanted.b : (ids.find((id) => id !== sides.a.partId) || ids[0]);
  $('part-a').value = sides.a.partId;
  $('part-b').value = sides.b.partId;

  await loadConfigs();
  bindChrome();
  bindKeys();

  // Audit hook: read-only handles, no hidden behaviour.
  window.tempCompare = {
    sides,
    state,
    selectEntry,
    setMicro(on, threshold) { state.hideMicro = Boolean(on); if (threshold) state.micro = threshold; renderPath(); },
    setDirection(direction) { state.direction = direction; renderPath(); },
    resolve: (key) => resolveSelection(sides[key].store.get(), sides[key].model),
  };

  await reload({ parts: ['a', 'b'] });
}

async function loadConfigs() {
  try {
    const response = await fetch(`api/similarity/configs?${lineParams()}`, { cache: 'no-store' });
    state.configs = response.ok ? await response.json() : null;
  } catch (error) {
    state.configs = null;
  }
  const select = $('config-select');
  select.innerHTML = '';
  const presets = (state.configs && state.configs.configs) || [];
  for (const preset of presets) {
    const option = document.createElement('option');
    option.value = preset.name;
    option.textContent = preset.name;
    select.append(option);
  }
  if (state.weightsText) {
    const option = document.createElement('option');
    option.value = '__custom__';
    option.textContent = `custom (${state.weightsText})`;
    select.append(option);
    select.value = '__custom__';
  } else {
    if (!state.configName || !presets.some((p) => p.name === state.configName)) {
      state.configName = (state.configs && state.configs.default) || (presets[0] && presets[0].name) || null;
    }
    if (state.configName) select.value = state.configName;
  }
  select.addEventListener('change', () => {
    if (select.value === '__custom__') return;
    state.configName = select.value;
    state.weightsText = '';
    reload({ parts: [] });
  });
}

/** Reload the named sides' parts, then the pair result; keeps the URL honest. */
async function reload({ parts }) {
  $('loading').classList.remove('hidden');
  state.error = null;
  const url = new URL(location.href);
  url.searchParams.set('a', sides.a.partId);
  url.searchParams.set('b', sides.b.partId);
  if (state.configName) url.searchParams.set('config', state.configName); else url.searchParams.delete('config');
  if (state.weightsText) url.searchParams.set('weights', state.weightsText); else url.searchParams.delete('weights');
  history.replaceState(null, '', url);
  $('find-similar').href = `index.html?view=similarity&run=1&query=${encodeURIComponent(sides.a.partId)}`
    + (state.configName ? `&config=${encodeURIComponent(state.configName)}` : '')
    + (state.weightsText ? `&weights=${encodeURIComponent(state.weightsText)}` : '');

  try {
    await Promise.all(parts.map((key) => loadSide(sides[key])));
    await loadPair();
  } catch (error) {
    state.error = error.message;
    console.error(error);
  } finally {
    $('loading').classList.add('hidden');
  }
  renderScore();
  renderPath();
}

async function loadSide(side) {
  side.model = await loadPart(side.partId, state.line);
  side.store.update({
    selection: null,
    hover: null,
    hiddenTypes: new Set(side.model.defaultHiddenFeatureTypes),
    visibleEdgeTypes: new Set(),
  });
  side.viewer.setModel(side.model);
  $(`hud-${side.key}`).textContent = `${side.key.toUpperCase()} · ${side.model.partId} · ${side.model.stats.rawCadFaceCount} faces`;
}

async function loadPair() {
  const params = new URLSearchParams({ a: sides.a.partId, b: sides.b.partId });
  if (state.configName) params.set('config', state.configName);
  if (state.weightsText) params.set('weights', state.weightsText);
  const response = await fetch(`api/similarity/compare?${lineParams(params)}`, { cache: 'no-store' });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `compare: HTTP ${response.status}`);
  state.payload = payload;
  state.selected = null;
}

// ------------------------------------------------------------------ score

function renderScore() {
  const root = $('score');
  root.innerHTML = '';
  if (state.error) {
    root.append(item('error', state.error, '', 'error'));
    return;
  }
  const payload = state.payload;
  if (!payload) return;
  const list = document.createElement('ul');
  list.className = 'score-list';

  // One score today (feature-level GED): the similarity to read, the
  // distance it comes from beside it.  The list form is what a second score
  // would slot into.
  const main = item('similarity', `${Number(payload.similarity_percent).toFixed(2)} %`,
    `distance ${fmt(payload.normalized_distance, 4)}`);
  main.classList.add('main');
  main.title = `feature-level GED · normalised distance ${Number(payload.normalized_distance).toFixed(6)}`
    + ' · similarity = 100 · (1 − distance)';
  list.append(main);

  // Channel bar: every breakdown key that is a cost.
  const breakdown = payload.breakdown || {};
  const costKeys = Object.keys(breakdown).filter((key) => /_cost$/.test(key));
  const total = costKeys.reduce((sum, key) => sum + Number(breakdown[key] || 0), 0);
  const bar = document.createElement('div');
  bar.className = 'score-bar';
  const legend = document.createElement('div');
  legend.className = 'score-legend';
  costKeys.forEach((key, position) => {
    const value = Number(breakdown[key] || 0);
    const piece = document.createElement('span');
    piece.className = `piece c${position % 6}`;
    piece.style.width = total > 0 ? `${(value / total) * 100}%` : '0%';
    piece.title = `${key} ${fmt(value, 4)}`;
    bar.append(piece);
    const tag = document.createElement('span');
    tag.className = `tag c${position % 6}`;
    tag.innerHTML = `<i></i>${label(key)} <b>${fmt(value, 4)}</b>`;
    legend.append(tag);
  });
  const channels = item('cost channels', '', '');
  channels.classList.add('channels');
  channels.querySelector('.value').append(bar);
  channels.querySelector('.sub').append(legend);
  list.append(channels);

  // One line of provenance: the configuration (the hash appears here only;
  // its weights and the raw / upper bound in the tooltip), when the two
  // parts were built, whether the pair came from the cache.
  const config = payload.config || {};
  const meta = document.createElement('li');
  meta.className = 'score-item meta';
  const chip = text('sim-hash', `${config.name} · #${config.hash}`);
  // The pair payload does not say whether its configuration is a preset;
  // the configs payload lists the presets with their hashes, so ask that.
  const presets = (state.configs && state.configs.configs) || [];
  const isPreset = config.preset !== undefined ? Boolean(config.preset) : presets.some((preset) => preset.hash === config.hash);
  chip.dataset.kind = isPreset ? 'preset' : 'custom';
  chip.title = Object.entries(config.weights || {}).map(([k, v]) => `${k} = ${v}`)
    .concat(['', `raw cost ${fmt(payload.raw_cost, 4)} / upper bound ${fmt(payload.upper_bound, 4)} (mean of both directions)`])
    .join('\n');
  const produced = payload.producedAt || {};
  const built = text('', `built a ${short(produced.a)} · b ${short(produced.b)}`);
  built.title = `${produced.a || '—'}\n${produced.b || '—'}`;
  const result = text('', payload.cached ? 'cached' : 'computed now');
  result.title = `${payload.computedAt || ''}${payload.timingMs !== undefined ? ' · ' + payload.timingMs + ' ms' : ''}`;
  meta.append(chip, built, result);
  list.append(meta);

  root.append(list);
}

function text(className, content) {
  const element = document.createElement('span');
  if (className) element.className = className;
  element.textContent = content;
  return element;
}

function item(labelText, valueText, subText, tone) {
  const li = document.createElement('li');
  li.className = 'score-item' + (tone ? ' ' + tone : '');
  const lab = document.createElement('span');
  lab.className = 'label';
  lab.textContent = labelText;
  const val = document.createElement('span');
  val.className = 'value';
  val.textContent = valueText;
  const sub = document.createElement('span');
  sub.className = 'sub';
  sub.textContent = subText;
  li.append(lab, val, sub);
  return li;
}

// -------------------------------------------------------------- edit path

function renderPath() {
  const head = $('path-head');
  const body = $('path');
  head.innerHTML = '';
  body.innerHTML = '';
  state.entries = [];
  const payload = state.payload;
  if (!payload || !payload.edit_path) {
    body.append(note(state.error || 'no pair result'));
    return;
  }
  const directions = Object.keys(payload.edit_path);
  if (!directions.includes(state.direction)) state.direction = directions[0];
  const block = payload.edit_path[state.direction] || {};

  // --- header: direction, micro filter, totals -------------------------
  const seg = document.createElement('div');
  seg.className = 'segmented cmp-dir';
  seg.style.setProperty('--segments', String(directions.length));
  seg.dataset.active = String(directions.indexOf(state.direction));
  seg.innerHTML = '<span class="seg-thumb" aria-hidden="true"></span>';
  for (const direction of directions) {
    const button = document.createElement('button');
    button.type = 'button';
    button.dataset.direction = direction;
    button.setAttribute('aria-selected', String(direction === state.direction));
    button.textContent = direction.replace(/_to_/, ' → ');
    button.title = direction === 'a_to_b' ? 'edits that turn a into b' : 'edits that turn b into a';
    button.addEventListener('click', () => { state.direction = direction; state.selected = null; clearSelection(); renderPath(); });
    seg.append(button);
  }
  const micro = document.createElement('label');
  micro.className = 'cmp-micro';
  micro.title = 'fold edits whose cost is below the threshold (normalised-metric drift with L_ref, zero-cost preserved edges)';
  const microSwitch = switchButton('hide micro edits', state.hideMicro, () => { state.hideMicro = !state.hideMicro; renderPath(); });
  const microInput = document.createElement('input');
  microInput.type = 'number';
  microInput.step = 'any';
  microInput.min = '0';
  microInput.value = String(state.micro);
  microInput.className = 'cmp-micro-input';
  microInput.title = 'micro-edit threshold (cost)';
  microInput.addEventListener('change', () => {
    const value = Number(microInput.value);
    state.micro = Number.isFinite(value) && value >= 0 ? value : DEFAULT_MICRO;
    microInput.value = String(state.micro);
    renderPath();
  });
  micro.append(microSwitch, document.createTextNode('<'), microInput);
  head.append(seg, micro);

  // --- groups ----------------------------------------------------------
  const groups = groupEntries(block);
  let listedCost = 0;
  let allCost = 0;
  let hiddenCount = 0;
  let allCount = 0;
  for (const group of groups) {
    allCost += group.total;
    allCount += group.entries.length;
    const visible = group.entries.filter((entry) => !isMicro(entry));
    hiddenCount += group.entries.length - visible.length;
    listedCost += visible.reduce((sum, entry) => sum + Number(entry.cost || 0), 0);
    body.append(groupElement(group, visible));
  }

  const other = directions.find((direction) => direction !== state.direction);
  const otherTotal = other ? directionTotal(payload.edit_path[other]) : null;
  const mean = otherTotal === null ? allCost : 0.5 * (allCost + otherTotal);
  const consistent = Math.abs(mean - Number(payload.raw_cost)) < 1e-6;
  const totals = document.createElement('div');
  totals.className = 'cmp-totals' + (consistent ? '' : ' mismatch');
  totals.innerHTML =
    `<span>listed <b>${fmt(listedCost, 4)}</b> of <b>${fmt(allCost, 4)}</b> = raw cost ${state.direction.replace(/_to_/, '→')}`
    + (hiddenCount ? ` · ${hiddenCount} of ${allCount} edit(s) below ${state.micro} folded` : ` · ${allCount} edit(s)`)
    + '</span>'
    + (other ? `<span>mean(${directions.map((d) => d.replace(/_to_/, '→')).join(', ')}) = <b>${fmt(mean, 4)}</b> `
      + `${consistent ? '=' : '≠'} result raw cost <b>${fmt(payload.raw_cost, 4)}</b>${consistent ? ' ✓' : ' ✗'}</span>` : '');
  body.append(totals);

  if (state.selected !== null && state.selected >= state.entries.length) state.selected = null;
  paintSelection();
}

/** Every section's entries, grouped by operation in display order. */
function groupEntries(block) {
  const sections = Object.keys(block).sort((x, y) => {
    const ix = SECTION_ORDER.indexOf(x);
    const iy = SECTION_ORDER.indexOf(y);
    return (ix < 0 ? 99 : ix) - (iy < 0 ? 99 : iy) || x.localeCompare(y);
  });
  const groups = [];
  for (const section of sections) {
    const value = block[section];
    const list = Array.isArray(value) ? value : (value && typeof value === 'object' ? [value] : []);
    const byOperation = new Map();
    for (const entry of list) {
      const operation = entry.operation || section;
      const key = `${section}:${operation}`;
      if (!byOperation.has(key)) byOperation.set(key, { key, section, operation, entries: [], total: 0 });
      const group = byOperation.get(key);
      group.entries.push(entry);
      group.total += Number(entry.cost || 0);
    }
    groups.push(...byOperation.values());
  }
  return groups;
}

function directionTotal(block) {
  let total = 0;
  for (const value of Object.values(block || {})) {
    const list = Array.isArray(value) ? value : (value && typeof value === 'object' ? [value] : []);
    for (const entry of list) total += Number(entry.cost || 0);
  }
  return total;
}

function isMicro(entry) {
  return state.hideMicro && Number(entry.cost || 0) < state.micro;
}

function groupElement(group, visible) {
  const wrap = document.createElement('div');
  wrap.className = 'cmp-group';
  const collapsed = state.collapsed.has(group.key);
  const head = document.createElement('button');
  head.type = 'button';
  head.className = 'cmp-group-head';
  head.setAttribute('aria-expanded', String(!collapsed));
  const hidden = group.entries.length - visible.length;
  head.innerHTML =
    `<span class="caret">${collapsed ? '▸' : '▾'}</span>`
    + `<span class="section">${escapeHtml(group.section.replace(/_/g, ' '))}</span>`
    + `<span class="op">${escapeHtml(String(group.operation).replace(/_/g, ' '))}</span>`
    + `<span class="count">${group.entries.length}${hidden ? ` <i>(${hidden} folded)</i>` : ''}</span>`
    + `<span class="total">Σ ${fmt(group.total, 4)}</span>`;
  head.addEventListener('click', () => {
    if (state.collapsed.has(group.key)) state.collapsed.delete(group.key); else state.collapsed.add(group.key);
    renderPath();
  });
  wrap.append(head);
  if (collapsed) return wrap;
  const list = document.createElement('div');
  list.className = 'cmp-entries';
  if (!visible.length) {
    list.append(note(group.entries.length ? 'all edits of this kind are below the micro threshold' : 'none'));
  }
  for (const entry of visible) list.append(entryElement(entry, group));
  wrap.append(list);
  return wrap;
}

// Which features an entry names on each side.  Strings are node ids, edge
// endpoint arrays are node ids; anything else (a fixed aggregate's raw dict,
// the scale record) names no feature and is a summary row.
function sideIds(entry, which) {
  const ids = [];
  const single = entry[which];
  if (typeof single === 'string') ids.push(single);
  const pair = entry[`${which}_edge`];
  if (Array.isArray(pair)) for (const id of pair) if (typeof id === 'string') ids.push(id);
  return ids;
}

function entryElement(entry, group) {
  const left = sideIds(entry, 'source');
  const right = sideIds(entry, 'target');
  const clickable = left.length > 0 || right.length > 0;
  const row = document.createElement('div');
  row.className = 'cmp-entry' + (clickable ? ' clickable' : ' summary');
  const index = clickable ? state.entries.push({ entry, group, left, right, element: row }) - 1 : null;
  if (index !== null) row.dataset.index = String(index);

  const line = document.createElement('div');
  line.className = 'cmp-entry-line';
  const sideA = document.createElement('span');
  sideA.className = 'side a';
  sideA.append(...idTags(left, entry.source_type, sides.a.model));
  if (!left.length && entry.source && typeof entry.source === 'object') sideA.append(tag(entry.type || 'aggregate', 'agg'));
  const arrow = document.createElement('span');
  arrow.className = 'arrow';
  arrow.textContent = left.length && right.length ? '→' : (left.length ? '✕' : (right.length ? '+' : '·'));
  const sideB = document.createElement('span');
  sideB.className = 'side b';
  sideB.append(...idTags(right, entry.target_type, sides.b.model));
  if (!right.length && entry.target && typeof entry.target === 'object') sideB.append(tag(entry.type || 'aggregate', 'agg'));
  if (entry.type && (left.length || right.length)) sideA.prepend(tag(entry.type, 'rel'));
  const cost = document.createElement('span');
  cost.className = 'cost' + (Number(entry.cost || 0) === 0 ? ' zero' : '');
  cost.textContent = fmt(entry.cost, 4);
  line.append(sideA, arrow, sideB, cost);
  row.append(line);

  // A compact summary always — the largest contributor, and how many more
  // there are — with the whole list in the tooltip; the full field dump only
  // while the entry is the active one (see the stylesheet).
  const bits = summaryBits(entry);
  if (bits.length) {
    const brief = document.createElement('div');
    brief.className = 'cmp-summary';
    brief.textContent = bits[0] + (bits.length > 1 ? `  +${bits.length - 1} more` : '');
    brief.title = bits.join(' · ');
    row.append(brief);
  }
  const detail = detailElement(entry);
  if (detail) row.append(detail);

  if (clickable) {
    row.addEventListener('click', () => selectEntry(index));
    row.addEventListener('dblclick', () => { selectEntry(index); frameEntry(index); });
  }
  return row;
}

function idTags(ids, typeName, model) {
  const out = [];
  ids.forEach((id, position) => {
    const node = model && model.nodeById.get(id);
    const chip = document.createElement('span');
    chip.className = 'id' + (node ? '' : ' unknown');
    if (node) chip.style.setProperty('--chip', node.color);
    chip.textContent = id;
    chip.title = node
      ? `${node.featureType} · ${node.cadFaceIds.length} raw face(s)`
      : `${typeName ? typeName + ' · ' : ''}not in this part's view (hidden type or missing)`;
    out.push(chip);
    if (position < ids.length - 1) out.push(document.createTextNode(' — '));
  });
  return out;
}

function tag(text, cls) {
  const el = document.createElement('span');
  el.className = `tag ${cls || ''}`;
  el.textContent = text;
  return el;
}

// The rest of the entry, rendered by shape: scalars as key/value, arrays of
// objects as a small table, other objects as nested key/value.
const LINE_FIELDS = new Set(['operation', 'source', 'target', 'source_type', 'target_type', 'source_edge', 'target_edge', 'type', 'cost']);

function detailElement(entry) {
  const rest = Object.entries(entry).filter(([key]) => !LINE_FIELDS.has(key));
  // The aggregate rows keep their raw dictionaries: they are the whole point
  // of a fixed-node substitution.
  if (typeof entry.source === 'object' && entry.source && !Array.isArray(entry.source)) rest.unshift(['source', entry.source]);
  if (typeof entry.target === 'object' && entry.target && !Array.isArray(entry.target)) rest.unshift(['target', entry.target]);
  if (!rest.length) return null;
  const box = document.createElement('div');
  box.className = 'cmp-detail';
  for (const [key, value] of rest) box.append(renderValue(key, value));
  return box;
}

/**
 * What the entry carries besides ids and cost, by shape, as short phrases:
 * the attribute rows that moved first (largest distance first), then the
 * scalar fields.  Nothing here knows an attribute or channel name.
 */
function summaryBits(entry) {
  const attributes = [];
  const scalars = [];
  const isObject = (v) => v && typeof v === 'object' && !Array.isArray(v);
  for (const [key, value] of Object.entries(entry)) {
    if (LINE_FIELDS.has(key)) continue;
    if (Array.isArray(value) && value.length && value.every(isObject)) {
      for (const row of value) {
        const moved = Number(row.distance) > 0 || (row.status && row.status !== 'compared');
        if (!moved) continue;
        const name = row.attribute ?? row.name ?? key;
        const span = row.left !== undefined || row.right !== undefined ? ` ${scalar(row.left)}→${scalar(row.right)}` : '';
        const dist = row.distance !== undefined ? ` (${fmt(row.distance, 3)})` : (row.status ? ` (${row.status})` : '');
        attributes.push({ weight: Number(row.distance) || 0, text: `${name}${span}${dist}` });
      }
    } else if (isObject(value)) {
      for (const [k, v] of Object.entries(value)) {
        if (isObject(v) || Array.isArray(v)) continue;
        if (v === 0 || v === null || v === undefined || v === '') continue;
        scalars.push(`${k} ${scalar(v)}`);
      }
    } else if (value !== null && value !== undefined && typeof value !== 'object' && key !== 'attributes_compared') {
      if (typeof value === 'number' && value === 0) continue;
      scalars.push(`${key.replace(/_/g, ' ')} ${scalar(value)}`);
    }
  }
  attributes.sort((x, y) => y.weight - x.weight);
  return [...attributes.map((item) => item.text), ...scalars];
}

function renderValue(key, value) {
  const wrap = document.createElement('div');
  wrap.className = 'kv';
  if (Array.isArray(value) && value.length && value.every((v) => v && typeof v === 'object' && !Array.isArray(v))) {
    wrap.append(kvLabel(key));
    wrap.append(miniTable(value));
    return wrap;
  }
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    wrap.append(kvLabel(key));
    const nested = document.createElement('div');
    nested.className = 'nested';
    for (const [k, v] of Object.entries(value)) nested.append(renderValue(k, v));
    wrap.append(nested);
    return wrap;
  }
  wrap.append(kvLabel(key));
  const val = document.createElement('span');
  val.className = 'v' + (value === null || value === undefined ? ' null' : '');
  val.textContent = Array.isArray(value) ? value.map(scalar).join(', ') : scalar(value);
  wrap.append(val);
  return wrap;
}

function kvLabel(key) {
  const label = document.createElement('span');
  label.className = 'k';
  label.textContent = key.replace(/_/g, ' ');
  return label;
}

function miniTable(rows) {
  const columns = [...new Set(rows.flatMap((row) => Object.keys(row)))];
  const table = document.createElement('table');
  table.className = 'cmp-mini';
  const head = document.createElement('tr');
  for (const column of columns) {
    const th = document.createElement('th');
    th.textContent = column.replace(/_/g, ' ');
    head.append(th);
  }
  table.append(head);
  for (const row of rows) {
    const tr = document.createElement('tr');
    const changed = Number(row.distance) > 0;
    if (changed) tr.className = 'changed';
    for (const column of columns) {
      const td = document.createElement('td');
      const value = row[column];
      td.textContent = value === undefined ? '' : scalar(value);
      if (value === null) td.className = 'null';
      tr.append(td);
    }
    table.append(tr);
  }
  return table;
}

function scalar(value) {
  if (value === null || value === undefined) return '—';
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : fmt(value, 4);
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
}

// ------------------------------------------------------------- selection

function selectEntry(index) {
  if (index === null || index === undefined || !state.entries[index]) {
    state.selected = null;
    clearSelection();
    paintSelection();
    return;
  }
  state.selected = index;
  const { left, right } = state.entries[index];
  sides.a.store.update({ selection: left.length ? { kind: 'features', featureIds: left } : null });
  sides.b.store.update({ selection: right.length ? { kind: 'features', featureIds: right } : null });
  paintSelection();
  const element = state.entries[index].element;
  if (element && element.scrollIntoView) element.scrollIntoView({ block: 'nearest' });
}

function clearSelection() {
  sides.a.store.update({ selection: null, hover: null });
  sides.b.store.update({ selection: null, hover: null });
}

function paintSelection() {
  state.entries.forEach((item, position) => item.element.classList.toggle('active', position === state.selected));
}

function frameEntry(index) {
  const picked = state.entries[index];
  if (!picked) return;
  for (const [key, ids] of [['a', picked.left], ['b', picked.right]]) {
    const side = sides[key];
    if (!ids.length || !side.model) continue;
    const faces = [];
    for (const id of ids) {
      const node = side.model.nodeById.get(id);
      if (node) faces.push(...node.cadFaceIds);
    }
    if (faces.length) side.viewer.frameFaces(faces);
  }
}

function step(delta) {
  if (!state.entries.length) return;
  const next = state.selected === null ? (delta > 0 ? 0 : state.entries.length - 1)
    : (state.selected + delta + state.entries.length) % state.entries.length;
  selectEntry(next);
}

/** Clicking a face in one view selects the edit that names its feature. */
function viewerCallbacks(side) {
  return {
    onPickFace(cadFaceId) {
      if (cadFaceId === null || cadFaceId === undefined) { selectEntry(null); return; }
      const row = side.model && side.model.faceById.get(cadFaceId);
      const featureId = row && row.featureIds[0];
      const which = side.key === 'a' ? 'left' : 'right';
      const index = featureId ? state.entries.findIndex((item) => item[which].includes(featureId)) : -1;
      if (index >= 0) { selectEntry(index); return; }
      state.selected = null;
      paintSelection();
      side.store.update({ selection: { kind: 'face', cadFaceId } });
      sides[side.key === 'a' ? 'b' : 'a'].store.update({ selection: null });
    },
    onHoverFace(cadFaceId, event) {
      side.store.update({ hover: cadFaceId === null || cadFaceId === undefined ? null : { kind: 'face', cadFaceId } });
      tooltip(side, cadFaceId, event);
    },
  };
}

function tooltip(side, cadFaceId, event) {
  const tip = $(`tip-${side.key}`);
  if (cadFaceId === null || cadFaceId === undefined || !event || !side.model) { tip.classList.remove('on'); return; }
  const row = side.model.faceById.get(cadFaceId);
  if (!row) { tip.classList.remove('on'); return; }
  tip.textContent = `${row.displayFaceId || '—'} · cad ${row.cadFaceId}\n${row.featureIds.length ? row.featureIds.join(', ') : 'unclaimed'}`;
  const rect = $(`viewport-${side.key}`).getBoundingClientRect();
  tip.style.left = `${event.clientX - rect.left}px`;
  tip.style.top = `${event.clientY - rect.top}px`;
  tip.classList.add('on');
}

// ---------------------------------------------------------------- chrome

function bindChrome() {
  $('swap').addEventListener('click', () => {
    const a = sides.a.partId;
    sides.a.partId = sides.b.partId;
    sides.b.partId = a;
    $('part-a').value = sides.a.partId;
    $('part-b').value = sides.b.partId;
    reload({ parts: ['a', 'b'] });
  });
  document.querySelector('.toolbar').addEventListener('click', (event) => {
    const button = event.target.closest('button[data-action]');
    if (!button) return;
    if (button.dataset.action === 'fit') fitBoth();
    if (button.dataset.action === 'link') setLink(!state.linkCameras);
  });
  // Both viewers report camera moves; with the link on, the other side
  // follows relative to its own part (same direction, same distance ratio).
  let applying = false;
  for (const [key, otherKey] of [['a', 'b'], ['b', 'a']]) {
    sides[key].viewer.onCameraChange(() => {
      if (!state.linkCameras || applying) return;
      const source = sides[key].viewer;
      const target = sides[otherKey].viewer;
      const from = source.bounds();
      const to = target.bounds();
      if (!from || !to || !from.radius || !to.radius) return;
      const goal = source.cameraGoal();
      const ratio = to.radius / from.radius;
      const position = goal.position.map((v, i) => to.center[i] + (v - from.center[i]) * ratio);
      const focus = goal.target.map((v, i) => to.center[i] + (v - from.center[i]) * ratio);
      applying = true;
      target.setCamera(position, focus);
      applying = false;
    });
  }
}

function setLink(on) {
  state.linkCameras = on;
  const button = document.querySelector('[data-action="link"]');
  if (button) button.setAttribute('aria-checked', String(on));
  if (on && sides.a.viewer && sides.b.viewer) {
    // Start aligned: b follows a once.
    const goal = sides.a.viewer.cameraGoal();
    const from = sides.a.viewer.bounds();
    const to = sides.b.viewer.bounds();
    if (from && to && from.radius && to.radius) {
      const ratio = to.radius / from.radius;
      sides.b.viewer.setCamera(
        goal.position.map((v, i) => to.center[i] + (v - from.center[i]) * ratio),
        goal.target.map((v, i) => to.center[i] + (v - from.center[i]) * ratio),
      );
    }
  }
}

function fitBoth() {
  if (state.selected !== null) { frameEntry(state.selected); return; }
  for (const side of Object.values(sides)) {
    const resolved = side.model ? resolveSelection(side.store.get(), side.model) : null;
    if (resolved && resolved.allFaceIds.size) side.viewer.frameFaces([...resolved.allFaceIds]);
    else side.viewer.fit();
  }
}

function bindKeys() {
  window.addEventListener('keydown', (event) => {
    const target = event.target;
    if (target instanceof Element && target.matches('input, select, textarea')) return;
    const key = event.key.toLowerCase();
    if (key === 'n' || key === 'j' || event.key === 'ArrowDown') { event.preventDefault(); step(1); }
    else if (key === 'p' || key === 'k' || event.key === 'ArrowUp') { event.preventDefault(); step(-1); }
    else if (key === 'f') fitBoth();
    else if (key === 'l') setLink(!state.linkCameras);
    else if (event.key === 'Escape') selectEntry(null);
    else if (event.key === 'Enter' && state.selected !== null) frameEntry(state.selected);
  });
}

// --------------------------------------------------------------- helpers

function switchButton(labelText, on, onClick) {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'switch compact';
  button.setAttribute('role', 'switch');
  button.setAttribute('aria-checked', String(on));
  button.innerHTML = `<span class="switch-label">${labelText}</span><span class="switch-track"><span class="switch-thumb"></span></span>`;
  button.addEventListener('click', onClick);
  return button;
}

function note(text) {
  const el = document.createElement('div');
  el.className = 'cmp-note';
  el.textContent = text;
  return el;
}

function label(key) {
  return key.replace(/_cost$/, '').replace(/^n_/, '').replace(/^edges_/, 'e. ').replace(/_/g, ' ');
}

function fmt(value, digits) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : '—';
}

function fmtCount(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return '—';
  return Number.isInteger(number) ? String(number) : number.toFixed(1);
}

function short(iso) {
  if (!iso) return '—';
  const match = /^\d{4}-(\d{2}-\d{2})T(\d{2}:\d{2})/.exec(iso);
  return match ? `${match[1]} ${match[2]}` : iso;
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (char) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]
  ));
}

boot().catch((error) => {
  $('loading').innerHTML = `<span class="spin"></span>${error.message}`;
  console.error(error);
});
