// Similarity view of the launcher page: a query part, a candidate set (all
// other parts or a named local group), the cost weights (presets plus eight
// adjustable scalars) and the ranking the server returns.
//
// Everything on screen is what GET /api/similarity/* answers.  The sliders are
// generated from the weight specs the server sends and the table columns from
// the keys of the first row's breakdown, so no weight key, channel name or
// feature name is written in this file.
//
// Since node 21' both kernel lines have a corpus, and `line` rides along on
// every request: the server holds one service per line, each with its own
// representation and its own cache.  Switching lines therefore invalidates
// the result on screen (the two scales are not comparable) rather than
// re-labelling it - the view clears the ranking and waits for a new Query.  Candidate groups and local presets
// live in localStorage only (same pattern as prefs.js); the server holds no
// state for this page.

import { DOTS, createMenu } from './menu.js';

const GROUPS_KEY = 'gatherSeedSimilarityGroups';    // { name: [partId, …] }
const PRESETS_KEY = 'gatherSeedSimilarityPresets';  // { name: { W_…: value } }
const STATE_KEY = 'gatherSeedSimilarityState';      // last query settings
const LOCAL_PREFIX = 'local:';

export function createSimilarity(root, options = {}) {
  const st = {
    parts: [],               // index parts (partId + featureCounts …)
    specs: null,             // /api/similarity/configs payload
    localPresets: readJson(PRESETS_KEY, {}),
    groups: readJson(GROUPS_KEY, {}),
    query: null,
    configName: null,        // server preset name or "local:<name>"
    weights: null,           // full weight dictionary, all keys
    advanced: false,
    top: 0,
    prefilter: 0,
    candidateMode: 'all',    // 'all' | 'selected'
    candidates: [],          // the ad-hoc candidate list (partIds)
    groupName: null,         // group the list was loaded from / saved to
    pickerOpen: false,
    pickerSearch: '',
    resolved: null,          // {name, hash, weights, preset} of the current weights
    result: null,            // last rank payload
    busy: false,
    error: null,
    note: null,
    pendingRun: false,
    built: false,
    openRow: null,           // ranking row whose breakdown is expanded
  };
  Object.assign(st, sanitizeState(readJson(STATE_KEY, {})));

  const el = {};
  let resolveTimer = null;
  let noteTimer = null;

  // ------------------------------------------------------------ public api

  function setParts(parts) {
    st.parts = parts || [];
    if (st.query && !st.parts.some((part) => part.partId === st.query)) st.query = null;
    if (!st.query && st.parts.length) st.query = st.parts[0].partId;
    if (st.built) {
      renderQueryOptions();
      renderCandidates();
    }
  }

  /**
   * Preselect the query part; `run` sends the query once the specs are in.
   * `extra` may carry a preset name, weight overrides (K=V,… or JSON text)
   * and the advanced switch — the same things a compare link carries.
   */
  function setQuery(partId, run = false, extra = {}) {
    if (partId) st.query = partId;
    st.pendingRun = Boolean(run);
    if (extra.configName) st.pendingConfig = extra.configName;
    if (extra.weights) st.pendingWeights = extra.weights;
    if (extra.advanced !== undefined) st.advanced = Boolean(extra.advanced);
    if (Array.isArray(extra.candidates) && extra.candidates.length) {
      st.candidates = [...new Set(extra.candidates)];
      st.candidateMode = 'selected';
    }
    if (st.built) {
      renderQueryOptions();
      renderCandidates();
      if (st.pendingRun && st.specs) runQuery();
    }
    persist();
  }

  async function render() {
    if (!st.built) build();
    if (!st.specs) await loadSpecs();
    renderAll();
    if (st.pendingRun && st.specs && !st.busy) runQuery();
  }

  // --------------------------------------------------------------- specs

  // The line rides on every request; the server validates it against its own
  // registry, this file never builds a path from it.
  function withLine(params) {
    const line = typeof options.line === 'function' ? options.line() : options.line;
    if (line) params.set('line', line);
    return params;
  }

  async function loadSpecs() {
    try {
      const response = await fetch(`api/similarity/configs?${withLine(new URLSearchParams())}`,
                                   { cache: 'no-store' });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      st.specs = await response.json();
    } catch (error) {
      st.error = `similarity API unavailable: ${error.message}`;
      st.specs = null;
      return;
    }
    const presets = st.specs.configs || [];
    const defaultName = st.specs.default || (presets[0] && presets[0].name);
    if (st.pendingConfig && presetByName(st.pendingConfig)) {
      st.configName = st.pendingConfig;
      st.weights = { ...presetByName(st.configName).weights };
    }
    st.pendingConfig = null;
    if (!st.configName || !presetByName(st.configName)) st.configName = defaultName;
    const base = presetByName(st.configName);
    if (!st.weights || !sameKeys(st.weights, base.weights)) st.weights = { ...base.weights };
    if (st.pendingWeights) {
      // A link carries the whole configuration: preset + these overrides,
      // not these overrides on top of whatever this browser had last.
      st.weights = { ...base.weights };
      for (const [key, value] of Object.entries(parseWeightText(st.pendingWeights))) {
        if (key in st.weights) st.weights[key] = value;
      }
      st.pendingWeights = null;
    }
    await resolveNow();
  }

  function serverPresets() {
    return (st.specs && st.specs.configs) || [];
  }

  function presetByName(name) {
    if (!name) return null;
    if (name.startsWith(LOCAL_PREFIX)) {
      const weights = st.localPresets[name.slice(LOCAL_PREFIX.length)];
      return weights ? { name, weights, local: true } : null;
    }
    return serverPresets().find((preset) => preset.name === name) || null;
  }

  /** The server preset the request is based on: the chosen one, or the default. */
  function basePreset() {
    const chosen = presetByName(st.configName);
    if (chosen && !chosen.local) return chosen;
    return presetByName(st.specs && st.specs.default) || serverPresets()[0] || null;
  }

  function specKeys() {
    return Object.keys((st.specs && st.specs.weights) || {});
  }

  function defaults() {
    const base = basePreset();
    return base ? base.weights : {};
  }

  /** Overrides relative to the base preset, as the CLI / API take them. */
  function overrides() {
    const base = defaults();
    const out = {};
    for (const key of specKeys()) {
      if (st.weights && base[key] !== undefined && Number(st.weights[key]) !== Number(base[key])) out[key] = Number(st.weights[key]);
    }
    return out;
  }

  function weightsParam() {
    const diff = overrides();
    const keys = Object.keys(diff);
    return keys.length ? keys.map((key) => `${key}=${trim(diff[key])}`).join(',') : '';
  }

  // Ask the server what this configuration is called and hashes to.  Cheap:
  // the route only resolves the preset and the overrides, no pair is solved.
  async function resolveNow() {
    const base = basePreset();
    if (!base) return;
    const params = new URLSearchParams({ config: base.name });
    const diff = weightsParam();
    if (diff) params.set('weights', diff);
    try {
      const response = await fetch(`api/similarity/configs?${withLine(params)}`, { cache: 'no-store' });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
      st.resolved = payload.resolved || base;
    } catch (error) {
      st.resolved = null;
      st.error = error.message;
    }
    if (st.built) renderHashChips();
  }

  function scheduleResolve() {
    clearTimeout(resolveTimer);
    resolveTimer = setTimeout(resolveNow, 180);
  }

  // --------------------------------------------------------------- build

  function build() {
    st.built = true;
    root.innerHTML = '';
    root.classList.add('sim-body');

    const grid = div('sim-grid');
    el.queryPanel = panel('Query');
    el.weightsPanel = panel('Cost weights');
    grid.append(el.queryPanel.panel, el.weightsPanel.panel);
    el.resultsPanel = panel('Ranking');
    el.resultsPanel.panel.classList.add('sim-results');
    root.append(grid, el.resultsPanel.panel);

    buildQueryPanel();
    buildWeightsPanel();
    buildResultsPanel();
  }

  function panel(title) {
    const box = div('stats-panel sim-panel');
    const head = div('stats-panel-head');
    const h2 = document.createElement('h2');
    h2.textContent = title;
    head.append(h2);
    const body = div('sim-panel-body');
    box.append(head, body);
    return { panel: box, head, body };
  }

  // -- query panel ----------------------------------------------------------

  function buildQueryPanel() {
    const { body } = el.queryPanel;

    const queryField = field('Query part');
    el.query = document.createElement('select');
    el.query.className = 'sim-select';
    el.query.addEventListener('change', () => { st.query = el.query.value; persist(); renderCandidates(); });
    queryField.append(el.query);

    // Candidates: every other part, or a list the reviewer puts together —
    // parts added one by one from the library (+), or a named group loaded
    // into the list.  Groups live in this browser only.
    const candField = field('Candidates');
    const modes = div('sim-cand-modes');
    el.modeAll = radio('sim-cand-mode', 'all', 'all other parts');
    el.modeSel = radio('sim-cand-mode', 'selected', 'selected parts');
    modes.append(el.modeAll.wrap, el.modeSel.wrap);
    el.modeAll.input.addEventListener('change', () => { st.candidateMode = 'all'; persist(); renderCandidates(); });
    el.modeSel.input.addEventListener('change', () => { st.candidateMode = 'selected'; persist(); renderCandidates(); });

    el.candBox = div('sim-cand-box');
    el.candList = div('sim-cand-list');
    el.addBtn = document.createElement('button');
    el.addBtn.type = 'button';
    el.addBtn.className = 'type-chip sim-cand-add';
    el.addBtn.title = 'add parts from the library';
    el.addBtn.innerHTML = '<span class="plus">+</span>Add parts';
    el.addBtn.addEventListener('click', () => { st.pickerOpen = !st.pickerOpen; renderCandidates(); });
    el.picker = div('sim-picker');
    el.picker.hidden = true;
    const groupRow = div('sim-group-row');
    el.group = document.createElement('select');
    el.group.className = 'sim-select sim-group-select';
    el.group.title = 'load a saved group into the list';
    el.group.addEventListener('change', () => loadGroup(el.group.value || null));
    el.groupSave = smallButton('Save', 'overwrite the selected group with the current list', saveGroup);
    el.groupSaveAs = smallButton('Save as…', 'keep the current list under a new name (this browser only)', saveGroupAs);
    el.groupRename = smallButton('Rename', 'rename the selected group', renameGroup);
    el.groupDelete = smallButton('Delete', 'delete the selected group', deleteGroup);
    groupRow.append(span('sim-field-label', 'group'), el.group, el.groupSave, el.groupSaveAs, el.groupRename, el.groupDelete);
    el.candCount = span('sim-cand-count');
    el.candBox.append(el.candList, el.picker, groupRow, el.candCount);
    candField.append(modes, el.candBox);

    const limits = div('sim-limits');
    el.top = numberInput('top', 'show only the best N (0 = all)');
    el.prefilter = numberInput('prefilter keep', 'solve the GED only for the N best pre-filter candidates (0 = all)');
    el.top.input.addEventListener('change', () => { st.top = clampInt(el.top.input.value); el.top.input.value = st.top; persist(); });
    el.prefilter.input.addEventListener('change', () => { st.prefilter = clampInt(el.prefilter.input.value); el.prefilter.input.value = st.prefilter; persist(); });
    el.run = document.createElement('button');
    el.run.type = 'button';
    el.run.className = 'btn-primary sim-run';
    el.run.textContent = 'Query';
    el.run.addEventListener('click', runQuery);
    limits.append(el.top.wrap, el.prefilter.wrap, spacer(), el.run);

    body.append(queryField, candField, limits);
  }

  function renderQueryOptions() {
    if (!el.query) return;
    el.query.innerHTML = '';
    for (const part of st.parts) {
      const option = document.createElement('option');
      option.value = part.partId;
      option.textContent = part.partId;
      el.query.append(option);
    }
    if (st.query) el.query.value = st.query;
    if (!el.query.value && st.parts.length) { st.query = st.parts[0].partId; el.query.value = st.query; }
  }

  function renderCandidates() {
    if (!el.candList) return;
    const known = new Set(st.parts.map((part) => part.partId));
    st.candidates = (st.candidates || []).filter((id) => known.has(id));
    const names = Object.keys(st.groups).sort();
    if (st.groupName && !names.includes(st.groupName)) st.groupName = null;
    const selected = st.candidateMode === 'selected';

    el.modeAll.input.checked = !selected;
    el.modeSel.input.checked = selected;
    el.modeAll.label.textContent = `all other parts (${Math.max(0, st.parts.length - 1)})`;
    el.modeSel.label.textContent = `selected parts (${st.candidates.filter((id) => id !== st.query).length})`;
    el.candBox.hidden = !selected;
    if (!selected) return;

    // The list: one chip per chosen part, the + at the end.
    el.candList.innerHTML = '';
    for (const id of st.candidates) {
      const chip = document.createElement('span');
      chip.className = 'type-chip on sim-cand' + (id === st.query ? ' is-query' : '');
      chip.innerHTML = '<span class="swatch"></span>';
      chip.append(document.createTextNode(id));
      if (id === st.query) chip.title = 'the query itself is never a candidate';
      const remove = document.createElement('button');
      remove.type = 'button';
      remove.className = 'sim-cand-remove';
      remove.title = 'remove from the list';
      remove.textContent = '×';
      remove.addEventListener('click', () => { st.candidates = st.candidates.filter((other) => other !== id); persist(); renderCandidates(); });
      chip.append(remove);
      el.candList.append(chip);
    }
    if (!st.candidates.length) el.candList.append(span('sim-cand-empty', 'no part chosen yet'));
    el.candList.append(el.addBtn);
    el.addBtn.classList.toggle('open', st.pickerOpen);

    // The picker: the library with a check per part and a search box.
    el.picker.hidden = !st.pickerOpen;
    el.picker.innerHTML = '';
    if (st.pickerOpen) {
      const head = div('sim-picker-head');
      const search = document.createElement('input');
      search.type = 'search';
      search.placeholder = 'filter parts…';
      search.value = st.pickerSearch;
      search.addEventListener('input', () => { st.pickerSearch = search.value; renderPickerRows(rows); });
      const all = smallButton('Add all shown', 'add every listed part', () => {
        const chosen = new Set(st.candidates);
        for (const part of pickerParts()) chosen.add(part.partId);
        st.candidates = [...chosen];
        persist();
        renderCandidates();
      });
      const none = smallButton('Clear list', 'remove every part from the list', () => { st.candidates = []; persist(); renderCandidates(); });
      const done = smallButton('Done', 'close the picker', () => { st.pickerOpen = false; renderCandidates(); });
      head.append(search, spacer(), all, none, done);
      const rows = div('sim-picker-rows');
      el.picker.append(head, rows);
      renderPickerRows(rows);
      search.focus();
    }

    // Groups: load into the list, save the list, rename / delete.
    el.group.innerHTML = '';
    const blank = document.createElement('option');
    blank.value = '';
    blank.textContent = names.length ? '— load a group —' : '— no saved group —';
    el.group.append(blank);
    for (const name of names) {
      const option = document.createElement('option');
      option.value = name;
      option.textContent = `${name} (${(st.groups[name] || []).length})`;
      el.group.append(option);
    }
    el.group.value = st.groupName || '';
    el.groupSave.disabled = !st.groupName;
    el.groupRename.disabled = !st.groupName;
    el.groupDelete.disabled = !st.groupName;
    el.groupSaveAs.disabled = !st.candidates.length;
    const effective = st.candidates.filter((id) => id !== st.query).length;
    const dirty = st.groupName && !sameList(st.groups[st.groupName] || [], st.candidates);
    el.candCount.textContent = `${effective} candidate(s)`
      + (st.groupName ? ` · group ${st.groupName}${dirty ? ' (list changed — Save to keep)' : ''}` : '');
  }

  function pickerParts() {
    const needle = st.pickerSearch.trim().toLowerCase();
    return st.parts.filter((part) => part.partId !== st.query
      && (!needle || part.partId.toLowerCase().includes(needle)
        || Object.keys(part.featureCounts || {}).some((type) => type.toLowerCase().includes(needle))));
  }

  function renderPickerRows(rows) {
    rows.innerHTML = '';
    const chosen = new Set(st.candidates);
    const parts = pickerParts();
    if (!parts.length) rows.append(span('sim-cand-empty', 'no part matches'));
    for (const part of parts) {
      const row = document.createElement('label');
      row.className = 'sim-picker-row' + (chosen.has(part.partId) ? ' on' : '');
      const box = document.createElement('input');
      box.type = 'checkbox';
      box.checked = chosen.has(part.partId);
      box.addEventListener('change', () => {
        const next = new Set(st.candidates);
        if (box.checked) next.add(part.partId); else next.delete(part.partId);
        st.candidates = [...next];
        persist();
        renderCandidates();
      });
      const name = span('sim-picker-name', part.partId);
      const meta = span('sim-picker-meta', `${part.featureNodeCount ?? '—'} features · ${part.relationEdgeCount ?? '—'} relations`);
      row.append(box, name, meta);
      rows.append(row);
    }
  }

  function loadGroup(name) {
    st.groupName = name || null;
    if (name && st.groups[name]) st.candidates = [...st.groups[name]];
    persist();
    renderCandidates();
  }

  function saveGroup() {
    if (!st.groupName) return;
    st.groups[st.groupName] = [...st.candidates];
    writeJson(GROUPS_KEY, st.groups);
    renderCandidates();
    say(`group "${st.groupName}" saved`);
  }

  function saveGroupAs() {
    const name = window.prompt('Name for this candidate group:', st.groupName || '');
    if (!name || !name.trim()) return;
    const clean = name.trim();
    st.groups[clean] = [...st.candidates];
    st.groupName = clean;
    writeJson(GROUPS_KEY, st.groups);
    persist();
    renderCandidates();
    say(`group "${clean}" saved`);
  }

  function renameGroup() {
    if (!st.groupName) return;
    const name = window.prompt(`Rename group "${st.groupName}" to:`, st.groupName);
    if (!name || !name.trim() || name.trim() === st.groupName) return;
    const clean = name.trim();
    st.groups[clean] = st.groups[st.groupName] || [];
    delete st.groups[st.groupName];
    st.groupName = clean;
    writeJson(GROUPS_KEY, st.groups);
    persist();
    renderCandidates();
  }

  function deleteGroup() {
    if (!st.groupName) return;
    if (!window.confirm(`Delete group "${st.groupName}"? (only this browser's list, no part is touched)`)) return;
    delete st.groups[st.groupName];
    st.groupName = null;
    writeJson(GROUPS_KEY, st.groups);
    persist();
    renderCandidates();
  }

  function candidateIds() {
    if (st.candidateMode !== 'selected') return null;
    return (st.candidates || []).filter((id) => id !== st.query);
  }

  // -- weights panel --------------------------------------------------------

  function buildWeightsPanel() {
    const { head, body } = el.weightsPanel;
    el.preset = document.createElement('select');
    el.preset.className = 'sim-select sim-preset';
    el.preset.title = 'cost preset; the sliders start from its values';
    el.preset.addEventListener('change', () => selectPreset(el.preset.value));
    el.hashChip = span('sim-hash', '');
    el.hashChip.title = 'configuration = the eight values below; the hash is sha256 over the values (never the name)';
    const info = span('info-dot', 'i');
    info.title = 'Distances computed under different weights are not comparable with each other; '
      + 'the hash names the configuration a number belongs to.';
    head.append(el.preset, el.hashChip, info, spacer());
    el.advanced = toggle('Advanced', 'show all eight weights (simple mode: the four main channels)', () => {
      st.advanced = !st.advanced;
      persist();
      renderSliders();
    });
    head.append(el.advanced);

    el.sliders = div('sim-sliders');
    const foot = div('sim-weights-foot');
    el.resetBtn = smallButton('Reset to default', 'back to the values of the selected preset', resetWeights);
    const dots = span('');
    dots.innerHTML = DOTS;
    const more = createMenu(dots, [
      { label: 'Save as local preset…', title: 'keep these values under a name in this browser', onSelect: saveLocalPreset },
      { label: 'Copy JSON', title: 'the eight values as a JSON object', onSelect: () => copyText(JSON.stringify(fullWeights(), null, 2), 'weights JSON copied') },
      { label: 'Copy CLI command', title: 'the ged rank command that reproduces this query', onSelect: () => copyText(cliCommand(), 'CLI command copied') },
    ], { summaryClass: 'icon-btn', title: 'more actions' });
    foot.append(el.resetBtn, more.element);
    body.append(el.sliders, foot);
  }

  function renderPresetOptions() {
    el.preset.innerHTML = '';
    const server = document.createElement('optgroup');
    server.label = 'presets';
    for (const preset of serverPresets()) {
      const option = document.createElement('option');
      option.value = preset.name;
      option.textContent = preset.name;
      server.append(option);
    }
    el.preset.append(server);
    const names = Object.keys(st.localPresets).sort();
    if (names.length) {
      const local = document.createElement('optgroup');
      local.label = 'local presets (this browser)';
      for (const name of names) {
        const option = document.createElement('option');
        option.value = LOCAL_PREFIX + name;
        option.textContent = name;
        local.append(option);
      }
      el.preset.append(local);
    }
    if (st.configName) el.preset.value = st.configName;
  }

  function selectPreset(name) {
    const preset = presetByName(name);
    if (!preset) return;
    st.configName = name;
    st.weights = { ...defaults(), ...preset.weights };
    persist();
    renderSliders();
    resolveNow();
  }

  function resetWeights() {
    st.weights = { ...defaults() };
    if (presetByName(st.configName)?.local) st.configName = basePreset().name;
    persist();
    renderPresetOptions();
    renderSliders();
    resolveNow();
  }

  function saveLocalPreset() {
    const name = window.prompt('Name for this local preset:');
    if (!name || !name.trim()) return;
    const clean = name.trim();
    st.localPresets[clean] = fullWeights();
    writeJson(PRESETS_KEY, st.localPresets);
    st.configName = LOCAL_PREFIX + clean;
    persist();
    renderPresetOptions();
    renderHashChips();
    say(`saved local preset "${clean}"`);
  }

  function fullWeights() {
    const out = {};
    for (const key of specKeys()) out[key] = Number(st.weights?.[key] ?? defaults()[key]);
    return out;
  }

  /** Sliders from the spec table: simple = the keys flagged simple, else all. */
  function renderSliders() {
    el.advanced.setAttribute('aria-checked', String(st.advanced));
    el.sliders.classList.toggle('advanced', st.advanced);
    el.sliders.innerHTML = '';
    const specs = (st.specs && st.specs.weights) || {};
    const base = defaults();
    for (const key of Object.keys(specs)) {
      const spec = specs[key];
      if (!st.advanced && !spec.simple) continue;
      el.sliders.append(slider(key, spec, Number(st.weights?.[key] ?? base[key]), Number(base[key])));
    }
    // Weights changed in advanced mode stay in force in simple mode; say so
    // only when it is the case, in the Advanced switch's tooltip.
    const hiddenChanged = Object.keys(overrides()).filter((key) => specs[key] && !specs[key].simple);
    el.advanced.title = 'show all weights (simple mode: the four main channels)'
      + (hiddenChanged.length ? ` · changed here: ${hiddenChanged.join(', ')}` : '');
    el.advanced.classList.toggle('has-changes', !st.advanced && hiddenChanged.length > 0);
  }

  function slider(key, spec, value, fallback) {
    const wrap = div('wslider');
    wrap.dataset.key = key;
    const min = Number(spec.min);
    const max = Number(spec.max);
    const step = Number(spec.step) || 0.01;
    const decimals = Math.max(0, Math.min(4, Math.ceil(-Math.log10(step))));
    const fmt = (v) => Number(v).toFixed(decimals);
    const pct = (v) => `${Math.max(0, Math.min(100, ((v - min) / (max - min)) * 100))}%`;

    const head = div('wslider-head');
    const label = span('wslider-label', spec.label || key);
    label.title = [spec.hint, `${key} · ${trim(min)} – ${trim(max)}`].filter(Boolean).join('\n');
    const keyTag = span('wslider-key', key);
    const valueInput = document.createElement('input');
    valueInput.type = 'number';
    valueInput.className = 'wslider-value';
    valueInput.min = String(min);
    valueInput.max = String(max);
    valueInput.step = String(step);
    valueInput.value = fmt(value);
    valueInput.title = 'type an exact value';
    head.append(label, keyTag, spacer(), valueInput);

    const track = div('wslider-track');
    const fill = span('wslider-fill');
    const mark = span('wslider-default');
    mark.style.left = pct(fallback);
    mark.title = `preset value ${fmt(fallback)}`;
    const range = document.createElement('input');
    range.type = 'range';
    range.className = 'wslider-range';
    range.min = String(min);
    range.max = String(max);
    range.step = String(step);
    range.value = String(value);
    range.setAttribute('aria-label', `${spec.label || key} (${key})`);
    track.append(fill, mark, range);

    const foot = div('wslider-foot');
    foot.append(span('wslider-end', trim(min)), span('wslider-hint', spec.hint || ''), span('wslider-end', trim(max)));
    wrap.append(head, track, foot);

    function paint(v) {
      wrap.style.setProperty('--pct', pct(v));
      wrap.classList.toggle('modified', Math.abs(v - fallback) > 1e-12);
      mark.classList.toggle('at-thumb', Math.abs(v - fallback) < (max - min) * 0.012);
    }
    function commit(v, fromRange) {
      let next = Number(v);
      if (!Number.isFinite(next)) next = fallback;
      next = Math.min(max, Math.max(min, next));
      st.weights[key] = Number(next.toFixed(6));
      if (!fromRange) range.value = String(next);
      valueInput.value = fmt(next);
      paint(next);
      persist();
      scheduleResolve();
    }
    range.addEventListener('input', () => commit(range.value, true));
    valueInput.addEventListener('change', () => commit(valueInput.value, false));
    valueInput.addEventListener('keydown', (event) => { if (event.key === 'Enter') valueInput.blur(); });
    // Double-click the thumb area: back to the preset value for this key.
    track.addEventListener('dblclick', () => commit(fallback, false));
    paint(value);
    return wrap;
  }

  function renderHashChips() {
    if (!el.hashChip) return;
    const resolved = st.resolved;
    const local = presetByName(st.configName)?.local && sameValues(presetByName(st.configName).weights, fullWeights())
      ? st.configName.slice(LOCAL_PREFIX.length) : null;
    if (!resolved) {
      el.hashChip.textContent = '#—';
      el.hashChip.dataset.kind = 'unknown';
    } else {
      const name = local ? `${local} (local)` : resolved.name;
      el.hashChip.textContent = `${name} · #${resolved.hash}`;
      el.hashChip.dataset.kind = resolved.preset ? 'preset' : 'custom';
    }
    if (el.resultsMeta) renderResults();
  }

  // -- results panel --------------------------------------------------------

  function buildResultsPanel() {
    const { head, body } = el.resultsPanel;
    el.resultsMeta = span('sim-meta', '');
    el.legend = span('score-legend sim-legend', '');
    head.append(el.resultsMeta, spacer(), el.legend);
    el.tableWrap = div('sim-table-wrap');
    el.empty = div('sim-empty');
    body.append(el.tableWrap, el.empty);
  }

  function renderResults() {
    const result = st.result;
    el.tableWrap.innerHTML = '';
    if (st.busy) {
      el.resultsMeta.textContent = 'querying…';
      el.empty.textContent = '';
      el.empty.hidden = true;
      return;
    }
    if (!result) {
      el.resultsMeta.textContent = '';
      el.empty.hidden = false;
      el.empty.textContent = st.error ? st.error : 'No query yet — pick a query part and press Query.';
      el.empty.classList.toggle('error', Boolean(st.error));
      return;
    }
    const cfg = result.config || {};
    const corpus = result.corpus || {};
    const failed = Object.keys(corpus.failed || {});
    const cached = (result.ranked || []).filter((row) => row.cached).length;
    el.resultsMeta.textContent =
      `${result.query} · ${cfg.name} · ${result.candidates} candidate(s)`
      + ` (${corpus.loaded} loaded, ${failed.length} failed) · ${Math.round(result.timingMs)} ms`
      + ` · ${cached}/${(result.ranked || []).length} cached`;
    el.resultsMeta.title = [`configuration ${cfg.name} #${cfg.hash}`]
      .concat(failed.map((id) => `${id}: ${corpus.failed[id]}`)).join('\n');
    const stale = st.resolved && st.resolved.hash !== cfg.hash;
    el.resultsMeta.classList.toggle('stale', Boolean(stale));
    if (stale) el.resultsMeta.title = 'the weights changed since this ranking was computed — press Query to recompute';

    const rows = result.ranked || [];
    el.legend.innerHTML = '';
    if (!rows.length) {
      el.empty.hidden = false;
      el.empty.classList.remove('error');
      el.empty.textContent = failed.length
        ? `no candidate could be ranked (${failed.length} failed: ${failed.join(', ')})`
        : 'no candidates — the corpus has only the query part';
      return;
    }
    el.empty.hidden = true;
    const costKeys = Object.keys(rows[0].breakdown || {}).filter((key) => /_cost$/.test(key));
    costKeys.forEach((key, position) => {
      const tag = span(`tag c${position % 6}`);
      tag.innerHTML = `<i></i>${channelLabel(key)}`;
      tag.title = key;
      el.legend.append(tag);
    });
    el.tableWrap.append(table(rows, cfg, costKeys));
  }

  /**
   * The ranking: rank, part, similarity (the number to read) with the
   * distance beside it, one stacked bar of the cost channels, a cache dot
   * and the Compare action.  Clicking a row opens its full breakdown — every
   * key the server sent, rendered by name, no channel known here.
   */
  function table(rows, cfg, costKeys) {
    const tbl = document.createElement('table');
    tbl.className = 'h-table sim-table';
    const thead = document.createElement('thead');
    const hr = document.createElement('tr');
    for (const [text, cls, title] of [
      ['#', 'num', 'rank'], ['part', '', 'candidate part'],
      ['similarity', 'num', '100 · (1 − distance)'], ['distance', 'num', 'normalised GED in [0, 1]'],
      ['channels', 'chan', 'share of the raw cost per channel: ' + costKeys.map(channelLabel).join(', ')],
      ['', 'num', 'served from the pair cache'], ['', 'num', ''],
    ]) {
      const th = document.createElement('th');
      th.textContent = text;
      th.className = cls;
      th.title = title;
      th.style.cursor = 'default';
      hr.append(th);
    }
    thead.append(hr);
    // The bars are scaled against the heaviest candidate on screen, so lengths
    // stay comparable down the column.
    const heaviest = rows.reduce((most, row) => Math.max(
      most,
      costKeys.reduce((sum, key) => sum + Number((row.breakdown || {})[key] || 0), 0),
    ), 0);
    const tbody = document.createElement('tbody');
    for (const row of rows) {
      const tr = document.createElement('tr');
      if (row.rank === 1) tr.classList.add('nearest');
      const open = st.openRow === row.partId;
      if (open) tr.classList.add('open');
      tr.title = row.rank === 1 ? 'nearest neighbour' : '';
      const compareHref = compareLink(row.partId, cfg);
      const cachedCell = td(row.cached ? '●' : '○', 'num cached' + (row.cached ? ' yes' : ''));
      cachedCell.title = row.cached ? 'served from the pair cache' : 'computed for this query';
      tr.append(
        td(String(row.rank), 'num rank'),
        partCell(row.partId),
        td(`${Number(row.similarity).toFixed(2)} %`, 'num sim'),
        td(Number(row.distance).toFixed(4), 'num dist'),
        channelCell(row.breakdown || {}, costKeys, heaviest),
        cachedCell,
        linkCell('Compare', compareHref),
      );
      tr.addEventListener('click', (event) => {
        if (event.target.closest('a, button')) return;
        st.openRow = open ? null : row.partId;
        renderResults();
      });
      tbody.append(tr);
      if (open) tbody.append(detailRow(row, hr.children.length));
    }
    tbl.append(thead, tbody);
    return tbl;
  }

  /**
   * The channel split for one candidate.  The bar's *length* is its total cost
   * against the heaviest row on screen, so a cheap match reads as a short bar
   * and the column says something the ranking beside it does not already; the
   * segments inside it keep their proportions.  Channels are taken in the order
   * the server sent them and coloured by position - never by name.
   */
  function channelCell(breakdown, costKeys, heaviest) {
    const cell = document.createElement('td');
    cell.className = 'chan';
    const total = costKeys.reduce((sum, key) => sum + Number(breakdown[key] || 0), 0);
    const track = div('score-bar sim-chan');
    const bar = div('sim-chan-fill');
    bar.style.width = heaviest > 0 ? `${Math.max((total / heaviest) * 100, 2)}%` : '100%';
    costKeys.forEach((key, position) => {
      const piece = span(`piece c${position % 6}`);
      piece.style.width = total > 0 ? `${(Number(breakdown[key] || 0) / total) * 100}%` : '0%';
      piece.title = `${channelLabel(key)} ${channelValue(key, breakdown[key])}`;
      bar.append(piece);
    });
    track.append(bar);
    cell.title = costKeys.map((key) => `${channelLabel(key)} ${channelValue(key, breakdown[key])}`).join(' · ')
      + `
total ${total.toFixed(4)}`;
    cell.append(track);
    return cell;
  }

  /** Everything the server said about one candidate, as label / value pairs. */
  function detailRow(row, columns) {
    const tr = document.createElement('tr');
    tr.className = 'sim-detail';
    const cell = document.createElement('td');
    cell.colSpan = columns;
    const grid = div('sim-detail-grid');
    const add = (label, value, title) => {
      const item = span('sim-detail-item');
      const k = span('k', label);
      if (title) k.title = title;
      item.append(k, span('v', value));
      grid.append(item);
    };
    for (const [key, value] of Object.entries(row.breakdown || {})) add(channelLabel(key), channelValue(key, value), key);
    add('raw cost', trim(row.rawCost), 'mean of both directions before normalisation');
    add('upper bound', trim(row.upperBound), 'normaliser of the distance');
    add('prefilter', Number(row.prefilterScore).toFixed(3), 'leaf-histogram L1 + L_ref ratio term (cheap pre-filter score)');
    add('built', row.producedAt || '—', 'producedAt of the candidate');
    cell.append(grid);
    tr.append(cell);
    return tr;
  }

  function compareLink(partId, cfg) {
    const params = withLine(new URLSearchParams({ a: st.result.query, b: partId }));
    const base = basePreset();
    if (base) params.set('config', base.name);
    const diff = weightsParamFor(cfg);
    if (diff) params.set('weights', diff);
    return `compare.html?${params}`;
  }

  /** Overrides of the configuration a result was computed with (not the sliders). */
  function weightsParamFor(cfg) {
    const base = basePreset();
    if (!base || !cfg || !cfg.weights) return '';
    const keys = Object.keys(cfg.weights).filter((key) => Number(cfg.weights[key]) !== Number(base.weights[key]));
    return keys.map((key) => `${key}=${trim(cfg.weights[key])}`).join(',');
  }

  function partCell(partId) {
    const cell = document.createElement('td');
    cell.className = 'part';
    const link = document.createElement('a');
    link.href = `viewer.html?part=${encodeURIComponent(partId)}`;
    link.textContent = partId;
    link.title = 'open the part page';
    cell.append(link);
    return cell;
  }

  function linkCell(text, href) {
    const cell = document.createElement('td');
    cell.className = 'num';
    const link = document.createElement('a');
    link.href = href;
    link.className = 'btn-text';
    link.textContent = text;
    cell.append(link);
    return cell;
  }

  // ----------------------------------------------------------------- query

  async function runQuery() {
    st.pendingRun = false;
    if (!st.query || !st.specs || st.busy) return;
    const base = basePreset();
    const params = new URLSearchParams({ query: st.query });
    if (base) params.set('config', base.name);
    const diff = weightsParam();
    if (diff) params.set('weights', diff);
    const candidates = candidateIds();
    if (candidates !== null) params.set('candidates', candidates.join(','));
    if (st.top) params.set('top', String(st.top));
    if (st.prefilter) params.set('prefilter', String(st.prefilter));

    st.busy = true;
    st.error = null;
    el.run.disabled = true;
    el.run.textContent = 'Querying…';
    renderResults();
    try {
      if (candidates !== null && candidates.length === 0) throw new Error('no candidate chosen — add parts with + or switch to "all other parts"');
      const response = await fetch(`api/similarity/rank?${withLine(params)}`, { cache: 'no-store' });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
      st.result = payload;
      if (payload.config) st.resolved = payload.config;
    } catch (error) {
      st.error = error.message;
      st.result = null;
    } finally {
      st.busy = false;
      el.run.disabled = false;
      el.run.textContent = 'Query';
    }
    renderHashChips();
    renderResults();
  }

  function cliCommand() {
    const base = basePreset();
    const parts = ['python run_pipeline.py ged rank', '--query', st.query || '<partId>'];
    if (base) parts.push('--config', base.name);
    const diff = weightsParam();
    if (diff) parts.push('--weights', diff);
    if (st.top) parts.push('--top', String(st.top));
    if (st.prefilter) parts.push('--prefilter-keep', String(st.prefilter));
    const candidates = candidateIds();
    if (candidates && candidates.length) parts.push('--candidates', ...candidates);
    parts.push('--json');
    return parts.join(' ');
  }

  // ------------------------------------------------------------- render all

  function renderAll() {
    renderQueryOptions();
    renderCandidates();
    el.top.input.value = st.top;
    el.prefilter.input.value = st.prefilter;
    if (st.specs) {
      renderPresetOptions();
      renderSliders();
    } else {
      el.sliders.innerHTML = '';
      el.sliders.append(Object.assign(div('sim-empty error'), { textContent: st.error || 'similarity API unavailable' }));
    }
    renderHashChips();
    renderResults();
  }

  // --------------------------------------------------------------- helpers

  function persist() {
    writeJson(STATE_KEY, {
      query: st.query,
      configName: st.configName,
      weights: st.weights,
      advanced: st.advanced,
      top: st.top,
      prefilter: st.prefilter,
      candidateMode: st.candidateMode,
      candidates: st.candidates,
      groupName: st.groupName,
    });
  }

  function say(text) {
    st.note = text;
    clearTimeout(noteTimer);
    const note = root.querySelector('.sim-toast') || span('sim-toast', '');
    note.textContent = text;
    el.weightsPanel.head.append(note);
    note.classList.add('on');
    noteTimer = setTimeout(() => note.classList.remove('on'), 1800);
  }

  async function copyText(text, message) {
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) await navigator.clipboard.writeText(text);
      else legacyCopy(text);
      say(message);
    } catch (error) {
      window.prompt('Copy:', text);
    }
  }

  /** The line changed: the ranking on screen was computed on another
      scale, so it is dropped rather than relabelled, and the presets are
      re-read because the server answers per line. */
  async function setLine() {
    st.result = null;
    st.error = null;
    st.note = null;
    await loadSpecs();
    await resolveNow();
    if (st.built) render();
  }

  return { render, setParts, setQuery, runQuery, setLine, state: st };
}

// ------------------------------------------------------------------ helpers

function channelLabel(key) {
  // A display transform of the key, not a lookup: "node_cost" -> "node",
  // "n_type_change" -> "type change", "edges_del" -> "e. del".
  return key.replace(/_cost$/, '').replace(/^n_/, '').replace(/^edges_/, 'e. ').replace(/_/g, ' ');
}

function channelValue(key, value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return '—';
  if (Number.isInteger(number)) return String(number);
  return /_cost$/.test(key) ? number.toFixed(4) : number.toFixed(1);
}

function trim(value) {
  const number = Number(value);
  return Number.isFinite(number) ? String(Number(number.toFixed(6))) : String(value);
}

/** "W_REL=0.8,C_DEL=1" or a JSON object text -> {KEY: number}; unknown text -> {}. */
function parseWeightText(text) {
  const out = {};
  if (!text) return out;
  const raw = String(text).trim();
  try {
    if (raw.startsWith('{')) {
      const parsed = JSON.parse(raw);
      for (const [key, value] of Object.entries(parsed || {})) out[String(key).toUpperCase()] = Number(value);
      return out;
    }
  } catch (error) {
    return out;
  }
  for (const item of raw.split(/[,;]/)) {
    const [key, value] = item.split('=');
    if (key && value !== undefined && Number.isFinite(Number(value))) out[key.trim().toUpperCase()] = Number(value);
  }
  return out;
}

function clampInt(value) {
  const number = Math.floor(Number(value));
  return Number.isFinite(number) && number > 0 ? number : 0;
}

function sameKeys(a, b) {
  const ka = Object.keys(a || {}).sort();
  const kb = Object.keys(b || {}).sort();
  return ka.length === kb.length && ka.every((key, index) => key === kb[index]);
}

function sameList(a, b) {
  const left = [...a].sort();
  const right = [...b].sort();
  return left.length === right.length && left.every((id, index) => id === right[index]);
}

function sameValues(a, b) {
  if (!sameKeys(a, b)) return false;
  return Object.keys(a).every((key) => Number(a[key]) === Number(b[key]));
}

function sanitizeState(saved) {
  const out = {};
  if (typeof saved.query === 'string') out.query = saved.query;
  if (typeof saved.configName === 'string') out.configName = saved.configName;
  if (saved.weights && typeof saved.weights === 'object') out.weights = { ...saved.weights };
  out.advanced = Boolean(saved.advanced);
  out.top = clampInt(saved.top);
  out.prefilter = clampInt(saved.prefilter);
  out.candidateMode = saved.candidateMode === 'selected' ? 'selected' : 'all';
  if (Array.isArray(saved.candidates)) out.candidates = saved.candidates.filter((id) => typeof id === 'string');
  if (typeof saved.groupName === 'string') out.groupName = saved.groupName;
  return out;
}

function readJson(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    const value = raw ? JSON.parse(raw) : null;
    return value && typeof value === 'object' ? value : fallback;
  } catch (error) {
    return fallback;
  }
}

function writeJson(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch (error) {
    // A blocked storage must not break the page.
  }
}

function legacyCopy(text) {
  const area = document.createElement('textarea');
  area.value = text;
  area.style.position = 'fixed';
  area.style.opacity = '0';
  document.body.append(area);
  area.select();
  document.execCommand('copy');
  area.remove();
}

function div(className) {
  const element = document.createElement('div');
  element.className = className;
  return element;
}

function span(className, text) {
  const element = document.createElement('span');
  element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function spacer() {
  return span('spacer');
}

function td(text, className) {
  const cell = document.createElement('td');
  cell.textContent = text;
  cell.className = className || '';
  return cell;
}

function field(labelText) {
  const wrap = div('sim-field');
  wrap.append(span('sim-field-label', labelText));
  return wrap;
}

function radio(name, value, labelText) {
  const wrap = document.createElement('label');
  wrap.className = 'sim-radio';
  const input = document.createElement('input');
  input.type = 'radio';
  input.name = name;
  input.value = value;
  const label = span('sim-radio-label', labelText);
  wrap.append(input, label);
  return { wrap, input, label };
}

function numberInput(labelText, title) {
  const wrap = document.createElement('label');
  wrap.className = 'sim-number';
  wrap.title = title;
  const label = span('sim-field-label', labelText);
  const input = document.createElement('input');
  input.type = 'number';
  input.min = '0';
  input.step = '1';
  input.value = '0';
  wrap.append(label, input);
  return { wrap, input };
}

function smallButton(text, title, onClick) {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'btn-text';
  button.textContent = text;
  button.title = title;
  button.addEventListener('click', onClick);
  return button;
}

function toggle(labelText, title, onClick) {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'switch compact';
  button.setAttribute('role', 'switch');
  button.setAttribute('aria-checked', 'false');
  button.title = title;
  button.innerHTML = `<span class="switch-label">${labelText}</span><span class="switch-track"><span class="switch-thumb"></span></span>`;
  button.addEventListener('click', onClick);
  return button;
}
