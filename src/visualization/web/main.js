// Wiring: one store, three dumb views.
//
// The graph and the 3D viewer never call each other.  Both raise intents here,
// the store resolves them, and both re-render from the same resolved selection.

import { loadIndex, loadPart, loadPartGeometry, setExpectedRun } from './data.js';
import { RUN_MOVED_TEXT, reloadPage } from './runwatch.js';
import { escapeHtml } from './html.js';
import { currentLine, lineLabel, setupLineSwitch } from './line.js';
import { createStore, initialState, resolveSelection } from './store.js';
import { createGraph } from './graph.js';
import { createViewer3D } from './viewer3d.js';
import { createInspector } from './inspector.js';
import { createHeuristicsPanel } from './heuristics.js';
import { externalGapsShown, setExternalGapsShown } from './prefs.js';
import { setupTheme } from './theme.js';
import { describeJob, fetchParts, postRefresh } from './refresh.js';

const store = createStore(initialState());
const $ = (id) => document.getElementById(id);

let model = null;
let graph = null;
let viewer = null;
let inspector = null;
let heuristics = null;
let indexData = null;
// Why the shown line has no result for the open part (null = it has one).
let lineGap = null;
// Set when the open part could not be loaded at all.  Distinct from lineGap:
// that one still has geometry to show, this one has nothing.
let loadError = null;

// ------------------------------------------------------------------ intents

const callbacks = {
  // First click selects, a second click on the already-selected feature flies
  // the camera to it.  Selecting stays cheap, and the camera only moves when
  // the reviewer asks for it — no mode to remember.
  onPickFeature(featureId) {
    const current = store.get().selection;
    const repeat = Boolean(featureId)
      && current
      && current.kind === 'feature'
      && current.featureId === featureId;

    store.update({ selection: featureId ? { kind: 'feature', featureId } : null });
    if (!repeat) return;
    const node = model.nodeById.get(featureId);
    if (node) viewer.focusFeature(node.cadFaceIds, directionHints(node));
  },
  onPickEdge(edgeId) {
    if (edgeId === null || edgeId === undefined) {
      store.update({ selection: null });
      return;
    }
    const current = store.get().selection;
    const repeat = current && current.kind === 'edge' && current.edgeId === edgeId;
    store.update({ selection: { kind: 'edge', edgeId } });
    if (!repeat) return;
    const resolved = resolveSelection(store.get(), model);
    if (resolved.allFaceIds.size) viewer.frameFaces([...resolved.allFaceIds]);
  },
  onPickFace(cadFaceId) {
    if (cadFaceId === null || cadFaceId === undefined) {
      store.update({ selection: null });
      return;
    }
    store.update({ selection: { kind: 'face', cadFaceId } });
    const row = model.faceById.get(cadFaceId);
    if (row && row.featureIds.length) graph.centerOn(row.featureIds[0]);
  },
  onHoverFeature(featureId) {
    store.update({ hover: featureId ? { kind: 'feature', featureId } : null });
  },
  onHoverFace(cadFaceId, event) {
    const empty = cadFaceId === null || cadFaceId === undefined;
    store.update({ hover: empty ? null : { kind: 'face', cadFaceId } });
    updateTooltip(cadFaceId, event);
  },
  onFrameFaces(cadFaceIds) { viewer.frameFaces(cadFaceIds); },
  // A heuristic verdict names its evaluation unit in the same identities the
  // rest of the page uses, so picking one is the ordinary selection round
  // trip — an edge verdict selects the edge, everything else selects the
  // first feature and frames every feature the verdict was computed on.
  onPickHeuristic(record) {
    const edgeId = (record.targets.edgeIds || [])[0];
    if (edgeId && model.edgeById.has(edgeId)) {
      store.update({ selection: { kind: 'edge', edgeId } });
    } else {
      const featureId = (record.targets.nodeIds || [])[0];
      if (!featureId || !model.nodeById.has(featureId)) return;
      store.update({ selection: { kind: 'feature', featureId } });
      graph.centerOn(featureId);
    }
    const faces = new Set();
    for (const nodeId of record.targets.nodeIds || []) {
      const node = model.nodeById.get(nodeId);
      for (const cadFaceId of (node && node.cadFaceIds) || []) faces.add(cadFaceId);
    }
    if (!faces.size) {
      const resolved = resolveSelection(store.get(), model);
      for (const cadFaceId of resolved.allFaceIds) faces.add(cadFaceId);
    }
    if (faces.size) viewer.frameFaces([...faces]);
  },
};

/**
 * Directional attributes to seed the camera search with.
 *
 * Taken as "any attribute that is three numbers" rather than a list of known
 * names: the ontology renamed its axis attributes once already
 * (`axisDirection` became `feature_axis` / `machining_direction` /
 * `surface_normal`), and a hard-coded name silently degrades to no hint at all.
 * A vector that turns out not to be a useful viewpoint simply loses to the
 * sphere sweep during scoring, so a wrong guess costs nothing.
 */
function directionHints(node) {
  const hints = [];
  for (const value of Object.values(node.properties || {})) {
    if (!Array.isArray(value) || value.length !== 3) continue;
    if (!value.every((component) => Number.isFinite(component))) continue;
    hints.push(value);
  }
  return hints;
}

// -------------------------------------------------------------------- boot

async function boot() {
  const index = await loadIndex();
  indexData = index;
  // Everything this page fetches from here on is demanded from *this* build;
  // a store that publishes another one mid-load is refused, not mixed in.
  setExpectedRun(index.runId);

  const select = $('part-select');
  for (const part of index.parts) {
    const option = document.createElement('option');
    option.value = part.partId;
    option.textContent = part.partId;
    select.append(option);
  }

  const params = new URLSearchParams(location.search);
  const requested = params.get('part');
  const partId = index.parts.some((p) => p.partId === requested) ? requested : index.parts[0].partId;
  select.value = partId;

  // Deep link from the statistics view: open this part with one heuristic
  // already expanded in the panel.
  const focus = params.get('h');
  if (focus) store.update({ heuristicFocus: focus });

  select.addEventListener('change', () => {
    const url = new URL(location.href);
    url.searchParams.set('part', select.value);
    history.replaceState(null, '', url);
    openPart(select.value);
  });

  graph = createGraph($('cy'), store, callbacks);
  viewer = createViewer3D($('viewport'), store, callbacks);
  inspector = createInspector($('inspector'), store, callbacks);
  heuristics = createHeuristicsPanel($('heuristics'), store, callbacks);

  // Switching line swaps the data source and the accent tokens, nothing
  // else: the same part reloads from the other line's artifacts.
  setupLineSwitch($('line-seg'), () => {
    updateSimilarLink();
    openPart(select.value);
  });
  updateSimilarLink();

  bindToolbar();
  bindSplitters();
  bindKeys();
  syncToggles();

  // Audit hook: lets the linkage be checked from a script without clicking
  // through the UI.  Read-only handles, no hidden behaviour.
  window.tempVisual = {
    store,
    graph,
    viewer,
    heuristics,
    prefs: { externalGapsShown, setExternalGapsShown },
    callbacks,
    get model() { return model; },
    get line() { return currentLine(); },
    get lineGap() { return lineGap; },
    resolveSelection: () => resolveSelection(store.get(), model),
  };

  await openPart(partId);
  startStoreWatch(index.runId);
}

/** The index's word on what *line* managed to produce for *partId*. */
function lineInfo(partId, line) {
  const entry = (indexData?.parts || []).find((part) => part.partId === partId);
  return (entry && entry.lines && entry.lines[line]) || null;
}

/**
 * A line whose mesh can stand in for one that has no result for this part.
 *
 * The candidates come from the index's own list of lines, in the order the
 * run wrote them - not from a fixed occ/nx pair, which is what stood here.
 * The registry allows more than two lines, and the pair would have sent a
 * third line to fetch the NX mesh whether NX had built this part or not.
 */
function meshDonor(partId, shownLine) {
  const entry = (indexData?.parts || []).find((part) => part.partId === partId);
  for (const [candidate, state] of Object.entries((entry && entry.lines) || {})) {
    if (candidate === shownLine) continue;
    if (state && state.status === 'PASS') return candidate;
  }
  return null;
}

async function openPart(partId) {
  $('loading').classList.remove('hidden');
  try {
    const line = currentLine();
    const info = lineInfo(partId, line);
    lineGap = null;
    loadError = null;
    if (info && info.status === 'PASS') {
      model = await loadPart(partId, line);
    } else {
      // This line has nothing for this part.  Named, not blank: the geometry
      // another line tessellated stays reviewable and the banner says why.
      lineGap = { line, status: (info && info.status) || 'NOT_BUILT', info };
      const donor = meshDonor(partId, line);
      if (!donor) {
        throw new Error(`no line has built this part, so there is no mesh to show it with`);
      }
      model = await loadPartGeometry(partId, donor, line);
    }
    // Defaults travel with the data: which feature types start hidden, and
    // which relation types start visible.  A relation registered later shows
    // up here switched off without any change to this file.
    store.update({
      selection: null,
      hover: null,
      search: '',
      hiddenTypes: new Set(model.defaultHiddenFeatureTypes),
      visibleEdgeTypes: new Set(model.defaultVisibleEdgeTypes),
    });
    $('graph-search').value = '';

    graph.setModel(model);
    viewer.setModel(model);
    inspector.setModel(model);
    heuristics.setModel(model);

    renderTypeFilter();
    renderRelationFilter();
    $('find-similar').href = `index.html?view=similarity&run=1&query=${encodeURIComponent(model.partId)}`;
    storeWatch.loadedProducedAt = loadedLineProducedAt();
    renderBanner();
  } catch (error) {
    // Nothing of the part that was on screen may survive this.  By the time
    // the load fails the picker and the URL already name the new part, so
    // leaving the old solid, graph and verdicts up shows one part's geometry
    // under another part's name — which reads as correct and is the worst of
    // the failure modes.  The views are emptied first, then the reason is
    // named twice: once in the banner, where the other named degradations
    // live, and once where the details of the part would have been.
    model = null;
    // A refused build is known here before the next poll would find it, and
    // the banner should say so in the same paint that empties the views.
    if (error && error.name === 'RunChanged') {
      storeWatch.movedTo = error.runId || 'a newer build';
    }
    showNoPart(partId, error);
    console.error(error);
  } finally {
    $('loading').classList.add('hidden');
  }
}

/** Empty every view and say why the part on the picker is not on screen. */
function showNoPart(partId, error) {
  model = null;
  lineGap = null;
  loadError = { partId, message: (error && error.message) || String(error) };
  // The views are emptied before the store is touched: each of them ignores a
  // state update once its model is gone, so this order cannot redraw the part
  // that just went away.
  graph.setModel(null);
  viewer.setModel(null);
  inspector.setModel(null);
  heuristics.setModel(null);
  store.update({
    selection: null,
    hover: null,
    search: '',
    hiddenTypes: new Set(),
    visibleEdgeTypes: new Set(),
  });
  $('graph-search').value = '';
  renderTypeFilter();
  renderRelationFilter();
  $('find-similar').href = `index.html?view=similarity&run=1&query=${encodeURIComponent(partId)}`;
  storeWatch.loadedProducedAt = null;
  renderBanner();
  const note = document.createElement('div');
  note.className = 'insp-empty';
  note.textContent = `${partId} could not be opened: ${loadError.message}`;
  $('inspector').append(note);
}

// ------------------------------------------------------------------- chrome

function renderTypeFilter() {
  const container = $('type-filter');
  container.innerHTML = '';
  if (!model) return; // no part on screen: there is nothing to filter
  const hidden = store.get().hiddenTypes;

  for (const entry of model.featureTypes) {
    const chip = document.createElement('button');
    chip.className = 'type-chip' + (hidden.has(entry.type) ? '' : ' on');
    chip.style.setProperty('--chip', entry.color);
    // The chip says what a person calls the class; the key it stands for is
    // in the tooltip, because that is what every id in the run is built from.
    const swatch = document.createElement('span');
    swatch.className = 'swatch';
    chip.append(swatch, document.createTextNode(`${entry.label} ${entry.count}`));
    chip.title = entry.label === entry.type ? entry.type : `${entry.label} · ${entry.type}`;
    chip.addEventListener('click', () => {
      const next = new Set(store.get().hiddenTypes);
      if (next.has(entry.type)) next.delete(entry.type); else next.add(entry.type);
      store.update({ hiddenTypes: next });
      renderTypeFilter();
      // Hiding a feature type changes how many edges of each relation type
      // can be drawn, so the relation chips have to be recounted.
      renderRelationFilter();
    });
    container.append(chip);
  }
}

// An edge is drawn only when its type is enabled *and* both endpoint nodes
// survive the feature filter.  Most LOCATED_ON edges terminate on PlanarFace,
// which is hidden by default, so switching the type on can look like nothing
// happened.  The chip therefore counts what is actually drawable.
function drawableEdgeCount(edgeType) {
  const hidden = store.get().hiddenTypes;
  let total = 0;
  for (const edge of model.edges) {
    if (edge.edgeType !== edgeType) continue;
    const ends = [model.nodeById.get(edge.source), model.nodeById.get(edge.target)];
    if (ends.every((node) => node && !hidden.has(node.featureType))) total += 1;
  }
  return total;
}

function renderRelationFilter() {
  const container = $('relation-filter');
  container.innerHTML = '';
  if (!model) return; // no part on screen: there is nothing to filter
  const visible = store.get().visibleEdgeTypes;

  for (const entry of model.relationTypes) {
    const drawable = drawableEdgeCount(entry.edgeType);
    const chip = document.createElement('button');
    chip.className = 'type-chip' + (visible.has(entry.edgeType) ? ' on' : '');
    chip.style.setProperty('--chip', entry.color);
    chip.title = drawable === entry.count
      ? `${entry.count} ${entry.edgeType} edge(s) in this part`
      : `${entry.count} ${entry.edgeType} edge(s) in this part, ${drawable} drawable`
        + `\n${entry.count - drawable} have an endpoint whose feature type is hidden`;
    // The drawable count is the number that matters; the total sits beside
    // it, quieter, with the reason in the tooltip.
    chip.innerHTML = `<span class="swatch"></span>${escapeHtml(entry.edgeType)} `
      + (drawable === entry.count
        ? `<span class="n">${entry.count}</span>`
        : `<span class="n">${drawable}</span><span class="of">/${entry.count}</span>`);
    chip.addEventListener('click', () => {
      const next = new Set(store.get().visibleEdgeTypes);
      if (next.has(entry.edgeType)) next.delete(entry.edgeType);
      else next.add(entry.edgeType);
      store.update({ visibleEdgeTypes: next });
      renderRelationFilter();
    });
    container.append(chip);
  }
}

function updateTooltip(cadFaceId, event) {
  const tip = $('viewport-tip');
  if (cadFaceId === null || cadFaceId === undefined || !event || !model) {
    tip.classList.remove('on');
    return;
  }
  const row = model.faceById.get(cadFaceId);
  if (!row) { tip.classList.remove('on'); return; }
  const owner = row.featureIds.length ? row.featureIds.join(', ') : 'unclaimed';
  tip.textContent = `${row.displayFaceId || '—'} · cad ${row.cadFaceId}\n${owner}`;
  const rect = $('viewport').getBoundingClientRect();
  tip.style.left = `${event.clientX - rect.left}px`;
  tip.style.top = `${event.clientY - rect.top}px`;
  tip.classList.add('on');
}

// ------------------------------------------------------------------ toolbar

// Since node 21' each line has its own corpus, representation and cache, so
// the link is live on both; the query it opens is always within the line the
// page is showing, because the two scales are not comparable.
function updateSimilarLink() {
  const link = $('find-similar');
  if (!link) return;
  link.classList.remove('disabled');
  link.removeAttribute('aria-disabled');
  link.title = 'Rank the other parts of this line by graph edit distance to this one';
}

function bindToolbar() {
  document.querySelector('.toolbar').addEventListener('click', (event) => {
    const button = event.target.closest('button[data-action]');
    if (button) doAction(button.dataset.action);
  });

  $('graph-search').addEventListener('input', (event) => store.update({ search: event.target.value }));
  $('graph-layout').addEventListener('change', (event) => store.update({ layout: event.target.value }));

  setupTheme($('theme-toggle'), (theme) => store.update({ theme }));
}

function doAction(action) {
  const state = store.get();
  if (action === 'fit') {
    const selection = resolveSelection(state, model);
    if (selection.allFaceIds.size) viewer.frameFaces([...selection.allFaceIds]);
    else viewer.fit();
    graph.fit();
  } else if (action === 'isolate') {
    store.update({ isolate: !state.isolate });
  } else if (action === 'edges') {
    store.update({ showEdges: !state.showEdges });
  } else if (action === 'clear') {
    store.update({ selection: null, hover: null }); // Esc only, no button
  }
  syncToggles();
}

function syncToggles() {
  const state = store.get();
  const set = (selector, on) => {
    const element = document.querySelector(selector);
    if (element) element.setAttribute('aria-checked', String(on));
  };
  set('[data-action="isolate"]', state.isolate);
  set('[data-action="edges"]', state.showEdges);
}

function bindKeys() {
  window.addEventListener('keydown', (event) => {
    const target = event.target;
    if (target instanceof Element && target.matches('input, select, textarea')) return;
    const key = event.key.toLowerCase();
    if (key === 'f') doAction('fit');
    else if (key === 'i') doAction('isolate');
    else if (key === 'e') doAction('edges');
    else if (event.key === 'Escape') doAction('clear');
  });
}

// ---------------------------------------------------------------- splitters

function bindSplitters() {
  const workspace = $('workspace');
  const sideStack = $('side-stack');

  // Ratios, not pixels — the split survives a window resize.
  let graphFraction = 0.4;
  let inspectorWidth = 330;
  // The right column is itself split: details on top, heuristic verdicts
  // below.  A ratio again, so the two panes keep their proportion.
  let inspectorFraction = 0.55;

  function layout() {
    const total = workspace.getBoundingClientRect().width;
    if (!total) return;
    inspectorWidth = clamp(inspectorWidth, 240, Math.max(240, total * 0.4));
    const available = total - inspectorWidth - 12;
    const left = clamp(available * graphFraction, 180, Math.max(180, available - 240));
    workspace.style.gridTemplateColumns = `${left}px 6px 1fr 6px ${inspectorWidth}px`;
    layoutSide();
  }

  function layoutSide() {
    const height = sideStack.getBoundingClientRect().height;
    if (!height) return;
    const top = clamp(height * inspectorFraction, 120, Math.max(120, height - 140));
    sideStack.style.gridTemplateRows = `${top}px 6px minmax(0, 1fr)`;
  }

  // The column's height is only final once the workspace has been laid out
  // and the panes have their content, which is after the first layout() call.
  // Watching it keeps the ratio honest instead of freezing whatever height
  // happened to be measured first.
  if (typeof ResizeObserver === 'function') {
    new ResizeObserver(() => layoutSide()).observe(sideStack);
  }

  function makeDraggable(splitter, apply) {
    splitter.addEventListener('pointerdown', (event) => {
      event.preventDefault();
      splitter.classList.add('dragging');
      splitter.setPointerCapture(event.pointerId);

      const onMove = (moveEvent) => {
        const rect = workspace.getBoundingClientRect();
        apply(moveEvent.clientX - rect.left, rect.width);
        graph.resize();
        viewer.invalidate();
      };
      const onUp = () => {
        splitter.classList.remove('dragging');
        splitter.removeEventListener('pointermove', onMove);
        splitter.removeEventListener('pointerup', onUp);
        graph.fit();
      };
      splitter.addEventListener('pointermove', onMove);
      splitter.addEventListener('pointerup', onUp);
    });
  }

  makeDraggable($('splitter-1'), (x, total) => {
    graphFraction = clamp(x / Math.max(1, total - inspectorWidth - 12), 0.12, 0.82);
    layout();
  });
  makeDraggable($('splitter-2'), (x, total) => {
    inspectorWidth = total - x - 6;
    layout();
  });

  // The horizontal one moves in Y, so it gets its own tiny drag handler
  // rather than a Y-aware branch inside the vertical one.
  const rowSplitter = $('splitter-3');
  rowSplitter.addEventListener('pointerdown', (event) => {
    event.preventDefault();
    rowSplitter.classList.add('dragging');
    rowSplitter.setPointerCapture(event.pointerId);
    const onMove = (moveEvent) => {
      const rect = sideStack.getBoundingClientRect();
      inspectorFraction = clamp((moveEvent.clientY - rect.top) / Math.max(1, rect.height), 0.12, 0.88);
      layoutSide();
    };
    const onUp = () => {
      rowSplitter.classList.remove('dragging');
      rowSplitter.removeEventListener('pointermove', onMove);
      rowSplitter.removeEventListener('pointerup', onUp);
    };
    rowSplitter.addEventListener('pointermove', onMove);
    rowSplitter.addEventListener('pointerup', onUp);
  });

  // A resize re-splits the panes and tells the two canvases their new size.
  // Neither is re-framed: a window getting bigger is not a request to throw
  // away where the reviewer had panned to, and Fit and F do it on demand.
  //
  // The canvases are told once the resize has stopped rather than on every
  // event of it.  Hosts that animate their viewport -- the in-app browser
  // pane grows over about four seconds as it opens -- fire a resize a frame,
  // and cy.resize() reallocates all three cytoscape canvas layers on each
  // one: measured at 59 reallocations in four seconds, which reads as the
  // whole graph flickering while nobody is touching anything.  The leading
  // call keeps a one-off resize instant; the trailing one is the size a drag
  // or an animation actually settles on.  The pane split itself is not
  // debounced -- it is a style write that costs nothing when the value has
  // not changed, and it is what keeps the panes tracking the window live.
  const RESIZE_SETTLE_MS = 120;
  let resizeTimer = null;
  let resizing = false;
  function syncCanvasesToPanes() {
    graph.resize();
    viewer.invalidate();
  }
  window.addEventListener('resize', () => {
    layout();
    if (!resizing) {
      resizing = true;
      syncCanvasesToPanes();
    }
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      resizing = false;
      syncCanvasesToPanes();
    }, RESIZE_SETTLE_MS);
  });
  layout();
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(value, max));
}

// ------------------------------------------------------------ store banner
//
// One line under the header when the part on screen is not the part the
// store would build today: built under older rules, source changed, queued or
// building right now, rebuilt since this page loaded, failed, or its STEP is
// gone.  Nothing is reloaded under the reviewer's feet — the banner offers the
// action and waits.

const storeWatch = {
  rows: new Map(),          // partId -> api/parts row
  status: null,             // last api/status payload
  runId: null,              // run id this page loaded
  movedTo: null,            // run id on disk, once it is not the one above
  loadedProducedAt: null,   // producedAt of the open part's shown line when it loaded
  timer: null,
  rowFor(partId) { return this.rows.get(partId) || null; },
};

const POLL_IDLE = 5000;
const POLL_RUNNING = 2000;

function startStoreWatch(runId) {
  storeWatch.runId = runId;
  pollStore();
}

async function pollStore() {
  clearTimeout(storeWatch.timer);
  let status = null;
  try {
    status = await fetch('api/status', { cache: 'no-store' }).then((r) => r.json());
  } catch {
    status = null;
  }
  storeWatch.status = status;
  // The run id this page loaded is finally read by somebody.  Once the store
  // serves a different build, every artifact on screen came from the previous
  // one - not just the open part - and the whole page is what has to be
  // re-taken.  Offered, never performed: the reviewer decides when.
  if (status && status.dataRunId && storeWatch.runId && status.dataRunId !== storeWatch.runId) {
    storeWatch.movedTo = status.dataRunId;
  }
  const rows = await fetchParts();
  if (rows) storeWatch.rows = new Map(rows.map((row) => [row.partId, row]));
  if (model && storeWatch.loadedProducedAt === null) {
    storeWatch.loadedProducedAt = loadedLineProducedAt();
  }
  renderBanner();
  if (status && status.state === 'disabled') return; // read-only: nothing will move
  const busy = status && (status.state === 'running' || (status.queue || []).length);
  storeWatch.timer = setTimeout(pollStore, busy ? POLL_RUNNING : POLL_IDLE);
}

function renderBanner() {
  const banner = $('store-banner');
  if (!banner) return;
  banner.innerHTML = '';
  banner.hidden = true;
  banner.dataset.state = '';

  // The run itself moving outranks everything: it is not one part that is
  // out of date but every artifact the page is holding, and the answer is a
  // page reload rather than any per-part action below.
  if (storeWatch.movedTo) {
    banner.dataset.state = 'updated';
    const chip = document.createElement('span');
    chip.className = 'st-chip';
    chip.textContent = 'rebuilt';
    const text = document.createElement('span');
    text.className = 'banner-text';
    text.textContent = RUN_MOVED_TEXT
      + (loadError ? ` ${loadError.partId} is not on screen: ${loadError.message}.` : '');
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'btn banner-btn';
    button.textContent = 'Reload page';
    button.addEventListener('click', reloadPage);
    banner.append(chip, text, button);
    banner.hidden = false;
    return;
  }

  // A part that could not be loaded outranks everything below, and it is the
  // only state in which the rest of this function has no part to describe.
  if (loadError) {
    banner.dataset.state = 'missing';
    const chip = document.createElement('span');
    chip.className = 'st-chip new';
    chip.textContent = 'not loaded';
    const text = document.createElement('span');
    text.className = 'banner-text';
    text.textContent = `${loadError.partId} could not be loaded — ${loadError.message}.`
      + ' Nothing of this part is on screen.';
    banner.append(chip, text);
    banner.hidden = false;
    return;
  }
  if (!model) return;
  const row = storeWatch.rowFor(model.partId);
  const status = storeWatch.status;
  const writable = Boolean(status) && status.state !== 'disabled';

  // The line gap outranks store drift: when the page is showing bare
  // geometry because the shown line has no result, that is the one thing the
  // reviewer must not miss.
  if (lineGap && model.geometryOnly) {
    const label = lineLabel(lineGap.line);
    const detail = lineGap.info && (lineGap.info.reason || lineGap.info.error);
    const kinds = { NOT_BUILT: 'missing', FAIL: 'failed' };
    const texts = {
      NOT_BUILT: `${label} was not computed for this part${detail ? ' — ' + detail : ''}; the next refresh tries again.`,
      FAIL: `The ${label} line failed for this part${detail ? ' — ' + detail : ''}.`,
    };
    banner.dataset.state = kinds[lineGap.status] || 'missing';
    const chip = document.createElement('span');
    chip.className = 'st-chip ' + (banner.dataset.state === 'missing' ? 'new' : banner.dataset.state);
    chip.textContent = lineGap.status === 'FAIL' ? 'failed' : 'not computed';
    const text = document.createElement('span');
    text.className = 'banner-text';
    text.textContent = `${texts[lineGap.status] || texts.NOT_BUILT} Showing geometry only.`;
    banner.append(chip, text);
    banner.hidden = false;
    return;
  }
  if (!row) return;

  const light = row.light || {};
  const line = (row.lines || {})[currentLine()] || {};
  let text = '';
  let kind = '';
  let action = null;

  if (row.job) {
    kind = 'building';
    text = row.job.state === 'running'
      ? `This part is being rebuilt now (${describeJob(row.job)}).`
      : `A rebuild of this part is queued (${row.job.jobId}).`;
  } else if (storeWatch.loadedProducedAt && line.producedAt && line.producedAt !== storeWatch.loadedProducedAt) {
    kind = 'updated';
    text = 'This part was rebuilt after the page loaded — the view shows the previous results.';
    action = { label: 'Reload part', run: () => { storeWatch.loadedProducedAt = null; openPart(model.partId); } };
  } else if (line.status === 'FAIL') {
    kind = 'failed';
    text = `The last build of this line failed${line.error ? ': ' + line.error : ''}; the results on screen are older.`;
    if (writable) action = { label: 'Re-run', run: () => rerunCurrent() };
  } else if (light.state === 'source_missing' || light.state === 'untracked') {
    kind = 'missing';
    text = 'The source STEP of this part is no longer in the input directory; the next refresh drops it.';
  } else if (light.state === 'changed') {
    kind = 'stale';
    text = 'The source STEP changed since this was built — re-run to update.';
    if (writable) action = { label: 'Re-run', run: () => rerunCurrent() };
  } else if ((row.rulesChanged || []).length) {
    kind = 'stale';
    text = `Built with older rules (${row.rulesChanged.join(', ')}) — re-run to update.`;
    if (writable) action = { label: 'Re-run', run: () => rerunCurrent() };
  } else {
    return;
  }

  banner.dataset.state = kind;
  const label = document.createElement('span');
  label.className = 'banner-text';
  label.textContent = text;
  banner.append(label);
  if (action) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'btn banner-btn';
    button.textContent = action.label;
    button.addEventListener('click', action.run);
    banner.append(button);
  } else if (!writable && (kind === 'stale' || kind === 'failed')) {
    const hint = document.createElement('span');
    hint.className = 'banner-hint';
    hint.textContent = 'read-only server — re-run from a terminal';
    banner.append(hint);
  }
  banner.hidden = false;
}

async function rerunCurrent() {
  if (!model) return;
  try {
    // Rebuild even if the store considers the part fresh: mode "all", one id.
    await postRefresh({ mode: 'all', only: [model.partId] });
  } catch (error) {
    console.error(error);
  }
  clearTimeout(storeWatch.timer);
  storeWatch.timer = setTimeout(pollStore, 300);
}

boot().catch((error) => {
  $('loading').innerHTML = `<span class="spin"></span>${escapeHtml(error.message)}`;
  console.error(error);
});

/** The build time of the shown line for the open part, or null. */
function loadedLineProducedAt() {
  const row = storeWatch.rowFor(model?.partId);
  return ((row && row.lines && row.lines[currentLine()]) || {}).producedAt || null;
}
