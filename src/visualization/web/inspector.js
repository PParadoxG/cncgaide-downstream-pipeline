// Details panel.  Renders what the main line already recorded, generically:
// attributes come straight out of graph_view.json `properties`, so a new
// ontology attribute appears here without any code change.

import { featureLabel } from './data.js';
import { resolveSelection } from './store.js';

export function createInspector(container, store, callbacks) {
  let model = null;

  function setModel(nextModel) {
    model = nextModel;
    render(store.get());
  }

  function render(state) {
    if (!model) return;
    const selection = resolveSelection(state, model);
    container.innerHTML = '';

    if (selection.kind === 'feature') renderFeature(container, model, state, selection, callbacks);
    else if (selection.kind === 'face') renderFace(container, model, state, selection, callbacks);
    else if (selection.kind === 'edge') renderEdge(container, model, state, selection, callbacks);
    else renderPartSummary(container, model);
  }

  store.subscribe(render);
  return { setModel };
}

// --------------------------------------------------------------- renderers

function renderFeature(root, model, state, selection, callbacks) {
  const node = model.nodeById.get(selection.focusFeatureId);
  const missing = Object.keys(node.missingAttributes || {});

  const typeBadge = badge(featureLabel(model, node.featureType), null, node.color);
  typeBadge.title = node.featureType;
  root.append(title(node.color, node.nodeId, [
    typeBadge,
    ...(missing.length ? [badge(`${missing.length} missing`, 'warn')] : []),
  ]));

  const read = model.usedAttributes && model.usedAttributes[node.featureType];
  // An empty list is an answer - "nothing on this class is read" - so the test
  // is whether the line said anything at all, not whether the list is non-empty.
  if (Array.isArray(read)) {
    // The line said which attribute names the pipeline reads on this class, so
    // the table shows those and folds everything else — the attributes nobody
    // read, plus the ones the line could not obtain — into one line.  The
    // comparison is `key in list`; nothing here knows what any name means, and
    // a line that supplies no list gets the old, fully expanded table.
    const wanted = new Set(read);
    const shown = {};
    const rest = {};
    for (const [key, value] of Object.entries(node.properties || {})) {
      (wanted.has(key) ? shown : rest)[key] = value;
    }
    Object.assign(rest, node.missingAttributes || {});
    root.append(section('Attributes', kvTable(shown)));
    const count = Object.keys(rest).length;
    if (count) root.append(section('', foldedTable(`Also published \u00b7 ${count}`, rest)));
  } else {
    root.append(section('Attributes', kvTable(node.properties)));
    if (missing.length) root.append(section('Missing', kvTable(node.missingAttributes)));
  }

  // The line's own measured quantities, kept apart from the recogniser's
  // attributes end to end (a value the pipeline derived must never sit where
  // a recogniser's value would).  Rendered generically: rows are whatever
  // keys the block carries, the full record rides in the tooltip.
  if (Object.keys(node.derived || {}).length) {
    root.append(section('Derived', derivedTable(node.derived)));
  }

  root.append(section(
    `Faces · ${node.displayFaceIds.length} display / ${node.cadFaceIds.length} raw`,
    faceList(node.displayFaceDetails, state, callbacks),
  ));

  const neighbours = [...(model.neighbours.get(node.nodeId) || [])];
  if (neighbours.length) {
    const list = document.createElement('div');
    for (const neighbourId of neighbours.sort()) {
      const neighbour = model.nodeById.get(neighbourId);
      list.append(linkButton(neighbourId, () => callbacks.onPickFeature(neighbourId), neighbour && neighbour.color));
    }
    root.append(section(`Direct adjacent · ${neighbours.length}`, list));
  }

  if (Object.keys(node.diagnostics || {}).length) {
    root.append(section('Diagnostics', kvTable(node.diagnostics)));
  }
}

function renderFace(root, model, state, selection, callbacks) {
  const row = model.faceById.get(selection.focusFaceId);
  const claimed = row.featureIds.length > 0;
  const owner = claimed ? model.nodeById.get(row.featureIds[0]) : null;

  root.append(title(owner ? owner.color : '#8d97a3', row.displayFaceId || `cad ${row.cadFaceId}`, [
    badge(row.surfaceType || 'unknown'),
    ...(claimed ? [] : [badge('unclaimed', 'warn')]),
  ]));

  root.append(section('Identity', kvTable({
    'cadFaceId': row.cadFaceId,
    'partmodelFaceId': row.partmodelFaceId,
    'displayFaceId': row.displayFaceId,
    'merged from': row.mergedCadFaceIds.length > 1 ? row.mergedCadFaceIds.join(', ') : '—',
  })));

  if (claimed) {
    const list = document.createElement('div');
    for (const featureId of row.featureIds) {
      const node = model.nodeById.get(featureId);
      list.append(linkButton(featureId, () => callbacks.onPickFeature(featureId), node && node.color));
    }
    root.append(section('Feature', list));
    if (owner) root.append(section('Attributes', kvTable(owner.properties)));
  } else {
    const note = document.createElement('div');
    note.className = 'insp-empty';
    note.textContent = 'No manufacturing feature claims this face. Identity above still comes from id_map.';
    root.append(section('Feature', note));
  }

  const actions = document.createElement('div');
  actions.append(linkButton('zoom to face', () => callbacks.onFrameFaces([row.cadFaceId])));
  if (row.mergedCadFaceIds.length > 1) {
    actions.append(linkButton('zoom to merge group', () => callbacks.onFrameFaces(row.mergedCadFaceIds)));
  }
  root.append(section('', actions));
}

function renderEdge(root, model, state, selection, callbacks) {
  const edge = model.edgeById.get(selection.focusEdgeId);
  if (!edge) return renderPartSummary(root, model);
  const source = model.nodeById.get(edge.source);
  const target = model.nodeById.get(edge.target);
  const color = (model.relationTypes.find((entry) => entry.edgeType === edge.edgeType) || {}).color || '#8d97a3';

  root.append(title(color, edge.edgeId, [badge(edge.edgeType, null, color), badge(edge.directed ? 'directed' : 'undirected')]));
  const endpoints = document.createElement('div');
  endpoints.append(
    linkButton(`${source ? featureLabel(model, source.featureType) + ' · ' : ''}${edge.source}`, () => callbacks.onPickFeature(edge.source), source && source.color),
    linkButton(`${target ? featureLabel(model, target.featureType) + ' · ' : ''}${edge.target}`, () => callbacks.onPickFeature(edge.target), target && target.color),
  );
  root.append(section('Endpoints', endpoints));
  // The relation's own properties render exactly like a node's attribute
  // table: whatever the ontology names, no per-relation display branch.
  if (Object.keys(edge.properties || {}).length) {
    root.append(section('Relation properties', kvTable(edge.properties)));
  }
  root.append(section('Relation evidence', kvTable(edge.evidence || {})));
  if (Object.keys(edge.diagnostics || {}).length) {
    root.append(section('Relation diagnostics', kvTable(edge.diagnostics)));
  }

  const refs = edge.evidenceFaceRefs || [];
  const witness = document.createElement('div');
  for (const reference of refs) {
    const label = `${reference.displayFaceId || reference.partmodelFaceId} · cad ${(reference.cadFaceIds || []).join(', ') || '—'}`;
    witness.append(linkButton(label, () => callbacks.onFrameFaces(reference.cadFaceIds || [])));
  }
  root.append(section(`Witness faces · ${refs.length}`, refs.length ? witness : emptyNote('No mapped witness face in this evidence.')));
  const actions = document.createElement('div');
  actions.append(linkButton('zoom to relation witnesses', () => callbacks.onFrameFaces([...selection.allFaceIds])));
  root.append(section('', actions));
}

function renderPartSummary(root, model) {
  const stats = model.stats;
  root.append(title('var(--accent)', model.partId, []));

  // One card: the part's numbers, a hairline, then its feature types with
  // their colours — the legend of the 3D view.
  const summary = document.createElement('div');
  summary.append(kvTable({
    'raw CAD faces': stats.rawCadFaceCount,
    'PartModel faces': stats.manifestPartmodelFaceCount,
    'features': stats.featureNodeCount,
    'relations': model.edges.length,
    'merged faces': Object.keys(stats.mergedDisplayFaces).length,
    'unclaimed faces': stats.unclaimedRawCadFaceIds.length,
  }));
  const divider = document.createElement('div');
  divider.className = 'insp-divider';
  summary.append(divider, typeTable(model.featureTypes));
  root.append(section('Part', summary));

  const help = document.createElement('div');
  help.className = 'insp-empty';
  help.textContent = 'Click a node, relation edge, or face to inspect evidence and highlight its mapped geometry.';
  root.append(section('', help));
}

function emptyNote(text) {
  const empty = document.createElement('div');
  empty.className = 'insp-empty';
  empty.textContent = text;
  return empty;
}

// ----------------------------------------------------------------- widgets

function title(color, id, badges) {
  const wrap = document.createElement('div');
  wrap.className = 'insp-title';
  const dot = document.createElement('span');
  dot.className = 'dot';
  dot.style.background = color;
  const label = document.createElement('span');
  label.className = 'id';
  label.textContent = id;
  wrap.append(dot, label, ...badges);
  return wrap;
}

function badge(text, kind, color) {
  const element = document.createElement('span');
  if (color) {
    element.className = 'chip-tint';
    element.style.setProperty('--chip', color);
  } else {
    element.className = 'badge' + (kind ? ' ' + kind : '');
  }
  element.textContent = text;
  return element;
}

function section(heading, body) {
  const wrap = document.createElement('div');
  wrap.className = 'insp-section';
  if (heading) {
    const h = document.createElement('h3');
    h.textContent = heading;
    wrap.append(h);
  }
  wrap.append(body);
  return wrap;
}

// Above this many rows a table opens folded.  A feature with a hundred
// attributes is a fact about the vocabulary in play, not a mistake — NX's
// six-step blind hole carries 107 and declares 143 more it will not hand
// over — but an unfolded hundred-row table pushes everything else off the
// panel.  The threshold is on the row count alone: nothing here knows or
// cares which class or which attribute it is looking at.
const FOLD_ROWS_ABOVE = 24;

function kvTable(object) {
  const entries = Object.entries(object || {});
  if (!entries.length) {
    const empty = document.createElement('div');
    empty.className = 'insp-empty';
    empty.textContent = '—';
    return empty;
  }
  const table = document.createElement('table');
  table.className = 'kv';
  for (const [key, value] of entries) {
    const tr = document.createElement('tr');
    const k = document.createElement('td');
    k.className = 'k';
    k.textContent = key;
    const v = document.createElement('td');
    v.className = 'v' + (value === null || value === undefined ? ' null' : '');
    v.textContent = formatValue(value);
    tr.append(k, v);
    table.append(tr);
  }
  if (entries.length <= FOLD_ROWS_ABOVE) return table;

  const details = document.createElement('details');
  details.className = 'kv-fold';
  const summary = document.createElement('summary');
  summary.textContent = `${entries.length} entries`;
  details.append(summary, table);
  return details;
}

/** The derived block: one row per measured quantity, the record in the tooltip. */
function derivedTable(object) {
  const table = document.createElement('table');
  table.className = 'kv';
  for (const [key, entry] of Object.entries(object || {})) {
    const record = entry && typeof entry === 'object' && !Array.isArray(entry) ? entry : { value: entry };
    const tr = document.createElement('tr');
    tr.title = JSON.stringify(record);
    const k = document.createElement('td');
    k.className = 'k';
    k.textContent = key;
    const v = document.createElement('td');
    const refused = record.value === null || record.value === undefined;
    v.className = 'v' + (refused ? ' null' : '');
    v.textContent = refused ? (record.reason ? `null · ${record.reason}` : 'null')
      : formatValue(record.value);
    tr.append(k, v);
    table.append(tr);
  }
  return table;
}

/** A kv table that always starts folded, behind a caption. */
function foldedTable(caption, object) {
  const details = document.createElement('details');
  details.className = 'kv-fold';
  const summary = document.createElement('summary');
  summary.textContent = caption;
  const table = kvTable(object);
  // kvTable folds long tables itself; inside a fold that would be two clicks.
  details.append(summary, table.tagName === 'DETAILS' ? table.lastChild : table);
  return details;
}

/** Feature types with their counts and colours, as the kv table but with a dot. */
function typeTable(featureTypes) {
  const table = document.createElement('table');
  table.className = 'kv';
  for (const entry of featureTypes) {
    const tr = document.createElement('tr');
    const k = document.createElement('td');
    k.className = 'k';
    const dot = document.createElement('i');
    dot.className = 'dot';
    dot.style.background = entry.color;
    k.append(dot, document.createTextNode(entry.label));
    k.title = entry.type;
    const v = document.createElement('td');
    v.className = 'v';
    v.textContent = String(entry.count);
    tr.append(k, v);
    table.append(tr);
  }
  return table;
}

function formatValue(value) {
  if (value === null || value === undefined) return 'null';
  if (Array.isArray(value)) return '[' + value.map(formatValue).join(', ') + ']';
  if (typeof value === 'number') {
    return Number.isInteger(value) ? String(value) : String(Number(value.toFixed(4)));
  }
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
}

function faceList(details, state, callbacks) {
  const wrap = document.createElement('div');
  const focusFaceId = state.selection && state.selection.kind === 'face' ? state.selection.cadFaceId : null;

  for (const detail of details) {
    for (const cadFaceId of detail.cadFaceIds) {
      const row = document.createElement('div');
      row.className = 'face-row' + (cadFaceId === focusFaceId ? ' active' : '');
      const fid = document.createElement('span');
      fid.className = 'fid';
      fid.textContent = detail.displayFaceId;
      const meta = document.createElement('span');
      meta.className = 'meta';
      meta.textContent = `cad ${cadFaceId} · ${detail.surfaceType || '?'}`;
      row.append(fid, meta);
      row.addEventListener('click', () => callbacks.onPickFace(cadFaceId));
      wrap.append(row);
    }
  }
  if (!wrap.childElementCount) {
    const empty = document.createElement('div');
    empty.className = 'insp-empty';
    empty.textContent = '—';
    return empty;
  }
  return wrap;
}

/** A link to another element: a tinted chip when it has a colour (a feature
 *  or relation), a plain text button otherwise. */
function linkButton(label, handler, color) {
  const button = document.createElement('button');
  button.type = 'button';
  if (color) {
    button.className = 'chip-tint mono link-chip';
    button.style.setProperty('--chip', color);
  } else {
    button.className = 'btn-text';
  }
  button.textContent = label;
  button.addEventListener('click', handler);
  return button;
}
