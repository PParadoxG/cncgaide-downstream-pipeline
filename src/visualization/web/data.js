// Loading + indexing of the artifacts produced by visualization/viewer_data.py.
//
// Nothing here derives new geometry semantics: the feature/face identity comes
// straight from the main-line id_map.json and graph_view.json that the
// preprocessor copied into view.json.

const NEUTRAL = '#8d97a3';

// ------------------------------------------------------------ which build
//
// A part is four files, fetched in parallel, and the store publishes a new
// build one os.replace at a time: "old view.json, new mesh.json" is reachable
// while a refresh lands, and the page would draw it without noticing (node 26
// G-03).  The guard is a demand rather than a check after the fact — every
// request for run data carries `If-Match` naming the build the page started
// on, and the server answers 409 instead of a file from another build
// (server.py::_if_match_satisfied).  A 409 becomes `RunChanged`, which the
// pages already know what to do with: it is a part that failed to load, and
// the named degradation path takes the part off screen and says why.
//
// Unset (a page that never learned a run id) means no demand and no change
// in behaviour, which is also what an older page gets from the new server.

let expectedRun = null;

/** The build this page reads.  Set once, from the index the page loaded. */
export function setExpectedRun(runId) {
  expectedRun = runId || null;
}

export function currentExpectedRun() {
  return expectedRun;
}

/** The validator the server stamps run data with; `null` when unknown. */
export function runValidator(runId) {
  return runId ? `W/"${runId}"` : null;
}

export class RunChanged extends Error {
  constructor(runId) {
    super('the run was rebuilt while this page was reading it'
      + (runId ? ` (the store now serves ${runId})` : ''));
    this.name = 'RunChanged';
    this.runId = runId || null;
  }
}

function runOptions() {
  const validator = runValidator(expectedRun);
  return validator ? { cache: 'no-store', headers: { 'If-Match': validator } } : { cache: 'no-store' };
}

async function fetchRunFile(url) {
  const response = await fetch(url, runOptions());
  if (response.status === 409) {
    let runId = null;
    try { runId = (await response.json()).runId || null; } catch { runId = null; }
    throw new RunChanged(runId);
  }
  return assertOk(response);
}

/**
 * The build every response claimed, when they all claim the same one.
 *
 * Belt to the `If-Match` braces, and the only guard a page has before it
 * knows its run id: the validators of the files that were just assembled into
 * one model have to agree, or the model is a mixture of two builds.
 */
function oneRunOnly(responses) {
  const tags = new Set(responses.map((r) => r && r.headers && r.headers.get('ETag')).filter(Boolean));
  if (tags.size > 1) throw new RunChanged(null);
  return [...tags][0] || null;
}

export async function loadIndex() {
  const response = await fetch('data/index.json');
  if (!response.ok) throw new Error(`viewer index unavailable (HTTP ${response.status})`);
  return response.json();
}

// A part directory holds one folder per line, both carrying the *same* file
// names, so switching lines is a change of one path segment and nothing else.
// The mesh is per line too: each line tessellates with its own kernel.
//
// The folder *is* the line id (contract/registry.py), so there is no mapping
// to keep in step here — and no default either.  A caller that forgets to say
// which line it wants is refused rather than handed one: the fallback this
// replaced returned the open-source folder, which is how the statistics view
// came to read v5.1 verdicts under the NX line's counts without anything
// failing.  An unknown line asks for a folder that is not there and says so.
export function lineFolder(line) {
  if (!line) throw new Error('lineFolder: no line named');
  return line;
}

function partBase(partId, line) {
  return `data/${encodeURIComponent(partId)}/${lineFolder(line)}/`;
}

export async function loadPart(partId, line) {
  const base = partBase(partId, line);
  const [viewResponse, meshResponse, bufferResponse, heuristics] = await Promise.all([
    fetchRunFile(base + 'view.json'),
    fetchRunFile(base + 'mesh.json'),
    fetchRunFile(base + 'mesh.bin'),
    // Optional on purpose: a run produced before the heuristic layer existed
    // still opens, with the panel saying so instead of the page failing.
    loadHeuristics(base),
  ]);
  const runId = oneRunOnly([viewResponse, meshResponse, bufferResponse]);
  const [view, mesh, buffer] = await Promise.all([
    viewResponse.json(), meshResponse.json(), bufferResponse.arrayBuffer(),
  ]);
  const model = buildModel(view, mesh, buffer, heuristics);
  model.line = line;
  model.runValidator = runId;
  return model;
}

/**
 * Geometry with no recognition result on top: every face neutral and
 * unclaimed, no nodes, no edges, no verdicts.  What the part page shows when
 * the chosen line has nothing for this part — the reason is named in the
 * banner, and the shape stays reviewable.  *meshFrom* is the line whose mesh
 * to borrow — the other one, since the chosen line built nothing — while
 * *shownAs* stays the line the page is on, because that is the question the
 * reader asked and the banner answers.
 */
export async function loadPartGeometry(partId, meshFrom, shownAs) {
  const base = partBase(partId, meshFrom);
  const [meshResponse, bufferResponse] = await Promise.all([
    fetchRunFile(base + 'mesh.json'),
    fetchRunFile(base + 'mesh.bin'),
  ]);
  const runId = oneRunOnly([meshResponse, bufferResponse]);
  const [mesh, buffer] = await Promise.all([meshResponse.json(), bufferResponse.arrayBuffer()]);
  const faces = {};
  for (const record of mesh.faces) {
    faces[String(record.cadFaceId)] = {
      cadFaceId: record.cadFaceId,
      displayFaceId: null,
      partmodelFaceId: null,
      surfaceType: null,
      featureIds: [],
      mergedCadFaceIds: [],
      hasMesh: record.indexCount > 0,
    };
  }
  const view = {
    partId,
    sourceStep: null,
    nodes: [],
    edges: [],
    relationTypes: [],
    faces,
    featureCounts: {},
    defaultHiddenFeatureTypes: [],
    stats: {
      rawCadFaceCount: mesh.cadFaceCount,
      featureNodeCount: 0,
      unclaimedRawCadFaceIds: Object.keys(faces).map(Number).sort((a, b) => a - b),
    },
  };
  const model = buildModel(view, mesh, buffer, null);
  model.geometryOnly = true;
  model.line = shownAs || meshFrom;
  model.runValidator = runId;
  return model;
}

export async function loadHeuristics(base) {
  let response;
  try {
    response = await fetch(base + 'heuristics.json', runOptions());
  } catch (error) {
    return null;
  }
  // "This build has no heuristics" and "this is not your build" are different
  // answers: the first is the missing-layer case this function exists for,
  // the second must reach the caller or the part quietly loses its verdicts
  // while the rest of it comes from the older build.
  if (response.status === 409) {
    let runId = null;
    try { runId = (await response.json()).runId || null; } catch { runId = null; }
    throw new RunChanged(runId);
  }
  try {
    return response.ok ? await response.json() : null;
  } catch (error) {
    return null;
  }
}

function assertOk(response) {
  if (!response.ok) throw new Error(`${response.url}: HTTP ${response.status}`);
  return response;
}

function buildModel(view, mesh, buffer, heuristics) {
  const nodeById = new Map();
  for (const node of view.nodes) nodeById.set(node.nodeId, node);

  const faceById = new Map();
  for (const key of Object.keys(view.faces)) {
    const row = view.faces[key];
    faceById.set(row.cadFaceId, row);
  }

  // Feature types present, in a stable order: most common first, then by name.
  const typeCounts = new Map();
  for (const node of view.nodes) {
    typeCounts.set(node.featureType, (typeCounts.get(node.featureType) || 0) + 1);
  }
  const colorByType = new Map();
  for (const node of view.nodes) colorByType.set(node.featureType, node.color);

  // `label` is the words a human reads for this class, `type` stays the key.
  // On the NX line they differ (STEP1HOLE / "Through Hole"); on the
  // open-source line the class name already is the words, and the map is
  // absent, so `label` falls back to `type` with no branch anywhere.
  const labels = view.featureLabels || {};
  const featureTypes = [...typeCounts.entries()]
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .map(([type, count]) => ({
      type,
      label: labels[type] || type,
      count,
      color: colorByType.get(type) || NEUTRAL,
    }));

  const meshFaces = new Map();
  for (const record of mesh.faces) meshFaces.set(record.cadFaceId, record);

  // Relation neighbours, indexed once so edge selection and the graph
  // highlight can both look them up cheaply.  The active relation-type set
  // is applied later by resolveSelection; this index deliberately keeps all
  // materialized edge types available.
  const edges = view.edges || [];
  const edgeById = new Map();
  const neighbours = new Map();
  const incidentEdges = new Map();
  for (const edge of edges) {
    edgeById.set(edge.edgeId, edge);
    for (const [a, b] of [[edge.source, edge.target], [edge.target, edge.source]]) {
      if (!neighbours.has(a)) neighbours.set(a, new Set());
      neighbours.get(a).add(b);
      if (!incidentEdges.has(a)) incidentEdges.set(a, []);
      incidentEdges.get(a).push(edge);
    }
  }

  // Relation types come from the run, so a rule registered after this file was
  // written still gets a chip, a colour and a default.
  const relationTypes = view.relationTypes || [];

  return {
    partId: view.partId,
    // No `outputTag`: the per-part view no longer states which run it sits in
    // (F-14), and nothing on any page ever read the field.  The tag is on the
    // index, which is where the footer takes it from.
    sourceStep: view.sourceStep,
    nodes: view.nodes,
    relationTypes,
    defaultHiddenFeatureTypes: view.defaultHiddenFeatureTypes || [],
    defaultVisibleEdgeTypes: relationTypes.filter((r) => r.defaultVisible).map((r) => r.edgeType),
    edges,
    edgeById,
    neighbours,
    incidentEdges,
    nodeById,
    faceById,
    faceIds: [...faceById.keys()].sort((a, b) => a - b),
    featureTypes,
    featureLabels: labels,
    // What this line evaluated none of, and why it says so — written by the
    // line, read by the panels.  A layer named here is a deliberate silence;
    // a layer absent from an old run is simply unknown, and the two must not
    // look the same on screen.
    notEvaluated: view.notEvaluated || {},
    // `{class: names}` when the line says which attributes the pipeline reads,
    // `null` when it does not - and `null` means "expand everything", which is
    // what a line whose attributes are all its own wants.
    usedAttributes: view.usedAttributes || null,
    // Verdicts of the heuristic layer, exactly as written by the pipeline.
    // null when the run predates that layer.
    heuristics: heuristics && Array.isArray(heuristics.records) ? heuristics : null,
    stats: view.stats,
    mesh,
    meshFaces,
    buffer,
  };
}

/**
 * Colour a raw CAD face carries.
 *
 * Filtered-out feature types normally fall back to neutral so the CAD view is
 * not drowned in PlanarFace colour, but a face the user explicitly selected
 * keeps its owner's colour — otherwise clicking a filtered face would look
 * like clicking nothing.
 */
/** The words a human reads for one class; the class name when it has none. */
export function featureLabel(model, featureType) {
  return (model && model.featureLabels && model.featureLabels[featureType]) || featureType;
}

export function baseFaceColor(model, cadFaceId, hiddenTypes, ignoreHidden, neutral) {
  const fallback = neutral || NEUTRAL;
  const row = model.faceById.get(cadFaceId);
  if (!row || !row.featureIds.length) return fallback;
  for (const featureId of row.featureIds) {
    const node = model.nodeById.get(featureId);
    if (node && (ignoreHidden || !hiddenTypes.has(node.featureType))) return node.color;
  }
  return fallback;
}

export const NEUTRAL_COLOR = NEUTRAL;
