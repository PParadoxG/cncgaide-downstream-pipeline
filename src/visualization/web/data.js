// Loading + indexing of the artifacts produced by visualization/viewer_data.py.
//
// Nothing here derives new geometry semantics: the feature/face identity comes
// straight from the main-line id_map.json and graph_view.json that the
// preprocessor copied into view.json.

const NEUTRAL = '#8d97a3';

export async function loadIndex() {
  const response = await fetch('data/index.json');
  if (!response.ok) throw new Error(`viewer index unavailable (HTTP ${response.status})`);
  return response.json();
}

// A part directory holds one folder per line, both carrying the *same* file
// names, so switching lines is a change of one path segment and nothing else.
// The mesh is per line too: each line tessellates with its own kernel.
export function lineFolder(line) {
  return line === 'nx' ? 'nx' : 'occ';
}

function partBase(partId, line) {
  return `data/${encodeURIComponent(partId)}/${lineFolder(line)}/`;
}

export async function loadPart(partId, line) {
  const base = partBase(partId, line);
  const [view, mesh, buffer, heuristics] = await Promise.all([
    fetch(base + 'view.json').then(assertOk).then((r) => r.json()),
    fetch(base + 'mesh.json').then(assertOk).then((r) => r.json()),
    fetch(base + 'mesh.bin').then(assertOk).then((r) => r.arrayBuffer()),
    // Optional on purpose: a run produced before the heuristic layer existed
    // still opens, with the panel saying so instead of the page failing.
    loadHeuristics(base),
  ]);
  const model = buildModel(view, mesh, buffer, heuristics);
  model.line = line || 'occ';
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
  const [mesh, buffer] = await Promise.all([
    fetch(base + 'mesh.json').then(assertOk).then((r) => r.json()),
    fetch(base + 'mesh.bin').then(assertOk).then((r) => r.arrayBuffer()),
  ]);
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
    outputTag: null,
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
  model.line = shownAs || meshFrom || 'occ';
  return model;
}

export async function loadHeuristics(base) {
  try {
    const response = await fetch(base + 'heuristics.json');
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
    sourceStep: view.sourceStep,
    outputTag: view.outputTag,
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
