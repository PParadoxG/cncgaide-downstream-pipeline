// Single selection/visibility store shared by the graph and the 3D viewer.
//
// Neither view talks to the other.  Both dispatch intents here and both
// re-render from the resolved state, so "click a feature" and "click a face"
// are the same round trip in opposite directions.

export function createStore(initial) {
  let state = initial;
  const subscribers = new Set();

  return {
    get: () => state,
    update(patch) {
      const next = typeof patch === 'function' ? patch(state) : patch;
      state = Object.assign({}, state, next);
      subscribers.forEach((fn) => fn(state));
    },
    subscribe(fn) {
      subscribers.add(fn);
      fn(state);
      return () => subscribers.delete(fn);
    },
  };
}

export function initialState() {
  return {
    // null | {kind:'feature', featureId} | {kind:'face', cadFaceId}
    // | {kind:'edge', edgeId} | {kind:'features', featureIds: [...]}
    selection: null,
    hover: null,
    // Both filters are re-seeded from each part's own data on load; these are
    // only what the page holds before the first part arrives.
    hiddenTypes: new Set(['PlanarFace']),
    visibleEdgeTypes: new Set(),
    isolate: false,
    showEdges: false,
    // Heuristic panel: which row is expanded, and the two narrowing switches.
    // Both default off — the panel shows the whole part until asked otherwise,
    // so its contents never move on their own when a selection changes.
    heuristicFocus: null,
    heuristicsFailsOnly: false,
    heuristicsPickedOnly: false,
    search: '',
    layout: 'grouped',
    theme: document.documentElement.dataset.theme || 'light',
  };
}

/**
 * Turn the raw selection into everything both views need.
 *
 * primaryFaceIds  raw CAD faces the user is pointing at right now
 * relatedFaceIds  the remaining raw CAD faces of the same feature(s)
 * featureIds      feature nodes that should read as selected in the graph
 */
export function resolveSelection(state, model) {
  const empty = {
    kind: 'none',
    featureIds: new Set(),
    primaryFaceIds: new Set(),
    relatedFaceIds: new Set(),
    allFaceIds: new Set(),
    relationNeighbourFaceIds: new Set(),
    relationNeighbourFeatureIds: new Set(),
    focusFaceId: null,
    focusFeatureId: null,
    focusEdgeId: null,
    edgeIds: new Set(),
  };
  const selection = state.selection;
  if (!selection) return empty;

  if (selection.kind === 'feature') {
    const node = model.nodeById.get(selection.featureId);
    if (!node) return empty;
    const faces = new Set(node.cadFaceIds);
    const relationNeighbourFaces = new Set();
    const relationNeighbourFeatures = new Set();
    for (const edge of model.incidentEdges.get(node.nodeId) || []) {
      if (!state.visibleEdgeTypes.has(edge.edgeType)) continue;
      const neighbourId = edge.source === node.nodeId ? edge.target : edge.source;
      const neighbour = model.nodeById.get(neighbourId);
      if (!neighbour) continue;
      relationNeighbourFeatures.add(neighbourId);
      for (const cadFaceId of neighbour.cadFaceIds || []) relationNeighbourFaces.add(cadFaceId);
    }
    return {
      kind: 'feature',
      featureIds: new Set([node.nodeId]),
      primaryFaceIds: faces,
      relatedFaceIds: new Set(),
      allFaceIds: faces,
      relationNeighbourFaceIds: relationNeighbourFaces,
      relationNeighbourFeatureIds: relationNeighbourFeatures,
      focusFaceId: null,
      focusFeatureId: node.nodeId,
      focusEdgeId: null,
      edgeIds: new Set(),
    };
  }

  if (selection.kind === 'face') {
    const row = model.faceById.get(selection.cadFaceId);
    if (!row) return empty;
    const primary = new Set([selection.cadFaceId]);
    const related = new Set();
    const featureIds = new Set();
    for (const featureId of row.featureIds) {
      featureIds.add(featureId);
      const node = model.nodeById.get(featureId);
      if (!node) continue;
      for (const cadFaceId of node.cadFaceIds) {
        if (!primary.has(cadFaceId)) related.add(cadFaceId);
      }
    }
    // Sibling raw faces of the same merged PartModel face are always related,
    // even when nothing claimed the face — that is the seam/merge case.
    for (const cadFaceId of row.mergedCadFaceIds) {
      if (!primary.has(cadFaceId)) related.add(cadFaceId);
    }
    return {
      kind: 'face',
      featureIds,
      primaryFaceIds: primary,
      relatedFaceIds: related,
      allFaceIds: new Set([...primary, ...related]),
      relationNeighbourFaceIds: new Set(),
      relationNeighbourFeatureIds: new Set(),
      focusFaceId: selection.cadFaceId,
      focusFeatureId: row.featureIds[0] || null,
      focusEdgeId: null,
      edgeIds: new Set(),
    };
  }

  // Several features as one primary selection — what an edit-path entry of
  // the compare page names on one side (both endpoints of a relation edge).
  // The part page never creates this kind.
  if (selection.kind === 'features') {
    const featureIds = new Set();
    const faces = new Set();
    for (const featureId of selection.featureIds || []) {
      const node = model.nodeById.get(featureId);
      if (!node) continue;
      featureIds.add(featureId);
      for (const cadFaceId of node.cadFaceIds || []) faces.add(cadFaceId);
    }
    if (!featureIds.size) return empty;
    return {
      kind: 'features',
      featureIds,
      primaryFaceIds: faces,
      relatedFaceIds: new Set(),
      allFaceIds: faces,
      relationNeighbourFaceIds: new Set(),
      relationNeighbourFeatureIds: new Set(),
      focusFaceId: null,
      focusFeatureId: [...featureIds][0],
      focusEdgeId: null,
      edgeIds: new Set(),
    };
  }

  if (selection.kind === 'edge') {
    const edge = model.edgeById.get(selection.edgeId);
    if (!edge) return empty;
    const featureIds = new Set([edge.source, edge.target]);
    const primary = new Set();
    for (const reference of edge.evidenceFaceRefs || []) {
      for (const cadFaceId of reference.cadFaceIds || []) primary.add(Number(cadFaceId));
    }
    const related = new Set();
    for (const featureId of featureIds) {
      const node = model.nodeById.get(featureId);
      if (!node) continue;
      for (const cadFaceId of node.cadFaceIds || []) {
        if (!primary.has(cadFaceId)) related.add(cadFaceId);
      }
    }
    return {
      kind: 'edge',
      featureIds,
      primaryFaceIds: primary,
      relatedFaceIds: related,
      allFaceIds: new Set([...primary, ...related]),
      relationNeighbourFaceIds: new Set(),
      relationNeighbourFeatureIds: new Set(),
      focusFaceId: primary.size ? [...primary][0] : null,
      focusFeatureId: edge.source,
      focusEdgeId: edge.edgeId,
      edgeIds: new Set([edge.edgeId]),
    };
  }

  return empty;
}

export function resolveHover(state, model) {
  const hover = state.hover;
  if (!hover) return { faceIds: new Set(), featureIds: new Set() };
  if (hover.kind === 'face') {
    return { faceIds: new Set([hover.cadFaceId]), featureIds: new Set() };
  }
  if (hover.kind === 'feature') {
    const node = model.nodeById.get(hover.featureId);
    return {
      faceIds: new Set(node ? node.cadFaceIds : []),
      featureIds: new Set([hover.featureId]),
    };
  }
  return { faceIds: new Set(), featureIds: new Set() };
}

/** A raw CAD face is visible when no owning feature type is filtered out. */
export function isFaceVisible(state, model, cadFaceId) {
  const row = model.faceById.get(cadFaceId);
  if (!row) return true;
  if (!row.featureIds.length) return true; // unclaimed faces are always reviewable
  return row.featureIds.some((featureId) => {
    const node = model.nodeById.get(featureId);
    return !node || !state.hiddenTypes.has(node.featureType);
  });
}
