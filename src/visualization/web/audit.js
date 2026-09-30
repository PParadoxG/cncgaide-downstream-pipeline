// Maintenance check for the web viewer.  Load it from any viewer page console:
//
//   await fetch('web/audit.js').then(r => r.text()).then(eval)
//   await runViewerAudit();      // part page: two-way Feature <-> CAD binding
//                                // launcher: store + similarity API
//                                // compare page: both views, edit path, highlights
//   await runAutoViewAudit();    // part page: does the camera see the feature
//   await runTimingAudit();      // compare page / launcher: a superseded
//                                // answer must not land (needs the server's
//                                // debug delay switch; see "timing" below)
//
// It walks every part, every feature and every raw CAD face using only the
// public window.tempVisual / window.tempCompare hooks plus real cytoscape/DOM
// events, so it exercises the same code paths a human reviewer would.  Run it
// after touching anything under visualization/, and after a main-line change
// to features or relations.

/** The kernel lines this page offers, read off the switch.
 *
 *  The list used to be written out here as a pair.  The registry allows more
 *  than two lines (contract/registry.py), and a third one would have been
 *  skipped by every sweep below without a word - an audit that quietly stops
 *  covering something is worse than no audit.
 */
function pageLines() {
  const ids = [...document.querySelectorAll('#line-seg button[data-line]')]
    .map((button) => button.dataset.line)
    .filter(Boolean);
  if (!ids.length) throw new Error('runViewerAudit: this page has no line switch');
  return ids;
}

/** Hold every transition still for the length of a measurement.
 *
 *  The geometry pass asks where a control ended up, and a control that slides
 *  there answers with wherever it is mid-slide.  Waiting it out is not enough:
 *  in a document the host never paints - a hidden tab, an embedded pane, a
 *  headless run - the transition does not advance at all, so the thumb stays
 *  at its starting value for ever and the pass reports it as sitting under the
 *  wrong tab.  That was two standing false failures, in exactly the runs where
 *  nobody could look at the screen to know better.
 *
 *  Freezing transitions makes the final value the current value in one frame,
 *  in every host.  It is the layout being checked here, never the animation.
 */
function freezeTransitions() {
  const style = document.createElement('style');
  style.textContent = '*, *::before, *::after '
    + '{ transition: none !important; animation: none !important; }';
  document.head.append(style);
  void document.body.offsetHeight; // force the frozen values to be computed
  return () => style.remove();
}

window.runViewerAudit = async function runViewerAudit() {
  if (window.tempCompare) return withGeometry(auditComparePage());
  if (document.getElementById('part-select') && window.tempVisual) {
    const results = await auditPartPage();
    for (const line of pageLines()) results.push(await auditSimilarityApi(line));
    return withGeometry(results);
  }
  if (document.getElementById('view-tabs')) return withGeometry(auditLauncherPage());
  throw new Error('runViewerAudit: not a viewer page');
};

/** Every page gets the geometry pass, last, once its own checks have run.
 *
 *  Geometry can only be read off what is on screen, so where a page keeps
 *  several views behind one control every view is brought up in turn and the
 *  reviewer's own choice is put back.  Without that the pass would only ever
 *  measure the view that happens to open first, which is the one that has
 *  never been the broken one. */
async function withGeometry(pending) {
  const results = await pending;
  const thaw = freezeTransitions();
  try {
    const tabs = [...document.querySelectorAll('#view-tabs button[data-view]')];
    if (!tabs.length) {
      results.push(auditGeometry());
      return results;
    }
    const chosen = tabs.find((tab) => tab.getAttribute('aria-selected') === 'true') || tabs[0];
    for (const tab of tabs) {
      tab.click();
      await settled();
      const pass = auditGeometry();
      pass.partId = `(geometry ${tab.dataset.view})`;
      results.push(pass);
    }
    chosen.click();
    await settled();
    return results;
  } finally {
    thaw();
  }
}

/** Wait until the boxes stop moving.
 *
 *  A view switch slides the thumb over a fifth of a second, and measuring
 *  part way through reports it as sitting under the wrong tab.  Rather than
 *  copy that duration out of the stylesheet -- where it can change without
 *  anyone thinking of this file -- watch the geometry itself.  Two equal
 *  frames are not enough: the embedded browser pane holds the thumb still
 *  for a few frames *before* the transition starts, which reads as settled
 *  and the pass then measures the old position.  So the geometry must hold
 *  still for a stretch of wall time longer than any trigger-to-movement
 *  latency -- a property of render scheduling, not of the stylesheet.
 *
 *  The geometry pass freezes transitions before it calls this (see
 *  freezeTransitions), so what is being waited out there is the relayout a
 *  view switch causes, not a slide. */
async function settled(limit = 80) {
  let previous = null;
  let heldSince = performance.now();
  for (let i = 0; i < limit; i += 1) {
    const boxes = [...document.querySelectorAll('.seg-thumb')]
      .map((element) => {
        const box = element.getBoundingClientRect();
        return `${box.left.toFixed(1)},${box.width.toFixed(1)}`;
      })
      .join('|');
    const now = `${boxes}#${document.body.scrollHeight}x${document.body.scrollWidth}`;
    if (now !== previous) {
      previous = now;
      heldSince = performance.now();
    } else if (performance.now() - heldSince >= 150) {
      return;
    }
    await frame();
  }
}

async function auditPartPage() {
  const results = [];
  const select = document.getElementById('part-select');
  const partIds = [...select.options].map((option) => option.value);

  // The whole per-part sweep runs once per line: the page must hold its
  // contracts over whichever line's artifacts it is showing.  A part a line
  // has nothing for is checked for its *named* gap state instead of the full
  // sweep — "nothing here" must still say why.
  const indexPayload = await fetch('data/index.json', { cache: 'no-store' }).then((r) => r.json());
  const lineMeta = new Map((indexPayload.parts || []).map(
    (part) => [part.partId, part.lines || {}],
  ));
  const originalLine = window.tempVisual.line;
  const accentSeen = {};

  async function switchLine(line) {
    if (window.tempVisual.line === line) return;
    const button = document.querySelector(`#line-seg button[data-line="${line}"]`);
    if (!button) throw new Error('line switch not on the page');
    button.click();
    await waitFor(() => window.tempVisual.line === line
      && window.tempVisual.model && window.tempVisual.model.line === line);
    await frame();
  }

  for (const line of pageLines()) {
  await switchLine(line);
  accentSeen[line] = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim();
  for (const partId of partIds) {
    select.value = partId;
    select.dispatchEvent(new Event('change'));
    await waitFor(() => window.tempVisual.model && window.tempVisual.model.partId === partId
      && window.tempVisual.model.line === line);
    await frame();

    const tv = window.tempVisual;
    const model = tv.model;
    const store = tv.store;
    const failures = [];

    // ---- 0. line state: a named gap, or counts that match the index ----
    const meta = (lineMeta.get(partId) || {})[line] || null;
    if (meta && meta.status === 'PASS') {
      if (model.geometryOnly) failures.push(`${line} result exists but the page shows geometry only`);
      if (model.nodes.length !== meta.nodeCount) {
        failures.push(`${line} view has ${model.nodes.length} nodes, the index says ${meta.nodeCount}`);
      }
      if (model.edges.length !== meta.edgeCount) {
        failures.push(`${line} view has ${model.edges.length} edges, the index says ${meta.edgeCount}`);
      }
    } else {
      // Nothing from this line: geometry only, and the banner names why.
      if (!model.geometryOnly || !tv.lineGap) {
        failures.push('line gap not represented as geometry-only');
      }
      if (model.nodes.length) failures.push('gap state still shows feature nodes');
      const banner = document.getElementById('store-banner');
      if (!banner || banner.hidden || !banner.querySelector('.st-chip')) {
        failures.push('line gap without a named banner chip');
      }
      results.push({ partId, line, lineGap: tv.lineGap && tv.lineGap.status, failures });
      continue;
    }

    // ---- 1. mesh <-> raw CAD face count -----------------------------------
    const meshFaceIds = Object.keys(tv.viewer.debugFaces()).map(Number).sort((a, b) => a - b);
    if (meshFaceIds.length !== model.stats.rawCadFaceCount) {
      failures.push(`mesh count ${meshFaceIds.length} != rawCadFaceCount ${model.stats.rawCadFaceCount}`);
    }
    for (const cadFaceId of meshFaceIds) {
      if (!model.faceById.has(cadFaceId)) failures.push(`mesh face ${cadFaceId} has no id_map row`);
    }

    // ---- 2. the data's own display defaults are honoured -------------------
    for (const type of model.defaultHiddenFeatureTypes) {
      if (!store.get().hiddenTypes.has(type)) failures.push(`${type} not hidden by default`);
    }
    const visibleEdgeTypes = store.get().visibleEdgeTypes;
    for (const entry of model.relationTypes) {
      if (entry.defaultVisible && !visibleEdgeTypes.has(entry.edgeType)) {
        failures.push(`${entry.edgeType} should start visible`);
      }
      if (!entry.defaultVisible && visibleEdgeTypes.has(entry.edgeType)) {
        failures.push(`${entry.edgeType} should start hidden`);
      }
    }
    // The background layer is whatever the *data* names as hidden by default;
    // each line names its own, so the audit must not carry one line's word
    // for it or it silently stops checking anything on the other.
    const background = new Set(model.defaultHiddenFeatureTypes);
    const shownNodes = tv.graph.cy.nodes('[!isGroup]').map((element) => element.id());
    const shownNodesSet = new Set(shownNodes);
    for (const nodeId of shownNodes) {
      if (background.has(model.nodeById.get(nodeId).featureType)) {
        failures.push(`background node ${nodeId} visible in graph`);
      }
    }

    // ---- 3. relation edges are copied, never invented ----------------------
    const knownEdgeIds = new Set(model.edges.map((edge) => edge.edgeId));
    tv.graph.cy.edges().forEach((element) => {
      if (!knownEdgeIds.has(element.id())) failures.push(`edge ${element.id()} not in view.json`);
      if (!shownNodesSet.has(element.data('source')) || !shownNodesSet.has(element.data('target'))) {
        failures.push(`edge ${element.id()} touches a hidden node`);
      }
      if (!visibleEdgeTypes.has(element.data('edgeType'))) {
        failures.push(`edge ${element.id()} drawn for a switched-off relation type`);
      }
    });

    // ---- 4. feature -> CAD highlight ---------------------------------------
    let featureChecks = 0;
    for (const node of model.nodes) {
      tv.callbacks.onPickFeature(node.nodeId);
      const faces = tv.viewer.debugFaces();
      // Relation neighbours receive a deliberately weaker context glow.  The
      // feature audit is about the primary selection, so inspect the semantic
      // highlight role rather than deriving strength from feature RGB values.
      const lit = Object.keys(faces)
        .filter((id) => faces[id].highlightRole === 'primary')
        .map(Number)
        .sort((a, b) => a - b);
      const expected = [...node.cadFaceIds].sort((a, b) => a - b);
      if (lit.join(',') !== expected.join(',')) {
        failures.push(`feature ${node.nodeId}: lit [${lit}] != owned [${expected}]`);
      }
      featureChecks += 1;
    }
    tv.callbacks.onPickFeature(null);

    // ---- 4b. the derived block renders apart from the attributes -----------
    //
    // A quantity this pipeline measured must never appear where a value the
    // recogniser stated would (node 19' 6).  On the page that means its own
    // section: every key of a node's `derived` block shows up under "Derived"
    // and none of them under "Attributes".
    let derivedChecks = 0;
    for (const node of model.nodes.filter((item) => item.derived
        && Object.keys(item.derived).length)) {
      tv.callbacks.onPickFeature(node.nodeId);
      const named = (heading) => {
        const section = [...document.querySelectorAll('.insp-section')]
          .find((element) => (element.querySelector('h3') || {}).textContent === heading);
        return section ? section.textContent : '';
      };
      const derived = named('Derived');
      const attributes = named('Attributes');
      for (const key of Object.keys(node.derived)) {
        derivedChecks += 1;
        if (!derived.includes(key)) failures.push(`${node.nodeId}: ${key} is not in the Derived section`);
        if (attributes.includes(key)) failures.push(`${node.nodeId}: ${key} leaked into Attributes`);
      }
    }
    tv.callbacks.onPickFeature(null);

    // ---- 5. CAD face -> feature reverse selection --------------------------
    let claimedChecks = 0;
    let unclaimedChecks = 0;
    let mergedChecks = 0;
    for (const cadFaceId of model.faceIds) {
      tv.callbacks.onPickFace(cadFaceId);
      const row = model.faceById.get(cadFaceId);
      const resolved = tv.resolveSelection();

      if (!resolved.primaryFaceIds.has(cadFaceId)) {
        failures.push(`face ${cadFaceId}: not primary after pick`);
      }
      const graphSelected = tv.graph.cy.nodes('.sel').map((element) => element.id()).sort();
      const expectedSelected = row.featureIds
        .filter((featureId) => shownNodes.includes(featureId))
        .sort();

      if (row.featureIds.length) {
        claimedChecks += 1;
        // A hidden-type owner is legitimately absent from the graph, but the
        // inspector must still name it.
        if (graphSelected.join(',') !== expectedSelected.join(',')) {
          failures.push(`face ${cadFaceId}: graph sel [${graphSelected}] != [${expectedSelected}]`);
        }
        const faceStates = tv.viewer.debugFaces();
        const lit = Object.keys(faceStates).filter((id) => faceStates[id].lit);
        if (!lit.map(Number).includes(cadFaceId)) failures.push(`face ${cadFaceId}: not lit after pick`);
      } else {
        unclaimedChecks += 1;
        if (graphSelected.length) failures.push(`unclaimed face ${cadFaceId} selected graph node(s)`);
        const title = document.querySelector('.insp-title .id');
        if (!title) failures.push(`unclaimed face ${cadFaceId}: inspector empty`);
      }

      if (row.mergedCadFaceIds.length > 1) {
        mergedChecks += 1;
        for (const sibling of row.mergedCadFaceIds) {
          if (!resolved.allFaceIds.has(sibling)) {
            failures.push(`face ${cadFaceId}: merged sibling ${sibling} not highlighted`);
          }
        }
      }
    }
    tv.callbacks.onPickFace(null);

    // ---- 6. relation edge -> witness CAD highlight -----------------------
    // Expose every relation type *and* every feature type.  Enabling the edge
    // types alone is not enough: an edge is only drawn when both endpoints
    // survive the node filter, and the background layer is hidden by default,
    // which is exactly where most LOCATED_ON edges terminate.
    const previousEdgeTypes = store.get().visibleEdgeTypes;
    const previousHiddenTypes = store.get().hiddenTypes;
    store.update({
      visibleEdgeTypes: new Set(model.relationTypes.map((entry) => entry.edgeType)),
      hiddenTypes: new Set(),
    });
    await frame();
    let edgeChecks = 0;
    let witnessChecks = 0;
    for (const edge of model.edges) {
      tv.callbacks.onPickEdge(edge.edgeId);
      const resolved = tv.resolveSelection();
      if (resolved.kind !== 'edge' || resolved.focusEdgeId !== edge.edgeId) {
        failures.push(`edge ${edge.edgeId}: selection did not resolve as edge`);
      }
      const selectedEdges = tv.graph.cy.edges('.sel').map((element) => element.id());
      if (selectedEdges.includes(edge.edgeId) === false) {
        failures.push(`edge ${edge.edgeId}: graph edge not selected`);
      }
      const faceState = tv.viewer.debugFaces();
      for (const reference of edge.evidenceFaceRefs || []) {
        for (const cadFaceId of reference.cadFaceIds || []) {
          witnessChecks += 1;
          if (!faceState[cadFaceId] || !faceState[cadFaceId].lit) {
            failures.push(`edge ${edge.edgeId}: witness cad face ${cadFaceId} not highlighted`);
          }
        }
      }
      edgeChecks += 1;
    }
    tv.callbacks.onPickEdge(null);

    // ---- 6b. an edge is drawn only when both endpoints are visible --------
    // The contract the check above silently assumed away.  Hide one feature
    // type again and every edge touching it must leave the graph.
    let endpointVisibilityChecks = 0;
    const hiddenProbe = model.nodes.find((node) => (model.incidentEdges.get(node.nodeId) || []).length);
    if (hiddenProbe) {
      store.update({ hiddenTypes: new Set([hiddenProbe.featureType]) });
      await frame();
      const drawn = new Set(tv.graph.cy.edges().map((element) => element.id()));
      for (const edge of model.edges) {
        const touchesHidden = [edge.source, edge.target].some((nodeId) => {
          const node = model.nodeById.get(nodeId);
          return node && node.featureType === hiddenProbe.featureType;
        });
        if (touchesHidden && drawn.has(edge.edgeId)) {
          failures.push(`edge ${edge.edgeId}: drawn although an endpoint type is hidden`);
        }
        endpointVisibilityChecks += 1;
      }
    }
    store.update({ visibleEdgeTypes: previousEdgeTypes, hiddenTypes: previousHiddenTypes });
    await frame();

    // ---- 7. feature relation context glow -------------------------------
    // The context is derived from the currently enabled edge types, not from
    // a relation-name branch.  Pick a visible, face-bearing feature with a
    // visible, face-bearing relation neighbour so both positive and
    // switched-off cases are exercised on every suitable part.
    const contextCandidate = model.nodes.find((node) => {
      if (background.has(node.featureType) || !node.cadFaceIds.length) return false;
      return (model.incidentEdges.get(node.nodeId) || []).some((edge) => {
        const neighbourId = edge.source === node.nodeId ? edge.target : edge.source;
        const neighbour = model.nodeById.get(neighbourId);
        return neighbour
          && !background.has(neighbour.featureType)
          && neighbour.cadFaceIds.length;
      });
    });
    let relationContextChecks = 0;
    if (contextCandidate) {
      const contextTypes = new Set(
        (model.incidentEdges.get(contextCandidate.nodeId) || []).map((edge) => edge.edgeType),
      );
      // A neighbour's faces, minus any the selected feature owns itself.
      // Two features can own the same raw face - NX's do routinely, a boss
      // and the corner notch that opens onto it share one - and such a face
      // stays lit as the selection's own whatever the relation types say.
      // Counting it as a neighbour face makes the switched-off case below
      // read a correct highlight as a failure.
      const neighbourFaces = new Set();
      for (const edge of model.incidentEdges.get(contextCandidate.nodeId) || []) {
        const neighbourId = edge.source === contextCandidate.nodeId ? edge.target : edge.source;
        const neighbour = model.nodeById.get(neighbourId);
        if (!neighbour || background.has(neighbour.featureType)) continue;
        for (const cadFaceId of neighbour.cadFaceIds || []) {
          if (!contextCandidate.cadFaceIds.includes(cadFaceId)) neighbourFaces.add(cadFaceId);
        }
      }

      store.update({ visibleEdgeTypes: contextTypes });
      tv.callbacks.onPickFeature(contextCandidate.nodeId);
      const resolvedContext = tv.resolveSelection();
      const positiveFaces = tv.viewer.debugFaces();
      const mainStrength = Math.max(...contextCandidate.cadFaceIds.map(
        (cadFaceId) => positiveFaces[cadFaceId]?.emissiveStrength || 0,
      ));
      const weakFaces = [...neighbourFaces].map((cadFaceId) => positiveFaces[cadFaceId]);
      if (!resolvedContext.relationNeighbourFaceIds.size || !weakFaces.length) {
        failures.push(`feature ${contextCandidate.nodeId}: relation context is empty`);
      } else if (weakFaces.some((face) => (
        face?.highlightRole !== 'relation-neighbour'
        || (face.emissiveStrength || 0) <= 0
        || (face.emissiveStrength || 0) >= mainStrength
      ))) {
        failures.push(`feature ${contextCandidate.nodeId}: neighbour glow is not weak`);
      }

      store.update({ visibleEdgeTypes: new Set() });
      const hiddenResolved = tv.resolveSelection();
      const hiddenFaces = tv.viewer.debugFaces();
      if (hiddenResolved.relationNeighbourFaceIds.size) {
        failures.push(`feature ${contextCandidate.nodeId}: hidden relation still resolves neighbours`);
      }
      if ([...neighbourFaces].some((cadFaceId) => (
        hiddenFaces[cadFaceId]?.highlightRole !== 'none'
        || (hiddenFaces[cadFaceId]?.emissiveStrength || 0) !== 0
      ))) {
        failures.push(`feature ${contextCandidate.nodeId}: hidden relation still lights neighbours`);
      }
      relationContextChecks = 1;
      tv.callbacks.onPickFeature(null);
      store.update({ visibleEdgeTypes: previousEdgeTypes });
      await frame();
    }

    // ---- 8. isolate hides everything else ----------------------------------
    // A feature rather than a background face, so isolate is exercised on
    // something the reviewer would actually isolate.
    const sample = model.nodes.find((node) => !background.has(node.featureType)) || model.nodes[0];
    if (sample) {
      tv.callbacks.onPickFeature(sample.nodeId);
      store.update({ isolate: true });
      const faces = tv.viewer.debugFaces();
      for (const [cadFaceId, face] of Object.entries(faces)) {
        const shouldShow = sample.cadFaceIds.includes(Number(cadFaceId));
        if (face.visible !== shouldShow) failures.push(`isolate: face ${cadFaceId} visible=${face.visible}`);
      }
      store.update({ isolate: false, selection: null });
    }

    // ---- 9. heuristic verdicts render and address real geometry -----------
    // The panel must be able to show every verdict of the part and every
    // verdict must name identities the page can resolve; a verdict pointing
    // at a node or edge the viewer does not know would be a broken drill-down.
    let heuristicChecks = 0;
    let heuristicSquareChecks = 0;
    const payload = model.heuristics;
    if (payload) {
      for (const record of payload.records) {
        heuristicChecks += 1;
        if (!payload.heuristics[record.hId]) {
          failures.push(`heuristic ${record.hId}: verdict without a catalogue entry`);
        }
        for (const nodeId of record.targets.nodeIds || []) {
          if (!model.nodeById.has(nodeId)) {
            failures.push(`heuristic ${record.hId}: verdict names unknown node ${nodeId}`);
          }
        }
        for (const edgeId of record.targets.edgeIds || []) {
          if (!model.edgeById.has(edgeId)) {
            failures.push(`heuristic ${record.hId}: verdict names unknown edge ${edgeId}`);
          }
        }
      }

      // Default view: one square per *evaluated* unit.  Not-applicable
      // verdicts are counted, never drawn — they say the rule does not talk
      // about that instance, which is not a result to scan for.
      store.update({
        selection: null,
        heuristicFocus: null,
        heuristicsFailsOnly: false,
        heuristicsPickedOnly: false,
      });
      // The audit checks the full set, so it asks for the external-input
      // gaps the reviewer normally leaves switched off — and puts their own
      // choice back when it is done.
      const externalGapsWereShown = tv.prefs.externalGapsShown();
      tv.prefs.setExternalGapsShown(true);
      await frame();
      const evaluated = payload.records.filter((record) => record.outcome !== 'NOT_APPLICABLE');
      const externalGaps = evaluated.filter(
        (record) => (record.evidence || {}).blocked === 'externalInputMissing',
      ).length;
      const drawn = () => document.querySelectorAll('#heuristics .heur-squares .heur-sq').length;
      if (drawn() !== evaluated.length) {
        failures.push(`heuristic panel draws ${drawn()} squares for ${evaluated.length} evaluated verdicts`);
      }
      heuristicSquareChecks = drawn();

      // Selecting geometry must *not* change what the panel lists; it only
      // marks the squares that talk about the selection.
      const probe = model.nodes.find(
        (node) => evaluated.some((record) => (record.targets.nodeIds || []).includes(node.nodeId)),
      );
      if (probe) {
        tv.callbacks.onPickFeature(probe.nodeId);
        await frame();
        if (drawn() !== evaluated.length) {
          failures.push(`heuristic panel: selecting ${probe.nodeId} changed the list to ${drawn()}`);
        }
        const marked = document.querySelectorAll('#heuristics .heur-squares .heur-sq.active').length;
        const about = evaluated.filter(
          (record) => (record.targets.nodeIds || []).includes(probe.nodeId),
        ).length;
        if (marked !== about) {
          failures.push(`heuristic panel: ${probe.nodeId} marks ${marked} squares, expected ${about}`);
        }

        // Only the explicit switch narrows the list.
        store.update({ heuristicsPickedOnly: true });
        await frame();
        if (drawn() !== about) {
          failures.push(`heuristic panel: picked-only shows ${drawn()} of ${about} verdicts`);
        }
        store.update({ heuristicsPickedOnly: false });
        tv.callbacks.onPickFeature(null);
        await frame();
      }

      // The not-met switch must never widen the set.
      store.update({ heuristicsFailsOnly: true });
      await frame();
      const failing = evaluated.filter((record) => record.outcome === 'FAIL').length;
      if (drawn() !== failing) {
        failures.push(`heuristic panel: not-met only shows ${drawn()} of ${failing} verdicts`);
      }
      store.update({ heuristicsFailsOnly: false });
      await frame();

      // Switching the external-input gaps back off removes exactly those.
      tv.prefs.setExternalGapsShown(false);
      await frame();
      if (drawn() !== evaluated.length - externalGaps) {
        failures.push(
          `heuristic panel: leaving out the external gaps left ${drawn()}, expected `
          + `${evaluated.length - externalGaps}`,
        );
      }
      tv.prefs.setExternalGapsShown(externalGapsWereShown);
      await frame();
    }

    results.push({
      partId,
      line,
      rawCadFaces: model.stats.rawCadFaceCount,
      meshedFaces: meshFaceIds.length,
      featureNodes: model.nodes.length,
      graphNodes: shownNodes.length,
      graphEdges: tv.graph.cy.edges().length,
      featureChecks,
      derivedChecks,
      claimedFaceChecks: claimedChecks,
      unclaimedFaceChecks: unclaimedChecks,
      mergedFaceChecks: mergedChecks,
      edgeChecks,
      witnessChecks,
      endpointVisibilityChecks,
      relationContextChecks,
      heuristicChecks,
      heuristicSquareChecks,
      failures,
    });
  }
  }

  // ---- 9b. the two lines are visually distinct, and the audit leaves the
  // page on the line it found it on.
  const lineFailures = [];
  if (accentSeen.occ === accentSeen.nx) {
    lineFailures.push(`accent token did not change across lines (${accentSeen.occ})`);
  }
  await switchLine(originalLine);
  if (window.tempVisual.line !== originalLine
      || document.documentElement.dataset.line !== originalLine) {
    lineFailures.push('audit failed to restore the original line');
  }
  results.push({ partId: '(line)', original: originalLine, accents: accentSeen, failures: lineFailures });

  // ---- 10. store: the part page has its banner slot, and the store API
  // knows every part the index lists (live freshness for each of them).
  const storeFailures = [];
  let storeChecks = 0;
  if (!document.getElementById('store-banner')) storeFailures.push('store banner element missing');
  try {
    const response = await fetch('api/parts', { cache: 'no-store' });
    if (response.ok) {
      const rows = (await response.json()).parts || [];
      const known = new Map(rows.map((row) => [row.partId, row]));
      for (const partId of partIds) {
        storeChecks += 1;
        const row = known.get(partId);
        if (!row) { storeFailures.push(`store does not know ${partId}`); continue; }
        if (!row.light || !row.light.state) storeFailures.push(`${partId}: no light state`);
        if (!row.lines || !row.lines.occ) storeFailures.push(`${partId}: no per-line state`);
        if (row.status !== 'PASS') storeFailures.push(`${partId}: listed in the index but record status is ${row.status}`);
      }
    } else {
      storeFailures.push(`api/parts HTTP ${response.status}`);
    }
  } catch (error) {
    storeFailures.push(`api/parts unreachable: ${error.message}`);
  }
  results.push({ partId: '(store)', storeChecks, failures: storeFailures });

  return results;
}

// ---- similarity API (launcher and part page) ------------------------------
//
// configs reachable; a ranking of the first part answers with rows whose
// distances lie in [0, 1] and ascend; a pair result carries the edit path
// whose per-direction sums average to the raw cost.  Run once per line since
// node 21': each line has its own corpus, representation and cache, and an
// answer on one scale says nothing about the other.
async function auditSimilarityApi(line) {
  const failures = [];
  const suffix = line ? `&line=${encodeURIComponent(line)}` : '';
  const only = line ? `?line=${encodeURIComponent(line)}` : '';
  let checks = 0;
  const get = async (path) => {
    const response = await fetch(path, { cache: 'no-store' });
    const payload = await response.json().catch(() => ({}));
    return { ok: response.ok, status: response.status, payload };
  };
  try {
    const configs = await get(`api/similarity/configs${only}`);
    checks += 1;
    if (!configs.ok) failures.push(`configs HTTP ${configs.status}`);
    else {
      if (!Array.isArray(configs.payload.configs) || !configs.payload.configs.length) failures.push('configs: no preset listed');
      if (!configs.payload.default) failures.push('configs: no default preset');
      if (!configs.payload.weights || !Object.keys(configs.payload.weights).length) failures.push('configs: no weight specs');
      if (line && configs.payload.line !== line) failures.push(`configs: answered for ${configs.payload.line}, asked ${line}`);
      if (!configs.payload.representation) failures.push('configs: no representation named');
      // The compare page folds micro edits at the run's own threshold; if
      // the answer stops carrying it the page silently falls back to a
      // number written in its own source, which is what this replaced.
      if (!(Number(configs.payload.microEditCost) >= 0)) failures.push('configs: no micro-edit threshold');
    }
    const index = await get('data/index.json');
    const ids = (index.payload.parts || []).map((part) => part.partId);
    if (ids.length >= 2) {
      const rank = await get(`api/similarity/rank?query=${encodeURIComponent(ids[0])}${suffix}`);
      checks += 1;
      if (!rank.ok) failures.push(`rank HTTP ${rank.status}: ${rank.payload.error || ''}`);
      else {
        const rows = rank.payload.ranked || [];
        if (!rows.length) failures.push('rank: no row');
        let previous = -1;
        for (const row of rows) {
          if (!(row.distance >= 0 && row.distance <= 1)) failures.push(`rank: ${row.partId} distance ${row.distance} outside [0,1]`);
          if (row.distance < previous - 1e-12) failures.push(`rank: ${row.partId} out of order`);
          previous = row.distance;
          if (!row.breakdown || typeof row.breakdown !== 'object') failures.push(`rank: ${row.partId} without breakdown`);
        }
        if (!rank.payload.config || !rank.payload.config.hash) failures.push('rank: no config hash');
        if (line && rank.payload.line !== line) failures.push(`rank: answered for ${rank.payload.line}, asked ${line}`);
      }
      const pair = await get(`api/similarity/compare?a=${encodeURIComponent(ids[0])}&b=${encodeURIComponent(ids[1])}${suffix}`);
      checks += 1;
      if (!pair.ok) failures.push(`compare HTTP ${pair.status}: ${pair.payload.error || ''}`);
      else {
        const path = pair.payload.edit_path;
        if (!path || !path.a_to_b || !path.b_to_a) failures.push('compare: edit_path missing a direction');
        else {
          const totals = ['a_to_b', 'b_to_a'].map((direction) => editPathTotal(path[direction]));
          if (Math.abs(0.5 * (totals[0] + totals[1]) - pair.payload.raw_cost) > 1e-6) {
            failures.push(`compare: mean of direction sums ${totals} != raw_cost ${pair.payload.raw_cost}`);
          }
        }
        if (!pair.payload.config || !pair.payload.config.hash) failures.push('compare: no config hash');
        if (line && pair.payload.line !== line) failures.push(`compare: answered for ${pair.payload.line}, asked ${line}`);
      }
    }
  } catch (error) {
    failures.push(`similarity API unreachable: ${error.message}`);
  }
  return { partId: `(similarity${line ? ':' + line : ''})`, similarityChecks: checks, failures };
}

function editPathTotal(block) {
  let total = 0;
  for (const value of Object.values(block || {})) {
    const list = Array.isArray(value) ? value : (value && typeof value === 'object' ? [value] : []);
    for (const entry of list) total += Number(entry.cost || 0);
  }
  return total;
}

// ---- launcher page ---------------------------------------------------------
async function auditLauncherPage() {
  const failures = [];
  let checks = 0;
  for (const id of ['view-parts', 'view-heuristics', 'view-similarity']) {
    checks += 1;
    if (!document.getElementById(id)) failures.push(`${id} missing`);
  }
  const tabs = document.querySelectorAll('#view-tabs button[data-view]');
  checks += 1;
  if (tabs.length !== 3) failures.push(`expected 3 view tabs, found ${tabs.length}`);
  const results = [{ partId: '(launcher)', launcherChecks: checks, failures }];
  results.push(await auditLauncherProviders());
  for (const line of pageLines()) results.push(await auditSimilarityApi(line));
  return results;
}

// The line switch on the launcher: every line renders its own numbers and,
// since node 21', every line answers Similarity from its own corpus - so the
// checks that used to assert the NX side was switched off now assert it is
// switched on.  Nothing here names a line: the sweep is over whatever the
// switch offers, and each pass is checked against that line's own section of
// the index.  The audit puts the reviewer's choice back at the end.
async function auditLauncherProviders() {
  const failures = [];
  let checks = 0;
  const seg = document.getElementById('line-seg');
  const buttons = [...(seg ? seg.querySelectorAll('button[data-line]') : [])];
  checks += 1;
  if (buttons.length < 2) {
    failures.push(`expected a line switch with at least two lines, found ${buttons.length}`);
    return { partId: '(line)', lineChecks: checks, failures };
  }
  const lines = buttons.map((button) => button.dataset.line);
  const labels = new Map(buttons.map((button) => [button.dataset.line, button.textContent.trim()]));
  const original = document.documentElement.dataset.line || lines[0];
  const accent = () => getComputedStyle(document.documentElement).getPropertyValue('--accent').trim();
  const index = await fetch('data/index.json', { cache: 'no-store' }).then((r) => r.json());
  const click = async (line) => {
    seg.querySelector(`button[data-line="${CSS.escape(line)}"]`).click();
    await frame();
  };

  const accents = {};
  const simTab = document.querySelector('#view-tabs button[data-view="similarity"]');
  const statsFailures = [];
  for (const line of lines) {
    await click(line);
    accents[line] = accent();
    checks += 1;
    if (!document.getElementById('run-line').textContent.includes(labels.get(line))) {
      failures.push(`footer does not name the ${line} line`);
    }
    checks += 1;
    if (simTab.disabled) failures.push(`similarity tab disabled on the ${line} line`);
    checks += 1;
    if (document.querySelector('#rows .part-row .col-actions')
        && ![...document.querySelectorAll('#rows .part-row .col-actions button')]
          .some((button) => button.textContent === 'Similar')) {
      failures.push(`row Similar action missing on the ${line} line`);
    }
    // A row this line built shows that line's own numbers.
    const sample = (index.parts || []).find(
      (part) => part.lines && part.lines[line] && part.lines[line].status === 'PASS',
    );
    if (sample) {
      checks += 1;
      const row = document.querySelector(`#rows .part-row[data-part-id="${CSS.escape(sample.partId)}"]`);
      const cells = row ? row.querySelectorAll('.col-num') : [];
      const features = cells[1] && Number(cells[1].textContent);
      if (features !== sample.lines[line].nodeCount) {
        failures.push(`${sample.partId}: features cell ${features} != ${line} nodeCount `
          + `${sample.lines[line].nodeCount}`);
      }
    }
    // The statistics view folds the shown line's own verdicts, so its numbers
    // have to move with the line and match that line's own index section.
    statsFailures.push(...await auditStatsFold(index, line));
  }
  checks += 1;
  if (new Set(Object.values(accents)).size !== lines.length) {
    failures.push('two lines share an accent token: the page cannot be told apart');
  }
  await click(original);
  failures.push(...statsFailures);
  return { partId: '(line)', lineChecks: checks + statsFailures.length + 2, accents, failures };
}

/** The heuristics view, for the line that is currently shown.
 *
 * Checked against the index's own fold for that line rather than against a
 * list of ids: the two dialects publish different heuristic sets and node 19a
 * is explicit that their rates are not one metric, so what the audit asserts
 * is that the page shows *this* line's set and *this* line's totals. */
async function auditStatsFold(index, line) {
  const failures = [];
  const tab = document.querySelector('#view-tabs button[data-view="heuristics"]');
  const previousView = document.querySelector('#view-tabs button[aria-selected="true"]');
  tab.click();
  await frame();

  const section = ((index.heuristics || {}).lines || {})[line] || {};
  const corpus = section.corpus || {};
  const shown = [...document.querySelectorAll('.h-table tbody tr:not(.open-drill) .col-id')]
    .map((cell) => cell.textContent.trim())
    .filter(Boolean);
  const expected = [...new Set([...Object.keys(section.catalogue || {}), ...Object.keys(corpus)])];
  if (!expected.length) failures.push(`${line}: index carries no heuristic fold for this line`);
  for (const hId of shown) {
    if (!expected.includes(hId)) failures.push(`${line}: rule table lists ${hId}, which this line does not publish`);
  }
  if (shown.length !== expected.length) {
    failures.push(`${line}: rule table lists ${shown.length} heuristics, the index folds ${expected.length}`);
  }
  // The matrix has one row per part this line actually built.
  const builtParts = (index.parts || []).filter(
    (part) => ((part.lines || {})[line] || {}).heuristicSummary,
  ).length;
  const matrixRows = document.querySelectorAll('.matrix tbody tr').length;
  if (builtParts && matrixRows !== builtParts) {
    failures.push(`${line}: matrix has ${matrixRows} part rows, ${builtParts} parts carry a summary`);
  }
  if (previousView) previousView.click();
  await frame();
  return failures;
}

// ---- compare page ----------------------------------------------------------
//
// Both views render their part; the edit path lists every non-micro entry of
// the shown direction; selecting an entry lights exactly the named features
// on the named sides (a delete only on the left, an insert only on the
// right); the micro switch changes what is listed, never the totals.
async function auditComparePage() {
  const tc = window.tempCompare;
  const failures = [];
  let checks = 0;
  await waitFor(() => tc.state.payload && tc.sides.a.model && tc.sides.b.model, 20000);
  await frame();
  for (const key of ['a', 'b']) {
    const side = tc.sides[key];
    const faces = Object.keys(side.viewer.debugFaces()).length;
    checks += 1;
    if (faces !== side.model.stats.rawCadFaceCount) failures.push(`view ${key}: ${faces} meshes for ${side.model.stats.rawCadFaceCount} faces`);
  }
  if (!document.querySelector('#score .score-item.main')) failures.push('score strip without the main score');

  const payload = tc.state.payload;
  const direction = tc.state.direction;
  const block = payload.edit_path[direction];
  const listedCount = () => document.querySelectorAll('#path .cmp-entry').length;
  const allEntries = Object.values(block).flatMap((value) => (Array.isArray(value) ? value : [value]));
  tc.setMicro(false);
  await frame();
  checks += 1;
  if (listedCount() !== allEntries.length) failures.push(`edit path lists ${listedCount()} of ${allEntries.length} entries with micro folding off`);
  tc.setMicro(true, 1e-3);
  await frame();
  const expectedShown = allEntries.filter((entry) => Number(entry.cost || 0) >= 1e-3).length;
  checks += 1;
  if (listedCount() !== expectedShown) failures.push(`edit path lists ${listedCount()} of ${expectedShown} entries with micro folding on`);
  const totals = document.querySelector('#path .cmp-totals');
  if (!totals || totals.classList.contains('mismatch')) failures.push('per-direction sums do not match the raw cost');

  // Highlight per side for a few entries of each kind.
  const lit = (key) => Object.entries(tc.sides[key].viewer.debugFaces())
    .filter(([, face]) => face.highlightRole === 'primary').map(([id]) => Number(id)).sort((x, y) => x - y);
  const expectedFaces = (key, ids) => {
    const out = new Set();
    for (const id of ids) {
      const node = tc.sides[key].model.nodeById.get(id);
      for (const face of (node && node.cadFaceIds) || []) out.add(face);
    }
    return [...out].sort((x, y) => x - y);
  };
  const seen = new Set();
  let highlightChecks = 0;
  for (let index = 0; index < tc.state.entries.length; index += 1) {
    const item = tc.state.entries[index];
    const kind = `${item.group.key}`;
    if (seen.has(kind) && highlightChecks > 12) continue;
    seen.add(kind);
    tc.selectEntry(index);
    await frame();
    highlightChecks += 1;
    for (const [key, ids] of [['a', item.left], ['b', item.right]]) {
      const expected = expectedFaces(key, ids);
      const actual = lit(key);
      if (expected.join(',') !== actual.join(',')) {
        failures.push(`${kind} #${index}: side ${key} lit [${actual}] expected [${expected}]`);
      }
    }
  }
  tc.selectEntry(null);
  await frame();
  return [{
    partId: `(compare ${tc.sides.a.partId} vs ${tc.sides.b.partId})`,
    compareChecks: checks,
    highlightChecks,
    entries: tc.state.entries.length,
    failures,
  }];
}

function frame() {
  // setTimeout, not requestAnimationFrame: a background/hidden tab never fires
  // rAF and the audit would hang forever.
  return new Promise((resolve) => setTimeout(resolve, 40));
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// ------------------------------------------------------------------ timing
//
//   await runTimingAudit();
//
// The two places where an answer can arrive after it stopped being the
// answer (node 26 G-09): a compare reload superseded by a second one, and a
// ranking superseded by a line switch.  Both were fixed with a request
// generation, and neither could be *tested* - the race needs a slow response,
// and every response here is fast.
//
// The fixture is the server's `?delayMs=N`, which only exists when the server
// was started with GATHERSEED_VIEWER_DEBUG_DELAY set:
//
//   $env:GATHERSEED_VIEWER_DEBUG_DELAY='1'
//   python run_pipeline.py view --port 18777 --no-browser --no-refresh-api
//
// The audit does not change the page: it wraps `fetch` for the length of one
// check so the *first* matching request is asked to be slow, drives the page
// the way a reviewer would, and puts `fetch` back.

/** Ask the server to be slow for the first request matching *pattern*. */
function slowFirst(pattern, ms) {
  const original = window.fetch;
  let used = false;
  window.fetch = function slowed(input, init) {
    const url = String(typeof input === 'string' ? input : (input && input.url) || '');
    if (!used && pattern.test(url)) {
      used = true;
      const separator = url.includes('?') ? '&' : '?';
      return original.call(this, `${url}${separator}delayMs=${ms}`, init);
    }
    return original.call(this, input, init);
  };
  return () => { window.fetch = original; return used; };
}

/** Is the delay switch on?  Timed, because an ignored parameter is silent. */
async function delaysHonoured(ms = 700) {
  const start = performance.now();
  await fetch(`api/status?delayMs=${ms}`, { cache: 'no-store' }).catch(() => null);
  return performance.now() - start > ms * 0.6;
}

window.runTimingAudit = async function runTimingAudit() {
  if (!(await delaysHonoured())) {
    return [{
      partId: '(timing)', timingChecks: 0, skipped: true,
      failures: [],
      note: 'server started without GATHERSEED_VIEWER_DEBUG_DELAY - no slow response to race with',
    }];
  }
  if (window.tempCompare) return auditCompareRace();
  if (window.tempLauncher) return auditRankRace();
  throw new Error('runTimingAudit: run it on the compare page or the launcher');
};

/**
 * Two reloads, the first one slow: the page must end up showing the part the
 * second one asked for, with a pair result computed on that part.
 */
async function auditCompareRace() {
  const tc = window.tempCompare;
  const failures = [];
  const ids = [...document.querySelectorAll('#part-a option')].map((option) => option.value);
  const back = tc.sides.a.partId;
  const other = ids.find((id) => id !== back && id !== tc.sides.b.partId);
  if (!other) return [{ partId: '(timing)', timingChecks: 0, failures: ['need three parts to race'] }];

  const select = document.getElementById('part-a');
  const restore = slowFirst(/\/data\/|api\/similarity\/compare/, 2500);
  try {
    select.value = other;
    select.dispatchEvent(new Event('change'));
    await sleep(120);
    select.value = back;
    select.dispatchEvent(new Event('change'));
    // Long enough for the slow one to come back and be dropped.
    await sleep(3500);
  } finally {
    if (!restore()) failures.push('no request was slowed - the pattern matched nothing');
  }
  await waitFor(() => tc.state.payload && tc.sides.a.model, 20000);
  if (tc.sides.a.partId !== back) failures.push(`side a settled on ${tc.sides.a.partId}, expected ${back}`);
  if (tc.sides.a.model.partId !== back) failures.push(`side a shows ${tc.sides.a.model.partId}, expected ${back}`);
  const hud = (document.getElementById('hud-a') || {}).textContent || '';
  if (!hud.includes(back)) failures.push(`hud a reads "${hud}", expected ${back}`);
  const pair = tc.state.payload || {};
  const named = [pair.a, pair.b].filter(Boolean);
  if (named.length && !named.includes(back)) failures.push(`pair result is about ${named.join('/')}, not ${back}`);
  return [{ partId: '(timing compare)', timingChecks: 4, superseded: other, failures }];
}

/**
 * A ranking overtaken by a line switch: the answer computed on the line being
 * left must not land under the line arrived at - the one comparison node 21'
 * says cannot be made.
 */
async function auditRankRace() {
  const panel = window.tempLauncher.similarity;
  const failures = [];
  const lines = pageLines();
  const startLine = window.tempLauncher.line;
  const otherLine = lines.find((id) => id !== startLine);
  if (!otherLine) return [{ partId: '(timing)', timingChecks: 0, failures: ['need two lines to race'] }];

  // The panel builds itself when its view is first opened, so the race has
  // to be run on the view the reviewer would be looking at.
  const tab = document.querySelector('#view-tabs button[data-view="similarity"]');
  if (tab) tab.click();
  await waitFor(() => panel.state.built && panel.state.specs && panel.state.query, 20000);

  const buttons = [...document.querySelectorAll('#line-seg button[data-line]')];
  const click = (id) => buttons.find((button) => button.dataset.line === id).click();
  const restore = slowFirst(/api\/similarity\/rank/, 2500);
  let landed = null;
  try {
    panel.runQuery();
    await sleep(120);
    click(otherLine);
    // Long enough for the slow ranking to come back and be dropped.
    await sleep(3500);
    landed = panel.state.result;
  } finally {
    if (!restore()) failures.push('no ranking request was slowed');
    click(startLine);
    await sleep(400);
  }
  // Read before the line is put back, because putting it back drops the
  // result too - what is being checked is what the *switch* left behind.
  if (landed && landed.line === startLine) {
    failures.push(`a ranking computed on ${startLine} survived the switch to ${otherLine}`);
  }
  if (panel.state.busy) failures.push('the panel is still busy after the race');
  return [{ partId: '(timing rank)', timingChecks: 2, leftLine: startLine, arrivedAt: otherLine,
            landedLine: landed ? landed.line : null, failures }];
}

async function waitFor(predicate, timeout = 10000) {
  const start = performance.now();
  while (!predicate()) {
    if (performance.now() - start > timeout) throw new Error('timeout waiting for condition');
    await new Promise((resolve) => setTimeout(resolve, 30));
  }
}

// Does the picture fit in its box?
//
// Everything above asks whether what is drawn agrees with what was recorded.
// That is half the question, and it is the half that keeps passing while the
// page is visibly broken: a panel squeezed to a fifth of its height still
// holds the right numbers, a slider drawn across the wrong element is still
// one of three tabs, a canvas a quarter too large for its pane still renders
// the right model.  All three shipped.  These checks read geometry only, and
// none of them knows the name of a single feature, relation or rule.
function auditGeometry() {
  const failures = [];
  let checks = 0;

  // The page scrolls down, never sideways.
  checks += 1;
  const root = document.documentElement;
  const sideways = root.scrollWidth - root.clientWidth;
  if (sideways > 1) failures.push(`the page scrolls sideways by ${sideways}px`);

  // An element that hides its overflow and is overflowed anyway is holding
  // content nobody can reach.  A one-line ellipsis is the legitimate use of
  // the same declaration, and it excludes itself by its own two properties
  // rather than by a list of class names.
  for (const element of document.querySelectorAll('*')) {
    const style = getComputedStyle(element);
    if (style.overflowY !== 'hidden' || style.display === 'none') continue;
    if (style.textOverflow === 'ellipsis' && style.whiteSpace === 'nowrap') continue;
    if (!element.clientHeight) continue;
    checks += 1;
    const cut = element.scrollHeight - element.clientHeight;
    if (cut > 2) failures.push(`${describeElement(element)} clips ${cut}px of its own content`);
  }

  // A segmented control's thumb is one segment wide, sits inside the control
  // it belongs to, and sits under the tab that is selected.  The width is
  // derived from the control's own box rather than from the stylesheet's
  // numbers, so padding and border can change without touching this.
  for (const seg of document.querySelectorAll('.segmented')) {
    const thumb = seg.querySelector('.seg-thumb');
    const buttons = [...seg.querySelectorAll('button')];
    const box = seg.getBoundingClientRect();
    if (!thumb || !buttons.length || !box.width) continue;
    const style = getComputedStyle(seg);
    const inner = box.width
      - parseFloat(style.borderLeftWidth) - parseFloat(style.borderRightWidth)
      - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
    const segments = Number(style.getPropertyValue('--segments')) || buttons.length;
    const thumbBox = thumb.getBoundingClientRect();
    const expected = inner / segments;

    checks += 3;
    if (Math.abs(thumbBox.width - expected) > 2) {
      failures.push(`${describeElement(seg)}: thumb is ${thumbBox.width.toFixed(0)}px where one of `
        + `${segments} segments is ${expected.toFixed(0)}px`);
    }
    if (thumbBox.left < box.left - 1 || thumbBox.right > box.right + 1) {
      failures.push(`${describeElement(seg)}: thumb is drawn outside its own control`);
    }
    const active = buttons[Number(seg.dataset.active) || 0];
    const activeBox = active && active.getBoundingClientRect();
    const centre = (thumbBox.left + thumbBox.right) / 2;
    if (activeBox && (centre < activeBox.left || centre > activeBox.right)) {
      failures.push(`${describeElement(seg)}: thumb is not under "${active.textContent.trim()}"`);
    }
  }

  return { partId: '(geometry)', geometryChecks: checks, failures };
}

/** Enough of an element to find it in the sheet, and no more. */
function describeElement(element) {
  const id = element.id ? `#${element.id}` : '';
  const classes = typeof element.className === 'string' && element.className.trim()
    ? '.' + element.className.trim().split(/\s+/).join('.')
    : '';
  return (element.tagName.toLowerCase() + id + classes).slice(0, 70);
}

// Second audit: does the auto-view land somewhere the feature is actually
// exposed?  "Exposed" is measured as the share of the feature's own surface
// samples whose first hit from the camera is a face of that same feature —
// the right question for concave features, where a ray aimed at the centroid
// of a bore correctly passes straight through.
window.runAutoViewAudit = async function runAutoViewAudit() {
  const results = [];
  const select = document.getElementById('part-select');
  const partIds = [...select.options].map((option) => option.value);

  for (const partId of partIds) {
    select.value = partId;
    select.dispatchEvent(new Event('change'));
    await waitFor(() => window.tempVisual.model && window.tempVisual.model.partId === partId);
    await frame();

    const tv = window.tempVisual;
    const model = tv.model;
    const baseline = tv.viewer.cameraGoal();  // the default fitted view

    let improved = 0;
    let same = 0;
    let worse = 0;
    let beforeTotal = 0;
    let afterTotal = 0;
    let zeroAfter = 0;
    const regressions = [];

    for (const node of model.nodes) {
      const samples = tv.viewer.debugSamples(node.cadFaceIds, 20);
      if (!samples.length) continue;

      const before = visibility(tv, node, samples, baseline.position);
      tv.callbacks.onPickFeature(node.nodeId); // first click: select only
      tv.callbacks.onPickFeature(node.nodeId); // second click: move the camera
      const goal = tv.viewer.cameraGoal();
      const after = visibility(tv, node, samples, goal.position);

      beforeTotal += before;
      afterTotal += after;
      if (after > before + 1e-6) improved += 1;
      else if (after < before - 1e-6) { worse += 1; regressions.push(`${node.nodeId} ${before.toFixed(2)}->${after.toFixed(2)}`); }
      else same += 1;
      if (after === 0) zeroAfter += 1;
    }
    tv.callbacks.onPickFeature(null);

    const counted = improved + same + worse;
    results.push({
      partId,
      features: counted,
      meanVisibilityDefaultView: +(beforeTotal / counted).toFixed(3),
      meanVisibilityAutoView: +(afterTotal / counted).toFixed(3),
      improved,
      unchanged: same,
      worse,
      invisibleAfter: zeroAfter,
      regressions: regressions.slice(0, 6),
    });
  }
  return results;
};

/** Share of a feature's surface samples the camera can actually see. */
function visibility(tv, node, samples, cameraPosition) {
  const owned = new Set(node.cadFaceIds);
  let seen = 0;
  for (const sample of samples) {
    const hit = tv.viewer.probeFace(cameraPosition, sample.point);
    if (hit !== null && owned.has(hit)) seen += 1;
  }
  return seen / samples.length;
}
