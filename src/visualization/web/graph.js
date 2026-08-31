// Cytoscape.js feature graph.  Every registered relation is copied from the
// generated view; the relation chips decide which edge types are drawn.

import { resolveSelection, resolveHover } from './store.js';

const GROUP_PREFIX = 'type::';
// Fallback tile geometry.  Width and column count are recomputed per part
// from its longest name (see tileGrid); these are what an empty graph and the
// first style build use.
const NODE_W = 132;
const NODE_H = 26;
const COLS = 3;
const GAP_X = 9;
const GAP_Y = 6;
const GROUP_GAP = 46;

export function createGraph(container, store, callbacks) {
  if (window.cytoscapeFcose) cytoscape.use(window.cytoscapeFcose);

  // Recomputed on every rebuild from the labels actually being drawn.
  let grid = { width: NODE_W, columns: COLS };

  const cy = cytoscape({
    container,
    style: buildStyle(grid.width),
    wheelSensitivity: 0.22,
    boxSelectionEnabled: false,
    autounselectify: true,
    elements: [],
  });

  let model = null;

  cy.on('tap', 'node[!isGroup]', (event) => callbacks.onPickFeature(event.target.id()));
  cy.on('tap', 'edge', (event) => callbacks.onPickEdge(event.target.id()));
  cy.on('tap', (event) => { if (event.target === cy) callbacks.onPickFeature(null); });
  cy.on('mouseover', 'node[!isGroup]', (event) => callbacks.onHoverFeature(event.target.id()));
  cy.on('mouseout', 'node[!isGroup]', () => callbacks.onHoverFeature(null));

  function setModel(nextModel) {
    model = nextModel;
    lastSignature = '';
    rebuild(store.get());
  }

  function visibleNodes(state) {
    const search = state.search.trim().toLowerCase();
    return model.nodes.filter((node) => {
      if (state.hiddenTypes.has(node.featureType)) return false;
      if (search && !node.nodeId.toLowerCase().includes(search)) return false;
      return true;
    });
  }

  let lastSignature = '';
  let lastStructure = '';

  function rebuild(state) {
    if (!model) return;
    const nodes = visibleNodes(state);
    const enabledEdgeTypes = [...state.visibleEdgeTypes].sort().join(',');
    // Two signatures, because these are not the same kind of change.  The
    // structure - which tiles exist, and under which layout - is what a
    // layout run answers.  The relation chips only decide which of the tiles
    // already on screen get joined up, and no tile moves for that.  Running
    // the layout for it anyway also refits, and since cy.fit() frames the
    // edges as well as the tiles, switching a chip on shifted the whole
    // viewport - and switching it off again did not shift it back.
    const structure = [state.layout, nodes.map((n) => n.nodeId).join(',')].join('|');
    const signature = [structure, enabledEdgeTypes].join('|');
    if (signature === lastSignature) return;
    const structureChanged = structure !== lastStructure;
    lastSignature = signature;
    lastStructure = structure;

    const nodeIds = new Set(nodes.map((node) => node.nodeId));
    const typesPresent = model.featureTypes
      .map((entry) => entry.type)
      .filter((type) => nodes.some((node) => node.featureType === type));

    const elements = [];
    // The group is headed by the words a person reads for the class, not by
    // the key: the key is already inside every node id in the group.
    const labelOf = new Map(model.featureTypes.map((entry) => [entry.type, entry.label]));
    for (const type of typesPresent) {
      elements.push({ data: { id: GROUP_PREFIX + type, label: labelOf.get(type) || type, isGroup: 1 } });
    }
    const tileLabels = [];
    for (const node of nodes) {
      const label = tileLabel(node);
      tileLabels.push(label);
      elements.push({
        data: {
          id: node.nodeId,
          parent: GROUP_PREFIX + node.featureType,
          label,
          color: node.color,
          faceCount: node.cadFaceIds.length,
        },
      });
    }
    // The tiles are as wide as this part's longest name and no wider, so a
    // vocabulary of long class names does not clip and one of short names
    // does not leave the grid full of empty tile.
    grid = tileGrid(tileLabels, tokens().meta);
    cy.style(buildStyle(grid.width));
    // An edge is drawn when its type is switched on *and* both of its endpoints
     // survived the feature filter.  Types combine as a union.
    const edgeElements = [];
    let edgeCount = 0;
    for (const edge of model.edges) {
      if (!state.visibleEdgeTypes.has(edge.edgeType)) continue;
      if (!nodeIds.has(edge.source) || !nodeIds.has(edge.target)) continue;
      edgeElements.push({
        data: {
          id: edge.edgeId,
          source: edge.source,
          target: edge.target,
          edgeType: edge.edgeType,
          color: edgeColor(model, edge.edgeType),
          weight: edge.sharedEdgeCount || 1,
        },
      });
      edgeCount += 1;
    }

    cy.batch(() => {
      // Tiles are only torn down when the structure changed; for a chip the
      // edges alone are swapped, so every tile keeps the position it has -
      // which is what keeps the zoom and the pan still.
      if (structureChanged) {
        cy.elements().remove();
        cy.add(elements);
      } else {
        cy.edges().remove();
      }
      cy.add(edgeElements);
    });

    if (structureChanged) runLayout(state, nodes, typesPresent);
    applyState(store.get());
    callbacks.onGraphStats({ shown: nodes.length, total: model.nodes.length, edges: edgeCount });
  }

  function runLayout(state, nodes, typesPresent) {
    if (state.layout === 'fcose' && window.cytoscapeFcose) {
      cy.layout({
        name: 'fcose',
        quality: 'proof',
        randomize: true,
        animate: false,
        nodeSeparation: 85,
        idealEdgeLength: 110,
        padding: 22,
        nodeDimensionsIncludeLabels: true,
      }).run();
    } else if (state.layout === 'concentric') {
      cy.layout({
        name: 'concentric',
        animate: false,
        padding: 22,
        concentric: (element) => element.degree() || 1,
        levelWidth: () => 2,
        minNodeSpacing: 16,
      }).run();
    } else {
      const positions = groupedPositions(nodes, typesPresent);
      cy.layout({
        name: 'preset',
        animate: false,
        padding: 22,
        positions: (element) => positions[element.id()] || undefined,
        fit: true,
      }).run();
    }
    fit();
  }

  /**
   * Fit, then align to the top when the graph is shorter than the pane: a
   * small graph reads as a list from the top, not as an island floating in
   * the middle of a tall empty canvas.
   */
  function fit() {
    // Measure against the container as it stands.  Cytoscape caches its own
    // size and re-reads it only on resize(), and the first fit of a part runs
    // while the workspace grid is still being applied: measured on opening
    // 084_818_012_solid_001, cytoscape believed it had 260x582 inside a pane
    // that was 254x422, so the framing came out 2.8% too large and the tiles
    // spilled past the right edge until somebody pressed Fit.
    cy.resize();
    cy.fit(undefined, 22);
    const box = cy.elements().boundingBox();
    const contentHeight = box.h * cy.zoom();
    const viewHeight = cy.height();
    if (contentHeight + 44 < viewHeight) {
      const pan = cy.pan();
      cy.pan({ x: pan.x, y: pan.y - ((viewHeight - contentHeight) / 2 - 22) });
    }
  }

  function groupedPositions(nodes, typesPresent) {
    const positions = {};
    const { width, columns } = grid;
    let y = 0;
    for (const type of typesPresent) {
      const members = nodes.filter((node) => node.featureType === type);
      members.forEach((node, i) => {
        positions[node.nodeId] = {
          x: (i % columns) * (width + GAP_X),
          y: y + Math.floor(i / columns) * (NODE_H + GAP_Y),
        };
      });
      y += Math.ceil(members.length / columns) * (NODE_H + GAP_Y) + GROUP_GAP;
    }
    return positions;
  }

  function applyState(state) {
    if (!model) return;
    rebuild(state);

    const selection = resolveSelection(state, model);
    const hover = resolveHover(state, model);
    const active = selection.featureIds.size > 0 || selection.edgeIds.size > 0;

    // A selected feature keeps neighbours reached through the currently
    // visible relation types legible instead of dimming them away.
    const neighbourIds = new Set();
    for (const neighbour of selection.relationNeighbourFeatureIds) neighbourIds.add(neighbour);

    cy.batch(() => {
      cy.nodes('[!isGroup]').forEach((element) => {
        const id = element.id();
        const selected = selection.featureIds.has(id);
        const neighbour = neighbourIds.has(id);
        element.toggleClass('sel', selected);
        element.toggleClass('near', !selected && neighbour);
        element.toggleClass('hover', hover.featureIds.has(id));
        element.toggleClass('dim', active && !selected && !neighbour);
      });
      cy.edges().forEach((element) => {
        const selectedEdge = selection.edgeIds.has(element.id());
        const incident = selection.featureIds.has(element.data('source'))
          || selection.featureIds.has(element.data('target'));
        element.toggleClass('sel', selectedEdge || (selection.kind !== 'edge' && incident));
        element.toggleClass('dim', active && !selectedEdge && (selection.kind === 'edge' || !incident));
      });
    });
  }

  function centerOn(featureId) {
    const element = cy.getElementById(featureId);
    if (element.empty()) return;
    // Replace the pan in flight instead of queueing behind it.  Cytoscape
    // queues animations by default, so picking features faster than one per
    // 180ms - which is what browsing a list of them looks like, and what the
    // audit does a few hundred times - builds a backlog that goes on panning
    // the graph long after the last click.  Measured after one audit run:
    // 690 distinct pan positions over six otherwise idle seconds, with the
    // window still and nothing being resized.
    cy.stop(true);
    cy.animate({ center: { eles: element }, duration: 180 });
  }

  function applyTheme() {
    cy.style(buildStyle(grid.width));
    applyState(store.get());
  }

  function resize() { cy.resize(); }

  let lastTheme = readTheme();
  store.subscribe((state) => {
    if (state.theme !== lastTheme) {
      lastTheme = state.theme;
      applyTheme();
      return;
    }
    applyState(state);
  });

  return { setModel, fit, resize, centerOn, applyTheme, cy };
}

/** "ThroughHole_003" -> "TH 003": enough to identify, short enough to read. */
/**
 * What one tile reads: the words a human uses for the class, then this
 * instance's ordinal — `Through Hole 001`, `Multi-Step (6) Blind Hole 001`.
 *
 * It used to be an initialism taken from the node id (`ThroughHole_001` ->
 * `TH 001`), which worked only because the open-source line's class names are
 * CamelCase: the rule was "delete everything that is not A–Z".  On NX's
 * UPPER_SNAKE types that rule deletes the digit NX uses to encode the segment
 * count, so `STEP1HOLE_001` and `STEP2HOLE_001` both came out as
 * `STEPHOLE 001` — two different classes, one tile label — and
 * `HOLE_FREE_SHAPED_STRAIGHT_002` came out as an unbroken 22-character run
 * that overflowed the tile.  A full label cannot collide with another class's
 * and needs no rule at all.
 *
 * The ordinal keeps its leading zeros, so the tile text and the node id line
 * up character for character; the id itself stays in the details panel, where
 * identifiers belong.
 */
// The tile says *which one*, the group heading above it says *what*.  Node
// 19' D15: printing the friendly name on every tile repeated the class name
// once per instance under a heading that already said it, and made the tiles
// wide enough to lose a column.  The node id is the primary key and cannot
// collide - unlike the abbreviation node 18' removed, which turned both
// STEP1HOLE and STEP2HOLE into "STEPHOLE".
function tileLabel(node) {
  return node.nodeId;
}

// Roughly how wide one character of the tile font is, as a fraction of the
// font size.  Only used to size the tiles to their longest label, so an
// approximation is enough — a tile that is a few pixels wide of perfect is
// invisible, a clipped name is not.
const CHAR_WIDTH_RATIO = 0.55;
const TILE_PADDING = 18;
const TILE_MIN_W = 96;
const TILE_MAX_W = 260;

/** Tile width and column count for one part's longest label. */
function tileGrid(labels, fontSize) {
  const longest = labels.reduce((max, text) => Math.max(max, text.length), 0);
  const width = Math.min(TILE_MAX_W,
    Math.max(TILE_MIN_W, Math.round(longest * fontSize * CHAR_WIDTH_RATIO) + TILE_PADDING));
  // Keep a row roughly as wide as the panel: three narrow tiles read better
  // than three that force the reader to zoom out.
  const columns = width <= 150 ? 3 : width <= 240 ? 2 : 1;
  return { width, columns };
}

function edgeColor(model, edgeType) {
  const entry = (model.relationTypes || []).find((item) => item.edgeType === edgeType);
  return (entry && entry.color) || '#888888';
}

function readTheme() {
  return document.documentElement.dataset.theme === 'light' ? 'light' : 'dark';
}

/** The stylesheet's tokens, read at build time so the canvas labels use the
 *  same families and the meta size as the rest of the page. */
function tokens() {
  const style = getComputedStyle(document.documentElement);
  const read = (name, fallback) => (style.getPropertyValue(name) || fallback).trim();
  return {
    groupBg: read('--panel-2', '#2b2a27'),
    groupText: read('--text-3', '#7a756c'),
    ring: read('--text', '#f2efe7'),
    mono: read('--mono', 'ui-monospace, Consolas, monospace'),
    sans: read('--sans', 'system-ui, sans-serif'),
    meta: parseFloat(read('--fs-meta', '11px')) || 11,
  };
}

function buildStyle(tileWidth) {
  const t = tokens();
  const width = tileWidth || NODE_W;

  return [
    {
      // A tinted tile, like the chips elsewhere: a wash of the type colour, a
      // hairline in it, the label in the page's text colour.
      selector: 'node[!isGroup]',
      style: {
        'width': width,
        'height': NODE_H,
        'shape': 'round-rectangle',
        'corner-radius': 8,
        'background-color': 'data(color)',
        'background-opacity': 0.2,
        'border-width': 1,
        'border-color': 'data(color)',
        'border-opacity': 0.75,
        'label': 'data(label)',
        'color': t.ring,
        // A name, not an identifier: the tile reads "Through Hole 001", and
        // the id it stands for is in the details panel, in mono, where the
        // page keeps identifiers.  Ellipsis is a safety net for a class name
        // longer than the widest tile, not the normal case.
        'font-family': t.sans,
        'font-size': t.meta,
        'font-weight': 600,
        'text-valign': 'center',
        'text-halign': 'center',
        'text-wrap': 'ellipsis',
        'text-max-width': width - 12,
        'transition-property': 'opacity, border-width',
        'transition-duration': '90ms',
      },
    },
    {
      selector: 'node[?isGroup]',
      style: {
        'shape': 'round-rectangle',
        'corner-radius': 12,
        'background-color': t.groupBg,
        'background-opacity': 1,
        'border-width': 0,
        'label': 'data(label)',
        'color': t.groupText,
        'font-family': t.sans,
        'font-size': t.meta,
        'text-valign': 'top',
        'text-halign': 'center',
        'text-margin-y': -3,
        'padding': 10,
      },
    },
    {
      selector: 'edge',
      style: {
        'width': 1,
        'line-color': 'data(color)',
        'curve-style': 'bezier',
        'opacity': 0.55,
        'transition-property': 'opacity, width',
        'transition-duration': '90ms',
      },
    },
    { selector: 'edge.sel', style: { 'width': 2, 'line-color': t.ring, 'opacity': 0.9 } },
    { selector: 'edge.dim', style: { 'opacity': 0.1 } },
    { selector: 'node.dim', style: { 'opacity': 0.22 } },
    { selector: 'node.near', style: { 'opacity': 0.75 } },
    { selector: 'node.hover', style: { 'border-width': 2, 'border-color': t.ring, 'border-opacity': 1, 'opacity': 1 } },
    { selector: 'node.sel', style: { 'border-width': 3, 'border-color': t.ring, 'border-opacity': 1, 'opacity': 1, 'z-index': 20 } },
  ];
}
