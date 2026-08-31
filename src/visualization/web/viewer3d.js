// three.js viewer: one Mesh per *raw CAD face*.
//
// Keeping the raw face as the drawing unit is the whole point of the
// experiment — every triangle the user can click carries the same integer id
// the STEP traversal assigned, which id_map.json already bridges to the
// PartModel face and to the feature.  No identity is re-derived in the browser.

import * as THREE from 'three';
import { OrbitControls } from '../vendor/OrbitControls.js';
import { baseFaceColor, NEUTRAL_COLOR } from './data.js';
import { resolveSelection, resolveHover, isFaceVisible } from './store.js';

/**
 * Lighting and shading are theme-dependent on purpose.
 *
 * The same absolute luminance reads much darker against a bright surround, so
 * a rig tuned on the dark background looks muddy the moment the page turns
 * light.  Two things change with the theme: the light rig gains a much
 * stronger ambient/hemisphere term (a bright room bounces light back onto the
 * part), and de-emphasised faces fade *upward* toward the background instead
 * of downward toward black.
 */
const THEMES = {
  dark: {
    ambient: 0.50,
    hemisphere: 0.45,
    hemisphereSky: 0x3a3b42,
    hemisphereGround: 0x121110,
    key: 1.90,
    fill: 0.75,
    rim: 0.45,
    neutral: '#8d97a3',
    mute: '#69707b',   // unrelated faces settle here
    muteScale: 0.62,   // ...and darken, because the surround is dark
    idleMix: 0.18,
    dimMix: 0.72,
    highlightTint: 0xffffff,
    outline: 0xffffff,
  },
  light: {
    ambient: 0.88,
    hemisphere: 1.05,
    hemisphereSky: 0xfffdf7,
    hemisphereGround: 0xd6d0c2,
    key: 1.30,
    fill: 0.42,
    rim: 0.26,
    neutral: '#cac2b2',
    mute: '#d2cbbc',   // unrelated faces settle here
    muteScale: 1.02,   // ...and stay light, so they read as "quiet", not "dark"
    idleMix: 0.14,
    dimMix: 0.68,
    highlightTint: 0x6b6357,
    outline: 0x1f1e1d,
  },
};

export function createViewer3D(container, store, callbacks) {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x0e1116, 1);
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 10000);
  camera.up.set(0, 0, 1); // CAD convention: Z up

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.14;
  controls.rotateSpeed = 0.85;

  // Three-point rig, repositioned from the camera basis every frame so the
  // part is always lit from the viewer's shoulder, plus an ambient floor and a
  // hemisphere term standing in for bounce off the surrounding page.
  const lightTarget = new THREE.Object3D();
  const keyLight = new THREE.DirectionalLight(0xffffff, 1.9);
  const fillLight = new THREE.DirectionalLight(0xbdd6f2, 0.75);
  const rimLight = new THREE.DirectionalLight(0xffffff, 0.45);
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
  const hemisphereLight = new THREE.HemisphereLight(0xffffff, 0x000000, 0.45);
  for (const light of [keyLight, fillLight, rimLight]) light.target = lightTarget;
  scene.add(lightTarget, keyLight, fillLight, rimLight, ambientLight, hemisphereLight);

  // Orientation triad, drawn as a small second pass in the corner.
  const axisScene = new THREE.Scene();
  const axisCamera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
  axisCamera.up.set(0, 0, 1);
  axisScene.add(buildAxisTriad());

  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();

  let model = null;
  let meshByFace = new Map();
  let pickables = [];
  let edgeLines = null;
  let selectedEdges = null;
  // X-ray overlay of the selected faces: the same geometry drawn once more
  // without a depth test, so a highlighted feature shows through the solid
  // from every angle (the outline used to do that job and read badly).
  let selectionOverlay = [];
  let edgePositions = null;
  let allPositions = null;
  let allNormals = null;
  let allIndices = null;
  let center = new THREE.Vector3();
  let radius = 1;
  let needsRender = true;
  let disposed = false;
  let forcedSize = null;

  // ------------------------------------------------------------ scene build

  function clear() {
    for (const mesh of meshByFace.values()) {
      mesh.geometry.dispose();
      mesh.material.dispose();
      scene.remove(mesh);
    }
    meshByFace = new Map();
    pickables = [];
    for (const lines of [edgeLines, selectedEdges]) {
      if (!lines) continue;
      scene.remove(lines);
      lines.geometry.dispose();
      lines.material.dispose();
    }
    edgeLines = null;
    selectedEdges = null;
    clearSelectionOverlay();
  }

  function clearSelectionOverlay() {
    for (const overlay of selectionOverlay) {
      scene.remove(overlay);
      overlay.material.dispose(); // geometry is shared with the face mesh
    }
    selectionOverlay = [];
  }

  function setModel(nextModel) {
    clear();
    model = nextModel;

    const buffers = model.mesh.buffers;
    const positions = new Float32Array(model.buffer, buffers.positions.byteOffset, buffers.positions.count * 3);
    const normals = new Float32Array(model.buffer, buffers.normals.byteOffset, buffers.normals.count * 3);
    const indices = new Uint32Array(model.buffer, buffers.indices.byteOffset, buffers.indices.count);
    edgePositions = new Float32Array(model.buffer, buffers.edgePositions.byteOffset, buffers.edgePositions.count * 3);
    allPositions = positions;
    allNormals = normals;
    allIndices = indices;

    for (const record of model.mesh.faces) {
      if (!record.indexCount) continue;
      const from = record.vertexOffset * 3;
      const to = from + record.vertexCount * 3;

      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(positions.subarray(from, to), 3));
      geometry.setAttribute('normal', new THREE.BufferAttribute(normals.subarray(from, to), 3));
      geometry.setIndex(new THREE.BufferAttribute(
        indices.subarray(record.indexOffset, record.indexOffset + record.indexCount), 1));
      geometry.computeBoundingSphere();

      const material = new THREE.MeshStandardMaterial({
        color: 0x8d97a3,
        metalness: 0.05,
        roughness: 0.6,
        side: THREE.DoubleSide,
        // Push faces back so the edge overlay — whose points are literally
        // these mesh vertices — draws cleanly in front instead of stitching.
        polygonOffset: true,
        polygonOffsetFactor: 2,
        polygonOffsetUnits: 2,
      });

      const mesh = new THREE.Mesh(geometry, material);
      mesh.userData.cadFaceId = record.cadFaceId;
      mesh.renderOrder = 1;
      scene.add(mesh);
      meshByFace.set(record.cadFaceId, mesh);
      pickables.push(mesh);
    }

    const edgeGeometry = new THREE.BufferGeometry();
    edgeGeometry.setAttribute('position', new THREE.BufferAttribute(edgePositions, 3));
    edgeLines = new THREE.LineSegments(
      edgeGeometry,
      new THREE.LineBasicMaterial({ color: 0x121820, depthWrite: false }),
    );
    edgeLines.renderOrder = 2;
    scene.add(edgeLines);

    const box = model.mesh.boundingBox;
    const min = new THREE.Vector3(box.min[0], box.min[1], box.min[2]);
    const max = new THREE.Vector3(box.max[0], box.max[1], box.max[2]);
    center = min.clone().add(max).multiplyScalar(0.5);
    radius = Math.max(1e-3, min.distanceTo(max) * 0.5);
    lightTarget.position.copy(center);

    camera.near = radius / 500;
    camera.far = radius * 500;
    fit();
    applyTheme(store.get().theme);
  }

  // ----------------------------------------------------------------- camera

  function fit() {
    const distance = (radius / Math.sin((camera.fov * Math.PI) / 360)) * 1.2;
    const direction = new THREE.Vector3(0.75, -1, 0.6).normalize();
    camera.position.copy(center).addScaledVector(direction, distance);
    controls.target.copy(center);
    camera.updateProjectionMatrix();
    controls.update();
    needsRender = true;
  }

  function frameFaces(cadFaceIds, viewDirection) {
    const box = new THREE.Box3();
    let found = false;
    for (const cadFaceId of cadFaceIds) {
      const mesh = meshByFace.get(cadFaceId);
      if (!mesh) continue;
      if (!mesh.geometry.boundingBox) mesh.geometry.computeBoundingBox();
      box.union(mesh.geometry.boundingBox);
      found = true;
    }
    if (!found) { fit(); return; }
    const target = box.getCenter(new THREE.Vector3());
    const localRadius = Math.max(radius * 0.03, box.getSize(new THREE.Vector3()).length() * 0.5);
    const framing = (localRadius / Math.sin((camera.fov * Math.PI) / 360)) * 2.2;
    const direction = (viewDirection ? viewDirection.clone() : camera.position.clone().sub(controls.target)).normalize();
    const distance = Math.max(framing, clearanceDistance(target, direction) + localRadius);
    glideTo(target.clone().addScaledVector(direction, distance), target);
  }

  /**
   * How far along `direction` the camera has to sit to be outside the solid.
   *
   * Framing a small feature inside a pocket or a bore can otherwise put the
   * camera below the surrounding material, where it looks at the inside of the
   * neighbouring walls instead of the feature.
   */
  function clearanceDistance(target, direction) {
    const probe = new THREE.Raycaster(target.clone(), direction.clone());
    probe.far = radius * 8;
    const hits = probe.intersectObjects(pickables, false);
    return hits.length ? hits[hits.length - 1].distance : 0;
  }

  // Camera moves are animated: a hard cut between two viewpoints costs the
  // reviewer the mental link between what they clicked and what they now see.
  let glide = null;

  function glideTo(position, target) {
    glide = {
      fromPosition: camera.position.clone(),
      toPosition: position,
      fromTarget: controls.target.clone(),
      toTarget: target,
      start: performance.now(),
      duration: 420,
    };
    needsRender = true;
  }

  function stepGlide() {
    if (!glide) return false;
    const t = Math.min(1, (performance.now() - glide.start) / glide.duration);
    const eased = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
    camera.position.lerpVectors(glide.fromPosition, glide.toPosition, eased);
    controls.target.lerpVectors(glide.fromTarget, glide.toTarget, eased);
    controls.update();
    if (t >= 1) glide = null;
    return true;
  }

  // --------------------------------------------- exposed-side camera framing

  /**
   * Move the camera to a viewpoint from which the feature is actually exposed.
   *
   * Every candidate direction is scored from the camera position it would
   * actually produce — not from a parallel ray at infinity, because at real
   * framing distance the rays fan out enough to change the answer.  For each
   * candidate: place a virtual camera, shoot a ray at each surface sample, and
   * count the sample only when the first face hit belongs to this feature.
   *
   * A fully enclosed feature scores zero everywhere and falls back to plain
   * framing from the current viewpoint.
   */
  function focusFeature(cadFaceIds, directionHints) {
    const samples = sampleFeatureSurface(cadFaceIds, 16);
    const target = featureTarget(cadFaceIds);
    if (!samples.length || !target) { frameFaces(cadFaceIds); return; }

    const owned = new Set(cadFaceIds);
    const view = currentViewDirection();
    const probe = new THREE.Raycaster();
    probe.far = radius * 40;
    const ray = new THREE.Vector3();

    let best = null;
    let bestScore = 0;
    for (const direction of candidateDirections(cadFaceIds, directionHints)) {
      const position = cameraPositionFor(target.center, target.localRadius, direction);
      let score = 0;
      for (const sample of samples) {
        ray.copy(sample.point).sub(position);
        const distance = ray.length();
        if (distance < 1e-9) continue;
        ray.divideScalar(distance);
        const facing = -sample.normal.dot(ray);
        if (facing <= 0.02) continue; // facing away from this camera
        probe.set(position, ray);
        const hits = probe.intersectObjects(pickables, false);
        if (!hits.length || !owned.has(hits[0].object.userData.cadFaceId)) continue;
        // Head-on is best, but a surface that can only be seen edge-on still
        // beats no visibility at all — that is the through-hole bore case.
        score += (0.15 + 0.85 * facing) * sample.weight;
      }
      // Small bias toward what is already on screen, so clicking through
      // neighbouring features does not spin the part around.
      score *= 1 + 0.12 * Math.max(0, direction.dot(view));
      if (score > bestScore) { bestScore = score; best = direction; }
    }

    if (!best) { frameFaces(cadFaceIds); return; }
    frameFaces(cadFaceIds, best);
  }

  /** Bounding centre and radius of a feature's own faces. */
  function featureTarget(cadFaceIds) {
    const box = new THREE.Box3();
    let found = false;
    for (const cadFaceId of cadFaceIds) {
      const mesh = meshByFace.get(cadFaceId);
      if (!mesh) continue;
      if (!mesh.geometry.boundingBox) mesh.geometry.computeBoundingBox();
      box.union(mesh.geometry.boundingBox);
      found = true;
    }
    if (!found) return null;
    return {
      center: box.getCenter(new THREE.Vector3()),
      localRadius: Math.max(radius * 0.03, box.getSize(new THREE.Vector3()).length() * 0.5),
    };
  }

  /** The exact position frameFaces() would use for this direction. */
  function cameraPositionFor(center, localRadius, direction) {
    const framing = (localRadius / Math.sin((camera.fov * Math.PI) / 360)) * 2.2;
    const distance = Math.max(framing, clearanceDistance(center, direction) + localRadius);
    return center.clone().addScaledVector(direction, distance);
  }

  function currentViewDirection() {
    return camera.position.clone().sub(controls.target).normalize();
  }

  function sampleFeatureSurface(cadFaceIds, limit) {
    const samples = [];
    if (!allPositions) return samples;

    const records = cadFaceIds
      .map((cadFaceId) => model.meshFaces.get(cadFaceId))
      .filter((record) => record && record.indexCount);
    const totalTriangles = records.reduce((sum, record) => sum + record.indexCount / 3, 0);
    if (!totalTriangles) return samples;
    const stride = Math.max(1, Math.floor(totalTriangles / limit));

    let counter = 0;
    const a = new THREE.Vector3();
    const b = new THREE.Vector3();
    const c = new THREE.Vector3();

    for (const record of records) {
      for (let i = 0; i < record.indexCount; i += 3) {
        if (counter++ % stride !== 0) continue;
        const base = record.indexOffset + i;
        const v0 = (record.vertexOffset + allIndices[base]) * 3;
        const v1 = (record.vertexOffset + allIndices[base + 1]) * 3;
        const v2 = (record.vertexOffset + allIndices[base + 2]) * 3;
        a.fromArray(allPositions, v0);
        b.fromArray(allPositions, v1);
        c.fromArray(allPositions, v2);

        const normal = new THREE.Vector3()
          .fromArray(allNormals, v0)
          .add(new THREE.Vector3().fromArray(allNormals, v1))
          .add(new THREE.Vector3().fromArray(allNormals, v2));
        if (normal.lengthSq() < 1e-12) continue;
        normal.normalize();

        const area = b.clone().sub(a).cross(c.clone().sub(a)).length() * 0.5;
        const point = a.clone().add(b).add(c).multiplyScalar(1 / 3);
        samples.push({ point, normal, weight: area });
      }
    }
    return samples;
  }

  function candidateDirections(cadFaceIds, directionHints) {
    const directions = [];

    // The feature's own mean outward normal is usually the right answer.
    const mean = new THREE.Vector3();
    for (const cadFaceId of cadFaceIds) {
      const record = model.meshFaces.get(cadFaceId);
      if (!record || !record.normal) continue;
      mean.addScaledVector(new THREE.Vector3().fromArray(record.normal), record.area || 1);
    }
    if (mean.lengthSq() > 1e-9) directions.push(mean.normalize().clone());

    // Both ends of every directional attribute the feature carries — an axis,
    // a machining direction, a surface normal.  Which one wins is decided by
    // the visibility score below, not by its name.
    for (const hint of directionHints || []) {
      const axis = new THREE.Vector3().fromArray(hint);
      if (axis.lengthSq() <= 1e-9) continue;
      axis.normalize();
      directions.push(axis.clone(), axis.clone().negate());
    }

    // Even coverage of the sphere for everything else.
    const count = 34;
    const golden = Math.PI * (3 - Math.sqrt(5));
    for (let i = 0; i < count; i += 1) {
      const z = 1 - (2 * i) / (count - 1);
      const r = Math.sqrt(Math.max(0, 1 - z * z));
      const theta = golden * i;
      directions.push(new THREE.Vector3(Math.cos(theta) * r, Math.sin(theta) * r, z));
    }
    return directions;
  }

  // ------------------------------------------------------------- appearance

  const scratch = new THREE.Color();
  const neutral = new THREE.Color(NEUTRAL_COLOR);
  const mute = new THREE.Color(NEUTRAL_COLOR);
  const tint = new THREE.Color(0xffffff);
  let palette = THEMES.dark;
  let themeIsLight = false;

  function applyTheme(theme) {
    themeIsLight = theme === 'light';
    palette = themeIsLight ? THEMES.light : THEMES.dark;

    const style = getComputedStyle(document.documentElement);
    const background = (style.getPropertyValue('--viewport') || '#1f1e1c').trim();
    const edgeColor = (style.getPropertyValue('--edge3d') || '#12110f').trim();
    renderer.setClearColor(new THREE.Color(background), 1);

    ambientLight.intensity = palette.ambient;
    hemisphereLight.intensity = palette.hemisphere;
    hemisphereLight.color.set(palette.hemisphereSky);
    hemisphereLight.groundColor.set(palette.hemisphereGround);
    keyLight.intensity = palette.key;
    fillLight.intensity = palette.fill;
    rimLight.intensity = palette.rim;

    neutral.set(palette.neutral);
    mute.set(palette.mute);
    tint.set(palette.highlightTint);

    if (edgeLines) edgeLines.material.color.set(edgeColor);
    if (selectedEdges) selectedEdges.material.color.set(palette.outline);
    applyState(store.get());
  }

  function applyState(state) {
    if (!model) return;
    const selection = resolveSelection(state, model);
    const hover = resolveHover(state, model);
    const hasSelection = selection.allFaceIds.size > 0;
    const relationNeighbourFaces = selection.relationNeighbourFaceIds || new Set();

    for (const [cadFaceId, mesh] of meshByFace) {
      const inSelection = selection.allFaceIds.has(cadFaceId);
      const inRelationNeighbour = relationNeighbourFaces.has(cadFaceId) && !inSelection;
      mesh.userData.highlightRole = 'none';
      mesh.userData.emissiveStrength = 0;

      if (state.isolate && hasSelection && !inSelection) {
        mesh.visible = false;
        continue;
      }
      mesh.visible = true;

      const typeVisible = isFaceVisible(state, model, cadFaceId);
      const isPrimary = selection.primaryFaceIds.has(cadFaceId);
      const isHovered = hover.faceIds.has(cadFaceId);

      // Selection outranks the type filter: a face the reviewer explicitly
      // picked must light up even when its feature type is filtered out.
      scratch.set(baseFaceColor(model, cadFaceId, state.hiddenTypes, inSelection, palette.neutral));
      let emissive = 0;
      let highlightRole = 'none';

      if (isPrimary) {
        emissive = themeIsLight ? 0.22 : 0.34;
        highlightRole = 'primary';
      } else if (inSelection) {
        emissive = themeIsLight ? 0.09 : 0.13;
        highlightRole = 'selected';
      } else if (inRelationNeighbour && typeVisible) {
        // Relation context is deliberately weaker than the selected feature;
        // it is useful orientation, not a second selection outline.
        scratch.lerp(tint, themeIsLight ? 0.12 : 0.09);
        emissive = themeIsLight ? 0.025 : 0.045;
        highlightRole = 'relation-neighbour';
      } else if (!typeVisible) {
        scratch.lerp(mute, 0.9).multiplyScalar(palette.muteScale);
      } else if (hasSelection) {
        scratch.lerp(mute, palette.dimMix).multiplyScalar(palette.muteScale);
      } else {
        scratch.lerp(neutral, palette.idleMix);
      }

      if (isHovered) {
        scratch.lerp(tint, 0.28);
        emissive = Math.max(emissive, themeIsLight ? 0.12 : 0.22);
      }

      mesh.material.color.copy(scratch);
      mesh.material.emissive.copy(scratch).multiplyScalar(emissive);
      mesh.userData.highlightRole = highlightRole;
      mesh.userData.emissiveStrength = emissive;
    }

    if (edgeLines) edgeLines.visible = state.showEdges;
    updateSelectionOverlay(selection, state);
    // The B-Rep outline of the selection belongs to the edge layer: it is
    // drawn only while the Edges switch is on, together with the x-ray faces.
    if (state.showEdges) updateSelectedEdges(selection); else updateSelectedEdges({ allFaceIds: new Set() });
    needsRender = true;
  }

  /**
   * Every selected face once more, on top of everything and see-through.
   * Primary faces are stronger than related / neighbour ones; the colour is
   * the face's own (already highlighted) colour so the overlay reads as
   * "this feature", not as a foreign marker.
   */
  function updateSelectionOverlay(selection, state) {
    clearSelectionOverlay();
    if (!selection.allFaceIds.size) return;
    for (const cadFaceId of selection.allFaceIds) {
      const mesh = meshByFace.get(cadFaceId);
      if (!mesh || (state.isolate && !mesh.visible)) continue;
      const primary = selection.primaryFaceIds.has(cadFaceId);
      const material = new THREE.MeshBasicMaterial({
        color: mesh.material.color.clone().lerp(tint, themeIsLight ? 0.18 : 0.28),
        transparent: true,
        opacity: primary ? (themeIsLight ? 0.42 : 0.38) : (themeIsLight ? 0.24 : 0.2),
        depthTest: false,
        depthWrite: false,
        side: THREE.DoubleSide,
      });
      const overlay = new THREE.Mesh(mesh.geometry, material);
      overlay.renderOrder = 4;
      overlay.userData.cadFaceId = cadFaceId;
      scene.add(overlay);
      selectionOverlay.push(overlay);
    }
  }

  function updateSelectedEdges(selection) {
    if (selectedEdges) {
      scene.remove(selectedEdges);
      selectedEdges.geometry.dispose();
      selectedEdges.material.dispose();
      selectedEdges = null;
    }
    if (!selection.allFaceIds.size || !edgePositions) return;

    const chunks = [];
    let total = 0;
    for (const cadFaceId of selection.allFaceIds) {
      const record = model.meshFaces.get(cadFaceId);
      if (!record || !record.edgeVertexCount) continue;
      const start = record.edgeVertexOffset * 3;
      const end = start + record.edgeVertexCount * 3;
      chunks.push(edgePositions.subarray(start, end));
      total += end - start;
    }
    if (!total) return;

    const merged = new Float32Array(total);
    let cursor = 0;
    for (const chunk of chunks) { merged.set(chunk, cursor); cursor += chunk.length; }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(merged, 3));
    selectedEdges = new THREE.LineSegments(
      geometry,
      new THREE.LineBasicMaterial({
        color: palette.outline,
        transparent: true,
        opacity: 0.9,
        depthTest: false,
      }),
    );
    selectedEdges.renderOrder = 3;
    scene.add(selectedEdges);
  }

  // ---------------------------------------------------------------- picking

  function pick(event) {
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
    const hits = raycaster.intersectObjects(pickables.filter((mesh) => mesh.visible), false);
    return hits.length ? hits[0].object.userData.cadFaceId : null;
  }

  let downAt = null;
  renderer.domElement.addEventListener('pointerdown', (event) => {
    downAt = { x: event.clientX, y: event.clientY };
    glide = null; // the user grabbing the mouse always outranks an animation
  });
  renderer.domElement.addEventListener('pointerup', (event) => {
    if (event.button !== 0 || !downAt) return;
    const moved = Math.hypot(event.clientX - downAt.x, event.clientY - downAt.y);
    downAt = null;
    if (moved > 4) return; // that drag was an orbit, not a click
    callbacks.onPickFace(pick(event));
  });
  renderer.domElement.addEventListener('dblclick', (event) => {
    const cadFaceId = pick(event);
    if (cadFaceId !== null) frameFaces([cadFaceId]);
  });

  let hoverQueued = false;
  renderer.domElement.addEventListener('pointermove', (event) => {
    if (hoverQueued) return;
    hoverQueued = true;
    requestAnimationFrame(() => {
      hoverQueued = false;
      callbacks.onHoverFace(pick(event), event);
    });
  });
  renderer.domElement.addEventListener('pointerleave', () => callbacks.onHoverFace(null, null));

  // ------------------------------------------------------------------- loop

  function resize() {
    const width = container.clientWidth;
    const height = container.clientHeight;
    if (!width || !height) return;
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    needsRender = true;
  }
  const observer = new ResizeObserver(resize);
  observer.observe(container);
  resize();

  const camDir = new THREE.Vector3();
  const camRight = new THREE.Vector3();
  const camUp = new THREE.Vector3();

  function positionLights() {
    camDir.copy(camera.position).sub(controls.target).normalize();
    camRight.crossVectors(camDir, camera.up).normalize();
    camUp.crossVectors(camRight, camDir).normalize();
    const d = radius * 6;
    keyLight.position.copy(center).addScaledVector(camDir, d).addScaledVector(camRight, d * 0.55).addScaledVector(camUp, d * 0.5);
    fillLight.position.copy(center).addScaledVector(camDir, d * 0.9).addScaledVector(camRight, -d * 0.7).addScaledVector(camUp, -d * 0.35);
    rimLight.position.copy(center).addScaledVector(camDir, -d * 0.8).addScaledVector(camUp, d * 0.6);
  }

  function tick() {
    if (disposed) return;
    requestAnimationFrame(tick);
    if (stepGlide()) needsRender = true;
    if (controls.update()) needsRender = true;
    if (!needsRender) return;
    renderFrame();
  }

  function renderFrame() {
    needsRender = false;

    positionLights();

    const width = forcedSize ? forcedSize.width : container.clientWidth;
    const height = forcedSize ? forcedSize.height : container.clientHeight;
    renderer.setScissorTest(false);
    renderer.setViewport(0, 0, width, height);
    renderer.render(scene, camera);

    const size = 74;
    renderer.setViewport(8, 8, size, size);
    renderer.setScissor(8, 8, size, size);
    renderer.setScissorTest(true);
    axisCamera.position.copy(camDir).multiplyScalar(3.4);
    axisCamera.up.copy(camera.up);
    axisCamera.lookAt(0, 0, 0);
    renderer.autoClear = false;
    renderer.clearDepth();
    renderer.render(axisScene, axisCamera);
    renderer.autoClear = true;
    renderer.setScissorTest(false);
    renderer.setViewport(0, 0, width, height);
  }
  tick();

  let lastTheme = store.get().theme;
  store.subscribe((state) => {
    if (state.theme !== lastTheme) {
      lastTheme = state.theme;
      applyTheme(state.theme);
      return;
    }
    applyState(state);
  });

  return {
    setModel,
    fit,
    frameFaces,
    focusFeature,
    applyTheme,
    invalidate() { resize(); },
    // Forces one synchronous frame; used by the audit script, since a hidden
    // tab never fires requestAnimationFrame.
    renderNow() { renderFrame(); return renderer.domElement; },
    // Where the camera is heading (or already is).  Used by the audit to check
    // that the chosen viewpoint really does see the feature.
    cameraGoal() {
      const position = glide ? glide.toPosition : camera.position;
      const target = glide ? glide.toTarget : controls.target;
      return { position: position.toArray(), target: target.toArray() };
    },
    /** Put the camera somewhere right now (no glide); used to link two views. */
    setCamera(position, target) {
      glide = null;
      camera.position.fromArray(position);
      controls.target.fromArray(target);
      controls.update();
      needsRender = true;
    },
    /** Called after every user-driven or animated camera move. */
    onCameraChange(listener) {
      controls.addEventListener('change', () => listener());
      return () => controls.removeEventListener('change', listener);
    },
    /** Centre and radius of the loaded part, for size-relative camera links. */
    bounds() {
      return model ? { center: center.toArray(), radius } : null;
    },
    /**
     * Render once at an explicit size and hand back the canvas.  A hidden tab
     * reports a zero-sized container, which would otherwise make every pixel
     * measurement meaningless.
     */
    debugRenderAt(width, height) {
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      forcedSize = { width, height };
      renderFrame();
      forcedSize = null;
      return renderer.domElement;
    },
    /** Overlay state, so the audit can check the edge layer without pixels. */
    debugOverlay() {
      return {
        canvas: [renderer.domElement.width, renderer.domElement.height],
        container: [container.clientWidth, container.clientHeight],
        edgeLines: edgeLines && {
          visible: edgeLines.visible,
          color: edgeLines.material.color.getHexString(),
          segments: edgeLines.geometry.getAttribute('position').count / 2,
        },
        selectedEdges: selectedEdges && {
          color: selectedEdges.material.color.getHexString(),
          segments: selectedEdges.geometry.getAttribute('position').count / 2,
        },
        xray: selectionOverlay.map((overlay) => overlay.userData.cadFaceId).sort((a, b) => a - b),
      };
    },
    /** Surface sample points of a feature, for visibility scoring in the audit. */
    debugSamples(cadFaceIds, limit) {
      return sampleFeatureSurface(cadFaceIds, limit || 24)
        .map((sample) => ({ point: sample.point.toArray(), normal: sample.normal.toArray() }));
    },
    /** First raw CAD face hit by a ray from `from` towards `to`. */
    probeFace(from, to) {
      const origin = new THREE.Vector3().fromArray(from);
      const direction = new THREE.Vector3().fromArray(to).sub(origin).normalize();
      const probe = new THREE.Raycaster(origin, direction);
      probe.far = radius * 40;
      const hits = probe.intersectObjects(pickables, false);
      return hits.length ? hits[0].object.userData.cadFaceId : null;
    },
    // Audit hook: current per-face appearance, so the highlight can be checked
    // programmatically instead of by eye.
    debugFaces() {
      const out = {};
      for (const [cadFaceId, mesh] of meshByFace) {
        out[cadFaceId] = {
          visible: mesh.visible,
          color: mesh.material.color.getHexString(),
          emissive: mesh.material.emissive.getHexString(),
          lit: mesh.material.emissive.getHex() !== 0,
          highlightRole: mesh.userData.highlightRole || 'none',
          emissiveStrength: mesh.userData.emissiveStrength || 0,
          emissiveLuma: (
            mesh.material.emissive.r + mesh.material.emissive.g + mesh.material.emissive.b
          ) / 3,
        };
      }
      return out;
    },
    dispose() {
      disposed = true;
      observer.disconnect();
      clear();
      renderer.dispose();
      renderer.domElement.remove();
    },
  };
}

function buildAxisTriad() {
  const group = new THREE.Group();
  const axes = [
    [new THREE.Vector3(1, 0, 0), 0xff5a5a, 'X'],
    [new THREE.Vector3(0, 1, 0), 0x62d26a, 'Y'],
    [new THREE.Vector3(0, 0, 1), 0x5aa8ff, 'Z'],
  ];
  for (const [direction, color, label] of axes) {
    const geometry = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), direction.clone()]);
    group.add(new THREE.Line(geometry, new THREE.LineBasicMaterial({ color })));

    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = 64;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#' + color.toString(16).padStart(6, '0');
    ctx.font = 'bold 46px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, 32, 34);

    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
      map: new THREE.CanvasTexture(canvas),
      transparent: true,
      depthTest: false,
    }));
    sprite.position.copy(direction).multiplyScalar(1.32);
    sprite.scale.setScalar(0.6);
    group.add(sprite);
  }
  return group;
}
