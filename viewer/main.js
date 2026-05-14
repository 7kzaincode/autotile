import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const STUD = 1.0;
const BRICK_H = 1.2;

// --- DOM ---
const $ = (id) => document.getElementById(id);
const app = $('app');
const hudFile = $('hud-file');
const hudGrid = $('hud-grid');
const hudBricks = $('hud-bricks');
const hudFloating = $('hud-floating');
const hudThumb = $('hud-thumb');
const fileInput = $('file-input');
const downloadBtn = $('download-btn');
const photoInput = $('photo-input');
const resolutionSelect = $('resolution-select');
const modelSelect = $('model-select');
const generateBtn = $('generate-btn');
const redecomposeBtn = $('redecompose-btn');
const upSelect = $('up-select');
const optTiles = $('opt-tiles');
const optSlopes = $('opt-slopes');
const optSlopeInv = $('opt-slope-inv');
const optBaseplate = $('opt-baseplate');
const optBackMode = $('opt-back-mode');
const optCluster = $('opt-cluster');
const optBlur = $('opt-blur');
const optMaxColors = $('opt-max-colors');
const optPreCluster = $('opt-pre-cluster');
const optMirror = $('opt-mirror');
const optMirrorGeom = $('opt-mirror-geom');
const optVoxelSymmetry = $('opt-voxel-symmetry');
const optVoxelHoleFill = $('opt-voxel-hole-fill');
const optVoxelSupersample = $('opt-voxel-supersample');
const optRegionColors = $('opt-region-colors');
const optHollow = $('opt-hollow');
const optDropFloaters = $('opt-drop-floaters');
const optAutoSupport = $('opt-auto-support');
const optDarkenEdges = $('opt-darken-edges');
const optAccentFeatures = $('opt-accent-features');
const optSmooth = $('opt-smooth');
const optUsePhotoPalette = $('opt-use-photo-palette');
const optPhotoPaletteSize = $('opt-photo-palette-size');
const photoPaletteOpts = $('photo-palette-opts');
const optSemanticColor = $('opt-semantic-color');
const optSemanticRegions = $('opt-semantic-regions');
const optGptVision = $('opt-gpt-vision');
const optUseSam = $('opt-use-sam');
const optBudget = $('opt-budget');

optUsePhotoPalette.addEventListener('change', () => {
  photoPaletteOpts.style.display = optUsePhotoPalette.checked ? 'block' : 'none';
});
const optRembg = $('opt-rembg');
const optStylize = $('opt-stylize');
const optStylizePreset = $('opt-stylize-preset');
const optStylizeStrength = $('opt-stylize-strength');
const stylizeOpts = $('stylize-opts');
const photoBackInput = $('photo-back-input');
const photoLeftInput = $('photo-left-input');
const photoRightInput = $('photo-right-input');

optStylize.addEventListener('change', () => {
  stylizeOpts.style.display = optStylize.checked ? 'block' : 'none';
});

// --- Subject preset — applies sensible defaults for the chosen subject type
// (must mirror server-side subject_preset.py so users can preview/edit).
const subjectSelect = document.getElementById('subject-select');

let SUBJECT_PRESETS = {
  pet: {
    resolution:'64', up:'auto', rembg:true, mirror:false, mirror_geom:false,
    voxel_symmetry:false, voxel_hole_fill:true, voxel_supersample:'0', hollow:true,
    drop_floaters:true, smooth:'2', tiles:true, slopes:false, slope_inv:false,
    accent_features:true, use_gpt_vision:true, use_sam:true,
    semantic_color:true, semantic_regions:'8',
    max_colors:'10', pre_cluster:'16', back_mode:'uv',
  },
  vehicle: {
    resolution:'64', up:'auto', rembg:true, mirror:true, mirror_geom:true,
    voxel_symmetry:true, voxel_hole_fill:true, voxel_supersample:'0', hollow:true,
    drop_floaters:true, smooth:'2', tiles:true, slopes:true, slope_inv:true,
    accent_features:false, use_gpt_vision:true, use_sam:true,
    semantic_color:true, semantic_regions:'10',
    max_colors:'14', pre_cluster:'20', back_mode:'uv',
  },
  building: {
    resolution:'64', up:'auto', rembg:true, mirror:false, mirror_geom:false,
    voxel_symmetry:false, voxel_hole_fill:true, voxel_supersample:'0', hollow:true,
    drop_floaters:true, smooth:'2', tiles:true, slopes:true, slope_inv:false,
    accent_features:false, use_gpt_vision:true, use_sam:true,
    semantic_color:true, semantic_regions:'8',
    max_colors:'10', pre_cluster:'16', back_mode:'uv',
  },
  other: {
    resolution:'64', up:'auto', rembg:true, mirror:false, mirror_geom:false,
    voxel_symmetry:false, voxel_hole_fill:true, voxel_supersample:'0', hollow:true,
    drop_floaters:true, smooth:'2', tiles:true, slopes:true, slope_inv:true,
    accent_features:false, use_gpt_vision:true, use_sam:true,
    semantic_color:true, semantic_regions:'8',
    max_colors:'10', pre_cluster:'16', back_mode:'uv',
  },
};

function smoothValueFromServer(value, fallback = '2') {
  return { none: '0', light: '2', medium: '5', heavy: '5' }[value] || fallback;
}

function normalizeServerPreset(cfg, fallback = {}) {
  return {
    resolution: String(cfg.resolution ?? fallback.resolution ?? '64'),
    up: String(cfg.up ?? fallback.up ?? 'auto'),
    rembg: Boolean(cfg.remove_bg ?? fallback.rembg ?? true),
    mirror: Boolean(cfg.mirror ?? fallback.mirror ?? false),
    mirror_geom: Boolean(cfg.mirror_geometry ?? fallback.mirror_geom ?? false),
    voxel_symmetry: Boolean(cfg.voxel_symmetry ?? fallback.voxel_symmetry ?? false),
    voxel_hole_fill: Boolean(cfg.voxel_hole_fill ?? fallback.voxel_hole_fill ?? true),
    voxel_supersample: String(cfg.voxel_supersample ?? fallback.voxel_supersample ?? '0'),
    hollow: Boolean(cfg.hollow ?? fallback.hollow ?? true),
    drop_floaters: Boolean(cfg.drop_floaters ?? fallback.drop_floaters ?? true),
    smooth: smoothValueFromServer(cfg.mesh_smoothing, fallback.smooth ?? '2'),
    tiles: Boolean(cfg.tiles ?? fallback.tiles ?? true),
    slopes: Boolean(cfg.slopes ?? fallback.slopes ?? true),
    slope_inv: Boolean(cfg.slope_inv ?? fallback.slope_inv ?? true),
    accent_features: Boolean(cfg.accent_features ?? fallback.accent_features ?? false),
    use_gpt_vision: Boolean(cfg.use_gpt_vision ?? fallback.use_gpt_vision ?? true),
    use_sam: Boolean(cfg.use_sam ?? fallback.use_sam ?? true),
    semantic_color: Boolean(cfg.semantic_color ?? fallback.semantic_color ?? true),
    semantic_regions: String(cfg.semantic_regions ?? fallback.semantic_regions ?? '8'),
    max_colors: String(cfg.max_colors ?? fallback.max_colors ?? '10'),
    pre_cluster: String(cfg.pre_cluster ?? fallback.pre_cluster ?? '16'),
    back_mode: String(cfg.back_mode ?? fallback.back_mode ?? 'uv'),
  };
}

async function hydrateSubjectPresets() {
  try {
    const res = await fetch('/api/presets');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const incoming = data.presets || {};
    const next = { ...SUBJECT_PRESETS };
    for (const [name, cfg] of Object.entries(incoming)) {
      next[name] = normalizeServerPreset(cfg, SUBJECT_PRESETS[name]);
    }
    SUBJECT_PRESETS = next;
    applySubjectPreset(subjectSelect.value);
  } catch (err) {
    console.warn('preset fetch failed; using bundled defaults', err);
  }
}

function applySubjectPreset(name) {
  const p = SUBJECT_PRESETS[name];
  if (!p) return;
  resolutionSelect.value = p.resolution;
  upSelect.value = p.up;
  optRembg.checked = p.rembg;
  optMirror.checked = p.mirror;
  optMirrorGeom.checked = p.mirror_geom;
  optVoxelSymmetry.checked = p.voxel_symmetry;
  optVoxelHoleFill.checked = p.voxel_hole_fill;
  optVoxelSupersample.value = p.voxel_supersample;
  optHollow.checked = p.hollow;
  optDropFloaters.checked = p.drop_floaters;
  optSmooth.value = p.smooth;
  optTiles.checked = p.tiles;
  optSlopes.checked = p.slopes;
  optSlopeInv.checked = p.slope_inv;
  optAccentFeatures.checked = p.accent_features;
  optGptVision.checked = p.use_gpt_vision;
  optUseSam.checked = p.use_sam;
  optSemanticColor.checked = p.semantic_color;
  optSemanticRegions.value = p.semantic_regions;
  optMaxColors.value = p.max_colors;
  optPreCluster.value = p.pre_cluster;
  optBackMode.value = p.back_mode;
  // Always turn stylize OFF — it's net-negative for color fidelity.
  optStylize.checked = false;
  stylizeOpts.style.display = 'none';
}

// Only fire setStatus on USER change (not on initial load) — at load time
// `statusEl` hasn't been initialized yet, and calling setStatus would hit
// the temporal-dead-zone and throw, breaking the rest of main.js.
subjectSelect.addEventListener('change', () => {
  applySubjectPreset(subjectSelect.value);
  setStatus(`✓ ${subjectSelect.value} preset applied`, 'ok');
});
// Apply default preset on load so the form reflects what will be sent.
applySubjectPreset(subjectSelect.value);
hydrateSubjectPresets();

// Auto-pick back_mode based on whether the AI model returns textured meshes.
// Textured models bake colors in — we should not overwrite via photo projection.
modelSelect.addEventListener('change', () => {
  // 'uv' is always the right default — it samples from the PHOTO using each
  // voxel's outward normal. Works for textured AND shape-only models.
  optBackMode.value = 'uv';
});
const statusEl = $('status');
const selectionPanel = $('selection');
const selType = $('sel-type');
const selPos = $('sel-pos');
const selColor = $('sel-color');
const palettePanel = $('palette');
const paletteGrid = $('palette-grid');
const toolbar = $('toolbar');
const returnLegoBtn = $('return-lego-btn');
const undoBtn = $('undo-btn');
const redoBtn = $('redo-btn');
const showFloatingBtn = $('show-floating-btn');
const partsToggleBtn = $('parts-toggle-btn');
const downloadLdrawBtn = $('download-ldraw-btn');
const downloadObjBtn = $('download-obj-btn');
const downloadCsvBtn = $('download-csv-btn');
const downloadBricklinkBtn = $('download-bricklink-btn');
const partsListPanel = $('parts-list');
const partsTotal = $('parts-total');
const partsUnique = $('parts-unique');
const partsCost = $('parts-cost');
const partsTime = $('parts-time');
const partsTableBody = partsListPanel.querySelector('tbody');

// --- scene ---
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf7f7f5);  // Notion canvas bg

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 5000);
camera.position.set(60, 50, 60);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
app.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 10, 0);

scene.add(new THREE.AmbientLight(0xffffff, 0.7));
const key = new THREE.DirectionalLight(0xffffff, 0.9); key.position.set(40, 60, 30); scene.add(key);
const fill = new THREE.DirectionalLight(0xffffff, 0.4); fill.position.set(-40, 20, -20); scene.add(fill);
// Light theme grid — subtle dark lines on light bg
const gridHelper = new THREE.GridHelper(80, 40, 0xc7c7c2, 0xe9e9e7); scene.add(gridHelper);

// --- editor state ---
const state = {
  sourceData: null,
  bricks: [],
  palette: [],
  paletteById: new Map(),
  gridShape: [0, 0, 0],
  brickMesh: null,
  studMesh: null,
  studToBrick: [],
  selected: new Set(),         // multi-select set of brick indices
  activeColorId: null,
  lastMeshName: null,
  lastPhotoName: null,
  lastFacePhotoName: null,
  lastOtherSidePhotoName: null,
  lastBodyView: null,
  lastOtherSideView: null,
  lastLegoLabel: null,
  rawMeshObject: null,
  floatingIdx: new Set(),      // brick indices with no support below
  showFloating: false,
  history: [],                 // undo stack: arrays of bricks
  future: [],                  // redo stack
  _cameraFramed: false,
};

const HIGHLIGHT = new THREE.Color(0xffffff);
const FLOATING  = new THREE.Color(0xff4040);

// Piece-kind taxonomy — drives geometry, height, stud rendering.
// Heights are in units of brick-height (BRICK_H); plates/tiles are 1/3,
// cones / cheese_slope are 2/3.
function pieceProfile(b) {
  const kind = b.kind || 'brick';
  if (kind === 'plate' || kind === 'tile' || kind === 'round_plate' || kind === 'round_tile')
    return { heightFrac: 1/3, hasStuds: kind !== 'tile' && kind !== 'round_tile' };
  if (kind === 'cone' || kind === 'cheese_slope')
    return { heightFrac: 2/3, hasStuds: false };
  if (kind === 'dome')
    return { heightFrac: 1.0, hasStuds: false };
  // brick, plate ish (full-height), slope, slope_inv, round_brick
  return { heightFrac: 1.0, hasStuds: kind !== 'slope' && kind !== 'slope_inv' };
}

function geometryForKind(kind) {
  // Each returns a unit geometry centered at origin with size 1x1x1 nominal.
  // We then scale per-instance via matrix.
  if (kind === 'round_brick' || kind === 'round_plate' || kind === 'round_tile') {
    const g = new THREE.CylinderGeometry(0.48, 0.48, 1, 24);
    return g;
  }
  if (kind === 'cone') {
    const g = new THREE.ConeGeometry(0.48, 1, 24);
    return g;
  }
  if (kind === 'dome') {
    // Half-sphere with flat bottom (radius 0.5)
    const g = new THREE.SphereGeometry(0.5, 24, 12, 0, Math.PI * 2, 0, Math.PI / 2);
    g.translate(0, -0.5, 0);  // origin at base
    return g;
  }
  if (kind === 'cheese_slope') {
    // Triangular wedge: square base, slopes up to a thin edge on one side
    const verts = new Float32Array([
      -0.5,-0.5,-0.5,   0.5,-0.5,-0.5,   0.5, 0.5,-0.5,
      -0.5,-0.5,-0.5,   0.5, 0.5,-0.5,  -0.5,-0.5, 0.5,
       0.5,-0.5,-0.5,   0.5,-0.5, 0.5,   0.5, 0.5,-0.5,
       0.5,-0.5, 0.5,  -0.5,-0.5, 0.5,   0.5, 0.5,-0.5,
      -0.5,-0.5,-0.5,  -0.5,-0.5, 0.5,   0.5,-0.5,-0.5,
      -0.5,-0.5, 0.5,   0.5,-0.5, 0.5,   0.5,-0.5,-0.5,
    ]);
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(verts, 3));
    g.computeVertexNormals();
    return g;
  }
  // brick / plate / tile / slope / slope_inv → regular cube; height handled via scale
  return new THREE.BoxGeometry(1, 1, 1);
}

function hasStuds(b) {
  if (isFaceMounted(b)) return false;
  return pieceProfile(b).hasStuds;
}

function isFaceMounted(b) {
  return b && ['-y', '+y', '-x', '+x'].includes(b.mount || b.face_axis);
}

// --- history (undo/redo) ---

function snapshot() {
  // Deep-copy bricks (each brick is a flat dict — JSON dance is fine)
  state.history.push(JSON.stringify(state.bricks));
  if (state.history.length > 80) state.history.shift();
  state.future.length = 0;
  refreshUndoButtons();
}

function undo() {
  if (state.history.length === 0) return;
  state.future.push(JSON.stringify(state.bricks));
  state.bricks = JSON.parse(state.history.pop());
  state.selected.clear();
  buildModel();
  computeFloating();
  refreshAllPanels();
}

function redo() {
  if (state.future.length === 0) return;
  state.history.push(JSON.stringify(state.bricks));
  state.bricks = JSON.parse(state.future.pop());
  state.selected.clear();
  buildModel();
  computeFloating();
  refreshAllPanels();
}

function refreshUndoButtons() {
  undoBtn.disabled = state.history.length === 0;
  redoBtn.disabled = state.future.length === 0;
}

// --- model build ---

function clearModel() {
  // Old single-mesh path
  for (const m of [state.brickMesh, state.studMesh]) {
    if (!m) continue;
    scene.remove(m);
    m.geometry.dispose();
    m.material.dispose();
  }
  state.brickMesh = null;
  state.studMesh = null;
  state.studToBrick = [];
  // Per-kind instanced meshes
  if (state.kindMeshes) {
    for (const m of Object.values(state.kindMeshes)) {
      scene.remove(m);
      m.geometry.dispose();
      m.material.dispose();
    }
  }
  state.kindMeshes = {};
  // Reverse map: (brickIdx) → {mesh, instanceId} so selection can find the right one
  state.brickToInstance = [];
}

function clearRawMeshPreview() {
  if (!state.rawMeshObject) return;
  scene.remove(state.rawMeshObject);
  state.rawMeshObject.traverse(obj => {
    if (obj.geometry) obj.geometry.dispose();
    if (obj.material) {
      if (Array.isArray(obj.material)) obj.material.forEach(m => m.dispose());
      else obj.material.dispose();
    }
  });
  state.rawMeshObject = null;
  if (returnLegoBtn) {
    returnLegoBtn.hidden = true;
    returnLegoBtn.disabled = true;
  }
}

function frameObject(obj) {
  const box = new THREE.Box3().setFromObject(obj);
  if (box.isEmpty()) return;
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z, 1e-6);
  const scale = 46 / maxDim;
  obj.scale.multiplyScalar(scale);

  const scaledBox = new THREE.Box3().setFromObject(obj);
  const scaledCenter = scaledBox.getCenter(new THREE.Vector3());
  const minY = scaledBox.min.y;
  obj.position.x -= scaledCenter.x;
  obj.position.z -= scaledCenter.z;
  obj.position.y -= minY;

  const finalBox = new THREE.Box3().setFromObject(obj);
  const finalSize = finalBox.getSize(new THREE.Vector3());
  const height = Math.max(finalSize.y, 1);
  controls.target.set(0, height / 2, 0);
  camera.position.set(Math.max(finalSize.x, finalSize.z) * 1.25 + 12,
                      height * 0.8 + 10,
                      Math.max(finalSize.x, finalSize.z) * 1.25 + 12);
  controls.update();
}

async function loadRawMeshPreview(url, label = 'raw mesh') {
  if (!url) return;
  setStatus(`<span class="spinner"></span>loading raw mesh preview…`);
  clearModel();
  clearRawMeshPreview();
  state.selected.clear();
  state.floatingIdx.clear();
  hudFile.textContent = label;
  if (state.bricks.length && returnLegoBtn) {
    returnLegoBtn.hidden = false;
    returnLegoBtn.disabled = false;
    toolbar.classList.add('visible');
  }
  const cleanUrl = (url || '').split('?')[0].toLowerCase();
  const isGltf = cleanUrl.endsWith('.glb') || cleanUrl.endsWith('.gltf');
  hudGrid.textContent = isGltf ? 'raw GLB' : 'raw OBJ';
  hudBricks.textContent = '—';
  hudFloating.textContent = '—';

  const loader = isGltf ? new GLTFLoader() : new OBJLoader();
  loader.load(
    url,
    (loaded) => {
      const obj = isGltf ? loaded.scene : loaded;
      const mat = new THREE.MeshLambertMaterial({
        color: 0xd8d8d4,
        side: THREE.DoubleSide,
      });
      obj.traverse(child => {
        if (!child.isMesh) return;
        child.material = mat;
        child.castShadow = false;
        child.receiveShadow = false;
      });
      state.rawMeshObject = obj;
      scene.add(obj);
      frameObject(obj);
      setStatus(`✓ raw mesh preview loaded`, 'ok');
    },
    undefined,
    (err) => {
      console.error(err);
      setStatus(`mesh preview failed: ${err?.message || err}`, 'err');
    },
  );
}

function buildModel() {
  clearRawMeshPreview();
  clearModel();
  const bricks = state.bricks;
  const palette = state.paletteById;

  // 1) Group bricks by kind so we can build one InstancedMesh per kind with
  //    the right geometry (cube / cylinder / cone / dome / wedge).
  const byKind = new Map();
  bricks.forEach((b, i) => {
    const kind = b.kind || 'brick';
    if (!byKind.has(kind)) byKind.set(kind, []);
    byKind.get(kind).push({ b, i });
  });

  // 2) Pre-count studs across all stud-having bricks
  const studCount = bricks.reduce(
    (acc, b) => acc + (hasStuds(b) ? b.size_x * b.size_y : 0),
    0
  );
  const studGeom = new THREE.CylinderGeometry(0.3, 0.3, 0.18, 12);
  const studMat = new THREE.MeshLambertMaterial();
  const studMesh = new THREE.InstancedMesh(studGeom, studMat, Math.max(studCount, 1));

  const m = new THREE.Matrix4();
  const color = new THREE.Color();
  const gs = state.gridShape;
  const offX = (gs[0] * STUD) / 2;
  const offZ = (gs[1] * STUD) / 2;
  const faceMountQuat = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, 0));

  state.studToBrick = new Array(studCount);
  state.brickToInstance = new Array(bricks.length);
  let studIdx = 0;

  // 3) For each kind, build one InstancedMesh with the right geometry
  for (const [kind, group] of byKind) {
    const geom = geometryForKind(kind);
    const mat  = new THREE.MeshLambertMaterial();
    const mesh = new THREE.InstancedMesh(geom, mat, group.length);
    mesh.userData = { kind };
    let inst = 0;
    for (const { b, i } of group) {
      const rgb = palette.get(b.color) || [180, 180, 180];
      color.setRGB(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255);

      const sx = b.size_x * STUD;
      const sz = b.size_y * STUD;
      if (isFaceMounted(b)) {
        const mount = b.mount || b.face_axis || '-y';
        const faceWidth = Math.max(STUD, b.size_x * STUD);
        const faceHeight = Math.max(STUD, b.size_y * BRICK_H);
        const faceThickness = 0.18;
        let pos;
        let scale;
        if (mount === '-x' || mount === '+x') {
          const depthWidth = Math.max(STUD, b.size_x * STUD);
          const px = mount === '-x'
            ? b.x * STUD - offX - faceThickness / 2 - 0.03
            : (b.x + 1) * STUD - offX + faceThickness / 2 + 0.03;
          pos = new THREE.Vector3(
            px,
            b.z * BRICK_H + faceHeight / 2,
            b.y * STUD + depthWidth / 2 - offZ,
          );
          scale = new THREE.Vector3(faceThickness, faceHeight, depthWidth);
        } else {
          const py = mount === '-y'
            ? b.y * STUD - offZ - faceThickness / 2 - 0.03
            : (b.y + 1) * STUD - offZ + faceThickness / 2 + 0.03;
          pos = new THREE.Vector3(
            b.x * STUD + faceWidth / 2 - offX,
            b.z * BRICK_H + faceHeight / 2,
            py,
          );
          scale = new THREE.Vector3(faceWidth, faceHeight, faceThickness);
        }
        m.compose(pos, faceMountQuat, scale);
        mesh.setMatrixAt(inst, m);
        mesh.setColorAt(inst, color);
        state.brickToInstance[i] = { mesh, instanceId: inst };
        inst++;
        continue;
      }
      const { heightFrac } = pieceProfile(b);
      const sy = BRICK_H * heightFrac;
      const cx = b.x * STUD + sx / 2 - offX;
      const cz = b.y * STUD + sz / 2 - offZ;
      const baseY = (b.z === -1) ? -sy : (b.z * BRICK_H);
      const cy = baseY + sy / 2;

      m.makeScale(sx, sy, sz);
      m.setPosition(cx, cy, cz);
      mesh.setMatrixAt(inst, m);
      mesh.setColorAt(inst, color);
      state.brickToInstance[i] = { mesh, instanceId: inst };
      inst++;

      // Studs on top (only stud-having kinds)
      if (!hasStuds(b)) continue;
      for (let dx = 0; dx < b.size_x; dx++) {
        for (let dz = 0; dz < b.size_y; dz++) {
          const stx = b.x * STUD + (dx + 0.5) * STUD - offX;
          const stz = b.y * STUD + (dz + 0.5) * STUD - offZ;
          const sty = baseY + sy + 0.09;
          m.makeScale(1, 1, 1);
          m.setPosition(stx, sty, stz);
          studMesh.setMatrixAt(studIdx, m);
          studMesh.setColorAt(studIdx, color);
          state.studToBrick[studIdx] = i;
          studIdx++;
        }
      }
    }
    mesh.count = group.length;
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    scene.add(mesh);
    state.kindMeshes[kind] = mesh;
  }

  studMesh.count = studCount;
  studMesh.instanceMatrix.needsUpdate = true;
  if (studMesh.instanceColor) studMesh.instanceColor.needsUpdate = true;
  scene.add(studMesh);
  state.studMesh = studMesh;

  const heightDim = gs[2] * BRICK_H;
  const maxDim = Math.max(gs[0], gs[1]) * STUD;
  controls.target.set(0, heightDim / 2, 0);
  if (!state._cameraFramed) {
    camera.position.set(maxDim * 1.4, heightDim * 1.2 + 10, maxDim * 1.4);
    state._cameraFramed = true;
  }
  controls.update();

  hudGrid.textContent = `${gs[0]}×${gs[1]}×${gs[2]}`;
  hudBricks.textContent = bricks.length.toLocaleString();
  applySelectionTint();
  applyFloatingTint();
}

// --- selection / highlighting / coloring ---

function setBrickInstanceColor(brickIdx, color) {
  const ref = state.brickToInstance[brickIdx];
  if (!ref) return;
  ref.mesh.setColorAt(ref.instanceId, color);
  ref.mesh.instanceColor.needsUpdate = true;
}

function applySelectionTint() {
  if (!state.kindMeshes || Object.keys(state.kindMeshes).length === 0) return;
  const color = new THREE.Color();
  state.bricks.forEach((b, i) => {
    if (state.selected.has(i)) {
      setBrickInstanceColor(i, HIGHLIGHT);
    } else {
      const rgb = state.paletteById.get(b.color) || [180, 180, 180];
      color.setRGB(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255);
      setBrickInstanceColor(i, color);
    }
  });
}

function applyFloatingTint() {
  if (!state.showFloating || !state.kindMeshes) return;
  for (const i of state.floatingIdx) {
    if (state.selected.has(i)) continue;  // selection wins
    setBrickInstanceColor(i, FLOATING);
  }
}

function setBrickColor(brickIdx, colorId) {
  const b = state.bricks[brickIdx];
  if (!b) return;
  b.color = colorId;
  const rgb = state.paletteById.get(colorId) || [180, 180, 180];
  const color = new THREE.Color(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255);
  setBrickInstanceColor(brickIdx, color);
  for (let s = 0; s < state.studToBrick.length; s++) {
    if (state.studToBrick[s] === brickIdx) state.studMesh.setColorAt(s, color);
  }
  if (state.studMesh.instanceColor) state.studMesh.instanceColor.needsUpdate = true;
}

function deleteSelected() {
  if (state.selected.size === 0) return;
  snapshot();
  // Remove in reverse to keep indices valid
  const sorted = [...state.selected].sort((a, b) => b - a);
  for (const i of sorted) state.bricks.splice(i, 1);
  state.selected.clear();
  buildModel();
  computeFloating();
  refreshAllPanels();
}

// --- floating bricks (no support beneath) ---

function computeFloating() {
  state.floatingIdx.clear();
  // Build a quick occupancy index keyed on (x, y, z) -> brickIdx
  const occ = new Map();
  const key = (x, y, z) => `${x},${y},${z}`;
  state.bricks.forEach((b, i) => {
    for (let dx = 0; dx < b.size_x; dx++) {
      for (let dy = 0; dy < b.size_y; dy++) {
        occ.set(key(b.x + dx, b.y + dy, b.z), i);
      }
    }
  });
  state.bricks.forEach((b, i) => {
    if (b.z <= 0) return; // bottom layer/baseplate is grounded
    let supported = false;
    for (let dx = 0; dx < b.size_x && !supported; dx++) {
      for (let dy = 0; dy < b.size_y && !supported; dy++) {
        if (occ.has(key(b.x + dx, b.y + dy, b.z - 1))) supported = true;
      }
    }
    if (!supported) state.floatingIdx.add(i);
  });
  hudFloating.textContent = state.floatingIdx.size.toLocaleString();
  if (state.showFloating) applySelectionTint(), applyFloatingTint();
}

// --- raycasting / pick ---

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
let pointerDown = null;
let pointerShift = false;

renderer.domElement.addEventListener('pointerdown', (e) => {
  pointerDown = { x: e.clientX, y: e.clientY };
  pointerShift = e.shiftKey;
});
renderer.domElement.addEventListener('pointerup', (e) => {
  if (!pointerDown) return;
  const moved = Math.hypot(e.clientX - pointerDown.x, e.clientY - pointerDown.y);
  pointerDown = null;
  if (moved > 4) return;
  pickAt(e.clientX, e.clientY, pointerShift || e.shiftKey);
});

function pickAt(cx, cy, multi) {
  if (!state.kindMeshes || Object.keys(state.kindMeshes).length === 0) return;
  pointer.x = (cx / window.innerWidth) * 2 - 1;
  pointer.y = -(cy / window.innerHeight) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const candidates = [...Object.values(state.kindMeshes), state.studMesh].filter(Boolean);
  const hits = raycaster.intersectObjects(candidates, false);
  if (hits.length === 0) {
    if (!multi) clearSelection();
    return;
  }
  const hit = hits[0];
  let brickIdx = -1;
  if (hit.object === state.studMesh) {
    brickIdx = state.studToBrick[hit.instanceId];
  } else {
    // Find which brick this (mesh, instanceId) corresponds to
    for (let i = 0; i < state.brickToInstance.length; i++) {
      const ref = state.brickToInstance[i];
      if (ref && ref.mesh === hit.object && ref.instanceId === hit.instanceId) {
        brickIdx = i;
        break;
      }
    }
  }
  if (brickIdx < 0) return;

  if (multi) {
    if (state.selected.has(brickIdx)) state.selected.delete(brickIdx);
    else state.selected.add(brickIdx);
  } else {
    state.selected.clear();
    state.selected.add(brickIdx);
  }
  applySelectionTint();
  applyFloatingTint();
  refreshSelectionPanel();
}

function clearSelection() {
  if (state.selected.size === 0) return;
  state.selected.clear();
  applySelectionTint();
  applyFloatingTint();
  refreshSelectionPanel();
}

// --- panels ---

function refreshAllPanels() {
  refreshSelectionPanel();
  refreshPaletteSwatches();
  refreshUndoButtons();
  if (partsListPanel.classList.contains('visible')) refreshPartsList();
}

function refreshSelectionPanel() {
  if (state.selected.size === 0) {
    selectionPanel.classList.remove('visible');
    palettePanel.classList.remove('visible');
    return;
  }
  selectionPanel.classList.add('visible');
  palettePanel.classList.add('visible');
  if (state.selected.size === 1) {
    const idx = [...state.selected][0];
    const b = state.bricks[idx];
    if (!b) return;
    selType.textContent = b.brick_type;
    selPos.textContent = `(${b.x}, ${b.y}, ${b.z})`;
    const name = state.palette.find(p => p.id === b.color)?.name || 'unknown';
    selColor.textContent = name;
    refreshPaletteSwatches(b.color);
  } else {
    selType.textContent = `${state.selected.size} bricks`;
    selPos.textContent = '—';
    const colors = new Set([...state.selected].map(i => state.bricks[i]?.color));
    selColor.textContent = colors.size === 1
      ? (state.palette.find(p => p.id === [...colors][0])?.name || 'unknown')
      : 'mixed';
    refreshPaletteSwatches(colors.size === 1 ? [...colors][0] : null);
  }
}

function refreshPaletteSwatches(activeId) {
  paletteGrid.querySelectorAll('.swatch').forEach(el => {
    el.classList.toggle('active', activeId != null && Number(el.dataset.colorId) === activeId);
  });
}

function buildPaletteUI() {
  paletteGrid.innerHTML = '';
  for (const p of state.palette) {
    const sw = document.createElement('div');
    sw.className = 'swatch';
    sw.style.background = `rgb(${p.rgb.join(',')})`;
    sw.title = p.name;
    sw.dataset.colorId = p.id;
    sw.addEventListener('click', () => {
      if (state.selected.size === 0) return;
      snapshot();
      for (const idx of state.selected) setBrickColor(idx, p.id);
      // keep selection tint
      for (const idx of state.selected) setBrickInstanceColor(idx, HIGHLIGHT);
      refreshPaletteSwatches(p.id);
      if (partsListPanel.classList.contains('visible')) refreshPartsList();
    });
    paletteGrid.appendChild(sw);
  }
}

// --- parts list ---

function localPartsList() {
  const counts = new Map();
  for (const b of state.bricks) {
    const k = `${b.kind || 'brick'}|${b.brick_type}|${b.color}`;
    counts.set(k, (counts.get(k) || 0) + 1);
  }
  const rows = [];
  for (const [k, qty] of counts) {
    const [kind, bt, cid] = k.split('|');
    const colorId = Number(cid);
    const pal = state.palette.find(p => p.id === colorId);
    rows.push({ kind, brick_type: bt, color_id: colorId, color_name: pal?.name || 'unknown',
                rgb: pal?.rgb || [180, 180, 180], qty });
  }
  rows.sort((a, b) => b.qty - a.qty);
  return rows;
}

// Per-brick-type price hints, mirrored from build_stats.py for instant UI feedback.
const PRICE_HINT = {
  '1x1':0.05,'1x2':0.07,'1x3':0.10,'1x4':0.12,
  '1x6':0.18,'1x8':0.25,'1x10':0.30,'1x12':0.40,'1x16':0.60,
  '2x2':0.10,'2x3':0.15,'2x4':0.20,
  '2x6':0.30,'2x8':0.40,'2x10':0.55,
  '4x4':0.30,'4x6':0.50,'4x8':0.75,'4x10':1.00,
};
const KIND_PRICE_MULTIPLIER = {
  brick: 1.0, plate: 0.6, tile: 0.7, slope: 1.3, slope_inv: 1.4,
};
const SEC_PER_BRICK = 15;

function refreshPartsList() {
  const rows = localPartsList();
  partsTotal.textContent = state.bricks.length.toLocaleString();
  partsUnique.textContent = rows.length.toLocaleString();
  let cost = 0;
  for (const b of state.bricks) {
    cost += (PRICE_HINT[b.brick_type] || 0.10) * (KIND_PRICE_MULTIPLIER[b.kind || 'brick'] || 1.0);
  }
  const minutes = Math.round((state.bricks.length * SEC_PER_BRICK) / 60);
  partsCost.textContent = '$' + cost.toFixed(2);
  partsTime.textContent = minutes >= 60 ? `${Math.floor(minutes/60)}h${minutes%60}m` : `${minutes}m`;
  partsTableBody.innerHTML = '';
  for (const r of rows) {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${r.kind} ${r.brick_type}</td>
      <td><span class="color-dot" style="background:rgb(${r.rgb.join(',')})"></span>${r.color_name}</td>
      <td class="qty">${r.qty}</td>`;
    partsTableBody.appendChild(tr);
  }
}

// --- keyboard ---
window.addEventListener('keydown', (e) => {
  const meta = e.metaKey || e.ctrlKey;
  if (meta && e.key.toLowerCase() === 'z' && !e.shiftKey) { e.preventDefault(); undo(); return; }
  if (meta && ((e.key.toLowerCase() === 'z' && e.shiftKey) || e.key.toLowerCase() === 'y')) {
    e.preventDefault(); redo(); return;
  }
  if (e.key === 'Escape') clearSelection();
  else if (e.key === 'Delete' || e.key === 'Backspace') {
    if (state.selected.size > 0) { e.preventDefault(); deleteSelected(); }
  }
});

// --- toolbar ---
undoBtn.addEventListener('click', undo);
redoBtn.addEventListener('click', redo);

showFloatingBtn.addEventListener('click', () => {
  state.showFloating = !state.showFloating;
  showFloatingBtn.classList.toggle('active', state.showFloating);
  applySelectionTint();
  applyFloatingTint();
});

partsToggleBtn.addEventListener('click', () => {
  const visible = partsListPanel.classList.toggle('visible');
  partsToggleBtn.classList.toggle('active', visible);
  if (visible) refreshPartsList();
});

// --- downloads (server-side) ---

async function postPayload(url) {
  const payload = { ...state.sourceData, bricks: state.bricks };
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
  return res;
}

async function downloadBlob(res, filename) {
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}

downloadLdrawBtn.addEventListener('click', async () => {
  try { await downloadBlob(await postPayload('/api/ldraw'), 'model.ldr'); }
  catch (err) { setStatus(`ldraw export failed: ${err.message}`, 'err'); }
});
downloadObjBtn.addEventListener('click', async () => {
  try { await downloadBlob(await postPayload('/api/obj'), 'model.zip'); }
  catch (err) { setStatus(`obj export failed: ${err.message}`, 'err'); }
});
downloadCsvBtn.addEventListener('click', async () => {
  try { await downloadBlob(await postPayload('/api/parts-list?format=csv'), 'parts.csv'); }
  catch (err) { setStatus(`csv export failed: ${err.message}`, 'err'); }
});
downloadBricklinkBtn.addEventListener('click', async () => {
  try { await downloadBlob(await postPayload('/api/parts-list?format=bricklink-xml'), 'bricklink_wanted.xml'); }
  catch (err) { setStatus(`bricklink xml export failed: ${err.message}`, 'err'); }
});

// --- photo upload / generation ---

function setStatus(msg, kind = '') {
  statusEl.className = kind;
  statusEl.innerHTML = msg;
}

// Pretty labels for known pipeline stages — what to show the user when
// the server publishes "▶ [stage-name] …" via SSE.
const STAGE_PRETTY = {
  'upload':           'Uploading photo',
  'rembg':            'Removing background',
  'pose-analysis':    'Reading pet pose',
  'mesh-input':       'Preparing mesh input',
  'stylize':          'Stylizing photo',
  'photo-to-mesh':    'Generating 3D mesh (this can take 1–5 min)',
  'load-mesh':        'Loading mesh',
  'voxelize':         'Voxelizing mesh',
  'gpt-vision':       'Asking GPT-4o-mini about colors',
  'project-photo':    'Projecting photo onto voxels',
  'mirror-geometry':  'Mirroring geometry',
  'hollow':           'Hollowing interior',
  'mirror-colors':    'Mirroring colors',
  'region-color':     'Clustering color regions',
  'sam-semantic':     'Running SAM 2 segmentation',
  'gpt-semantic':     'Painting GPT bbox regions',
  'kmeans-semantic':  'Running k-means segmentation',
  'photo-palette':    'Extracting photo palette',
  'pet-color-map':    'Cleaning pet coat colors',
  'quantize':         'Snapping to LEGO palette',
  'mirror-palette':   'Mirroring palette IDs',
  'decompose':        'Decomposing into bricks',
  'postprocess':      'Post-processing (tiles, slopes)',
  'accent-features':  'Placing eyes / nose / mouth',
};

// Live-update the form status from the SSE log stream. The most recent
// "▶ [stage]" line drives the status text — actual server progress, not
// a synthetic timer.
let _pipelineRunning = false;
function setStageFromLog(msg) {
  if (!_pipelineRunning) return;
  const m = msg.match(/^▶\s*\[([\w-]+)\]/);
  if (!m) return;
  const pretty = STAGE_PRETTY[m[1]] || m[1];
  setStatus(`<span class="spinner"></span>${pretty}…`);
}
function startStageTicker() {
  _pipelineRunning = true;
  setStatus(`<span class="spinner"></span>Starting…`);
}
function stopStageTicker() { _pipelineRunning = false; }

generateBtn.addEventListener('click', async () => {
  const photo = photoInput.files[0];
  if (!photo) { setStatus('pick a front face photo first', 'err'); return; }
  const isPet = subjectSelect.value === 'pet';
  const sidePhoto = photoLeftInput.files[0] || photoRightInput.files[0];
  if (isPet && !sidePhoto) {
    setStatus('pets need a front face photo plus at least one side full-body photo', 'err');
    return;
  }
  generateBtn.disabled = true;
  startStageTicker();
  // Show the input photo as a thumbnail in the HUD so the user can see what they uploaded
  const reader = new FileReader();
  reader.onload = () => { hudThumb.src = reader.result; hudThumb.style.display = 'block'; };
  reader.readAsDataURL(photo);

  const fd = new FormData();
  fd.append('photo', photo);
  if (photoBackInput.files[0])  fd.append('photo_back',  photoBackInput.files[0]);
  if (photoLeftInput.files[0])  fd.append('photo_left',  photoLeftInput.files[0]);
  if (photoRightInput.files[0]) fd.append('photo_right', photoRightInput.files[0]);
  fd.append('subject', subjectSelect.value);
  fd.append('preset_client_applied', 'true');
  fd.append('remove_bg', String(optRembg.checked));
  fd.append('stylize', String(optStylize.checked));
  fd.append('stylize_preset', optStylizePreset.value);
  fd.append('stylize_strength', optStylizeStrength.value);
  fd.append('resolution', resolutionSelect.value);
  fd.append('model', modelSelect.value);
  fd.append('up', upSelect.value);
  fd.append('tiles', String(optTiles.checked));
  fd.append('slopes', String(optSlopes.checked));
  fd.append('slope_inv', String(optSlopeInv.checked));
  fd.append('baseplate', String(optBaseplate.checked));
  fd.append('back_mode', optBackMode.value);
  fd.append('cluster_colors', optCluster.value);
  fd.append('blur_radius', optBlur.value);
  fd.append('max_colors', optMaxColors.value);
  fd.append('pre_cluster', optPreCluster.value);
  fd.append('mirror', String(optMirror.checked));
  fd.append('mirror_geometry', String(optMirrorGeom.checked));
  fd.append('voxel_symmetry', String(optVoxelSymmetry.checked));
  fd.append('voxel_hole_fill', String(optVoxelHoleFill.checked));
  fd.append('voxel_supersample', optVoxelSupersample.value);
  fd.append('region_colors', optRegionColors.value);
  fd.append('hollow', String(optHollow.checked));
  fd.append('drop_floaters', String(optDropFloaters.checked));
  fd.append('auto_support', String(optAutoSupport.checked));
  fd.append('darken_edges', String(optDarkenEdges.checked));
  fd.append('accent_features', String(optAccentFeatures.checked));
  fd.append('smooth_iterations', optSmooth.value);
  fd.append('use_photo_palette', String(optUsePhotoPalette.checked));
  fd.append('photo_palette_size', optPhotoPaletteSize.value);
  fd.append('semantic_color', String(optSemanticColor.checked));
  fd.append('semantic_regions', optSemanticRegions.value);
  fd.append('use_gpt_vision', String(optGptVision.checked));
  fd.append('use_sam', String(optUseSam.checked));
  fd.append('budget', optBudget.value);

  try {
    const res = await fetch('/api/generate', { method: 'POST', body: fd });
    if (!res.ok) throw new Error(await res.text() || `HTTP ${res.status}`);
    const data = await res.json();
    applyData(data, photo.name);
    state.lastMeshName = data._mesh_name || null;
    state.lastPhotoName = data._photo_name || null;
    state.lastFacePhotoName = data._face_photo_name || null;
    state.lastOtherSidePhotoName = data._other_side_photo_name || null;
    state.lastBodyView = data._body_view || null;
    state.lastOtherSideView = data._other_side_view || null;
    redecomposeBtn.disabled = !state.lastMeshName;
    setStatus(`✓ generated ${data.bricks.length} bricks`, 'ok');
  } catch (err) {
    console.error(err);
    setStatus(`error: ${err.message}`, 'err');
  } finally {
    stopStageTicker();
    generateBtn.disabled = false;
  }
});

redecomposeBtn.addEventListener('click', async () => {
  if (!state.lastMeshName) return;
  redecomposeBtn.disabled = true;
  setStatus(`<span class="spinner"></span>re-decomposing mesh (up=${upSelect.value}, res=${resolutionSelect.value})…`);
  const fd = new FormData();
  fd.append('mesh_name', state.lastMeshName);
  fd.append('subject', subjectSelect.value);
  fd.append('preset_client_applied', 'true');
  fd.append('resolution', resolutionSelect.value);
  fd.append('up', upSelect.value);
  fd.append('tiles', String(optTiles.checked));
  fd.append('slopes', String(optSlopes.checked));
  fd.append('slope_inv', String(optSlopeInv.checked));
  fd.append('baseplate', String(optBaseplate.checked));
  fd.append('back_mode', optBackMode.value);
  fd.append('cluster_colors', optCluster.value);
  fd.append('blur_radius', optBlur.value);
  fd.append('max_colors', optMaxColors.value);
  fd.append('pre_cluster', optPreCluster.value);
  fd.append('mirror', String(optMirror.checked));
  fd.append('mirror_geometry', String(optMirrorGeom.checked));
  fd.append('voxel_symmetry', String(optVoxelSymmetry.checked));
  fd.append('voxel_hole_fill', String(optVoxelHoleFill.checked));
  fd.append('voxel_supersample', optVoxelSupersample.value);
  fd.append('region_colors', optRegionColors.value);
  fd.append('hollow', String(optHollow.checked));
  fd.append('drop_floaters', String(optDropFloaters.checked));
  fd.append('auto_support', String(optAutoSupport.checked));
  fd.append('darken_edges', String(optDarkenEdges.checked));
  fd.append('accent_features', String(optAccentFeatures.checked));
  fd.append('smooth_iterations', optSmooth.value);
  fd.append('use_photo_palette', String(optUsePhotoPalette.checked));
  fd.append('photo_palette_size', optPhotoPaletteSize.value);
  fd.append('semantic_color', String(optSemanticColor.checked));
  fd.append('semantic_regions', optSemanticRegions.value);
  fd.append('use_gpt_vision', String(optGptVision.checked));
  fd.append('use_sam', String(optUseSam.checked));
  fd.append('budget', optBudget.value);
  if (state.lastPhotoName) fd.append('photo_name', state.lastPhotoName);
  if (state.lastFacePhotoName) fd.append('face_photo_name', state.lastFacePhotoName);
  if (state.lastOtherSidePhotoName) fd.append('other_side_photo_name', state.lastOtherSidePhotoName);
  if (state.lastBodyView) fd.append('body_view', state.lastBodyView);
  if (state.lastOtherSideView) fd.append('other_side_view', state.lastOtherSideView);
  try {
    const res = await fetch('/api/redecompose', { method: 'POST', body: fd });
    if (!res.ok) throw new Error(await res.text() || `HTTP ${res.status}`);
    const data = await res.json();
    applyData(data, state.lastMeshName);
    setStatus(`✓ rebuilt with ${data.bricks.length} bricks`, 'ok');
  } catch (err) {
    console.error(err);
    setStatus(`error: ${err.message}`, 'err');
  } finally {
    redecomposeBtn.disabled = false;
  }
});

// --- load / save ---

function applyData(data, label) {
  state.sourceData = data;
  state.bricks = data.bricks.map(b => ({ ...b }));
  state.palette = data.palette;
  state.paletteById = new Map(data.palette.map(p => [p.id, p.rgb]));
  state.gridShape = data.grid_shape;
  state.selected.clear();
  state.history = [];
  state.future = [];
  state._cameraFramed = false;
  state.lastMeshName = data._mesh_name || state.lastMeshName;
  state.lastPhotoName = data._photo_name || state.lastPhotoName;
  state.lastFacePhotoName = data._face_photo_name || state.lastFacePhotoName;
  state.lastOtherSidePhotoName = data._other_side_photo_name || state.lastOtherSidePhotoName;
  state.lastBodyView = data._body_view || state.lastBodyView;
  state.lastOtherSideView = data._other_side_view || state.lastOtherSideView;
  buildModel();
  buildPaletteUI();
  computeFloating();
  refreshAllPanels();
  hudFile.textContent = label;
  state.lastLegoLabel = label;
  toolbar.classList.add('visible');
  if (state.lastMeshName) redecomposeBtn.disabled = false;
}

returnLegoBtn.addEventListener('click', () => {
  if (!state.bricks.length) return;
  buildModel();
  computeFloating();
  refreshAllPanels();
  hudFile.textContent = state.lastLegoLabel || hudFile.textContent || 'LEGO model';
  setStatus('✓ LEGO brick view restored', 'ok');
});

function loadFromUrl(url, label) {
  fetch(url)
    .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
    .then(data => applyData(data, label))
    .catch(err => { hudFile.textContent = `error: ${err.message}`; console.error(err); });
}

fileInput.addEventListener('change', (e) => {
  const f = e.target.files[0];
  if (!f) return;
  const reader = new FileReader();
  reader.onload = () => {
    try { applyData(JSON.parse(reader.result), f.name); }
    catch (err) { hudFile.textContent = `error: ${err.message}`; }
  };
  reader.readAsText(f);
});

downloadBtn.addEventListener('click', () => {
  if (!state.sourceData) return;
  const payload = { ...state.sourceData, bricks: state.bricks };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = (hudFile.textContent || 'bricks').replace(/\.json$/, '') + '.edited.json';
  a.click();
  URL.revokeObjectURL(url);
});

// --- loop ---

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();

hudFile.textContent = 'no model loaded — upload a photo →';

// --- Dev bar: live logs, cache, run-status ──────────────────────────────────

(function initDevBar() {
  const devbar = $('devbar');
  const logList = $('log-list');
  const logLevel = $('log-level');
  const logFilter = $('log-filter');
  const logAutoscroll = $('log-autoscroll');
  const logClearBtn = $('log-clear-btn');
  const cachePane = $('cache-pane');
  const statusPane = $('status-pane');
  const pipelineChip = $('devbar-pipeline-chip');
  const gptChip = $('devbar-gpt-chip');
  const meshChip = $('devbar-mesh-chip');
  const recentChip = $('devbar-recent');
  if (!devbar) return;

  function toggle() {
    if (devbar.classList.contains('collapsed')) {
      devbar.classList.remove('collapsed');
      devbar.classList.add('expanded');
      document.body.classList.add('devbar-open');
    } else {
      devbar.classList.add('collapsed');
      devbar.classList.remove('expanded');
      document.body.classList.remove('devbar-open');
    }
  }
  window.__toggleDevbar = toggle;
  // Robust click binding (replaces the inline onclick that was unreliable)
  const devbarHeader = document.getElementById('devbar-header');
  if (devbarHeader) {
    devbarHeader.addEventListener('click', (e) => {
      // Don't toggle when clicking a chip or interactive element inside
      if (e.target.closest('button, input, select')) return;
      toggle();
    });
  }

  document.querySelectorAll('.devbar-tab').forEach(t => {
    t.addEventListener('click', (e) => {
      e.stopPropagation();
      const which = t.dataset.tab;
      document.querySelectorAll('.devbar-tab').forEach(x => x.classList.toggle('active', x === t));
      document.querySelectorAll('.devbar-pane').forEach(p => {
        p.classList.toggle('active', p.dataset.tab === which);
      });
      if (which === 'cache')  refreshCache();
      if (which === 'status') refreshStatus();
    });
  });

  // --- Log rendering ---
  function fmtTs(ts) {
    const d = new Date(ts * 1000);
    return d.toTimeString().slice(0, 8);
  }
  function rowVisible(row) {
    const lvl = logLevel.value;
    if (lvl && row.dataset.level !== lvl) return false;
    const q = logFilter.value.trim().toLowerCase();
    if (q && !row.textContent.toLowerCase().includes(q)) return false;
    return true;
  }
  function applyFilters() {
    logList.querySelectorAll('.log-row').forEach(r => {
      r.style.display = rowVisible(r) ? '' : 'none';
    });
  }
  function appendLog(entry) {
    const row = document.createElement('div');
    row.className = `log-row ${entry.level || 'info'}`;
    row.dataset.level = entry.level || 'info';
    row.innerHTML = `<span class="ts">${fmtTs(entry.ts)}</span>${escapeHtml(entry.msg)}`;
    logList.appendChild(row);
    if (!rowVisible(row)) row.style.display = 'none';
    // Keep DOM bounded
    while (logList.children.length > 800) logList.removeChild(logList.firstChild);
    if (logAutoscroll.checked) logList.scrollTop = logList.scrollHeight;
    // Update header chip from last meaningful line
    if (entry.level === 'error') {
      pipelineChip.textContent = 'error';
      pipelineChip.className = 'chip err';
    } else if (entry.msg.startsWith('[subject]') || entry.msg.includes('[gpt]') || entry.msg.includes('[hunyuan')) {
      pipelineChip.textContent = 'running';
      pipelineChip.className = 'chip warn';
    }
    recentChip.textContent = entry.msg.slice(0, 60);
  }
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[c]));
  }

  logLevel.addEventListener('change', applyFilters);
  logFilter.addEventListener('input', applyFilters);
  logClearBtn.addEventListener('click', async () => {
    await fetch('/api/logs', { method: 'DELETE' });
    logList.innerHTML = '';
  });

  // --- SSE stream (auto-reconnect on drop) ---
  let es = null;
  function connect() {
    try { es && es.close(); } catch (_) {}
    es = new EventSource('/api/logs/stream');
    es.onmessage = (e) => {
      try {
        const entry = JSON.parse(e.data);
        appendLog(entry);
        setStageFromLog(entry.msg);  // form-status mirrors actual server stage
      } catch (_) {}
    };
    es.onerror = () => {
      // Browsers auto-retry; force reconnect after 3s if it stays broken.
      setTimeout(connect, 3000);
      try { es.close(); } catch (_) {}
    };
  }
  connect();

  // --- Cache pane ---
  async function refreshCache() {
    try {
      const r = await fetch('/api/cache'); const data = await r.json();
      cachePane.innerHTML = '';
      const totalFiles = ['gpt_cache', 'meshes', 'outputs', 'photos']
        .reduce((sum, key) => sum + (data[key]?.count || 0), 0);
      const toolbar = document.createElement('div');
      toolbar.style.cssText = 'display:flex;align-items:center;gap:12px;margin:0 0 12px';
      toolbar.innerHTML = `
        <div style="flex:1;color:var(--text-dim);font-size:12px">
          ${totalFiles} cached file${totalFiles === 1 ? '' : 's'}
        </div>`;
      const clearAllBtn = document.createElement('button');
      clearAllBtn.className = 'btn small danger';
      clearAllBtn.style.width = 'auto';
      clearAllBtn.style.margin = '0';
      clearAllBtn.textContent = 'clear all';
      clearAllBtn.disabled = totalFiles === 0;
      clearAllBtn.addEventListener('click', async () => {
        clearAllBtn.disabled = true;
        clearAllBtn.textContent = 'clearing…';
        try {
          const res = await fetch('/api/cache', { method: 'DELETE' });
          if (!res.ok) throw new Error(await res.text() || `HTTP ${res.status}`);
          refreshCache();
          updateChips();
        } catch (err) {
          cachePane.textContent = `clear all failed: ${err.message || err}`;
        }
      });
      toolbar.appendChild(clearAllBtn);
      cachePane.appendChild(toolbar);
      const order = [
        ['gpt_cache', 'GPT cache', 'gpt',
         'OpenAI vision results cached by photo hash. Safe to clear — will re-call API on next run.'],
        ['meshes', '3D meshes', 'meshes',
         'AI-generated meshes from TRELLIS/Hunyuan. Safe to clear — will regenerate on next photo.'],
        ['outputs', 'Outputs (brick JSON)', 'outputs',
         'Generated brick JSON files. Safe to clear.'],
        ['photos', 'Uploaded photos', 'photos',
         'Photos you uploaded (including bg-removed and stylized versions). Safe to clear.'],
      ];
      for (const [key, label, kind, desc] of order) {
        const c = data[key] || {};
        const card = document.createElement('div');
        card.className = 'cache-card';
        const latest = (c.latest || []).slice(-3).join(', ') || 'Empty';
        card.innerHTML = `
          <div class="name">${label}</div>
          <div class="stats">
            ${c.count || 0} files · ${(c.size_mb || 0).toFixed(2)} MB
            <div class="hint" style="font-size:11px;margin-top:2px">${desc}</div>
            <div class="hint" style="font-size:11px;opacity:0.5">latest: ${escapeHtml(latest)}</div>
          </div>`;
        const btn = document.createElement('button');
        btn.className = 'btn small danger'; btn.style.width = 'auto'; btn.style.margin = '0';
        btn.textContent = 'clear';
        btn.addEventListener('click', async () => {
          btn.disabled = true;
          btn.textContent = 'clearing…';
          try {
            const res = await fetch(`/api/cache/${kind}`, { method: 'DELETE' });
            if (!res.ok) throw new Error(await res.text() || `HTTP ${res.status}`);
            refreshCache();
            updateChips();
          } catch (err) {
            btn.disabled = false;
            btn.textContent = 'clear';
            cachePane.textContent = `clear ${label} failed: ${err.message || err}`;
          }
        });
        card.appendChild(btn);
        cachePane.appendChild(card);
      }
    } catch (e) {
      cachePane.textContent = `cache fetch failed: ${e}`;
    }
  }

  async function refreshStatus() {
    try {
      const r = await fetch('/api/status'); const data = await r.json();
      const run = data.run || {};
      const arts = run.artifacts || [];
      let artsHtml = '';
      if (arts.length) {
        artsHtml = `
          <h4 style="margin-top:16px;font-size:12px;color:var(--text-emph);font-weight:600">Pipeline artifacts</h4>
          <div style="display:flex;flex-direction:column;gap:8px;margin-top:8px">`;
        for (const a of arts) {
          const label = escapeHtml(a.label || 'artifact');
          const url = a.url || '';
          const safeUrl = escapeHtml(url);
          if (a.kind === 'image') {
            artsHtml += `
              <div style="display:flex;align-items:center;gap:12px;padding:8px;background:var(--bg-sidebar);border-radius:6px">
                <img src="${safeUrl}" alt="${label}" style="max-width:64px;max-height:64px;border-radius:4px;border:1px solid var(--border)" />
                <div style="flex:1">
                  <div style="font-weight:500;color:var(--text-emph)">${label}</div>
                  <a href="${safeUrl}" target="_blank" style="font-size:11px;color:var(--accent);text-decoration:none">open ↗</a>
                </div>
              </div>`;
          } else if (a.kind === 'mesh') {
            artsHtml += `
              <div style="display:flex;align-items:center;gap:12px;padding:8px;background:var(--bg-sidebar);border-radius:6px">
                <div style="width:64px;height:64px;background:var(--bg-input);border-radius:4px;display:flex;align-items:center;justify-content:center;color:var(--text-faint)">3D</div>
                <div style="flex:1">
                  <div style="font-weight:500;color:var(--text-emph)">${label}</div>
                  <button class="btn small" data-preview-mesh="${safeUrl}" data-preview-label="${label}" style="margin:6px 8px 0 0">Preview</button>
                  <a href="${safeUrl}" target="_blank" style="font-size:11px;color:var(--accent);text-decoration:none">download .obj ↓</a>
                </div>
              </div>`;
          } else {
            artsHtml += `
              <div style="padding:8px;background:var(--bg-sidebar);border-radius:6px">
                <div style="font-weight:500;color:var(--text-emph)">${label}</div>
                <a href="${safeUrl}" target="_blank" style="font-size:11px;color:var(--accent);text-decoration:none">${safeUrl} ↗</a>
              </div>`;
          }
        }
        artsHtml += `</div>`;
      }
      statusPane.innerHTML = `
        <div><b>status:</b> ${escapeHtml(run.status || 'idle')}</div>
        <div><b>body photo:</b> ${escapeHtml(run.photo || run.side_photo || '—')}</div>
        <div><b>face photo:</b> ${escapeHtml(run.face_photo || '—')}</div>
        <div><b>subject:</b> ${escapeHtml(run.subject || '—')}</div>
        <div><b>pose:</b> ${escapeHtml(run.pose || '—')}</div>
        <div><b>semantic paint:</b> ${escapeHtml(run.semantic_paint_mode || '—')}</div>
        <div><b>resolution:</b> ${escapeHtml(run.resolution || '—')}</div>
        <div><b>chosen up:</b> ${escapeHtml(run.chosen_up || '—')}</div>
        <div><b>photo front:</b> ${escapeHtml(run.front_axis || '—')}</div>
        <div><b>voxel shape:</b> ${escapeHtml((run.voxel_shape || []).join?.('×') || '—')}</div>
        <div><b>voxel AA:</b> ${escapeHtml(run.voxel_supersample || '—')}x</div>
        <div><b>bricks generated:</b> ${escapeHtml(run.bricks ?? '—')}</div>
        <div><b>used GPT vision:</b> ${run.used_gpt ? 'yes' : 'no'}</div>
        <div><b>stage:</b> ${escapeHtml(run.stage || '—')}</div>
        <div><b>last update:</b> ${run.updated_ts ? new Date(run.updated_ts * 1000).toLocaleTimeString() : '—'}</div>
        <div><b>log entries:</b> ${escapeHtml(data.log_count)}</div>
        ${artsHtml}`;
      statusPane.querySelectorAll('[data-preview-mesh]').forEach(btn => {
        btn.addEventListener('click', () => {
          loadRawMeshPreview(btn.dataset.previewMesh, btn.dataset.previewLabel || 'raw mesh');
        });
      });
    } catch (e) {
      statusPane.textContent = `status fetch failed: ${e}`;
    }
  }

  async function updateChips() {
    try {
      const r = await fetch('/api/cache'); const data = await r.json();
      gptChip.textContent = `gpt: ${data.gpt_cache?.count ?? '—'}`;
      meshChip.textContent = `meshes: ${data.meshes?.count ?? '—'}`;
    } catch (_) {}
    try {
      const r = await fetch('/api/status'); const d = await r.json();
      const s = d.run?.status || 'idle';
      pipelineChip.textContent = s;
      pipelineChip.className = 'chip ' + (s === 'done' ? 'green' : s === 'error' ? 'err' : s === 'idle' ? '' : 'warn');
    } catch (_) {}
  }

  updateChips();
  setInterval(updateChips, 4000);

  // Auto-refresh the Run tab when it's visible AND a pipeline is running,
  // so newly-recorded artifacts appear within ~2s of being created.
  setInterval(() => {
    const runActive = document.querySelector(
      '.devbar-pane.active[data-tab="status"]');
    if (runActive && _pipelineRunning) refreshStatus();
  }, 2000);
})();
