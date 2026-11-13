// X-ray Mirror
// Camera -> MediaPipe face + hand tracking -> anatomy (BodyParts3D) pinned to the head with three.js.
// Everything runs on the device. Nothing is uploaded.

import * as THREE from './lib/three.module.min.js';
import { FaceLandmarker, HandLandmarker } from './lib/vision_bundle.mjs';

const $ = (id) => document.getElementById(id);
const here = (p) => new URL(p, import.meta.url).href;
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const params = new URLSearchParams(location.search);

/* ------------------------------------------------------------------ */
/* Settings                                                            */
/* ------------------------------------------------------------------ */

const STORE_KEY = 'xray-mirror-v1';
const DEFAULTS = {
  layers: { skull: true, brain: true, muscles: true, eyes: true, outline: true, hands: true },
  labels: false,
  look: 'xray',
  brightness: 0.3,
  through: 0.55,
  feather: 0.17,
  reveal: 'lens-hand',
  lens: 0.5,
  tracking: false,
  fit: { size: 1, width: 1, height: 1, depth: 1, x: 0, y: 0, z: 0 },
};
const S = loadSettings();

function loadSettings() {
  const s = structuredClone(DEFAULTS);
  try {
    const saved = JSON.parse(localStorage.getItem(STORE_KEY) || 'null');
    if (saved) {
      Object.assign(s, saved);
      s.layers = Object.assign({}, DEFAULTS.layers, saved.layers);
      s.fit = Object.assign({}, DEFAULTS.fit, saved.fit);
    }
  } catch (e) { /* storage may be unavailable */ }
  return s;
}
let saveTimer = 0;
function saveSettings() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    try { localStorage.setItem(STORE_KEY, JSON.stringify(S)); } catch (e) { /* ignore */ }
  }, 250);
}

/* ------------------------------------------------------------------ */
/* Renderer, scenes, materials                                         */
/* ------------------------------------------------------------------ */

const stage = $('stage'), frame = $('frame'), video = $('video'), still = $('still');
const hud = $('hud'), hctx = hud.getContext('2d');

const VFOV = 63; // MediaPipe's face geometry assumes this vertical field of view
const FOCAL = 1 / Math.tan(THREE.MathUtils.degToRad(VFOV / 2));

const renderer = new THREE.WebGLRenderer({ canvas: $('gl'), alpha: true, antialias: false, premultipliedAlpha: true, powerPreference: 'high-performance' });
renderer.autoClear = false;
renderer.setClearColor(0x000000, 0);
const camera = new THREE.PerspectiveCamera(VFOV, 4 / 3, 4, 600); // units: centimetres
let rt = new THREE.WebGLRenderTarget(4, 4, { samples: 4, depthBuffer: true });

const sceneFill = new THREE.Scene();
const sceneMain = new THREE.Scene();
const sceneOut = new THREE.Scene();
const orthoCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

const VERT = /* glsl */`
  varying vec3 vN; varying vec3 vV; varying vec3 vP;
  void main(){
    vP = position;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    vN = normalMatrix * normal;
    vV = -mv.xyz;
    gl_Position = projectionMatrix * mv;
  }`;

// X-ray: additive edge glow, no depth test, so every structure shines through the others.
const FRAG_XRAY = /* glsl */`
  uniform vec3 uColor; uniform float uGain;
  varying vec3 vN; varying vec3 vV;
  void main(){
    float ndv = abs(dot(normalize(vN), normalize(vV)));
    float f = 1.0 - ndv;
    float i = 0.018 + 0.982 * pow(f, 3.0);
    gl_FragColor = vec4(uColor * i * uGain, 0.0);
  }`;

// Solid: simple lit surface, written premultiplied so the whole model can be faded as one.
const FRAG_SOLID = /* glsl */`
  uniform vec3 uColor; uniform float uAlpha; uniform float uBright;
  uniform vec3 uEyeL; uniform vec3 uEyeR; uniform float uIris;
  varying vec3 vN; varying vec3 vV; varying vec3 vP;
  void main(){
    vec3 n = normalize(vN); vec3 v = normalize(vV);
    if (dot(n, v) < 0.0) n = -n;
    vec3 L = normalize(vec3(0.35, 0.6, 0.72));
    float d = max(dot(n, L), 0.0);
    float ndv = max(dot(n, v), 0.0);
    vec3 base = uColor;
    if (uIris > 0.5) {
      vec3 c = vP.x > 0.0 ? uEyeL : uEyeR;
      vec3 e = vP - c;
      float r = length(e.xz);
      if (e.y < 0.0) base = mix(vec3(0.03), mix(vec3(0.32, 0.2, 0.12), base, smoothstep(6.0, 6.6, r)), smoothstep(2.4, 2.9, r));
    }
    vec3 col = base * (0.26 + 0.60 * d + 0.24 * ndv);
    col += 0.10 * pow(max(dot(reflect(-L, n), v), 0.0), 22.0);
    col *= uBright;
    gl_FragColor = vec4(col * uAlpha, uAlpha);
  }`;

// Head fill: darkens the real face inside the head silhouette and draws the blue outline.
const FRAG_FILL = /* glsl */`
  uniform vec3 uFill; uniform float uFillA; uniform vec3 uRim; uniform float uRimGain;
  varying vec3 vN; varying vec3 vV; varying vec3 vP;
  void main(){
    float ndv = max(dot(normalize(vN), normalize(vV)), 0.0);
    float neck = smoothstep(1430.0, 1466.0, vP.z);
    float edge = smoothstep(0.03, 0.42, ndv);
    float a = uFillA * edge * neck;
    vec3 rim = uRim * pow(1.0 - ndv, 2.2) * uRimGain * neck;
    gl_FragColor = vec4(uFill * a + rim, a);
  }`;

const FRAG_OUT = /* glsl */`
  uniform sampler2D tMap; uniform vec2 uRes; uniform vec3 uLens; uniform float uFeather; uniform float uMode; uniform vec3 uHalf;
  varying vec2 vUv;
  void main(){
    vec4 c = texture2D(tMap, vUv);
    vec2 p = vUv * uRes;
    float m = 1.0;
    if (uMode > 0.5 && uMode < 1.5) {
      float d = distance(p, uLens.xy);
      m = 1.0 - smoothstep(uLens.z * (1.0 - uFeather) - 1.0, uLens.z, d);
    } else if (uMode > 1.5) {
      float w = max(uFeather * uRes.y * 0.2, 1.5);
      m = smoothstep(-w, w, dot(p, uHalf.xy) - uHalf.z);
    }
    gl_FragColor = c * m;
  }`;

const LAYER_STYLE = {
  skull:   { xray: [0.78, 0.89, 1.00], gain: 0.30, solid: [0.91, 0.87, 0.78] },
  brain:   { xray: [1.00, 0.46, 0.64], gain: 0.20, solid: [0.86, 0.60, 0.64] },
  muscles: { xray: [1.00, 0.28, 0.30], gain: 0.16, solid: [0.70, 0.24, 0.22] },
  eyes:    { xray: [1.00, 0.92, 0.62], gain: 0.55, solid: [0.95, 0.94, 0.90] },
  hands:   { xray: [0.86, 0.93, 1.00], gain: 0.70, solid: [0.91, 0.87, 0.78] },
};
const MESH_LAYER = { skull: 'skull', jaw: 'skull', brain: 'brain', muscles: 'muscles', musjaw: 'muscles', eyes: 'eyes' };

const mats = {}; // layer -> { xray, solid }
function makeMaterials(meta) {
  for (const [layer, st] of Object.entries(LAYER_STYLE)) {
    const xray = new THREE.ShaderMaterial({
      vertexShader: VERT, fragmentShader: FRAG_XRAY,
      uniforms: { uColor: { value: new THREE.Color().fromArray(st.xray) }, uGain: { value: st.gain } },
      side: THREE.DoubleSide, depthTest: false, depthWrite: false, transparent: true,
      blending: THREE.CustomBlending, blendEquation: THREE.AddEquation,
      blendSrc: THREE.OneFactor, blendDst: THREE.OneFactor, blendSrcAlpha: THREE.ZeroFactor, blendDstAlpha: THREE.OneFactor,
    });
    const solid = new THREE.ShaderMaterial({
      vertexShader: VERT, fragmentShader: FRAG_SOLID,
      uniforms: {
        uColor: { value: new THREE.Color().fromArray(st.solid) }, uAlpha: { value: 1 }, uBright: { value: 1 },
        uEyeL: { value: new THREE.Vector3().fromArray(meta.eyeL) }, uEyeR: { value: new THREE.Vector3().fromArray(meta.eyeR) },
        uIris: { value: layer === 'eyes' ? 1 : 0 },
      },
      side: THREE.DoubleSide, blending: THREE.NoBlending, depthTest: true, depthWrite: true,
    });
    mats[layer] = { xray, solid };
  }
}
const fillMat = new THREE.ShaderMaterial({
  vertexShader: VERT, fragmentShader: FRAG_FILL,
  uniforms: {
    uFill: { value: new THREE.Color(0.02, 0.05, 0.11) }, uFillA: { value: 0.5 },
    uRim: { value: new THREE.Color(0.16, 0.36, 0.95) }, uRimGain: { value: 1 },
  },
  side: THREE.FrontSide, blending: THREE.NoBlending, depthTest: true, depthWrite: true,
});
const outMat = new THREE.ShaderMaterial({
  vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }',
  fragmentShader: FRAG_OUT,
  uniforms: {
    tMap: { value: null }, uRes: { value: new THREE.Vector2(1, 1) }, uLens: { value: new THREE.Vector3() },
    uFeather: { value: 0.17 }, uMode: { value: 1 }, uHalf: { value: new THREE.Vector3(1, 0, 0) },
  },
  depthTest: false, depthWrite: false, transparent: true,
  blending: THREE.CustomBlending, blendEquation: THREE.AddEquation,
  blendSrc: THREE.OneFactor, blendDst: THREE.OneMinusSrcAlphaFactor, blendSrcAlpha: THREE.OneFactor, blendDstAlpha: THREE.OneMinusSrcAlphaFactor,
});
sceneOut.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), outMat));

/* ------------------------------------------------------------------ */
/* Anatomy                                                             */
/* ------------------------------------------------------------------ */

let META = null;
const head = new THREE.Group(); head.matrixAutoUpdate = false; head.visible = false; sceneMain.add(head);
const headFill = new THREE.Group(); headFill.matrixAutoUpdate = false; headFill.visible = false; sceneFill.add(headFill);
const jaw = new THREE.Group(); jaw.matrixAutoUpdate = false; head.add(jaw);
const layerMeshes = { skull: [], brain: [], muscles: [], eyes: [], hands: [] };
const A = new THREE.Matrix4();        // BodyParts3D millimetres -> MediaPipe canonical face space (cm)
const fitPivot = new THREE.Vector3(); // head centre in canonical space

async function loadAnatomy() {
  const buf = await (await fetch(here('./assets/anatomy.bin'))).arrayBuffer();
  const headerLen = new DataView(buf).getUint32(0, true);
  const meta = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, headerLen)));
  const base = 4 + headerLen, mn = meta.min, mx = meta.max;
  META = meta;
  makeMaterials(meta);
  A.fromArray(meta.A);
  fitPivot.fromArray(meta.headCenter).applyMatrix4(A);
  for (const m of meta.meshes) {
    const q = new Uint16Array(buf, base + m.pos, m.vcount * 3);
    const p = new Float32Array(m.vcount * 3);
    const sx = (mx[0] - mn[0]) / 65535, sy = (mx[1] - mn[1]) / 65535, sz = (mx[2] - mn[2]) / 65535;
    for (let i = 0; i < p.length; i += 3) { p[i] = mn[0] + q[i] * sx; p[i + 1] = mn[1] + q[i + 1] * sy; p[i + 2] = mn[2] + q[i + 2] * sz; }
    const idx = m.idxBytes === 2 ? new Uint16Array(buf, base + m.idx, m.icount) : new Uint32Array(buf, base + m.idx, m.icount);
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(p, 3));
    g.setIndex(new THREE.BufferAttribute(idx, 1));
    g.computeVertexNormals();
    if (m.name === 'skin') {
      const mesh = new THREE.Mesh(g, fillMat); mesh.frustumCulled = false; headFill.add(mesh);
    } else {
      const layer = MESH_LAYER[m.name];
      const mesh = new THREE.Mesh(g, mats[layer].xray); mesh.frustumCulled = false;
      (m.name === 'jaw' || m.name === 'musjaw' ? jaw : head).add(mesh);
      layerMeshes[layer].push(mesh);
    }
  }
}

/* ------------------------------------------------------------------ */
/* Hand bones (procedural, driven by the 21 hand landmarks)            */
/* ------------------------------------------------------------------ */

const handsGroup = new THREE.Group(); sceneMain.add(handsGroup);
const BONE_PROFILE = [[0, 0], [0.62, 0.0], [0.95, 0.035], [1.0, 0.09], [0.86, 0.16], [0.6, 0.27], [0.5, 0.42], [0.5, 0.6], [0.58, 0.76], [0.8, 0.88], [0.86, 0.94], [0.7, 0.99], [0, 1.0]];
const boneGeo = new THREE.LatheGeometry(BONE_PROFILE.map(([r, y]) => new THREE.Vector2(r, y)), 12);
const carpalGeo = new THREE.SphereGeometry(1, 10, 8);
// [from, to, knuckle radius as a fraction of palm length]; 'C' = carpal end of the metacarpal
const HAND_BONES = [[1, 2, 0.085], [2, 3, 0.078], [3, 4, 0.066]];
for (const b of [5, 9, 13, 17]) {
  const k = b === 17 ? 0.86 : 1;
  HAND_BONES.push(['C', b, 0.074 * k], [b, b + 1, 0.07 * k], [b + 1, b + 2, 0.06 * k], [b + 2, b + 3, 0.052 * k]);
}
const CARPALS = [[-0.19, 0.03], [-0.065, 0.015], [0.065, 0.015], [0.19, 0.04], [-0.21, 0.155], [-0.07, 0.15], [0.07, 0.15], [0.21, 0.16]];
const HAND_LINKS = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]];

const handSlots = [0, 1].map(() => {
  const g = new THREE.Group(); g.visible = false; handsGroup.add(g);
  const bones = HAND_BONES.map(() => { const m = new THREE.Mesh(boneGeo, null); m.matrixAutoUpdate = false; m.frustumCulled = false; g.add(m); return m; });
  const carpals = CARPALS.map(() => { const m = new THREE.Mesh(carpalGeo, null); m.matrixAutoUpdate = false; m.frustumCulled = false; g.add(m); return m; });
  layerMeshes.hands.push(...bones, ...carpals);
  return { group: g, bones, carpals, pts: Array.from({ length: 21 }, () => new THREE.Vector3()), uv: [], has: false, seen: 0, label: '', anchors: {} };
});

const _a = new THREE.Vector3(), _b = new THREE.Vector3(), _x = new THREE.Vector3(), _y = new THREE.Vector3(), _z = new THREE.Vector3(), _n = new THREE.Vector3(), _u = new THREE.Vector3(), _s = new THREE.Vector3(), _o = new THREE.Vector3();

function placeBone(mesh, a, b, r, normal) {
  _y.subVectors(b, a); const len = _y.length(); if (len < 1e-4) { mesh.visible = false; return; }
  mesh.visible = true; _y.divideScalar(len);
  _x.crossVectors(_y, normal); if (_x.lengthSq() < 1e-6) _x.set(1, 0, 0); _x.normalize();
  _z.crossVectors(_x, _y);
  const gap = len * 0.045, L = len - gap * 2;
  _o.copy(a).addScaledVector(_y, gap);
  mesh.matrix.makeBasis(_x.clone().multiplyScalar(r), _y.clone().multiplyScalar(L), _z.clone().multiplyScalar(r * 0.84)).setPosition(_o);
}

// Lift the 2D hand landmarks into camera space. Depth comes from the hand's real size (world landmarks, metres).
function handTo3D(lm, wlm, out, aspect) {
  const X = (i) => (2 * lm[i].x - 1) * aspect, Y = (i) => 1 - 2 * lm[i].y;
  const est = [];
  for (const [i, j] of [[0, 5], [0, 17], [5, 17], [0, 9], [0, 13], [5, 13]]) {
    const li = Math.hypot(X(i) - X(j), Y(i) - Y(j));
    const lw = Math.hypot(wlm[i].x - wlm[j].x, wlm[i].y - wlm[j].y) * 100;
    if (li > 1e-4 && lw > 0.5) est.push(FOCAL * lw / li);
  }
  if (!est.length) return false;
  est.sort((p, q) => p - q);
  const Z0 = clamp(est[est.length >> 1], 12, 250);
  let zm = 0; for (let i = 0; i < 21; i++) zm += wlm[i].z; zm /= 21;
  for (let i = 0; i < 21; i++) {
    const Z = Z0 + (wlm[i].z - zm) * 100;
    out[i].set(X(i) * Z / FOCAL, Y(i) * Z / FOCAL, -Z);
  }
  return true;
}

const _tmpPts = Array.from({ length: 21 }, () => new THREE.Vector3());
function updateHands(res, now, aspect) {
  const used = [false, false];
  const n = res ? res.landmarks.length : 0;
  for (let h = 0; h < n; h++) {
    const label = res.handedness?.[h]?.[0]?.categoryName || '';
    let slot = label === 'Left' ? 0 : 1;
    if (used[slot]) slot = 1 - slot;
    if (used[slot]) continue;
    if (!handTo3D(res.landmarks[h], res.worldLandmarks[h], _tmpPts, aspect)) continue;
    used[slot] = true;
    const H = handSlots[slot];
    const fresh = !H.has || now - H.seen > 400;
    for (let i = 0; i < 21; i++) {
      if (fresh) H.pts[i].copy(_tmpPts[i]);
      else {
        const d = H.pts[i].distanceTo(_tmpPts[i]);
        H.pts[i].lerp(_tmpPts[i], clamp(0.42 + d * 0.5, 0.42, 1));
      }
    }
    H.uv = res.landmarks[h]; H.has = true; H.seen = now; H.label = label;
    layoutHand(H);
  }
  for (const H of handSlots) {
    if (H.has && now - H.seen > 220) H.has = false;
    H.group.visible = H.has && S.layers.hands;
  }
}

function layoutHand(H) {
  const P = H.pts;
  _u.subVectors(P[9], P[0]); const palm = _u.length(); if (palm < 1e-3) return; _u.divideScalar(palm);
  _s.subVectors(P[5], P[17]); _s.addScaledVector(_u, -_s.dot(_u)).normalize();
  _n.crossVectors(_u, _s).normalize();
  HAND_BONES.forEach(([from, to, k], i) => {
    if (from === 'C') _a.lerpVectors(P[0], P[to], 0.3); else _a.copy(P[from]);
    _b.copy(P[to]);
    placeBone(H.bones[i], _a, _b, k * palm * (from === 1 ? 1.1 : 1), _n);
  });
  CARPALS.forEach(([sx, uy], i) => {
    _o.copy(P[0]).addScaledVector(_s, sx * palm).addScaledVector(_u, uy * palm);
    const r = palm * 0.062;
    H.carpals[i].matrix.makeBasis(_s.clone().multiplyScalar(r * 1.05), _u.clone().multiplyScalar(r * 0.92), _n.clone().multiplyScalar(r * 0.8)).setPosition(_o);
  });
  H.anchors = {
    'Distal phalanx': new THREE.Vector3().lerpVectors(P[7], P[8], 0.5),
    'Middle phalanx': new THREE.Vector3().lerpVectors(P[6], P[7], 0.5),
    'Proximal phalanx': new THREE.Vector3().lerpVectors(P[5], P[6], 0.5),
    'Metacarpal': new THREE.Vector3().lerpVectors(P[0], P[9], 0.68),
    'Carpals': new THREE.Vector3().copy(P[0]).addScaledVector(_u, 0.09 * palm),
  };
}

/* ------------------------------------------------------------------ */
/* Head pose                                                           */
/* ------------------------------------------------------------------ */

const pose = { pos: new THREE.Vector3(), quat: new THREE.Quaternion(), has: false, seen: 0, jaw: 0, lm: null, t: 0 };
const _m = new THREE.Matrix4(), _m2 = new THREE.Matrix4(), _p = new THREE.Vector3(), _q = new THREE.Quaternion(), _sc = new THREE.Vector3();
const ONE = new THREE.Vector3(1, 1, 1);

function updatePose(res, now) {
  const mats4 = res && res.facialTransformationMatrixes;
  if (!mats4 || !mats4.length) { if (now - pose.seen > 350) pose.has = false; return; }
  _m.fromArray(mats4[0].data); _m.decompose(_p, _q, _sc);
  const dt = Math.max(1, now - pose.t) / 1000;
  if (!pose.has || now - pose.seen > 500) { pose.pos.copy(_p); pose.quat.copy(_q); }
  else {
    const speed = pose.pos.distanceTo(_p) / dt;          // cm per second
    pose.pos.lerp(_p, clamp(0.28 + speed * 0.03, 0.28, 1));
    const ang = pose.quat.angleTo(_q) / dt;              // radians per second
    pose.quat.slerp(_q, clamp(0.3 + ang * 0.35, 0.3, 1));
  }
  let j = 0;
  const cats = res.faceBlendshapes?.[0]?.categories;
  if (cats) for (const c of cats) if (c.categoryName === 'jawOpen') { j = c.score; break; }
  pose.jaw += (j - pose.jaw) * 0.55;
  pose.has = true; pose.seen = now; pose.t = now; pose.lm = res.faceLandmarks[0];
}

function applyPose() {
  const vis = pose.has;
  head.visible = vis; headFill.visible = vis;
  if (!vis) return;
  const f = S.fit, mx = src.mirrored ? -1 : 1;
  _m.compose(pose.pos, pose.quat, ONE);
  _m2.makeTranslation(fitPivot.x + f.x * mx, fitPivot.y + f.y, fitPivot.z + f.z);
  _m.multiply(_m2);
  _m2.makeScale(f.size * f.width, f.size * f.height, f.size * f.depth); _m.multiply(_m2);
  _m2.makeTranslation(-fitPivot.x, -fitPivot.y, -fitPivot.z); _m.multiply(_m2);
  _m.multiply(A);
  head.matrix.copy(_m); head.matrixWorldNeedsUpdate = true;
  headFill.matrix.copy(_m); headFill.matrixWorldNeedsUpdate = true;
  // jaw hinge around the temporomandibular joints
  const jp = META.jawPivot, ang = clamp(pose.jaw, 0, 1) * 0.44;
  jaw.matrix.makeTranslation(jp[0], jp[1], jp[2]);
  _m2.makeRotationX(ang); jaw.matrix.multiply(_m2);
  _m2.makeTranslation(-jp[0], -jp[1], -jp[2]); jaw.matrix.multiply(_m2);
  jaw.matrixWorldNeedsUpdate = true;
}

/* ------------------------------------------------------------------ */
/* Video source                                                        */
/* ------------------------------------------------------------------ */

const src = { kind: 'none', el: null, w: 4, h: 3, mirrored: true, lastTime: -1, stillFrames: 0, facing: 'user', stream: null };
const F = { w: 1, h: 1, left: 0, top: 0, pxW: 1, pxH: 1, sw: 1, sh: 1, dpr: 1 };

function layout() {
  const sw = stage.clientWidth, sh = stage.clientHeight;
  if (!sw || !sh) return;
  const sc = Math.max(sw / src.w, sh / src.h);
  F.w = src.w * sc; F.h = src.h * sc; F.left = (sw - F.w) / 2; F.top = (sh - F.h) / 2; F.sw = sw; F.sh = sh;
  frame.style.width = F.w + 'px'; frame.style.height = F.h + 'px';
  frame.style.left = F.left + 'px'; frame.style.top = F.top + 'px';
  frame.classList.toggle('mirror', src.mirrored);
  const dpr = Math.min(window.devicePixelRatio || 1, 2); F.dpr = dpr;
  let pw = F.w * dpr, ph = F.h * dpr;
  const cap = 1500 / Math.max(pw, ph); if (cap < 1) { pw *= cap; ph *= cap; }
  F.pxW = Math.max(2, Math.round(pw)); F.pxH = Math.max(2, Math.round(ph));
  renderer.setSize(F.pxW, F.pxH, false);
  rt.setSize(F.pxW, F.pxH);
  camera.aspect = src.w / src.h; camera.updateProjectionMatrix();
  hud.width = Math.round(sw * dpr); hud.height = Math.round(sh * dpr);
}
new ResizeObserver(layout).observe(stage);
video.addEventListener('resize', () => { if (src.kind === 'camera' || src.kind === 'video') { src.w = video.videoWidth || src.w; src.h = video.videoHeight || src.h; layout(); } });

function stopSource() {
  if (src.stream) { src.stream.getTracks().forEach((t) => t.stop()); src.stream = null; }
  video.pause(); video.srcObject = null;
  if (video.src) { try { URL.revokeObjectURL(video.src); } catch (e) { /* ignore */ } video.removeAttribute('src'); video.load(); }
  src.kind = 'none'; pose.has = false; handSlots.forEach((h) => (h.has = false));
}

async function startCamera(facing) {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw Object.assign(new Error('no-camera-api'), { name: 'NoCameraApi' });
  }
  const stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { facingMode: facing, width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } } });
  stopSource();
  src.stream = stream; src.facing = facing;
  video.srcObject = stream; still.style.display = 'none'; video.style.display = 'block';
  await video.play().catch(() => {});
  await new Promise((r) => { if (video.videoWidth) r(); else video.addEventListener('loadedmetadata', r, { once: true }); });
  src.kind = 'camera'; src.el = video; src.w = video.videoWidth; src.h = video.videoHeight; src.lastTime = -1;
  const settings = stream.getVideoTracks()[0]?.getSettings?.() || {};
  src.mirrored = (settings.facingMode || facing) !== 'environment';
  layout();
  try {
    const devs = await navigator.mediaDevices.enumerateDevices();
    $('btn-flip').hidden = devs.filter((d) => d.kind === 'videoinput').length < 2;
  } catch (e) { $('btn-flip').hidden = true; }
}

async function useFile(file) {
  if (!file) return;
  if (file.type.startsWith('video/')) {
    stopSource();
    video.src = URL.createObjectURL(file); video.loop = true; video.muted = true;
    still.style.display = 'none'; video.style.display = 'block';
    await new Promise((r, j) => { video.addEventListener('loadedmetadata', r, { once: true }); video.addEventListener('error', () => j(new Error('This video could not be opened.')), { once: true }); });
    await video.play().catch(() => {});
    src.kind = 'video'; src.el = video; src.w = video.videoWidth; src.h = video.videoHeight; src.lastTime = -1; src.mirrored = false;
  } else {
    const url = URL.createObjectURL(file);
    const img = new Image();
    await new Promise((r, j) => { img.onload = r; img.onerror = () => j(new Error('This picture could not be opened.')); img.src = url; });
    stopSource();
    const sc = Math.min(1, 1280 / Math.max(img.naturalWidth, img.naturalHeight));
    still.width = Math.round(img.naturalWidth * sc); still.height = Math.round(img.naturalHeight * sc);
    still.getContext('2d').drawImage(img, 0, 0, still.width, still.height);
    URL.revokeObjectURL(url);
    video.style.display = 'none'; still.style.display = 'block';
    src.kind = 'still'; src.el = still; src.w = still.width; src.h = still.height; src.stillFrames = 0; src.mirrored = false;
  }
  $('btn-flip').hidden = true;
  layout();
}

/* ------------------------------------------------------------------ */
/* Tracking                                                            */
/* ------------------------------------------------------------------ */

let faceLm = null, handLm = null, ready = false, lastTs = 0, frameNo = 0;
const stats = { fps: 0, ms: 0, lastFrame: 0, delegate: '' };
let lastHands = null;

async function loadTrackers() {
  const fileset = { wasmLoaderPath: here('./lib/vision_wasm_internal.js'), wasmBinaryPath: here('./lib/vision_wasm_internal.wasm') };
  const make = async (delegate) => {
    const f = await FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: here('./assets/face_landmarker.task'), delegate },
      runningMode: 'VIDEO', numFaces: 1, outputFacialTransformationMatrixes: true, outputFaceBlendshapes: true,
    });
    const h = await HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: here('./assets/hand_landmarker.task'), delegate },
      runningMode: 'VIDEO', numHands: 2,
    });
    return [f, h];
  };
  const want = params.get('cpu') ? 'CPU' : 'GPU';
  try { [faceLm, handLm] = await make(want); stats.delegate = want; }
  catch (e) {
    if (want === 'CPU') throw e;
    console.warn('GPU tracking unavailable, using CPU', e);
    [faceLm, handLm] = await make('CPU'); stats.delegate = 'CPU';
  }
}

function detect(now) {
  let ts = Math.round(now); if (ts <= lastTs) ts = lastTs + 1; lastTs = ts;
  const t0 = performance.now();
  let fr = null;
  try { fr = faceLm.detectForVideo(src.el, ts); } catch (e) { console.warn(e); }
  updatePose(fr, now);
  const needHands = S.layers.hands || S.reveal === 'lens-hand' || S.tracking;
  if (needHands) {
    try { lastHands = handLm.detectForVideo(src.el, ts); } catch (e) { console.warn(e); lastHands = null; }
    updateHands(lastHands, now, src.w / src.h);
  } else { lastHands = null; updateHands(null, now, src.w / src.h); }
  const ms = performance.now() - t0;
  stats.ms += (ms - stats.ms) * 0.15;
  if (stats.lastFrame) { const f = 1000 / Math.max(1, now - stats.lastFrame); stats.fps += (f - stats.fps) * 0.12; }
  stats.lastFrame = now;
}

/* ------------------------------------------------------------------ */
/* Per-frame update, render, HUD                                       */
/* ------------------------------------------------------------------ */

const lens = { u: 0.5, v: 0.5, init: false };
const _v = new THREE.Vector3();

function toUV(camPoint, out) { // camera space (cm) -> normalised frame coords, v down
  _v.copy(camPoint).applyMatrix4(camera.projectionMatrix);
  out[0] = (_v.x + 1) / 2; out[1] = (1 - _v.y) / 2; return out;
}
function uvToStage(u, v, out) { // frame coords -> CSS px inside the stage (handles the mirror flip)
  out[0] = F.left + (src.mirrored ? 1 - u : u) * F.w; out[1] = F.top + v * F.h; return out;
}
const headUV = [0.5, 0.5], headTopUV = [0.5, 0.4], _uv = [0, 0], _st = [0, 0];
let headRadiusCss = 100;

function applySettingsToScene() {
  const xr = S.look === 'xray';
  const gain = 0.4 + S.brightness * 2.0, bright = 0.72 + S.brightness * 0.75;
  const solidAlpha = 1 - 0.78 * S.through;
  for (const [layer, list] of Object.entries(layerMeshes)) {
    const on = S.layers[layer];
    const m = mats[layer]; if (!m) continue;
    m.xray.uniforms.uGain.value = LAYER_STYLE[layer].gain * gain * (1 - 0.35 * S.through);
    m.solid.uniforms.uAlpha.value = solidAlpha; m.solid.uniforms.uBright.value = bright;
    for (const mesh of list) { mesh.material = xr ? m.xray : m.solid; if (layer !== 'hands') mesh.visible = on; }
  }
  fillMat.uniforms.uFillA.value = (1 - S.through) * (xr ? 0.94 : 0.8);
  fillMat.uniforms.uRimGain.value = S.layers.outline ? (0.55 + S.brightness * 1.3) : 0;
}

function updateLens() {
  let tu = 0.5, tv = 0.5, found = false;
  if (pose.has) {
    _p.fromArray(META.headCenter).applyMatrix4(head.matrix); toUV(_p, headUV);
    _b.fromArray(META.headCenter); _b.z += 60; _b.applyMatrix4(head.matrix); toUV(_b, headTopUV);
    _a.copy(_p); _a.x += 9.5; toUV(_a, _uv); headRadiusCss = Math.abs(_uv[0] - headUV[0]) * F.w;
  }
  if (S.reveal === 'lens-hand') {
    const H = handSlots.find((h) => h.has);
    if (H) { _a.set(0, 0, 0); for (const i of [0, 5, 9, 13, 17]) _a.add(H.pts[i]); _a.multiplyScalar(0.2); toUV(_a, _uv); tu = _uv[0]; tv = _uv[1]; found = true; }
  }
  if (!found && pose.has) { tu = headUV[0]; tv = headUV[1]; found = true; }
  if (found) {
    if (!lens.init) { lens.u = tu; lens.v = tv; lens.init = true; }
    else { lens.u += (tu - lens.u) * 0.3; lens.v += (tv - lens.v) * 0.3; }
  }
  const rCss = S.lens * 0.95 * Math.min(F.sw, F.sh) * 0.5 * 1.25;
  const k = F.pxW / F.w;
  const U = outMat.uniforms;
  U.uRes.value.set(F.pxW, F.pxH);
  U.uLens.value.set(lens.u * F.pxW, (1 - lens.v) * F.pxH, rCss * k);
  U.uFeather.value = S.feather;
  U.uMode.value = S.reveal === 'all' ? 0 : S.reveal === 'half' ? 2 : 1;
  if (S.reveal === 'half') {
    // split along the head's own vertical axis; reveal the half shown on the left of the screen
    const cx = headUV[0] * F.pxW, cy = (1 - headUV[1]) * F.pxH;
    let ux = (headTopUV[0] - headUV[0]) * F.pxW, uy = -(headTopUV[1] - headUV[1]) * F.pxH;
    const l = Math.hypot(ux, uy) || 1; ux /= l; uy /= l;
    let nx = uy, ny = -ux;
    if ((nx > 0) !== src.mirrored) { nx = -nx; ny = -ny; }
    U.uHalf.value.set(nx, ny, nx * cx + ny * cy);
  }
  lens.rCss = rCss;
}

function render() {
  renderer.setRenderTarget(rt);
  renderer.clear(true, true, true);
  renderer.render(sceneFill, camera);
  renderer.clearDepth();
  renderer.render(sceneMain, camera);
  renderer.setRenderTarget(null);
  renderer.clear(true, true, true);
  outMat.uniforms.tMap.value = rt.texture;
  renderer.render(sceneOut, orthoCam);
}

/* ---- HUD: lens ring, labels, tracking data (drawn un-mirrored so text reads normally) ---- */

function placeColumn(items, x, align, minY, maxY) {
  if (!items.length) return;
  items.sort((p, q) => p.y - q.y);
  const gap = 17;
  let y = -1e9;
  for (const it of items) { it.ly = Math.max(it.y, y + gap, minY); y = it.ly; }
  y = maxY + gap;
  for (let i = items.length - 1; i >= 0; i--) { items[i].ly = Math.min(items[i].ly, y - gap); y = items[i].ly; }
  hctx.textAlign = align;
  for (const it of items) {
    hctx.beginPath(); hctx.moveTo(it.x, it.y); hctx.lineTo(x + (align === 'right' ? 5 : -5), it.ly); hctx.stroke();
    hctx.beginPath(); hctx.arc(it.x, it.y, 1.6, 0, Math.PI * 2); hctx.fill();
    hctx.fillText(it.name, x, it.ly + 3.5);
  }
}

function drawHud() {
  const dpr = F.dpr;
  hctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  hctx.clearRect(0, 0, F.sw, F.sh);
  if (src.kind === 'none') return;

  if (S.reveal.startsWith('lens') && lens.init) {
    uvToStage(lens.u, lens.v, _st);
    hctx.strokeStyle = 'rgba(255,255,255,0.55)'; hctx.lineWidth = 1;
    hctx.beginPath(); hctx.arc(_st[0], _st[1], lens.rCss, 0, Math.PI * 2); hctx.stroke();
  }

  if (S.labels) {
    hctx.font = '11px "Segoe UI", system-ui, -apple-system, sans-serif';
    hctx.fillStyle = 'rgba(236,243,250,0.92)'; hctx.strokeStyle = 'rgba(236,243,250,0.42)'; hctx.lineWidth = 0.8;
    hctx.shadowColor = 'rgba(0,0,0,0.75)'; hctx.shadowBlur = 3;
    if (pose.has) {
      uvToStage(headUV[0], headUV[1], _st); const hx = _st[0];
      const L = [], R = [];
      for (const lb of META.labels) {
        if (!S.layers[lb.layer]) continue;
        _p.fromArray(lb.p); if (lb.jaw) _p.applyMatrix4(jaw.matrix); _p.applyMatrix4(head.matrix);
        toUV(_p, _uv); uvToStage(_uv[0], _uv[1], _st);
        (_st[0] < hx ? L : R).push({ name: lb.name, x: _st[0], y: _st[1] });
      }
      const off = Math.max(headRadiusCss * 1.5, 90);
      placeColumn(L, Math.max(hx - off, 86), 'right', 12, F.sh - 10);
      placeColumn(R, Math.min(hx + off, F.sw - 86), 'left', 12, F.sh - 10);
    }
    if (S.layers.hands) {
      for (const H of handSlots) {
        if (!H.has) continue;
        toUV(H.pts[9], _uv); uvToStage(_uv[0], _uv[1], _st); const cx = _st[0];
        toUV(H.pts[0], _uv); const w0 = uvToStage(_uv[0], _uv[1], [0, 0]);
        const size = Math.hypot(_st[0] - w0[0], _st[1] - w0[1]);
        const items = [];
        for (const [name, p] of Object.entries(H.anchors)) { toUV(p, _uv); uvToStage(_uv[0], _uv[1], _st); items.push({ name, x: _st[0], y: _st[1] }); }
        uvToStage(headUV[0], headUV[1], _st);
        const leftSide = pose.has ? cx < _st[0] : cx < F.sw / 2;
        const x = leftSide ? Math.max(cx - size * 1.5, 92) : Math.min(cx + size * 1.5, F.sw - 92);
        placeColumn(items, x, leftSide ? 'right' : 'left', 12, F.sh - 10);
      }
    }
    hctx.shadowBlur = 0;
  }

  if (S.tracking) {
    if (pose.lm) {
      hctx.fillStyle = 'rgba(120,220,255,0.75)';
      for (let i = 0; i < pose.lm.length; i++) { uvToStage(pose.lm[i].x, pose.lm[i].y, _st); hctx.fillRect(_st[0] - 0.6, _st[1] - 0.6, 1.3, 1.3); }
    }
    if (lastHands) {
      hctx.strokeStyle = 'rgba(160,255,190,0.8)'; hctx.fillStyle = 'rgba(160,255,190,0.95)'; hctx.lineWidth = 1;
      for (const lm of lastHands.landmarks) {
        hctx.beginPath();
        for (const [i, j] of HAND_LINKS) { uvToStage(lm[i].x, lm[i].y, _st); hctx.moveTo(_st[0], _st[1]); uvToStage(lm[j].x, lm[j].y, _st); hctx.lineTo(_st[0], _st[1]); }
        hctx.stroke();
        for (const p of lm) { uvToStage(p.x, p.y, _st); hctx.fillRect(_st[0] - 1.5, _st[1] - 1.5, 3, 3); }
      }
    }
    const e = new THREE.Euler().setFromQuaternion(pose.quat, 'YXZ'), deg = THREE.MathUtils.radToDeg;
    const lines = [
      `${stats.fps.toFixed(0)} fps   tracking ${stats.ms.toFixed(0)} ms   ${stats.delegate}`,
      `source ${src.w}×${src.h}   render ${F.pxW}×${F.pxH}`,
      pose.has ? `head  x ${pose.pos.x.toFixed(1)}  y ${pose.pos.y.toFixed(1)}  z ${(-pose.pos.z).toFixed(1)} cm` : 'head  not found',
      pose.has ? `yaw ${deg(e.y).toFixed(0)}°  pitch ${deg(e.x).toFixed(0)}°  roll ${deg(e.z).toFixed(0)}°  jaw ${(pose.jaw * 100).toFixed(0)}%` : '',
      `hands ${handSlots.filter((h) => h.has).length}`,
    ].filter(Boolean);
    hctx.font = '11px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
    hctx.textAlign = 'left';
    const w = Math.max(...lines.map((l) => hctx.measureText(l).width)) + 16, h = lines.length * 15 + 10;
    hctx.fillStyle = 'rgba(8,13,19,0.72)'; hctx.fillRect(10, 10, w, h);
    hctx.fillStyle = 'rgba(200,232,250,0.95)';
    lines.forEach((l, i) => hctx.fillText(l, 18, 27 + i * 15));
  }
}

let noteTimer = 0;
function updateNote(now) {
  const el = $('note');
  let msg = '';
  if (ready && src.kind !== 'none' && !pose.has && now - pose.seen > 1200 && now - noteTimer > 1200) {
    msg = src.kind === 'still' ? 'No face found in this picture' : 'Looking for a face…';
  }
  if (msg) { el.textContent = msg; el.hidden = false; } else el.hidden = true;
}

function tick(now) {
  requestAnimationFrame(tick);
  if (!ready || src.kind === 'none' || document.hidden) return;
  let fresh = false;
  if (src.kind === 'still') { src.stillFrames++; fresh = src.stillFrames < 20 || src.stillFrames % 30 === 0; }
  else if (video.readyState >= 2 && video.currentTime !== src.lastTime) { src.lastTime = video.currentTime; fresh = true; }
  if (fresh) { frameNo++; detect(now); }
  applyPose();
  applySettingsToScene();
  updateLens();
  render();
  drawHud();
  updateNote(now);
}

/* ------------------------------------------------------------------ */
/* Controls                                                            */
/* ------------------------------------------------------------------ */

function paintRange(el) { el.style.setProperty('--p', ((el.value - el.min) / (el.max - el.min)) * 100 + '%'); }
function bindRange(id, get, set, out, fmt) {
  const el = $(id);
  const show = () => { paintRange(el); if (out) $(out).textContent = fmt(get()); };
  el.value = get(); show();
  el.addEventListener('input', () => { set(parseFloat(el.value)); show(); saveSettings(); });
  return () => { el.value = get(); show(); };
}
const pct = (v) => Math.round(v * 100) + '%';
const cm = (v) => (Math.abs(v) < 0.05 ? 0 : v).toFixed(1) + ' cm';
const refreshers = [];

function initControls() {
  for (const k of Object.keys(DEFAULTS.layers)) {
    const el = $('l-' + k); el.checked = S.layers[k];
    el.addEventListener('change', () => { S.layers[k] = el.checked; saveSettings(); });
  }
  const syncLayers = () => { for (const k of Object.keys(DEFAULTS.layers)) $('l-' + k).checked = S.layers[k]; };
  const PRESETS = {
    skull: { skull: true, brain: false, muscles: false, eyes: false },
    brain: { skull: false, brain: true, muscles: false, eyes: false },
    muscles: { skull: false, brain: false, muscles: true, eyes: true },
    everything: { skull: true, brain: true, muscles: true, eyes: true, outline: true, hands: true },
  };
  document.querySelectorAll('[data-preset]').forEach((b) => b.addEventListener('click', () => { Object.assign(S.layers, PRESETS[b.dataset.preset]); syncLayers(); saveSettings(); }));

  for (const [id, key] of [['labels', 'labels'], ['tracking', 'tracking']]) {
    const el = $(id); el.checked = S[key];
    el.addEventListener('change', () => { S[key] = el.checked; saveSettings(); });
  }
  const setLook = (v) => { S.look = v; $('look-xray').setAttribute('aria-pressed', v === 'xray'); $('look-solid').setAttribute('aria-pressed', v === 'solid'); saveSettings(); };
  $('look-xray').addEventListener('click', () => setLook('xray'));
  $('look-solid').addEventListener('click', () => setLook('solid'));
  setLook(S.look);

  bindRange('brightness', () => S.brightness, (v) => (S.brightness = v));
  bindRange('through', () => S.through, (v) => (S.through = v));
  bindRange('feather', () => S.feather, (v) => (S.feather = v), 'o-feather', pct);
  bindRange('lens', () => S.lens, (v) => (S.lens = v));
  const rv = $('reveal'); rv.value = S.reveal; if (!rv.value) { S.reveal = 'lens-hand'; rv.value = S.reveal; }
  rv.addEventListener('change', () => { S.reveal = rv.value; saveSettings(); });

  for (const k of ['size', 'width', 'height', 'depth']) refreshers.push(bindRange('f-' + k, () => S.fit[k], (v) => (S.fit[k] = v), 'o-' + k, pct));
  for (const k of ['x', 'y', 'z']) refreshers.push(bindRange('f-' + k, () => S.fit[k], (v) => (S.fit[k] = v), 'o-' + k, cm));
  $('reset-fit').addEventListener('click', () => { Object.assign(S.fit, DEFAULTS.fit); refreshers.forEach((f) => f()); saveSettings(); });

  $('btn-start').addEventListener('click', onStartCamera);
  $('btn-start-file').addEventListener('click', () => $('file').click());
  $('btn-file').addEventListener('click', () => $('file').click());
  $('btn-flip').addEventListener('click', async () => {
    try { await startCamera(src.facing === 'user' ? 'environment' : 'user'); } catch (e) { console.warn(e); }
  });
  $('file').addEventListener('change', async (ev) => {
    const file = ev.target.files && ev.target.files[0]; ev.target.value = '';
    if (!file) return;
    try { await useFile(file); await whenReady(); $('start').hidden = true; $('btn-file').hidden = false; }
    catch (e) { startMsg(e.message || 'That file could not be opened.', true); }
  });
}

function startMsg(text, err) { const el = $('start-msg'); el.textContent = text; el.classList.toggle('err', !!err); }

let loadPromise = null, loadError = null;
function whenReady() { return loadPromise; }

async function onStartCamera() {
  const btn = $('btn-start'); btn.disabled = true;
  try {
    startMsg('Waiting for camera permission…');
    await startCamera('user');
    if (!ready) startMsg('Loading anatomy and tracking models…');
    await whenReady();
    if (loadError) throw loadError;
    $('start').hidden = true; $('btn-file').hidden = false;
  } catch (e) {
    console.warn(e);
    let msg;
    if (e === loadError) msg = 'The tracking models could not be loaded. Reload the page to try again.';
    else if (e.name === 'NoCameraApi' || !window.isSecureContext) msg = 'The camera is not available here. Open this page from an https:// address, or use a photo or video.';
    else if (e.name === 'NotAllowedError' || e.name === 'SecurityError') msg = 'Camera access is blocked here. Allow the camera for this page, or use a photo or video.';
    else if (e.name === 'NotFoundError' || e.name === 'OverconstrainedError') msg = 'No camera was found. You can use a photo or video instead.';
    else msg = 'The camera could not be started. You can use a photo or video instead.';
    startMsg(msg, true);
  } finally { btn.disabled = false; }
}

/* ------------------------------------------------------------------ */
/* Boot                                                                */
/* ------------------------------------------------------------------ */

initControls();
layout();
loadPromise = (async () => {
  try {
    await Promise.all([loadAnatomy(), loadTrackers()]);
    ready = true;
    if (src.kind === 'none') startMsg('Ready.');
  } catch (e) {
    console.error(e); loadError = e;
    startMsg('The tracking models could not be loaded. Reload the page to try again.', true);
  }
})();
requestAnimationFrame(tick);

// Test hook: ?src=<image url> uses a still picture as the source.
if (params.get('src')) {
  (async () => {
    const blob = await (await fetch(params.get('src'))).blob();
    await useFile(new File([blob], 'test', { type: blob.type }));
    await whenReady(); $('start').hidden = true; $('btn-file').hidden = false;
  })();
}
window.__xray = { S, pose, handSlots, stats, src, updateHands, lens };
