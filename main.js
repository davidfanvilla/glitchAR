import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.165.0/build/three.module.js";

const container = document.querySelector("#scene");
const startCameraButton = document.querySelector("#start-camera");
const preview = document.querySelector(".camera-preview");
const portraitPreview = document.querySelector("#portrait-preview");
const handVideo = document.querySelector("#hand-video");
const handCanvas = document.querySelector("#hand-canvas");
const handStatus = document.querySelector("#hand-status");
const handCtx = handCanvas.getContext("2d");
const liveCode = document.querySelector("#live-code");
const MEDIAPIPE_CDN = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
const HAND_MODEL_CDN =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 100);
camera.position.set(0, 0, 8.35);

const group = new THREE.Group();
scene.add(group);

const keyLight = new THREE.DirectionalLight(0xfff0df, 2.4);
keyLight.position.set(2.7, 3.6, 5);
scene.add(keyLight);
scene.add(new THREE.AmbientLight(0xffffff, 1.2));

const state = {
  fold: 0.72,
  depth: 0.62,
  targetX: 0,
  targetY: 0,
  targetZ: 0,
  seed: 7,
  handsDetected: 1,
  mode: "bolita",
  mouseFallback: false,
  pointer: { active: false, x: 0.5, y: 0.5, pressure: 0.22, crumple: 0 },
  hand: { active: true, x: 0.5, y: 0.52, pressure: 0.82, crumple: 0.96, openness: 0.05 },
  twoHands: { active: false, span: 0, angle: 0 },
};
window.paperPortrait = { state };

const debugParams = new URLSearchParams(window.location.search);
if (debugParams.get("crumple") === "1") {
  state.hand.active = true;
  state.hand.pressure = 1;
  state.hand.crumple = 1;
}
if (debugParams.get("mouse") === "1") {
  state.mouseFallback = true;
  state.hand.active = false;
  state.hand.crumple = 0;
  state.hand.pressure = 0;
  state.mode = "mouse";
}

let portraitMesh;
let wireMesh;
let handLandmarker;
let lastVideoTime = -1;
let trackingStarted = false;
let geometryBase = [];
let portraitAspect = 1071 / 1468;

const handConnections = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [5, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [9, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [13, 17],
  [17, 18],
  [18, 19],
  [19, 20],
  [0, 17],
];

const loader = new THREE.TextureLoader();
loadPortrait();

async function loadPortrait() {
  const portraitUrl = await getPortraitTextureUrl();
  loader.load(portraitUrl, (texture) => {
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.anisotropy = renderer.capabilities.getMaxAnisotropy();
    const cleanTexture = makeTransparentTexture(texture.image);
    buildPortrait(cleanTexture);
    resize();
    animate();
  });
}

async function getPortraitTextureUrl() {
  try {
    const response = await fetch("./portrait.b64", { cache: "force-cache" });
    if (response.ok) {
      const base64 = (await response.text()).trim();
      if (base64) {
        const dataUrl = `data:image/jpeg;base64,${base64}`;
        if (portraitPreview) portraitPreview.src = dataUrl;
        return dataUrl;
      }
    }
  } catch {
    // Local file previews cannot always fetch sibling text files.
  }

  const fallbackUrl = "./assets/portrait.png";
  if (portraitPreview) portraitPreview.src = fallbackUrl;
  return fallbackUrl;
}

function makeTransparentTexture(image) {
  const canvas = document.createElement("canvas");
  canvas.width = image.naturalWidth || image.width;
  canvas.height = image.naturalHeight || image.height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(image, 0, 0);
  const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height);

  for (let index = 0; index < pixels.data.length; index += 4) {
    const r = pixels.data[index];
    const g = pixels.data[index + 1];
    const b = pixels.data[index + 2];
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const brightNeutral = max > 184 && max - min < 34;
    if (brightNeutral) pixels.data[index + 3] = 0;
  }

  ctx.putImageData(pixels, 0, 0);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = renderer.capabilities.getMaxAnisotropy();
  return tex;
}

function buildPortrait(texture) {
  portraitAspect = texture.image.width / texture.image.height;
  const geometry = createFoldGeometry(portraitAspect);
  const material = new THREE.MeshStandardMaterial({
    map: texture,
    transparent: true,
    alphaTest: 0.02,
    roughness: 0.74,
    metalness: 0.02,
    side: THREE.DoubleSide,
    flatShading: true,
  });

  portraitMesh = new THREE.Mesh(geometry, material);
  portraitMesh.scale.set(1, 1, 1);
  portraitMesh.rotation.y = -0.06;
  group.add(portraitMesh);

  const wireMaterial = new THREE.MeshBasicMaterial({
    color: 0xffd6b4,
    transparent: true,
    opacity: 0.58,
    wireframe: true,
    depthWrite: false,
  });
  wireMesh = new THREE.Mesh(geometry.clone(), wireMaterial);
  wireMesh.scale.copy(portraitMesh.scale);
  wireMesh.rotation.copy(portraitMesh.rotation);
  wireMesh.visible = false;
  group.add(wireMesh);
}

function createFoldGeometry(aspect = portraitAspect) {
  const rows = 44;
  const cols = 32;
  const height = 4.72;
  const width = height * aspect;
  const positions = [];
  const uvs = [];
  const indices = [];
  const pointIndex = new Map();
  geometryBase = [];

  for (let y = 0; y <= rows; y += 1) {
    for (let x = 0; x <= cols; x += 1) {
      const u = x / cols;
      const v = 1 - y / rows;

      const px = (u - 0.5) * width;
      const py = (v - 0.5) * height;
      const creaseA = Math.sin((u * 5.7 + v * 3.1 + state.seed) * Math.PI);
      const creaseB = Math.cos((u * 9.2 - v * 4.8 + state.seed * 0.37) * Math.PI);
      const wrinkle = Math.sin((u * 31 + v * 19 + state.seed) * Math.PI);
      const theta = hash(u * 8.1, v * 5.7, state.seed) * Math.PI * 2;
      const phi = Math.acos(hash(u * 13.3, v * 9.4, state.seed + 4) * 2 - 1);
      const noise = hash(u * 41.9, v * 23.7, state.seed + 11);
      const z = creaseA * 0.025 + creaseB * 0.018;
      pointIndex.set(`${x},${y}`, positions.length / 3);
      geometryBase.push({ x: px, y: py, z, u, v, phase: creaseA, creaseA, creaseB, wrinkle, theta, phi, noise });
      positions.push(px, py, z);
      uvs.push(u, v);
    }
  }

  for (let y = 0; y < rows; y += 1) {
    for (let x = 0; x < cols; x += 1) {
      const a = pointIndex.get(`${x},${y}`);
      const b = pointIndex.get(`${x + 1},${y}`);
      const c = pointIndex.get(`${x},${y + 1}`);
      const d = pointIndex.get(`${x + 1},${y + 1}`);
      if ([a, b, c, d].some((value) => value === undefined)) continue;

      if ((x + y + state.seed) % 2 === 0) {
        indices.push(a, c, b, b, c, d);
      } else {
        indices.push(a, c, d, a, d, b);
      }
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}

function updateFold(time) {
  if (!portraitMesh) return;

  const position = portraitMesh.geometry.attributes.position;
  const wirePosition = wireMesh.geometry.attributes.position;
  const fold = state.fold;
  const depth = state.depth;
  const interaction = getInteraction();
  const crumple = clamp(interaction.crumple || 0, 0, 1);
  const handPress = clamp(interaction.pressure || 0, 0, 1);
  const radius = state.hand.active ? 0.16 : 0.11;
  const centerX = (interaction.x - 0.5) * 3.1;
  const centerY = (interaction.y - 0.5) * 4.28;
  const ballRadius = 0.48 + (1 - depth) * 0.16;

  for (let i = 0; i < geometryBase.length; i += 1) {
    const p = geometryBase[i];
    const dx = p.u - interaction.x;
    const dy = p.v - interaction.y;
    const localPressure = Math.exp(-(dx * dx + dy * dy) / radius) * handPress;
    const paperRidge =
      Math.sin((p.u * 34 + p.v * 21 + state.seed) * Math.PI) *
      Math.cos((p.u * 18 - p.v * 29) * Math.PI);
    const pinch = Math.sin((dx * 22 - dy * 17 + state.seed) * Math.PI) * localPressure;
    const breath = Math.sin(time * 0.0014 + p.phase * 2.3) * 0.035;
    const sheetZ =
      p.z +
      (p.creaseA * 0.2 + p.creaseB * 0.12 + p.wrinkle * 0.06) * fold * depth * (1 - crumple * 0.65) +
      (paperRidge * 0.62 + p.creaseA * 0.38 + pinch) * localPressure * (1 - crumple * 0.28) +
      breath;
    const hinge = Math.sin((p.u - 0.5) * Math.PI) * fold * 0.04 * (1 - crumple);
    const pull = localPressure * 0.18;
    const sheetX = p.x + hinge * p.y - dx * pull;
    const sheetY =
      p.y +
      Math.cos((p.v + state.seed) * Math.PI * 3) * fold * 0.02 -
      dy * pull * 1.4;

    const sx = Math.cos(p.theta) * Math.sin(p.phi);
    const sy = Math.sin(p.theta) * Math.sin(p.phi);
    const sz = Math.cos(p.phi);
    const crushedRadius = ballRadius * (0.68 + p.noise * 0.52);
    const crinkle = (paperRidge * 0.32 + p.creaseA * 0.2 + p.wrinkle * 0.12) * depth;
    const ballX = centerX + sx * crushedRadius * (0.86 + p.noise * 0.1);
    const ballY = centerY + sy * crushedRadius;
    const ballZ = sz * crushedRadius + crinkle;
    const easedCrumple = crumple * crumple * (3 - 2 * crumple);
    const x = lerp(sheetX, ballX, easedCrumple);
    const y = lerp(sheetY, ballY, easedCrumple);
    const z = lerp(sheetZ, ballZ, easedCrumple);

    position.setXYZ(i, x, y, z);
    wirePosition.setXYZ(i, x, y, z + 0.006);
  }

  position.needsUpdate = true;
  wirePosition.needsUpdate = true;
  portraitMesh.geometry.computeVertexNormals();
}

function getInteraction() {
  if (state.hand.active) return state.hand;
  if (state.pointer.active) return state.pointer;
  return { active: false, x: 0.54, y: 0.56, pressure: 0, crumple: 0 };
}

function resize() {
  const { clientWidth, clientHeight } = container;
  renderer.setSize(clientWidth, clientHeight);
  camera.aspect = clientWidth / clientHeight;
  camera.updateProjectionMatrix();
  const mobile = clientWidth < 620;
  group.position.set(0, mobile ? 0.02 : -0.03, 0);
  group.scale.setScalar(mobile ? 0.9 : 1);
}

function animate(time = 0) {
  detectHand(time);
  updateFold(time);
  group.rotation.x += shortestAngle(group.rotation.x, state.targetX) * 0.075;
  group.rotation.y += shortestAngle(group.rotation.y, state.targetY) * 0.075;
  group.rotation.z += shortestAngle(group.rotation.z, state.targetZ) * 0.08;
  updateLiveCode();
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}

window.addEventListener("resize", resize);
window.addEventListener("pointermove", (event) => {
  if (state.mouseFallback && !state.hand.active) {
    state.targetY = (event.clientX / window.innerWidth - 0.5) * 0.5;
    state.targetX = -(event.clientY / window.innerHeight - 0.5) * 0.32;
  }
  const rect = container.getBoundingClientRect();
  const inScene =
    event.clientX >= rect.left &&
    event.clientX <= rect.right &&
    event.clientY >= rect.top &&
    event.clientY <= rect.bottom;
  state.pointer.active = state.mouseFallback && inScene && !state.hand.active;
  if (inScene) {
    state.pointer.x = clamp((event.clientX - rect.left) / rect.width, 0, 1);
    state.pointer.y = clamp(1 - (event.clientY - rect.top) / rect.height, 0, 1);
    state.pointer.pressure += (0.34 - state.pointer.pressure) * 0.2;
    state.pointer.crumple += (0.18 - state.pointer.crumple) * 0.12;
  }
});

window.addEventListener("pointerleave", () => {
  state.pointer.active = false;
});

startCameraButton.addEventListener("click", startHandTracking);

async function startHandTracking() {
  if (trackingStarted) {
    stopHandTracking();
    return;
  }

  if (window.location.protocol === "file:") {
    startCameraButton.textContent = "Abrir con servidor";
    setHandStatus("Abre con servidor", true);
    return;
  }

  if (!navigator.mediaDevices?.getUserMedia) {
    setHandStatus("Camara no disponible", true);
    return;
  }

  setHandStatus("Cargando", true);
  startCameraButton.disabled = true;

  try {
    const { visionModule, vision, modelAssetPath } = await loadHandVision();
    handLandmarker = await createHandLandmarker(visionModule.HandLandmarker, vision, modelAssetPath);
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        width: { ideal: 720 },
        height: { ideal: 540 },
        facingMode: "user",
      },
    });
    handVideo.srcObject = stream;
    await handVideo.play();
    trackingStarted = true;
    preview.classList.add("is-live");
    startCameraButton.textContent = "Detener mano";
    startCameraButton.disabled = false;
    setHandStatus("Busca tu mano", true);
  } catch (error) {
    console.error(error);
    state.hand.active = false;
    state.pointer.active = false;
    trackingStarted = false;
    startCameraButton.disabled = false;
    startCameraButton.textContent = "Detectar mano";
    setHandStatus(getCameraErrorMessage(error), true);
  }
}

async function loadHandVision() {
  try {
    const visionModule = await import("./assets/mediapipe/vision_bundle.mjs");
    const vision = await visionModule.FilesetResolver.forVisionTasks("./assets/mediapipe/wasm");
    return {
      visionModule,
      vision,
      modelAssetPath: "./assets/mediapipe/hand_landmarker.task",
    };
  } catch {
    const visionModule = await import(`${MEDIAPIPE_CDN}/vision_bundle.mjs`);
    const vision = await visionModule.FilesetResolver.forVisionTasks(`${MEDIAPIPE_CDN}/wasm`);
    return {
      visionModule,
      vision,
      modelAssetPath: HAND_MODEL_CDN,
    };
  }
}

async function createHandLandmarker(HandLandmarker, vision, modelAssetPath) {
  const options = {
    baseOptions: {
      modelAssetPath,
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.55,
    minHandPresenceConfidence: 0.55,
    minTrackingConfidence: 0.5,
  };

  try {
    return await HandLandmarker.createFromOptions(vision, options);
  } catch {
    return HandLandmarker.createFromOptions(vision, {
      ...options,
      baseOptions: { ...options.baseOptions, delegate: "CPU" },
    });
  }
}

function stopHandTracking() {
  const stream = handVideo.srcObject;
  if (stream) stream.getTracks().forEach((track) => track.stop());
  handVideo.srcObject = null;
  trackingStarted = false;
  state.hand.active = false;
  state.twoHands.active = false;
  state.hand.crumple = 0;
  state.hand.pressure = 0;
  state.targetX = 0;
  state.targetY = 0;
  state.targetZ = 0;
  state.handsDetected = 1;
  state.mode = "bolita";
  state.hand.active = true;
  state.hand.x = 0.5;
  state.hand.y = 0.52;
  state.hand.pressure = 0.82;
  state.hand.crumple = 0.96;
  state.hand.openness = 0.05;
  preview.classList.remove("is-live");
  handCtx.clearRect(0, 0, handCanvas.width, handCanvas.height);
  startCameraButton.textContent = "Detectar mano";
  setHandStatus("Listo");
}

function detectHand(time) {
  if (!trackingStarted || !handLandmarker || handVideo.readyState < 2) return;
  if (handVideo.currentTime === lastVideoTime) return;
  lastVideoTime = handVideo.currentTime;

  const result = handLandmarker.detectForVideo(handVideo, time);
  const hands = result.landmarks || [];
  state.handsDetected = hands.length;
  drawHands(hands);

  if (!hands.length) {
    state.hand.active = false;
    state.twoHands.active = false;
    state.mode = "buscando";
    state.hand.crumple += (0 - state.hand.crumple) * 0.12;
    state.targetX += shortestAngle(state.targetX, 0) * 0.08;
    state.targetY += shortestAngle(state.targetY, 0) * 0.08;
    state.targetZ += shortestAngle(state.targetZ, 0) * 0.08;
    setHandStatus("Buscando");
    return;
  }

  const analyzed = hands.map(analyzeHand);

  if (analyzed.length >= 2) {
    updateTwoHandRotation(analyzed);
    return;
  }

  state.twoHands.active = false;
  updateSingleHand(analyzed[0]);
}

function analyzeHand(landmarks) {
  const palm = averagePoints([landmarks[0], landmarks[5], landmarks[9], landmarks[13], landmarks[17]]);
  const bounds = getHandBounds(landmarks);
  const scale = Math.max(bounds.width, bounds.height);
  const palmSize = Math.max(distance(landmarks[5], landmarks[17]), 0.001);
  const fingerTips = [landmarks[4], landmarks[8], landmarks[12], landmarks[16], landmarks[20]];
  const tipDistance = fingerTips.reduce((sum, point) => sum + distance(point, palm), 0) / fingerTips.length;
  const fingertipSpread = getHandBounds(fingerTips);
  const extensionRatio = tipDistance / palmSize;
  const spreadRatio = Math.max(fingertipSpread.width, fingertipSpread.height) / palmSize;
  const openness = clamp((extensionRatio - 1.25) / 1.25 + spreadRatio * 0.16, 0, 1);
  const fist = clamp(1 - openness, 0, 1);
  const pressure = clamp(0.12 + fist * 0.9 + (scale - 0.18) * 0.55, 0, 1);

  return {
    landmarks,
    palm,
    sceneX: 1 - palm.x,
    sceneY: 1 - palm.y,
    openness,
    fist,
    pressure,
    scale,
  };
}

function updateSingleHand(hand) {
  state.mode = hand.fist > 0.68 ? "bolita" : hand.openness > 0.68 ? "desplegado" : "plegando";
  state.hand.active = true;
  state.hand.x += (hand.sceneX - state.hand.x) * 0.36;
  state.hand.y += (hand.sceneY - state.hand.y) * 0.36;
  state.hand.pressure += (hand.pressure - state.hand.pressure) * 0.28;
  state.hand.crumple += (hand.fist - state.hand.crumple) * 0.32;
  state.hand.openness += (hand.openness - state.hand.openness) * 0.32;
  state.targetX += shortestAngle(state.targetX, 0) * 0.05;
  state.targetY += shortestAngle(state.targetY, 0) * 0.05;
  state.targetZ += shortestAngle(state.targetZ, 0) * 0.05;
  state.pointer.active = false;
  setHandStatus(hand.fist > 0.68 ? "Bolita" : hand.openness > 0.68 ? "Abierta" : "Plegando");
}

function updateTwoHandRotation(hands) {
  state.mode = "rotacion 3D";
  const sorted = [...hands].sort((a, b) => a.sceneX - b.sceneX);
  const left = sorted[0];
  const right = sorted[1];
  const centerX = (left.sceneX + right.sceneX) * 0.5;
  const centerY = (left.sceneY + right.sceneY) * 0.5;
  const dx = right.sceneX - left.sceneX;
  const dy = right.sceneY - left.sceneY;
  const span = Math.hypot(dx, dy);
  const angle = Math.atan2(dy, dx);
  const depthTwist = clamp((left.palm.z - right.palm.z) * 7, -1, 1);
  const averageOpenness = (left.openness + right.openness) * 0.5;
  const targetPitch = clamp((centerY - 0.5) * 1.45, -0.95, 0.95);
  const targetYaw = clamp((centerX - 0.5) * 1.4 + depthTwist * 0.85, -1.15, 1.15);
  const targetRoll = clamp(angle, -1.35, 1.35);

  state.twoHands.active = true;
  state.twoHands.span += (span - state.twoHands.span) * 0.24;
  state.twoHands.angle += shortestAngle(state.twoHands.angle, angle) * 0.24;
  state.hand.active = true;
  state.hand.x += (centerX - state.hand.x) * 0.3;
  state.hand.y += (centerY - state.hand.y) * 0.3;
  state.hand.pressure += (0.18 - state.hand.pressure) * 0.24;
  state.hand.crumple += (0 - state.hand.crumple) * 0.2;
  state.hand.openness += (averageOpenness - state.hand.openness) * 0.24;
  state.targetX += shortestAngle(state.targetX, targetPitch) * 0.24;
  state.targetY += shortestAngle(state.targetY, targetYaw) * 0.24;
  state.targetZ += shortestAngle(state.targetZ, targetRoll) * 0.24;
  state.pointer.active = false;
  setHandStatus("Girando");
}

function drawHands(hands) {
  const width = handVideo.videoWidth || 1;
  const height = handVideo.videoHeight || 1;
  if (handCanvas.width !== width || handCanvas.height !== height) {
    handCanvas.width = width;
    handCanvas.height = height;
  }

  handCtx.clearRect(0, 0, width, height);
  if (!hands.length) return;

  handCtx.lineWidth = Math.max(3, width * 0.006);
  handCtx.strokeStyle = "rgba(255, 214, 180, 0.92)";
  handCtx.fillStyle = "rgba(255, 255, 255, 0.95)";

  hands.forEach((landmarks) => {
    handConnections.forEach(([a, b]) => {
      handCtx.beginPath();
      handCtx.moveTo(landmarks[a].x * width, landmarks[a].y * height);
      handCtx.lineTo(landmarks[b].x * width, landmarks[b].y * height);
      handCtx.stroke();
    });

    landmarks.forEach((point) => {
      handCtx.beginPath();
      handCtx.arc(point.x * width, point.y * height, Math.max(4, width * 0.008), 0, Math.PI * 2);
      handCtx.fill();
    });
  });
}

function getCameraErrorMessage(error) {
  const name = error?.name || "";
  if (name === "NotAllowedError" || name === "PermissionDeniedError") return "Permite camara";
  if (name === "NotFoundError" || name === "DevicesNotFoundError") return "Sin camara";
  if (window.location.protocol === "file:") return "Abre con servidor";
  return "No cargo camara";
}

function setHandStatus(text, visible = false) {
  handStatus.textContent = text;
  preview.classList.toggle("show-status", visible);
}

function updateLiveCode() {
  if (!liveCode) return;
  liveCode.textContent = `const mano = ${Math.min(state.handsDetected, 1)};
const manos = ${state.handsDetected};
const modo = "${state.mode}";

// 1 mano: cerrar = bolita, abrir = desplegar
papel.arruga = ${state.hand.crumple.toFixed(2)};
papel.presion = ${state.hand.pressure.toFixed(2)};

// 2 manos: rotacion libre del retrato
portrait.rotation.x = ${group.rotation.x.toFixed(2)};
portrait.rotation.y = ${group.rotation.y.toFixed(2)};
portrait.rotation.z = ${group.rotation.z.toFixed(2)};

if (mano === 1) {
  deformarFoto(papel.arruga);
}

if (manos === 2) {
  girarEnTresEjes(x, y, z);
}`;
}

function averagePoints(points) {
  const total = points.reduce(
    (sum, point) => ({ x: sum.x + point.x, y: sum.y + point.y, z: sum.z + (point.z || 0) }),
    { x: 0, y: 0, z: 0 },
  );
  return {
    x: total.x / points.length,
    y: total.y / points.length,
    z: total.z / points.length,
  };
}

function getHandBounds(points) {
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  return {
    width: Math.max(...xs) - Math.min(...xs),
    height: Math.max(...ys) - Math.min(...ys),
  };
}

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function shortestAngle(from, to) {
  return Math.atan2(Math.sin(to - from), Math.cos(to - from));
}

function hash(x, y, seed) {
  return fract(Math.sin(x * 12.9898 + y * 78.233 + seed * 37.719) * 43758.5453);
}

function fract(value) {
  return value - Math.floor(value);
}
