const statusEl = document.querySelector("#webar-status");
const handCanvas = document.querySelector("#hand-canvas");
const handCtx = handCanvas.getContext("2d");
const PORTRAIT_BASE64_PATH = "./portrait.b64";
const MEDIAPIPE_VERSION = "0.10.14";
const MEDIAPIPE_CDN = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
const HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

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

resizeHandCanvas();
window.addEventListener("resize", resizeHandCanvas);

registerWhenReady();

function registerWhenReady() {
  if (!window.AFRAME?.THREE) {
    setStatus("Cargando WebAR");
    window.setTimeout(registerWhenReady, 80);
    return;
  }

  const THREE = window.AFRAME.THREE;

  window.AFRAME.registerComponent("fold-portrait", {
    init() {
      this.THREE = THREE;
      this.group = new THREE.Group();
      this.group.position.set(0, 0.05, 0);
      this.el.object3D.add(this.group);

      this.state = {
        fold: 0.72,
        depth: 0.62,
        seed: 7,
        markerVisible: false,
        targetX: 0,
        targetY: 0,
        targetZ: 0,
        hand: { active: false, x: 0.5, y: 0.52, pressure: 0, crumple: 0, openness: 1 },
      };

      this.geometryBase = [];
      this.lastVideoTime = -1;
      this.lastDetectAt = 0;
      this.video = null;
      this.ready = false;
      this.handLandmarker = null;

      this.el.addEventListener("markerFound", () => {
        this.state.markerVisible = true;
        setStatus("Marcador detectado. Muestra tu mano");
      });

      this.el.addEventListener("markerLost", () => {
        this.state.markerVisible = false;
        setStatus("Apunta al marcador Hiro");
      });

      this.loadPortrait();
      this.initHands();
    },

    async loadPortrait() {
      try {
        const response = await fetch(PORTRAIT_BASE64_PATH);
        if (!response.ok) throw new Error(`Portrait request failed: ${response.status}`);
        const portraitUrl = `data:image/jpeg;base64,${(await response.text()).trim()}`;
        const loader = new this.THREE.TextureLoader();
        loader.load(
          portraitUrl,
          (texture) => {
            applyColorSpace(texture, this.THREE);
            const cleanTexture = makeTransparentTexture(texture.image, this.THREE);
            const geometry = createFoldGeometry(this, texture.image.width / texture.image.height);
            const material = new this.THREE.MeshBasicMaterial({
              map: cleanTexture,
              transparent: true,
              alphaTest: 0.02,
              side: this.THREE.DoubleSide,
            });

            this.portraitMesh = new this.THREE.Mesh(geometry, material);
            this.portraitMesh.rotation.x = -Math.PI * 0.5;
            this.group.add(this.portraitMesh);
            this.ready = true;
            setStatus("Foto cargada. Apunta al marcador Hiro");
          },
          undefined,
          (error) => {
            console.error(error);
            setStatus("No pude cargar tu foto");
          },
        );
      } catch (error) {
        console.error(error);
        setStatus("No pude cargar tu foto");
      }
    },

    async initHands() {
      try {
        const visionModule = await import(`${MEDIAPIPE_CDN}/vision_bundle.mjs`);
        const vision = await visionModule.FilesetResolver.forVisionTasks(`${MEDIAPIPE_CDN}/wasm`);
        this.handLandmarker = await visionModule.HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: HAND_MODEL_URL,
            delegate: "CPU",
          },
          runningMode: "VIDEO",
          numHands: 2,
          minHandDetectionConfidence: 0.52,
          minHandPresenceConfidence: 0.52,
          minTrackingConfidence: 0.46,
        });
        setStatus("Apunta al marcador Hiro");
      } catch (error) {
        console.error(error);
        setStatus("No pude cargar deteccion de mano");
      }
    },

    tick(time) {
      if (!this.ready) return;
      this.detectHands(time);
      this.updatePaper(time);
      this.group.rotation.x += shortestAngle(this.group.rotation.x, this.state.targetX) * 0.08;
      this.group.rotation.y += shortestAngle(this.group.rotation.y, this.state.targetY) * 0.08;
      this.group.rotation.z += shortestAngle(this.group.rotation.z, this.state.targetZ) * 0.09;
    },

    detectHands(time) {
      if (!this.handLandmarker || time - this.lastDetectAt < 92) return;
      this.lastDetectAt = time;
      this.video = this.video?.readyState >= 2 ? this.video : findCameraVideo();
      if (!this.video || this.video.readyState < 2 || !this.video.videoWidth) return;
      if (this.video.currentTime === this.lastVideoTime) return;
      this.lastVideoTime = this.video.currentTime;

      const result = this.handLandmarker.detectForVideo(this.video, time);
      const hands = result.landmarks || [];
      drawHands(hands);

      if (!hands.length) {
        this.state.hand.active = false;
        this.state.hand.crumple += (0 - this.state.hand.crumple) * 0.12;
        this.state.targetX += shortestAngle(this.state.targetX, 0) * 0.08;
        this.state.targetY += shortestAngle(this.state.targetY, 0) * 0.08;
        this.state.targetZ += shortestAngle(this.state.targetZ, 0) * 0.08;
        if (this.state.markerVisible) setStatus("Marcador detectado. Muestra tu mano");
        return;
      }

      const analyzed = hands.map(analyzeHand);
      if (analyzed.length >= 2) {
        updateTwoHandRotation(this.state, analyzed);
        setStatus("Dos manos: girando el retrato");
        return;
      }

      updateSingleHand(this.state, analyzed[0]);
      setStatus(
        analyzed[0].fist > 0.68 ? "Mano cerrada: bolita de papel" : analyzed[0].openness > 0.68 ? "Mano abierta: desplegado" : "Plegando",
      );
    },

    updatePaper(time) {
      if (!this.portraitMesh) return;
      const geometry = this.portraitMesh.geometry;
      const position = geometry.attributes.position;
      const interaction = this.state.hand.active
        ? this.state.hand
        : { active: false, x: 0.5, y: 0.52, pressure: 0, crumple: 0 };
      const crumple = clamp(interaction.crumple || 0, 0, 1);
      const handPress = clamp(interaction.pressure || 0, 0, 1);
      const centerX = (interaction.x - 0.5) * 0.8;
      const centerY = (interaction.y - 0.5) * 1.08;
      const ballRadius = 0.13 + (1 - this.state.depth) * 0.04;

      for (let i = 0; i < this.geometryBase.length; i += 1) {
        const p = this.geometryBase[i];
        const dx = p.u - interaction.x;
        const dy = p.v - interaction.y;
        const localPressure = Math.exp(-(dx * dx + dy * dy) / 0.16) * handPress;
        const paperRidge =
          Math.sin((p.u * 34 + p.v * 21 + this.state.seed) * Math.PI) *
          Math.cos((p.u * 18 - p.v * 29) * Math.PI);
        const breath = Math.sin(time * 0.0014 + p.phase * 2.3) * 0.006;
        const sheetZ =
          p.z +
          (p.creaseA * 0.04 + p.creaseB * 0.026 + p.wrinkle * 0.012) *
            this.state.fold *
            this.state.depth *
            (1 - crumple * 0.65) +
          (paperRidge * 0.13 + p.creaseA * 0.08) * localPressure * (1 - crumple * 0.28) +
          breath;
        const hinge = Math.sin((p.u - 0.5) * Math.PI) * this.state.fold * 0.012 * (1 - crumple);
        const sheetX = p.x + hinge * p.y - dx * localPressure * 0.052;
        const sheetY = p.y + Math.cos((p.v + this.state.seed) * Math.PI * 3) * this.state.fold * 0.006 - dy * localPressure * 0.068;
        const sx = Math.cos(p.theta) * Math.sin(p.phi);
        const sy = Math.sin(p.theta) * Math.sin(p.phi);
        const sz = Math.cos(p.phi);
        const crushedRadius = ballRadius * (0.68 + p.noise * 0.52);
        const ballX = centerX + sx * crushedRadius * (0.86 + p.noise * 0.1);
        const ballY = centerY + sy * crushedRadius;
        const ballZ = sz * crushedRadius + (paperRidge * 0.06 + p.creaseA * 0.044 + p.wrinkle * 0.026) * this.state.depth;
        const t = crumple * crumple * (3 - 2 * crumple);
        position.setXYZ(i, lerp(sheetX, ballX, t), lerp(sheetY, ballY, t), lerp(sheetZ, ballZ, t));
      }

      position.needsUpdate = true;
      geometry.computeVertexNormals();
    },
  });

  const marker = document.querySelector("#hiro-marker");
  if (marker && !marker.components?.["fold-portrait"]) {
    marker.setAttribute("fold-portrait", "");
  }
}

function createFoldGeometry(component, aspect) {
  const THREE = component.THREE;
  const rows = 44;
  const cols = 32;
  const height = 1.16;
  const width = height * aspect;
  const positions = [];
  const uvs = [];
  const indices = [];
  const pointIndex = new Map();
  component.geometryBase = [];

  for (let y = 0; y <= rows; y += 1) {
    for (let x = 0; x <= cols; x += 1) {
      const u = x / cols;
      const v = 1 - y / rows;
      const px = (u - 0.5) * width;
      const py = (v - 0.5) * height;
      const creaseA = Math.sin((u * 5.7 + v * 3.1 + component.state.seed) * Math.PI);
      const creaseB = Math.cos((u * 9.2 - v * 4.8 + component.state.seed * 0.37) * Math.PI);
      const wrinkle = Math.sin((u * 31 + v * 19 + component.state.seed) * Math.PI);
      const theta = hash(u * 8.1, v * 5.7, component.state.seed) * Math.PI * 2;
      const phi = Math.acos(hash(u * 13.3, v * 9.4, component.state.seed + 4) * 2 - 1);
      const noise = hash(u * 41.9, v * 23.7, component.state.seed + 11);
      const z = creaseA * 0.005 + creaseB * 0.003;
      pointIndex.set(`${x},${y}`, positions.length / 3);
      component.geometryBase.push({ x: px, y: py, z, u, v, phase: creaseA, creaseA, creaseB, wrinkle, theta, phi, noise });
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
      if ((x + y + component.state.seed) % 2 === 0) indices.push(a, c, b, b, c, d);
      else indices.push(a, c, d, a, d, b);
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}

function makeTransparentTexture(image, THREE) {
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
    if (max > 184 && max - min < 34) pixels.data[index + 3] = 0;
  }

  ctx.putImageData(pixels, 0, 0);
  const texture = new THREE.CanvasTexture(canvas);
  applyColorSpace(texture, THREE);
  return texture;
}

function applyColorSpace(texture, THREE) {
  if ("colorSpace" in texture && THREE.SRGBColorSpace) texture.colorSpace = THREE.SRGBColorSpace;
  if ("encoding" in texture && THREE.sRGBEncoding) texture.encoding = THREE.sRGBEncoding;
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
  return { landmarks, palm, sceneX: palm.x, sceneY: 1 - palm.y, openness, fist, pressure };
}

function updateSingleHand(state, hand) {
  state.hand.active = true;
  state.hand.x += (hand.sceneX - state.hand.x) * 0.36;
  state.hand.y += (hand.sceneY - state.hand.y) * 0.36;
  state.hand.pressure += (hand.pressure - state.hand.pressure) * 0.28;
  state.hand.crumple += (hand.fist - state.hand.crumple) * 0.32;
  state.hand.openness += (hand.openness - state.hand.openness) * 0.32;
  state.targetX += shortestAngle(state.targetX, 0) * 0.05;
  state.targetY += shortestAngle(state.targetY, 0) * 0.05;
  state.targetZ += shortestAngle(state.targetZ, 0) * 0.05;
}

function updateTwoHandRotation(state, hands) {
  const sorted = [...hands].sort((a, b) => a.sceneX - b.sceneX);
  const left = sorted[0];
  const right = sorted[1];
  const centerX = (left.sceneX + right.sceneX) * 0.5;
  const centerY = (left.sceneY + right.sceneY) * 0.5;
  const dx = right.sceneX - left.sceneX;
  const dy = right.sceneY - left.sceneY;
  const angle = Math.atan2(dy, dx);
  const depthTwist = clamp((left.palm.z - right.palm.z) * 7, -1, 1);
  state.hand.active = true;
  state.hand.x += (centerX - state.hand.x) * 0.3;
  state.hand.y += (centerY - state.hand.y) * 0.3;
  state.hand.pressure += (0.18 - state.hand.pressure) * 0.24;
  state.hand.crumple += (0 - state.hand.crumple) * 0.2;
  state.targetX += shortestAngle(state.targetX, clamp((centerY - 0.5) * 1.45, -0.95, 0.95)) * 0.24;
  state.targetY += shortestAngle(state.targetY, clamp((centerX - 0.5) * 1.4 + depthTwist * 0.85, -1.15, 1.15)) * 0.24;
  state.targetZ += shortestAngle(state.targetZ, clamp(angle, -1.35, 1.35)) * 0.24;
}

function drawHands(hands) {
  handCtx.clearRect(0, 0, handCanvas.width, handCanvas.height);
  if (!hands.length) return;
  const width = handCanvas.width;
  const height = handCanvas.height;
  handCtx.lineWidth = Math.max(3, width * 0.0035);
  handCtx.strokeStyle = "rgba(255, 214, 180, 0.9)";
  handCtx.fillStyle = "rgba(255, 255, 255, 0.9)";

  hands.forEach((landmarks) => {
    handConnections.forEach(([a, b]) => {
      handCtx.beginPath();
      handCtx.moveTo(landmarks[a].x * width, landmarks[a].y * height);
      handCtx.lineTo(landmarks[b].x * width, landmarks[b].y * height);
      handCtx.stroke();
    });
    landmarks.forEach((point) => {
      handCtx.beginPath();
      handCtx.arc(point.x * width, point.y * height, Math.max(3, width * 0.0048), 0, Math.PI * 2);
      handCtx.fill();
    });
  });
}

function findCameraVideo() {
  return [...document.querySelectorAll("video")].find((video) => video.readyState >= 2 && video.videoWidth > 0);
}

function resizeHandCanvas() {
  handCanvas.width = window.innerWidth * window.devicePixelRatio;
  handCanvas.height = window.innerHeight * window.devicePixelRatio;
}

function setStatus(text) {
  statusEl.textContent = text;
}

function averagePoints(points) {
  const total = points.reduce(
    (sum, point) => ({ x: sum.x + point.x, y: sum.y + point.y, z: sum.z + (point.z || 0) }),
    { x: 0, y: 0, z: 0 },
  );
  return { x: total.x / points.length, y: total.y / points.length, z: total.z / points.length };
}

function getHandBounds(points) {
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  return { width: Math.max(...xs) - Math.min(...xs), height: Math.max(...ys) - Math.min(...ys) };
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
