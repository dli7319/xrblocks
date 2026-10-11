// app/capture.js
/**
 * Frame sources for the spatial-colocation demo.
 *
 * Every source exposes the same interface:
 *   kind       'camera' | 'synthetic'
 *   video      <video> (camera) or <canvas> (synthetic) element
 *   canvas     <canvas> painted every animation frame; getFrame() reads pixels from it
 *   getFrame() -> {data: Uint8ClampedArray (RGBA, row-major), width, height} | null
 *   getHeadPose() -> 4x4 T_ref_head (nested number[][]) or null when tracking is unknown
 *   stop()
 *
 * startSynthetic() renders a deterministic textured "room" (seeded random patches at varied
 * depths) from a slowly moving virtual camera via 2D canvas + getImageData. Because the
 * scene is a pure function of (seed, elapsed time), two browser tabs see the SAME room from
 * different viewpoints — exactly what cross-device relocalization needs. It backs
 * ?synthetic=1 and is the automatic fallback when getUserMedia is denied.
 */

const FOV_FALLBACK_DEG = 60; // keep in sync with lib/orb.js DEFAULT_FOV_DEG
const SYNTH_SEED = 0xc010ca7e; // fixed: every device must render the identical room
const SYNTH_PATCHES = 320;

/** Deterministic PRNG (mulberry32). */
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Unique high-contrast texture patch -> distinctive ORB descriptors. */
function makePatchTexture(rng) {
  const S = 96;
  const c = document.createElement('canvas');
  c.width = S;
  c.height = S;
  const g = c.getContext('2d');
  const cells = 5 + Math.floor(rng() * 5);
  const cell = S / cells;
  for (let gy = 0; gy < cells; gy++) {
    for (let gx = 0; gx < cells; gx++) {
      g.fillStyle = rng() < 0.5 ? '#000000' : '#ffffff';
      g.fillRect(gx * cell, gy * cell, cell + 1, cell + 1);
    }
  }
  // Border frame guarantees edge structure regardless of the random fill.
  g.strokeStyle = rng() < 0.5 ? '#ffffff' : '#000000';
  g.lineWidth = 5;
  g.strokeRect(3, 3, S - 6, S - 6);
  return c;
}

/** Deterministic scene: textured quads at varied depths in front of the map origin. */
function buildSyntheticScene() {
  const rng = mulberry32(SYNTH_SEED);
  const patches = [];
  for (let i = 0; i < SYNTH_PATCHES; i++) {
    patches.push({
      x: (rng() * 2 - 1) * 1.7,
      y: (rng() * 2 - 1) * 1.1,
      z: -1.6 - rng() * 2.0, // 1.6..3.6 m ahead (-z forward, map frame = kf0 head)
      size: 0.18 + rng() * 0.14, // meters
      tex: makePatchTexture(rng),
    });
  }
  return patches;
}

/** Slow, smooth virtual head trajectory. Periodic => any tab start time sees the room.
 * Includes roll: without rotation about the optical axis the focal anisotropy
 * (fy vs fx) is nearly unobservable and self-calibration drifts. */
function syntheticPoseAt(t) {
  return {
    x: 0.3 * Math.sin(t * 0.17),
    y: 0.12 * Math.sin(t * 0.11 + 0.6),
    z: 0.18 * Math.sin(t * 0.13 + 2.1),
    yaw: 0.12 * Math.sin(t * 0.29),
    pitch: 0.05 * Math.sin(t * 0.21 + 1.3),
    roll: 0.09 * Math.sin(t * 0.23 + 0.7),
  };
}

function norm3(v) {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}

function cross3(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

/** Camera axes for yaw/pitch/roll (radians): forward f (+z OpenCV-style in cam
 * frame), right r, down v. Roll rotates r/v about the optical axis. */
function cameraAxes(yaw, pitch, roll = 0) {
  const f = norm3([
    Math.sin(yaw) * Math.cos(pitch),
    Math.sin(pitch),
    -Math.cos(yaw) * Math.cos(pitch),
  ]);
  let r = norm3([-f[2], 0, f[0]]); // cross(f, worldUp) with worldUp = (0,1,0)
  let v = cross3(f, r); // points down in a y-up world
  if (roll) {
    const c = Math.cos(roll);
    const s = Math.sin(roll);
    const r2 = [r[0] * c + v[0] * s, r[1] * c + v[1] * s, r[2] * c + v[2] * s];
    const v2 = [v[0] * c - r[0] * s, v[1] * c - r[1] * s, v[2] * c - r[2] * s];
    r = r2;
    v = v2;
  }
  return {r, v, f};
}

/**
 * Animated synthetic camera replacement. Same interface as startCamera(), plus a real
 * getHeadPose() (the trajectory is ground truth), so the whole pipeline — keyframing,
 * triangulation, PnP relocalization, presence — works headlessly across two tabs.
 */
export function startSynthetic({
  width = 640,
  height = 480,
  fovDeg = FOV_FALLBACK_DEG,
} = {}) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d', {willReadFrequently: true});
  const scene = buildSyntheticScene();
  const fpx = width / (2 * Math.tan((fovDeg * Math.PI) / 360));
  const cx = width / 2;
  const cy = height / 2;
  const t0 = performance.now();
  let raf = 0;
  let running = true;
  // Exact frame/pose pairing: the pose the scene was drawn under, recorded at
  // paint time. Frames carry it so downstream geometry (mapping, calibration)
  // never mixes a frame with a pose from a different instant.
  let lastDrawTMs = 0;
  let lastDrawPose = null;

  function poseMatrixAt(t) {
    const p = syntheticPoseAt(t);
    const {r, v, f} = cameraAxes(p.yaw, p.pitch, p.roll);
    return [
      [r[0], v[0], f[0], p.x],
      [r[1], v[1], f[1], p.y],
      [r[2], v[2], f[2], p.z],
      [0, 0, 0, 1],
    ];
  }

  function draw() {
    if (!running) return;
    const t = (performance.now() - t0) / 1000;
    const p = syntheticPoseAt(t);
    const {r, v, f} = cameraAxes(p.yaw, p.pitch, p.roll);
    lastDrawTMs = t0 + t * 1000;
    lastDrawPose = poseMatrixAt(t);
    ctx.fillStyle = '#0b0d10'; // featureless background: only patches make ORB features
    ctx.fillRect(0, 0, width, height);
    const items = [];
    for (const s of scene) {
      const dx = s.x - p.x;
      const dy = s.y - p.y;
      const dz = s.z - p.z;
      const depth = dx * f[0] + dy * f[1] + dz * f[2];
      if (depth <= 0.3) continue;
      const px = cx + (fpx * (dx * r[0] + dy * r[1] + dz * r[2])) / depth;
      const py = cy + (fpx * (dx * v[0] + dy * v[1] + dz * v[2])) / depth;
      const half = (fpx * s.size) / (2 * depth);
      if (px < -half || px > width + half || py < -half || py > height + half)
        continue;
      items.push({px, py, half, depth, tex: s.tex});
    }
    // Painter's algorithm: occlusion is view-consistent across devices.
    items.sort((a, b) => b.depth - a.depth);
    for (const it of items) {
      ctx.drawImage(
        it.tex,
        it.px - it.half,
        it.py - it.half,
        it.half * 2,
        it.half * 2
      );
    }
    raf = requestAnimationFrame(draw);
  }
  draw();

  return {
    kind: 'synthetic',
    video: canvas,
    canvas,
    width,
    height,
    getFrame() {
      if (!running) return null;
      const img = ctx.getImageData(0, 0, width, height);
      return {
        data: img.data,
        width,
        height,
        tMs: lastDrawTMs,
        T_ref_head: lastDrawPose,
      };
    },
    getHeadPose() {
      return poseMatrixAt((performance.now() - t0) / 1000);
    },
    stop() {
      running = false;
      cancelAnimationFrame(raf);
    },
  };
}

/** getUserMedia-backed source. Frames are downscaled to <= `width` px wide RGBA canvases. */
export async function startCamera({width = 640} = {}) {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode: {ideal: 'environment'},
      width: {ideal: width},
      height: {ideal: 480},
    },
  });
  const video = document.createElement('video');
  video.setAttribute('playsinline', '');
  video.muted = true;
  video.autoplay = true;
  video.srcObject = stream;
  await video.play();

  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d', {willReadFrequently: true});
  let raf = 0;
  let running = true;
  // Wall-clock epoch for the video's media clock: frame wall time ≈ epoch +
  // video.currentTime*1000. Frames carry that timestamp so callers can pair
  // them with the pose from the matching instant (see lib/posehistory.js).
  let videoEpoch = 0;
  let lastDrawTMs = 0;

  function draw() {
    if (!running) return;
    if (video.videoWidth > 0) {
      if (!videoEpoch)
        videoEpoch = performance.now() - video.currentTime * 1000;
      lastDrawTMs = videoEpoch + video.currentTime * 1000;
      const w = Math.min(width, video.videoWidth);
      const h = Math.max(
        1,
        Math.round((w * video.videoHeight) / video.videoWidth)
      );
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
      ctx.drawImage(video, 0, 0, w, h);
    }
    raf = requestAnimationFrame(draw);
  }
  draw();

  return {
    kind: 'camera',
    video,
    canvas,
    getFrame() {
      if (!running || canvas.width === 0 || canvas.height === 0) return null;
      const img = ctx.getImageData(0, 0, canvas.width, canvas.height);
      return {
        data: img.data,
        width: canvas.width,
        height: canvas.height,
        tMs: lastDrawTMs || performance.now(),
      };
    },
    getHeadPose() {
      return null; // desktop / no-XR: caller falls back to SfM-derived head poses
    },
    stop() {
      running = false;
      cancelAnimationFrame(raf);
      stream.getTracks().forEach((tr) => tr.stop());
      video.srcObject = null;
    },
  };
}

/**
 * Best-effort WebXR immersive-ar head tracking layered on top of any frame source.
 * Returns a controller:
 *   supported  device reports immersive-ar support
 *   active     session running (renderer.xr.getCamera() drives getHeadPose())
 *   needsGesture  requestSession() must be called from a user gesture -> expose a button
 *   enable()   request the session (call from a click handler); true on success
 *   getHeadPose()  T_ref_head from the XR camera matrixWorld (local-floor), or null
 *   stop()
 * Never throws: if simultaneous camera+XR fails (expected risk on Android Chrome) the
 * caller keeps the camera source and falls back to SfM-derived head poses.
 */
export async function startXRIfNeeded(renderer) {
  const state = {
    supported: false,
    active: false,
    needsGesture: false,
    session: null,
    error: null,
  };

  async function adopt(session) {
    state.session = session;
    renderer.xr.enabled = true;
    try {
      state.refSpace = await session.requestReferenceSpace('local-floor');
    } catch (err) {
      state.refSpace = await session.requestReferenceSpace('viewer');
    }
    state.active = true;
    state.needsGesture = false;
    session.addEventListener('end', () => {
      state.active = false;
      state.session = null;
    });
  }

  state.enable = async function enable() {
    if (state.active) return true;
    try {
      const session = await navigator.xr.requestSession('immersive-ar', {
        optionalFeatures: ['local-floor'],
      });
      await adopt(session);
      return true;
    } catch (err) {
      state.error = String((err && err.message) || err);
      return false;
    }
  };

  state.getHeadPose = function getHeadPose() {
    if (!state.active) return null;
    try {
      const cam = renderer.xr.getCamera();
      if (!cam) return null;
      cam.updateMatrixWorld();
      const e = cam.matrixWorld.elements; // column-major THREE.Matrix4
      return [
        [e[0], e[4], e[8], e[12]],
        [e[1], e[5], e[9], e[13]],
        [e[2], e[6], e[10], e[14]],
        [0, 0, 0, 1],
      ];
    } catch (err) {
      return null;
    }
  };

  state.stop = function stop() {
    if (state.session) {
      try {
        state.session.end();
      } catch (err) {
        /* already ended */
      }
    }
    state.active = false;
    state.session = null;
  };

  try {
    if (!navigator.xr) {
      state.error = 'WebXR unavailable';
      return state;
    }
    state.supported = await navigator.xr.isSessionSupported('immersive-ar');
    if (!state.supported) {
      state.error = 'immersive-ar not supported here';
      return state;
    }
    const existing = renderer.xr.getSession();
    if (existing) {
      await adopt(existing);
    } else {
      state.needsGesture = true; // expose an "Enable XR" button
    }
  } catch (err) {
    state.error = String((err && err.message) || err);
  }
  return state;
}
