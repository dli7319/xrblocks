// app/main.js
/**
 * Spatial Colocation demo — app wiring.
 *
 * State machine: idle -> build | relocalize -> live, with the frame source either the
 * device camera or the synthetic procedural scene (?synthetic=1, or auto-fallback when
 * getUserMedia is denied). Pipeline: ORB extraction <=10 Hz -> keyframes -> SpatialMap
 * (lib/map.js) -> PnP relocalization (lib/relocalize.js) -> PeerJS presence.
 *
 * Worker A modules are imported exactly per the implementation spec (../lib/*.js).
 */
import {loadCv} from '../lib/cv-runtime.js';
import {PoseHistory} from '../lib/posehistory.js';
import {
  extractOrb,
  estimateIntrinsics,
  MAX_FEATURES,
  DEFAULT_FOV_DEG,
} from '../lib/orb.js';
import {matchDescriptors} from '../lib/matching.js';
import {
  estimateEssential,
  triangulate,
  poseFromRt,
  invertRigid,
  matMul,
} from '../lib/geometry.js';
import {
  createMap,
  addKeyframe,
  setMapOrigin,
  serializeMap,
  deserializeMap,
} from '../lib/map.js';
import {
  relocalize,
  headPoseFromCameraPose,
  MIN_RELOC_INLIERS,
} from '../lib/relocalize.js';
import {estimateHandEyeRotation} from '../lib/calibration.js';
import {OnlineCalibrator, KFromCalibrator} from '../lib/selfcalib.js';
import {startCamera, startSynthetic, startXRIfNeeded} from './capture.js';
import {createNet} from './net.js';
import * as store from './store.js';
import {Viz} from './viz.js';
import {UI} from './ui.js';

// Shared constants — values copied exactly from IMPLEMENTATION_SPEC.md.
const KEYFRAME_MIN_MS = 500;
const KEYFRAME_MIN_MOVE_M = 0.05;
const KEYFRAME_MIN_ROT_DEG = 3;
const PRESENCE_INTERVAL_MS = 100;
const MIN_PARALLAX_DEG = 1.0;
const PROCESS_INTERVAL_MS = 100; // feature extraction <= 10 Hz
const RELOC_MIN_INTERVAL_MS = 250; // PnP search <= 4 Hz, keeps the UI live
const SFM_STEP_M = 0.03; // per-frame translation guess in camera-only (no XR) mode
const POSE_TTL_MS = 4000;
const MAP_FALLBACK_MS = 3500;
const MAP_RETRY_MS = 5000;
const MIN_MATCHES_FOR_POSE = 12;
const CALIB_MIN_SAMPLES = 30;
const CALIB_STEP_INTERVAL_MS = 2000; // online self-calibration LM solve cadence
const CALIB_PAIR_MIN_ROT_DEG = 1.2; // long-baseline pairing: real epipolar leverage
const CALIB_PAIR_MIN_TRANS_M = 0.02;
const CALIB_MAX_RMS_PX = 8; // adoption plausibility gate

const IDENTITY = [
  [1, 0, 0, 0],
  [0, 1, 0, 0],
  [0, 0, 1, 0],
  [0, 0, 0, 1],
];

// Convention adapter for the online calibrator: `app/capture.js`'s synthetic
// `getHeadPose()` and the SfM pose chain express the camera frame OpenCV-style
// (x right, y down, z forward — see `cameraAxes` in capture.js), while
// SELF_CALIB_SPEC pins the calibrator to the three/WebXR convention (forward
// -Z, y up). For those sources the true `T_head_camera` therefore carries this
// fixed frame flip; composing it into the calibrator PRIOR keeps a local
// optimizer in the right basin (priors are initialization only) and makes the
// adopted extrinsics pipeline-correct (`T_ref_cam = T_ref_head · T_head_camera`
// then lands in three-convention for the cube projection / triangulation).
// XR head poses are already three-convention and use the prior as-is.
const CV_CAM_FLIP = [
  [1, 0, 0, 0],
  [0, -1, 0, 0],
  [0, 0, -1, 0],
  [0, 0, 0, 1],
];

function randId(n) {
  return Math.random()
    .toString(36)
    .slice(2, 2 + n);
}

function clone4(T) {
  return T.map((r) => r.slice());
}

function rot3(T) {
  return [
    [T[0][0], T[0][1], T[0][2]],
    [T[1][0], T[1][1], T[1][2]],
    [T[2][0], T[2][1], T[2][2]],
  ];
}

/** Rotation angle (deg) between two 4x4 poses. */
function rotationDegBetween(Ta, Tb) {
  const Ra = rot3(Ta);
  const Rb = rot3(Tb);
  // trace(Ra · Rb^T) = sum over i,j of Ra[i][j] * Rb[i][j]
  let tr = 0;
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) tr += Ra[i][j] * Rb[i][j];
  }
  const c = Math.min(1, Math.max(-1, (tr - 1) / 2));
  return (Math.acos(c) * 180) / Math.PI;
}

function translationDist(Ta, Tb) {
  return Math.hypot(
    Ta[0][3] - Tb[0][3],
    Ta[1][3] - Tb[1][3],
    Ta[2][3] - Tb[2][3]
  );
}

const params = new URLSearchParams(location.search);
const room =
  (params.get('room') || randId(6))
    .replace(/[^a-zA-Z0-9_-]/g, '')
    .slice(0, 24) || randId(6);
const startMode = params.get('mode');
const wantSynthetic = params.get('synthetic') === '1';
const fovDeg = params.get('fov')
  ? parseFloat(params.get('fov')) || DEFAULT_FOV_DEG
  : DEFAULT_FOV_DEG;
const label = (params.get('label') || `Dev-${randId(4)}`).slice(0, 24);
// Real-device camera feeds lag the clock by up to ~1 s (capture + ISP +
// transport). Frames carry media-clock timestamps and are paired with the pose
// of their capture instant; if a browser's media clock tracks ARRIVAL rather
// than capture, a constant bias remains — measure it via state.frameAgeMs and
// the calib line, then compensate here (?frameLagMs=N shifts frame times back).
const frameLagMs = Math.max(0, Number(params.get('frameLagMs')) || 0);

const state = {
  mode: 'idle',
  source: null,
  xr: null,
  K: null,
  KFrameW: 0,
  map: null,
  mapName: null,
  hasMap: false,
  T_head_camera: clone4(IDENTITY),
  RHeadCam: null, // calibrated rotation (optional refinement)
  tHeadCamManual: [0, 0, 0],
  extrinsicsManual: false,
  calib: {head: [], cam: [], lastH: null, done: false},
  lastEssR: null,
  lastKf: null, // {ts, T}
  kf0Pose: null, // builder's T_ref_head at first keyframe of a fresh map
  prevKps: null,
  prevDesc: null,
  sfm: null, // {T, prevKps} camera-only head-pose chain
  T_map_head: null,
  relocRef: null, // {T_map_head, T_ref_head} — propagate pose between relocalizations
  lastReloc: null, // {inliers, ts}
  matches: 0,
  calibrator: null, // OnlineCalibrator (lib/selfcalib.js) — online K + extrinsics
  calibFrameW: 0,
  calibKey: null, // frame width + pose-convention key for calibrator creation
  calibPriorF: 0, // prior focal (plausibility gate for adoption)
  headPoseCvFrame: null, // true when head poses use the OpenCV-style cam frame
  prevHeadPose: null,
  calibRef: null, // long-baseline pairing reference {kps, desc, headPose}
  poseHistory: new PoseHistory(),
  lastCalibStep: 0,
  calibConvergedLogged: false,
  poses: new Map(), // peerId -> {label, T_map_head, inliers, ts}
  cube: {placed: false, autoPlaced: false, mapPos: null},
  lastBroadcast: 0,
  lastMapReq: 0,
  mapRetryTimer: null,
  lastProcess: 0,
  lastRelocStep: 0,
  lastDeviceRender: 0,
  fps: 0,
  fpsSamples: [],
};

const helpers = {
  cv: null,
  estimateEssential,
  triangulate,
  matchDescriptors,
  MIN_PARALLAX_DEG,
};

// ---- UI ----------------------------------------------------------------------

const ui = new UI({
  onMode: (mode) => setMode(mode),
  onPlaceCube: () => placeCube(),
  onSourceToggle: () =>
    startSource(state.source?.kind === 'synthetic' ? 'camera' : 'synthetic'),
  onXrEnable: async () => {
    if (!state.xr) return;
    const ok = await state.xr.enable();
    ui.setXr(
      ok ? 'on' : `off (${state.xr.error || 'failed'})`,
      ok ? 'ok' : 'warn'
    );
    ui.showXrButton(false);
    if (ok) {
      ui.log('XR head tracking active (local-floor)');
      ui.setExtrinsicsStatus('XR on — collecting hand-eye pairs', 'ok');
    } else {
      ui.log(
        `XR failed: ${state.xr.error} — head pose ≈ camera pose (SfM chain)`,
        'warn'
      );
    }
  },
  onSaveMap: () => saveMapNow(),
  onLoadMap: (name) => loadMapByName(name),
  onExtrinsics: (t) => applyExtrinsics(t),
});

const viz = new Viz({
  container: document.getElementById('viz-host'),
  overlay: document.getElementById('overlay'),
});

// ---- networking --------------------------------------------------------------

const net = createNet({room, label, hasMap: false});

net.on('status', ({status, detail, isHost}) => {
  ui.setNet({status, detail, isHost, peerCount: net.peerCount});
  ui.log(
    `net: ${status}${detail ? ` — ${detail}` : ''}`,
    status === 'error' ? 'error' : ''
  );
  // The peer may connect after Relocalize mode asked for a map — ask again once online.
  if (status === 'joined' || status === 'host') requestMapSoon();
});
net.on('peers', (peers) => {
  ui.setNet({
    status: net.isHost ? 'host' : 'joined',
    peerCount: Math.max(0, peers.length - 1),
  });
});
net.on('pose', (msg) => {
  if (!msg || !msg.peerId || msg.peerId === net.peerId) return;
  state.poses.set(msg.peerId, {
    label: msg.label,
    T_map_head: msg.T_map_head,
    inliers: msg.inliers,
    ts: performance.now(),
  });
});
net.on('map-request', (msg) => {
  if (!state.hasMap || !state.map) return;
  try {
    const bytes = serializeMap(state.map);
    net.sendMap(bytes, state.mapName || `${room}-map`, msg.peerId);
    ui.log(`sent map (${bytes.length} B) to ${msg.peerId}`);
  } catch (err) {
    ui.log(`serialize failed: ${err.message}`, 'error');
  }
});
net.on('map-data', ({from, name, bytes}) => {
  try {
    const m = deserializeMap(bytes);
    adoptMap(m, name);
    ui.log(
      `map "${name}" received from ${from} (${m.landmarks?.length || 0} landmarks)`
    );
    ui.setBadge(`map: ${name}`);
  } catch (err) {
    ui.log(`bad map from ${from}: ${err.message}`, 'error');
  }
});
window.addEventListener('beforeunload', () => {
  net.disconnect();
  if (state.source) state.source.stop();
  if (state.xr) state.xr.stop();
});

// ---- mode machine ------------------------------------------------------------

/** Ask the room for a map (rate-limited; used on mode entry and on net connect). */
function requestMapSoon() {
  if (state.mode !== 'relocalize' || state.map) return;
  const now = performance.now();
  if (now - state.lastMapReq < 2000) return;
  state.lastMapReq = now;
  net.requestMap();
}

function setMode(mode) {
  if (!['build', 'relocalize', 'live'].includes(mode)) return;
  state.mode = mode;
  ui.setMode(mode);
  if (mode === 'build') {
    ui.setBadge('build: sweep slowly across textured surfaces');
    ui.log('Build: ORB -> keyframes -> triangulated landmark map');
  } else if (mode === 'relocalize') {
    acquireMapForReloc();
    ui.log('Relocalize: PnP against the stored map');
  } else if (mode === 'live') {
    if (!state.map) {
      ui.setBadge('live: no map yet — Relocalize first');
      ui.log(
        'Live without a map: broadcast starts after relocalization',
        'warn'
      );
    } else {
      ui.setBadge(
        state.T_map_head
          ? 'live: broadcasting head pose'
          : 'live: waiting for a pose'
      );
      ui.log('Live: broadcasting presence at 10 Hz');
    }
  }
}

// ---- frame source ------------------------------------------------------------

async function startSource(kind) {
  if (state.source) {
    state.source.stop();
    state.source = null;
  }
  try {
    state.source =
      kind === 'synthetic'
        ? startSynthetic({fovDeg})
        : await startCamera({width: 640});
  } catch (err) {
    if (kind === 'camera') {
      ui.log(
        `camera unavailable (${err.message}) — using synthetic feed`,
        'warn'
      );
      state.source = startSynthetic({fovDeg});
    } else {
      throw err;
    }
  }
  const host = document.getElementById('feed-host');
  host.textContent = '';
  state.source.canvas.classList.add('feed-canvas');
  host.appendChild(state.source.canvas);
  ui.setSource(
    state.source.kind,
    `${state.source.canvas.width}x${state.source.canvas.height}`
  );
  ui.setSourceButton(
    state.source.kind === 'synthetic' ? 'Use camera' : 'Use synthetic'
  );
  // Keep the feed pane's aspect ratio locked to the frame (canvas dims settle after play()).
  window.setTimeout(() => {
    const c = state.source && state.source.canvas;
    if (c && c.width > 0) {
      document.getElementById('feed-pane').style.aspectRatio =
        `${c.width} / ${c.height}`;
      ui.setSource(state.source.kind, `${c.width}x${c.height}`);
    }
  }, 400);
  // Frame dims may have changed -> recompute intrinsics; feature tracks don't transfer.
  state.K = null;
  state.KFrameW = 0;
  state.prevKps = null;
  state.prevDesc = null;
  state.sfm = null;
  // New source/camera setup -> a fresh online calibrator (priors re-derived
  // from the next ensureIntrinsics/estimateIntrinsics guess).
  state.calibrator = null;
  state.calibFrameW = 0;
  state.calibKey = null;
  state.prevHeadPose = null;
  state.calibRef = null;
  state.lastCalibStep = 0;
  state.calibConvergedLogged = false;
}

// ---- head pose ---------------------------------------------------------------

/** Maintain the SfM camera-pose chain (camera-only fallback + calibration pairs). */
function chainSfmPose(kps, matchesList) {
  state.lastEssR = null;
  if (!state.sfm) state.sfm = {T: clone4(IDENTITY), prevKps: null};
  if (
    state.sfm.prevKps &&
    matchesList.length >= MIN_MATCHES_FOR_POSE &&
    state.K
  ) {
    try {
      const ess = estimateEssential(
        state.cv,
        state.sfm.prevKps,
        kps,
        matchesList,
        state.K
      );
      if (ess && ess.R && ess.t) {
        const t = [
          ess.t[0] * SFM_STEP_M,
          ess.t[1] * SFM_STEP_M,
          ess.t[2] * SFM_STEP_M,
        ];
        // recoverPose: X_2 = R·X_1 + t  =>  T_cam1_cam2 = [R|t]; chain T_ref_cam2 = T_ref_cam1 · inv(T_cam1_cam2)
        const T_rel = poseFromRt(ess.R, t);
        state.sfm.T = matMul(state.sfm.T, invertRigid(T_rel));
        state.lastEssR = ess.R;
      }
    } catch (err) {
      /* keep previous pose */
    }
  }
  state.sfm.prevKps = kps;
  return state.sfm.T;
}

/** Optional hand-eye rotation calibration (XR head vs camera chain). */
function collectCalibration(headPose, camRel) {
  if (state.calib.done) return;
  const H = rot3(headPose);
  if (state.calib.lastH && camRel) {
    const A = matMul3T(state.calib.lastH, H); // R_head_prev^T · R_head_now
    state.calib.head.push(A);
    state.calib.cam.push(camRel);
    if (state.calib.head.length > 240) {
      state.calib.head.shift();
      state.calib.cam.shift();
    }
    if (
      state.calib.head.length >= CALIB_MIN_SAMPLES &&
      state.calib.head.length % CALIB_MIN_SAMPLES < 2
    ) {
      tryCalibrate();
    }
  }
  state.calib.lastH = H;
}

function matMul3T(A, B) {
  // A^T · B for 3x3 row-major matrices.
  const out = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      out[i][j] = A[0][i] * B[0][j] + A[1][i] * B[1][j] + A[2][i] * B[2][j];
    }
  }
  return out;
}

function tryCalibrate() {
  try {
    const res = estimateHandEyeRotation(state.calib.head, state.calib.cam);
    if (res && res.R_head_camera) {
      // The ONLINE estimate (selfcalib) drives T_head_camera once constrained;
      // the one-shot hand-eye rotation only fills in before that.
      const online = state.calibrator && state.calibrator.quality.constrained;
      if (!online) {
        state.RHeadCam = res.R_head_camera;
        if (!state.extrinsicsManual) rebuildHeadCamera();
      }
      ui.setExtrinsicsStatus(
        `calibrated ✓ ${res.samples ?? state.calib.head.length} samples`,
        'ok'
      );
      ui.log('hand-eye rotation calibrated — T_head_camera updated');
      state.calib.done = true;
    }
  } catch (err) {
    // Insufficient excitation or rejected data: identity extrinsics stay (documented).
    state.calib.done = true;
  }
}

function currentHeadPose(kps, matchesList, frame) {
  const sfmPose = chainSfmPose(kps, matchesList); // also feeds calibration camera increments
  // Prefer the pose at the FRAME's capture instant (exact for synthetic
  // sources, history-sampled for video): mixing a frame with a later pose
  // motion-smears every geometric constraint (mapping, calibration).
  const aligned = frameAlignedPose(frame);
  if (state.xr && state.xr.active) {
    const T = state.xr.getHeadPose();
    if (T) {
      state.headPoseCvFrame = false; // WebXR: three/WebXR convention
      collectCalibration(aligned || T, state.lastEssR);
      return aligned || T;
    }
  }
  if (state.source && state.source.kind === 'synthetic') {
    state.headPoseCvFrame = true; // capture.js synthetic: OpenCV-style cam frame
    return aligned || state.source.getHeadPose();
  }
  state.headPoseCvFrame = true; // SfM chain: OpenCV-style cam frame
  return aligned || sfmPose; // camera-only fallback: head pose ≈ camera pose (T_head_camera identity-ish)
}

/** Pose at the frame's capture instant, or null when unavailable. */
function frameAlignedPose(frame) {
  if (!frame) return null;
  if (frame.T_ref_head) return frame.T_ref_head;
  if (frame.tMs && state.poseHistory) {
    return state.poseHistory.sample(frame.tMs - frameLagMs);
  }
  return null;
}

function rebuildHeadCamera() {
  const R = state.RHeadCam || [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
  const t = state.tHeadCamManual;
  state.T_head_camera = [
    [R[0][0], R[0][1], R[0][2], t[0]],
    [R[1][0], R[1][1], R[1][2], t[1]],
    [R[2][0], R[2][1], R[2][2], t[2]],
    [0, 0, 0, 1],
  ];
}

function applyExtrinsics({tx, ty, tz}) {
  state.tHeadCamManual = [tx || 0, ty || 0, tz || 0];
  state.extrinsicsManual = true;
  rebuildHeadCamera();
  // The manual value becomes the calibrator PRIOR (and locks the lever arm in
  // adoptOnlineCalibration); the online estimate re-converges from data.
  if (state.calibrator) state.calibrator.reset(state.K, state.T_head_camera);
  state.lastCalibStep = 0;
  state.calibConvergedLogged = false;
  ui.setExtrinsicsStatus(
    `manual offset [${tx || 0}, ${ty || 0}, ${tz || 0}] m`,
    'ok'
  );
  ui.log('T_head_camera translation override applied');
}

// ---- processing --------------------------------------------------------------

function ensureIntrinsics(frame) {
  if (!state.K || state.KFrameW !== frame.width) {
    state.K = estimateIntrinsics(frame.width, frame.height, fovDeg);
    state.KFrameW = frame.width;
  }
  return state.K;
}

// ---- online self-calibration (lib/selfcalib.js) -----------------------------

/**
 * (Re)create the online calibrator on source toggles, frame-dimension changes
 * and pose-convention flips. Prior K = the estimateIntrinsics guess, prior T =
 * the manual/hand-eye override (with the CV_CAM_FLIP frame flip for OpenCV-
 * style pose sources) — BOTH initialization only; once `quality.constrained`
 * the online estimate drives state.K / state.T_head_camera.
 */
function ensureCalibrator(frame) {
  const key = `${frame.width}|${state.headPoseCvFrame ? 'cv' : 'three'}`;
  if (state.calibrator && state.calibKey === key) return state.calibrator;
  const priorT = state.headPoseCvFrame
    ? matMul(CV_CAM_FLIP, state.T_head_camera)
    : state.T_head_camera;
  state.calibrator = new OnlineCalibrator({
    width: frame.width,
    height: frame.height,
    priorK: state.K,
    priorTHeadCamera: priorT,
  });
  state.calibKey = key;
  state.calibFrameW = frame.width;
  state.calibPriorF = state.K ? state.K[0][0] : 0;
  state.lastCalibStep = 0;
  state.calibConvergedLogged = false;
  // Old tracks cannot pair across a dimension/convention change.
  state.prevKps = null;
  state.prevDesc = null;
  state.prevHeadPose = null;
  return state.calibrator;
}

/**
 * Feed one LONG-BASELINE pair (reference -> current). Consecutive frames on
 * slow feeds move only ~0.2 deg / 5 mm, which is noise-dominated for
 * self-calibration — the fit drifts along near-null directions. Pairing against
 * a reference >= CALIB_PAIR_MIN_ROT_DEG / CALIB_PAIR_MIN_TRANS_M away gives the
 * epipolar constraints real leverage. Head poses from the scale-ambiguous SfM
 * chain are fine: the constraint is scale-invariant.
 */
function feedCalibration(headPose, keypoints, descriptors) {
  const calib = state.calibrator;
  if (!calib || !headPose || !descriptors) return;
  const ref = state.calibRef;
  if (!ref || !ref.headPose || !ref.desc) {
    state.calibRef = {kps: keypoints, desc: descriptors, headPose};
    return;
  }
  const pairRot = rotationDegBetween(ref.headPose, headPose);
  const pairTrans = translationDist(ref.headPose, headPose);
  if (pairRot < CALIB_PAIR_MIN_ROT_DEG && pairTrans < CALIB_PAIR_MIN_TRANS_M) {
    return; // keep accumulating motion against the reference
  }
  const matches = matchDescriptors(ref.desc, descriptors);
  if (matches.length) {
    calib.addPair({
      T_ref_head_i: ref.headPose,
      T_ref_head_j: headPose,
      pts_i: ref.kps,
      pts_j: keypoints,
      matches,
    });
  }
  state.calibRef = {kps: keypoints, desc: descriptors, headPose};
}

/**
 * Adoption plausibility gate: a constrained-but-diverged estimate (weak or
 * degenerate motion fits noise along near-null directions) must not corrupt
 * the pipeline — only estimates near the prior's scale with a sane residual
 * level are adopted.
 */
function calibrationPlausible() {
  const calib = state.calibrator;
  if (!calib || !calib.quality.constrained) return false;
  const k = KFromCalibrator(calib);
  const q = calib.quality;
  const rmsPx = (q.rmsEpipolar * (k.fx + k.fy)) / 2;
  // Tight scale gate (15%): weakly-constrained fits can be internally
  // consistent yet tens of percent off — those must never reach the pipeline.
  return (
    state.calibPriorF > 0 &&
    Math.abs(k.fx / state.calibPriorF - 1) < 0.15 &&
    Math.abs(k.fy / state.calibPriorF - 1) < 0.15 &&
    Math.abs(k.cx - state.K[0][2]) < 0.25 * state.K[0][2] &&
    Math.abs(k.cy - state.K[1][2]) < 0.25 * state.K[1][2] &&
    rmsPx < CALIB_MAX_RMS_PX
  );
}

/** Compact stats/status line: `calib: fx … fy … (rms …px, … pairs, ✓)`. */
function calibLine(adopted) {
  const calib = state.calibrator;
  if (!calib) return 'calib: gathering…';
  const q = calib.quality;
  if (!q.constrained) return `calib: gathering… (${q.pairs} pairs)`;
  const k = KFromCalibrator(calib);
  const rmsPx = (q.rmsEpipolar * (k.fx + k.fy)) / 2;
  const mark = adopted ? '✓' : '✗ implausible';
  return (
    `calib: fx ${k.fx.toFixed(1)} fy ${k.fy.toFixed(1)} ` +
    `(rms ${rmsPx.toFixed(1)}px, ${q.pairs} pairs, ${mark})`
  );
}

/** Apply the online estimate to the pipeline when constrained + plausible. */
function adoptOnlineCalibration() {
  const calib = state.calibrator;
  if (!calib || !calibrationPlausible()) return false;
  const K = calib.K;
  state.K = [K[0].slice(), K[1].slice(), K[2].slice()];
  const T = calib.T_head_camera;
  state.RHeadCam = rot3(T);
  if (state.extrinsicsManual) {
    // Manual extrinsics LOCK the lever arm: keep the user's translation.
    state.T_head_camera = [
      [T[0][0], T[0][1], T[0][2], state.tHeadCamManual[0]],
      [T[1][0], T[1][1], T[1][2], state.tHeadCamManual[1]],
      [T[2][0], T[2][1], T[2][2], state.tHeadCamManual[2]],
      [0, 0, 0, 1],
    ];
  } else {
    state.T_head_camera = clone4(T);
  }
  return true;
}

/** Run the LM solve every ~2 s of data; adopt + report when constrained. */
function calibStep(now) {
  const calib = state.calibrator;
  if (!calib) return;
  if (
    state.lastCalibStep &&
    now - state.lastCalibStep < CALIB_STEP_INTERVAL_MS
  ) {
    return;
  }
  state.lastCalibStep = now;
  try {
    calib.step();
  } catch (err) {
    ui.log(`calib step failed: ${err.message}`, 'warn');
    return;
  }
  const adopted = adoptOnlineCalibration();
  ui.setExtrinsicsStatus(calibLine(adopted), adopted ? 'ok' : 'warn');
  if (adopted && !state.calibConvergedLogged) {
    state.calibConvergedLogged = true;
    const k = KFromCalibrator(calib);
    const q = calib.quality;
    ui.log(
      `online calibration converged: fx=${k.fx.toFixed(1)} ` +
        `cy=${k.cy.toFixed(1)} rms=${(q.rmsEpipolar * k.fx).toFixed(2)}px`
    );
  }
}

function keyframeGate(now, T) {
  if (!state.lastKf) return true;
  if (now - state.lastKf.ts < KEYFRAME_MIN_MS) return false;
  return (
    translationDist(T, state.lastKf.T) >= KEYFRAME_MIN_MOVE_M ||
    rotationDegBetween(T, state.lastKf.T) >= KEYFRAME_MIN_ROT_DEG
  );
}

function buildStep(now, headPose, kps, descriptors, frame) {
  if (!state.map) {
    state.map = createMap({
      width: frame.width,
      height: frame.height,
      K: state.K,
    });
    state.kf0Pose = null;
    ui.log(`map created (${frame.width}x${frame.height}, fov ${fovDeg}°)`);
  }
  if (!keyframeGate(now, headPose)) return;
  const fresh = state.map.keyframes.length === 0;
  if (fresh) setMapOrigin(state.map, headPose);
  let res;
  try {
    res = addKeyframe(
      state.map,
      {
        timestamp: now,
        T_ref_head: headPose,
        T_head_camera: state.T_head_camera,
        kps,
        descriptors,
      },
      helpers
    );
  } catch (err) {
    ui.log(`addKeyframe error: ${err.message}`, 'error');
    return;
  }
  if (!res || res.newKeyframe === false) return;
  state.lastKf = {ts: now, T: headPose};
  viz.setLandmarks(state.map.landmarks);
  if (fresh && !state.kf0Pose) {
    state.kf0Pose = headPose;
  } else if (state.kf0Pose) {
    // Own pose in map frame (also drives the XR world alignment).
    state.T_map_head = matMul(invertRigid(state.kf0Pose), headPose);
  }
  if (!state.hasMap) {
    state.hasMap = true;
    net.setHasMap(true);
    net.announce();
    ui.log(`map is now shareable — ${state.map.landmarks.length} landmarks`);
  }
}

function liveRelocalize() {
  if (!state.map) return null;
  try {
    return relocalize(
      state.map,
      {kps: state.prevKps, descriptors: state.prevDesc, K: state.K},
      {cv: state.cv}
    );
  } catch (err) {
    return null;
  }
}

function relocStep(headPose) {
  // The relocalization solve is heavy on large maps; cap it well below frame
  // rate so the page stays interactive while searching for the map.
  const now = performance.now();
  if (now - state.lastRelocStep < RELOC_MIN_INTERVAL_MS) return;
  state.lastRelocStep = now;
  const r = liveRelocalize();
  if (r) {
    state.lastReloc = {inliers: r.inliers, ts: performance.now()};
    state.T_map_head = headPoseFromCameraPose(
      r.T_map_camera,
      state.T_head_camera
    );
    state.relocRef = {T_map_head: state.T_map_head, T_ref_head: headPose};
    viz.setOriginVisible(true);
    ui.setStats({inliers: r.inliers});
    ui.setBadge(`relocalized ✓ ${r.inliers} inliers`);
    ui.log(`relocalized ✓ ${r.inliers} inliers (need >= ${MIN_RELOC_INLIERS})`);
    if (state.mode === 'relocalize') setMode('live');
  } else {
    ui.setBadge('no pose yet — keep scanning the mapped area');
  }
}

function liveStep(now, headPose) {
  const r = liveRelocalize();
  if (r) {
    state.T_map_head = headPoseFromCameraPose(
      r.T_map_camera,
      state.T_head_camera
    );
    state.relocRef = {T_map_head: state.T_map_head, T_ref_head: headPose};
    state.lastReloc = {inliers: r.inliers, ts: now};
    viz.setOriginVisible(true);
  } else if (state.relocRef && headPose) {
    // Propagate between successful relocalizations using the head-pose track.
    state.T_map_head = matMul(
      state.relocRef.T_map_head,
      matMul(invertRigid(state.relocRef.T_ref_head), headPose)
    );
  }
  if (state.T_map_head) {
    ui.setStats({inliers: state.lastReloc ? state.lastReloc.inliers : null});
  }
}

/** Apply a 4x4 pose to a 3D point. */
function applyPoint(T, p) {
  return [
    T[0][0] * p[0] + T[0][1] * p[1] + T[0][2] * p[2] + T[0][3],
    T[1][0] * p[0] + T[1][1] * p[1] + T[1][2] * p[2] + T[1][3],
    T[2][0] * p[0] + T[2][1] * p[1] + T[2][2] * p[2] + T[2][3],
  ];
}

/**
 * Place (or re-place) the world-locked stability cube 1 m ahead of the camera.
 * The cube is written ONCE into the map frame and never moved afterwards: if
 * tracking is stable it stays glued to the world while the camera moves.
 */
function placeCube() {
  const ahead = [0, 0, -1];
  const mapPos = state.T_map_head ? applyPoint(state.T_map_head, ahead) : ahead;
  state.cube = {placed: true, autoPlaced: true, mapPos};
  viz.setCube(mapPos);
  ui.log(
    `stability cube placed 1.0 m ahead (map frame: ${mapPos
      .map((v) => v.toFixed(2))
      .join(', ')})`
  );
}

function renderDevices(now) {
  const list = [];
  if (state.T_map_head) {
    list.push({
      peerId: net.peerId,
      label,
      T_map_head: state.T_map_head,
      inliers: state.lastReloc ? state.lastReloc.inliers : 0,
      self: true,
    });
  }
  for (const [pid, p] of state.poses) {
    if (now - p.ts > POSE_TTL_MS) {
      state.poses.delete(pid);
      continue;
    }
    list.push({
      peerId: pid,
      label: p.label,
      T_map_head: p.T_map_head,
      inliers: p.inliers,
      self: false,
    });
  }
  ui.setDevices(list);
  viz.setDevices(list);
}

function processFrame(frame, now) {
  ensureIntrinsics(frame);
  ensureCalibrator(frame);
  const {keypoints, descriptors} = extractOrb(state.cv, frame, {
    maxFeatures: MAX_FEATURES,
  });
  viz.drawKeypoints(keypoints, frame.width, frame.height);

  let matchesList = [];
  if (state.prevDesc) {
    try {
      matchesList = matchDescriptors(state.prevDesc, descriptors);
    } catch (err) {
      matchesList = [];
    }
  }
  state.matches = matchesList.length;
  // Observed frame age (capture-estimate -> processing): a persistent large
  // value on a real device means the media clock tracks arrival, not capture —
  // compensate with ?frameLagMs.
  if (frame.tMs) state.frameAgeMs = now - frame.tMs;

  const headPose = currentHeadPose(keypoints, matchesList, frame);
  state.poseHistory.push(frame.tMs || now, headPose);
  // Online self-calibration consumes LONG-BASELINE pairs (reference frame ->
  // current) built from the same ORB features the pipeline already extracts.
  feedCalibration(headPose, keypoints, descriptors);
  calibStep(now);
  state.prevHeadPose = headPose;
  state.prevKps = keypoints;
  state.prevDesc = descriptors;

  if (state.mode === 'build')
    buildStep(now, headPose, keypoints, descriptors, frame);
  else if (state.mode === 'relocalize') relocStep(headPose);
  else if (state.mode === 'live') liveStep(now, headPose);

  // Stability cube: auto-place once tracking is established, then project its
  // map-fixed position onto the feed every frame. The projected square must
  // stay glued to one spot of the scene — visible drift = tracking drift.
  if (!state.cube.autoPlaced && state.T_map_head) placeCube();
  if (state.cube.placed && state.T_map_head) {
    const T_map_cam = matMul(state.T_map_head, state.T_head_camera);
    const pCam = applyPoint(invertRigid(T_map_cam), state.cube.mapPos);
    // Head poses use the three/WebXR camera convention (forward = −Z, y up),
    // while K is a pinhole matrix (forward = +Z, y down). Convert at this
    // boundary: depth = −z, X_cv = x, Y_cv = −y.
    const depth = -pCam[2];
    if (depth > 0.15) {
      const u = (state.K[0][0] * pCam[0]) / depth + state.K[0][2];
      const v = (-state.K[1][1] * pCam[1]) / depth + state.K[1][2];
      const sizePx = (state.K[0][0] * 0.25) / depth;
      if (u > -60 && v > -60 && u < frame.width + 60 && v < frame.height + 60) {
        viz.drawCubeMarker(u, v, sizePx);
      }
    }
  }

  // Presence and the device roster run in EVERY mode: a phone that is still
  // building or searching for the map must still see (and be seen by) the rest
  // of the room. Builders broadcast as soon as they have a map-frame pose.
  if (state.T_map_head && now - state.lastBroadcast >= PRESENCE_INTERVAL_MS) {
    net.sendPose(
      state.T_map_head,
      state.lastReloc ? state.lastReloc.inliers : 0
    );
    state.lastBroadcast = now;
  }
  if (now - state.lastDeviceRender >= 200) {
    renderDevices(now);
    state.lastDeviceRender = now;
  }

  state.fpsSamples.push(now);
  while (state.fpsSamples.length && now - state.fpsSamples[0] > 1000)
    state.fpsSamples.shift();
  state.fps = state.fpsSamples.length;

  // XR world alignment: render map-frame content in the live reference space.
  if (state.xr && state.xr.active) {
    viz.setMapPose(state.T_map_head ? invertRigid(state.T_map_head) : null);
  }

  ui.setStats({
    keyframes: state.map ? state.map.keyframes.length : 0,
    landmarks: state.map ? state.map.landmarks.length : 0,
    matches: state.matches,
    fps: state.fps,
  });
}

function frameLoop(now) {
  requestAnimationFrame(frameLoop);
  // Dense (60 Hz) XR pose history so video frames can be paired with the pose
  // of their capture instant even with capture latency.
  if (state.xr && state.xr.active) {
    const T = state.xr.getHeadPose();
    if (T) state.poseHistory.push(now, T);
  }
  if (!state.cv || !state.source) return;
  if (now - state.lastProcess < PROCESS_INTERVAL_MS) return;
  state.lastProcess = now;
  const frame = state.source.getFrame();
  if (!frame) return;
  try {
    processFrame(frame, now);
  } catch (err) {
    ui.log(`frame error: ${err && err.message ? err.message : err}`, 'error');
  }
}

// ---- map persistence ---------------------------------------------------------

function adoptMap(m, name) {
  state.map = m;
  state.mapName = name || state.mapName;
  state.hasMap = true;
  state.lastKf = null;
  state.kf0Pose = null;
  state.T_map_head = null;
  state.relocRef = null;
  state.lastReloc = null;
  // The old cube pose lived in the previous map frame — re-place once tracked.
  state.cube = {placed: false, autoPlaced: false, mapPos: null};
  viz.setCube(null);
  viz.setLandmarks(m.landmarks);
  viz.setOriginVisible(false);
  viz.setDevices([]);
  ui.setStats({
    keyframes: m.keyframes?.length || 0,
    landmarks: m.landmarks?.length || 0,
    inliers: null,
  });
  net.setHasMap(true);
}

async function saveMapNow() {
  if (!state.map) {
    ui.log('nothing to save yet', 'warn');
    return;
  }
  const name =
    ui.mapName ||
    `${room}-${label}-${new Date().toISOString().slice(11, 19).replace(/:/g, '')}`;
  try {
    const bytes = serializeMap(state.map);
    await store.saveMap(name, bytes);
    state.mapName = name;
    state.hasMap = true;
    net.setHasMap(true);
    net.announce();
    await refreshMapList();
    ui.log(`saved map "${name}" (${bytes.length} B) + announced to room`);
  } catch (err) {
    ui.log(`save failed: ${err.message}`, 'error');
  }
}

async function loadMapByName(name) {
  try {
    const bytes = await store.loadMap(name);
    if (!bytes) {
      ui.log(`map "${name}" not in IndexedDB`, 'warn');
      return false;
    }
    const m = deserializeMap(bytes);
    adoptMap(m, name);
    ui.log(`loaded map "${name}" (${m.landmarks?.length || 0} landmarks)`);
    ui.setBadge(`map: ${name}`);
    return true;
  } catch (err) {
    ui.log(`load failed: ${err.message}`, 'error');
    return false;
  }
}

async function refreshMapList() {
  try {
    ui.setMapList(await store.listMaps());
  } catch (err) {
    ui.log(`IndexedDB unavailable: ${err.message}`, 'warn');
  }
}

function acquireMapForReloc() {
  state.map = null;
  state.mapName = null;
  state.T_map_head = null;
  state.relocRef = null;
  state.lastReloc = null;
  state.lastKf = null;
  state.kf0Pose = null;
  viz.setLandmarks([]);
  viz.setOriginVisible(false);
  viz.setDevices([]);
  ui.setStats({inliers: null, keyframes: 0, landmarks: 0});
  ui.setBadge('asking room for map…');
  state.lastMapReq = performance.now();
  net.requestMap();
  window.setTimeout(async () => {
    if (state.map || state.mode !== 'relocalize') return;
    const maps = await store.listMaps().catch(() => []);
    if (maps.length) {
      const ok = await loadMapByName(maps[0].name);
      if (ok) ui.setBadge(`map: ${maps[0].name} (from storage)`);
    }
    if (state.map || state.mode !== 'relocalize') return;
    ui.setBadge('waiting for a map from the room…');
    ui.log('no map in room or storage yet — retrying the room', 'warn');
    // Network joins routinely outlast the first fallback window, so keep asking
    // the room (rate-limited inside requestMapSoon) until a map arrives.
    if (!state.mapRetryTimer) {
      state.mapRetryTimer = window.setInterval(() => {
        if (state.map || state.mode !== 'relocalize') {
          window.clearInterval(state.mapRetryTimer);
          state.mapRetryTimer = null;
          return;
        }
        requestMapSoon();
      }, MAP_RETRY_MS);
    }
  }, MAP_FALLBACK_MS);
}

// ---- boot --------------------------------------------------------------------

async function init() {
  ui.log(
    `room=${room} label=${label} fov=${fovDeg}° synthetic=${wantSynthetic ? 1 : 0}`
  );
  document.getElementById('room-label').textContent = `${room} · ${label}`;

  if (['build', 'relocalize', 'live'].includes(startMode)) setMode(startMode);

  await startSource(wantSynthetic ? 'synthetic' : 'camera');

  // WebXR head tracking (optional; gesture-gated when the browser requires one).
  state.xr = await startXRIfNeeded(viz.renderer);
  if (state.xr.active) {
    ui.setXr('on', 'ok');
    ui.log('XR immersive-ar active — head poses from local-floor space');
  } else if (state.xr.needsGesture) {
    ui.setXr('tap to enable', 'warn');
    ui.showXrButton(true);
    ui.log(
      'WebXR AR supported — tap "Enable XR" for real head poses (optional)'
    );
  } else {
    ui.setXr('off', 'warn');
    ui.log(
      `head poses from ${
        state.source.kind === 'synthetic'
          ? 'synthetic trajectory'
          : 'SfM chain (scale-ambiguous)'
      } (${state.xr.error || 'no XR'})`
    );
  }

  // OpenCV (async CDN load).
  try {
    state.cv = await loadCv();
    ui.log('OpenCV ready');
  } catch (err) {
    ui.log(`loadCv failed: ${err.message} — polling global cv`, 'warn');
    state.cv = await new Promise((resolve, reject) => {
      const t0 = performance.now();
      const timer = setInterval(() => {
        const g = globalThis.cv;
        if (g && g.Mat) {
          clearInterval(timer);
          resolve(g);
        } else if (performance.now() - t0 > 30000) {
          clearInterval(timer);
          reject(new Error('cv never became available'));
        }
      }, 100);
    });
  }
  helpers.cv = state.cv;

  if (!['build', 'relocalize', 'live'].includes(startMode)) {
    ui.log(
      'pick a mode: Build (map) · Relocalize (consume a map) · Live (presence)'
    );
  }
  await refreshMapList();
  if (new URLSearchParams(location.search).has('debug')) {
    // In-page inspection/drive hook for debugging transports on real devices.
    window.__scoloc = {
      state,
      net,
      viz,
      placeCube,
      requestMapSoon,
      refreshMapList,
      // Online self-calibration handle: __scoloc.calib.K vs the prior K.
      get calib() {
        return state.calibrator;
      },
    };
  }
  requestAnimationFrame(frameLoop);
}

init().catch((err) => {
  ui.log(`init failed: ${err && err.message ? err.message : err}`, 'error');
  ui.setBadge('init failed — see log');
});
