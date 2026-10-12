/**
 * Compute-pipeline Web Worker for the XR Blocks simulator testbed.
 *
 * Everything expensive runs HERE, off the render thread:
 *   ORB extraction (~70ms/frame on slow CPUs), descriptor matching, the
 *   online self-calibration LM solves, map growth (keyframe matching +
 *   triangulation) and the relocalization PnP search.
 *
 * The worker OWNS the compute state (calibrator, map, pairing tracks); the
 * app keeps presentation state and receives compact results per frame
 * (counts, landmark buffers for the viz, map->ref transforms, calib
 * snapshots, log/badge events) plus RPC replies for map serialization.
 *
 * THIS IS A CLASSIC WORKER ON PURPOSE: docs.opencv.org sends no CORS
 * headers, so fetch()/ESM-import of opencv.js always fail from a worker —
 * but importScripts() is exempt from CORS (same as a page <script> tag).
 * Project ESM modules load via dynamic import(), which classic workers
 * support. The UMD bundle's thenable readiness shim is disarmed before the
 * namespace travels anywhere (see lib/cv-runtime.js).
 */

const OPENCV_URL = 'https://docs.opencv.org/4.x/opencv.js';

// Pipeline constants (mirrored from app/sim-main.js — kept local so the
// worker carries no app imports).
const CALIB_STEP_INTERVAL_MS = 300;
const CALIB_PAIR_MIN_ROT_DEG = 2.5;
const CALIB_PAIR_MIN_TRANS_M = 0.04;
const RELOC_MIN_INTERVAL_MS = 250;

// Convention flip (same as app/main.js's CV_CAM_FLIP): the map pipeline's
// triangulator (lib/geometry.js) expects OPENCV-convention camera poses
// (x right, y down, +z forward), while XR/WebXR head poses are three-
// convention (y up, -z forward). The head->map composition must carry this
// flip or the reconstruction mirrors through the camera plane (points land
// around the camera path instead of on the walls). The CALIBRATOR stays on
// three-convention extrinsics (identity-ish) — only the map path flips.
const CV_CAM_FLIP = [
  [1, 0, 0, 0],
  [0, -1, 0, 0],
  [0, 0, -1, 0],
  [0, 0, 0, 1],
];

let orbLib = null;
let matchingLib = null;
let geometryLib = null;
let mapLib = null;
let relocalizeLib = null;
let selfcalibLib = null;
let cvPromise = null;

// ---- compute state ---------------------------------------------------------
const S = {
  calibrator: null,
  calibFrameW: 0,
  calibRef: null,
  lastCalibStep: 0,
  prevKps: null,
  prevDesc: null,
  prevHeadPose: null,
  frameMatchTick: 0,
  map: null,
  mapName: null,
  lastKf: null,
  kf0Pose: null,
  hasMapAnnounced: false,
  relocRef: null,
  lastReloc: null,
  lastRelocStep: 0,
  landmarkSentCount: -1,
};

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Remove the thenable readiness shim (see lib/cv-runtime.js docblock). */
function disarm(cv) {
  if (typeof cv.then === 'function') {
    try {
      delete cv.then;
    } catch {
      /* ignore */
    }
    if (typeof cv.then === 'function') cv.then = undefined;
    if (typeof cv.then === 'function') {
      throw new Error('cv exposes a non-removable .then shim');
    }
  }
  return cv;
}

async function loadCvInWorker() {
  if (!(globalThis.cv && typeof globalThis.cv.Mat === 'function')) {
    importScripts(OPENCV_URL); // synchronous, CORS-exempt like a <script> tag
    const deadline = Date.now() + 30000;
    while (typeof (globalThis.cv && globalThis.cv.Mat) !== 'function') {
      if (typeof globalThis.cv === 'function' && globalThis.cv.then) {
        try {
          globalThis.cv.then(() => {});
        } catch {
          /* readiness shim; polling below still works */
        }
      }
      if (Date.now() > deadline) {
        throw new Error('opencv.js: cv.Mat never appeared (30s)');
      }
      await sleep(15);
    }
  }
  return disarm(globalThis.cv);
}

async function loadLibs() {
  if (!orbLib) {
    const [orb, matching, geometry, map, relocalize, selfcalib] =
      await Promise.all([
        import('../lib/orb.js'),
        import('../lib/matching.js'),
        import('../lib/geometry.js'),
        import('../lib/map.js'),
        import('../lib/relocalize.js'),
        import('../lib/selfcalib.js'),
      ]);
    orbLib = orb;
    matchingLib = matching;
    geometryLib = geometry;
    mapLib = map;
    relocalizeLib = relocalize;
    selfcalibLib = selfcalib;
  }
}

function ensureCv() {
  if (!cvPromise) cvPromise = loadCvInWorker();
  return cvPromise;
}

function matMulLocal(A, B) {
  return geometryLib.matMul(A, B);
}

function helpers() {
  return {
    cv: globalThis.cv,
    estimateEssential: geometryLib.estimateEssential,
    triangulate: geometryLib.triangulate,
    matchDescriptors: matchingLib.matchDescriptors,
    MIN_PARALLAX_DEG: orbLib.MIN_PARALLAX_DEG,
  };
}

// ---- pipeline (ported from app/sim-main.js) --------------------------------

// Pose helpers (copied from app/sim-main.js's local implementations —
// geometry.js exports degBetween for vectors, not 4x4 poses).
function rotationDegBetween(Ta, Tb) {
  let tr = 0;
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) tr += Ta[i][j] * Tb[i][j];
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

function keyframeGate(now, T) {
  if (!S.lastKf) return true;
  if (now - S.lastKf.ts < orbLib.KEYFRAME_MIN_MS) return false;
  return (
    translationDist(T, S.lastKf.T) >= orbLib.KEYFRAME_MIN_MOVE_M ||
    rotationDegBetween(T, S.lastKf.T) >= orbLib.KEYFRAME_MIN_ROT_DEG
  );
}

function pair(T_map_head, headPose) {
  return {
    T_map_head,
    T_map_ref: geometryLib.matMul(
      T_map_head,
      geometryLib.invertRigid(headPose)
    ),
  };
}

function ensureCalibrator(meta) {
  if (S.calibrator && S.calibFrameW === meta.width) return;
  S.calibrator = new selfcalibLib.OnlineCalibrator({
    width: meta.width,
    height: meta.height,
    priorK: meta.K,
    priorTHeadCamera: meta.T_head_camera,
  });
  S.calibFrameW = meta.width;
  S.lastCalibStep = 0;
  S.calibRef = null;
  S.prevKps = null;
  S.prevDesc = null;
  S.prevHeadPose = null;
}

function feedCalibration(headPose, keypoints, descriptors) {
  const calib = S.calibrator;
  if (!calib || !headPose || !descriptors) return;
  const ref = S.calibRef;
  if (!ref || !ref.headPose || !ref.desc) {
    S.calibRef = {kps: keypoints, desc: descriptors, headPose};
    return;
  }
  const pairRot = rotationDegBetween(ref.headPose, headPose);
  const pairTrans = translationDist(ref.headPose, headPose);
  if (pairRot < CALIB_PAIR_MIN_ROT_DEG && pairTrans < CALIB_PAIR_MIN_TRANS_M) {
    return; // keep accumulating motion against the reference
  }
  const matches = matchingLib.matchDescriptors(ref.desc, descriptors);
  if (matches.length) {
    calib.addPair({
      T_ref_head_i: ref.headPose,
      T_ref_head_j: headPose,
      pts_i: ref.kps,
      pts_j: keypoints,
      matches,
    });
  }
  S.calibRef = {kps: keypoints, desc: descriptors, headPose};
}

function calibSnapshot() {
  const calib = S.calibrator;
  if (!calib) return null;
  return {
    K: calib.K.map((row) => row.slice()),
    T_head_camera: calib.T_head_camera.map((row) => row.slice()),
    quality: {...calib.quality},
    priorK: calib.priorK.map((row) => row.slice()),
  };
}

function buildStep(now, headPose, kps, descriptors, meta, events) {
  const result = {
    mapCreated: false,
    newKeyframe: false,
    hasMapNow: false,
    T_map_head: null,
    T_map_ref: null,
  };
  if (!S.map) {
    S.map = mapLib.createMap({
      width: meta.width,
      height: meta.height,
      K: meta.K,
    });
    S.kf0Pose = null;
    S.hasMapAnnounced = false;
    S.landmarkSentCount = -1;
    result.mapCreated = true;
    events.push({
      kind: 'log',
      msg: `map created (${meta.width}x${meta.height})`,
    });
  }
  if (!keyframeGate(now, headPose)) return result;
  const fresh = S.map.keyframes.length === 0;
  if (fresh) mapLib.setMapOrigin(S.map, headPose);
  let res;
  try {
    res = mapLib.addKeyframe(
      S.map,
      {
        timestamp: now,
        T_ref_head: headPose,
        T_head_camera: matMulLocal(meta.T_head_camera, CV_CAM_FLIP),
        kps,
        descriptors,
      },
      helpers()
    );
  } catch (err) {
    events.push({
      kind: 'log',
      msg: `addKeyframe error: ${err.message}`,
      level: 'error',
    });
    return result;
  }
  if (!res || res.newKeyframe === false) return result;
  result.newKeyframe = true;
  S.lastKf = {ts: now, T: headPose};
  if (fresh && !S.kf0Pose) {
    S.kf0Pose = headPose;
  } else if (S.kf0Pose) {
    // Own pose in map frame (map frame = head frame at the first keyframe).
    const p = pair(
      geometryLib.matMul(geometryLib.invertRigid(S.kf0Pose), headPose),
      headPose
    );
    result.T_map_head = p.T_map_head;
    result.T_map_ref = p.T_map_ref;
  }
  if (!S.hasMapAnnounced) {
    S.hasMapAnnounced = true;
    result.hasMapNow = true;
    events.push({
      kind: 'log',
      msg: `map is now shareable — ${S.map.landmarks.length} landmarks`,
    });
  }
  return result;
}

function liveRelocalize(meta) {
  if (!S.map) return null;
  try {
    return relocalizeLib.relocalize(
      S.map,
      {kps: S.prevKps, descriptors: S.prevDesc, K: meta.K},
      {cv: globalThis.cv}
    );
  } catch (err) {
    return null;
  }
}

function relocStep(now, headPose, meta, events) {
  const result = {
    reloc: null,
    T_map_head: null,
    T_map_ref: null,
    wantMode: null,
  };
  if (now - S.lastRelocStep < RELOC_MIN_INTERVAL_MS) return result;
  S.lastRelocStep = now;
  const r = liveRelocalize(meta);
  if (r) {
    S.lastReloc = {inliers: r.inliers, ts: now};
    const T_map_head = relocalizeLib.headPoseFromCameraPose(
      r.T_map_camera,
      matMulLocal(meta.T_head_camera, CV_CAM_FLIP)
    );
    const p = pair(T_map_head, headPose);
    result.T_map_head = p.T_map_head;
    result.T_map_ref = p.T_map_ref;
    result.reloc = {inliers: r.inliers};
    S.relocRef = {T_map_head, T_ref_head: headPose};
    events.push({kind: 'stats', stats: {inliers: r.inliers}});
    events.push({kind: 'badge', text: `relocalized ✓ ${r.inliers} inliers`});
    events.push({
      kind: 'log',
      msg: `relocalized ✓ ${r.inliers} inliers (need >= ${relocalizeLib.MIN_RELOC_INLIERS})`,
    });
    result.wantMode = 'live';
  } else {
    events.push({
      kind: 'badge',
      text: 'no pose yet — keep scanning the mapped area',
    });
  }
  return result;
}

function liveStep(now, headPose, meta) {
  const result = {reloc: null, T_map_head: null, T_map_ref: null};
  const r = liveRelocalize(meta);
  if (r) {
    const T_map_head = relocalizeLib.headPoseFromCameraPose(
      r.T_map_camera,
      matMulLocal(meta.T_head_camera, CV_CAM_FLIP)
    );
    const p = pair(T_map_head, headPose);
    S.relocRef = {T_map_head, T_ref_head: headPose};
    S.lastReloc = {inliers: r.inliers, ts: now};
    result.T_map_head = p.T_map_head;
    result.T_map_ref = p.T_map_ref;
    result.reloc = {inliers: r.inliers};
  } else if (S.relocRef && headPose) {
    // Propagate between successful relocalizations via the head-pose track.
    const p = pair(
      geometryLib.matMul(
        S.relocRef.T_map_head,
        geometryLib.matMul(
          geometryLib.invertRigid(S.relocRef.T_ref_head),
          headPose
        )
      ),
      headPose
    );
    result.T_map_head = p.T_map_head;
    result.T_map_ref = p.T_map_ref;
  }
  return result;
}

function landmarkSnapshot() {
  const landmarks = S.map ? S.map.landmarks : [];
  const n = landmarks.length;
  const pos = new Float32Array(n * 3);
  const desc = new Uint8Array(n * 32);
  for (let i = 0; i < n; i++) {
    pos[3 * i] = landmarks[i].position[0];
    pos[3 * i + 1] = landmarks[i].position[1];
    pos[3 * i + 2] = landmarks[i].position[2];
    const d = landmarks[i].descriptor;
    for (let k = 0; k < 32; k++) desc[32 * i + k] = d[k] | 0;
  }
  return {count: n, pos, desc};
}

async function onFrame(data) {
  await loadLibs();
  const cv = await ensureCv();
  const {buffer, width, height, meta} = data;
  const image = {data: new Uint8ClampedArray(buffer), width, height};
  const {keypoints, descriptors} = orbLib.extractOrb(cv, image, {
    maxFeatures: orbLib.MAX_FEATURES,
  });

  const events = [];
  const now = meta.now;
  let matchesCount = 0;
  ensureCalibrator(meta);

  // Frame-to-frame match feeds ONLY the UI counter — every 3rd frame.
  S.frameMatchTick += 1;
  if (S.prevDesc && S.frameMatchTick % 3 === 0) {
    try {
      matchesCount = matchingLib.matchDescriptors(S.prevDesc, descriptors).length;
    } catch (err) {
      matchesCount = 0;
    }
  }

  feedCalibration(meta.headPose, keypoints, descriptors);

  // LM solve at its own cadence; the app adopts from the snapshot.
  let stepped = false;
  if (!S.lastCalibStep || now - S.lastCalibStep >= CALIB_STEP_INTERVAL_MS) {
    S.lastCalibStep = now;
    try {
      S.calibrator.step(5);
      stepped = true;
    } catch (err) {
      events.push({
        kind: 'log',
        msg: `calib step failed: ${err.message}`,
        level: 'warn',
      });
    }
  }

  S.prevHeadPose = meta.headPose;
  S.prevKps = keypoints;
  S.prevDesc = descriptors;

  let modeResult = {T_map_head: null, T_map_ref: null};
  if (meta.mode === 'build') {
    modeResult = buildStep(
      now,
      meta.headPose,
      keypoints,
      descriptors,
      meta,
      events
    );
  } else if (meta.mode === 'relocalize') {
    modeResult = relocStep(now, meta.headPose, meta, events);
    if (modeResult.wantMode) events.push({kind: 'mode', mode: modeResult.wantMode});
  } else if (meta.mode === 'live') {
    modeResult = liveStep(now, meta.headPose, meta);
  }

  // Keep the app's tracked poses current even between keyframes.
  if (!modeResult.T_map_head && S.kf0Pose && meta.mode === 'build') {
    const p = pair(
      geometryLib.matMul(geometryLib.invertRigid(S.kf0Pose), meta.headPose),
      meta.headPose
    );
    modeResult.T_map_head = p.T_map_head;
    modeResult.T_map_ref = p.T_map_ref;
  }

  const count = S.map ? S.map.landmarks.length : 0;
  const out = {
    type: 'result',
    meta: {now, tMs: meta.tMs, headPose: meta.headPose},
    matches: matchesCount,
    calib: calibSnapshot(),
    stepped,
    events,
    T_map_head: modeResult.T_map_head,
    T_map_ref: modeResult.T_map_ref,
    reloc: modeResult.reloc || null,
    mapInfo: S.map
      ? {
          width: S.map.width,
          height: S.map.height,
          landmarks: count,
          keyframes: S.map.keyframes.length,
        }
      : null,
    mapCreated: !!modeResult.mapCreated,
    hasMapNow: !!modeResult.hasMapNow,
    newKeyframe: !!modeResult.newKeyframe,
    landmarks: null,
  };
  if (count !== S.landmarkSentCount) {
    const snap = landmarkSnapshot();
    S.landmarkSentCount = snap.count;
    out.landmarks = snap;
    self.postMessage(out, [snap.pos.buffer, snap.desc.buffer]);
    return;
  }
  self.postMessage(out);
}

// ---- message routing -------------------------------------------------------

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === 'frame') {
      await onFrame(msg);
      return;
    }
    await loadLibs();
    if (msg.type === 'init') {
      S.calibrator = null;
      ensureCalibrator(msg);
      self.postMessage({type: 'rpcResult', id: msg.id, ok: true});
    } else if (msg.type === 'serializeMap') {
      if (!S.map) {
        self.postMessage({type: 'rpcResult', id: msg.id, ok: false});
      } else {
        const bytes = mapLib.serializeMap(S.map);
        self.postMessage({type: 'rpcResult', id: msg.id, ok: true, bytes}, [
          bytes.buffer,
        ]);
      }
    } else if (msg.type === 'loadMap') {
      const m = mapLib.deserializeMap(msg.bytes);
      S.map = m;
      S.mapName = msg.name || null;
      S.lastKf = null;
      S.kf0Pose = null;
      S.hasMapAnnounced = true;
      S.relocRef = null;
      S.lastReloc = null;
      S.landmarkSentCount = -1;
      const snap = landmarkSnapshot();
      self.postMessage(
        {
          type: 'rpcResult',
          id: msg.id,
          ok: true,
          mapInfo: {
            width: m.width,
            height: m.height,
            landmarks: m.landmarks.length,
            keyframes: m.keyframes.length,
          },
          landmarks: snap,
        },
        [snap.pos.buffer, snap.desc.buffer]
      );
    } else if (msg.type === 'reset') {
      S.calibrator = null;
      S.map = null;
      S.calibRef = null;
      S.prevKps = null;
      S.prevDesc = null;
      S.landmarkSentCount = -1;
      self.postMessage({type: 'rpcResult', id: msg.id, ok: true});
    }
  } catch (err) {
    const error = err && err.message ? err.message : String(err);
    if (msg && msg.id !== undefined) {
      self.postMessage({type: 'rpcResult', id: msg.id, ok: false, error});
    } else {
      self.postMessage({type: 'workerError', error});
    }
  }
};
