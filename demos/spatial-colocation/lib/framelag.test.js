/**
 * Frame-lag (camera feed delay) estimation tests.
 *
 * The estimator recovers the constant pipeline latency between a frame's
 * timestamp and its true exposure time: frames are CONTENT at time c but
 * stamped `tMs = c + lag`, and the uncompensated app pairs them with poses
 * sampled at `tMs - 0`. `estimateFrameLag` re-samples the raw pose history at
 * `tMs - L` and must find L = lag.
 *
 * The generator reuses the PINNED projection convention from
 * selfcalib.test.js: `u = fx·X/(-Z) + cx`, `v = -fy·Y/(-Z) + cy` with the
 * three/WebXR -Z-forward frame, and an independent pose-math stack.
 *
 * Run standalone: node lib/framelag.test.js
 * Also runs under Vitest via the colocated framelag.test.ts shim.
 */
import assert from 'node:assert/strict';

import {OnlineCalibrator} from './selfcalib.js';

let checks = 0;
function check(name, fn) {
  fn();
  checks++;
  console.log(`ok - ${name}`);
}

// ---- deterministic PRNG + noise --------------------------------------------

function mulberry32(seed) {
  let a = seed >>> 0;
  return function rand() {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gauss(rng) {
  const u = Math.max(1e-12, rng());
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}

// ---- independent pose math (deliberately NOT imported from the lib) --------

function rot3(T) {
  return [
    [T[0][0], T[0][1], T[0][2]],
    [T[1][0], T[1][1], T[1][2]],
    [T[2][0], T[2][1], T[2][2]],
  ];
}

function matMul4(A, B) {
  const out = [
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
  ];
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += A[i][k] * B[k][j];
      out[i][j] = s;
    }
  }
  return out;
}

function invertRigid(T) {
  const R = rot3(T);
  const Rt = [
    [R[0][0], R[1][0], R[2][0]],
    [R[0][1], R[1][1], R[2][1]],
    [R[0][2], R[1][2], R[2][2]],
  ];
  const t = [T[0][3], T[1][3], T[2][3]];
  return [
    [Rt[0][0], Rt[0][1], Rt[0][2], -(Rt[0][0] * t[0] + Rt[0][1] * t[1] + Rt[0][2] * t[2])],
    [Rt[1][0], Rt[1][1], Rt[1][2], -(Rt[1][0] * t[0] + Rt[1][1] * t[1] + Rt[1][2] * t[2])],
    [Rt[2][0], Rt[2][1], Rt[2][2], -(Rt[2][0] * t[0] + Rt[2][1] * t[1] + Rt[2][2] * t[2])],
    [0, 0, 0, 1],
  ];
}

function poseFromRt(R, t) {
  return [
    [R[0][0], R[0][1], R[0][2], t[0]],
    [R[1][0], R[1][1], R[1][2], t[1]],
    [R[2][0], R[2][1], R[2][2], t[2]],
    [0, 0, 0, 1],
  ];
}

function applyPoint(T, p) {
  return [
    T[0][0] * p[0] + T[0][1] * p[1] + T[0][2] * p[2] + T[0][3],
    T[1][0] * p[0] + T[1][1] * p[1] + T[1][2] * p[2] + T[1][3],
    T[2][0] * p[0] + T[2][1] * p[1] + T[2][2] * p[2] + T[2][3],
  ];
}

function rotEuler(yaw, pitch, roll) {
  const cy = Math.cos(yaw);
  const sy = Math.sin(yaw);
  const cp = Math.cos(pitch);
  const sp = Math.sin(pitch);
  const cr = Math.cos(roll);
  const sr = Math.sin(roll);
  return [
    [cy * cr + sy * sp * sr, -cy * sr + sy * sp * cr, sy * cp],
    [cp * sr, cp * cr, -sp],
    [-sy * cr + cy * sp * sr, sy * sr + cy * sp * cr, cy * cp],
  ];
}

// ---- synthetic world --------------------------------------------------------

const W = 640;
const H = 480;
const TRUE_K = {fx: 520, fy: 515, cx: 319.5, cy: 239.5};
const TRUE_T_HEAD_CAMERA = [
  [1, 0, 0, 0],
  [0, 1, 0, 0],
  [0, 0, 1, 0],
  [0, 0, 0, 1],
];
const PRIOR_K = [
  [TRUE_K.fx, 0, TRUE_K.cx],
  [0, TRUE_K.fy, TRUE_K.cy],
  [0, 0, 1],
];

/**
 * Continuous head trajectory in seconds. `varying` selects angular
 * acceleration (lag-OBSERVABLE); the degenerate variant turns at a constant
 * rate with constant-velocity translation (lag-INVARIANT: both pair poses
 * shift together and the relative transform is unchanged).
 */
function trajAt(t, varying) {
  const yaw = varying ? 0.6 * t + 1.3 * Math.sin(1.7 * t) : 0;
  const pitch = varying ? 0.5 * Math.sin(1.1 * t + 0.7) : 0;
  const roll = varying ? 0.4 * Math.sin(0.8 * t + 2.0) : 0;
  const pos = varying
    ? [
        0.4 * Math.sin(0.9 * t + 0.3),
        0.25 * Math.sin(0.6 * t + 1.2),
        -0.35 * Math.cos(0.7 * t) + 0.35,
      ]
    : [0.2 * t, 0, -0.1 * t]; // straight line, no rotation: lag-invariant
  return poseFromRt(rotEuler(yaw, pitch, roll), pos);
}

function makeScene(rng, n) {
  const pts = [];
  for (let i = 0; i < n; i++) {
    const d = 1 + 5 * rng();
    pts.push([(rng() * 2 - 1) * 0.45 * d, (rng() * 2 - 1) * 0.3 * d, -d]);
  }
  return pts;
}

function projectPoint(T_ref_head, P, noisePx, rng) {
  const T_ref_cam = matMul4(T_ref_head, TRUE_T_HEAD_CAMERA);
  const p = applyPoint(invertRigid(T_ref_cam), P);
  const depth = -p[2];
  if (!(depth > 0.3)) return null;
  const u = (TRUE_K.fx * p[0]) / depth + TRUE_K.cx + noisePx * gauss(rng);
  const v = (-TRUE_K.fy * p[1]) / depth + TRUE_K.cy + noisePx * gauss(rng);
  if (u < 2 || u > W - 2 || v < 2 || v > H - 2) return null;
  return {x: u, y: v};
}

/**
 * Feed the calibrator pairs from a trajectory with an INJECTED timestamp lag:
 * frames are content at `c` but stamped `tMs = c + lagMs`, and the
 * uncompensated app pairs them with poses sampled at `tMs - 0` — exactly the
 * wrong pairing the estimator must discover.
 */
function feedWithLag(calib, {lagMs, varying, seed = 7, frames = 26, dtMs = 200}) {
  const rng = mulberry32(seed);
  const scene = makeScene(rng, 400);
  const contentAt = (k) => k * dtMs;
  for (let k = 0; k + 3 < frames; k++) {
    for (const gap of [1, 2, 3]) {
      if (k + gap >= frames) break;
      const ci = contentAt(k);
      const cj = contentAt(k + gap);
      const tMsI = ci + lagMs;
      const tMsJ = cj + lagMs;
      const pts_i = [];
      const pts_j = [];
      const matches = [];
      for (const P of scene) {
        const pi = projectPoint(trajAt(ci / 1000, varying), P, 0.5, rng);
        const pj = projectPoint(trajAt(cj / 1000, varying), P, 0.5, rng);
        if (!pi || !pj) continue;
        pts_i.push(pi);
        pts_j.push(pj);
        matches.push({queryIdx: pts_i.length - 1, trainIdx: pts_j.length - 1});
      }
      if (matches.length < 12) continue;
      calib.addPair({
        T_ref_head_i: trajAt(tMsI / 1000, varying), // uncompensated pairing
        T_ref_head_j: trajAt(tMsJ / 1000, varying),
        pts_i,
        pts_j,
        matches,
        tMs_i: tMsI,
        tMs_j: tMsJ,
      });
    }
  }
}

function makeCalib() {
  return new OnlineCalibrator({
    width: W,
    height: H,
    priorK: PRIOR_K,
    priorTHeadCamera: TRUE_T_HEAD_CAMERA,
    priorWeight: 0,
  });
}

// ---- tests -----------------------------------------------------------------

check('recovers an injected 300ms feed delay', () => {
  const calib = makeCalib();
  feedWithLag(calib, {lagMs: 300, varying: true});
  const est = calib.estimateFrameLag((t) => (t < 0 ? null : trajAt(t / 1000, true)), {
    maxLagMs: 1000,
  });
  assert.ok(est, 'estimate should exist on varying-omega motion');
  console.log(
    `   estimated ${est.lagMs} ms (truth 300) · spread ${est.spreadRatio.toFixed(2)}x · ${est.pairsUsed} pairs`
  );
  assert.ok(Math.abs(est.lagMs - 300) <= 30, `lag ${est.lagMs} not within 30ms of 300`);
  assert.ok(est.pairsUsed >= 8);
});

check('recovers 0ms and 750ms delays within the sweep resolution', () => {
  for (const lag of [0, 750]) {
    const calib = makeCalib();
    feedWithLag(calib, {lagMs: lag, varying: true});
    const est = calib.estimateFrameLag(
      (t) => (t < 0 ? null : trajAt(t / 1000, true)),
      {maxLagMs: 1000}
    );
    assert.ok(est, `estimate should exist for lag ${lag}`);
    console.log(`   estimated ${est.lagMs} ms (truth ${lag})`);
    assert.ok(Math.abs(est.lagMs - lag) <= 30, `lag ${est.lagMs} not within 30ms of ${lag}`);
  }
});

check('returns null on rotation-free linear motion (unobservable)', () => {
  const calib = makeCalib();
  feedWithLag(calib, {lagMs: 300, varying: false});
  const est = calib.estimateFrameLag(
    (t) => (t < 0 ? null : trajAt(t / 1000, false)),
    {maxLagMs: 1000}
  );
  assert.equal(est, null);
});

check('returns null with too few timestamped pairs', () => {
  const calib = makeCalib();
  const est = calib.estimateFrameLag((t) => (t < 0 ? null : trajAt(t / 1000, true)));
  assert.equal(est, null);
});

console.log(`${checks} checks passed`);
