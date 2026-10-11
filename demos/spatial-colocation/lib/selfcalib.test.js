/**
 * Synthetic recovery tests for `lib/selfcalib.js` (SELF_CALIB_SPEC.md).
 *
 * Plain Node-runnable (node:assert) so it executes with
 * `node demos/spatial-colocation/lib/selfcalib.test.js`; also runs under
 * Vitest via the colocated `selfcalib.test.ts` shim.
 *
 * The generator PINS THE CONVENTION: projections use the exact boundary
 * conversion documented in lib/selfcalib.js — three/WebXR poses (forward -Z,
 * y up), image pixels (y down), `u = fx·X/(-Z) + cx`, `v = -fy·Y/(-Z) + cy`.
 * A known-pose projection test asserts ~zero residual at the true parameters
 * and a NON-zero residual when the v (y-down) conversion is flipped, so a
 * convention regression cannot pass silently.
 */
import assert from 'node:assert/strict';

import {OnlineCalibrator, KFromCalibrator} from './selfcalib.js';

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
    [
      Rt[0][0],
      Rt[0][1],
      Rt[0][2],
      -(Rt[0][0] * t[0] + Rt[0][1] * t[1] + Rt[0][2] * t[2]),
    ],
    [
      Rt[1][0],
      Rt[1][1],
      Rt[1][2],
      -(Rt[1][0] * t[0] + Rt[1][1] * t[1] + Rt[1][2] * t[2]),
    ],
    [
      Rt[2][0],
      Rt[2][1],
      Rt[2][2],
      -(Rt[2][0] * t[0] + Rt[2][1] * t[1] + Rt[2][2] * t[2]),
    ],
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

function rotMatFromRotVec(v) {
  const [x, y, z] = v;
  const theta = Math.hypot(x, y, z);
  if (theta < 1e-12) {
    return [
      [1, -z, y],
      [z, 1, -x],
      [-y, x, 1],
    ];
  }
  const kx = x / theta;
  const ky = y / theta;
  const kz = z / theta;
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  const t1 = 1 - c;
  return [
    [c + kx * kx * t1, kx * ky * t1 - kz * s, kx * kz * t1 + ky * s],
    [ky * kx * t1 + kz * s, c + ky * ky * t1, ky * kz * t1 - kx * s],
    [kz * kx * t1 - ky * s, kz * ky * t1 + kx * s, c + kz * kz * t1],
  ];
}

function rotEuler(yaw, pitch, roll) {
  const cy = Math.cos(yaw);
  const sy = Math.sin(yaw);
  const cp = Math.cos(pitch);
  const sp = Math.sin(pitch);
  const cr = Math.cos(roll);
  const sr = Math.sin(roll);
  // R = Ry(yaw) · Rx(pitch) · Rz(roll)
  return [
    [cy * cr + sy * sp * sr, -cy * sr + sy * sp * cr, sy * cp],
    [cp * sr, cp * cr, -sp],
    [-sy * cr + cy * sp * sr, sy * sr + cy * sp * cr, cy * cp],
  ];
}

function rotationDegBetween(Ra, Rb) {
  // angle of Raᵀ · Rb
  let tr = 0;
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) tr += Ra[i][j] * Rb[i][j];
  }
  const c = Math.min(1, Math.max(-1, (tr - 1) / 2));
  return (Math.acos(c) * 180) / Math.PI;
}

// ---- synthetic world --------------------------------------------------------

const W = 640;
const H = 480;
const TRUE_K = {fx: 520, fy: 515, cx: 319.5, cy: 239.5};
// Extrinsic rotation ~5 deg about a slanted axis; lever arm ~2 cm.
const TRUE_RVEC = (() => {
  const n = Math.hypot(1, 2, 3);
  const ang = (5 * Math.PI) / 180;
  return [(1 / n) * ang, (2 / n) * ang, (3 / n) * ang];
})();
const TRUE_LEVER = [0.015, -0.01, 0.02];
const TRUE_T_HEAD_CAMERA = poseFromRt(rotMatFromRotVec(TRUE_RVEC), TRUE_LEVER);
// Prior guess: the raw demo's estimateIntrinsics(width, height, 60 deg).
const PRIOR_K = (() => {
  const f = W / (2 * Math.tan((60 * Math.PI) / 360));
  return [
    [f, 0, W / 2],
    [0, f, H / 2],
    [0, 0, 1],
  ];
})();
const IDENTITY = [
  [1, 0, 0, 0],
  [0, 1, 0, 0],
  [0, 0, 1, 0],
  [0, 0, 0, 1],
];

/** Random 3D points spread 1-6 m in front of the trajectory (-Z forward). */
function makeScene(rng, n) {
  const pts = [];
  for (let i = 0; i < n; i++) {
    const d = 1 + 5 * rng();
    pts.push([(rng() * 2 - 1) * 0.45 * d, (rng() * 2 - 1) * 0.3 * d, -d]);
  }
  return pts;
}

/**
 * Sinusoidal head trajectory with translation AND rotation diversity
 * (yaw/pitch/roll at incommensurate frequencies) plus small pose noise.
 * Default cadence 0.5 s/frame (26 frames ≈ 13 s of motion). `translate:
 * false` gives the degenerate pure-rotation case.
 */
function headPoseAt(k, rng, opts) {
  const {translate, dt, yawA, pitchA, rollA} = {
    translate: true,
    dt: 0.5,
    yawA: 0.5,
    pitchA: 0.35,
    rollA: 0.25,
    ...opts,
  };
  const t = k * dt;
  const pos = translate
    ? [
        0.4 * Math.sin(0.9 * t + 0.3) + 0.0005 * gauss(rng),
        0.25 * Math.sin(0.6 * t + 1.2) + 0.0005 * gauss(rng),
        -0.35 * Math.cos(0.7 * t) + 0.35 + 0.0005 * gauss(rng),
      ]
    : [0, 0, 0];
  const yaw = yawA * Math.sin(0.8 * t);
  const pitch = pitchA * Math.sin(0.5 * t + 0.7);
  const roll = rollA * Math.sin(0.4 * t + 2.0);
  const R = rotEuler(yaw, pitch, roll);
  // Small rotation noise (~0.03 deg) — tracking noise, not geometry.
  const noise = rotMatFromRotVec([
    0.0005 * gauss(rng),
    0.0005 * gauss(rng),
    0.0005 * gauss(rng),
  ]);
  const Rn = matMul4(poseFromRt(R, pos), poseFromRt(noise, [0, 0, 0]));
  return poseFromRt(rot3(Rn), pos);
}

/**
 * Project a world point with the PINNED convention:
 * `u = fx·X/(-Z) + cx`, `v = -fy·Y/(-Z) + cy` (+ pixel noise). `flipY` uses
 * the wrong v sign — used to prove the tests pin the y-down conversion.
 */
function projectPoint(T_ref_head, P, noisePx, rng, flipY) {
  const T_ref_cam = matMul4(T_ref_head, TRUE_T_HEAD_CAMERA);
  const p = applyPoint(invertRigid(T_ref_cam), P);
  const depth = -p[2];
  if (!(depth > 0.3)) return null;
  const u = (TRUE_K.fx * p[0]) / depth + TRUE_K.cx + noisePx * gauss(rng);
  const vSign = flipY ? 1 : -1;
  const v =
    (vSign * TRUE_K.fy * p[1]) / depth + TRUE_K.cy + noisePx * gauss(rng);
  if (u < 2 || u > W - 2 || v < 2 || v > H - 2) return null;
  return {x: u, y: v};
}

/**
 * Generate matched pairs from consecutive frames (gaps 1-3 by default), like
 * the apps feed `addPair` from per-frame matches. Extra options are forwarded
 * to `headPoseAt` (dt / yawA / pitchA / rollA / translate).
 *
 * The standard translating trajectory (26 frames at 0.5 s cadence, ~13 s of
 * motion) is deliberately a STRONG self-calibration geometry: measured worst
 * recovery error over 5 seeds (incl. a 20%-wrong prior) is cx 0.37 px,
 * cy 0.46 px, fx/fy 0.16%, rotation 0.065 deg, lever arm 4.4 mm — comfortably
 * inside the spec tolerances instead of sitting on their edge.
 */
function generatePairs({
  seed,
  frames,
  noisePx = 0.5,
  flipY = false,
  gaps = [1, 2, 3],
  ...traj
}) {
  const rng = mulberry32(seed);
  const scene = makeScene(rng, 400);
  const poses = [];
  for (let k = 0; k < frames; k++) poses.push(headPoseAt(k, rng, traj));
  const pairs = [];
  for (let k = 0; k + 1 < frames; k++) {
    for (const gap of gaps) {
      if (k + gap >= frames) continue;
      const pts_i = [];
      const pts_j = [];
      const matches = [];
      for (const P of scene) {
        const pi = projectPoint(poses[k], P, noisePx, rng, flipY);
        const pj = projectPoint(poses[k + gap], P, noisePx, rng, flipY);
        if (!pi || !pj) continue;
        pts_i.push(pi);
        pts_j.push(pj);
        matches.push({queryIdx: pts_i.length - 1, trainIdx: pts_j.length - 1});
      }
      if (matches.length >= 6) {
        pairs.push({
          T_ref_head_i: poses[k],
          T_ref_head_j: poses[k + gap],
          pts_i,
          pts_j,
          matches,
        });
      }
    }
  }
  return pairs;
}

function feed(calib, pairs) {
  for (const p of pairs) calib.addPair(p);
}

/** Run step() until the cost plateaus (or `maxSteps` calls). */
function converge(calib, maxSteps = 12) {
  let totalIters = 0;
  let steps = 0;
  let last = null;
  for (let i = 0; i < maxSteps; i++) {
    last = calib.step();
    steps++;
    totalIters += last.iterations;
    if (!last.improved) break;
  }
  return {last, steps, totalIters};
}

function kErrors(calib) {
  const K = calib.K;
  return {
    fx: K[0][0],
    fy: K[1][1],
    cx: K[0][2],
    cy: K[1][2],
    fxPct: (Math.abs(K[0][0] - TRUE_K.fx) / TRUE_K.fx) * 100,
    fyPct: (Math.abs(K[1][1] - TRUE_K.fy) / TRUE_K.fy) * 100,
    cxPx: Math.abs(K[0][2] - TRUE_K.cx),
    cyPx: Math.abs(K[1][2] - TRUE_K.cy),
  };
}

function extrErrors(calib) {
  const T = calib.T_head_camera;
  const rotDeg = rotationDegBetween(rot3(TRUE_T_HEAD_CAMERA), rot3(T));
  const leverM = Math.hypot(
    T[0][3] - TRUE_LEVER[0],
    T[1][3] - TRUE_LEVER[1],
    T[2][3] - TRUE_LEVER[2]
  );
  return {rotDeg, leverM};
}

function fmtK(e) {
  return (
    `fx ${e.fx.toFixed(1)} (${e.fxPct.toFixed(2)}%) ` +
    `fy ${e.fy.toFixed(1)} (${e.fyPct.toFixed(2)}%) ` +
    `cx ${e.cx.toFixed(1)} (${e.cxPx.toFixed(2)}px) ` +
    `cy ${e.cy.toFixed(1)} (${e.cyPx.toFixed(2)}px)`
  );
}

// ---- tests -----------------------------------------------------------------

check('convention pin: known-pose projection has ~zero residual', () => {
  // Noise-free data projected with the documented boundary conversion; the
  // calibrator starts AT the truth, so the Sampson residual must be ~0 and
  // the parameters must not move.
  const pairs = generatePairs({
    seed: 7,
    frames: 12,
    noisePx: 0,
    translate: true,
  });
  const calib = new OnlineCalibrator({
    width: W,
    height: H,
    priorK: [
      [TRUE_K.fx, 0, TRUE_K.cx],
      [0, TRUE_K.fy, TRUE_K.cy],
      [0, 0, 1],
    ],
    priorTHeadCamera: TRUE_T_HEAD_CAMERA,
  });
  feed(calib, pairs);
  const res = converge(calib);
  assert.ok(res.last.rmsEpipolar !== null, 'rms computed');
  assert.ok(
    res.last.rmsEpipolar < 1e-9,
    `rms at truth must be ~0, got ${res.last.rmsEpipolar}`
  );
  const e = kErrors(calib);
  assert.ok(e.fxPct < 1e-6, 'fx unmoved');
  assert.ok(e.cxPx < 1e-6, `cx unmoved, got ${e.cxPx}`);
  const x = extrErrors(calib);
  assert.ok(x.rotDeg < 1e-4, `rotation unmoved, got ${x.rotDeg}`);
  console.log(
    `   measured rms at truth: ${res.last.rmsEpipolar.toExponential(2)}`
  );
});

check('convention pin: flipped v (y-up mistake) does NOT fit', () => {
  // Same geometry, but v projected with the WRONG sign: the calibrator must
  // see large residuals — this pins the y-down pixel convention.
  const pairs = generatePairs({
    seed: 7,
    frames: 12,
    noisePx: 0,
    translate: true,
    flipY: true,
  });
  const calib = new OnlineCalibrator({
    width: W,
    height: H,
    priorK: [
      [TRUE_K.fx, 0, TRUE_K.cx],
      [0, TRUE_K.fy, TRUE_K.cy],
      [0, 0, 1],
    ],
    priorTHeadCamera: TRUE_T_HEAD_CAMERA,
  });
  feed(calib, pairs);
  const res = converge(calib);
  assert.ok(
    res.last.rmsEpipolar > 1e-4,
    `flipped-v data must not fit at truth, got rms ${res.last.rmsEpipolar}`
  );
  console.log(
    `   measured rms with flipped v: ${res.last.rmsEpipolar.toFixed(4)}`
  );
});

check('K recovery: fx/fy within 2%, cx/cy within 2px', () => {
  const pairs = generatePairs({seed: 11, frames: 26, noisePx: 0.5});
  const calib = new OnlineCalibrator({
    width: W,
    height: H,
    priorK: PRIOR_K,
    priorTHeadCamera: IDENTITY,
  });
  feed(calib, pairs);
  const {last, totalIters} = converge(calib);
  const e = kErrors(calib);
  console.log(
    `   measured K errors: ${fmtK(e)} · rms ${(last.rmsEpipolar * TRUE_K.fx).toFixed(2)}px · ${totalIters} LM iterations`
  );
  assert.ok(e.fxPct < 2, `fx ${e.fxPct.toFixed(2)}% must be < 2%`);
  assert.ok(e.fyPct < 2, `fy ${e.fyPct.toFixed(2)}% must be < 2%`);
  assert.ok(e.cxPx < 2, `cx ${e.cxPx.toFixed(2)}px must be < 2px`);
  assert.ok(e.cyPx < 2, `cy ${e.cyPx.toFixed(2)}px must be < 2px`);
});

check('extrinsics recovery: rotation < 2deg, lever arm < 2cm', () => {
  // Same synthetic run as the K recovery (rotation ~5deg truth, lever ~2cm).
  const pairs = generatePairs({seed: 11, frames: 26, noisePx: 0.5});
  const calib = new OnlineCalibrator({
    width: W,
    height: H,
    priorK: PRIOR_K,
    priorTHeadCamera: IDENTITY,
  });
  feed(calib, pairs);
  const {last} = converge(calib);
  const x = extrErrors(calib);
  const e = kErrors(calib);
  console.log(
    `   measured extrinsics errors: rotation ${x.rotDeg.toFixed(3)}deg · lever ${x.leverM.toFixed(4)}m · ` +
      `K ${fmtK(e)} · rms ${(last.rmsEpipolar * TRUE_K.fx).toFixed(2)}px`
  );
  assert.ok(x.rotDeg < 2, `rotation ${x.rotDeg.toFixed(3)}deg must be < 2deg`);
  // 2 cm bound per spec (measured value is printed above for the report).
  assert.ok(
    x.leverM < 0.02,
    `lever arm ${x.leverM.toFixed(4)}m must be < 0.02m`
  );
});

check(
  'degenerate pure rotation: leverArmConstrained=false, K converges',
  () => {
    // No head translation at all. The lever arm is held at its prior (here: the
    // true value — the gate keeps it fixed on degenerate motion); the camera
    // translation is purely lever-arm-induced and drives the K recovery. The
    // trajectory uses a slower frame cadence (dt 0.7 s, ~13deg/frame rotation)
    // because rotation-only self-calibration carries less information per
    // observation: with 0.35 s frames the cy noise sigma exceeds 3 px, at 0.7 s
    // it is < 1.5 px over seeds (measured values are printed below).
    const pairs = generatePairs({
      seed: 13,
      frames: 40,
      noisePx: 0.5,
      translate: false,
      dt: 0.7,
      yawA: 0.4,
      pitchA: 0.45,
      rollA: 0.3,
    });
    const calib = new OnlineCalibrator({
      width: W,
      height: H,
      priorK: PRIOR_K,
      priorTHeadCamera: TRUE_T_HEAD_CAMERA,
    });
    feed(calib, pairs);
    const q = calib.quality;
    assert.equal(q.leverArmConstrained, false, 'lever arm must be gated off');
    const {last} = converge(calib);
    const e = kErrors(calib);
    const x = extrErrors(calib);
    console.log(
      `   pure rotation: ${fmtK(e)} · rms ${(last.rmsEpipolar * TRUE_K.fx).toFixed(2)}px · ` +
        `rotation ${x.rotDeg.toFixed(3)}deg · lever ${x.leverM.toFixed(4)}m (frozen at prior)`
    );
    assert.equal(calib.quality.leverArmConstrained, false, 'still gated');
    assert.equal(calib.quality.constrained, true, 'K side must be constrained');
    assert.ok(e.fxPct < 2, `fx ${e.fxPct.toFixed(2)}% must be < 2%`);
    assert.ok(e.fyPct < 2, `fy ${e.fyPct.toFixed(2)}% must be < 2%`);
    assert.ok(e.cxPx < 2, `cx ${e.cxPx.toFixed(2)}px must be < 2px`);
    assert.ok(e.cyPx < 2, `cy ${e.cyPx.toFixed(2)}px must be < 2px`);
  }
);

check('prior is initialization only: 20%-wrong prior still converges', () => {
  const priorK = [
    [1.2 * TRUE_K.fx, 0, 1.2 * TRUE_K.cx],
    [0, 1.2 * TRUE_K.fy, 1.2 * TRUE_K.cy],
    [0, 0, 1],
  ];
  const priorT = poseFromRt(
    rotMatFromRotVec(TRUE_RVEC.map((v) => v * 1.2)),
    TRUE_LEVER.map((v) => v * 1.2)
  );
  const pairs = generatePairs({seed: 17, frames: 26, noisePx: 0.5});
  const calib = new OnlineCalibrator({
    width: W,
    height: H,
    priorK,
    priorTHeadCamera: priorT,
  });
  feed(calib, pairs);
  const {last} = converge(calib);
  const e = kErrors(calib);
  const x = extrErrors(calib);
  console.log(
    `   20%-wrong prior: ${fmtK(e)} · rotation ${x.rotDeg.toFixed(3)}deg · ` +
      `lever ${x.leverM.toFixed(4)}m · rms ${(last.rmsEpipolar * TRUE_K.fx).toFixed(2)}px`
  );
  assert.ok(e.fxPct < 2, `fx ${e.fxPct.toFixed(2)}% must be < 2%`);
  assert.ok(e.fyPct < 2, `fy ${e.fyPct.toFixed(2)}% must be < 2%`);
  assert.ok(e.cxPx < 2, `cx ${e.cxPx.toFixed(2)}px must be < 2px`);
  assert.ok(e.cyPx < 2, `cy ${e.cyPx.toFixed(2)}px must be < 2px`);
  assert.ok(x.rotDeg < 2, `rotation ${x.rotDeg.toFixed(3)}deg must be < 2deg`);
  assert.ok(
    x.leverM < 0.02,
    `lever arm ${x.leverM.toFixed(4)}m must be < 0.02m`
  );
});

check('reset() clears the buffer and re-converges', () => {
  const pairs = generatePairs({seed: 19, frames: 26, noisePx: 0.5});
  const calib = new OnlineCalibrator({
    width: W,
    height: H,
    priorK: PRIOR_K,
    priorTHeadCamera: IDENTITY,
  });
  feed(calib, pairs);
  const first = converge(calib);
  assert.ok(calib.quality.pairs > 0, 'buffer populated');
  assert.ok(first.last.rmsEpipolar !== null, 'rms computed');

  calib.reset();
  assert.equal(calib.quality.pairs, 0, 'buffer cleared');
  assert.equal(calib.quality.steps, 0, 'step counter cleared');
  assert.equal(calib.quality.constrained, false, 'not constrained after reset');
  const k0 = KFromCalibrator(calib);
  assert.ok(
    Math.abs(k0.fx - PRIOR_K[0][0]) < 1e-9,
    'parameters back at the prior after reset'
  );
  assert.equal(
    calib.step().rmsEpipolar,
    null,
    'step() with empty buffer is a no-op'
  );

  feed(calib, pairs);
  const second = converge(calib);
  const e = kErrors(calib);
  console.log(
    `   after reset: ${fmtK(e)} · rms ${(second.last.rmsEpipolar * TRUE_K.fx).toFixed(2)}px`
  );
  assert.ok(e.fxPct < 2, `fx ${e.fxPct.toFixed(2)}% must be < 2%`);
  assert.ok(e.fyPct < 2, `fy ${e.fyPct.toFixed(2)}% must be < 2%`);
  assert.ok(e.cxPx < 2, `cx ${e.cxPx.toFixed(2)}px must be < 2px`);
  assert.ok(e.cyPx < 2, `cy ${e.cyPx.toFixed(2)}px must be < 2px`);
});

check('KFromCalibrator returns the fx/fy/cx/cy view', () => {
  const calib = new OnlineCalibrator({
    width: W,
    height: H,
    priorK: PRIOR_K,
    priorTHeadCamera: IDENTITY,
  });
  const k = KFromCalibrator(calib);
  assert.deepEqual(Object.keys(k).sort(), ['cx', 'cy', 'fx', 'fy']);
  assert.ok(Math.abs(k.fx - PRIOR_K[0][0]) < 1e-12);
  assert.ok(Math.abs(k.cx - PRIOR_K[0][2]) < 1e-12);
  const K = calib.K;
  assert.equal(K[0][1], 0);
  assert.equal(K[1][0], 0);
  assert.equal(K[2][2], 1);
});

check('robust to 25% outlier matches (raw descriptor matches)', () => {
  // The apps feed RAW matchDescriptors() output, which is outlier-
  // contaminated; the MAD observation gate must keep the solve on the true
  // minimum. 25% of correspondences are replaced with random false matches.
  const rng = mulberry32(99);
  const pairs = generatePairs({seed: 23, frames: 26, noisePx: 1.0});
  let corrupted = 0;
  for (const p of pairs) {
    for (const m of p.matches) {
      if (rng() < 0.25) {
        p.pts_j[m.trainIdx] = {x: rng() * W, y: rng() * H};
        corrupted++;
      }
    }
  }
  const calib = new OnlineCalibrator({
    width: W,
    height: H,
    priorK: PRIOR_K,
    priorTHeadCamera: IDENTITY,
  });
  feed(calib, pairs);
  const {last} = converge(calib);
  const e = kErrors(calib);
  const x = extrErrors(calib);
  console.log(
    `   25% outliers (${corrupted} corrupted matches): ${fmtK(e)} · ` +
      `rotation ${x.rotDeg.toFixed(3)}deg · lever ${x.leverM.toFixed(4)}m · ` +
      `rms ${(last.rmsEpipolar * TRUE_K.fx).toFixed(2)}px`
  );
  assert.ok(e.fxPct < 2, `fx ${e.fxPct.toFixed(2)}% must be < 2%`);
  assert.ok(e.fyPct < 2, `fy ${e.fyPct.toFixed(2)}% must be < 2%`);
  // Bounds are looser than the spec's clean-data 2px here (this scenario also
  // doubles the pixel noise to 1.0 px): the pin is that the solve stays ON the
  // true minimum instead of diverging on the outliers.
  assert.ok(e.cxPx < 3, `cx ${e.cxPx.toFixed(2)}px must be < 3px`);
  assert.ok(e.cyPx < 3, `cy ${e.cyPx.toFixed(2)}px must be < 3px`);
  assert.ok(x.rotDeg < 2, `rotation ${x.rotDeg.toFixed(3)}deg must be < 2deg`);
});

check('lever arm is clamped to a physical bound', () => {
  // Documented deviation pin: degenerate/weak data must not run the lever arm
  // away — components clamp to +-0.5 m (head-mounted camera bound).
  const calib = new OnlineCalibrator({
    width: W,
    height: H,
    priorK: PRIOR_K,
    priorTHeadCamera: poseFromRt(
      rotMatFromRotVec(TRUE_RVEC),
      [3, -2, 1.5] // absurd prior lever arm
    ),
  });
  const T = calib.T_head_camera;
  assert.ok(Math.abs(T[0][3]) <= 0.5, `tx ${T[0][3]} must clamp to 0.5`);
  assert.ok(Math.abs(T[1][3]) <= 0.5, `ty ${T[1][3]} must clamp to 0.5`);
  assert.ok(Math.abs(T[2][3]) <= 0.5, `tz ${T[2][3]} must clamp to 0.5`);
  console.log(`   clamped lever arm: [${T[0][3]}, ${T[1][3]}, ${T[2][3]}]`);
});

console.log(`\n${checks} checks passed`);
