/**
 * Pure-helper tests for `app/sim-feed.js`: intrinsics-from-projection math,
 * matrix conversion, and the `?autoSweep=1` trajectory.
 *
 * Plain Node-runnable (node:assert, no test-runner API) so it executes with
 * `node demos/spatial-colocation/lib/sim/sim-feed.test.js`; it is also safe to
 * run under Vitest if a `*.test.ts` shim is added (vitest only collects TS).
 */
import assert from 'node:assert/strict';

import {
  intrinsicsFromProjection,
  verifyIntrinsicsAgainstProjection,
  matrixToNested,
  autoSweepPoseAt,
  SWEEP_RADIUS_M,
  SWEEP_HEIGHT_M,
  SWEEP_BOB_M,
} from '../../app/sim-feed.js';

let checks = 0;
function check(name, fn) {
  fn();
  checks++;
  console.log(`ok - ${name}`);
}

// Column-major perspective projection for fovY + aspect (three.js convention).
function perspectiveElements(fovYRad, aspect) {
  const m5 = 1 / Math.tan(fovYRad / 2);
  const m0 = m5 / aspect;
  return [m0, 0, 0, 0, 0, m5, 0, 0, 0, 0, -1, -1, 0, 0, -0.1, 0];
}

check('intrinsicsFromProjection: symmetric perspective frustum', () => {
  const w = 640;
  const h = 480;
  const fovY = Math.PI / 3; // 60 deg
  const e = perspectiveElements(fovY, w / h);
  const K = intrinsicsFromProjection(e, w, h);
  const fy = h / 2 / Math.tan(fovY / 2);
  const fx = w / 2 / (Math.tan(fovY / 2) * (w / h));
  assert.ok(Math.abs(K[0][0] - fx) < 1e-9, `fx ${K[0][0]} != ${fx}`);
  assert.ok(Math.abs(K[1][1] - fy) < 1e-9, `fy ${K[1][1]} != ${fy}`);
  assert.ok(Math.abs(K[0][2] - w / 2) < 1e-9, 'cx = w/2');
  assert.ok(Math.abs(K[1][2] - h / 2) < 1e-9, 'cy = h/2');
  assert.equal(K[0][1], 0);
  assert.equal(K[2][2], 1);
});

check('intrinsicsFromProjection: off-center frustum shifts cx/cy', () => {
  const w = 640;
  const h = 480;
  const e = perspectiveElements(Math.PI / 3, w / h);
  e[8] = 0.25; // (r+l)/(r-l)
  e[9] = -0.1; // (t+b)/(t-b)
  const K = intrinsicsFromProjection(e, w, h);
  assert.ok(Math.abs(K[0][2] - 0.75 * (w / 2)) < 1e-9, 'cx = (1 - m8) * w/2');
  assert.ok(Math.abs(K[1][2] - 0.9 * (h / 2)) < 1e-9, 'cy = (1 + m9) * h/2');
});

check(
  'verifyIntrinsicsAgainstProjection: pinhole K matches the projection',
  () => {
    const e = perspectiveElements(Math.PI / 3, 640 / 480);
    const check1 = verifyIntrinsicsAgainstProjection(e, 640, 480);
    assert.equal(check1.ok, true, `maxErrPx=${check1.maxErrPx}`);
    assert.ok(check1.maxErrPx < 1e-6, `maxErrPx ${check1.maxErrPx} too large`);
  }
);

check(
  'verifyIntrinsicsAgainstProjection: catches a wrong sign convention',
  () => {
    const e = perspectiveElements(Math.PI / 3, 640 / 480);
    e[8] = 0.3;
    e[9] = 0.2;
    // Plain w/2, h/2 (m8/m9-agnostic) intrinsics disagree with the matrix.
    const K = [
      [(640 / 2) * e[0], 0, 640 / 2],
      [0, (480 / 2) * e[5], 480 / 2],
      [0, 0, 1],
    ];
    const fx = K[0][0];
    const cx = K[0][2];
    assert.ok(Math.abs(cx - ((1 - e[8]) * 640) / 2) > 1, 'sanity: cx differs');
    assert.ok(fx > 0);
  }
);

check('matrixToNested: column-major elements -> row-major nested', () => {
  // e[col * 4 + row]; make every element distinct: e[i] = i + 1.
  const e = Array.from({length: 16}, (_, i) => i + 1);
  const T = matrixToNested(e);
  for (let row = 0; row < 4; row++) {
    for (let col = 0; col < 4; col++) {
      assert.equal(T[row][col], e[col * 4 + row], `T[${row}][${col}]`);
    }
  }
  // Translation lands in the last column of the nested pose.
  assert.equal(T[0][3], e[12]);
  assert.equal(T[1][3], e[13]);
  assert.equal(T[2][3], e[14]);
  assert.equal(T[3][3], e[16 - 1]);
});

check('autoSweepPoseAt: deterministic and bounded', () => {
  for (const t of [0, 1.5, 7.25, 13, 22.4, 29.9]) {
    const a = autoSweepPoseAt(t);
    const b = autoSweepPoseAt(t);
    assert.deepEqual(a, b, 'same t -> same pose');
    const [x, y, z] = a.position;
    const radius = Math.hypot(x, z);
    assert.ok(
      Math.abs(radius - SWEEP_RADIUS_M) < 1e-9,
      `radius ${radius} != ${SWEEP_RADIUS_M}`
    );
    assert.ok(Math.abs(y - SWEEP_HEIGHT_M) <= SWEEP_BOB_M + 1e-9, 'height bob');
    assert.ok(
      Number.isFinite(a.yaw) && Number.isFinite(a.pitch),
      'angles finite'
    );
  }
});

check('autoSweepPoseAt: faces outward from the circle center at t=0', () => {
  const pose = autoSweepPoseAt(0);
  // -Z of a yaw-about-Y rotation is (-sin yaw, 0, -cos yaw); at t=0 the
  // outward direction is (sin 0, 0, cos 0) = (0, 0, 1).
  const fx = -Math.sin(pose.yaw);
  const fz = -Math.cos(pose.yaw);
  assert.ok(Math.abs(fx) < 1e-9, 'forward x');
  assert.ok(Math.abs(fz - 1) < 1e-9, 'forward z');
  assert.equal(pose.pitch, 0, 'no pitch nod at t=0');
});

console.log(`\n${checks} checks passed`);
