/** @vitest-environment node */
/**
 * Hand-eye rotation calibration: recovery of a known `R_head_camera` from a
 * synthetic motion sequence, plus rejection of degenerate input.
 */
import {describe, expect, test} from 'vitest';
import {
  matMul,
  poseFromRt,
  quatFromRotMat,
  rotMatFromQuat,
} from './geometry.js';
import {estimateHandEyeRotation} from './calibration.js';

function yaw(deg) {
  const r = (deg * Math.PI) / 180;
  return [
    [Math.cos(r), 0, Math.sin(r)],
    [0, 1, 0],
    [-Math.sin(r), 0, Math.cos(r)],
  ];
}
function pitch(deg) {
  const r = (deg * Math.PI) / 180;
  return [
    [1, 0, 0],
    [0, Math.cos(r), -Math.sin(r)],
    [0, Math.sin(r), Math.cos(r)],
  ];
}
function roll(deg) {
  const r = (deg * Math.PI) / 180;
  return [
    [Math.cos(r), -Math.sin(r), 0],
    [Math.sin(r), Math.cos(r), 0],
    [0, 0, 1],
  ];
}
function transpose3(M) {
  return [
    [M[0][0], M[1][0], M[2][0]],
    [M[0][1], M[1][1], M[2][1]],
    [M[0][2], M[1][2], M[2][2]],
  ];
}
function rotationErrorDeg(Ra, Rb) {
  const relative = matMul(Ra, transpose3(Rb));
  const trace = relative[0][0] + relative[1][1] + relative[2][2];
  return (
    (Math.acos(Math.max(-1, Math.min(1, (trace - 1) / 2))) * 180) / Math.PI
  );
}
function makeRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

/**
 * Model: `C_i = G * H_i * X` with an unknown constant frame alignment `G` and
 * the hand-eye rotation `X = R_head_camera`.
 */
function makeSequence({count = 16, X, G, jitterDeg = 0}) {
  const random = makeRandom(4242);
  const H = [];
  const C = [];
  for (let i = 0; i < count; i++) {
    // Excited head motion: oscillating yaw + pitch + a little roll.
    const head = matMul(
      matMul(yaw(25 * Math.sin(i * 0.7)), pitch(18 * Math.cos(i * 0.5))),
      roll(12 * Math.sin(i * 0.9))
    );
    H.push(head);
    let camera = matMul(matMul(G, head), X);
    if (jitterDeg > 0) {
      const jitter = matMul(
        matMul(
          yaw((random() - 0.5) * 2 * jitterDeg),
          pitch((random() - 0.5) * 2 * jitterDeg)
        ),
        roll((random() - 0.5) * 2 * jitterDeg)
      );
      camera = matMul(jitter, camera);
    }
    C.push(camera);
  }
  return {H, C};
}

describe('calibration', () => {
  test('recovers a known hand-eye rotation from an excited sequence', () => {
    const X = matMul(pitch(15), yaw(-32));
    const G = matMul(yaw(40), pitch(-20));
    const {H, C} = makeSequence({count: 16, X, G});

    const result = estimateHandEyeRotation(H, C);
    expect(result).not.toBeNull();
    expect(result.samples).toBe(16);
    expect(result.R_head_camera).toHaveLength(3);
    expect(rotationErrorDeg(result.R_head_camera, X)).toBeLessThan(1);
  });

  test('accepts 4x4 poses as well as 3x3 rotations', () => {
    const X = matMul(roll(24), yaw(18));
    const G = pitch(-35);
    const {H, C} = makeSequence({count: 14, X, G});

    const headPoses = H.map((R, i) => poseFromRt(R, [0.1 * i, 0.02 * i, 0.05]));
    const cameraPoses = C.map((R, i) =>
      poseFromRt(R, [0.05 * i, -0.03 * i, 0.01])
    );
    const result = estimateHandEyeRotation(headPoses, cameraPoses);
    expect(result).not.toBeNull();
    expect(rotationErrorDeg(result.R_head_camera, X)).toBeLessThan(1);
  });

  test('tolerates small rotational noise', () => {
    const X = matMul(pitch(12), yaw(28));
    const G = yaw(-45);
    const {H, C} = makeSequence({count: 30, X, G, jitterDeg: 0.4});
    const result = estimateHandEyeRotation(H, C);
    expect(result).not.toBeNull();
    expect(rotationErrorDeg(result.R_head_camera, X)).toBeLessThan(2);
  });

  test('returns null for too few samples', () => {
    const X = yaw(10);
    const G = pitch(20);
    const short = makeSequence({count: 9, X, G});
    expect(estimateHandEyeRotation(short.H, short.C)).toBeNull();
    expect(estimateHandEyeRotation([], [])).toBeNull();
    expect(estimateHandEyeRotation(null, null)).toBeNull();
    // Mismatched / invalid entries are dropped first.
    const withGarbage = makeSequence({count: 12, X, G});
    for (const index of [3, 4, 5, 6]) withGarbage.H[index] = [NaN, NaN, NaN];
    for (const index of [7, 8, 9, 10]) withGarbage.C[index] = 'nope';
    expect(estimateHandEyeRotation(withGarbage.H, withGarbage.C)).toBeNull();
  });

  test('returns null without sufficient rotational excitation', () => {
    const X = matMul(pitch(15), yaw(-32));
    const G = matMul(yaw(40), pitch(-20));
    const {H, C} = makeSequence({count: 16, X, G});

    // Head never moves between frames (< 2 deg).
    const still = H.map((R) => H[0]);
    const stillCamera = still.map((R) => matMul(matMul(G, R), X));
    expect(estimateHandEyeRotation(still, stillCamera)).toBeNull();

    // Head rotates, but only 0.5 deg between frames: below the 2 deg floor.
    const small = [];
    for (let i = 0; i < 16; i++) small.push(matMul(H[0], yaw(0.5 * i)));
    const smallCamera = small.map((R) => matMul(matMul(G, R), X));
    expect(estimateHandEyeRotation(small, smallCamera)).toBeNull();
  });

  test('returns null for a degenerate single-axis sequence', () => {
    const X = matMul(pitch(15), yaw(-32));
    const G = matMul(yaw(40), pitch(-20));
    const H = [];
    const C = [];
    for (let i = 0; i < 16; i++) {
      const head = yaw(6 * i); // all rotations share one axis
      H.push(head);
      C.push(matMul(matMul(G, head), X));
    }
    expect(estimateHandEyeRotation(H, C)).toBeNull();
  });

  test('quaternion helpers stay consistent with the calibration input', () => {
    const R = matMul(yaw(33), pitch(-17));
    const q = quatFromRotMat(R);
    const back = rotMatFromQuat(q);
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        expect(back[i][j]).toBeCloseTo(R[i][j], 9);
      }
    }
  });
});
