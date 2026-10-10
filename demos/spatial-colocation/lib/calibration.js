/**
 * Hand-eye *rotation* calibration (camera -> head extrinsics).
 *
 * Model: the camera and the head are rigidly linked by `X = R_head_camera`, so
 * a camera pose observed in an arbitrary constant frame satisfies
 *
 * ```text
 * C_i = G * H_i * X        (G = unknown constant frame alignment, rotations)
 * ```
 *
 * Relative motions give the conjugacy `dC = G * dH * G^-1`, i.e. `A X = X B`
 * with `A = dC`, `B = dH`, solved for `G` with the quaternion method (null
 * space of the stacked `L(a) - R(b)` system). `X` then follows from the
 * absolute poses: `X_i = H_i^T * G^T * C_i`, averaged over all samples.
 * The formulation also covers the already-aligned case `G = I` (`C_i = H_i X`).
 *
 * Translation of `T_head_camera` defaults to `[0, 0, 0]` (camera approximated
 * at the head center); the app exposes a manual override.
 */

import {matMul, quatFromRotMat, rotMatFromQuat} from './geometry.js';

/** Minimum number of valid pose samples. */
const MIN_SAMPLES = 10;
/** Minimum rotational excitation: max angle between consecutive head rotations. */
const MIN_EXCITATION_DEG = 2;

/**
 * Estimate the rotation part of `T_head_camera` from a motion sequence.
 *
 * @param {Array<number[][]>} headRotations `T_ref_head` (4x4) or 3x3 rotation per frame
 * @param {Array<number[][]>} cameraRotations `T_ref_cam` (4x4) or 3x3 rotation per frame
 * @returns {{R_head_camera: number[][], samples: number}|null} null when the
 *   sample count is too low or there is insufficient rotational excitation
 */
export function estimateHandEyeRotation(headRotations, cameraRotations) {
  if (!Array.isArray(headRotations) || !Array.isArray(cameraRotations)) {
    return null;
  }
  const count = Math.min(headRotations.length, cameraRotations.length);
  const H = [];
  const C = [];
  for (let i = 0; i < count; i++) {
    const h = toRotation(headRotations[i]);
    const c = toRotation(cameraRotations[i]);
    if (!h || !c) continue;
    H.push(h);
    C.push(c);
  }
  const samples = H.length;
  if (samples < MIN_SAMPLES) return null;

  // Excitation gate: the head must actually rotate between frames.
  let maxAngleDeg = 0;
  for (let i = 0; i + 1 < samples; i++) {
    const relative = matMul(H[i + 1], transpose3(H[i]));
    maxAngleDeg = Math.max(maxAngleDeg, rotationAngleDeg(relative));
  }
  if (maxAngleDeg < MIN_EXCITATION_DEG) return null;

  // Relative motions -> AX = XB (A = camera motion, B = head motion) -> G.
  const pairs = [];
  for (let i = 0; i + 1 < samples; i++) {
    pairs.push({
      a: quatFromRotMat(matMul(C[i + 1], transpose3(C[i]))),
      b: quatFromRotMat(matMul(H[i + 1], transpose3(H[i]))),
    });
  }
  const alignment = solveAxXb(pairs);
  if (!alignment) return null;
  const G = rotMatFromQuat(alignment);

  // Absolute recovery of X = R_head_camera, averaged over all samples.
  const quaternions = [];
  const Gt = transpose3(G);
  for (let i = 0; i < samples; i++) {
    const X = matMul(matMul(transpose3(H[i]), Gt), C[i]);
    if (!isFiniteMatrix(X)) continue;
    quaternions.push(quatFromRotMat(X));
  }
  if (quaternions.length < MIN_SAMPLES) return null;
  const average = averageQuaternions(quaternions);
  if (!average) return null;
  const R_head_camera = rotMatFromQuat(average);
  if (!isFiniteMatrix(R_head_camera)) return null;
  return {R_head_camera, samples};
}

/**
 * Solve `A_i X = X B_i` for a single rotation `X` using quaternions.
 *
 * Each pair contributes the linear constraint `(L(a) - R(b)) x = 0`; the
 * solution is the null space of the stacked system (smallest eigenvector of
 * the Gram matrix). A degenerate null space (more than one free direction,
 * e.g. single-axis excitation) is rejected.
 *
 * @param {Array<{a: number[], b: number[]}>} pairs quaternions `[x,y,z,w]`
 * @returns {number[]|null} unit quaternion `[x, y, z, w]`
 */
function solveAxXb(pairs) {
  if (pairs.length === 0) return null;
  const rows = [];
  for (const pair of pairs) {
    const left = leftMultiplyMatrix(pair.a);
    const right = rightMultiplyMatrix(pair.b);
    for (let i = 0; i < 4; i++) {
      const row = new Array(4);
      for (let j = 0; j < 4; j++) row[j] = left[i][j] - right[i][j];
      rows.push(row);
    }
  }

  const gram = [
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
  ];
  for (const row of rows) {
    for (let i = 0; i < 4; i++) {
      if (row[i] === 0) continue;
      for (let j = 0; j < 4; j++) gram[i][j] += row[i] * row[j];
    }
  }
  const eigen = symmetricEigen(gram, 4);
  if (!eigen) return null;
  // eigen.values ascending: [0] is the null space, [1] must be a real
  // constraint, otherwise X is not uniquely determined.
  const largest = eigen.values[3];
  if (!(largest > 0) || !(eigen.values[1] > 1e-6 * largest)) return null;
  const q = eigen.vectors[0];
  const norm = Math.hypot(q[0], q[1], q[2], q[3]);
  if (!(norm > 0)) return null;
  return [q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm];
}

/**
 * 4x4 matrix of left multiplication by quaternion `a` (`r = a (x) q`), acting
 * on vectors ordered `[x, y, z, w]`.
 */
function leftMultiplyMatrix(a) {
  const [ax, ay, az, aw] = a;
  return [
    [aw, -az, ay, ax],
    [az, aw, -ax, ay],
    [-ay, ax, aw, az],
    [-ax, -ay, -az, aw],
  ];
}

/**
 * 4x4 matrix of right multiplication by quaternion `b` (`r = q (x) b`),
 * acting on vectors ordered `[x, y, z, w]`.
 */
function rightMultiplyMatrix(b) {
  const [bx, by, bz, bw] = b;
  return [
    [bw, bz, -by, bx],
    [-bz, bw, bx, by],
    [by, -bx, bw, bz],
    [-bx, -by, -bz, bw],
  ];
}

/**
 * Sign-aligned mean of unit quaternions, re-normalized.
 *
 * @param {number[][]} quaternions `[x, y, z, w]`
 * @returns {number[]|null}
 */
function averageQuaternions(quaternions) {
  const reference = quaternions[0];
  let sum = [0, 0, 0, 0];
  for (const q of quaternions) {
    const aligned =
      q[0] * reference[0] +
        q[1] * reference[1] +
        q[2] * reference[2] +
        q[3] * reference[3] <
      0
        ? [-q[0], -q[1], -q[2], -q[3]]
        : q;
    sum = [
      sum[0] + aligned[0],
      sum[1] + aligned[1],
      sum[2] + aligned[2],
      sum[3] + aligned[3],
    ];
  }
  const norm = Math.hypot(sum[0], sum[1], sum[2], sum[3]);
  if (!(norm > 0)) return null;
  return [sum[0] / norm, sum[1] / norm, sum[2] / norm, sum[3] / norm];
}

/**
 * Accept a 3x3 rotation or a 4x4 pose and return its 3x3 rotation, or null
 * when the input is not a finite rotation.
 *
 * @returns {number[][]|null}
 */
function toRotation(value) {
  if (!Array.isArray(value)) return null;
  const rows = value.length;
  if (rows === 3 && Array.isArray(value[0]) && value[0].length === 3) {
    return isFiniteMatrix(value) ? value.map((row) => row.slice()) : null;
  }
  if (rows >= 3 && Array.isArray(value[0]) && value[0].length >= 3) {
    const R = [
      value[0].slice(0, 3),
      value[1].slice(0, 3),
      value[2].slice(0, 3),
    ];
    return isFiniteMatrix(R) ? R : null;
  }
  return null;
}

function transpose3(M) {
  return [
    [M[0][0], M[1][0], M[2][0]],
    [M[0][1], M[1][1], M[2][1]],
    [M[0][2], M[1][2], M[2][2]],
  ];
}

function rotationAngleDeg(T) {
  const trace = T[0][0] + T[1][1] + T[2][2];
  const cosine = Math.max(-1, Math.min(1, (trace - 1) / 2));
  return (Math.acos(cosine) * 180) / Math.PI;
}

function isFiniteMatrix(M) {
  for (const row of M) {
    if (!Array.isArray(row)) return false;
    for (const value of row) if (!isFinite(value)) return false;
  }
  return true;
}

/**
 * Cyclic Jacobi eigen decomposition of a symmetric matrix, sorted ascending.
 *
 * @param {number[][]} input square symmetric matrix
 * @param {number} n size
 * @returns {{values: number[], vectors: number[][]}|null}
 */
function symmetricEigen(input, n) {
  const a = input.map((row) => row.slice());
  const v = new Array(n);
  for (let i = 0; i < n; i++) {
    v[i] = new Array(n);
    for (let j = 0; j < n; j++) v[i][j] = i === j ? 1 : 0;
  }
  const maxSweeps = Math.max(30, 6 * n);
  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) off += a[p][q] * a[p][q];
    }
    if (!(off > 1e-300)) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = a[p][q];
        if (Math.abs(apq) < 1e-300) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * apq);
        const t =
          (theta >= 0 ? 1 : -1) /
          (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        const app = a[p][p];
        const aqq = a[q][q];
        for (let k = 0; k < n; k++) {
          if (k === p || k === q) continue;
          const akp = a[k][p];
          const akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
          a[p][k] = a[k][p];
          a[q][k] = a[k][q];
        }
        a[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq;
        a[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq;
        a[p][q] = 0;
        a[q][p] = 0;
        for (let k = 0; k < n; k++) {
          const vkp = v[k][p];
          const vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq;
          v[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }

  const order = [];
  for (let i = 0; i < n; i++) order.push(i);
  order.sort((i, j) => a[i][i] - a[j][j]);
  const values = [];
  const vectors = [];
  for (const idx of order) {
    if (!isFinite(a[idx][idx])) return null;
    const vec = new Array(n);
    let norm = 0;
    for (let k = 0; k < n; k++) {
      vec[k] = v[k][idx];
      norm += vec[k] * vec[k];
    }
    norm = Math.sqrt(norm);
    if (!(norm > 0)) return null;
    for (let k = 0; k < n; k++) vec[k] /= norm;
    values.push(a[idx][idx]);
    vectors.push(vec);
  }
  return {values, vectors};
}
