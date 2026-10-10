/** @vitest-environment node */
/**
 * Two-view geometry: triangulation accuracy, PnP pose recovery, essential
 * matrix estimation from synthetic correspondences, plus matrix utilities.
 */
import {beforeAll, describe, expect, test} from 'vitest';
import {loadCv} from './cv-runtime.js';
import {
  degBetween,
  estimateEssential,
  invertRigid,
  matMul,
  poseFromRt,
  quatFromRotMat,
  rotMatFromQuat,
  solvePnPRansac,
  triangulate,
} from './geometry.js';

let cv;

const K = [
  [520, 0, 320],
  [0, 515, 240],
  [0, 0, 1],
];

/** Deterministic LCG. */
function makeRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

/** Rotation about +Y by `deg`. */
function yaw(deg) {
  const r = (deg * Math.PI) / 180;
  return [
    [Math.cos(r), 0, Math.sin(r)],
    [0, 1, 0],
    [-Math.sin(r), 0, Math.cos(r)],
  ];
}

/** Rotation about +X by `deg`. */
function pitch(deg) {
  const r = (deg * Math.PI) / 180;
  return [
    [1, 0, 0],
    [0, Math.cos(r), -Math.sin(r)],
    [0, Math.sin(r), Math.cos(r)],
  ];
}

function matMul3(A, B) {
  const out = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      for (let k = 0; k < 3; k++) out[i][j] += A[i][k] * B[k][j];
    }
  }
  return out;
}

function transpose3(M) {
  return [
    [M[0][0], M[1][0], M[2][0]],
    [M[0][1], M[1][1], M[2][1]],
    [M[0][2], M[1][2], M[2][2]],
  ];
}

/** Angle in degrees between two rotations. */
function rotationErrorDeg(Ra, Rb) {
  const relative = matMul3(Ra, transpose3(Rb));
  const trace = relative[0][0] + relative[1][1] + relative[2][2];
  return (
    (Math.acos(Math.max(-1, Math.min(1, (trace - 1) / 2))) * 180) / Math.PI
  );
}

/**
 * Project a world point with `T_world_cam = [R|t]` (camera center = t,
 * camera axes = columns of R).
 */
function project(P, R, t, intrinsics) {
  const v = [P[0] - t[0], P[1] - t[1], P[2] - t[2]];
  const x = R[0][0] * v[0] + R[1][0] * v[1] + R[2][0] * v[2];
  const y = R[0][1] * v[0] + R[1][1] * v[1] + R[2][1] * v[2];
  const z = R[0][2] * v[0] + R[1][2] * v[1] + R[2][2] * v[2];
  if (!(z > 0)) return null;
  return [
    (intrinsics[0][0] * x) / z + intrinsics[0][2],
    (intrinsics[1][1] * y) / z + intrinsics[1][2],
    z,
  ];
}

/** Non-planar random scene in front of the origin. */
function makeScene(count, seed) {
  const random = makeRandom(seed);
  const points = [];
  for (let i = 0; i < count; i++) {
    points.push([
      (random() - 0.5) * 4,
      (random() - 0.5) * 3,
      2.5 + random() * 3.5,
    ]);
  }
  return points;
}

const identity = [
  [1, 0, 0, 0],
  [0, 1, 0, 0],
  [0, 0, 1, 0],
  [0, 0, 0, 1],
];

function poseFromRotation(R, t) {
  return poseFromRt(R, t);
}

describe('geometry', () => {
  beforeAll(async () => {
    cv = await loadCv();
  }, 60000);

  test('triangulate() recovers known 3D points from two known poses', () => {
    const scene = makeScene(24, 4242);
    const poseA = identity;
    const poseB = poseFromRotation(yaw(12), [0.35, 0.08, 0.05]);

    const ptsA = [];
    const ptsB = [];
    const expected = [];
    for (const point of scene) {
      const a = project(point, identity, [0, 0, 0], K);
      const b = project(point, yaw(12), [0.35, 0.08, 0.05], K);
      expect(a).not.toBeNull();
      expect(b).not.toBeNull();
      ptsA.push({x: a[0], y: a[1]});
      ptsB.push({x: b[0], y: b[1]});
      expected.push(point);
    }

    const recovered = triangulate(cv, poseA, poseB, ptsA, ptsB, K);
    expect(recovered).toBeInstanceOf(Float64Array);
    expect(recovered.length).toBe(expected.length * 3);
    for (let i = 0; i < expected.length; i++) {
      const dx = recovered[3 * i] - expected[i][0];
      const dy = recovered[3 * i + 1] - expected[i][1];
      const dz = recovered[3 * i + 2] - expected[i][2];
      expect(Math.hypot(dx, dy, dz)).toBeLessThan(1e-3);
    }
  });

  test('triangulate() returns NaN rows for points behind a camera', () => {
    // Camera B sits far down +Z, so points in front of A are behind B.
    const poseB = poseFromRotation(
      [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ],
      [0, 0, 10]
    );
    const point = [0.4, 0.2, 3];
    const a = project(point, identity, [0, 0, 0], K);
    expect(a).not.toBeNull();
    const recovered = triangulate(
      cv,
      identity,
      poseB,
      [{x: a[0], y: a[1]}],
      [{x: a[0], y: a[1]}],
      K
    );
    expect(Number.isNaN(recovered[0])).toBe(true);
    expect(Number.isNaN(recovered[1])).toBe(true);
    expect(Number.isNaN(recovered[2])).toBe(true);
  });

  test('solvePnPRansac() recovers a known pose from perfect correspondences', () => {
    const scene = makeScene(40, 1337);
    const R = matMul3(pitch(-8), yaw(15));
    const t = [0.4, -0.2, 0.1];
    const T_world_cam = poseFromRotation(R, t);

    const pts3d = [];
    const pts2d = [];
    for (const point of scene) {
      const pixel = project(point, R, t, K);
      if (!pixel) continue;
      pts3d.push({x: point[0], y: point[1], z: point[2]});
      pts2d.push({x: pixel[0], y: pixel[1]});
    }
    expect(pts3d.length).toBeGreaterThanOrEqual(30);

    const solution = solvePnPRansac(cv, pts3d, pts2d, K);
    expect(solution).not.toBeNull();
    expect(solution.inliers.length).toBeGreaterThanOrEqual(pts3d.length - 4);

    const T_cam_obj = invertRigid(T_world_cam);
    const rotationError = rotationErrorDeg(solution.T_cam_obj, T_cam_obj);
    expect(rotationError).toBeLessThan(0.5);

    const translationError = Math.hypot(
      solution.T_cam_obj[0][3] - T_cam_obj[0][3],
      solution.T_cam_obj[1][3] - T_cam_obj[1][3],
      solution.T_cam_obj[2][3] - T_cam_obj[2][3]
    );
    expect(translationError).toBeLessThan(0.01);

    // rvec/tvec are plain number arrays consistent with T_cam_obj.
    expect(solution.rvec).toHaveLength(3);
    expect(solution.tvec).toHaveLength(3);
    expect(solution.tvec.every((value) => Number.isFinite(value))).toBe(true);
  });

  test('solvePnPRansac() rejects gross outliers via RANSAC', () => {
    const scene = makeScene(36, 5150);
    const R = matMul3(pitch(6), yaw(-12));
    const t = [-0.3, 0.15, 0.05];
    const T_world_cam = poseFromRotation(R, t);
    const random = makeRandom(77);

    const pts3d = [];
    const pts2d = [];
    for (let i = 0; i < scene.length; i++) {
      const point = scene[i];
      const pixel = project(point, R, t, K);
      if (!pixel) continue;
      pts3d.push({x: point[0], y: point[1], z: point[2]});
      if (i % 6 === 0) {
        // Gross outlier: unrelated pixel.
        pts2d.push({x: random() * 640, y: random() * 480});
      } else {
        pts2d.push({x: pixel[0], y: pixel[1]});
      }
    }

    const solution = solvePnPRansac(cv, pts3d, pts2d, K);
    expect(solution).not.toBeNull();
    const T_cam_obj = invertRigid(T_world_cam);
    expect(rotationErrorDeg(solution.T_cam_obj, T_cam_obj)).toBeLessThan(2);
    const translationError = Math.hypot(
      solution.T_cam_obj[0][3] - T_cam_obj[0][3],
      solution.T_cam_obj[1][3] - T_cam_obj[1][3],
      solution.T_cam_obj[2][3] - T_cam_obj[2][3]
    );
    expect(translationError).toBeLessThan(0.03);
    // Outliers are excluded from the inlier set.
    expect(solution.inliers.length).toBeLessThan(pts3d.length);
    expect(solution.inliers.length).toBeGreaterThan(pts3d.length * 0.75);
  });

  test('solvePnPRansac() returns null when there are too few points', () => {
    expect(
      solvePnPRansac(cv, [{x: 0, y: 0, z: 1}], [{x: 1, y: 1}], K)
    ).toBeNull();
    expect(
      solvePnPRansac(
        cv,
        [
          {x: 0, y: 0, z: 1},
          {x: 1, y: 0, z: 1},
        ],
        [
          {x: 1, y: 1},
          {x: 2, y: 2},
        ],
        K
      )
    ).toBeNull();
  });

  test('estimateEssential() recovers the relative pose of a synthetic pair', () => {
    const scene = makeScene(70, 2024);
    const R = matMul3(pitch(-5), yaw(10));
    const t = [0.28, 0.06, 0.03];

    const kpsA = [];
    const kpsB = [];
    const matches = [];
    for (const point of scene) {
      const a = project(point, identity, [0, 0, 0], K);
      const b = project(point, R, t, K);
      if (!a || !b) continue;
      const index = matches.length;
      kpsA.push({x: a[0], y: a[1]});
      kpsB.push({x: b[0], y: b[1]});
      matches.push({queryIdx: index, trainIdx: index, distance: 0});
    }
    expect(matches.length).toBeGreaterThan(50);

    const essential = estimateEssential(cv, kpsA, kpsB, matches, K);
    expect(essential).not.toBeNull();

    // The returned (R, t) is the relative pose A -> B: X_B = R * X_A + t.
    // With `T_world_cam = [R|t]` per view that is R_B^T and -R_B^T * t_B.
    const Rrel = transpose3(R);
    const trel = [
      -(Rrel[0][0] * t[0] + Rrel[0][1] * t[1] + Rrel[0][2] * t[2]),
      -(Rrel[1][0] * t[0] + Rrel[1][1] * t[1] + Rrel[1][2] * t[2]),
      -(Rrel[2][0] * t[0] + Rrel[2][1] * t[1] + Rrel[2][2] * t[2]),
    ];
    expect(rotationErrorDeg(essential.R, Rrel)).toBeLessThan(2);
    expect(essential.inliers.length).toBeGreaterThan(30);

    // Translation is recovered up to scale, with the sign fixed by cheirality.
    const norm = Math.hypot(essential.t[0], essential.t[1], essential.t[2]);
    expect(norm).toBeCloseTo(1, 6);
    expect(degBetween(essential.t, trel)).toBeLessThan(5);
    expect(essential.essentialMask).toBeInstanceOf(Uint8Array);
    expect(essential.essentialMask.length).toBe(matches.length);
    const maskSum = essential.essentialMask.reduce(
      (sum, value) => sum + value,
      0
    );
    expect(maskSum).toBe(essential.inliers.length);

    // inliers are sorted indices into `matches`.
    expect([...essential.inliers].sort((a, b) => a - b)).toEqual(
      essential.inliers
    );
  });

  test('estimateEssential() returns null for degenerate input', () => {
    const matches = [
      {queryIdx: 0, trainIdx: 0},
      {queryIdx: 1, trainIdx: 1},
    ];
    expect(
      estimateEssential(
        cv,
        [
          {x: 1, y: 1},
          {x: 2, y: 2},
        ],
        [
          {x: 1, y: 1},
          {x: 2, y: 2},
        ],
        matches,
        K
      )
    ).toBeNull();
    expect(estimateEssential(cv, [], [], [], K)).toBeNull();
    expect(
      estimateEssential(
        cv,
        [{x: 0, y: 0}],
        [{x: 0, y: 0}],
        [{queryIdx: 0, trainIdx: 0}],
        K
      )
    ).toBeNull();
  });

  test('matMul() composes poses and matrices', () => {
    const a = poseFromRotation(yaw(90), [1, 0, 0]);
    const b = poseFromRotation(yaw(90), [0, 2, 0]);
    const composed = matMul(a, b);
    expect(composed).toHaveLength(4);
    // 90deg twice = 180deg about Y.
    expect(composed[0][0]).toBeCloseTo(-1, 9);
    expect(composed[2][2]).toBeCloseTo(-1, 9);
    // Translation of b moved into a's frame: t_a + R_a * t_b = [1,2,0].
    expect(composed[0][3]).toBeCloseTo(1, 9);
    expect(composed[1][3]).toBeCloseTo(2, 9);
    expect(composed[2][3]).toBeCloseTo(0, 9);

    // K (3x3) * [R|t] (3x4) -> 3x4.
    const projection = matMul(K, [
      [1, 0, 0, 0],
      [0, 1, 0, 5],
      [0, 0, 1, -3],
    ]);
    expect(projection).toHaveLength(3);
    expect(projection[2]).toEqual([0, 0, 1, -3]);
    // Row 0 of K dotted with the translation column [0, 5, -3].
    expect(projection[0][3]).toBeCloseTo(-3 * K[0][2], 9);
    expect(projection[1][3]).toBeCloseTo(5 * K[1][1] - 3 * K[1][2], 9);
  });

  test('invertRigid() inverts rigid poses', () => {
    const pose = poseFromRotation(
      matMul3(pitch(20), yaw(-35)),
      [0.4, -0.2, 1.5]
    );
    const inverse = invertRigid(pose);
    const product = matMul(pose, inverse);
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        const expected = i === j ? 1 : 0;
        expect(product[i][j]).toBeCloseTo(expected, 10);
      }
    }
    // Inverse of [R|t] is [R^T, -R^T t].
    expect(inverse[0][3]).toBeCloseTo(
      -(pose[0][0] * 0.4 + pose[1][0] * -0.2 + pose[2][0] * 1.5),
      10
    );
  });

  test('poseFromRt() builds a homogeneous pose', () => {
    const R = yaw(30);
    const pose = poseFromRt(R, [1, 2, 3]);
    expect(pose).toEqual([
      [R[0][0], R[0][1], R[0][2], 1],
      [R[1][0], R[1][1], R[1][2], 2],
      [R[2][0], R[2][1], R[2][2], 3],
      [0, 0, 0, 1],
    ]);
  });

  test('quaternions round-trip through rotation matrices', () => {
    expect(
      quatFromRotMat([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ])
    ).toEqual([0, 0, 0, 1]);

    const random = makeRandom(31415);
    for (let trial = 0; trial < 25; trial++) {
      const axis = [random() * 2 - 1, random() * 2 - 1, random() * 2 - 1];
      const axisNorm = Math.hypot(axis[0], axis[1], axis[2]) || 1;
      const angle = random() * Math.PI * 2;
      const c = Math.cos(angle / 2);
      const s = Math.sin(angle / 2);
      const R = rotMatFromQuat([
        (axis[0] / axisNorm) * s,
        (axis[1] / axisNorm) * s,
        (axis[2] / axisNorm) * s,
        c,
      ]);
      const q = quatFromRotMat(R);
      const back = rotMatFromQuat(q);
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
          expect(back[i][j]).toBeCloseTo(R[i][j], 9);
        }
      }
      // Orthonormal rotation.
      const shouldBeI = matMul3(R, transpose3(R));
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
          expect(shouldBeI[i][j]).toBeCloseTo(i === j ? 1 : 0, 9);
        }
      }
    }
  });

  test('degBetween() measures angles between vectors', () => {
    expect(degBetween([1, 0, 0], [1, 0, 0])).toBeCloseTo(0, 9);
    expect(degBetween([1, 0, 0], [0, 1, 0])).toBeCloseTo(90, 9);
    expect(degBetween([1, 0, 0], [-1, 0, 0])).toBeCloseTo(180, 9);
    expect(degBetween([0, 0, 0], [1, 0, 0])).toBe(0);
    expect(degBetween([3, 4], [0, 5])).toBeCloseTo(
      (Math.acos(4 / 5) * 180) / Math.PI,
      9
    );
  });
});
