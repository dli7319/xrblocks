/**
 * Two-view geometry, PnP and matrix utilities.
 *
 * Conventions (see IMPLEMENTATION_SPEC):
 * - Poses are nested `number[][]`; `T_a_b` is the pose of frame *b* expressed
 *   in frame *a*. `T_a_c = matMul(T_a_b, T_b_c)`.
 * - Camera model: pinhole `K = [[fx,0,cx],[0,fy,cy],[0,0,1]]`, image coords in
 *   pixels (u right, v down).
 * - `estimateEssential` returns the relative pose from view A to view B:
 *   `X_B = R * X_A + t` with `t` of unit norm (scale is unobservable).
 *
 * OpenCV.js note: `cv.findEssentialMat`, `cv.recoverPose` and
 * `cv.triangulatePoints` are *not* exported by any OpenCV.js build shipped
 * here (`@techstark/opencv-js` nor the official `docs.opencv.org/4.x/opencv.js`
 * CDN bundle), so those steps are implemented in plain JS. If a `cv` namespace
 * does expose them the OpenCV path is attempted first and any failure falls
 * back to the JS implementation, so both environments behave the same.
 * `cv.solvePnPRansac` and `cv.Rodrigues` are used directly where available.
 */

/** RANSAC iterations for the essential matrix estimate. */
const RANSAC_ITERATIONS = 150;
/** Sampson (symmetric epipolar) error threshold in pixels. */
const SAMPSON_THRESHOLD_PX = 2.0;
/** Reprojection threshold used to classify PnP inliers (pixels). */
const REPROJECTION_THRESHOLD_PX = 4.0;
/** Minimal sample size for the 8-point algorithm. */
const MIN_ESSENTIAL_SAMPLES = 8;

/* ------------------------------------------------------------------ *
 * Matrix utilities
 * ------------------------------------------------------------------ */

/**
 * Generic matrix product for nested number arrays (`4x4` per spec, but any
 * compatible dimensions work, e.g. `K(3x3) * [R|t](3x4)`).
 *
 * @param {number[][]} A
 * @param {number[][]} B
 * @returns {number[][]}
 */
export function matMul(A, B) {
  const rows = A.length;
  const inner = B.length;
  const cols = B[0].length;
  const out = new Array(rows);
  for (let i = 0; i < rows; i++) {
    const ai = A[i];
    const oi = new Array(cols);
    for (let j = 0; j < cols; j++) oi[j] = 0;
    for (let k = 0; k < inner; k++) {
      const aik = ai[k];
      if (aik === 0) continue;
      const bk = B[k];
      for (let j = 0; j < cols; j++) oi[j] += aik * bk[j];
    }
    out[i] = oi;
  }
  return out;
}

/**
 * Inverse of a rigid 4x4 pose: `[[R^T, -R^T t], [0 0 0 1]]`.
 *
 * @param {number[][]} T 4x4 pose
 * @returns {number[][]} 4x4 pose
 */
export function invertRigid(T) {
  const Rt = [
    [T[0][0], T[1][0], T[2][0]],
    [T[0][1], T[1][1], T[2][1]],
    [T[0][2], T[1][2], T[2][2]],
  ];
  const t = [T[0][3], T[1][3], T[2][3]];
  const nt = [
    -(Rt[0][0] * t[0] + Rt[0][1] * t[1] + Rt[0][2] * t[2]),
    -(Rt[1][0] * t[0] + Rt[1][1] * t[1] + Rt[1][2] * t[2]),
    -(Rt[2][0] * t[0] + Rt[2][1] * t[1] + Rt[2][2] * t[2]),
  ];
  return [
    [Rt[0][0], Rt[0][1], Rt[0][2], nt[0]],
    [Rt[1][0], Rt[1][1], Rt[1][2], nt[1]],
    [Rt[2][0], Rt[2][1], Rt[2][2], nt[2]],
    [0, 0, 0, 1],
  ];
}

/**
 * Build a 4x4 pose from a 3x3 rotation and a translation vector.
 *
 * @param {number[][]} R 3x3 rotation
 * @param {number[]} t length-3 translation
 * @returns {number[][]} 4x4 pose
 */
export function poseFromRt(R, t) {
  return [
    [R[0][0], R[0][1], R[0][2], t[0]],
    [R[1][0], R[1][1], R[1][2], t[1]],
    [R[2][0], R[2][1], R[2][2], t[2]],
    [0, 0, 0, 1],
  ];
}

/**
 * Rotation matrix -> quaternion `[x, y, z, w]` (normalized).
 *
 * @param {number[][]} R 3x3 rotation
 * @returns {number[]} `[x, y, z, w]`
 */
export function quatFromRotMat(R) {
  const trace = R[0][0] + R[1][1] + R[2][2];
  let x;
  let y;
  let z;
  let w;
  if (trace > 0) {
    const s = Math.sqrt(trace + 1) * 2;
    w = s / 4;
    x = (R[2][1] - R[1][2]) / s;
    y = (R[0][2] - R[2][0]) / s;
    z = (R[1][0] - R[0][1]) / s;
  } else if (R[0][0] > R[1][1] && R[0][0] > R[2][2]) {
    const s = Math.sqrt(1 + R[0][0] - R[1][1] - R[2][2]) * 2;
    w = (R[2][1] - R[1][2]) / s;
    x = s / 4;
    y = (R[0][1] + R[1][0]) / s;
    z = (R[0][2] + R[2][0]) / s;
  } else if (R[1][1] > R[2][2]) {
    const s = Math.sqrt(1 + R[1][1] - R[0][0] - R[2][2]) * 2;
    w = (R[0][2] - R[2][0]) / s;
    x = (R[0][1] + R[1][0]) / s;
    y = s / 4;
    z = (R[1][2] + R[2][1]) / s;
  } else {
    const s = Math.sqrt(1 + R[2][2] - R[0][0] - R[1][1]) * 2;
    w = (R[1][0] - R[0][1]) / s;
    x = (R[0][2] + R[2][0]) / s;
    y = (R[1][2] + R[2][1]) / s;
    z = s / 4;
  }
  const norm = Math.hypot(x, y, z, w) || 1;
  return [x / norm, y / norm, z / norm, w / norm];
}

/**
 * Quaternion `[x, y, z, w]` -> 3x3 rotation matrix.
 *
 * @param {number[]} q `[x, y, z, w]`
 * @returns {number[][]} 3x3 rotation
 */
export function rotMatFromQuat(q) {
  const norm = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
  const x = q[0] / norm;
  const y = q[1] / norm;
  const z = q[2] / norm;
  const w = q[3] / norm;
  return [
    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
  ];
}

/**
 * Angle in degrees between two vectors.
 *
 * @param {number[]} a
 * @param {number[]} b
 * @returns {number} degrees in `[0, 180]`, `0` for degenerate input
 */
export function degBetween(a, b) {
  const na = Math.hypot(a[0], a[1], a[2] ?? 0);
  const nb = Math.hypot(b[0], b[1], b[2] ?? 0);
  if (na === 0 || nb === 0) return 0;
  let dot = 0;
  for (let i = 0; i < 3; i++) dot += (a[i] ?? 0) * (b[i] ?? 0);
  const cosine = Math.max(-1, Math.min(1, dot / (na * nb)));
  return (Math.acos(cosine) * 180) / Math.PI;
}

/* ------------------------------------------------------------------ *
 * Two-view geometry
 * ------------------------------------------------------------------ */

/**
 * Two-view geometry from 2D-2D matches.
 *
 * Uses `cv.findEssentialMat`/`cv.recoverPose` when the build exports them,
 * otherwise a plain JS normalized 8-point RANSAC + essential decomposition.
 *
 * Cheirality: `t` is unit norm, its sign is chosen so that the inlier points
 * triangulate to positive depth in *both* views. Scale is unobservable, so
 * downstream triangulation must fix scale externally.
 *
 * @param {object} cv OpenCV namespace
 * @param {Array<{x: number, y: number}>} kpsA keypoints of view A
 * @param {Array<{x: number, y: number}>} kpsB keypoints of view B
 * @param {Array<{queryIdx: number, trainIdx: number}>} matches indices into kpsA/kpsB
 * @param {number[][]} K 3x3 intrinsics
 * @returns {{R: number[][], t: number[], inliers: number[], essentialMask: Uint8Array}|null}
 */
export function estimateEssential(cv, kpsA, kpsB, matches, K) {
  if (!Array.isArray(matches) || matches.length < MIN_ESSENTIAL_SAMPLES) {
    return null;
  }
  const pairs = [];
  for (let i = 0; i < matches.length; i++) {
    const m = matches[i];
    const a = kpsA[m.queryIdx];
    const b = kpsB[m.trainIdx];
    if (!a || !b) continue;
    const ax = pointX(a);
    const ay = pointY(a);
    const bx = pointX(b);
    const by = pointY(b);
    if (!isFinite(ax) || !isFinite(ay) || !isFinite(bx) || !isFinite(by)) {
      continue;
    }
    pairs.push({idx: i, x1: ax, y1: ay, x2: bx, y2: by});
  }
  if (pairs.length < MIN_ESSENTIAL_SAMPLES) return null;

  if (cv && typeof cv.findEssentialMat === 'function') {
    try {
      const fromCv = essentialWithCv(cv, pairs, K);
      if (fromCv) return fromCv;
    } catch {
      /* fall through to the JS implementation */
    }
  }
  return essentialWithJs(pairs, K);
}

/**
 * Plain JS essential matrix estimation: normalized 8-point RANSAC, manifold
 * projection, then decomposition + cheirality for R and unit t.
 *
 * @param {Array<{idx: number, x1: number, y1: number, x2: number, y2: number}>} pairs
 * @param {number[][]} K
 * @returns {{R: number[][], t: number[], inliers: number[], essentialMask: Uint8Array}|null}
 */
function essentialWithJs(pairs, K) {
  const random = makeRandom(0x2545f491);
  let bestSet = null;
  let bestCount = -1;
  for (let it = 0; it < RANSAC_ITERATIONS; it++) {
    const sample = sampleIndices(pairs.length, MIN_ESSENTIAL_SAMPLES, random);
    const E = eightPointE(
      sample.map((i) => pairs[i]),
      K
    );
    if (!E) continue;
    const set = collectEpipolarInliers(E, pairs, K);
    if (set.length > bestCount) {
      bestCount = set.length;
      bestSet = set;
      if (bestCount === pairs.length) break;
    }
  }
  if (!bestSet || bestCount < MIN_ESSENTIAL_SAMPLES) return null;

  // Refine on the consensus set (twice: refine, re-collect, refine again).
  let current = bestSet;
  let E = null;
  for (let pass = 0; pass < 3; pass++) {
    E = eightPointE(current, K);
    if (!E) break;
    const next = collectEpipolarInliers(E, pairs, K);
    if (next.length >= MIN_ESSENTIAL_SAMPLES && next.length > current.length) {
      current = next;
    } else {
      break;
    }
  }
  if (!E) E = eightPointE(current, K);
  if (!E) return null;
  current = collectEpipolarInliers(E, pairs, K);
  if (current.length < MIN_ESSENTIAL_SAMPLES) return null;

  const solution = decomposeWithCheirality(E, current, K);
  if (!solution) return null;

  const mask = new Uint8Array(pairs.length);
  const inliers = current.map((p) => {
    mask[p.idx] = 1;
    return p.idx;
  });
  inliers.sort((a, b) => a - b);
  return {R: solution.R, t: solution.t, inliers, essentialMask: mask};
}

/**
 * Estimate via `cv.findEssentialMat` + `cv.recoverPose` (builds that export
 * them). Throws on unexpected shapes so the caller can fall back to JS.
 *
 * @param {object} cv
 * @param {Array<{idx: number, x1: number, y1: number, x2: number, y2: number}>} pairs
 * @param {number[][]} K
 * @returns {{R: number[][], t: number[], inliers: number[], essentialMask: Uint8Array}|null}
 */
function essentialWithCv(cv, pairs, K) {
  const ptsA = new cv.Mat(pairs.length, 1, cv.CV_32FC2);
  const ptsB = new cv.Mat(pairs.length, 1, cv.CV_32FC2);
  const cam = mat3ToCv(cv, K);
  const ransac = typeof cv.RANSAC === 'number' ? cv.RANSAC : 3;
  for (let i = 0; i < pairs.length; i++) {
    ptsA.data32F[2 * i] = pairs[i].x1;
    ptsA.data32F[2 * i + 1] = pairs[i].y1;
    ptsB.data32F[2 * i] = pairs[i].x2;
    ptsB.data32F[2 * i + 1] = pairs[i].y2;
  }
  try {
    const essential = cv.findEssentialMat(ptsA, ptsB, cam, ransac, 0.999, 1.0);
    const E = Array.isArray(essential) ? essential[0] : essential;
    if (!E || E.rows !== 3 || E.cols !== 3) {
      throw new Error('findEssentialMat did not return a 3x3 matrix');
    }
    const Rm = new cv.Mat();
    const tm = new cv.Mat();
    const mask = new cv.Mat();
    try {
      let count = null;
      try {
        count = cv.recoverPose(E, ptsA, ptsB, cam, Rm, tm, mask);
      } catch {
        count = cv.recoverPose(E, ptsA, ptsB, cam, Rm, tm);
      }
      const R = matFromCv(Rm, 3, 3);
      const tRaw = readNumbers(tm, 3);
      if (!R || !tRaw)
        throw new Error('recoverPose returned unreadable output');
      const norm = Math.hypot(tRaw[0], tRaw[1], tRaw[2]);
      if (!(norm > 0))
        throw new Error('recoverPose returned a zero translation');
      const t = [tRaw[0] / norm, tRaw[1] / norm, tRaw[2] / norm];

      const inliers = [];
      const maskValues =
        mask.rows * mask.cols > 0
          ? readNumbers(mask, mask.rows * mask.cols)
          : null;
      for (let i = 0; i < pairs.length; i++) {
        const keep = maskValues
          ? maskValues[i] !== 0
          : count === null || i < count;
        if (keep) inliers.push(pairs[i].idx);
      }
      if (inliers.length < MIN_ESSENTIAL_SAMPLES) return null;
      const m = new Uint8Array(pairs.length);
      for (const idx of inliers) m[idx] = 1;
      return {R, t, inliers, essentialMask: m};
    } finally {
      Rm.delete();
      tm.delete();
      mask.delete();
      if (typeof E.delete === 'function') E.delete();
    }
  } finally {
    ptsA.delete();
    ptsB.delete();
    cam.delete();
  }
}

/**
 * Linear triangulation of `n` point pairs.
 *
 * `poseA`/`poseB` are `T_world_cam` (4x4). Points are pixel coordinates.
 * Returns a `Float64Array(n * 3)` of world points; rows are `NaN` when the
 * point cannot be placed in front of both cameras.
 *
 * @param {object} cv OpenCV namespace
 * @param {number[][]} poseA `T_world_cam` of view A
 * @param {number[][]} poseB `T_world_cam` of view B
 * @param {Array<{x: number, y: number}>} ptsA pixels in view A
 * @param {Array<{x: number, y: number}>} ptsB pixels in view B
 * @param {number[][]} K 3x3 intrinsics
 * @returns {Float64Array} `n * 3` world points
 */
export function triangulate(cv, poseA, poseB, ptsA, ptsB, K) {
  const n = Math.min(ptsA.length, ptsB.length);
  const out = new Float64Array(n * 3);
  out.fill(NaN);

  if (cv && typeof cv.triangulatePoints === 'function') {
    try {
      const viaCv = triangulateWithCv(cv, poseA, poseB, ptsA, ptsB, K, n);
      if (viaCv) return viaCv;
    } catch {
      /* fall through to the JS implementation */
    }
  }

  const TcwA = invertRigid(poseA);
  const TcwB = invertRigid(poseB);
  const P1 = matMul(K, toExtrinsic(TcwA));
  const P2 = matMul(K, toExtrinsic(TcwB));
  for (let i = 0; i < n; i++) {
    const X = triangulatePair(P1, P2, ptsA[i], ptsB[i]);
    if (!X) continue;
    const zA =
      TcwA[2][0] * X[0] + TcwA[2][1] * X[1] + TcwA[2][2] * X[2] + TcwA[2][3];
    const zB =
      TcwB[2][0] * X[0] + TcwB[2][1] * X[1] + TcwB[2][2] * X[2] + TcwB[2][3];
    if (!(zA > 0 && zB > 0)) continue;
    out[3 * i] = X[0];
    out[3 * i + 1] = X[1];
    out[3 * i + 2] = X[2];
  }
  return out;
}

/**
 * `cv.triangulatePoints` path (used only if the build exports it).
 *
 * @returns {Float64Array|null}
 */
function triangulateWithCv(cv, poseA, poseB, ptsA, ptsB, K, n) {
  const TcwA = invertRigid(poseA);
  const TcwB = invertRigid(poseB);
  const P1 = matMul(K, toExtrinsic(TcwA));
  const P2 = matMul(K, toExtrinsic(TcwB));
  const p1Mat = new cv.Mat(3, 4, cv.CV_64F);
  const p2Mat = new cv.Mat(3, 4, cv.CV_64F);
  const aMat = new cv.Mat(n, 1, cv.CV_64F + (1 << 3));
  const bMat = new cv.Mat(n, 1, cv.CV_64F + (1 << 3));
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 4; j++) {
      p1Mat.data64F[i * 4 + j] = P1[i][j];
      p2Mat.data64F[i * 4 + j] = P2[i][j];
    }
  }
  for (let i = 0; i < n; i++) {
    aMat.data64F[2 * i] = pointX(ptsA[i]);
    aMat.data64F[2 * i + 1] = pointY(ptsA[i]);
    bMat.data64F[2 * i] = pointX(ptsB[i]);
    bMat.data64F[2 * i + 1] = pointY(ptsB[i]);
  }
  try {
    const homog = cv.triangulatePoints(p1Mat, p2Mat, aMat, bMat);
    try {
      const values = homog.data64F || homog.data32F;
      if (!values) throw new Error('triangulatePoints returned no data');
      const out = new Float64Array(n * 3);
      out.fill(NaN);
      // cv.triangulatePoints returns a 4xN matrix (row-major); tolerate Nx4 too.
      const fourByN = homog.rows === 4;
      const element = fourByN
        ? (row, i) => values[row * homog.cols + i]
        : (row, i) => values[i * homog.cols + row];
      for (let i = 0; i < n; i++) {
        const w = element(3, i);
        if (!isFinite(w) || Math.abs(w) < 1e-12) continue;
        const X = [element(0, i) / w, element(1, i) / w, element(2, i) / w];
        const zA =
          TcwA[2][0] * X[0] +
          TcwA[2][1] * X[1] +
          TcwA[2][2] * X[2] +
          TcwA[2][3];
        const zB =
          TcwB[2][0] * X[0] +
          TcwB[2][1] * X[1] +
          TcwB[2][2] * X[2] +
          TcwB[2][3];
        if (!(zA > 0 && zB > 0)) continue;
        out[3 * i] = X[0];
        out[3 * i + 1] = X[1];
        out[3 * i + 2] = X[2];
      }
      return out;
    } finally {
      homog.delete();
    }
  } finally {
    p1Mat.delete();
    p2Mat.delete();
    aMat.delete();
    bMat.delete();
  }
}

/**
 * PnP RANSAC (EPNP, 100 iterations, 4 px reprojection error).
 *
 * @param {object} cv OpenCV namespace
 * @param {Array<{x: number, y: number, z: number}>} pts3d map-frame points
 * @param {Array<{x: number, y: number}>} pts2d pixel coordinates
 * @param {number[][]} K 3x3 intrinsics
 * @returns {{rvec: number[], tvec: number[], T_cam_obj: number[][], inliers: number[]}|null}
 */
export function solvePnPRansac(cv, pts3d, pts2d, K) {
  if (!cv || typeof cv.solvePnPRansac !== 'function') {
    throw new Error(
      'solvePnPRansac: the OpenCV namespace does not export solvePnPRansac'
    );
  }
  const n = Math.min(pts3d.length, pts2d.length);
  if (n < 4) return null;

  const objPoints = new cv.Mat(n, 1, cv.CV_32FC3);
  const imgPoints = new cv.Mat(n, 1, cv.CV_32FC2);
  const cam = mat3ToCv(cv, K);
  const dist = new cv.Mat();
  const ePnp = typeof cv.SOLVEPNP_EPNP === 'number' ? cv.SOLVEPNP_EPNP : 2;
  for (let i = 0; i < n; i++) {
    objPoints.data32F[3 * i] = pts3d[i].x;
    objPoints.data32F[3 * i + 1] = pts3d[i].y;
    objPoints.data32F[3 * i + 2] = pts3d[i].z;
    imgPoints.data32F[2 * i] = pointX(pts2d[i]);
    imgPoints.data32F[2 * i + 1] = pointY(pts2d[i]);
  }

  const attempts = [
    (rvec, tvec, inl) =>
      cv.solvePnPRansac(
        objPoints,
        imgPoints,
        cam,
        dist,
        rvec,
        tvec,
        false,
        100,
        4.0,
        0.99,
        inl,
        ePnp
      ),
    (rvec, tvec, inl) =>
      cv.solvePnPRansac(
        objPoints,
        imgPoints,
        cam,
        dist,
        rvec,
        tvec,
        false,
        100,
        4.0,
        0.99,
        inl
      ),
    (rvec, tvec, inl) =>
      cv.solvePnPRansac(objPoints, imgPoints, cam, dist, rvec, tvec, inl),
    (rvec, tvec, inl) =>
      cv.solvePnPRansac(objPoints, imgPoints, cam, dist, rvec, tvec),
  ];

  let parsed = null;
  try {
    for (const attempt of attempts) {
      const rvec = new cv.Mat();
      const tvec = new cv.Mat();
      const inl = new cv.Mat();
      try {
        attempt(rvec, tvec, inl);
        const rv = readNumbers(rvec, 3);
        const tv = readNumbers(tvec, 3);
        if (!rv || !tv || !isFiniteVector(rv) || !isFiniteVector(tv)) continue;
        parsed = {
          rvec: rv,
          tvec: tv,
          inliers: sanitizeInliers(readInliers(inl), n),
        };
        break;
      } catch {
        /* try the next signature */
      } finally {
        rvec.delete();
        tvec.delete();
        inl.delete();
      }
    }
  } finally {
    objPoints.delete();
    imgPoints.delete();
    cam.delete();
    dist.delete();
  }
  if (!parsed) return null;

  const R = rodriguesToMatrix(parsed.rvec);
  const T_cam_obj = poseFromRt(R, parsed.tvec);
  let inliers = parsed.inliers;
  if (!inliers || inliers.length < 4) {
    inliers = reprojectedInliers(pts3d, pts2d, T_cam_obj, K, n);
  }
  if (inliers.length < 4) return null;
  return {rvec: parsed.rvec, tvec: parsed.tvec, T_cam_obj, inliers};
}

/* ------------------------------------------------------------------ *
 * Pure JS geometry helpers
 * ------------------------------------------------------------------ */

/**
 * Normalized 8-point essential estimate for one minimal sample, projected
 * onto the essential manifold (`s, s, 0`).
 *
 * @returns {number[][]|null} 3x3 essential matrix
 */
function eightPointE(sample, K) {
  // The essential matrix is defined on *camera* coordinates (x = (u - cx)/fx),
  // not pixels, so de-project through K before the Hartley normalization.
  const fx = K[0][0];
  const fy = K[1][1];
  const cx = K[0][2];
  const cy = K[1][2];
  const camA = sample.map((p) => [(p.x1 - cx) / fx, (p.y1 - cy) / fy]);
  const camB = sample.map((p) => [(p.x2 - cx) / fx, (p.y2 - cy) / fy]);
  const normA = normalizePoints(camA);
  const normB = normalizePoints(camB);
  const A = [];
  for (let i = 0; i < sample.length; i++) {
    const [x1, y1] = applyTransform(normA.T, camA[i][0], camA[i][1]);
    const [x2, y2] = applyTransform(normB.T, camB[i][0], camB[i][1]);
    A.push([x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1]);
  }
  const nullSpace = nullVector(A, 9);
  if (!nullSpace || !isFiniteVector(nullSpace)) return null;
  const Ep = [
    [nullSpace[0], nullSpace[1], nullSpace[2]],
    [nullSpace[3], nullSpace[4], nullSpace[5]],
    [nullSpace[6], nullSpace[7], nullSpace[8]],
  ];
  // Undo Hartley normalization: E = T2^T * Ep * T1
  const E = matMul(matMul(transpose3(normB.T), Ep), normA.T);
  const projected = projectEssential(E);
  return projected;
}

/**
 * Sampson-error consensus set of an essential matrix (pixel space).
 *
 * @param {number[][]} E
 * @param {Array<{idx: number, x1: number, y1: number, x2: number, y2: number}>} pairs
 * @param {number[][]} K
 * @returns {Array<object>} inlying pairs
 */
function collectEpipolarInliers(E, pairs, K) {
  const F = fundamentalFromE(E, K);
  const thresholdSq = SAMPSON_THRESHOLD_PX * SAMPSON_THRESHOLD_PX;
  const inliers = [];
  for (const p of pairs) {
    if (sampsonSq(F, p.x1, p.y1, p.x2, p.y2) <= thresholdSq) inliers.push(p);
  }
  return inliers;
}

/**
 * `F = K^-T E K^-1`.
 */
function fundamentalFromE(E, K) {
  const fx = K[0][0];
  const fy = K[1][1];
  const cx = K[0][2];
  const cy = K[1][2];
  const Kinv = [
    [1 / fx, 0, -cx / fx],
    [0, 1 / fy, -cy / fy],
    [0, 0, 1],
  ];
  return matMul(matMul(transpose3(Kinv), E), Kinv);
}

/**
 * Squared Sampson distance in pixels between pixels `(x1,y1)` and `(x2,y2)`.
 */
function sampsonSq(F, x1, y1, x2, y2) {
  const a = [x1, y1, 1];
  const b = [x2, y2, 1];
  const Fa = mulVec3(F, a);
  const Ft = transpose3(F);
  const Fb = mulVec3(Ft, b);
  const cross = b[0] * Fa[0] + b[1] * Fa[1] + b[2] * Fa[2];
  const den = Fa[0] * Fa[0] + Fa[1] * Fa[1] + Fb[0] * Fb[0] + Fb[1] * Fb[1];
  if (!(den > 0)) return Infinity;
  return (cross * cross) / den;
}

/**
 * Project a 3x3 matrix onto the essential manifold (SVD -> `diag(s,s,0)`).
 *
 * @param {number[][]} E
 * @returns {number[][]|null}
 */
function projectEssential(E) {
  const svd = svd3(E);
  if (!svd) return null;
  const mid = (svd.s[0] + svd.s[1]) / 2;
  if (!(mid > 0)) return null;
  const U = svd.U;
  const V = svd.V;
  // Third columns carry no weight (sigma3 = 0), flip them for det = +1.
  if (determinant3(U) < 0) {
    U[0][2] = -U[0][2];
    U[1][2] = -U[1][2];
    U[2][2] = -U[2][2];
  }
  if (determinant3(V) < 0) {
    V[0][2] = -V[0][2];
    V[1][2] = -V[1][2];
    V[2][2] = -V[2][2];
  }
  const sigma = [
    [mid, 0, 0],
    [0, mid, 0],
    [0, 0, 0],
  ];
  const out = matMul(matMul(U, sigma), transpose3(V));
  return isFiniteMatrix(out) ? out : null;
}

/**
 * Decompose `E = U diag(s,s,0) V^T` into 4 (R, t) candidates and keep the one
 * that puts the most points in front of both cameras.
 *
 * @param {number[][]} E
 * @param {Array<{x1: number, y1: number, x2: number, y2: number}>} pairs
 * @param {number[][]} K
 * @returns {{R: number[][], t: number[]}|null}
 */
function decomposeWithCheirality(E, pairs, K) {
  const svd = svd3(E);
  if (!svd) return null;
  const U = svd.U;
  const V = svd.V;
  if (determinant3(U) < 0) {
    U[0][2] = -U[0][2];
    U[1][2] = -U[1][2];
    U[2][2] = -U[2][2];
  }
  if (determinant3(V) < 0) {
    V[0][2] = -V[0][2];
    V[1][2] = -V[1][2];
    V[2][2] = -V[2][2];
  }
  const W = [
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1],
  ];
  const Wt = transpose3(W);
  const Vt = transpose3(V);
  const R1 = matMul(matMul(U, W), Vt);
  const R2 = matMul(matMul(U, Wt), Vt);
  const t = [U[0][2], U[1][2], U[2][2]];
  const candidates = [
    {R: R1, t},
    {R: R1, t: [-t[0], -t[1], -t[2]]},
    {R: R2, t},
    {R: R2, t: [-t[0], -t[1], -t[2]]},
  ];

  const P1 = matMul(K, [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
  ]);
  let best = null;
  let bestScore = -1;
  for (const candidate of candidates) {
    const P2 = matMul(K, toExtrinsic(poseFromRt(candidate.R, candidate.t)));
    let score = 0;
    for (const p of pairs) {
      const X = triangulatePair(P1, P2, {x: p.x1, y: p.y1}, {x: p.x2, y: p.y2});
      if (!X) continue;
      const z1 = X[2];
      const z2 =
        candidate.R[2][0] * X[0] +
        candidate.R[2][1] * X[1] +
        candidate.R[2][2] * X[2] +
        candidate.t[2];
      if (z1 > 0 && z2 > 0) score++;
    }
    if (score > bestScore) {
      bestScore = score;
      best = candidate;
    }
  }
  if (!best || bestScore <= 0) return null;
  const norm = Math.hypot(best.t[0], best.t[1], best.t[2]);
  if (!(norm > 0)) return null;
  return {
    R: best.R,
    t: [best.t[0] / norm, best.t[1] / norm, best.t[2] / norm],
  };
}

/**
 * Linear triangulation of one pixel pair.
 *
 * @param {number[][]} P1 3x4 projection of view A
 * @param {number[][]} P2 3x4 projection of view B
 * @param {{x: number, y: number}} a pixel in view A
 * @param {{x: number, y: number}} b pixel in view B
 * @returns {number[]|null} `[X, Y, Z]` or null when degenerate
 */
function triangulatePair(P1, P2, a, b) {
  const x1 = pointX(a);
  const y1 = pointY(a);
  const x2 = pointX(b);
  const y2 = pointY(b);
  const rows = [
    rowCombination(P1, x1, 0),
    rowCombination(P1, y1, 1),
    rowCombination(P2, x2, 0),
    rowCombination(P2, y2, 1),
  ];
  // Row normalization keeps the normal equations well conditioned.
  for (const row of rows) {
    const norm = Math.hypot(row[0], row[1], row[2], row[3]);
    if (!(norm > 0)) return null;
    row[0] /= norm;
    row[1] /= norm;
    row[2] /= norm;
    row[3] /= norm;
  }
  const nullSpace = nullVector(rows, 4);
  if (!nullSpace || !isFiniteVector(nullSpace)) return null;
  const w = nullSpace[3];
  if (!isFinite(w) || Math.abs(w) < 1e-9) return null;
  return [nullSpace[0] / w, nullSpace[1] / w, nullSpace[2] / w];
}

/**
 * One DLT constraint row for `u = (P[0]·X) / (P[2]·X)`:
 * `u * P[2] - P[0]` (row 0) or `v * P[2] - P[1]` (row 1).
 *
 * @param {number[][]} P 3x4 projection matrix
 * @param {number} scale pixel coordinate
 * @param {0|1} row which projection row to use
 * @returns {number[]} length-4 row
 */
function rowCombination(P, scale, row) {
  const source = row === 1 ? P[1] : P[0];
  return [
    scale * P[2][0] - source[0],
    scale * P[2][1] - source[1],
    scale * P[2][2] - source[2],
    scale * P[2][3] - source[3],
  ];
}

/**
 * Unit vector spanning the null space of `M^T M` (smallest eigenvector).
 *
 * @param {number[][]} M rows
 * @param {number} cols
 * @returns {number[]|null}
 */
function nullVector(M, cols) {
  const gram = new Array(cols);
  for (let i = 0; i < cols; i++) {
    gram[i] = new Array(cols);
    for (let j = 0; j < cols; j++) gram[i][j] = 0;
  }
  for (const row of M) {
    for (let i = 0; i < cols; i++) {
      if (row[i] === 0) continue;
      for (let j = 0; j < cols; j++) gram[i][j] += row[i] * row[j];
    }
  }
  const eigen = symmetricEigen(gram, cols);
  if (!eigen || eigen.vectors.length === 0) return null;
  return eigen.vectors[0];
}

/**
 * Cyclic Jacobi eigen decomposition of a symmetric matrix.
 *
 * @param {number[][]} input square symmetric matrix
 * @param {number} n size
 * @returns {{values: number[], vectors: number[][]}|null} sorted ascending;
 *   `vectors[k]` is the unit eigenvector of `values[k]`
 */
function symmetricEigen(input, n) {
  const a = input.map((row) => row.slice());
  const v = identity2(n);
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

/**
 * SVD of a 3x3 matrix via `eigen(E^T E)` (descending singular values).
 *
 * @param {number[][]} M
 * @returns {{U: number[][], V: number[][], s: number[]}|null}
 */
function svd3(M) {
  const MtM = new Array(3);
  for (let i = 0; i < 3; i++) {
    MtM[i] = new Array(3);
    for (let j = 0; j < 3; j++) {
      MtM[i][j] = M[0][i] * M[0][j] + M[1][i] * M[1][j] + M[2][i] * M[2][j];
    }
  }
  const eigen = symmetricEigen(MtM, 3);
  if (!eigen) return null;
  // eigen is ascending -> flip to descending singular values.
  const order = [2, 1, 0];
  const V = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  const s = [];
  for (let col = 0; col < 3; col++) {
    const source = eigen.vectors[order[col]];
    const value = Math.sqrt(Math.max(eigen.values[order[col]], 0));
    s.push(value);
    for (let k = 0; k < 3; k++) V[k][col] = source[k];
  }
  const tol = 1e-12 * Math.max(s[0], 1);
  if (!(s[0] > 0)) return null;
  const u0 = normalize3(mulVec3(M, [V[0][0], V[1][0], V[2][0]]));
  if (!u0) return null;
  let u1 =
    s[1] > tol ? normalize3(mulVec3(M, [V[0][1], V[1][1], V[2][1]])) : null;
  if (!u1) {
    // Degenerate second column: build an orthonormal complement of u0.
    const axis = Math.abs(u0[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
    u1 = normalize3(cross3(u0, axis));
    if (!u1) return null;
  }
  // Gram-Schmidt for numerical safety.
  const dot01 = u1[0] * u0[0] + u1[1] * u0[1] + u1[2] * u0[2];
  u1 = normalize3([
    u1[0] - dot01 * u0[0],
    u1[1] - dot01 * u0[1],
    u1[2] - dot01 * u0[2],
  ]);
  if (!u1) return null;
  const u2 = normalize3(cross3(u0, u1));
  if (!u2) return null;
  const U = [
    [u0[0], u1[0], u2[0]],
    [u0[1], u1[1], u2[1]],
    [u0[2], u1[2], u2[2]],
  ];
  if (!isFiniteMatrix(U) || !isFiniteMatrix(V)) return null;
  return {U, V, s};
}

/* ------------------------------------------------------------------ *
 * Generic numeric helpers
 * ------------------------------------------------------------------ */

/**
 * Deterministic LCG so RANSAC results are reproducible across runs.
 *
 * @param {number} seed
 * @returns {() => number} `[0, 1)`
 */
function makeRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

/**
 * Sample `k` distinct indices from `[0, n)` (partial Fisher-Yates).
 */
function sampleIndices(n, k, random) {
  const indices = new Array(n);
  for (let i = 0; i < n; i++) indices[i] = i;
  for (let i = 0; i < k; i++) {
    const j = i + Math.floor(random() * (n - i));
    const tmp = indices[i];
    indices[i] = indices[j];
    indices[j] = tmp;
  }
  return indices.slice(0, k);
}

/**
 * Hartley normalization of 2D points.
 */
function normalizePoints(points) {
  let cx = 0;
  let cy = 0;
  for (const p of points) {
    cx += p[0];
    cy += p[1];
  }
  cx /= points.length;
  cy /= points.length;
  let meanDist = 0;
  for (const p of points) meanDist += Math.hypot(p[0] - cx, p[1] - cy);
  meanDist /= points.length;
  const scale = meanDist > 1e-12 ? Math.SQRT2 / meanDist : 1;
  return {
    T: [
      [scale, 0, -scale * cx],
      [0, scale, -scale * cy],
      [0, 0, 1],
    ],
  };
}

/**
 * Apply a 3x3 homogeneous transform to a pixel.
 */
function applyTransform(T, x, y) {
  const w = T[2][0] * x + T[2][1] * y + T[2][2];
  return [
    (T[0][0] * x + T[0][1] * y + T[0][2]) / w,
    (T[1][0] * x + T[1][1] * y + T[1][2]) / w,
  ];
}

/**
 * `[R|t]` from a `T_cam_world` pose.
 */
function toExtrinsic(TcamWorld) {
  return [
    [TcamWorld[0][0], TcamWorld[0][1], TcamWorld[0][2], TcamWorld[0][3]],
    [TcamWorld[1][0], TcamWorld[1][1], TcamWorld[1][2], TcamWorld[1][3]],
    [TcamWorld[2][0], TcamWorld[2][1], TcamWorld[2][2], TcamWorld[2][3]],
  ];
}

/**
 * Rodrigues (axis-angle) vector -> 3x3 rotation matrix.
 *
 * @param {number[]} r
 * @returns {number[][]}
 */
function rodriguesToMatrix(r) {
  const theta = Math.hypot(r[0], r[1], r[2]);
  if (theta < 1e-12) {
    return [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ];
  }
  const kx = r[0] / theta;
  const ky = r[1] / theta;
  const kz = r[2] / theta;
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  const v = 1 - c;
  return [
    [c + kx * kx * v, kx * ky * v - kz * s, kx * kz * v + ky * s],
    [ky * kx * v + kz * s, c + ky * ky * v, ky * kz * v - kx * s],
    [kz * kx * v - ky * s, kz * ky * v + kx * s, c + kz * kz * v],
  ];
}

/**
 * Recompute PnP inliers from reprojection error (used when the OpenCV build
 * does not return an inlier list).
 *
 * @returns {number[]}
 */
function reprojectedInliers(pts3d, pts2d, T_cam_obj, K, n) {
  const inliers = [];
  for (let i = 0; i < n; i++) {
    const X = pts3d[i];
    const xc =
      T_cam_obj[0][0] * X.x +
      T_cam_obj[0][1] * X.y +
      T_cam_obj[0][2] * X.z +
      T_cam_obj[0][3];
    const yc =
      T_cam_obj[1][0] * X.x +
      T_cam_obj[1][1] * X.y +
      T_cam_obj[1][2] * X.z +
      T_cam_obj[1][3];
    const zc =
      T_cam_obj[2][0] * X.x +
      T_cam_obj[2][1] * X.y +
      T_cam_obj[2][2] * X.z +
      T_cam_obj[2][3];
    if (!(zc > 0)) continue;
    const u = (K[0][0] * xc) / zc + K[0][2];
    const v = (K[1][1] * yc) / zc + K[1][2];
    const du = u - pointX(pts2d[i]);
    const dv = v - pointY(pts2d[i]);
    if (
      du * du + dv * dv <=
      REPROJECTION_THRESHOLD_PX * REPROJECTION_THRESHOLD_PX
    ) {
      inliers.push(i);
    }
  }
  return inliers;
}

/**
 * Copy a 3x3 OpenCV matrix into nested arrays.
 *
 * @returns {number[][]|null}
 */
function matFromCv(mat, rows, cols) {
  const values = readNumbers(mat, rows * cols);
  if (!values) return null;
  const out = [];
  for (let i = 0; i < rows; i++)
    out.push(values.slice(i * cols, (i + 1) * cols));
  return isFiniteMatrix(out) ? out : null;
}

/**
 * Map an OpenCV Mat depth constant to its typed-array accessor.
 * `CV_8U=0, CV_8S=1, CV_16U=2, CV_16S=3, CV_32S=4, CV_32F=5, CV_64F=6`.
 */
const DEPTH_KEYS = {
  0: ['data'],
  1: ['data', 'data8S'],
  2: ['data', 'data16U'],
  3: ['data', 'data16S'],
  4: ['data32S', 'data'],
  5: ['data32F', 'data'],
  6: ['data64F', 'data'],
};

/**
 * Read the first `n` values out of an OpenCV Mat of any numeric type.
 *
 * @returns {number[]|null}
 */
function readNumbers(mat, n) {
  if (!mat) return null;
  let cells = 0;
  try {
    cells = mat.rows * mat.cols;
  } catch {
    return null;
  }
  if (cells < n) return null;
  let depth = -1;
  try {
    if (typeof mat.depth === 'function') depth = mat.depth();
    else if (typeof mat.type === 'function') depth = mat.type() & 7;
  } catch {
    depth = -1;
  }
  const preferred = DEPTH_KEYS[depth] || [
    'data32S',
    'data32F',
    'data64F',
    'data',
  ];
  for (const key of preferred) {
    try {
      const array = mat[key];
      if (array && array.length >= n) {
        const out = new Array(n);
        for (let i = 0; i < n; i++) out[i] = array[i];
        return out;
      }
    } catch {
      /* try the next typed view */
    }
  }
  return null;
}

/**
 * Read PnP inlier indices from `cv.solvePnPRansac` output, or null.
 *
 * @returns {number[]|null}
 */
function readInliers(mat) {
  if (!mat) return null;
  let cells = 0;
  try {
    cells = mat.rows * mat.cols;
  } catch {
    return null;
  }
  if (cells <= 0) return null;
  const values = readNumbers(mat, cells);
  if (!values) return null;
  return values.map((value) => Math.trunc(value));
}

/**
 * Keep only indices inside `[0, n)` and unique, sorted.
 *
 * @returns {number[]|null}
 */
function sanitizeInliers(indices, n) {
  if (!indices) return null;
  const seen = new Set();
  const out = [];
  for (const index of indices) {
    if (index < 0 || index >= n || seen.has(index)) continue;
    seen.add(index);
    out.push(index);
  }
  out.sort((a, b) => a - b);
  return out;
}

/**
 * 3x3 nested matrix -> `cv.Mat(CV_64F)` (caller deletes it).
 *
 * @returns {object}
 */
function mat3ToCv(cv, K) {
  const mat = new cv.Mat(3, 3, cv.CV_64F);
  mat.data64F.set([
    K[0][0],
    K[0][1],
    K[0][2],
    K[1][0],
    K[1][1],
    K[1][2],
    K[2][0],
    K[2][1],
    K[2][2],
  ]);
  return mat;
}

function transpose3(M) {
  return [
    [M[0][0], M[1][0], M[2][0]],
    [M[0][1], M[1][1], M[2][1]],
    [M[0][2], M[1][2], M[2][2]],
  ];
}

function mulVec3(M, v) {
  return [
    M[0][0] * v[0] + M[0][1] * v[1] + M[0][2] * v[2],
    M[1][0] * v[0] + M[1][1] * v[1] + M[1][2] * v[2],
    M[2][0] * v[0] + M[2][1] * v[1] + M[2][2] * v[2],
  ];
}

function cross3(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function normalize3(v) {
  const norm = Math.hypot(v[0], v[1], v[2]);
  if (!(norm > 0)) return null;
  return [v[0] / norm, v[1] / norm, v[2] / norm];
}

function determinant3(M) {
  return (
    M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) -
    M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
    M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0])
  );
}

function identity2(n) {
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = new Array(n);
    for (let j = 0; j < n; j++) out[i][j] = i === j ? 1 : 0;
  }
  return out;
}

function pointX(p) {
  return p && p.x !== undefined ? p.x : p[0];
}

function pointY(p) {
  return p && p.y !== undefined ? p.y : p[1];
}

function isFiniteVector(v) {
  for (const value of v) if (!isFinite(value)) return false;
  return true;
}

function isFiniteMatrix(M) {
  for (const row of M) {
    if (!Array.isArray(row)) return false;
    for (const value of row) if (!isFinite(value)) return false;
  }
  return true;
}
