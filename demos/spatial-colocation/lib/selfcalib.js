// lib/selfcalib.js
/**
 * Online camera self-calibration (SELF_CALIB_SPEC.md).
 *
 * Estimates the 10 unknowns — intrinsics `fx, fy, cx, cy` plus the
 * camera-to-head extrinsics `T_head_camera` (rotation vector `rx, ry, rz` and
 * lever arm `tx, ty, tz`) — from metric head-pose motion + 2D-2D matches, by
 * damped Gauss-Newton (Levenberg) on Sampson-normalized epipolar residuals
 * with NUMERIC (central-difference) Jacobians and a 10x10 normal system
 * solved by Gaussian elimination with partial pivoting. Priors are
 * OPTIMIZATION INITIALIZATION ONLY; the estimate converges from data.
 *
 * Conventions (pinned by the known-pose projection test in
 * `lib/selfcalib.test.js`):
 * - poses: 4x4 nested `number[][]`, three/WebXR convention (forward = -Z,
 *   y up), `T_a_b` = pose of frame b expressed in frame a
 *   (`T_a_c = T_a_b · T_b_c`);
 * - pixels: image convention (origin top-left, u right, v down);
 * - pixel -> three-convention normalized bearing at the boundary:
 *   `x = (u - cx) / fx`, `y = -(v - cy) / fy`, `z = -1` (z = -1 = in front).
 *
 * Per matched pair (frames i, j), with `T_head_camera = X`:
 *   `T_ref_cam = T_ref_head · X`
 *   the pair transform in the i -> j direction is `X^-1 · A · X = (R_ij, t_ij)`
 *   where `A = inv(T_ref_head_j) · T_ref_head_i` is precomputed per pair, so
 *   `p_j = R_ij · p_i + t_ij` and the epipolar constraint is
 *   `x_jᵀ E x_i = 0` with `E = [t_ij]_× R_ij` (the spec's residual form;
 *   `T_a_b` stays in the IMPLEMENTATION_SPEC convention everywhere else).
 * The residual is the Sampson-normalized form from the spec:
 *   `r = (x_jᵀ E x_i) / sqrt(gx² + gy²)`
 * with Sampson gradient terms `(gx, gy)` = the first two components of
 * `E x_i` (the gradient of the constraint w.r.t. x_j's normalized image
 * coordinates — the point-to-epipolar-line normalization). Pairs are skipped
 * where the denominator is degenerate (< 1e-9) or either bearing fails the
 * in-front (z) guard. Because E is defined up to scale the residual is
 * invariant to the (possibly scale-ambiguous) magnitude of the input head
 * translations.
 */

const MAX_PAIRS = 400; // ring buffer cap (frame pairs), drop oldest
const MAX_MATCHES_PER_PAIR = 64; // per-pair match subsample (compute bound)
const MIN_MATCHES_PER_PAIR = 6; // below this a pair is not worth storing
const MIN_OBS = 30; // residuals required to run the optimizer
const MIN_PAIRS_FOR_CONSTRAINT = 10;
const MAX_LM_ITERATIONS = 12; // per step(); shorter cycles re-trim more often
const FD_STEP = 1e-5; // central-difference step (relative to param scale)
const LAMBDA0 = 1e-3;
const LAMBDA_MIN = 1e-10;
const LAMBDA_MAX = 1e10;
const DEN_EPS = 1e-9; // degenerate Sampson denominator threshold
const TRIM_SIGMA = 2.5; // robust observation gate (MAD sigmas, per step())
const TRIM_FLOOR = 1e-7; // ~5e-5 px — keeps perfect/near-perfect data intact
const LEVER_MAX_M = 0.5; // physical lever-arm bound (head-mounted camera)
const SMALL_DELTA = 1e-9; // stop when max |dp| / (1 + |p|) drops below
const PLATEAU = 1e-12; // stop when relative cost gain drops below

// Observability gates.
const ROT_AXIS_MIN_RAD = (0.1 * Math.PI) / 180; // per-pair rotation counts
const ROT_AXIS_DISTINCT_RAD = (20 * Math.PI) / 180; // "distinct axes" separation
const ROT_TOTAL_MIN_RAD = (5 * Math.PI) / 180; // total rotation excitation
const ROT_EIGEN_RATIO_MIN = 0.005; // 2nd/1st scatter eigenvalue ratio
const MIN_ROT_AXES = 4;
const TRANS_DIR_MIN_M = 0.001; // per-pair translation counts
const TRANS_DIVERSE_MIN = 0.08; // RMS direction spread to unlock the lever arm
const MIN_TRANS_DIRS = 4;

// ---- small matrix helpers (nested number[][], self-contained) ---------------

/** 3x3 rotation part of a 4x4 pose. */
function rot3(T) {
  return [
    [T[0][0], T[0][1], T[0][2]],
    [T[1][0], T[1][1], T[1][2]],
    [T[2][0], T[2][1], T[2][2]],
  ];
}

function transpose3(R) {
  return [
    [R[0][0], R[1][0], R[2][0]],
    [R[0][1], R[1][1], R[2][1]],
    [R[0][2], R[1][2], R[2][2]],
  ];
}

function mul3(A, B) {
  const out = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      out[i][j] = A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j];
    }
  }
  return out;
}

function mul3v(R, v) {
  return [
    R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
    R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
    R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2],
  ];
}

/** Rodrigues: rotation vector (axis * angle, radians) -> 3x3. */
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

/** 3x3 -> rotation vector (axis * angle). Handles the theta ~ 0 / ~ pi cases. */
function rotVecFromRotMat(R) {
  const trace = R[0][0] + R[1][1] + R[2][2];
  const cos = Math.min(1, Math.max(-1, (trace - 1) / 2));
  const theta = Math.acos(cos);
  const w = [R[2][1] - R[1][2], R[0][2] - R[2][0], R[1][0] - R[0][1]];
  if (theta < 1e-8) {
    return [w[0] / 2, w[1] / 2, w[2] / 2];
  }
  if (Math.PI - theta < 1e-6) {
    // R ~ -I + 2 kkᵀ -> axis from the symmetric part (sign-safe).
    const m = [
      [(R[0][0] + 1) / 2, (R[0][1] + R[1][0]) / 4, (R[0][2] + R[2][0]) / 4],
      [(R[0][1] + R[1][0]) / 4, (R[1][1] + 1) / 2, (R[1][2] + R[2][1]) / 4],
      [(R[0][2] + R[2][0]) / 4, (R[1][2] + R[2][1]) / 4, (R[2][2] + 1) / 2],
    ];
    let idx = 0;
    if (m[1][1] > m[idx][idx]) idx = 1;
    if (m[2][2] > m[idx][idx]) idx = 2;
    const axis = [0, 0, 0];
    axis[idx] = Math.sqrt(Math.max(0, m[idx][idx]));
    for (let i = 0; i < 3; i++) {
      if (i !== idx) axis[i] = m[i][idx] / (axis[idx] || 1);
    }
    const n = Math.hypot(axis[0], axis[1], axis[2]) || 1;
    return [
      (theta * axis[0]) / n,
      (theta * axis[1]) / n,
      (theta * axis[2]) / n,
    ];
  }
  const s = Math.sin(theta);
  return [
    (w[0] * theta) / (2 * s),
    (w[1] * theta) / (2 * s),
    (w[2] * theta) / (2 * s),
  ];
}

/** [t]x · R (cross-product skew matrix times a 3x3). */
function skewTimes(t, R) {
  const [tx, ty, tz] = t;
  return [
    [
      tz * R[1][0] - ty * R[2][0],
      tz * R[1][1] - ty * R[2][1],
      tz * R[1][2] - ty * R[2][2],
    ],
    [
      tx * R[2][0] - tz * R[0][0],
      tx * R[2][1] - tz * R[0][1],
      tx * R[2][2] - tz * R[0][2],
    ],
    [
      ty * R[0][0] - tx * R[1][0],
      ty * R[0][1] - tx * R[1][1],
      ty * R[0][2] - tx * R[1][2],
    ],
  ];
}

/**
 * Relative transform `inv(Ti) · Tj` for rigid 4x4 poses, as (R, t) with
 * `p_i = R · p_j + t`.
 */
function relativeTransform(Ti, Tj) {
  const RiT = transpose3(rot3(Ti));
  const d = [Tj[0][3] - Ti[0][3], Tj[1][3] - Ti[1][3], Tj[2][3] - Ti[2][3]];
  return {
    R: mul3(RiT, rot3(Tj)),
    t: mul3v(RiT, d),
  };
}

/** Gaussian elimination with partial pivoting. Returns x or null. */
function solveLinear(A, b) {
  const n = b.length;
  const M = A.map((row) => row.slice());
  const x = b.slice();
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    }
    if (Math.abs(M[piv][col]) < 1e-300) return null;
    if (piv !== col) {
      const tmp = M[piv];
      M[piv] = M[col];
      M[col] = tmp;
      const tv = x[piv];
      x[piv] = x[col];
      x[col] = tv;
    }
    for (let r = col + 1; r < n; r++) {
      const f = M[r][col] / M[col][col];
      if (f === 0) continue;
      for (let c = col; c < n; c++) M[r][c] -= f * M[col][c];
      x[r] -= f * x[col];
    }
  }
  for (let r = n - 1; r >= 0; r--) {
    let s = x[r];
    for (let c = r + 1; c < n; c++) s -= M[r][c] * x[c];
    x[r] = s / M[r][r];
    if (!Number.isFinite(x[r])) return null;
  }
  return x;
}

// ---- robust observation gate ------------------------------------------------

/**
 * Robust observation gate (MAD): keep layout entries whose residual is within
 * TRIM_SIGMA robust sigmas of the median residual. Descriptor matches are
 * noisy AND outlier-contaminated; plain least squares over raw matches is not
 * robust, so this trim runs once per step() before the (unchanged) LM solve.
 * A degenerate spread (perfect data) keeps everything.
 *
 * @param {Int32Array} layout [entryIdx, obsOffset] pairs
 * @param {Float64Array} res residuals at the base parameters
 * @returns {Int32Array} the trimmed layout
 */
function robustTrim(layout, res) {
  const m = res.length;
  if (m < 12) return layout;
  const sorted = Float64Array.from(res).sort();
  const median = sorted[m >> 1];
  const dev = new Float64Array(m);
  for (let i = 0; i < m; i++) dev[i] = Math.abs(res[i] - median);
  const devSorted = Float64Array.from(dev).sort();
  const mad = devSorted[m >> 1] * 1.4826;
  const gate = Math.max(TRIM_SIGMA * mad, TRIM_FLOOR);
  const keep = [];
  for (let i = 0; i < m; i++) {
    if (dev[i] <= gate) {
      keep.push(layout[2 * i], layout[2 * i + 1]);
    }
  }
  return Int32Array.from(keep);
}

// ---- observability metrics --------------------------------------------------

/**
 * RMS spread of the unit head-translation directions across pairs
 * ("lever-arm-agnostic": it uses the head-frame pair translation, which does
 * not depend on the extrinsics being estimated). 0 for pure rotation or a
 * single fixed direction.
 */
function translationDiversity(buffer) {
  const dirs = [];
  for (const entry of buffer) {
    const t = entry.t;
    const n = Math.hypot(t[0], t[1], t[2]);
    if (n < TRANS_DIR_MIN_M) continue;
    dirs.push([t[0] / n, t[1] / n, t[2] / n]);
  }
  if (dirs.length < MIN_TRANS_DIRS) return 0;
  const mean = [0, 0, 0];
  for (const d of dirs) {
    mean[0] += d[0];
    mean[1] += d[1];
    mean[2] += d[2];
  }
  mean[0] /= dirs.length;
  mean[1] /= dirs.length;
  mean[2] /= dirs.length;
  let acc = 0;
  for (const d of dirs) {
    acc +=
      (d[0] - mean[0]) ** 2 + (d[1] - mean[1]) ** 2 + (d[2] - mean[2]) ** 2;
  }
  return Math.sqrt(acc / dirs.length);
}

/**
 * True when the head-relative rotations span >= 2 distinct axes
 * (>= ROT_AXIS_DISTINCT_RAD apart, with the second cluster carrying >= 0.5%
 * of the rotation energy) and total excitation >= ROT_TOTAL_MIN_RAD.
 */
function rotationDiverse(buffer) {
  const axes = []; // [ax, ay, az, angle]
  let total = 0;
  for (const entry of buffer) {
    const v = rotVecFromRotMat(entry.R);
    const ang = Math.hypot(v[0], v[1], v[2]);
    if (ang < ROT_AXIS_MIN_RAD) continue;
    axes.push([v[0] / ang, v[1] / ang, v[2] / ang, ang]);
    total += ang;
  }
  if (axes.length < MIN_ROT_AXES || total < ROT_TOTAL_MIN_RAD) return false;
  // Two axes must be at least ROT_AXIS_DISTINCT_RAD apart.
  let maxAngle = 0;
  for (let i = 0; i < axes.length; i++) {
    for (let j = i + 1; j < axes.length; j++) {
      const dot = Math.abs(
        axes[i][0] * axes[j][0] +
          axes[i][1] * axes[j][1] +
          axes[i][2] * axes[j][2]
      );
      const ang = Math.acos(Math.min(1, dot));
      if (ang > maxAngle) maxAngle = ang;
    }
  }
  if (maxAngle < ROT_AXIS_DISTINCT_RAD) return false;
  // Magnitude-weighted scatter of the axes: the second eigenvalue must carry
  // a real share of the energy (a pure single-axis rotation has rank 1).
  const S = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (const a of axes) {
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) S[i][j] += a[3] * a[i] * a[j];
    }
  }
  const eig = symmetricEigenvalues3(S);
  return eig[1] / Math.max(eig[0], 1e-300) >= ROT_EIGEN_RATIO_MIN;
}

/** Eigenvalues of a symmetric 3x3, descending (analytic form). */
function symmetricEigenvalues3(S) {
  const p1 = S[0][1] ** 2 + S[0][2] ** 2 + S[1][2] ** 2;
  if (p1 < 1e-300) {
    return [
      Math.max(S[0][0], S[1][1], S[2][2]),
      Math.min(
        Math.max(S[0][0], S[1][1]),
        Math.max(S[1][1], S[2][2]),
        Math.max(S[0][0], S[2][2])
      ),
      Math.min(S[0][0], S[1][1], S[2][2]),
    ];
  }
  const q = (S[0][0] + S[1][1] + S[2][2]) / 3;
  const p2 =
    (S[0][0] - q) ** 2 + (S[1][1] - q) ** 2 + (S[2][2] - q) ** 2 + 2 * p1;
  const p = Math.sqrt(p2 / 6);
  const B = S.map((row, i) => row.map((v, j) => (v - (i === j ? q : 0)) / p));
  const detB =
    B[0][0] * (B[1][1] * B[2][2] - B[1][2] * B[2][1]) -
    B[0][1] * (B[1][0] * B[2][2] - B[1][2] * B[2][0]) +
    B[0][2] * (B[1][0] * B[2][1] - B[1][1] * B[2][0]);
  let r = detB / 2;
  r = Math.min(1, Math.max(-1, r));
  const phi = Math.acos(r) / 3;
  const eig1 = q + 2 * p * Math.cos(phi);
  const eig3 = q + 2 * p * Math.cos(phi + (2 * Math.PI) / 3);
  const eig2 = 3 * q - eig1 - eig3;
  return [eig1, eig2, eig3].sort((a, b) => b - a);
}

// ---- the calibrator --------------------------------------------------------

/**
 * Online camera self-calibration: 10 params
 * `[fx, fy, cx, cy, rx, ry, rz, tx, ty, tz]` where `(rx, ry, rz)` is the
 * rotation vector of `T_head_camera` (camera pose in the head frame) and
 * `(tx, ty, tz)` its lever arm.
 */
export class OnlineCalibrator {
  /**
   * @param {{width: number, height: number, priorK: number[][],
   *   priorTHeadCamera: number[][]}} opts priorK: 3x3 nested,
   *   priorTHeadCamera: 4x4 nested (INITIALIZATION ONLY)
   */
  constructor({width, height, priorK, priorTHeadCamera}) {
    if (!(width > 0) || !(height > 0)) {
      throw new Error('OnlineCalibrator: width/height must be positive');
    }
    this.width = width;
    this.height = height;
    this.priorK = priorK ? priorK.map((r) => r.slice()) : null;
    this.priorT = priorTHeadCamera
      ? priorTHeadCamera.map((r) => r.slice())
      : null;
    this.buffer = []; // ring buffer of frame pairs
    this._steps = 0;
    this._rmsEpipolar = null;
    this.p = this._clamp(this._paramsFromPriors());
  }

  /** Parameter vector from the current priors (defaults if none supplied). */
  _paramsFromPriors() {
    const K = this.priorK;
    const T = this.priorT;
    const fx = K ? K[0][0] : this.width * 0.9;
    const fy = K ? K[1][1] : this.width * 0.9;
    const cx = K ? K[0][2] : this.width / 2;
    const cy = K ? K[1][2] : this.height / 2;
    let rv = [0, 0, 0];
    let t = [0, 0, 0];
    if (T) {
      rv = rotVecFromRotMat(rot3(T));
      t = [T[0][3], T[1][3], T[2][3]];
    }
    return [fx, fy, cx, cy, rv[0], rv[1], rv[2], t[0], t[1], t[2]];
  }

  /** Clamp fx, fy to (0.2w, 5w) and cx, cy to (-w, 2w) after each update. */
  _clamp(p) {
    const w = this.width;
    p[0] = Math.min(5 * w, Math.max(0.2 * w, p[0]));
    p[1] = Math.min(5 * w, Math.max(0.2 * w, p[1]));
    p[2] = Math.min(2 * w, Math.max(-w, p[2]));
    p[3] = Math.min(2 * w, Math.max(-w, p[3]));
    // Physical plausibility bound on the lever arm (documented deviation from
    // the spec's clamp list, which covers only the intrinsics): a head-mounted
    // camera sits within LEVER_MAX_M of the head, so the lever-arm components
    // are clamped to +-LEVER_MAX_M. This keeps degenerate/weak-motion data
    // from running away along the near-null directions of the normal system
    // (observed live: multi-meter lever arms fitting outlier noise).
    p[7] = Math.min(LEVER_MAX_M, Math.max(-LEVER_MAX_M, p[7]));
    p[8] = Math.min(LEVER_MAX_M, Math.max(-LEVER_MAX_M, p[8]));
    p[9] = Math.min(LEVER_MAX_M, Math.max(-LEVER_MAX_M, p[9]));
    return p;
  }

  /**
   * Store one consecutive-frame pair into the ring buffer (cap MAX_PAIRS,
   * drop oldest). Matches are stride-subsampled to MAX_MATCHES_PER_PAIR.
   *
   * @param {{T_ref_head_i: number[][], T_ref_head_j: number[][],
   *   pts_i: {x: number, y: number}[], pts_j: {x: number, y: number}[],
   *   matches: {queryIdx: number, trainIdx: number}[]}} obs pts_i/pts_j:
   *   keypoints; matches: queryIdx indexes pts_i (frame i), trainIdx pts_j
   */
  addPair({T_ref_head_i, T_ref_head_j, pts_i, pts_j, matches}) {
    if (
      !T_ref_head_i ||
      !T_ref_head_j ||
      !pts_i ||
      !pts_j ||
      !matches ||
      !matches.length
    ) {
      return;
    }
    // Pair transform in the i -> j direction (A maps head_i coords to
    // head_j coords): with X = T_head_camera, X^-1 A X maps cam_i coords to
    // cam_j coords, so p_j = R_ij p_i + t_ij and the spec's residual
    // x_jᵀ E x_i with E = [t_ij]x R_ij is the vanishing constraint.
    const rel = relativeTransform(T_ref_head_j, T_ref_head_i);
    const stride = Math.max(
      1,
      Math.ceil(matches.length / MAX_MATCHES_PER_PAIR)
    );
    const obs = [];
    for (let k = 0; k < matches.length; k += stride) {
      const m = matches[k];
      const a = pts_i[m.queryIdx];
      const b = pts_j[m.trainIdx];
      if (!a || !b) continue;
      if (
        !Number.isFinite(a.x) ||
        !Number.isFinite(a.y) ||
        !Number.isFinite(b.x) ||
        !Number.isFinite(b.y)
      ) {
        continue;
      }
      obs.push(a.x, a.y, b.x, b.y);
    }
    if (obs.length < MIN_MATCHES_PER_PAIR * 4) return;
    this.buffer.push({R: rel.R, t: rel.t, obs: Float64Array.from(obs)});
    while (this.buffer.length > MAX_PAIRS) this.buffer.shift();
  }

  /**
   * Epipolar matrix `E = [t_ij]_× R_ij` for one buffered pair at parameters
   * `p`, from the i -> j pair transform `X^-1 · A · X` with `X = [R t]`:
   * `R_ij = Rᵀ R_hh R`, `t_ij = Rᵀ (R_hh t + t_hh − t)`, so
   * `p_j = R_ij · p_i + t_ij`.
   */
  _entryE(p, entry) {
    const R = rotMatFromRotVec([p[4], p[5], p[6]]);
    const t = [p[7], p[8], p[9]];
    const Rt = transpose3(R);
    const Rij = mul3(Rt, mul3(entry.R, R));
    const v = mul3v(entry.R, t);
    const tij = mul3v(Rt, [
      v[0] + entry.t[0] - t[0],
      v[1] + entry.t[1] - t[1],
      v[2] + entry.t[2] - t[2],
    ]);
    return skewTimes(tij, Rij);
  }

  /**
   * Build the FIXED observation layout used for the whole step(): the
   * (entryIdx, obsOffset) pairs whose Sampson denominator is non-degenerate
   * at the base parameters. Central differences then cannot change the residual
   * count mid-solve.
   */
  _buildLayout(p) {
    const cx = p[2];
    const cy = p[3];
    const fx = p[0];
    const fy = p[1];
    const keep = [];
    for (let e = 0; e < this.buffer.length; e++) {
      const entry = this.buffer[e];
      const E = this._entryE(p, entry);
      const obs = entry.obs;
      for (let k = 0; k < obs.length; k += 4) {
        const xi0 = (obs[k] - cx) / fx;
        const xi1 = -(obs[k + 1] - cy) / fy;
        const g0 = E[0][0] * xi0 + E[0][1] * xi1 - E[0][2];
        const g1 = E[1][0] * xi0 + E[1][1] * xi1 - E[1][2];
        const den = Math.hypot(g0, g1);
        if (Number.isFinite(den) && den >= DEN_EPS) keep.push(e, k);
      }
    }
    return Int32Array.from(keep);
  }

  /**
   * Evaluate the Sampson-normalized epipolar residuals for parameter vector
   * `p` over the fixed layout into `out`. Returns the number of residuals
   * written (always `layout.length / 2`).
   */
  _evalResiduals(p, layout, out) {
    const fx = p[0];
    const fy = p[1];
    const cx = p[2];
    const cy = p[3];
    const buffer = this.buffer;
    let m = 0;
    let lastE = -1;
    let E = null;
    for (let i = 0; i < layout.length; i += 2) {
      const e = layout[i];
      const k = layout[i + 1];
      if (e !== lastE) {
        E = this._entryE(p, buffer[e]);
        lastE = e;
      }
      const obs = buffer[e].obs;
      // Bearings in three-convention (z = -1 = in front; the spec's CV z > 0
      // guard maps to z < 0 here and is vacuous for a pinhole K, kept as-is).
      const xi0 = (obs[k] - cx) / fx;
      const xi1 = -(obs[k + 1] - cy) / fy;
      const xj0 = (obs[k + 2] - cx) / fx;
      const xj1 = -(obs[k + 3] - cy) / fy;
      // Sampson gradient terms: g = E · x_i (gradient w.r.t. x_j coords).
      const g0 = E[0][0] * xi0 + E[0][1] * xi1 - E[0][2];
      const g1 = E[1][0] * xi0 + E[1][1] * xi1 - E[1][2];
      const g2 = E[2][0] * xi0 + E[2][1] * xi1 - E[2][2];
      const f = g0 * xj0 + g1 * xj1 - g2;
      const den = Math.max(Math.hypot(g0, g1), DEN_EPS);
      out[m++] = f / den;
    }
    return m;
  }

  /**
   * Run damped Gauss-Newton (Levenberg) on the buffered observations.
   * Up to MAX_LM_ITERATIONS iterations; stops on a small parameter delta or
   * a residual plateau. Updates gated by the observability metrics: K only
   * when the rotations span >= 2 distinct axes, the lever arm only when the
   * translation directions are diverse (else it stays at its prior).
   *
   * @returns {{iterations: number, rmsEpipolar: number|null,
   *   improved: boolean}}
   */
  step() {
    // Fixed observation layout at the base parameters: pairs with degenerate
    // Sampson denominators are dropped here; perturbed evaluations clamp.
    let layout = this._buildLayout(this.p);
    let m = layout.length / 2;
    if (m < MIN_OBS) {
      return {iterations: 0, rmsEpipolar: null, improved: false};
    }
    let res = new Float64Array(m);
    this._evalResiduals(this.p, layout, res);
    // Robust observation gate: the apps feed RAW descriptor matches, whose
    // outlier fraction would drag a plain least-squares solve off the true
    // minimum (verified live). Trim observations further than 3.5 robust
    // sigmas (MAD) from the median residual BEFORE the solve; the Gauss-
    // Newton/Levenberg math on the kept set is exactly the spec's. The layout
    // stays fixed for the whole solve.
    const trimmed = robustTrim(layout, res);
    if (trimmed.length !== layout.length) {
      layout = trimmed;
      m = layout.length / 2;
      if (m < MIN_OBS) {
        return {iterations: 0, rmsEpipolar: null, improved: false};
      }
      res = new Float64Array(m);
      this._evalResiduals(this.p, layout, res);
    }
    let cost = 0;
    for (let i = 0; i < m; i++) cost += res[i] * res[i];
    const costStart = cost;

    const rotDiverse = rotationDiverse(this.buffer);
    const transDiv = translationDiversity(this.buffer);
    const transDiverse =
      transDiv >= TRANS_DIVERSE_MIN && this.buffer.length >= MIN_TRANS_DIRS;
    // Freeze mask per parameter: K needs >= 2 distinct rotation axes, the
    // lever arm needs diverse translation directions. The extrinsics rotation
    // is always free (it is what rotation-only motion constrains).
    const freeze = [
      !rotDiverse,
      !rotDiverse,
      !rotDiverse,
      !rotDiverse,
      false,
      false,
      false,
      !transDiverse,
      !transDiverse,
      !transDiverse,
    ];

    const p = this.p.slice();
    const jac = new Float64Array(m * 10);
    const tmpP = new Float64Array(m);
    const tmpM = new Float64Array(m);
    const resTrial = new Float64Array(m);
    let lambda = LAMBDA0;
    let iterations = 0;
    let improved = false;

    for (let it = 0; it < MAX_LM_ITERATIONS; it++) {
      iterations = it + 1;
      // Numeric Jacobian (central differences).
      for (let k = 0; k < 10; k++) {
        if (freeze[k]) continue;
        const h = FD_STEP * Math.max(1, Math.abs(p[k]));
        const pp = p.slice();
        pp[k] = p[k] + h;
        const pm = p.slice();
        pm[k] = p[k] - h;
        this._evalResiduals(pp, layout, tmpP);
        this._evalResiduals(pm, layout, tmpM);
        for (let row = 0; row < m; row++) {
          jac[row * 10 + k] = (tmpP[row] - tmpM[row]) / (2 * h);
        }
      }
      // Normal equations A = JᵀJ, b = -Jᵀr.
      const A = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      ];
      const b = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
      for (let row = 0; row < m; row++) {
        const r = res[row];
        for (let i = 0; i < 10; i++) {
          const ji = jac[row * 10 + i];
          if (ji === 0) continue;
          b[i] -= ji * r;
          for (let j = i; j < 10; j++) A[i][j] += ji * jac[row * 10 + j];
        }
      }
      for (let i = 0; i < 10; i++) {
        for (let j = 0; j < i; j++) A[i][j] = A[j][i];
      }
      // Frozen parameters: identity row/col, zero rhs -> zero delta.
      // Additionally freeze parameters whose RELATIVE curvature is negligible:
      // with degenerate motion (e.g. no roll about the optical axis) some
      // intrinsics directions are unconstrained, and letting them move only
      // fits noise — they ride the clamps to nonsense values. Relative scaling
      // ((1+|p|)^2) puts all ten parameters on comparable footing.
      let relDiagMax = 0;
      for (let k = 0; k < 10; k++) {
        if (freeze[k]) continue;
        const s = 1 + Math.abs(p[k]);
        const rel = A[k][k] * s * s;
        if (rel > relDiagMax) relDiagMax = rel;
      }
      for (let k = 0; k < 10; k++) {
        const unconstrained =
          !freeze[k] &&
          relDiagMax > 0 &&
          A[k][k] * (1 + Math.abs(p[k])) ** 2 < relDiagMax * 1e-6;
        if (freeze[k] || unconstrained) {
          for (let j = 0; j < 10; j++) {
            A[k][j] = 0;
            A[j][k] = 0;
          }
          A[k][k] = 1;
          b[k] = 0;
        }
      }
      // Marquardt damping.
      const Ad = A.map((row, i) =>
        row.map((v, j) => (i === j ? v + lambda * Math.max(v, 1e-12) : v))
      );
      const delta = solveLinear(Ad, b);
      if (!delta) {
        lambda *= 10;
        if (lambda > LAMBDA_MAX) break;
        continue;
      }
      // Trust region on the per-iteration step: even constrained directions
      // may not jump far in one iteration (keeps the solve in the basin).
      const TRUST = [
        0.02 * (Math.abs(p[0]) || 10),
        0.02 * (Math.abs(p[1]) || 10),
        3,
        3,
        0.005,
        0.005,
        0.005,
        0.01,
        0.01,
        0.01,
      ];
      const pTrial = p.slice();
      for (let k = 0; k < 10; k++) {
        pTrial[k] += Math.max(-TRUST[k], Math.min(TRUST[k], delta[k]));
      }
      this._clamp(pTrial);
      this._evalResiduals(pTrial, layout, resTrial);
      let costTrial = 0;
      for (let i = 0; i < m; i++) costTrial += resTrial[i] * resTrial[i];
      if (costTrial < cost) {
        const relGain = (cost - costTrial) / Math.max(cost, 1e-300);
        let dmax = 0;
        for (let k = 0; k < 10; k++) {
          if (freeze[k]) continue;
          const d = Math.abs(delta[k]) / (1 + Math.abs(p[k]));
          if (d > dmax) dmax = d;
        }
        for (let k = 0; k < 10; k++) p[k] = pTrial[k];
        res.set(resTrial);
        cost = costTrial;
        improved = true;
        lambda = Math.max(lambda / 3, LAMBDA_MIN);
        if (dmax < SMALL_DELTA || relGain < PLATEAU) break;
      } else {
        lambda *= 10;
        if (lambda > LAMBDA_MAX) break;
      }
    }

    for (let k = 0; k < 10; k++) this.p[k] = p[k];
    this._steps += iterations;
    this._rmsEpipolar = Math.sqrt(cost / m);
    return {
      iterations,
      rmsEpipolar: this._rmsEpipolar,
      improved: cost < costStart,
    };
  }

  /** Current intrinsics estimate: 3x3 nested `[[fx,0,cx],[0,fy,cy],[0,0,1]]`. */
  get K() {
    return [
      [this.p[0], 0, this.p[2]],
      [0, this.p[1], this.p[3]],
      [0, 0, 1],
    ];
  }

  /** Current extrinsics estimate: `T_head_camera` (4x4 nested). */
  get T_head_camera() {
    const R = rotMatFromRotVec([this.p[4], this.p[5], this.p[6]]);
    return [
      [R[0][0], R[0][1], R[0][2], this.p[7]],
      [R[1][0], R[1][1], R[1][2], this.p[8]],
      [R[2][0], R[2][1], R[2][2], this.p[9]],
      [0, 0, 0, 1],
    ];
  }

  /**
   * @returns {{pairs: number, rmsEpipolar: number|null, constrained: boolean,
   *   leverArmConstrained: boolean, steps: number}}
   */
  get quality() {
    const rotDiverse = rotationDiverse(this.buffer);
    const transDiverse =
      translationDiversity(this.buffer) >= TRANS_DIVERSE_MIN &&
      this.buffer.length >= MIN_TRANS_DIRS;
    return {
      pairs: this.buffer.length,
      rmsEpipolar: this._rmsEpipolar,
      constrained:
        rotDiverse &&
        this.buffer.length >= MIN_PAIRS_FOR_CONSTRAINT &&
        this._rmsEpipolar !== null,
      leverArmConstrained:
        transDiverse &&
        this.buffer.length >= MIN_PAIRS_FOR_CONSTRAINT &&
        this._rmsEpipolar !== null,
      steps: this._steps,
    };
  }

  /**
   * Clear the buffer and re-initialize from priors (arguments replace the
   * stored priors; omitted arguments keep the previously stored ones).
   */
  reset(priorK, priorTHeadCamera) {
    if (priorK) this.priorK = priorK.map((r) => r.slice());
    if (priorTHeadCamera) {
      this.priorT = priorTHeadCamera.map((r) => r.slice());
    }
    this.buffer = [];
    this._steps = 0;
    this._rmsEpipolar = null;
    this.p = this._clamp(this._paramsFromPriors());
  }
}

/**
 * Convenience accessor for the current intrinsics estimate.
 * @param {OnlineCalibrator} calibrator
 * @returns {{fx: number, fy: number, cx: number, cy: number}}
 */
export function KFromCalibrator(calibrator) {
  const K = calibrator.K;
  return {fx: K[0][0], fy: K[1][1], cx: K[0][2], cy: K[1][2]};
}
