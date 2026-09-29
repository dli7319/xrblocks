/**
 * Object-Centric KinectFusion (TSDF + RGB Volumetric Fusion & Marching Cubes)
 * for XR Circle-to-Digitize.
 *
 * Pure TypeScript / TypedArray implementation with zero DOM or Three.js runtime
 * dependencies so it can execute inside a Web Worker at 72+ FPS.
 */

export interface Vec3Tuple {
  x: number;
  y: number;
  z: number;
}

export interface SupportPlane3D {
  nx: number;
  ny: number;
  nz: number;
  /** Plane equation: nx*x + ny*y + nz*z + d = 0 (signed distance > 0 on object side). */
  d: number;
}

export interface BoundingBox2D {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  cx: number;
  cy: number;
}

export interface VolumeSeedParams {
  /** Raw depth buffer (`Float32Array` or `Uint16Array`) of size `depthWidth * depthHeight`. */
  depthData: Float32Array | Uint16Array;
  depthWidth: number;
  depthHeight: number;
  rawValueToMeters: number;
  /** 4x4 column-major matrix: World -> Depth Camera View Space. */
  depthViewMatrix: Float32Array;
  /** 4x4 column-major matrix: Depth Camera Clip Space -> Depth Camera View Space. */
  depthProjectionInverseMatrix: Float32Array;
  /** Optional 4x4 column-major matrix: Normalized View Coords -> Normalized Depth Buffer Coords. */
  normDepthBufferFromNormViewMatrix?: Float32Array;
  /** 4x4 column-major matrix: World -> RGB Camera Clip Space. */
  rgbClipFromWorldMatrix: Float32Array;
  /** Binary foreground mask (`1` = foreground, `0` = background) in RGB camera space. */
  cameraBinaryMask: Uint8Array;
  maskWidth: number;
  maskHeight: number;
  /** Minimum allowed volume size in meters (default 0.16). */
  minSizeMeters?: number;
  /** Maximum allowed volume size in meters (default 0.75). */
  maxSizeMeters?: number;
  /** Padding multiplier around detected 3D extent (default 1.35). */
  paddingFactor?: number;
}

export interface VolumeSeedResult {
  center: Vec3Tuple;
  sizeMeters: number;
  boundsMin: Vec3Tuple;
  boundsMax: Vec3Tuple;
  supportFloorY: number;
  supportPlane: SupportPlane3D | null;
  sampleCount: number;
  medianDepthMeters: number;
}

export interface TSDFIntegrationFrame {
  /** Raw depth buffer (`Float32Array` or `Uint16Array`) of size `depthWidth * depthHeight`. */
  depthData: Float32Array | Uint16Array;
  depthWidth: number;
  depthHeight: number;
  rawValueToMeters: number;
  /** 4x4 column-major matrix: World -> Depth Camera View Space. */
  depthViewMatrix: Float32Array;
  /** 4x4 column-major matrix: Depth Camera View Space -> Depth Camera Clip Space. */
  depthProjectionMatrix: Float32Array;
  /** Optional 4x4 column-major matrix: Normalized View Coords -> Normalized Depth Buffer Coords. */
  normDepthBufferFromNormViewMatrix?: Float32Array;
  /** Optional 4x4 column-major matrix: World -> RGB Camera Clip Space. */
  rgbClipFromWorldMatrix?: Float32Array;
  /** Optional RGB(A) image buffer of size `rgbWidth * rgbHeight * 4`. */
  rgbaData?: Uint8ClampedArray | Uint8Array;
  rgbWidth?: number;
  rgbHeight?: number;
  /** Optional binary mask (`1` = foreground, `0` = background) in RGB camera space. */
  cameraBinaryMask?: Uint8Array;
  maskWidth?: number;
  maskHeight?: number;
  /** Whether to carve voxels projecting outside `cameraBinaryMask` (default true when mask provided). */
  carveOutsideMask?: boolean;
  /** Maximum allowed depth discontinuity between adjacent depth pixels in meters (default 0.045). */
  maxDepthEdgeDeltaMeters?: number;
}

export interface ExtractedTSDFMesh {
  positions: Float32Array;
  normals: Float32Array;
  colors: Float32Array;
  triangleCount: number;
}

/**
 * Inverts a 4x4 column-major matrix into `out`. Returns `true` if invertible.
 */
export function invertMatrix4ColMajor(
  m: Float32Array,
  out: Float32Array = new Float32Array(16)
): boolean {
  const n11 = m[0],
    n21 = m[1],
    n31 = m[2],
    n41 = m[3];
  const n12 = m[4],
    n22 = m[5],
    n32 = m[6],
    n42 = m[7];
  const n13 = m[8],
    n23 = m[9],
    n33 = m[10],
    n43 = m[11];
  const n14 = m[12],
    n24 = m[13],
    n34 = m[14],
    n44 = m[15];

  const t11 =
    n23 * n34 * n42 -
    n24 * n33 * n42 +
    n24 * n32 * n43 -
    n22 * n34 * n43 -
    n23 * n32 * n44 +
    n22 * n33 * n44;
  const t12 =
    n14 * n33 * n42 -
    n13 * n34 * n42 -
    n14 * n32 * n43 +
    n12 * n34 * n43 +
    n13 * n32 * n44 -
    n12 * n33 * n44;
  const t13 =
    n13 * n24 * n42 -
    n14 * n23 * n42 +
    n14 * n22 * n43 -
    n12 * n24 * n43 -
    n13 * n22 * n44 +
    n12 * n23 * n44;
  const t14 =
    n14 * n23 * n32 -
    n13 * n24 * n32 -
    n14 * n22 * n33 +
    n12 * n24 * n33 +
    n13 * n22 * n34 -
    n12 * n23 * n34;

  const det = n11 * t11 + n21 * t12 + n31 * t13 + n41 * t14;
  if (Math.abs(det) < 1e-12) {
    return false;
  }
  const detInv = 1.0 / det;

  out[0] = t11 * detInv;
  out[1] =
    (n24 * n33 * n41 -
      n23 * n34 * n41 -
      n24 * n31 * n43 +
      n21 * n34 * n43 +
      n23 * n31 * n44 -
      n21 * n33 * n44) *
    detInv;
  out[2] =
    (n22 * n34 * n41 -
      n24 * n32 * n41 +
      n24 * n31 * n42 -
      n21 * n34 * n42 -
      n22 * n31 * n44 +
      n21 * n32 * n44) *
    detInv;
  out[3] =
    (n23 * n32 * n41 -
      n22 * n33 * n41 -
      n23 * n31 * n42 +
      n21 * n33 * n42 +
      n22 * n31 * n43 -
      n21 * n32 * n43) *
    detInv;

  out[4] = t12 * detInv;
  out[5] =
    (n13 * n34 * n41 -
      n14 * n33 * n41 +
      n14 * n31 * n43 -
      n11 * n34 * n43 -
      n13 * n31 * n44 +
      n11 * n33 * n44) *
    detInv;
  out[6] =
    (n14 * n32 * n41 -
      n12 * n34 * n41 -
      n14 * n31 * n42 +
      n11 * n34 * n42 +
      n12 * n31 * n44 -
      n11 * n32 * n44) *
    detInv;
  out[7] =
    (n12 * n33 * n41 -
      n13 * n32 * n41 +
      n13 * n31 * n42 -
      n11 * n33 * n42 -
      n12 * n31 * n43 +
      n11 * n32 * n43) *
    detInv;

  out[8] = t13 * detInv;
  out[9] =
    (n14 * n23 * n41 -
      n13 * n24 * n41 -
      n14 * n21 * n43 +
      n11 * n24 * n43 +
      n13 * n21 * n44 -
      n11 * n23 * n44) *
    detInv;
  out[10] =
    (n12 * n24 * n41 -
      n14 * n22 * n41 +
      n14 * n21 * n42 -
      n11 * n24 * n42 -
      n12 * n21 * n44 +
      n11 * n22 * n44) *
    detInv;
  out[11] =
    (n13 * n22 * n41 -
      n12 * n23 * n41 -
      n13 * n21 * n42 +
      n11 * n23 * n42 +
      n12 * n21 * n43 -
      n11 * n22 * n43) *
    detInv;

  out[12] = t14 * detInv;
  out[13] =
    (n13 * n24 * n31 -
      n14 * n23 * n31 +
      n14 * n21 * n33 -
      n11 * n24 * n33 -
      n13 * n21 * n34 +
      n11 * n23 * n34) *
    detInv;
  out[14] =
    (n14 * n22 * n31 -
      n12 * n24 * n31 -
      n14 * n21 * n32 +
      n11 * n24 * n32 +
      n12 * n21 * n34 -
      n11 * n22 * n34) *
    detInv;
  out[15] =
    (n12 * n23 * n31 -
      n13 * n22 * n31 +
      n13 * n21 * n32 -
      n11 * n23 * n32 -
      n12 * n21 * n33 +
      n11 * n22 * n33) *
    detInv;

  return true;
}

/**
 * Erodes a 2D binary mask by `radius` pixels to prevent silhouette depth bleed.
 */
export function erodeBinaryMask(
  mask: Uint8Array,
  width: number,
  height: number,
  radius = 2
): Uint8Array {
  if (radius <= 0) {
    return new Uint8Array(mask);
  }
  const out = new Uint8Array(width * height);
  for (let y = radius; y < height - radius; y++) {
    const rowOffset = y * width;
    for (let x = radius; x < width - radius; x++) {
      if (!mask[rowOffset + x]) continue;
      let allFg = true;
      for (let dy = -radius; dy <= radius && allFg; dy++) {
        const rOff = (y + dy) * width + x;
        for (let dx = -radius; dx <= radius; dx++) {
          if (!mask[rOff + dx]) {
            allFg = false;
            break;
          }
        }
      }
      if (allFg) {
        out[rowOffset + x] = 1;
      }
    }
  }
  return out;
}

/**
 * Dilates a 2D binary mask by `radius` pixels using separable horizontal + vertical passes.
 */
export function dilateBinaryMask(
  mask: Uint8Array,
  width: number,
  height: number,
  radius = 4
): Uint8Array {
  if (radius <= 0) {
    return new Uint8Array(mask);
  }
  const horiz = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) {
    const row = y * width;
    for (let x = 0; x < width; x++) {
      if (!mask[row + x]) continue;
      const x0 = Math.max(0, x - radius);
      const x1 = Math.min(width - 1, x + radius);
      for (let k = x0; k <= x1; k++) {
        horiz[row + k] = 1;
      }
    }
  }

  const out = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) {
    const row = y * width;
    for (let x = 0; x < width; x++) {
      if (!horiz[row + x]) continue;
      const y0 = Math.max(0, y - radius);
      const y1 = Math.min(height - 1, y + radius);
      for (let k = y0; k <= y1; k++) {
        out[k * width + x] = 1;
      }
    }
  }
  return out;
}

/**
 * Pre-filters a depth map in meters, zeroing out pixels at steep depth discontinuities
 * (flying/veil pixels at object boundaries).
 */
export function filterDepthDiscontinuities(
  depthData: Float32Array | Uint16Array,
  width: number,
  height: number,
  rawValueToMeters: number,
  maxDeltaMeters = 0.045,
  minDepthMeters = 0.08,
  maxDepthMeters = 4.5
): Float32Array {
  const n = width * height;
  const meters = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const d = depthData[i] * rawValueToMeters;
    meters[i] =
      Number.isFinite(d) && d >= minDepthMeters && d <= maxDepthMeters ? d : 0;
  }

  const filtered = new Float32Array(meters);
  for (let y = 1; y < height - 1; y++) {
    const row = y * width;
    for (let x = 1; x < width - 1; x++) {
      const idx = row + x;
      const c = meters[idx];
      if (c <= 0) continue;

      const l = meters[idx - 1];
      const r = meters[idx + 1];
      const u = meters[idx - width];
      const d = meters[idx + width];

      if (
        l <= 0 ||
        r <= 0 ||
        u <= 0 ||
        d <= 0 ||
        Math.abs(c - l) > maxDeltaMeters ||
        Math.abs(c - r) > maxDeltaMeters ||
        Math.abs(c - u) > maxDeltaMeters ||
        Math.abs(c - d) > maxDeltaMeters
      ) {
        filtered[idx] = 0;
      }
    }
  }
  return filtered;
}

/**
 * Fits a 3D supporting table/floor plane (`nx*x + ny*y + nz*z + d = 0`) using
 * deterministic 3-point RANSAC on the 3D points just outside the 2D SAM mask border.
 */
export function fitSupportPlaneRansac(
  borderPoints: Vec3Tuple[],
  objectPoints: Vec3Tuple[],
  cameraWorldPos: Vec3Tuple,
  inlierThresholdMeters = 0.012
): SupportPlane3D | null {
  if (borderPoints.length < 12 || objectPoints.length < 6) {
    return null;
  }

  let bestPlane: SupportPlane3D | null = null;
  let bestInlierCount = 0;
  const nPts = borderPoints.length;
  const iterations = Math.min(90, nPts * 2);

  for (let iter = 0; iter < iterations; iter++) {
    // Deterministic quasi-random triplet sampling
    const i0 = (iter * 7 + 1) % nPts;
    const i1 = (iter * 13 + 5 + Math.floor(nPts / 3)) % nPts;
    const i2 = (iter * 29 + 11 + Math.floor((2 * nPts) / 3)) % nPts;
    if (i0 === i1 || i1 === i2 || i0 === i2) continue;

    const p0 = borderPoints[i0];
    const p1 = borderPoints[i1];
    const p2 = borderPoints[i2];

    const ux = p1.x - p0.x;
    const uy = p1.y - p0.y;
    const uz = p1.z - p0.z;
    const vx = p2.x - p0.x;
    const vy = p2.y - p0.y;
    const vz = p2.z - p0.z;

    let nx = uy * vz - uz * vy;
    let ny = uz * vx - ux * vz;
    let nz = ux * vy - uy * vx;
    const len = Math.hypot(nx, ny, nz);
    if (len < 1e-5) continue;
    nx /= len;
    ny /= len;
    nz /= len;

    // Orient plane normal so it points toward the camera (or +Y upward)
    const toCamX = cameraWorldPos.x - p0.x;
    const toCamY = cameraWorldPos.y - p0.y;
    const toCamZ = cameraWorldPos.z - p0.z;
    if (nx * toCamX + ny * toCamY + nz * toCamZ < 0) {
      nx = -nx;
      ny = -ny;
      nz = -nz;
    }

    const d = -(nx * p0.x + ny * p0.y + nz * p0.z);

    let inliers = 0;
    for (let k = 0; k < nPts; k++) {
      const pt = borderPoints[k];
      const dist = Math.abs(nx * pt.x + ny * pt.y + nz * pt.z + d);
      if (dist <= inlierThresholdMeters) {
        inliers++;
      }
    }

    if (inliers > bestInlierCount) {
      bestInlierCount = inliers;
      bestPlane = {nx, ny, nz, d};
    }
  }

  if (!bestPlane || bestInlierCount < Math.max(8, nPts * 0.28)) {
    return null;
  }

  // Verify the fitted plane acts as a supporting surface beneath/behind the object
  // (at least 60% of the object points must lie on the positive side of the plane)
  let aboveCount = 0;
  for (const op of objectPoints) {
    const signedDist =
      bestPlane.nx * op.x +
      bestPlane.ny * op.y +
      bestPlane.nz * op.z +
      bestPlane.d;
    if (signedDist >= -0.006) {
      aboveCount++;
    }
  }
  if (aboveCount < objectPoints.length * 0.6) {
    return null;
  }

  return bestPlane;
}

/**
 * Seeds a 3D object bounding volume in world space by unprojecting WebXR depth samples
 * that project inside the 2D EfficientSAM segmentation mask, and fits + subtracts
 * the surrounding table support plane.
 */
export function seedVolumeFromMaskAndDepth(
  params: VolumeSeedParams
): VolumeSeedResult | null {
  const {
    depthData,
    depthWidth,
    depthHeight,
    rawValueToMeters,
    depthViewMatrix,
    depthProjectionInverseMatrix,
    normDepthBufferFromNormViewMatrix,
    rgbClipFromWorldMatrix,
    cameraBinaryMask,
    maskWidth,
    maskHeight,
    minSizeMeters = 0.16,
    maxSizeMeters = 0.75,
    paddingFactor = 1.35,
  } = params;

  const worldFromDepthView = new Float32Array(16);
  if (!invertMatrix4ColMajor(depthViewMatrix, worldFromDepthView)) {
    return null;
  }

  const cameraWorldPos: Vec3Tuple = {
    x: worldFromDepthView[12],
    y: worldFromDepthView[13],
    z: worldFromDepthView[14],
  };

  const normViewFromNormDepthBuffer = new Float32Array(16);
  let hasNormTransform = false;
  if (normDepthBufferFromNormViewMatrix) {
    hasNormTransform = invertMatrix4ColMajor(
      normDepthBufferFromNormViewMatrix,
      normViewFromNormDepthBuffer
    );
  }

  const cleanDepth = filterDepthDiscontinuities(
    depthData,
    depthWidth,
    depthHeight,
    rawValueToMeters
  );

  const erodedMask = erodeBinaryMask(
    cameraBinaryMask,
    maskWidth,
    maskHeight,
    2
  );
  const innerDilated = dilateBinaryMask(
    cameraBinaryMask,
    maskWidth,
    maskHeight,
    3
  );
  const outerDilated = dilateBinaryMask(
    cameraBinaryMask,
    maskWidth,
    maskHeight,
    20
  );

  interface Sample3D {
    wx: number;
    wy: number;
    wz: number;
    depth: number;
  }

  const fgSamplesEroded: Sample3D[] = [];
  const fgSamplesRaw: Sample3D[] = [];
  const borderSamples: Vec3Tuple[] = [];

  const pInv = depthProjectionInverseMatrix;
  const wFromV = worldFromDepthView;
  const cFromW = rgbClipFromWorldMatrix;

  for (let dy = 1; dy < depthHeight - 1; dy++) {
    const by = (dy + 0.5) / depthHeight;
    const rowOffset = dy * depthWidth;
    for (let dx = 1; dx < depthWidth - 1; dx++) {
      const dMeters = cleanDepth[rowOffset + dx];
      if (dMeters <= 0) continue;

      const bx = (dx + 0.5) / depthWidth;
      let viewU = bx;
      let viewTopV = by;
      if (hasNormTransform) {
        const m = normViewFromNormDepthBuffer;
        viewU = m[0] * bx + m[4] * by + m[12];
        viewTopV = m[1] * bx + m[5] * by + m[13];
      }

      const ndcX = 2.0 * viewU - 1.0;
      const ndcY = 1.0 - 2.0 * viewTopV;

      const rw = pInv[3] * ndcX + pInv[7] * ndcY - pInv[11] + pInv[15];
      if (Math.abs(rw) < 1e-8) continue;
      const rx = (pInv[0] * ndcX + pInv[4] * ndcY - pInv[8] + pInv[12]) / rw;
      const ry = (pInv[1] * ndcX + pInv[5] * ndcY - pInv[9] + pInv[13]) / rw;
      const rz = (pInv[2] * ndcX + pInv[6] * ndcY - pInv[10] + pInv[14]) / rw;
      if (rz >= -1e-5) continue;

      const scale = -dMeters / rz;
      const vx = rx * scale;
      const vy = ry * scale;
      const vz = -dMeters;

      const wx = wFromV[0] * vx + wFromV[4] * vy + wFromV[8] * vz + wFromV[12];
      const wy = wFromV[1] * vx + wFromV[5] * vy + wFromV[9] * vz + wFromV[13];
      const wz = wFromV[2] * vx + wFromV[6] * vy + wFromV[10] * vz + wFromV[14];

      const cw = cFromW[3] * wx + cFromW[7] * wy + cFromW[11] * wz + cFromW[15];
      if (cw <= 1e-5) continue;
      const cNdcX =
        (cFromW[0] * wx + cFromW[4] * wy + cFromW[8] * wz + cFromW[12]) / cw;
      const cNdcY =
        (cFromW[1] * wx + cFromW[5] * wy + cFromW[9] * wz + cFromW[13]) / cw;

      const mx = Math.floor((cNdcX + 1.0) * 0.5 * maskWidth);
      const my = Math.floor((1.0 - (cNdcY + 1.0) * 0.5) * maskHeight);
      if (mx < 0 || mx >= maskWidth || my < 0 || my >= maskHeight) continue;

      const mIdx = my * maskWidth + mx;
      if (erodedMask[mIdx]) {
        fgSamplesEroded.push({wx, wy, wz, depth: dMeters});
      }
      if (cameraBinaryMask[mIdx]) {
        fgSamplesRaw.push({wx, wy, wz, depth: dMeters});
      } else if (outerDilated[mIdx] && !innerDilated[mIdx]) {
        borderSamples.push({x: wx, y: wy, z: wz});
      }
    }
  }

  let samples = fgSamplesEroded.length >= 6 ? fgSamplesEroded : fgSamplesRaw;
  if (samples.length === 0) {
    return null;
  }

  // Fit supporting table plane from the annular border pixels surrounding the mask
  const supportPlane = fitSupportPlaneRansac(
    borderSamples,
    samples.map((s) => ({x: s.wx, y: s.wy, z: s.wz})),
    cameraWorldPos
  );

  // If a table plane was found, reject any sample that lies on or below the table plane (< 8 mm)
  if (supportPlane) {
    const abovePlane = samples.filter((s) => {
      const h =
        supportPlane.nx * s.wx +
        supportPlane.ny * s.wy +
        supportPlane.nz * s.wz +
        supportPlane.d;
      return h >= 0.008;
    });
    if (abovePlane.length >= 5) {
      samples = abovePlane;
    }
  }

  samples.sort((a, b) => a.depth - b.depth);
  const startIdx = Math.floor(samples.length * 0.12);
  const endIdx = Math.max(startIdx + 1, Math.ceil(samples.length * 0.88));
  const inliers = samples.slice(startIdx, endIdx);

  let minX = Infinity,
    minY = Infinity,
    minZ = Infinity;
  let maxX = -Infinity,
    maxY = -Infinity,
    maxZ = -Infinity;
  let sumX = 0,
    sumY = 0,
    sumZ = 0;

  for (const s of inliers) {
    if (s.wx < minX) minX = s.wx;
    if (s.wy < minY) minY = s.wy;
    if (s.wz < minZ) minZ = s.wz;
    if (s.wx > maxX) maxX = s.wx;
    if (s.wy > maxY) maxY = s.wy;
    if (s.wz > maxZ) maxZ = s.wz;
    sumX += s.wx;
    sumY += s.wy;
    sumZ += s.wz;
  }

  const count = inliers.length;
  const cx = sumX / count;
  const cy = sumY / count;
  const cz = sumZ / count;

  const spanX = maxX - minX;
  const spanY = maxY - minY;
  const spanZ = maxZ - minZ;
  const maxSpan = Math.max(spanX, spanY, spanZ, Math.max(spanX, spanY) * 0.85);
  const sizeMeters = Math.max(
    minSizeMeters,
    Math.min(maxSizeMeters, maxSpan * paddingFactor)
  );

  // Tight anisotropic 3D bounding box padded by 3.5 cm (and extended slightly away from camera for back-surface fusion)
  const padX = Math.max(0.03, spanX * 0.22);
  const padY = Math.max(0.03, spanY * 0.22);
  const padZ = Math.max(
    0.04,
    Math.max(spanZ, Math.max(spanX, spanY) * 0.45) * 0.35
  );

  const boundsMin: Vec3Tuple = {
    x: minX - padX,
    y: minY - padY * 0.5,
    z: minZ - padZ,
  };
  const boundsMax: Vec3Tuple = {
    x: maxX + padX,
    y: maxY + padY,
    z: maxZ + padZ,
  };

  const medianDepthMeters = inliers[Math.floor(inliers.length * 0.5)].depth;

  return {
    center: {x: cx, y: cy, z: cz},
    sizeMeters,
    boundsMin,
    boundsMax,
    supportFloorY: minY - 0.004,
    supportPlane,
    sampleCount: count,
    medianDepthMeters,
  };
}

/**
 * Object-Centric TSDF + RGB Voxel Grid with 8x8x8 macro-block acceleration,
 * RANSAC support-plane clipping, and Marching Cubes surface extraction.
 */
export class ObjectTSDFVolume {
  readonly resolution: number;
  readonly sizeMeters: number;
  readonly voxelSize: number;
  readonly truncationDist: number;
  readonly center: Vec3Tuple;
  readonly origin: Vec3Tuple;
  readonly supportFloorY: number;
  readonly boundsMin: Vec3Tuple;
  readonly boundsMax: Vec3Tuple;
  supportPlane: SupportPlane3D | null;

  readonly tsdf: Float32Array;
  readonly weight: Float32Array;
  readonly colorR: Uint8Array;
  readonly colorG: Uint8Array;
  readonly colorB: Uint8Array;

  readonly blockSize = 8;
  readonly blocksPerAxis: number;
  readonly blockMaxWeight: Float32Array;

  fusedFrameCount = 0;
  maxWeight = 32.0;

  constructor(
    center: Vec3Tuple,
    sizeMeters: number,
    resolution = 64,
    supportFloorY = -Infinity,
    supportPlane: SupportPlane3D | null = null,
    boundsMin?: Vec3Tuple,
    boundsMax?: Vec3Tuple
  ) {
    this.resolution = resolution;
    this.sizeMeters = sizeMeters;
    this.voxelSize = sizeMeters / resolution;
    this.truncationDist = Math.max(0.015, this.voxelSize * 3.0);
    this.center = {...center};
    const half = sizeMeters * 0.5;
    this.origin = {
      x: center.x - half,
      y: center.y - half,
      z: center.z - half,
    };
    this.supportFloorY = supportFloorY;
    this.supportPlane = supportPlane;
    this.boundsMin = boundsMin
      ? {...boundsMin}
      : {x: this.origin.x, y: this.origin.y, z: this.origin.z};
    this.boundsMax = boundsMax
      ? {...boundsMax}
      : {
          x: center.x + half,
          y: center.y + half,
          z: center.z + half,
        };

    const totalVoxels = resolution * resolution * resolution;
    this.tsdf = new Float32Array(totalVoxels).fill(1.0);
    this.weight = new Float32Array(totalVoxels);
    this.colorR = new Uint8Array(totalVoxels).fill(240);
    this.colorG = new Uint8Array(totalVoxels).fill(240);
    this.colorB = new Uint8Array(totalVoxels).fill(240);

    this.blocksPerAxis = Math.ceil(resolution / this.blockSize);
    const totalBlocks =
      this.blocksPerAxis * this.blocksPerAxis * this.blocksPerAxis;
    this.blockMaxWeight = new Float32Array(totalBlocks);
  }

  reset(): void {
    this.tsdf.fill(1.0);
    this.weight.fill(0.0);
    this.colorR.fill(240);
    this.colorG.fill(240);
    this.colorB.fill(240);
    this.blockMaxWeight.fill(0.0);
    this.fusedFrameCount = 0;
  }

  /**
   * Projects the tight 3D object bounds into the 2D RGB camera frame
   * to automatically generate a bounding box + center prompt for EfficientSAM.
   */
  projectBoundingBoxToCamera(
    rgbClipFromWorldMatrix: Float32Array,
    imgWidth: number,
    imgHeight: number
  ): BoundingBox2D | null {
    const m = rgbClipFromWorldMatrix;
    const cx = this.center.x;
    const cy = this.center.y;
    const cz = this.center.z;

    const xs = [this.boundsMin.x, this.boundsMax.x];
    const ys = [this.boundsMin.y, this.boundsMax.y];
    const zs = [this.boundsMin.z, this.boundsMax.z];

    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    let validCorners = 0;

    for (const wz of zs) {
      for (const wy of ys) {
        for (const wx of xs) {
          const cw = m[3] * wx + m[7] * wy + m[11] * wz + m[15];
          if (cw <= 1e-4) continue;
          const ndcX = (m[0] * wx + m[4] * wy + m[8] * wz + m[12]) / cw;
          const ndcY = (m[1] * wx + m[5] * wy + m[9] * wz + m[13]) / cw;
          const px = (ndcX + 1.0) * 0.5 * imgWidth;
          const py = (1.0 - (ndcY + 1.0) * 0.5) * imgHeight;
          if (px < minX) minX = px;
          if (py < minY) minY = py;
          if (px > maxX) maxX = px;
          if (py > maxY) maxY = py;
          validCorners++;
        }
      }
    }

    if (validCorners < 4) return null;

    const cCw = m[3] * cx + m[7] * cy + m[11] * cz + m[15];
    if (cCw <= 1e-4) return null;
    const cNdcX = (m[0] * cx + m[4] * cy + m[8] * cz + m[12]) / cCw;
    const cNdcY = (m[1] * cx + m[5] * cy + m[9] * cz + m[13]) / cCw;
    const centerPx = (cNdcX + 1.0) * 0.5 * imgWidth;
    const centerPy = (1.0 - (cNdcY + 1.0) * 0.5) * imgHeight;

    const x1 = Math.max(0, Math.min(imgWidth - 1, minX));
    const y1 = Math.max(0, Math.min(imgHeight - 1, minY));
    const x2 = Math.max(0, Math.min(imgWidth - 1, maxX));
    const y2 = Math.max(0, Math.min(imgHeight - 1, maxY));

    if (x2 - x1 < 8 || y2 - y1 < 8) return null;

    return {
      x1,
      y1,
      x2,
      y2,
      cx: Math.max(x1, Math.min(x2, centerPx)),
      cy: Math.max(y1, Math.min(y2, centerPy)),
    };
  }

  /**
   * Integrates a single WebXR depth frame (+ RGB image and 2D SAM mask)
   * into the TSDF volume. Rejects table support-plane voxels and carves
   * background silhouettes.
   */
  integrateFrame(frame: TSDFIntegrationFrame): number {
    const {
      depthData,
      depthWidth,
      depthHeight,
      rawValueToMeters,
      depthViewMatrix,
      depthProjectionMatrix,
      normDepthBufferFromNormViewMatrix,
      rgbClipFromWorldMatrix,
      rgbaData,
      rgbWidth = 0,
      rgbHeight = 0,
      cameraBinaryMask,
      maskWidth = 0,
      maskHeight = 0,
      carveOutsideMask = true,
      maxDepthEdgeDeltaMeters = 0.045,
    } = frame;

    const cleanDepth = filterDepthDiscontinuities(
      depthData,
      depthWidth,
      depthHeight,
      rawValueToMeters,
      maxDepthEdgeDeltaMeters
    );

    const hasMask = Boolean(
      cameraBinaryMask &&
        rgbClipFromWorldMatrix &&
        maskWidth > 0 &&
        maskHeight > 0
    );
    const erodedFgMask =
      hasMask && cameraBinaryMask
        ? erodeBinaryMask(cameraBinaryMask, maskWidth, maskHeight, 2)
        : null;

    const N = this.resolution;
    const N2 = N * N;
    const vs = this.voxelSize;
    const halfVs = vs * 0.5;
    const ox = this.origin.x + halfVs;
    const oy = this.origin.y + halfVs;
    const oz = this.origin.z + halfVs;
    const trunc = this.truncationDist;
    const invTrunc = 1.0 / trunc;
    const maxW = this.maxWeight;
    const floorY = this.supportFloorY;
    const plane = this.supportPlane;
    const bMin = this.boundsMin;
    const bMax = this.boundsMax;

    const V = depthViewMatrix;
    const P = depthProjectionMatrix;
    const normM = normDepthBufferFromNormViewMatrix;
    const hasRgbClip = Boolean(rgbClipFromWorldMatrix);
    const hasColors = Boolean(
      rgbaData && rgbClipFromWorldMatrix && rgbWidth > 0 && rgbHeight > 0
    );

    const tsdfArr = this.tsdf;
    const weightArr = this.weight;
    const rArr = this.colorR;
    const gArr = this.colorG;
    const bArr = this.colorB;
    const bpa = this.blocksPerAxis;
    const bSize = this.blockSize;
    const blockMaxW = this.blockMaxWeight;

    let updatedCount = 0;

    for (let iz = 0; iz < N; iz++) {
      const wz = oz + iz * vs;
      if (wz < bMin.z || wz > bMax.z) continue;

      const bz = (iz / bSize) | 0;
      const zOffset = iz * N2;

      const vzBaseZ = V[8] * wz + V[12];
      const vyBaseZ = V[9] * wz + V[13];
      const vxBaseZ = V[10] * wz + V[14];

      for (let iy = 0; iy < N; iy++) {
        const wy = oy + iy * vs;
        if (wy < floorY || wy < bMin.y || wy > bMax.y) continue;

        const by = (iy / bSize) | 0;
        const yzOffset = zOffset + iy * N;

        const vxBaseYZ = V[4] * wy + vzBaseZ;
        const vyBaseYZ = V[5] * wy + vyBaseZ;
        const vzBaseYZ = V[6] * wy + vxBaseZ;

        for (let ix = 0; ix < N; ix++) {
          const wx = ox + ix * vs;
          if (wx < bMin.x || wx > bMax.x) continue;

          const idx = yzOffset + ix;

          // Reject any voxel on or below the fitted supporting table plane (< 8 mm)
          if (plane) {
            const planeDist =
              plane.nx * wx + plane.ny * wy + plane.nz * wz + plane.d;
            if (planeDist < 0.008) {
              if (weightArr[idx] > 0) {
                weightArr[idx] = 0;
                tsdfArr[idx] = 1.0;
              }
              continue;
            }
          }

          // 1. World -> Depth Camera View Space
          const vx = V[0] * wx + vxBaseYZ;
          const vy = V[1] * wx + vyBaseYZ;
          const vz = V[2] * wx + vzBaseYZ;
          const zCam = -vz;
          if (zCam <= 0.08) continue;

          // 2. Depth Camera View Space -> Clip Space
          const clipW = P[3] * vx + P[7] * vy + P[11] * vz + P[15];
          if (clipW <= 1e-5) continue;
          const invClipW = 1.0 / clipW;
          const clipX = (P[0] * vx + P[4] * vy + P[8] * vz + P[12]) * invClipW;
          const clipY = (P[1] * vx + P[5] * vy + P[9] * vz + P[13]) * invClipW;
          if (clipX < -1.0 || clipX > 1.0 || clipY < -1.0 || clipY > 1.0) {
            continue;
          }

          const u = 0.5 * (clipX + 1.0);
          const v = 0.5 * (clipY + 1.0);
          let normBx = u;
          let normBy = 1.0 - v;
          if (normM) {
            const topV = 1.0 - v;
            normBx = normM[0] * u + normM[4] * topV + normM[12];
            normBy = normM[1] * u + normM[5] * topV + normM[13];
          }

          const dx = Math.round(normBx * (depthWidth - 1));
          const dy = Math.round(normBy * (depthHeight - 1));
          if (dx < 0 || dx >= depthWidth || dy < 0 || dy >= depthHeight) {
            continue;
          }

          const dMeasured = cleanDepth[dy * depthWidth + dx];

          // 3. Projection into RGB / Mask camera space
          let rgbU = u;
          let rgbTopV = 1.0 - v;
          if (hasRgbClip && rgbClipFromWorldMatrix) {
            const C = rgbClipFromWorldMatrix;
            const cw = C[3] * wx + C[7] * wy + C[11] * wz + C[15];
            if (cw <= 1e-5) continue;
            const cNdcX = (C[0] * wx + C[4] * wy + C[8] * wz + C[12]) / cw;
            const cNdcY = (C[1] * wx + C[5] * wy + C[9] * wz + C[13]) / cw;
            rgbU = 0.5 * (cNdcX + 1.0);
            rgbTopV = 1.0 - 0.5 * (cNdcY + 1.0);
          }

          if (hasMask && cameraBinaryMask) {
            const mx = Math.floor(rgbU * maskWidth);
            const my = Math.floor(rgbTopV * maskHeight);
            const inMaskBounds =
              mx >= 0 && mx < maskWidth && my >= 0 && my < maskHeight;
            const mIdx = inMaskBounds ? my * maskWidth + mx : -1;
            const isRawFg = mIdx >= 0 && cameraBinaryMask[mIdx] === 1;
            const isErodedFg =
              mIdx >= 0 && (erodedFgMask ? erodedFgMask[mIdx] === 1 : isRawFg);

            if (!isRawFg) {
              // Visual-hull silhouette carving: if voxel projects outside the 2D object mask
              // and is not occluded by a closer foreground surface, decay its weight.
              if (
                carveOutsideMask &&
                weightArr[idx] > 0 &&
                (dMeasured <= 0 || dMeasured > zCam - trunc * 0.25)
              ) {
                const newW = Math.max(0.0, weightArr[idx] - 1.0);
                weightArr[idx] = newW;
                if (newW === 0.0) {
                  tsdfArr[idx] = 1.0;
                }
              }
              continue;
            }

            // Only integrate positive surface depth inside the eroded mask interior
            if (!isErodedFg) {
              continue;
            }
          }

          if (dMeasured <= 0) continue;

          // 4. Signed distance evaluation
          const eta = dMeasured - zCam;
          if (eta < -trunc) continue;

          const sdf = Math.min(1.0, eta * invTrunc);
          const wObs = 1.0;
          const oldW = weightArr[idx];
          const oldF = tsdfArr[idx];
          const newW = Math.min(maxW, oldW + wObs);
          const newF = (oldW * oldF + wObs * sdf) / (oldW + wObs);

          tsdfArr[idx] = newF;
          weightArr[idx] = newW;
          updatedCount++;

          const bx = (ix / bSize) | 0;
          const bIdx = (bz * bpa + by) * bpa + bx;
          if (newW > blockMaxW[bIdx]) {
            blockMaxW[bIdx] = newW;
          }

          // 5. Fallback vertex color integration near surface
          if (hasColors && rgbaData && Math.abs(eta) <= trunc) {
            const cx = Math.floor(rgbU * rgbWidth);
            const cy = Math.floor(rgbTopV * rgbHeight);
            if (cx >= 0 && cx < rgbWidth && cy >= 0 && cy < rgbHeight) {
              const p = (cy * rgbWidth + cx) * 4;
              const obsR = rgbaData[p];
              const obsG = rgbaData[p + 1];
              const obsB = rgbaData[p + 2];
              if (oldW === 0) {
                rArr[idx] = obsR;
                gArr[idx] = obsG;
                bArr[idx] = obsB;
              } else {
                const alpha = wObs / (oldW + wObs);
                rArr[idx] = Math.round(rArr[idx] * (1 - alpha) + obsR * alpha);
                gArr[idx] = Math.round(gArr[idx] * (1 - alpha) + obsG * alpha);
                bArr[idx] = Math.round(bArr[idx] * (1 - alpha) + obsB * alpha);
              }
            }
          }
        }
      }
    }

    if (updatedCount > 0) {
      this.fusedFrameCount++;
    }
    return updatedCount;
  }

  /**
   * Computes the outward surface normal at integer grid coordinates `(x, y, z)`
   * using central differences on the TSDF scalar field.
   */
  private sampleGradientNormal(
    gx: number,
    gy: number,
    gz: number,
    out: Float32Array,
    offset: number
  ): void {
    const N = this.resolution;
    const N2 = N * N;
    const tsdf = this.tsdf;

    const ix = Math.max(1, Math.min(N - 2, Math.round(gx)));
    const iy = Math.max(1, Math.min(N - 2, Math.round(gy)));
    const iz = Math.max(1, Math.min(N - 2, Math.round(gz)));
    const base = iz * N2 + iy * N + ix;

    const dx = tsdf[base + 1] - tsdf[base - 1];
    const dy = tsdf[base + N] - tsdf[base - N];
    const dz = tsdf[base + N2] - tsdf[base - N2];
    const len = Math.hypot(dx, dy, dz);
    if (len > 1e-6) {
      out[offset] = dx / len;
      out[offset + 1] = dy / len;
      out[offset + 2] = dz / len;
    } else {
      out[offset] = 0;
      out[offset + 1] = 1;
      out[offset + 2] = 0;
    }
  }

  /**
   * Extracts the zero-crossing ($F(\mathbf{p}) = 0$) isosurface via block-accelerated
   * Marching Cubes, returning vertex positions, TSDF-gradient normals, and RGB vertex colors.
   */
  extractMesh(minWeight = 1.0): ExtractedTSDFMesh {
    const N = this.resolution;
    const N2 = N * N;
    const vs = this.voxelSize;
    const halfVs = vs * 0.5;
    const ox = this.origin.x + halfVs;
    const oy = this.origin.y + halfVs;
    const oz = this.origin.z + halfVs;

    const tsdf = this.tsdf;
    const weight = this.weight;
    const rArr = this.colorR;
    const gArr = this.colorG;
    const bArr = this.colorB;
    const bpa = this.blocksPerAxis;
    const bSize = this.blockSize;
    const blockMaxW = this.blockMaxWeight;

    const posList: number[] = [];
    const normList: number[] = [];
    const colList: number[] = [];

    const edgeVertX = new Float32Array(12);
    const edgeVertY = new Float32Array(12);
    const edgeVertZ = new Float32Array(12);
    const edgeGridX = new Float32Array(12);
    const edgeGridY = new Float32Array(12);
    const edgeGridZ = new Float32Array(12);
    const edgeColR = new Float32Array(12);
    const edgeColG = new Float32Array(12);
    const edgeColB = new Float32Array(12);
    const cornerF = new Float32Array(8);
    const cornerIdx = new Int32Array(8);
    const tempNorm = new Float32Array(9);

    const inv255 = 1.0 / 255.0;

    for (let bz = 0; bz < bpa; bz++) {
      const zStart = bz * bSize;
      const zEnd = Math.min(N - 1, zStart + bSize);

      for (let by = 0; by < bpa; by++) {
        const yStart = by * bSize;
        const yEnd = Math.min(N - 1, yStart + bSize);

        for (let bx = 0; bx < bpa; bx++) {
          const bIdx = (bz * bpa + by) * bpa + bx;
          if (blockMaxW[bIdx] < minWeight) continue;

          const xStart = bx * bSize;
          const xEnd = Math.min(N - 1, xStart + bSize);

          for (let iz = zStart; iz < zEnd; iz++) {
            const z0 = iz * N2;
            const z1 = (iz + 1) * N2;

            for (let iy = yStart; iy < yEnd; iy++) {
              const y0 = iy * N;
              const y1 = (iy + 1) * N;

              for (let ix = xStart; ix < xEnd; ix++) {
                const i0 = z0 + y0 + ix;
                const i1 = i0 + 1;
                const i2 = z0 + y1 + ix + 1;
                const i3 = z0 + y1 + ix;
                const i4 = z1 + y0 + ix;
                const i5 = i4 + 1;
                const i6 = z1 + y1 + ix + 1;
                const i7 = z1 + y1 + ix;

                if (
                  weight[i0] < minWeight ||
                  weight[i1] < minWeight ||
                  weight[i2] < minWeight ||
                  weight[i3] < minWeight ||
                  weight[i4] < minWeight ||
                  weight[i5] < minWeight ||
                  weight[i6] < minWeight ||
                  weight[i7] < minWeight
                ) {
                  continue;
                }

                const f0 = tsdf[i0];
                const f1 = tsdf[i1];
                const f2 = tsdf[i2];
                const f3 = tsdf[i3];
                const f4 = tsdf[i4];
                const f5 = tsdf[i5];
                const f6 = tsdf[i6];
                const f7 = tsdf[i7];

                let cubeIndex = 0;
                if (f0 < 0) cubeIndex |= 1;
                if (f1 < 0) cubeIndex |= 2;
                if (f2 < 0) cubeIndex |= 4;
                if (f3 < 0) cubeIndex |= 8;
                if (f4 < 0) cubeIndex |= 16;
                if (f5 < 0) cubeIndex |= 32;
                if (f6 < 0) cubeIndex |= 64;
                if (f7 < 0) cubeIndex |= 128;

                const edgeMask = MC_EDGE_TABLE[cubeIndex];
                if (edgeMask === 0) continue;

                cornerF[0] = f0;
                cornerF[1] = f1;
                cornerF[2] = f2;
                cornerF[3] = f3;
                cornerF[4] = f4;
                cornerF[5] = f5;
                cornerF[6] = f6;
                cornerF[7] = f7;

                cornerIdx[0] = i0;
                cornerIdx[1] = i1;
                cornerIdx[2] = i2;
                cornerIdx[3] = i3;
                cornerIdx[4] = i4;
                cornerIdx[5] = i5;
                cornerIdx[6] = i6;
                cornerIdx[7] = i7;

                for (let e = 0; e < 12; e++) {
                  if ((edgeMask & (1 << e)) === 0) continue;
                  const cA = MC_EDGE_CORNERS[e * 2];
                  const cB = MC_EDGE_CORNERS[e * 2 + 1];
                  const fa = cornerF[cA];
                  const fb = cornerF[cB];
                  const denom = fa - fb;
                  const t =
                    Math.abs(denom) > 1e-6
                      ? Math.max(0.0, Math.min(1.0, fa / denom))
                      : 0.5;
                  const oneMinusT = 1.0 - t;

                  const gx =
                    ix +
                    MC_CORNER_OFFSET_X[cA] * oneMinusT +
                    MC_CORNER_OFFSET_X[cB] * t;
                  const gy =
                    iy +
                    MC_CORNER_OFFSET_Y[cA] * oneMinusT +
                    MC_CORNER_OFFSET_Y[cB] * t;
                  const gz =
                    iz +
                    MC_CORNER_OFFSET_Z[cA] * oneMinusT +
                    MC_CORNER_OFFSET_Z[cB] * t;

                  edgeGridX[e] = gx;
                  edgeGridY[e] = gy;
                  edgeGridZ[e] = gz;
                  edgeVertX[e] = ox + gx * vs;
                  edgeVertY[e] = oy + gy * vs;
                  edgeVertZ[e] = oz + gz * vs;

                  const idxA = cornerIdx[cA];
                  const idxB = cornerIdx[cB];
                  edgeColR[e] =
                    (rArr[idxA] * oneMinusT + rArr[idxB] * t) * inv255;
                  edgeColG[e] =
                    (gArr[idxA] * oneMinusT + gArr[idxB] * t) * inv255;
                  edgeColB[e] =
                    (bArr[idxA] * oneMinusT + bArr[idxB] * t) * inv255;
                }

                const triRowOffset = cubeIndex * 16;
                for (let t = 0; t < 15; t += 3) {
                  const e0 = MC_TRI_TABLE[triRowOffset + t];
                  if (e0 === -1) break;
                  const e1 = MC_TRI_TABLE[triRowOffset + t + 1];
                  const e2 = MC_TRI_TABLE[triRowOffset + t + 2];

                  this.sampleGradientNormal(
                    edgeGridX[e0],
                    edgeGridY[e0],
                    edgeGridZ[e0],
                    tempNorm,
                    0
                  );
                  this.sampleGradientNormal(
                    edgeGridX[e1],
                    edgeGridY[e1],
                    edgeGridZ[e1],
                    tempNorm,
                    3
                  );
                  this.sampleGradientNormal(
                    edgeGridX[e2],
                    edgeGridY[e2],
                    edgeGridZ[e2],
                    tempNorm,
                    6
                  );

                  // Ensure CCW triangle winding matches the outward TSDF gradient normal
                  const ax = edgeVertX[e1] - edgeVertX[e0];
                  const ay = edgeVertY[e1] - edgeVertY[e0];
                  const az = edgeVertZ[e1] - edgeVertZ[e0];
                  const bxVec = edgeVertX[e2] - edgeVertX[e0];
                  const byVec = edgeVertY[e2] - edgeVertY[e0];
                  const bzVec = edgeVertZ[e2] - edgeVertZ[e0];
                  const cx = ay * bzVec - az * byVec;
                  const cy = az * bxVec - ax * bzVec;
                  const cz = ax * byVec - ay * bxVec;

                  const dot =
                    cx * tempNorm[0] + cy * tempNorm[1] + cz * tempNorm[2];

                  if (dot >= 0) {
                    posList.push(
                      edgeVertX[e0],
                      edgeVertY[e0],
                      edgeVertZ[e0],
                      edgeVertX[e1],
                      edgeVertY[e1],
                      edgeVertZ[e1],
                      edgeVertX[e2],
                      edgeVertY[e2],
                      edgeVertZ[e2]
                    );
                    normList.push(
                      tempNorm[0],
                      tempNorm[1],
                      tempNorm[2],
                      tempNorm[3],
                      tempNorm[4],
                      tempNorm[5],
                      tempNorm[6],
                      tempNorm[7],
                      tempNorm[8]
                    );
                    colList.push(
                      edgeColR[e0],
                      edgeColG[e0],
                      edgeColB[e0],
                      edgeColR[e1],
                      edgeColG[e1],
                      edgeColB[e1],
                      edgeColR[e2],
                      edgeColG[e2],
                      edgeColB[e2]
                    );
                  } else {
                    posList.push(
                      edgeVertX[e0],
                      edgeVertY[e0],
                      edgeVertZ[e0],
                      edgeVertX[e2],
                      edgeVertY[e2],
                      edgeVertZ[e2],
                      edgeVertX[e1],
                      edgeVertY[e1],
                      edgeVertZ[e1]
                    );
                    normList.push(
                      tempNorm[0],
                      tempNorm[1],
                      tempNorm[2],
                      tempNorm[6],
                      tempNorm[7],
                      tempNorm[8],
                      tempNorm[3],
                      tempNorm[4],
                      tempNorm[5]
                    );
                    colList.push(
                      edgeColR[e0],
                      edgeColG[e0],
                      edgeColB[e0],
                      edgeColR[e2],
                      edgeColG[e2],
                      edgeColB[e2],
                      edgeColR[e1],
                      edgeColG[e1],
                      edgeColB[e1]
                    );
                  }
                }
              }
            }
          }
        }
      }
    }

    return {
      positions: new Float32Array(posList),
      normals: new Float32Array(normList),
      colors: new Float32Array(colList),
      triangleCount: posList.length / 9,
    };
  }
}

const MC_CORNER_OFFSET_X = [0, 1, 1, 0, 0, 1, 1, 0];
const MC_CORNER_OFFSET_Y = [0, 0, 1, 1, 0, 0, 1, 1];
const MC_CORNER_OFFSET_Z = [0, 0, 0, 0, 1, 1, 1, 1];

const MC_EDGE_CORNERS = [
  0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7,
];

const MC_EDGE_TABLE = new Uint16Array([
  0x0, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c, 0x80c, 0x905, 0xa0f,
  0xb06, 0xc0a, 0xd03, 0xe09, 0xf00, 0x190, 0x99, 0x393, 0x29a, 0x596, 0x49f,
  0x795, 0x69c, 0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90, 0x230,
  0x339, 0x33, 0x13a, 0x636, 0x73f, 0x435, 0x53c, 0xa3c, 0xb35, 0x83f, 0x936,
  0xe3a, 0xf33, 0xc39, 0xd30, 0x3a0, 0x2a9, 0x1a3, 0xaa, 0x7a6, 0x6af, 0x5a5,
  0x4ac, 0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0, 0x460, 0x569,
  0x663, 0x76a, 0x66, 0x16f, 0x265, 0x36c, 0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a,
  0x963, 0xa69, 0xb60, 0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff, 0x3f5, 0x2fc,
  0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0, 0x650, 0x759, 0x453,
  0x55a, 0x256, 0x35f, 0x55, 0x15c, 0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53,
  0x859, 0x950, 0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc, 0xfcc,
  0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0, 0x8c0, 0x9c9, 0xac3, 0xbca,
  0xcc6, 0xdcf, 0xec5, 0xfcc, 0xcc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9,
  0x7c0, 0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c, 0x15c, 0x55,
  0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650, 0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6,
  0xfff, 0xcf5, 0xdfc, 0x2fc, 0x3f5, 0xff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
  0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c, 0x36c, 0x265, 0x16f,
  0x66, 0x76a, 0x663, 0x569, 0x460, 0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af,
  0xaa5, 0xbac, 0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa, 0x1a3, 0x2a9, 0x3a0, 0xd30,
  0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c, 0x53c, 0x435, 0x73f, 0x636,
  0x13a, 0x33, 0x339, 0x230, 0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895,
  0x99c, 0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99, 0x190, 0xf00, 0xe09,
  0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c, 0x70c, 0x605, 0x50f, 0x406, 0x30a,
  0x203, 0x109, 0x0,
]);

const MC_TRI_TABLE = new Int8Array([
  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 8, 3, -1,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 9, -1, -1, -1, -1, -1,
  -1, -1, -1, -1, -1, -1, -1, -1, 1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1,
  -1, -1, -1, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,
  8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 9, 2, 10, 0, 2, 9, -1,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1,
  -1, -1, -1, -1, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
  0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 9, 0, 2, 3, 11,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1,
  -1, -1, -1, -1, -1, 3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1,
  -1, 0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1, 3, 9, 0, 3, 11,
  9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1, 9, 8, 10, 10, 8, 11, -1, -1, -1, -1,
  -1, -1, -1, -1, -1, -1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
  -1, -1, 4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 9, 8,
  4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 1, 9, 4, 7, 1, 7, 3, 1, -1,
  -1, -1, -1, -1, -1, -1, 1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1,
  -1, 3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, 9, 2, 10, 9, 0, 2,
  8, 4, 7, -1, -1, -1, -1, -1, -1, -1, 2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1,
  -1, -1, -1, 8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 11, 4,
  7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1, 9, 0, 1, 8, 4, 7, 2, 3, 11,
  -1, -1, -1, -1, -1, -1, -1, 4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1,
  -1, 3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1, 1, 11, 10, 1, 4,
  11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1, 4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3,
  -1, -1, -1, -1, 4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1, 9,
  5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 9, 5, 4, 0, 8, 3,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1,
  -1, -1, -1, -1, -1, 8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1, 1,
  2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, 0, 8, 1, 2, 10, 4,
  9, 5, -1, -1, -1, -1, -1, -1, -1, 5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1,
  -1, -1, -1, 2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1, 9, 5, 4, 2,
  3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 11, 2, 0, 8, 11, 4, 9, 5,
  -1, -1, -1, -1, -1, -1, -1, 0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1,
  -1, -1, 2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1, 10, 3, 11, 10, 1,
  3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, 4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10,
  -1, -1, -1, -1, 5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1, 5, 4,
  8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, 9, 7, 8, 5, 7, 9, -1, -1,
  -1, -1, -1, -1, -1, -1, -1, -1, 9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1,
  -1, -1, 0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1, 1, 5, 3, 3, 5,
  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1,
  -1, -1, -1, -1, -1, 10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1, 8, 0,
  2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1, 2, 10, 5, 2, 5, 3, 3, 5, 7, -1,
  -1, -1, -1, -1, -1, -1, 7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1,
  -1, 9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1, 2, 3, 11, 0, 1, 8, 1,
  7, 8, 1, 5, 7, -1, -1, -1, -1, 11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1,
  -1, -1, -1, 9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1, 5, 7, 0, 5,
  0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1, 11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0,
  7, 5, 7, 0, -1, 11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
  10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 8, 3, 5, 10,
  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 9, 0, 1, 5, 10, 6, -1, -1, -1, -1,
  -1, -1, -1, -1, -1, -1, 1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1,
  -1, 1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 6, 5, 1, 2,
  6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1, 9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1,
  -1, -1, -1, -1, 5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1, 2, 3, 11,
  10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 11, 0, 8, 11, 2, 0, 10, 6,
  5, -1, -1, -1, -1, -1, -1, -1, 0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1,
  -1, -1, -1, 5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1, 6, 3, 11,
  6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1, 0, 8, 11, 0, 11, 5, 0, 5, 1, 5,
  11, 6, -1, -1, -1, -1, 3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1, 6,
  5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1, 5, 10, 6, 4, 7, 8, -1,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1,
  -1, -1, -1, -1, 1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, 10, 6,
  5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1, 6, 1, 2, 6, 5, 1, 4, 7, 8, -1,
  -1, -1, -1, -1, -1, -1, 1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1, 8,
  4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1, 7, 3, 9, 7, 9, 4, 3, 2, 9, 5,
  9, 6, 2, 6, 9, -1, 3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, 5,
  10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1, 0, 1, 9, 4, 7, 8, 2, 3, 11,
  5, 10, 6, -1, -1, -1, -1, 9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1,
  8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1, 5, 1, 11, 5, 11, 6, 1,
  0, 11, 7, 11, 4, 0, 4, 11, -1, 0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7,
  -1, 6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1, 10, 4, 9, 6, 4, 10,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1,
  -1, -1, -1, -1, -1, 10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1,
  8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1, 1, 4, 9, 1, 2, 4, 2, 6,
  4, -1, -1, -1, -1, -1, -1, -1, 3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1,
  -1, 0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8, 3, 2, 8, 2,
  4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, 10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1,
  -1, -1, -1, -1, -1, 0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1, 3,
  11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1, 6, 4, 1, 6, 1, 10, 4, 8, 1,
  2, 1, 11, 8, 11, 1, -1, 9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1,
  8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1, 3, 11, 6, 3, 6, 0, 0, 6, 4,
  -1, -1, -1, -1, -1, -1, -1, 6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1,
  -1, -1, 7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1, 0, 7, 3, 0,
  10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1, 10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8,
  0, -1, -1, -1, -1, 10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1, 1,
  2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1, 2, 6, 9, 2, 9, 1, 6, 7, 9, 0,
  9, 3, 7, 3, 9, -1, 7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1, 7,
  3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 3, 11, 10, 6, 8, 10,
  8, 9, 8, 6, 7, -1, -1, -1, -1, 2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7,
  -1, 1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1, 11, 2, 1, 11, 1, 7,
  10, 6, 1, 6, 7, 1, -1, -1, -1, -1, 8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3,
  6, -1, 0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 7, 8, 0, 7,
  0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1, 7, 11, 6, -1, -1, -1, -1, -1, -1,
  -1, -1, -1, -1, -1, -1, -1, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
  -1, -1, -1, 3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1,
  9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8, 1, 9, 8, 3, 1, 11, 7,
  6, -1, -1, -1, -1, -1, -1, -1, 10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1,
  -1, -1, -1, 1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, 2, 9, 0,
  2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, 6, 11, 7, 2, 10, 3, 10, 8, 3,
  10, 9, 8, -1, -1, -1, -1, 7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1,
  -1, -1, 7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1, 2, 7, 6, 2, 3,
  7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1, 1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6,
  -1, -1, -1, -1, 10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1, 10,
  7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1, 0, 3, 7, 0, 7, 10, 0, 10, 9,
  6, 10, 7, -1, -1, -1, -1, 7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1,
  -1, -1, 6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, 6, 11,
  3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1, 8, 6, 11, 8, 4, 6, 9, 0, 1, -1,
  -1, -1, -1, -1, -1, -1, 9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1,
  6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1, 1, 2, 10, 3, 0, 11,
  0, 6, 11, 0, 4, 6, -1, -1, -1, -1, 4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1,
  -1, -1, -1, 10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1, 8, 2, 3, 8, 4,
  2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, 0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1,
  -1, -1, -1, -1, -1, 1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1, 1, 9,
  4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1, 8, 1, 3, 8, 6, 1, 8, 4, 6, 6,
  10, 1, -1, -1, -1, -1, 10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1,
  -1, 4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1, 10, 9, 4, 6, 10, 4, -1,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1,
  -1, -1, -1, -1, 0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, 5, 0,
  1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, 11, 7, 6, 8, 3, 4, 3, 5, 4,
  3, 1, 5, -1, -1, -1, -1, 9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1,
  -1, 6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1, 7, 6, 11, 5, 4, 10,
  4, 2, 10, 4, 0, 2, -1, -1, -1, -1, 3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7,
  6, -1, 7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1, 9, 5, 4, 0, 8,
  6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1, 3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1,
  -1, -1, -1, 6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1, 9, 5, 4, 10, 1,
  6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1, 1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9,
  5, 4, -1, 4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1, 7, 6, 10, 7,
  10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1, 6, 9, 5, 6, 11, 9, 11, 8, 9, -1,
  -1, -1, -1, -1, -1, -1, 3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1,
  0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1, 6, 11, 3, 6, 3, 5, 5,
  3, 1, -1, -1, -1, -1, -1, -1, -1, 1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1,
  -1, -1, -1, 0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1, 11, 8, 5, 11,
  5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1, 6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3,
  -1, -1, -1, -1, 5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1, 9, 5, 6,
  9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1, 1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8,
  2, 6, 2, 8, -1, 1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
  3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1, 10, 1, 0, 10, 0, 6, 9, 5, 0, 5,
  6, 0, -1, -1, -1, -1, 0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1,
  -1, 10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 11, 5, 10,
  7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 11, 5, 10, 11, 7, 5, 8, 3,
  0, -1, -1, -1, -1, -1, -1, -1, 5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1,
  -1, -1, -1, 10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1, 11, 1, 2,
  11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1, 0, 8, 3, 1, 2, 7, 1, 7, 5, 7,
  2, 11, -1, -1, -1, -1, 9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1, 7,
  5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1, 2, 5, 10, 2, 3, 5, 3, 7, 5, -1,
  -1, -1, -1, -1, -1, -1, 8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1,
  9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1, 9, 8, 2, 9, 2, 1, 8, 7,
  2, 10, 2, 5, 7, 5, 2, -1, 1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1,
  -1, -1, 0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1, 9, 0, 3, 9, 3,
  5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1, 9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1,
  -1, -1, -1, -1, -1, 5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1,
  5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1, 0, 1, 9, 8, 4, 10, 8,
  10, 11, 10, 4, 5, -1, -1, -1, -1, 10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3,
  1, 4, -1, 2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1, 0, 4, 11, 0,
  11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1, 0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8,
  11, 8, 5, -1, 9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 5,
  10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1, 5, 10, 2, 5, 2, 4, 4, 2, 0, -1,
  -1, -1, -1, -1, -1, -1, 3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1, 5,
  10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1, 8, 4, 5, 8, 5, 3, 3, 5, 1,
  -1, -1, -1, -1, -1, -1, -1, 0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1,
  -1, -1, 8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1, 9, 4, 5, -1, -1,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 11, 7, 4, 9, 11, 9, 10, 11, -1,
  -1, -1, -1, -1, -1, -1, 0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1,
  1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1, 3, 1, 4, 3, 4, 8, 1,
  10, 4, 7, 4, 11, 10, 11, 4, -1, 4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1,
  -1, -1, 9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1, 11, 7, 4, 11, 4,
  2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1, 11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4,
  -1, -1, -1, -1, 2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1, 9, 10, 7,
  9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1, 3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10,
  0, 4, 0, 10, -1, 1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4,
  9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1, 4, 9, 1, 4, 1, 7, 0, 8, 1,
  8, 7, 1, -1, -1, -1, -1, 4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1,
  -1, 4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 9, 10, 8, 10,
  11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, 0, 9, 3, 9, 11, 11, 9, 10,
  -1, -1, -1, -1, -1, -1, -1, 0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1,
  -1, -1, 3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 2, 11,
  1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1, 3, 0, 9, 3, 9, 11, 1, 2, 9, 2,
  11, 9, -1, -1, -1, -1, 0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1,
  -1, 3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 3, 8, 2,
  8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1, 9, 10, 2, 0, 9, 2, -1, -1, -1,
  -1, -1, -1, -1, -1, -1, -1, 2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1,
  -1, 1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 3, 8, 9,
  1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 9, 1, -1, -1, -1, -1, -1, -1,
  -1, -1, -1, -1, -1, -1, -1, 0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
]);
