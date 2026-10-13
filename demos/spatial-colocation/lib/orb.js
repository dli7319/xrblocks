/**
 * ORB feature extraction and camera intrinsics helpers.
 *
 * All functions take the OpenCV namespace (`cv`) as an argument so the module
 * stays environment agnostic (Node tests use `@techstark/opencv-js`, the
 * browser uses the CDN global).
 */

/** Max ORB features per frame. */
export const MAX_FEATURES = 1000;
/** A/B toggle: illumination normalization inside extractOrb. */
export const NORMALIZE_ILLUMINATION = true;
/** Descriptor length in bytes (ORB = 256 bits). */
export const DESC_LEN = 32;
/** Default vertical-less field of view used to guess intrinsics (degrees). */
export const DEFAULT_FOV_DEG = 60;

/** Shared constants (see IMPLEMENTATION_SPEC "Conventions"). */
export const MIN_PARALLAX_DEG = 1.0;
export const KEYFRAME_MIN_MS = 500;
export const KEYFRAME_MIN_MOVE_M = 0.05;
export const KEYFRAME_MIN_ROT_DEG = 3;
export const PRESENCE_INTERVAL_MS = 100;

/**
 * Convert an RGBA (or RGB / already gray) raster into a single channel
 * `cv.Mat (CV_8UC1)`. Luma uses ITU-R BT.601 weights (0.299/0.587/0.114),
 * which matches `cv.COLOR_RGBA2GRAY` closely.
 *
 * @param {object} cv OpenCV namespace
 * @param {{data: Uint8ClampedArray|Uint8Array|Uint8Array, width: number, height: number}} image
 * @returns {object} cv.Mat (caller owns it and must `.delete()` it)
 */
export function toGrayMat(cv, image) {
  const width = image.width | 0;
  const height = image.height | 0;
  const data = image.data;
  if (!(width > 0) || !(height > 0)) {
    throw new Error(`toGrayMat: invalid image size ${width}x${height}`);
  }
  const pixels = width * height;
  if (!data || data.length < pixels) {
    throw new Error(
      `toGrayMat: image data length ${data ? data.length : 0} < width*height ${pixels}`
    );
  }

  const mat = new cv.Mat(height, width, cv.CV_8UC1);
  const dst = mat.data;
  if (data.length === pixels) {
    dst.set(data.subarray(0, pixels));
  } else if (data.length >= pixels * 4) {
    for (let i = 0; i < pixels; i++) {
      const j = i * 4;
      dst[i] =
        (299 * data[j] + 587 * data[j + 1] + 114 * data[j + 2] + 500) / 1000;
    }
  } else if (data.length >= pixels * 3) {
    for (let i = 0; i < pixels; i++) {
      const j = i * 3;
      dst[i] =
        (299 * data[j] + 587 * data[j + 1] + 114 * data[j + 2] + 500) / 1000;
    }
  } else {
    throw new Error(
      `toGrayMat: cannot interpret image data of length ${data.length} for ${width}x${height}`
    );
  }
  return mat;
}

/**
 * Per-tile mean/std equalization with bilinear interpolation (CLAHE-lite).
 * ORB's binary descriptors are only mildly invariant to exposure and color
 * shifts; day/night relighting (sun position, warmth, sky fill) moves the
 * whole gray distribution. Normalizing here means map building AND
 * relocalization see comparable gradients under any time of day.
 *
 * @param {Uint8Array} dst grayscale buffer, modified in place
 * @param {number} width image width
 * @param {number} height image height
 */
export function normalizeIllumination(dst, width, height) {
  const TILES = 8;
  const tw = Math.max(1, Math.floor(width / TILES));
  const th = Math.max(1, Math.floor(height / TILES));
  const gx = Math.ceil(width / tw);
  const gy = Math.ceil(height / th);
  const mean = new Float32Array(gx * gy);
  const std = new Float32Array(gx * gy);
  for (let ty = 0; ty < gy; ty++) {
    for (let tx = 0; tx < gx; tx++) {
      let s = 0;
      let s2 = 0;
      let n = 0;
      const y1 = Math.min(height, ty * th + th);
      const x1 = Math.min(width, tx * tw + tw);
      for (let y = ty * th; y < y1; y++) {
        for (let x = tx * tw; x < x1; x++) {
          const v = dst[y * width + x];
          s += v;
          s2 += v * v;
          n++;
        }
      }
      const m = s / Math.max(1, n);
      mean[ty * gx + tx] = m;
      std[ty * gx + tx] = Math.sqrt(Math.max(1, s2 / Math.max(1, n) - m * m));
    }
  }
  for (let y = 0; y < height; y++) {
    const fy = Math.min(gy - 1.001, Math.max(0, y / th - 0.5));
    const y0 = Math.floor(fy);
    const wy = fy - y0;
    const y1 = Math.min(gy - 1, y0 + 1);
    for (let x = 0; x < width; x++) {
      const fx = Math.min(gx - 1.001, Math.max(0, x / tw - 0.5));
      const x0 = Math.floor(fx);
      const wx = fx - x0;
      const x1 = Math.min(gx - 1, x0 + 1);
      const i00 = y0 * gx + x0;
      const i01 = y0 * gx + x1;
      const i10 = y1 * gx + x0;
      const i11 = y1 * gx + x1;
      const m =
        (mean[i00] * (1 - wx) + mean[i01] * wx) * (1 - wy) +
        (mean[i10] * (1 - wx) + mean[i11] * wx) * wy;
      const sd =
        (std[i00] * (1 - wx) + std[i01] * wx) * (1 - wy) +
        (std[i10] * (1 - wx) + std[i11] * wx) * wy;
      const v = (dst[y * width + x] - m) * (64 / Math.max(6, sd)) + 128;
      dst[y * width + x] = v < 0 ? 0 : v > 255 ? 255 : v;
    }
  }
}

/**
 * Extract ORB keypoints + binary descriptors from an RGBA raster.
 *
 * @param {object} cv OpenCV namespace
 * @param {{data: Uint8ClampedArray|Uint8Array, width: number, height: number}} image
 * @param {{maxFeatures?: number}} [options]
 * @returns {{keypoints: Array<{x: number, y: number, angle: number, size: number, octave: number}>, descriptors: Uint8Array}}
 */
export function extractOrb(cv, image, {maxFeatures = MAX_FEATURES} = {}) {
  const gray = toGrayMat(cv, image);
  // A/B toggle for measurements (day/night robustness pass).
  if (NORMALIZE_ILLUMINATION) {
    normalizeIllumination(gray.data, image.width | 0, image.height | 0);
  }
  const orb = createOrb(cv, maxFeatures);
  const vector = new cv.KeyPointVector();
  const descriptorsMat = new cv.Mat();
  const mask = new cv.Mat();
  try {
    orb.detectAndCompute(gray, mask, vector, descriptorsMat);
    const available = vector.size();
    const rows = descriptorsMat.rows > 0 ? descriptorsMat.rows : 0;
    const count = Math.min(available, rows);
    const keypoints = new Array(count);
    for (let i = 0; i < count; i++) {
      const kp = vector.get(i);
      keypoints[i] = {
        x: kp.pt.x,
        y: kp.pt.y,
        angle: kp.angle,
        size: kp.size,
        octave: kp.octave,
      };
    }
    const descriptors = new Uint8Array(count * DESC_LEN);
    if (count > 0) {
      descriptors.set(descriptorsMat.data.subarray(0, count * DESC_LEN));
    }
    return {keypoints, descriptors};
  } finally {
    gray.delete();
    mask.delete();
    vector.delete();
    descriptorsMat.delete();
    if (typeof orb.delete === 'function') orb.delete();
  }
}

/**
 * Pinhole K from image size + fov. `fx = fy = w / (2 * tan(fov / 2))`,
 * `cx = w / 2`, `cy = h / 2`.
 *
 * @param {number} width
 * @param {number} height
 * @param {number} [fovDeg]
 * @returns {number[][]} 3x3 intrinsic matrix
 */
export function estimateIntrinsics(width, height, fovDeg = DEFAULT_FOV_DEG) {
  const fx = width / (2 * Math.tan((fovDeg * Math.PI) / 180 / 2));
  const fy = fx;
  return [
    [fx, 0, width / 2],
    [0, fy, height / 2],
    [0, 0, 1],
  ];
}

/**
 * Create an ORB detector honouring `maxFeatures` across OpenCV.js builds
 * (`cv.ORB_create`, `cv.ORB.create` or `new cv.ORB`).
 *
 * @param {object} cv
 * @param {number} maxFeatures
 * @returns {object} ORB detector
 */
function createOrb(cv, maxFeatures) {
  if (typeof cv.ORB_create === 'function') {
    try {
      return cv.ORB_create(maxFeatures);
    } catch {
      return cv.ORB_create();
    }
  }
  try {
    return new cv.ORB(maxFeatures);
  } catch {
    return new cv.ORB();
  }
}
