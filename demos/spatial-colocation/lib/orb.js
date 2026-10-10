/**
 * ORB feature extraction and camera intrinsics helpers.
 *
 * All functions take the OpenCV namespace (`cv`) as an argument so the module
 * stays environment agnostic (Node tests use `@techstark/opencv-js`, the
 * browser uses the CDN global).
 */

/** Max ORB features per frame. */
export const MAX_FEATURES = 1000;
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
 * Extract ORB keypoints + binary descriptors from an RGBA raster.
 *
 * @param {object} cv OpenCV namespace
 * @param {{data: Uint8ClampedArray|Uint8Array, width: number, height: number}} image
 * @param {{maxFeatures?: number}} [options]
 * @returns {{keypoints: Array<{x: number, y: number, angle: number, size: number, octave: number}>, descriptors: Uint8Array}}
 */
export function extractOrb(cv, image, {maxFeatures = MAX_FEATURES} = {}) {
  const gray = toGrayMat(cv, image);
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
