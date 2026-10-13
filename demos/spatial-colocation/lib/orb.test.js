/** @vitest-environment node */
/**
 * ORB extraction on a procedurally generated textured image plus intrinsic
 * matrix math.
 */
import {beforeAll, describe, expect, test} from 'vitest';
import {
  DEFAULT_FOV_DEG,
  DESC_LEN,
  MAX_FEATURES,
  estimateIntrinsics,
  extractOrb,
  normalizeIllumination,
  toGrayMat,
} from './orb.js';
import {loadCv} from './cv-runtime.js';

let cv;

/**
 * Procedural texture: seeded noise plus a checkerboard of filled squares, so
 * there are many well-defined corners.
 *
 * @param {number} width
 * @param {number} height
 * @param {number} seed
 * @returns {{data: Uint8ClampedArray, width: number, height: number}}
 */
function makeTexturedImage(width, height, seed = 1234) {
  const data = new Uint8ClampedArray(width * height * 4);
  let state = seed >>> 0;
  const random = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
  for (let i = 0; i < width * height; i++) {
    const value = Math.floor(random() * 256);
    data[4 * i] = value;
    data[4 * i + 1] = value;
    data[4 * i + 2] = value;
    data[4 * i + 3] = 255;
  }
  const cell = 24;
  for (let by = 0; by < height; by += cell) {
    for (let bx = 0; bx < width; bx += cell) {
      if (((bx / cell + by / cell) | 0) % 2 === 0) continue;
      const shade = 255 - ((bx * 7 + by * 3) % 180);
      for (let y = by; y < Math.min(by + 12, height); y++) {
        for (let x = bx; x < Math.min(bx + 12, width); x++) {
          const j = 4 * (y * width + x);
          data[j] = shade;
          data[j + 1] = 255 - shade;
          data[j + 2] = shade >> 1;
        }
      }
    }
  }
  return {data, width, height};
}

describe('orb', () => {
  beforeAll(async () => {
    cv = await loadCv();
  }, 60000);

  test('extractOrb() finds many keypoints with 32-byte descriptors', () => {
    const image = makeTexturedImage(320, 240);
    const {keypoints, descriptors} = extractOrb(cv, image);

    expect(keypoints.length).toBeGreaterThan(50);
    expect(descriptors.length).toBe(keypoints.length * DESC_LEN);
    expect(descriptors).toBeInstanceOf(Uint8Array);
    for (const keypoint of keypoints) {
      expect(Number.isFinite(keypoint.x)).toBe(true);
      expect(Number.isFinite(keypoint.y)).toBe(true);
      expect(keypoint.x).toBeGreaterThanOrEqual(0);
      expect(keypoint.x).toBeLessThan(image.width);
      expect(keypoint.y).toBeGreaterThanOrEqual(0);
      expect(keypoint.y).toBeLessThan(image.height);
      expect(Number.isFinite(keypoint.angle)).toBe(true);
      expect(keypoint.size).toBeGreaterThan(0);
      expect(Number.isInteger(keypoint.octave)).toBe(true);
    }
  });

  test('extractOrb() honours maxFeatures', () => {
    const image = makeTexturedImage(320, 240, 99);
    const capped = extractOrb(cv, image, {maxFeatures: 40});
    expect(capped.keypoints.length).toBeGreaterThan(0);
    expect(capped.keypoints.length).toBeLessThanOrEqual(40);
    expect(capped.descriptors.length).toBe(capped.keypoints.length * DESC_LEN);
    expect(MAX_FEATURES).toBe(1000);
    expect(DEFAULT_FOV_DEG).toBe(60);
  });

  test('extractOrb() accepts single channel input', () => {
    const width = 160;
    const height = 120;
    const data = new Uint8ClampedArray(width * height);
    let state = 5;
    for (let i = 0; i < data.length; i++) {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      data[i] = state >>> 24;
    }
    const {keypoints, descriptors} = extractOrb(cv, {data, width, height});
    expect(keypoints.length).toBeGreaterThan(20);
    expect(descriptors.length).toBe(keypoints.length * DESC_LEN);
  });

  test('toGrayMat() converts RGBA with BT.601 luma', () => {
    const width = 4;
    const height = 1;
    const data = new Uint8ClampedArray([
      255,
      0,
      0,
      255, // red   -> 76
      0,
      255,
      0,
      255, // green -> 150
      0,
      0,
      255,
      255, // blue  -> 29
      255,
      255,
      255,
      255, // white -> 255
    ]);
    const mat = toGrayMat(cv, {data, width, height});
    try {
      expect(mat.rows).toBe(height);
      expect(mat.cols).toBe(width);
      expect(mat.type()).toBe(cv.CV_8UC1);
      expect(Array.from(mat.data)).toEqual([76, 150, 29, 255]);
    } finally {
      mat.delete();
    }
  });

  test('toGrayMat() rejects inconsistent buffers', () => {
    expect(() =>
      toGrayMat(cv, {data: new Uint8ClampedArray(4), width: 8, height: 8})
    ).toThrow();
    expect(() =>
      toGrayMat(cv, {data: new Uint8ClampedArray(64), width: 0, height: 8})
    ).toThrow();
  });

  test('estimateIntrinsics() matches the pinhole formula', () => {
    const fov90 = estimateIntrinsics(640, 480, 90);
    expect(fov90[0][0]).toBeCloseTo(320, 9); // 640 / (2 * tan(45deg))
    expect(fov90[1][1]).toBeCloseTo(320, 9);
    expect(fov90[0][2]).toBe(320);
    expect(fov90[1][2]).toBe(240);
    expect(fov90[2]).toEqual([0, 0, 1]);
    expect(fov90[0][1]).toBe(0);
    expect(fov90[1][0]).toBe(0);

    const expectedFx = 640 / (2 * Math.tan((60 * Math.PI) / 180 / 2));
    const byDefault = estimateIntrinsics(640, 480);
    expect(byDefault[0][0]).toBeCloseTo(expectedFx, 6);
    expect(byDefault[0][0]).toBeCloseTo(554.256, 2);
    expect(byDefault[1][1]).toBe(byDefault[0][0]);
  });

test('normalizeIllumination() equalizes exposure and contrast', () => {
  const w = 64;
  const h = 64;
  const a = new Uint8Array(w * h);
  const b = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const v = 40 + ((x * 7 + y * 13) % 120);
      a[y * w + x] = v;
      b[y * w + x] = Math.max(0, Math.min(255, v * 0.35 + 18)); // darker, flatter
    }
  }
  normalizeIllumination(a, w, h);
  normalizeIllumination(b, w, h);
  let diff = 0;
  for (let i = 0; i < w * h; i++) diff += Math.abs(a[i] - b[i]);
  const meanDiff = diff / (w * h);
  expect(meanDiff).toBeLessThan(8);
});

test('extractOrb() descriptors survive an exposure shift (day/night proxy)', () => {
  const base = makeTexturedImage(320, 240, 42);
  const dark = {
    width: base.width,
    height: base.height,
    data: new Uint8ClampedArray(base.data.length),
  };
  for (let i = 0; i < base.data.length; i += 4) {
    // darker + warmer (sun down): channels scale differently
    dark.data[i] = base.data[i] * 0.32 + 22;
    dark.data[i + 1] = base.data[i + 1] * 0.28 + 16;
    dark.data[i + 2] = base.data[i + 2] * 0.24 + 12;
    dark.data[i + 3] = 255;
  }
  const a = extractOrb(cv, base, {maxFeatures: 300});
  const b = extractOrb(cv, dark, {maxFeatures: 300});
  expect(a.keypoints.length).toBeGreaterThan(30);
  expect(b.keypoints.length).toBeGreaterThan(30);
  // position-matched pairs must mostly keep low Hamming distance
  let matched = 0;
  let stable = 0;
  for (const kb of b.keypoints) {
    let best = null;
    let bestD2 = 25; // 5px gate
    for (const ka of a.keypoints) {
      const d2 = (ka.x - kb.x) ** 2 + (ka.y - kb.y) ** 2;
      if (d2 < bestD2) {
        bestD2 = d2;
        best = ka;
      }
    }
    if (!best) continue;
    matched++;
    const ia = a.keypoints.indexOf(best);
    const ib = b.keypoints.indexOf(kb);
    let ham = 0;
    for (let k = 0; k < DESC_LEN; k++) {
      let x = a.descriptors[ia * DESC_LEN + k] ^ b.descriptors[ib * DESC_LEN + k];
      while (x) {
        ham += x & 1;
        x >>= 1;
      }
    }
    if (ham < 64) stable++;
  }
  expect(matched).toBeGreaterThan(20);
  expect(stable / matched).toBeGreaterThan(0.7);
});
});
