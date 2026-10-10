/** @vitest-environment node */
/**
 * Runtime bootstrap tests: the loader must resolve a ready OpenCV namespace in
 * Node and hand it back through a promise without dead-locking on the OpenCV.js
 * thenable readiness shim.
 */
import {beforeAll, describe, expect, test} from 'vitest';
import {isNode, loadCv} from './cv-runtime.js';

let cv;

describe('cv-runtime', () => {
  beforeAll(async () => {
    cv = await loadCv();
  }, 60000);

  test('isNode() detects the node test environment', () => {
    expect(isNode()).toBe(true);
    expect(typeof window).toBe('undefined');
  });

  test('loadCv() resolves a ready OpenCV namespace', () => {
    expect(cv).toBeTruthy();
    expect(typeof cv.Mat).toBe('function');
    expect(typeof cv.solvePnPRansac).toBe('function');
    expect(typeof cv.ORB).toBe('function');
    expect(typeof cv.KeyPointVector).toBe('function');
  });

  test('loadCv() disarms the thenable readiness shim', async () => {
    // Awaiting an object that still exposes `.then` recurses forever.
    expect(typeof cv.then).not.toBe('function');
    const again = await loadCv();
    expect(typeof again.Mat).toBe('function');
  });

  test('cv.Mat allocates and frees memory', () => {
    const mat = new cv.Mat(4, 4, cv.CV_8UC1);
    expect(mat.rows).toBe(4);
    expect(mat.cols).toBe(4);
    expect(mat.data.length).toBe(16);
    mat.delete();
  });
});
