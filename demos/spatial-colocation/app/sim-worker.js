/**
 * ORB extraction Web Worker for the XR Blocks simulator testbed.
 *
 * extractOrb costs ~70ms per 512x512 frame on a slow CPU — far past a frame
 * budget — so it runs here and the render thread never blocks on it.
 *
 * THIS IS A CLASSIC WORKER ON PURPOSE: docs.opencv.org sends no CORS headers,
 * so fetch()/ESM-import of opencv.js always fail from a worker — but
 * importScripts() is exempt from CORS (same as the page's <script> tag).
 * The project's ESM modules are then loaded with dynamic import(), which
 * classic workers do support. The thenable readiness shim of the UMD bundle
 * is disarmed before the namespace travels anywhere (see lib/cv-runtime.js).
 */

const OPENCV_URL = 'https://docs.opencv.org/4.x/opencv.js';

let orbModule = null;
let cvPromise = null;

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Remove the thenable readiness shim (see lib/cv-runtime.js docblock). */
function disarm(cv) {
  if (typeof cv.then === 'function') {
    try {
      delete cv.then;
    } catch {
      /* ignore */
    }
    if (typeof cv.then === 'function') cv.then = undefined;
    if (typeof cv.then === 'function') {
      throw new Error('cv exposes a non-removable .then shim');
    }
  }
  return cv;
}

async function loadCvInWorker() {
  if (!(globalThis.cv && typeof globalThis.cv.Mat === 'function')) {
    importScripts(OPENCV_URL); // synchronous, CORS-exempt like a <script> tag
    const deadline = Date.now() + 30000;
    while (typeof (globalThis.cv && globalThis.cv.Mat) !== 'function') {
      if (typeof globalThis.cv === 'function' && globalThis.cv.then) {
        try {
          globalThis.cv.then(() => {});
        } catch {
          /* readiness shim; polling below still works */
        }
      }
      if (Date.now() > deadline) {
        throw new Error('opencv.js: cv.Mat never appeared (30s)');
      }
      await sleep(15);
    }
  }
  return disarm(globalThis.cv);
}

function ensureCv() {
  if (!cvPromise) cvPromise = loadCvInWorker();
  return cvPromise;
}

async function ensureOrb() {
  if (!orbModule) orbModule = await import('../lib/orb.js');
  return orbModule;
}

self.onmessage = async (e) => {
  const {buffer, width, height, maxFeatures, meta} = e.data;
  try {
    const [cv, orb] = await Promise.all([ensureCv(), ensureOrb()]);
    const image = {
      data: new Uint8ClampedArray(buffer),
      width,
      height,
    };
    const {keypoints, descriptors} = orb.extractOrb(cv, image, {
      maxFeatures: maxFeatures || orb.MAX_FEATURES,
    });
    self.postMessage(
      {ok: true, meta, keypoints, descriptors},
      [descriptors.buffer]
    );
  } catch (err) {
    self.postMessage({
      ok: false,
      meta,
      error: err && err.message ? err.message : String(err),
    });
  }
};
