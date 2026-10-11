// app/sim-feed.js
/**
 * ORB feed sourced from the XR Blocks SDK's own rendered scene.
 *
 * `createSceneFeed()` renders `scene` from `camera` into an offscreen
 * `THREE.WebGLRenderTarget` on demand (the caller ticks it at <= 10 Hz) and
 * reads the pixels back as a top-left-origin RGBA frame — the same contract as
 * `app/capture.js`'s camera/synthetic sources, so the 4-phase pipeline in
 * `app/sim-main.js` sees pixels and head poses that describe the SAME world.
 *
 * Degradation: a renderer without a synchronous `readRenderTargetPixels`
 * (WebGPU) cannot feed ORB from GPU pixels, so the feed falls back to
 * `app/capture.js`'s `startSynthetic()` (and its synthetic head-pose
 * trajectory) and reports `kind: 'synthetic'` with a clear console warning —
 * the page degrades to the old testbed instead of breaking.
 */

import * as THREE from 'three';

import {startSynthetic} from './capture.js';

/** ORB feed width the pipeline expects (frames are downscaled to <= 640 px). */
export const FEED_WIDTH = 640;

/** Number of pixels of tolerance before the intrinsics self-check complains. */
const INTRINSICS_TOLERANCE_PX = 0.5;

// ---- pure helpers (unit-tested in lib/sim/*.test.js) -------------------------

/**
 * Pinhole intrinsics `K` from a three.js perspective `projectionMatrix`.
 *
 * `proj` may be a `THREE.Matrix4`, any `{elements}` holder, or a plain
 * 16-element array — all in three.js column-major order (`e[col * 4 + row]`).
 * For a camera-space point (x, y, z) with z < 0 the projection maps to pixels
 *   u = fx * x / (-z) + cx,  v = fy * (-y) / (-z) + cy
 * (OpenCV convention: origin top-left, v down) with
 *   fx = (w / 2) * e[0], cx = (1 - e[8]) * w / 2,
 *   fy = (h / 2) * e[5], cy = (1 + e[9]) * h / 2.
 * Derived by expanding ndc = P · (x, y, z, 1) / w_clip with w_clip = -z; the
 * signs are re-checked at runtime against the projection itself by
 * `verifyIntrinsicsAgainstProjection()`.
 *
 * @param {object|number[]} proj projection matrix (column-major elements)
 * @param {number} w render width in pixels
 * @param {number} h render height in pixels
 * @returns {number[][]} 3x3 `K` in `lib/orb.js` `estimateIntrinsics` shape
 */
export function intrinsicsFromProjection(proj, w, h) {
  const e = proj && proj.elements ? proj.elements : proj;
  if (!e || e.length < 16) {
    throw new Error(
      'intrinsicsFromProjection: expected 16 column-major elements'
    );
  }
  const fx = (w / 2) * e[0];
  const cx = ((1 - e[8]) * w) / 2;
  const fy = (h / 2) * e[5];
  const cy = ((1 + e[9]) * h) / 2;
  return [
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1],
  ];
}

/**
 * Compare the pinhole-K projection against the projection matrix itself on a
 * grid of camera-space points. The projection matrix is the source of truth
 * for where geometry lands in the render, so any disagreement means the
 * intrinsics the pipeline feeds OpenCV are wrong.
 *
 * @param {object|number[]} proj projection matrix (column-major elements)
 * @param {number} w render width in pixels
 * @param {number} h render height in pixels
 * @returns {{ok: boolean, maxErrPx: number}} agreement within tolerance
 */
export function verifyIntrinsicsAgainstProjection(proj, w, h) {
  const e = proj && proj.elements ? proj.elements : proj;
  const K = intrinsicsFromProjection(e, w, h);
  const fx = K[0][0];
  const cx = K[0][2];
  const fy = K[1][1];
  const cy = K[1][2];
  let maxErrPx = 0;
  for (const x of [-0.8, -0.2, 0.1, 0.6]) {
    for (const y of [-0.5, 0.0, 0.35]) {
      for (const depth of [0.5, 1.7, 4.2]) {
        const z = -depth;
        // Projection-matrix path (three.js clip -> ndc -> pixels).
        const clipX = e[0] * x + e[4] * y + e[8] * z + e[12];
        const clipY = e[1] * x + e[5] * y + e[9] * z + e[13];
        const clipW = e[3] * x + e[7] * y + e[11] * z + e[15];
        const u = ((clipX / clipW + 1) * w) / 2;
        const v = ((1 - clipY / clipW) * h) / 2;
        // Pinhole-K path (OpenCV convention, v down).
        const u2 = (fx * x) / -z + cx;
        const v2 = (fy * -y) / -z + cy;
        maxErrPx = Math.max(maxErrPx, Math.abs(u - u2), Math.abs(v - v2));
      }
    }
  }
  return {ok: maxErrPx <= INTRINSICS_TOLERANCE_PX, maxErrPx};
}

/**
 * Convert a three.js world matrix (column-major `elements`) to the nested
 * row-major `number[4][4]` pose convention of `lib/*` (`T_a_b[row][col]`).
 *
 * @param {object|number[]} m THREE.Matrix4 or 16 column-major elements
 * @returns {number[][]} 4x4 row-major pose
 */
export function matrixToNested(m) {
  const e = m && m.elements ? m.elements : m;
  if (!e || e.length < 16) {
    throw new Error('matrixToNested: expected 16 column-major elements');
  }
  const out = [];
  for (let row = 0; row < 4; row++) {
    const r = [];
    for (let col = 0; col < 4; col++) r.push(e[col * 4 + row]);
    out.push(r);
  }
  return out;
}

// ---- autoSweep virtual-head trajectory (pure; documented motion) ------------

/**
 * `?autoSweep=1` virtual-head trajectory — a deterministic, user-free sweep:
 * the head translates on a horizontal circle of `SWEEP_RADIUS_M` around
 * (0, SWEEP_HEIGHT_M, 0), completing one lap every `SWEEP_PERIOD_S` seconds
 * (slow circle), while yaw scans +/-`SWEEP_YAW_SCAN_DEG` around the outward
 * direction every `SWEEP_YAW_PERIOD_S` seconds and pitch nods
 * +/-`SWEEP_PITCH_DEG` every `SWEEP_PITCH_PERIOD_S` seconds. Height bobs
 * `SWEEP_BOB_M` every 12 s. The head faces outward from the circle center, so
 * the feature room (2-6 m out) stays in view and parallax grows steadily —
 * exactly the motion that reveals (in)stability of the world-locked cube.
 *
 * @param {number} tSec seconds since the sweep started
 * @returns {{position: number[], yaw: number, pitch: number}} pose to apply
 *   to the XB camera (yaw about +Y, pitch about +X, 'YXZ' euler order)
 */
export function autoSweepPoseAt(tSec) {
  const theta = (2 * Math.PI * tSec) / SWEEP_PERIOD_S;
  const position = [
    SWEEP_RADIUS_M * Math.sin(theta),
    SWEEP_HEIGHT_M + SWEEP_BOB_M * Math.sin((2 * Math.PI * tSec) / 12),
    SWEEP_RADIUS_M * Math.cos(theta),
  ];
  // Face outward from the circle center (yaw = theta + PI maps -Z to
  // (sin theta, 0, cos theta)) plus a slow scan around it.
  const yaw =
    theta +
    Math.PI +
    ((SWEEP_YAW_SCAN_DEG * Math.PI) / 180) *
      Math.sin((2 * Math.PI * tSec) / SWEEP_YAW_PERIOD_S);
  const pitch =
    ((SWEEP_PITCH_DEG * Math.PI) / 180) *
    Math.sin((2 * Math.PI * tSec) / SWEEP_PITCH_PERIOD_S);
  return {position, yaw, pitch};
}

export const SWEEP_PERIOD_S = 30;
export const SWEEP_RADIUS_M = 0.5;
export const SWEEP_HEIGHT_M = 1.5;
export const SWEEP_BOB_M = 0.05;
export const SWEEP_YAW_SCAN_DEG = 60;
export const SWEEP_YAW_PERIOD_S = 15;
export const SWEEP_PITCH_DEG = 5;
export const SWEEP_PITCH_PERIOD_S = 9;

// ---- scene feed --------------------------------------------------------------

/**
 * Create an ORB feed that reads the SDK's own rendered pixels.
 *
 * @param {object} options
 * @param {object} options.renderer XB renderer (`xb.core.renderer`)
 * @param {object} options.scene XB scene (`xb.core.scene`)
 * @param {object} options.camera XB camera (`xb.core.camera`) — the head
 * @param {number} [options.width] feed width in pixels (height follows aspect)
 * @returns {{getFrame: Function, kind: string, getHeadPose: Function, stop: Function}}
 *   `getFrame()` renders on demand and returns a reused
 *   `{data: Uint8ClampedArray, width, height}` RGBA buffer (top-left origin);
 *   `getHeadPose()` is null for scene feeds (the caller reads the XB camera)
 *   and the synthetic trajectory pose for the degraded feed.
 */
export function createSceneFeed({
  renderer,
  scene,
  camera,
  width = FEED_WIDTH,
} = {}) {
  if (!renderer || !scene || !camera) {
    throw new Error('createSceneFeed: renderer, scene and camera are required');
  }

  if (typeof renderer.readRenderTargetPixels !== 'function') {
    console.warn(
      '[sim-feed] renderer has no synchronous readRenderTargetPixels (WebGPU?) — ' +
        'falling back to the app/capture.js synthetic feed AND synthetic head ' +
        'poses; the page degrades to the old testbed.'
    );
    const height = Math.max(1, Math.round((width * 3) / 4));
    const synth = startSynthetic({width, height});
    return {
      kind: 'synthetic',
      getFrame: () => synth.getFrame(),
      getHeadPose: () => synth.getHeadPose(),
      stop: () => synth.stop(),
    };
  }

  const aspect = camera.aspect > 0 ? camera.aspect : 4 / 3;
  const height = Math.max(1, Math.round(width / aspect));
  const target = new THREE.WebGLRenderTarget(width, height, {
    depthBuffer: true,
    stencilBuffer: false,
  });
  const pixels = new Uint8Array(width * height * 4);
  const frame = new Uint8ClampedArray(width * height * 4);
  let verified = false;

  return {
    kind: 'scene',
    getFrame() {
      const previous = renderer.getRenderTarget();
      renderer.setRenderTarget(target);
      renderer.render(scene, camera);
      renderer.readRenderTargetPixels(target, 0, 0, width, height, pixels);
      renderer.setRenderTarget(previous ?? null);
      if (!verified) {
        // Verify the intrinsics signs on real render data: the projection
        // matrix decides where geometry lands, so K must agree with it.
        verified = true;
        const check = verifyIntrinsicsAgainstProjection(
          camera.projectionMatrix,
          width,
          height
        );
        if (!check.ok) {
          console.warn(
            `[sim-feed] intrinsics disagree with the projection matrix by ` +
              `${check.maxErrPx.toFixed(2)} px — check intrinsicsFromProjection()`
          );
        }
      }
      // readRenderTargetPixels returns rows bottom-up; flip to OpenCV order.
      const rowBytes = width * 4;
      for (let row = 0; row < height; row++) {
        const src = (height - 1 - row) * rowBytes;
        frame.set(pixels.subarray(src, src + rowBytes), row * rowBytes);
      }
      return {data: frame, width, height};
    },
    getHeadPose() {
      return null;
    },
    stop() {
      target.dispose();
    },
  };
}
