/**
 * Map-based relocalization: match a live frame against the stored landmark set
 * and solve PnP to recover `T_map_camera`.
 */

import {invertRigid, matMul, solvePnPRansac} from './geometry.js';
import {matchDescriptorsWords, packDescriptors} from './matching.js';
import {DESC_LEN} from './orb.js';

/** Minimum PnP inliers before a relocalization is accepted. */
export const MIN_RELOC_INLIERS = 12;

/** Best-distance correspondences fed to PnP; plenty for EPNP, cheap on big maps. */
export const MAX_PNP_CORRESPONDENCES = 300;

/** Per-map cache of packed landmark descriptor words (landmarks are append-only). */
const landmarkWordCache = new WeakMap();

/** Guided-refinement gate (px): the stage-1 pose replaces the ratio test. */
const REFINE_GATE_PX = 25;
/** Max descriptor Hamming distance for a gated match (RANSAC prunes the rest). */
const REFINE_MAX_HAMMING = 64;
/** Isolation toggle for A/B measurements. */
const REFINE_PASSES = 2;
/** Correspondence cap for refinement solves (they start from a good pose). */
const REFINE_MAX_CORRESPONDENCES = 600;

/** 256-entry popcount table for 32-byte descriptor Hamming distance. */
const POPCOUNT = (() => {
  const t = new Uint8Array(256);
  for (let i = 0; i < 256; i++) {
    let v = i;
    let c = 0;
    while (v) {
      c += v & 1;
      v >>= 1;
    }
    t[i] = c;
  }
  return t;
})();

function hamming32(a, ai, b, bi) {
  let d = 0;
  for (let k = 0; k < 32; k++) {
    d += POPCOUNT[a[ai + k] ^ b[bi + k]];
  }
  return d;
}

/**
 * Localize a live frame against a map.
 *
 * Strategy: match the frame descriptors against *all* map landmark descriptors
 * (Hamming + Lowe ratio test against the best two map candidates per frame
 * feature), keep the best match per landmark, collect 2D-3D pairs and run
 * `cv.solvePnPRansac`.
 *
 * @param {object} map SpatialMap
 * @param {{kps: Array<{x: number, y: number}>, descriptors: Uint8Array, K: number[][]}} frame
 * @param {{cv: object, minInliers?: number}} options
 * @returns {{T_map_camera: number[][], inliers: number, matchedLandmarkIds: number[]}|null}
 */
export function relocalize(map, frame, {cv, minInliers = MIN_RELOC_INLIERS}) {
  if (!cv) throw new Error('relocalize: an OpenCV namespace is required');
  if (!map || !Array.isArray(map.landmarks) || map.landmarks.length === 0) {
    return null;
  }
  const kps = frame && Array.isArray(frame.kps) ? frame.kps : [];
  const descriptors = frame && frame.descriptors;
  if (!descriptors || kps.length === 0) return null;

  const landmarkCount = map.landmarks.length;
  let cache = landmarkWordCache.get(map);
  if (!cache || cache.count !== landmarkCount) {
    const landmarkDescriptors = new Uint8Array(landmarkCount * DESC_LEN);
    for (let i = 0; i < landmarkCount; i++) {
      const descriptor = toBytes8(map.landmarks[i].descriptor);
      if (!descriptor || descriptor.length < DESC_LEN) {
        throw new Error(
          'relocalize: map landmark is missing a 32-byte descriptor'
        );
      }
      landmarkDescriptors.set(descriptor.subarray(0, DESC_LEN), i * DESC_LEN);
    }
    cache = {
      count: landmarkCount,
      words: packDescriptors(landmarkDescriptors, landmarkCount),
      bytes: landmarkDescriptors,
    };
    landmarkWordCache.set(map, cache);
  }
  const frameDescriptors = toBytes8(descriptors);
  const frameCount = Math.floor(frameDescriptors.length / DESC_LEN);

  // matchDescriptorsWords returns matches sorted by distance, so the first hit
  // for a landmark is its best one. PnP only needs a few hundred well-separated
  // correspondences; capping keeps the solve fast on large maps.
  const matches = matchDescriptorsWords(
    packDescriptors(frameDescriptors, frameCount),
    cache.words,
    frameCount,
    cache.count,
    {}
  );
  const seen = new Set();
  const pts3d = [];
  const pts2d = [];
  const landmarkIds = [];
  for (const match of matches) {
    if (seen.has(match.trainIdx)) continue;
    seen.add(match.trainIdx);
    const landmark = map.landmarks[match.trainIdx];
    const keypoint = kps[match.queryIdx];
    if (!landmark || !keypoint) continue;
    pts3d.push({
      x: landmark.position[0],
      y: landmark.position[1],
      z: landmark.position[2],
    });
    pts2d.push({x: keypoint.x, y: keypoint.y});
    landmarkIds.push(landmark.id);
    if (landmarkIds.length >= MAX_PNP_CORRESPONDENCES) break;
  }
  if (pts3d.length < 4) return null;

  const K = frame.K || map.K;
  const solution = solvePnPRansac(cv, pts3d, pts2d, K);
  if (!solution || solution.inliers.length < minInliers) return null;

  let bestSolution = solution;
  let bestIds = solution.inliers.map((index) => landmarkIds[index]);
  let bestPts3d = pts3d;
  let bestPts2d = pts2d;

  // Stage 2: geometrically guided refinement (x2). The stage-1 pose replaces
  // the Lowe ratio test — only landmarks projecting within REFINE_GATE_PX of
  // a detected keypoint are candidates, and the best descriptor distance
  // wins. This is what lifts inliers past the ~20 that the ratio test caps
  // single-frame matching at; RANSAC still rejects the remaining mismatches.
  const wpx = 2 * (K[0][2] || 256);
  const hpx = 2 * (K[1][2] || 256);
  for (let pass = 0; pass < REFINE_PASSES; pass++) {
    const T = bestSolution.T_cam_obj;
    const cell = REFINE_GATE_PX;
    const grid = new Map();
    for (let i = 0; i < landmarkCount; i++) {
      const L = map.landmarks[i];
      const lx = L.position[0];
      const ly = L.position[1];
      const lz = L.position[2];
      const X = T[0][0] * lx + T[0][1] * ly + T[0][2] * lz + T[0][3];
      const Y = T[1][0] * lx + T[1][1] * ly + T[1][2] * lz + T[1][3];
      const Z = T[2][0] * lx + T[2][1] * ly + T[2][2] * lz + T[2][3];
      if (Z <= 0.05) continue;
      const u = (K[0][0] * X) / Z + K[0][2];
      const v = (K[1][1] * Y) / Z + K[1][2];
      if (u < -cell || u > wpx + cell || v < -cell || v > hpx + cell) continue;
      const key = `${Math.floor(u / cell)},${Math.floor(v / cell)}`;
      let bucket = grid.get(key);
      if (!bucket) {
        bucket = [];
        grid.set(key, bucket);
      }
      bucket.push(i);
    }
    const refPts3d = [];
    const refPts2d = [];
    const refIds = [];
    // One correspondence per landmark (best Hamming wins): repeated textures
    // otherwise pile several votes onto one landmark and can outvote the
    // true pose in RANSAC.
    const claim = new Map();
    for (let q = 0; q < frameCount; q++) {
      const kx = kps[q].x;
      const ky = kps[q].y;
      const cx0 = Math.floor(kx / cell);
      const cy0 = Math.floor(ky / cell);
      let bestD = REFINE_MAX_HAMMING;
      let bestLm = -1;
      for (let gy2 = -1; gy2 <= 1; gy2++) {
        for (let gx2 = -1; gx2 <= 1; gx2++) {
          const bucket = grid.get(`${cx0 + gx2},${cy0 + gy2}`);
          if (!bucket) continue;
          for (const li of bucket) {
            const d = hamming32(
              frameDescriptors,
              q * DESC_LEN,
              cache.bytes,
              li * DESC_LEN
            );
            if (d < bestD) {
              bestD = d;
              bestLm = li;
            }
          }
        }
      }
      if (bestLm < 0) continue;
      const prev = claim.get(bestLm);
      if (prev === undefined || bestD < prev.d) {
        claim.set(bestLm, {d: bestD, q});
      }
    }
    for (const [li, {q}] of claim) {
      const L = map.landmarks[li];
      refPts3d.push({
        x: L.position[0],
        y: L.position[1],
        z: L.position[2],
      });
      refPts2d.push({x: kps[q].x, y: kps[q].y});
      refIds.push(L.id);
      if (refIds.length >= REFINE_MAX_CORRESPONDENCES) break;
    }
    if (refPts3d.length < 4) break;
    const refined = solvePnPRansac(cv, refPts3d, refPts2d, K);
    if (!refined || refined.inliers.length < bestSolution.inliers.length) {
      break;
    }
    bestSolution = refined;
    bestIds = refined.inliers.map((index) => refIds[index]);
    bestPts3d = refPts3d;
    bestPts2d = refPts2d;
  }

  return {
    T_map_camera: invertRigid(bestSolution.T_cam_obj),
    inliers: bestSolution.inliers.length,
    matchedLandmarkIds: bestIds,
  };
}

/**
 * Convert `T_map_camera` + known `T_head_camera` -> `T_map_head`
 * (for presence broadcasts): `T_map_head = T_map_camera * inverse(T_head_camera)`.
 *
 * @param {number[][]} T_map_camera
 * @param {number[][]} T_head_camera
 * @returns {number[][]} 4x4 `T_map_head`
 */
export function headPoseFromCameraPose(T_map_camera, T_head_camera) {
  return matMul(T_map_camera, invertRigid(T_head_camera));
}

/**
 * Coerce descriptor input to a `Uint8Array` view.
 *
 * @param {Uint8Array|number[]|null|undefined} descriptor
 * @returns {Uint8Array|null}
 */
function toBytes8(descriptor) {
  if (!descriptor) return null;
  if (descriptor instanceof Uint8Array) return descriptor;
  if (ArrayBuffer.isView(descriptor)) {
    return new Uint8Array(
      descriptor.buffer,
      descriptor.byteOffset,
      descriptor.byteLength
    );
  }
  return Uint8Array.from(descriptor);
}
