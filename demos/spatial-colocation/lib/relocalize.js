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

  const matchedLandmarkIds = solution.inliers.map(
    (index) => landmarkIds[index]
  );
  return {
    T_map_camera: invertRigid(solution.T_cam_obj),
    inliers: solution.inliers.length,
    matchedLandmarkIds,
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
