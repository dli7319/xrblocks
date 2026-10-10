/**
 * Sparse 3D landmark map: keyframe bookkeeping, triangulation-driven landmark
 * growth, origin normalization and compact serialization.
 *
 * `geometry.js` is deliberately NOT imported here - its functions are injected
 * through `helpers` so this module stays hermetic (see IMPLEMENTATION_SPEC,
 * `lib/map.js`). Small local matrix helpers are used for pose composition.
 *
 * SpatialMap shape:
 * ```
 * {
 *   version: 1,
 *   K, width, height,               // intrinsics used at capture time
 *   T_ref_map,                      // head pose of the origin (null = unset)
 *   keyframes: [{id, timestamp, T_map_head, T_head_camera, keypoints, descriptors}],
 *   landmarks: [{id, position, descriptor, observations}],
 * }
 * ```
 */

import {
  DESC_LEN,
  KEYFRAME_MIN_MS,
  KEYFRAME_MIN_MOVE_M,
  KEYFRAME_MIN_ROT_DEG,
  MIN_PARALLAX_DEG,
} from './orb.js';
import {
  hamming,
  matchDescriptors as defaultMatchDescriptors,
} from './matching.js';

/** Merge radius for landmark associations (meters). */
const MERGE_RADIUS_M = 0.05;
/** Max descriptor Hamming distance for a landmark association. */
const MERGE_MAX_HAMMING = 48;
/** Hard cap on map size; new observations still merge beyond it. */
export const MAX_LANDMARKS = 12000;
/** Spatial hash cell size; MERGE_RADIUS_M guarantees in-radius points are in the 27-neighborhood. */
const GRID_CELL_M = MERGE_RADIUS_M;
/** Previous keyframes triangulated against per new keyframe. */
const MAX_TRIANGULATION_PARTNERS = 3;
/** Minimum matches before attempting triangulation against a partner. */
const MIN_PARTNER_MATCHES = 4;
/** File magic for serialized maps ('SCM1'). */
const MAGIC = [0x53, 0x43, 0x4d, 0x31];
/** Bytes per keypoint in the binary blob: x, y, angle, size, octave (float64). */
const KEYPOINT_STRIDE = 5;
/** Fixed layout prefix: magic(4) + headerLength(4) + headerOffset(4). */
const PREFIX_BYTES = 12;
/** Serialized format version. */
const FORMAT_VERSION = 1;

/**
 * Create an empty map.
 *
 * @param {{width: number, height: number, K: number[][]}} options
 * @returns {object} SpatialMap
 */
export function createMap({width, height, K}) {
  if (!(width > 0) || !(height > 0)) {
    throw new Error(`createMap: invalid size ${width}x${height}`);
  }
  if (!Array.isArray(K) || K.length !== 3) {
    throw new Error('createMap: K must be a 3x3 matrix');
  }
  return {
    version: FORMAT_VERSION,
    K: cloneMatrix(K),
    width,
    height,
    T_ref_map: null,
    keyframes: [],
    landmarks: [],
  };
}

/**
 * Set/normalize the map origin: the given head pose becomes identity in the
 * map frame. Existing keyframes and landmarks are re-baked into the new frame.
 *
 * @param {object} map
 * @param {number[][]} [T_ref_head_kf0] head pose of the desired origin (identity when omitted)
 */
export function setMapOrigin(map, T_ref_head_kf0) {
  const next = T_ref_head_kf0 ? clonePose(T_ref_head_kf0) : identityPose();
  const previous = map.T_ref_map ? clonePose(map.T_ref_map) : identityPose();
  const delta = matMulLocal(invertRigidLocal(next), previous);
  for (const keyframe of map.keyframes) {
    keyframe.T_map_head = matMulLocal(delta, keyframe.T_map_head);
  }
  for (const landmark of map.landmarks) {
    landmark.position = applyTransform(delta, landmark.position);
  }
  invalidateGrid(map);
  map.T_ref_map = next;
}

/**
 * Add a keyframe (rate-limited + motion-gated) and grow the landmark set.
 *
 * `headPose` is `T_ref_head` in the builder's live reference space; it is
 * normalized with `map.T_ref_map` (auto-initialized from the first keyframe, or
 * set explicitly via `setMapOrigin`).
 *
 * New landmarks are triangulated against up to 3 previous keyframes with
 * sufficient parallax (`MIN_PARALLAX_DEG`), and merged into an existing
 * landmark when within 5 cm and Hamming distance <= 48.
 *
 * @param {object} map
 * @param {{timestamp: number, T_ref_head: number[][], T_head_camera: number[][], kps: Array<object>, descriptors: Uint8Array}} frame
 * @param {{cv: object, estimateEssential: Function, triangulate: Function, matchDescriptors?: Function, MIN_PARALLAX_DEG?: number}} helpers
 * @returns {{newLandmarks: number, newKeyframe: boolean}}
 */
export function addKeyframe(map, frame, helpers) {
  if (!helpers || typeof helpers.triangulate !== 'function') {
    throw new Error('addKeyframe: helpers.triangulate is required');
  }
  const cv = helpers.cv;
  if (!cv) throw new Error('addKeyframe: helpers.cv is required');
  const match = helpers.matchDescriptors || defaultMatchDescriptors;
  const minParallax =
    typeof helpers.MIN_PARALLAX_DEG === 'number'
      ? helpers.MIN_PARALLAX_DEG
      : MIN_PARALLAX_DEG;

  const kps = Array.isArray(frame.kps) ? frame.kps : [];
  const descriptors =
    frame.descriptors instanceof Uint8Array
      ? frame.descriptors
      : Uint8Array.from(frame.descriptors || []);
  const keypointCount = Math.min(
    kps.length,
    Math.floor(descriptors.length / DESC_LEN)
  );
  const timestamp = Number.isFinite(frame.timestamp) ? frame.timestamp : 0;
  const T_head_camera = frame.T_head_camera
    ? clonePose(frame.T_head_camera)
    : identityPose();

  if (map.keyframes.length === 0 && !map.T_ref_map) {
    map.T_ref_map = frame.T_ref_head
      ? clonePose(frame.T_ref_head)
      : identityPose();
  }
  const T_map_head = matMulLocal(
    invertRigidLocal(map.T_ref_map || identityPose()),
    frame.T_ref_head ? clonePose(frame.T_ref_head) : identityPose()
  );

  const previous = map.keyframes[map.keyframes.length - 1];
  if (previous && !isKeyframeAccepted(previous, T_map_head, timestamp)) {
    return {newLandmarks: 0, newKeyframe: false};
  }

  const keyframe = {
    id: nextId(map.keyframes),
    timestamp,
    T_map_head,
    T_head_camera,
    keypoints: [],
    descriptors: descriptors.slice(0, keypointCount * DESC_LEN),
  };
  for (let i = 0; i < keypointCount; i++) {
    const kp = kps[i];
    keyframe.keypoints.push({
      x: numberOr(kp.x, 0),
      y: numberOr(kp.y, 0),
      angle: numberOr(kp.angle, 0),
      size: numberOr(kp.size, 0),
      octave: numberOr(kp.octave, 0),
    });
  }

  const newLandmarks = growLandmarks(
    map,
    keyframe,
    match,
    helpers,
    cv,
    minParallax
  );
  map.keyframes.push(keyframe);
  return {newLandmarks, newKeyframe: true};
}

/**
 * Serialize a map to a compact `Uint8Array`: fixed prefix + binary blobs
 * (Float64 keypoints, raw descriptors) + JSON header. Round-trips exactly.
 *
 * @param {object} map
 * @returns {Uint8Array}
 */
export function serializeMap(map) {
  const keyframeMeta = [];
  const landmarkMeta = [];
  let keypointBytes = 0;
  for (const keyframe of map.keyframes) {
    keyframeMeta.push({
      id: keyframe.id,
      timestamp: keyframe.timestamp,
      T_map_head: keyframe.T_map_head,
      T_head_camera: keyframe.T_head_camera,
      kpOffset: keypointBytes,
      kpCount: keyframe.keypoints.length,
      descOffset: 0,
      descLength: keyframe.descriptors.length,
    });
    keypointBytes += keyframe.keypoints.length * KEYPOINT_STRIDE * 8;
  }
  // Descriptor offsets are absolute inside the blob section so the reader does
  // not have to rebuild section sizes.
  let cursor = keypointBytes;
  let keyframeDescBytes = 0;
  for (const meta of keyframeMeta) {
    meta.descOffset = cursor;
    cursor += meta.descLength;
    keyframeDescBytes += meta.descLength;
  }
  let landmarkDescBytes = 0;
  for (const landmark of map.landmarks) {
    landmarkMeta.push({
      id: landmark.id,
      position: landmark.position,
      observations: landmark.observations,
      descOffset: cursor,
      descLength: landmark.descriptor.length,
    });
    cursor += landmark.descriptor.length;
    landmarkDescBytes += landmark.descriptor.length;
  }

  const blobBytes = keypointBytes + keyframeDescBytes + landmarkDescBytes;
  const headerOffset = PREFIX_BYTES + blobBytes;
  const header = JSON.stringify({
    version: map.version ?? FORMAT_VERSION,
    width: map.width,
    height: map.height,
    K: map.K,
    T_ref_map: map.T_ref_map ?? null,
    keyframes: keyframeMeta,
    landmarks: landmarkMeta,
    layout: {
      keypointBytes,
      keyframeDescBytes,
      landmarkDescBytes,
    },
  });
  const headerBytes = new TextEncoder().encode(header);

  const out = new Uint8Array(PREFIX_BYTES + blobBytes + headerBytes.length);
  out.set(MAGIC, 0);
  const view = new DataView(out.buffer);
  view.setUint32(4, headerBytes.length, true);
  view.setUint32(8, headerOffset, true);

  // Float64 keypoint blob first, then the descriptor blobs; the offsets stored
  // in the header are absolute inside the blob section.
  let writeCursor = PREFIX_BYTES;
  for (const keyframe of map.keyframes) {
    for (const kp of keyframe.keypoints) {
      view.setFloat64(writeCursor, kp.x, true);
      view.setFloat64(writeCursor + 8, kp.y, true);
      view.setFloat64(writeCursor + 16, kp.angle, true);
      view.setFloat64(writeCursor + 24, kp.size, true);
      view.setFloat64(writeCursor + 32, kp.octave, true);
      writeCursor += KEYPOINT_STRIDE * 8;
    }
  }
  keyframeMeta.forEach((meta, index) => {
    out.set(map.keyframes[index].descriptors, PREFIX_BYTES + meta.descOffset);
  });
  landmarkMeta.forEach((meta, index) => {
    out.set(map.landmarks[index].descriptor, PREFIX_BYTES + meta.descOffset);
  });

  out.set(headerBytes, headerOffset);
  return out;
}

/**
 * Rebuild a map produced by {@link serializeMap}.
 *
 * @param {Uint8Array|ArrayBuffer|{buffer: ArrayBuffer, byteOffset: number, byteLength: number}} bytes
 * @returns {object} SpatialMap
 */
export function deserializeMap(bytes) {
  const view = toBytes(bytes);
  if (view.length < PREFIX_BYTES) {
    throw new Error('deserializeMap: buffer too small');
  }
  for (let i = 0; i < MAGIC.length; i++) {
    if (view[i] !== MAGIC[i]) {
      throw new Error(
        'deserializeMap: bad magic header (not a serialized map)'
      );
    }
  }
  const header = new DataView(view.buffer, view.byteOffset, view.byteLength);
  const headerLength = header.getUint32(4, true);
  const headerOffset = header.getUint32(8, true);
  if (headerOffset + headerLength > view.length) {
    throw new Error('deserializeMap: header exceeds buffer');
  }
  const json = new TextDecoder().decode(
    view.subarray(headerOffset, headerOffset + headerLength)
  );
  const parsed = JSON.parse(json);
  if (parsed.version !== FORMAT_VERSION) {
    throw new Error(`deserializeMap: unsupported version ${parsed.version}`);
  }

  const map = {
    version: parsed.version,
    width: parsed.width,
    height: parsed.height,
    K: parsed.K,
    T_ref_map: parsed.T_ref_map ?? null,
    keyframes: [],
    landmarks: [],
  };

  for (const meta of parsed.keyframes) {
    const keypoints = [];
    const kpBytes = meta.kpCount * KEYPOINT_STRIDE * 8;
    if (PREFIX_BYTES + meta.kpOffset + kpBytes > headerOffset) {
      throw new Error('deserializeMap: keypoint blob exceeds buffer');
    }
    const base = PREFIX_BYTES + meta.kpOffset;
    for (let i = 0; i < meta.kpCount; i++) {
      const at = base + i * KEYPOINT_STRIDE * 8;
      keypoints.push({
        x: header.getFloat64(at, true),
        y: header.getFloat64(at + 8, true),
        angle: header.getFloat64(at + 16, true),
        size: header.getFloat64(at + 24, true),
        octave: header.getFloat64(at + 32, true),
      });
    }
    const descLength = meta.descLength ?? meta.kpCount * DESC_LEN;
    if (PREFIX_BYTES + meta.descOffset + descLength > headerOffset) {
      throw new Error(
        'deserializeMap: keyframe descriptor blob exceeds buffer'
      );
    }
    map.keyframes.push({
      id: meta.id,
      timestamp: meta.timestamp,
      T_map_head: meta.T_map_head,
      T_head_camera: meta.T_head_camera,
      keypoints,
      descriptors: view.slice(
        PREFIX_BYTES + meta.descOffset,
        PREFIX_BYTES + meta.descOffset + descLength
      ),
    });
  }

  for (const meta of parsed.landmarks) {
    const descLength = meta.descLength ?? DESC_LEN;
    if (PREFIX_BYTES + meta.descOffset + descLength > headerOffset) {
      throw new Error(
        'deserializeMap: landmark descriptor blob exceeds buffer'
      );
    }
    map.landmarks.push({
      id: meta.id,
      position: meta.position,
      descriptor: view.slice(
        PREFIX_BYTES + meta.descOffset,
        PREFIX_BYTES + meta.descOffset + descLength
      ),
      observations: meta.observations,
    });
  }
  return map;
}

/* ------------------------------------------------------------------ *
 * Internals
 * ------------------------------------------------------------------ */

/**
 * Triangulate the new keyframe against previous keyframes and merge/add the
 * resulting landmarks.
 *
 * @returns {number} count of newly created landmarks
 */
function growLandmarks(map, keyframe, match, helpers, cv, minParallax) {
  const partners = map.keyframes.slice(-MAX_TRIANGULATION_PARTNERS).reverse();
  if (partners.length === 0) return 0;

  const T_cam_new = matMulLocal(keyframe.T_map_head, keyframe.T_head_camera);
  const centerNew = [T_cam_new[0][3], T_cam_new[1][3], T_cam_new[2][3]];
  let created = 0;

  for (const partner of partners) {
    const matches = match(keyframe.descriptors, partner.descriptors);
    if (!Array.isArray(matches) || matches.length < MIN_PARTNER_MATCHES) {
      continue;
    }

    let keep = matches;
    if (typeof helpers.estimateEssential === 'function') {
      try {
        const essential = helpers.estimateEssential(
          cv,
          keyframe.keypoints,
          partner.keypoints,
          matches,
          map.K
        );
        if (
          essential &&
          Array.isArray(essential.inliers) &&
          essential.inliers.length >= MIN_PARTNER_MATCHES
        ) {
          keep = essential.inliers
            .map((index) => matches[index])
            .filter(Boolean);
        }
      } catch {
        keep = matches; // essential estimation is only a filter
      }
    }
    if (keep.length < MIN_PARTNER_MATCHES) continue;

    const T_cam_prev = matMulLocal(partner.T_map_head, partner.T_head_camera);
    const centerPrev = [T_cam_prev[0][3], T_cam_prev[1][3], T_cam_prev[2][3]];
    const ptsNew = keep.map((m) => keyframe.keypoints[m.queryIdx]);
    const ptsPrev = keep.map((m) => partner.keypoints[m.trainIdx]);
    const points = helpers.triangulate(
      cv,
      T_cam_new,
      T_cam_prev,
      ptsNew,
      ptsPrev,
      map.K
    );
    if (!points || points.length < keep.length * 3) continue;

    for (let i = 0; i < keep.length; i++) {
      const x = points[3 * i];
      const y = points[3 * i + 1];
      const z = points[3 * i + 2];
      if (!isFinite(x) || !isFinite(y) || !isFinite(z)) continue;
      const position = [x, y, z];
      const parallax = degBetween(
        [
          position[0] - centerNew[0],
          position[1] - centerNew[1],
          position[2] - centerNew[2],
        ],
        [
          position[0] - centerPrev[0],
          position[1] - centerPrev[1],
          position[2] - centerPrev[2],
        ]
      );
      if (parallax < minParallax) continue;

      const matched = keep[i];
      const descriptor = keyframe.descriptors.subarray(
        matched.queryIdx * DESC_LEN,
        (matched.queryIdx + 1) * DESC_LEN
      );
      if (mergeIntoLandmark(map, position, descriptor)) continue;
      if (map.landmarks.length >= MAX_LANDMARKS) continue;
      const landmark = {
        id: nextId(map.landmarks),
        position,
        descriptor: descriptor.slice(),
        observations: 1,
      };
      map.landmarks.push(landmark);
      gridInsert(map, landmark);
      created += 1;
    }
  }
  return created;
}

/**
 * Merge `position`/`descriptor` into an existing landmark when within
 * `MERGE_RADIUS_M` and Hamming distance <= `MERGE_MAX_HAMMING`. Uses a spatial
 * hash so the search is O(1) in map size instead of O(n) per candidate.
 *
 * @returns {boolean} true when merged
 */
function mergeIntoLandmark(map, position, descriptor) {
  const radiusSq = MERGE_RADIUS_M * MERGE_RADIUS_M;
  const grid = ensureGrid(map);
  const cx = Math.floor(position[0] / GRID_CELL_M);
  const cy = Math.floor(position[1] / GRID_CELL_M);
  const cz = Math.floor(position[2] / GRID_CELL_M);
  for (let gx = cx - 1; gx <= cx + 1; gx++) {
    for (let gy = cy - 1; gy <= cy + 1; gy++) {
      for (let gz = cz - 1; gz <= cz + 1; gz++) {
        const bucket = grid.get(gx + ',' + gy + ',' + gz);
        if (!bucket) continue;
        for (const landmark of bucket) {
          const dx = landmark.position[0] - position[0];
          const dy = landmark.position[1] - position[1];
          const dz = landmark.position[2] - position[2];
          if (dx * dx + dy * dy + dz * dz > radiusSq) continue;
          if (hamming(landmark.descriptor, descriptor) > MERGE_MAX_HAMMING)
            continue;
          landmark.observations += 1;
          return true;
        }
      }
    }
  }
  return false;
}

/**
 * Spatial hash over landmark positions, kept OUTSIDE the map object so maps
 * stay serialization-clean (deep-equality, JSON, structured clone). Rebuilt
 * lazily whenever landmark geometry changed (origin transforms).
 *
 * @returns {Map<string, Array<object>>}
 */
const gridState = new WeakMap();

function ensureGrid(map) {
  let state = gridState.get(map);
  if (
    state &&
    state.grid &&
    state.stamp === state.geomStamp &&
    state.len === map.landmarks.length
  ) {
    return state.grid;
  }
  state = state || {geomStamp: 0};
  state.grid = new Map();
  gridState.set(map, state);
  for (const landmark of map.landmarks) gridInsert(map, landmark);
  state.stamp = state.geomStamp;
  state.len = map.landmarks.length;
  return state.grid;
}

/** Insert one landmark into the spatial hash (used by ensureGrid and new pushes). */
function gridInsert(map, landmark) {
  const state = gridState.get(map);
  if (!state || !state.grid) return;
  const key =
    Math.floor(landmark.position[0] / GRID_CELL_M) +
    ',' +
    Math.floor(landmark.position[1] / GRID_CELL_M) +
    ',' +
    Math.floor(landmark.position[2] / GRID_CELL_M);
  let bucket = state.grid.get(key);
  if (!bucket) {
    bucket = [];
    state.grid.set(key, bucket);
  }
  bucket.push(landmark);
  state.len = map.landmarks.length;
}

/** Invalidate the spatial hash after landmark positions moved. */
function invalidateGrid(map) {
  let state = gridState.get(map);
  if (!state) {
    state = {geomStamp: 0};
    gridState.set(map, state);
  }
  state.geomStamp += 1;
  state.grid = null;
}

/**
 * Keyframe gate: at least `KEYFRAME_MIN_MS` since the previous keyframe AND
 * enough head motion (translation or rotation). The first keyframe always
 * qualifies (handled by the caller).
 *
 * @returns {boolean}
 */
function isKeyframeAccepted(previous, T_map_head, timestamp) {
  const dt = timestamp - previous.timestamp;
  const timeOk = Number.isFinite(dt) ? dt >= KEYFRAME_MIN_MS : true;
  if (!timeOk) return false;

  const dx = T_map_head[0][3] - previous.T_map_head[0][3];
  const dy = T_map_head[1][3] - previous.T_map_head[1][3];
  const dz = T_map_head[2][3] - previous.T_map_head[2][3];
  const moved = Math.sqrt(dx * dx + dy * dy + dz * dz) >= KEYFRAME_MIN_MOVE_M;
  const rotated =
    rotationAngleDeg(
      matMulLocal(T_map_head, transposePose(previous.T_map_head))
    ) >= KEYFRAME_MIN_ROT_DEG;
  return moved || rotated;
}

/* ------------------------------------------------------------------ *
 * Local matrix helpers (geometry.js must not be imported here)
 * ------------------------------------------------------------------ */

function identityPose() {
  return [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ];
}

function cloneMatrix(M) {
  return M.map((row) => row.slice());
}

function clonePose(T) {
  if (!Array.isArray(T) || T.length < 3) {
    throw new Error('pose must be a 4x4 (or 3x4) nested array');
  }
  return T.slice(0, 4).map((row) => {
    const copy = row.slice();
    while (copy.length < 4) copy.push(0);
    return copy;
  });
}

function matMulLocal(A, B) {
  const rows = A.length;
  const inner = B.length;
  const cols = B[0].length;
  const out = new Array(rows);
  for (let i = 0; i < rows; i++) {
    const oi = new Array(cols).fill(0);
    for (let k = 0; k < inner; k++) {
      const aik = A[i][k];
      if (aik === 0) continue;
      for (let j = 0; j < cols; j++) oi[j] += aik * B[k][j];
    }
    out[i] = oi;
  }
  return out;
}

function invertRigidLocal(T) {
  const Rt = [
    [T[0][0], T[1][0], T[2][0]],
    [T[0][1], T[1][1], T[2][1]],
    [T[0][2], T[1][2], T[2][2]],
  ];
  const t = [T[0][3], T[1][3], T[2][3]];
  const nt = [
    -(Rt[0][0] * t[0] + Rt[0][1] * t[1] + Rt[0][2] * t[2]),
    -(Rt[1][0] * t[0] + Rt[1][1] * t[1] + Rt[1][2] * t[2]),
    -(Rt[2][0] * t[0] + Rt[2][1] * t[1] + Rt[2][2] * t[2]),
  ];
  return [
    [Rt[0][0], Rt[0][1], Rt[0][2], nt[0]],
    [Rt[1][0], Rt[1][1], Rt[1][2], nt[1]],
    [Rt[2][0], Rt[2][1], Rt[2][2], nt[2]],
    [0, 0, 0, 1],
  ];
}

function transposePose(T) {
  return [
    [T[0][0], T[1][0], T[2][0], 0],
    [T[0][1], T[1][1], T[2][1], 0],
    [T[0][2], T[1][2], T[2][2], 0],
    [0, 0, 0, 1],
  ];
}

function applyTransform(T, point) {
  return [
    T[0][0] * point[0] + T[0][1] * point[1] + T[0][2] * point[2] + T[0][3],
    T[1][0] * point[0] + T[1][1] * point[1] + T[1][2] * point[2] + T[1][3],
    T[2][0] * point[0] + T[2][1] * point[1] + T[2][2] * point[2] + T[2][3],
  ];
}

function rotationAngleDeg(T) {
  const trace = T[0][0] + T[1][1] + T[2][2];
  const cosine = Math.max(-1, Math.min(1, (trace - 1) / 2));
  return (Math.acos(cosine) * 180) / Math.PI;
}

function degBetween(a, b) {
  const na = Math.hypot(a[0], a[1], a[2]);
  const nb = Math.hypot(b[0], b[1], b[2]);
  if (na === 0 || nb === 0) return 0;
  const dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cosine = Math.max(-1, Math.min(1, dot / (na * nb)));
  return (Math.acos(cosine) * 180) / Math.PI;
}

function nextId(entries) {
  let max = -1;
  for (const entry of entries) {
    if (typeof entry.id === 'number' && entry.id > max) max = entry.id;
  }
  return max + 1;
}

function numberOr(value, fallback) {
  return Number.isFinite(value) ? value : fallback;
}

function toBytes(bytes) {
  if (bytes instanceof Uint8Array) return bytes;
  if (bytes instanceof ArrayBuffer) return new Uint8Array(bytes);
  if (ArrayBuffer.isView(bytes)) {
    return new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  }
  throw new Error('deserializeMap: expected a Uint8Array or ArrayBuffer');
}
