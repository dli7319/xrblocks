/** @vitest-environment node */
/**
 * SpatialMap: keyframe gating, triangulation-driven landmark growth, origin
 * normalization and exact serialization round-trips.
 *
 * The synthetic sequence projects a fixed set of textured 3D landmarks from a
 * slowly orbiting head (camera rigidly attached with a non-identity
 * extrinsic), then feeds the resulting keypoints/descriptors to `addKeyframe`
 * - direct correspondence synthesis, no image rendering needed.
 */
import {beforeAll, describe, expect, test} from 'vitest';
import {loadCv} from './cv-runtime.js';
import {
  estimateEssential,
  invertRigid,
  matMul,
  poseFromRt,
  triangulate,
} from './geometry.js';
import {matchDescriptors} from './matching.js';
import {DESC_LEN, KEYFRAME_MIN_MS, MIN_PARALLAX_DEG} from './orb.js';
import {
  addKeyframe,
  createMap,
  deserializeMap,
  serializeMap,
  setMapOrigin,
} from './map.js';

let cv;

const WIDTH = 640;
const HEIGHT = 480;

function yaw(deg) {
  const r = (deg * Math.PI) / 180;
  return [
    [Math.cos(r), 0, Math.sin(r)],
    [0, 1, 0],
    [-Math.sin(r), 0, Math.cos(r)],
  ];
}
function pitch(deg) {
  const r = (deg * Math.PI) / 180;
  return [
    [1, 0, 0],
    [0, Math.cos(r), -Math.sin(r)],
    [0, Math.sin(r), Math.cos(r)],
  ];
}
function makeRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}
function popcount(byte) {
  let v = byte;
  let count = 0;
  while (v) {
    count += v & 1;
    v >>= 1;
  }
  return count;
}
function hamming(a, b) {
  let distance = 0;
  for (let i = 0; i < DESC_LEN; i++) distance += popcount(a[i] ^ b[i]);
  return distance;
}
function withFlippedBits(row, count, salt) {
  const copy = Uint8Array.from(row);
  for (let i = 0; i < count; i++) {
    const bit = (salt + i * 11) % (DESC_LEN * 8);
    copy[bit >> 3] ^= 1 << (bit & 7);
  }
  return copy;
}
function stack(rows) {
  const out = new Uint8Array(rows.length * DESC_LEN);
  rows.forEach((row, i) => out.set(row, i * DESC_LEN));
  return out;
}
function project(P, R, t, K) {
  const v = [P[0] - t[0], P[1] - t[1], P[2] - t[2]];
  const x = R[0][0] * v[0] + R[1][0] * v[1] + R[2][0] * v[2];
  const y = R[0][1] * v[0] + R[1][1] * v[1] + R[2][1] * v[2];
  const z = R[0][2] * v[0] + R[1][2] * v[1] + R[2][2] * v[2];
  if (!(z > 0)) return null;
  return [(K[0][0] * x) / z + K[0][2], (K[1][1] * y) / z + K[1][2]];
}
function rotationOf(T) {
  return [
    [T[0][0], T[0][1], T[0][2]],
    [T[1][0], T[1][1], T[1][2]],
    [T[2][0], T[2][1], T[2][2]],
  ];
}
function translationOf(T) {
  return [T[0][3], T[1][3], T[2][3]];
}
function transformPoint(T, p) {
  return [
    T[0][0] * p[0] + T[0][1] * p[1] + T[0][2] * p[2] + T[0][3],
    T[1][0] * p[0] + T[1][1] * p[1] + T[1][2] * p[2] + T[1][3],
    T[2][0] * p[0] + T[2][1] * p[1] + T[2][2] * p[2] + T[2][3],
  ];
}
function identityPose() {
  return [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ];
}
function poseClose(actual, expected, tolerance = 1e-6) {
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      if (Math.abs(actual[i][j] - expected[i][j]) > tolerance) return false;
    }
  }
  return true;
}

/**
 * Synthetic orbit: head poses orbit the origin, camera = head * T_head_camera.
 *
 * @param {{viewCount?: number, fovDeg?: number, landmarkCount?: number}} [options]
 */
function makeSequence({viewCount = 6, fovDeg = 60, landmarkCount = 220} = {}) {
  const focal = WIDTH / (2 * Math.tan((fovDeg * Math.PI) / 180 / 2));
  const K = [
    [focal, 0, WIDTH / 2],
    [0, focal, HEIGHT / 2],
    [0, 0, 1],
  ];
  const random = makeRandom(20261010);

  const landmarks = [];
  const descriptorRows = [];
  while (landmarks.length < landmarkCount) {
    const descriptor = new Uint8Array(DESC_LEN);
    for (let i = 0; i < DESC_LEN; i++)
      descriptor[i] = Math.floor(random() * 256);
    if (descriptorRows.some((row) => hamming(row, descriptor) < 64)) continue;
    descriptorRows.push(descriptor);
    landmarks.push({
      position: [
        (random() - 0.5) * 4,
        (random() - 0.5) * 3,
        2.5 + random() * 3.5,
      ],
      descriptor,
    });
  }

  const T_head_camera = poseFromRt(pitch(8), [0.02, 0.05, 0.01]);
  const Origin = poseFromRt(yaw(25), [1.5, -0.3, 2]);
  const T_ref_head = (i) => matMul(Origin, T_head(i));
  const T_head = (i) => poseFromRt(yaw(2 * i), [0.25 * i, 0.03 * i, 0]);
  const T_camera = (i) => matMul(T_head(i), T_head_camera);

  const views = [];
  for (let i = 0; i < viewCount; i++) {
    const camera = T_camera(i);
    const kps = [];
    const rows = [];
    let state = (i + 1) * 7919;
    const jitter = () => {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      return (state / 4294967296 - 0.5) * 0.1;
    };
    for (const landmark of landmarks) {
      const pixel = project(
        landmark.position,
        rotationOf(camera),
        translationOf(camera),
        K
      );
      if (!pixel) continue;
      if (
        pixel[0] < 2 ||
        pixel[0] > WIDTH - 2 ||
        pixel[1] < 2 ||
        pixel[1] > HEIGHT - 2
      )
        continue;
      kps.push({
        x: pixel[0] + jitter(),
        y: pixel[1] + jitter(),
        angle: 0,
        size: 31,
        octave: 0,
      });
      rows.push(withFlippedBits(landmark.descriptor, 8 + (i % 9), i * 13 + 5));
    }
    views.push({kps, descriptors: stack(rows)});
  }

  return {
    K,
    landmarks,
    views,
    T_head_camera,
    Origin,
    T_ref_head,
    T_head,
    T_camera,
  };
}

function makeHelpers() {
  return {
    cv,
    estimateEssential,
    triangulate,
    matchDescriptors,
    MIN_PARALLAX_DEG,
  };
}

function frameFrom(sequence, index, extra = {}) {
  return {
    timestamp: index * 600,
    T_ref_head: sequence.T_ref_head(index),
    T_head_camera: sequence.T_head_camera,
    kps: sequence.views[index].kps,
    descriptors: sequence.views[index].descriptors,
    ...extra,
  };
}

describe('map', () => {
  beforeAll(async () => {
    cv = await loadCv();
  }, 60000);

  test('builds landmarks from a synthetic orbit sequence', () => {
    const sequence = makeSequence({viewCount: 6});
    const map = createMap({width: WIDTH, height: HEIGHT, K: sequence.K});
    const helpers = makeHelpers();

    let createdLandmarks = 0;
    for (let i = 0; i < 6; i++) {
      const result = addKeyframe(map, frameFrom(sequence, i), helpers);
      expect(result.newKeyframe).toBe(true);
      expect(result.newLandmarks).toBeGreaterThanOrEqual(0);
      createdLandmarks += result.newLandmarks;
    }

    expect(map.version).toBe(1);
    expect(map.width).toBe(WIDTH);
    expect(map.height).toBe(HEIGHT);
    expect(map.keyframes).toHaveLength(6);
    expect(map.landmarks.length).toBeGreaterThan(50);
    expect(createdLandmarks).toBe(map.landmarks.length);

    // Keyframe 0 is identity by definition, later ones follow the head track.
    expect(poseClose(map.keyframes[0].T_map_head, identityPose())).toBe(true);
    for (let i = 1; i < 6; i++) {
      const expected = matMul(
        invertRigid(sequence.T_head(0)),
        sequence.T_head(i)
      );
      expect(poseClose(map.keyframes[i].T_map_head, expected, 1e-9)).toBe(true);
      expect(map.keyframes[i].timestamp).toBe(i * 600);
      expect(map.keyframes[i].descriptors.length).toBe(
        map.keyframes[i].keypoints.length * DESC_LEN
      );
    }

    // Landmarks sit on the synthetic scene (map frame == head frame at view 0).
    const toMap = invertRigid(sequence.T_head(0));
    const gtPositions = sequence.landmarks.map((lm) =>
      transformPoint(toMap, lm.position)
    );
    for (const landmark of map.landmarks) {
      expect(landmark.descriptor).toBeInstanceOf(Uint8Array);
      expect(landmark.descriptor).toHaveLength(DESC_LEN);
      expect(landmark.observations).toBeGreaterThanOrEqual(1);
      const nearest = Math.min(
        ...gtPositions.map((p) =>
          Math.hypot(
            p[0] - landmark.position[0],
            p[1] - landmark.position[1],
            p[2] - landmark.position[2]
          )
        )
      );
      expect(nearest).toBeLessThan(0.05);
    }

    // Repeated observations merged instead of duplicated.
    const observed = map.landmarks.filter((lm) => lm.observations >= 2);
    expect(observed.length).toBeGreaterThan(20);
    const ids = new Set(map.landmarks.map((lm) => lm.id));
    expect(ids.size).toBe(map.landmarks.length);
    expect(map.keyframes.map((kf) => kf.id)).toEqual([0, 1, 2, 3, 4, 5]);
  });

  test('gates keyframes on elapsed time and head motion', () => {
    const sequence = makeSequence({viewCount: 7});
    const map = createMap({width: WIDTH, height: HEIGHT, K: sequence.K});
    const helpers = makeHelpers();
    addKeyframe(map, frameFrom(sequence, 0), helpers);
    addKeyframe(map, frameFrom(sequence, 1), helpers);
    expect(map.keyframes).toHaveLength(2);
    expect(KEYFRAME_MIN_MS).toBe(500);

    // Too soon since the previous keyframe.
    const tooSoon = addKeyframe(
      map,
      frameFrom(sequence, 2, {timestamp: 700}),
      helpers
    );
    expect(tooSoon).toEqual({newLandmarks: 0, newKeyframe: false});
    expect(map.keyframes).toHaveLength(2);

    // Long enough since the last one, but the head did not move.
    const stationary = addKeyframe(
      map,
      frameFrom(sequence, 1, {timestamp: 600 + 2000}),
      helpers
    );
    expect(stationary).toEqual({newLandmarks: 0, newKeyframe: false});
    expect(map.keyframes).toHaveLength(2);

    // Enough time + motion: accepted.
    const accepted = addKeyframe(map, frameFrom(sequence, 2), helpers);
    expect(accepted.newKeyframe).toBe(true);
    expect(map.keyframes).toHaveLength(3);
  });

  test('rejects frames without injected helpers', () => {
    const sequence = makeSequence({viewCount: 1});
    const map = createMap({width: WIDTH, height: HEIGHT, K: sequence.K});
    expect(() => addKeyframe(map, frameFrom(sequence, 0), {})).toThrow();
    expect(() => createMap({width: 0, height: 10, K: sequence.K})).toThrow();
  });

  test('setMapOrigin() normalizes the chosen keyframe to identity', () => {
    const sequence = makeSequence({viewCount: 5});
    const map = createMap({width: WIDTH, height: HEIGHT, K: sequence.K});
    const helpers = makeHelpers();

    // Pick view 2's head pose as the origin before any keyframe exists.
    setMapOrigin(map, sequence.T_ref_head(2));
    for (let i = 0; i < 5; i++)
      addKeyframe(map, frameFrom(sequence, i), helpers);

    expect(poseClose(map.keyframes[2].T_map_head, identityPose(), 1e-9)).toBe(
      true
    );
    for (let i = 0; i < 5; i++) {
      const expected = matMul(
        invertRigid(sequence.T_ref_head(2)),
        sequence.T_ref_head(i)
      );
      expect(poseClose(map.keyframes[i].T_map_head, expected, 1e-9)).toBe(true);
    }

    // Re-normalizing an existing map re-bakes keyframes and landmarks.
    const landmarkSnapshot = map.landmarks.map((lm) => [...lm.position]);
    const previousOrigin = map.T_ref_map;
    const newOrigin = sequence.T_ref_head(4);
    setMapOrigin(map, newOrigin);
    const delta = matMul(invertRigid(newOrigin), previousOrigin);
    expect(poseClose(map.keyframes[4].T_map_head, identityPose(), 1e-9)).toBe(
      true
    );
    for (let i = 0; i < 5; i++) {
      const expected = matMul(invertRigid(newOrigin), sequence.T_ref_head(i));
      expect(poseClose(map.keyframes[i].T_map_head, expected, 1e-9)).toBe(true);
    }
    map.landmarks.forEach((landmark, index) => {
      const expected = transformPoint(delta, landmarkSnapshot[index]);
      expect(landmark.position[0]).toBeCloseTo(expected[0], 9);
      expect(landmark.position[1]).toBeCloseTo(expected[1], 9);
      expect(landmark.position[2]).toBeCloseTo(expected[2], 9);
    });
  });

  test('serializeMap()/deserializeMap() round-trip exactly', () => {
    const sequence = makeSequence({viewCount: 5});
    const map = createMap({width: WIDTH, height: HEIGHT, K: sequence.K});
    const helpers = makeHelpers();
    for (let i = 0; i < 4; i++)
      addKeyframe(map, frameFrom(sequence, i), helpers);
    expect(map.landmarks.length).toBeGreaterThan(0);

    const bytes = serializeMap(map);
    expect(bytes).toBeInstanceOf(Uint8Array);
    expect(bytes.length).toBeGreaterThan(12);

    const restored = deserializeMap(bytes);
    expect(restored).toEqual(map);
    expect(restored.keyframes[0].descriptors).toBeInstanceOf(Uint8Array);
    expect(restored.landmarks[0].descriptor).toBeInstanceOf(Uint8Array);
    expect(restored.keyframes[0].keypoints[0]).toEqual(
      map.keyframes[0].keypoints[0]
    );

    // Serialization is stable: reserializing gives identical bytes.
    expect(serializeMap(restored)).toEqual(bytes);

    // The restored map keeps working as a map.
    const result = addKeyframe(restored, frameFrom(sequence, 4), helpers);
    expect(result.newKeyframe).toBe(true);
    expect(restored.landmarks.length).toBeGreaterThanOrEqual(
      map.landmarks.length
    );
  });

  test('deserializeMap() rejects buffers that are not maps', () => {
    expect(() => deserializeMap(new Uint8Array(4))).toThrow();
    expect(() =>
      deserializeMap(new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]))
    ).toThrow();
    const sequence = makeSequence({viewCount: 1});
    const map = createMap({width: WIDTH, height: HEIGHT, K: sequence.K});
    addKeyframe(map, frameFrom(sequence, 0), makeHelpers());
    const bytes = serializeMap(map);
    bytes[2] ^= 0xff; // corrupt the magic
    expect(() => deserializeMap(bytes)).toThrow(/magic/);

    const truncated = serializeMap(map).subarray(0, 20);
    expect(() => deserializeMap(truncated)).toThrow();
    // ArrayBuffer input is accepted too.
    const copy = new Uint8Array(serializeMap(map));
    expect(deserializeMap(copy.buffer)).toEqual(map);
  });
});
