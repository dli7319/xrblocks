/** @vitest-environment node */
/**
 * Relocalization against a stored map built from a synthetic sequence, using a
 * held-out view captured by a "different device" (different intrinsics,
 * keypoint noise, descriptor bit flips).
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
import {DESC_LEN, MIN_PARALLAX_DEG} from './orb.js';
import {addKeyframe, createMap} from './map.js';
import {
  MIN_RELOC_INLIERS,
  headPoseFromCameraPose,
  relocalize,
} from './relocalize.js';

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
function intrinsics(fovDeg) {
  const focal = WIDTH / (2 * Math.tan((fovDeg * Math.PI) / 180 / 2));
  return [
    [focal, 0, WIDTH / 2],
    [0, focal, HEIGHT / 2],
    [0, 0, 1],
  ];
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
function project(P, R, t, K) {
  const v = [P[0] - t[0], P[1] - t[1], P[2] - t[2]];
  const x = R[0][0] * v[0] + R[1][0] * v[1] + R[2][0] * v[2];
  const y = R[0][1] * v[0] + R[1][1] * v[1] + R[2][1] * v[2];
  const z = R[0][2] * v[0] + R[1][2] * v[1] + R[2][2] * v[2];
  if (!(z > 0)) return null;
  return [(K[0][0] * x) / z + K[0][2], (K[1][1] * y) / z + K[1][2]];
}
function rotationErrorDeg(Ra, Rb) {
  const relative = matMul(Ra, [
    [Rb[0][0], Rb[1][0], Rb[2][0]],
    [Rb[0][1], Rb[1][1], Rb[2][1]],
    [Rb[0][2], Rb[1][2], Rb[2][2]],
  ]);
  const trace = relative[0][0] + relative[1][1] + relative[2][2];
  return (
    (Math.acos(Math.max(-1, Math.min(1, (trace - 1) / 2))) * 180) / Math.PI
  );
}

/** Synthetic world: fixed landmarks + orbiting head, camera = head * extrinsic. */
function makeSequence({viewCount = 5, fovDeg = 60, landmarkCount = 220} = {}) {
  const K = intrinsics(fovDeg);
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
  const T_head = (i) => poseFromRt(yaw(2 * i), [0.25 * i, 0.03 * i, 0]);
  const T_camera = (i) => matMul(T_head(i), T_head_camera);
  const T_ref_head = (i) =>
    matMul(poseFromRt(yaw(25), [1.5, -0.3, 2]), T_head(i));

  /** Render one view; `salt` varies descriptor flips per device/frame. */
  const render = (headPose, intrinsicsMatrix, salt, jitterPx) => {
    const camera = matMul(headPose, T_head_camera);
    const kps = [];
    const rows = [];
    let state = salt * 7919 + 13;
    const jitter = () => {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      return (state / 4294967296 - 0.5) * 2 * jitterPx;
    };
    for (const landmark of landmarks) {
      const pixel = project(
        landmark.position,
        rotationOf(camera),
        translationOf(camera),
        intrinsicsMatrix
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
      rows.push(
        withFlippedBits(landmark.descriptor, 8 + (salt % 9), salt * 13 + 5)
      );
    }
    return {kps, descriptors: stack(rows)};
  };

  const views = [];
  for (let i = 0; i < viewCount; i++) {
    views.push(render(T_head(i), K, i + 1, 0.05));
  }

  return {
    K,
    landmarks,
    views,
    T_head_camera,
    T_head,
    T_camera,
    T_ref_head,
    render,
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

describe('relocalize', () => {
  beforeAll(async () => {
    cv = await loadCv();
  }, 60000);

  test('relocalizes a held-out view from a different device', () => {
    const sequence = makeSequence({viewCount: 5});
    const map = createMap({width: WIDTH, height: HEIGHT, K: sequence.K});
    const helpers = makeHelpers();
    for (let i = 0; i < 5; i++) {
      addKeyframe(
        map,
        {
          timestamp: i * 600,
          T_ref_head: sequence.T_ref_head(i),
          T_head_camera: sequence.T_head_camera,
          kps: sequence.views[i].kps,
          descriptors: sequence.views[i].descriptors,
        },
        helpers
      );
    }
    expect(map.landmarks.length).toBeGreaterThan(30);

    // Map frame = builder's head frame at keyframe 0.
    const toMap = invertRigid(sequence.T_head(0));
    // Held-out pose: between views 3 and 4, rendered with a *different* fov.
    const heldOutIndex = 3.5;
    const fovOther = 63;
    const frame = {
      K: intrinsics(fovOther),
      ...sequence.render(
        sequence.T_head(heldOutIndex),
        intrinsics(fovOther),
        42,
        0.3
      ),
    };
    expect(frame.kps.length).toBeGreaterThan(30);

    const result = relocalize(map, frame, {cv});
    expect(result).not.toBeNull();
    expect(result.inliers).toBeGreaterThanOrEqual(MIN_RELOC_INLIERS);
    expect(result.T_map_camera).toHaveLength(4);
    expect(result.matchedLandmarkIds).toHaveLength(result.inliers);
    const known = new Set(map.landmarks.map((lm) => lm.id));
    for (const id of result.matchedLandmarkIds)
      expect(known.has(id)).toBe(true);

    // Ground truth T_map_camera for the held-out view.
    const T_map_camera = matMul(toMap, sequence.T_camera(heldOutIndex));
    const rotationError = rotationErrorDeg(
      rotationOf(result.T_map_camera),
      rotationOf(T_map_camera)
    );
    const translationError = Math.hypot(
      result.T_map_camera[0][3] - T_map_camera[0][3],
      result.T_map_camera[1][3] - T_map_camera[1][3],
      result.T_map_camera[2][3] - T_map_camera[2][3]
    );
    console.log(
      `[relocalize] inliers=${result.inliers} rotErr=${rotationError.toFixed(3)}deg tErr=${translationError.toFixed(4)}m`
    );
    // Spec primary criterion: within 5 deg / 5 cm of ground truth (measured
    // ~0.02 deg / ~0.002 m for this sequence, well inside the margins).
    expect(rotationError).toBeLessThan(5);
    expect(translationError).toBeLessThan(0.05);
  }, 60000);

  test('headPoseFromCameraPose() converts back to the head frame', () => {
    const sequence = makeSequence({viewCount: 3});
    const toMap = invertRigid(sequence.T_head(0));
    const index = 2;
    const T_map_camera = matMul(toMap, sequence.T_camera(index));
    const headPose = headPoseFromCameraPose(
      T_map_camera,
      sequence.T_head_camera
    );
    const expected = matMul(toMap, sequence.T_head(index));
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        expect(headPose[i][j]).toBeCloseTo(expected[i][j], 9);
      }
    }
    // Identity extrinsics pass the camera pose straight through.
    const identityExtrinsic = [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
    ];
    expect(headPoseFromCameraPose(T_map_camera, identityExtrinsic)).toEqual(
      T_map_camera
    );
  });

  test('returns null when the map or frame is unusable', () => {
    const sequence = makeSequence({viewCount: 2});
    const emptyMap = createMap({width: WIDTH, height: HEIGHT, K: sequence.K});
    const frame = {K: sequence.K, ...sequence.views[0]};
    expect(relocalize(emptyMap, frame, {cv})).toBeNull();

    const map = createMap({width: WIDTH, height: HEIGHT, K: sequence.K});
    const helpers = makeHelpers();
    for (let i = 0; i < 2; i++) {
      addKeyframe(
        map,
        {
          timestamp: i * 600,
          T_ref_head: sequence.T_ref_head(i),
          T_head_camera: sequence.T_head_camera,
          kps: sequence.views[i].kps,
          descriptors: sequence.views[i].descriptors,
        },
        helpers
      );
    }
    expect(
      relocalize(
        map,
        {K: sequence.K, kps: [], descriptors: new Uint8Array(0)},
        {cv}
      )
    ).toBeNull();
    expect(
      relocalize(
        map,
        {
          K: sequence.K,
          kps: [{x: 1, y: 1}],
          descriptors: new Uint8Array(DESC_LEN),
        },
        {cv, minInliers: 1000}
      )
    ).toBeNull();
    expect(() => relocalize(map, frame, {})).toThrow(/OpenCV/);
  }, 60000);
});
