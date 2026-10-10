# Spatial Colocation Demo — Implementation Spec (binding)

**Repo:** `/opt/data/home/xrblocks-het` (git worktree, branch `het`, fork `dli7319/xrblocks`).
**Rule for all workers: NO git commands, NO changes outside `demos/spatial-colocation/` except the
root `package.json`/`package-lock.json` devDependency additions explicitly listed below. Do NOT touch
`src/`, `samples/`, `templates/`, `docs/`. Never push or open PRs against `google/xrblocks`.**

## Goal

A browser demo proving the concept: **cross-device colocation without a persistent-anchors
runtime**, via app-level visual mapping + relocalization. Four phases, all in one demo app:

- **Phase 0** — ORB feature extraction + cross-view matching on the live camera feed.
- **Phase 1** — calibration (camera→head extrinsics) + triangulation of a sparse 3D landmark map
  from camera frames + head poses.
- **Phase 2** — map persistence (serialize/deserialize + IndexedDB) + PnP relocalization against a
  stored map from a _different session/device_.
- **Phase 3** — networked map sharing + presence: devices on the same room join via PeerJS; a
  device that has relocalized broadcasts its head pose; every device sees the others relocalized
  to the same map.

Deliverable: static page `demos/spatial-colocation/index.html` served over HTTPS, usable from
phones (Android Chrome AR / Quest Browser) and desktop.

## Architecture

Plain ES modules in the browser — **no bundler, no TypeScript**. Core CV/geometry code is
environment-agnostic ESM JavaScript with JSDoc types, tested by colocated Vitest tests in Node
using `@techstark/opencv-js` (installed as a root devDependency by Worker A). In the browser,
OpenCV.js comes from the CDN (`https://docs.opencv.org/4.x/opencv.js`, global `cv`).

### File ownership (disjoint — do not edit files you don't own)

**Worker A (core CV/geometry/pipeline)** owns:

- `demos/spatial-colocation/lib/cv-runtime.js`
- `demos/spatial-colocation/lib/orb.js`
- `demos/spatial-colocation/lib/matching.js`
- `demos/spatial-colocation/lib/geometry.js`
- `demos/spatial-colocation/lib/map.js`
- `demos/spatial-colocation/lib/relocalize.js`
- `demos/spatial-colocation/lib/calibration.js`
- colocated `*.test.js` next to each of the above
- root `package.json` + `package-lock.json`: add devDependency `@techstark/opencv-js`

**Worker B (browser app, networking, visualization)** owns:

- `demos/spatial-colocation/index.html`
- `demos/spatial-colocation/app/main.js`
- `demos/spatial-colocation/app/capture.js`
- `demos/spatial-colocation/app/net.js`
- `demos/spatial-colocation/app/viz.js`
- `demos/spatial-colocation/app/store.js`
- `demos/spatial-colocation/app/ui.js`
- `demos/spatial-colocation/README.md`

Worker B imports Worker A's modules exactly as specified below (they exist as files by spec; if a
symbol is missing at integration time the integrator fixes it — do not redefine them in `app/`).

## Conventions (shared constants — identical everywhere)

- Units: meters, seconds. Right-handed. WebXR reference space: `local-floor`.
- **Map frame** = the builder's head frame at the instant the first keyframe was captured
  (keyframe 0 has head-pose = identity by definition).
- `T_a_b` = 4×4 row-major `Float64Array(16)` (or plain nested number[4][4] — pick nested
  `number[][]` for API simplicity; convert internally) pose of frame _b_ expressed in frame _a_.
  Composition: `T_a_c = T_a_b · T_b_c`.
- Camera model: pinhole, `K = [[fx,0,cx],[0,fy,cy],[0,0,1]]`. Image coords: pixels, origin
  top-left, u right, v down (OpenCV convention).
- Binary ORB descriptors: 32 bytes each, row-major `Uint8Array(n*32)`.

Shared constants (export from `lib/orb.js` and import where needed; B may hardcode but copy values
exactly):

```
MAX_FEATURES        = 1000
MATCH_RATIO         = 0.75      // Lowe ratio test
MIN_RELOC_INLIERS   = 12
MIN_PARALLAX_DEG    = 1.0       // min angle between rays to accept triangulation
KEYFRAME_MIN_MS     = 500
KEYFRAME_MIN_MOVE_M = 0.05
KEYFRAME_MIN_ROT_DEG = 3
DEFAULT_FOV_DEG     = 60
DESC_LEN            = 32
PRESENCE_INTERVAL_MS = 100
```

## Module interfaces (Worker A must export exactly these)

### `lib/cv-runtime.js`

```js
/** Loads the cv namespace. Browser: waits for global `cv` (CDN script). Node: imports @techstark/opencv-js. */
export async function loadCv(): Promise<CvNamespace>
export function isNode(): boolean
```

Implementation detail: detect node via `typeof process !== 'undefined' && process.versions?.node`
and no `window`; use dynamic `import('@techstark/opencv-js')` there (module default-exports cv, may
need to await `cv.onRuntimeInitialized`-style readiness).

### `lib/orb.js`

```js
export const MAX_FEATURES = 1000
export const DESC_LEN = 32
export const DEFAULT_FOV_DEG = 60
/**
 * image: {data: Uint8ClampedArray|Uint8Array (RGBA), width, height}
 * Returns {keypoints: [{x, y, angle, size, octave}], descriptors: Uint8Array(n*32)}
 */
export function extractOrb(cv, image, {maxFeatures = MAX_FEATURES} = {}): {keypoints, descriptors}
/** Grayscale RGBA->gray helper used above, exported for tests/viz. */
export function toGrayMat(cv, image): cv.Mat
/** Pinhole K from image size + fov. fx=fy=w/(2*tan(fov/2)), cx=w/2, cy=h/2. */
export function estimateIntrinsics(width, height, fovDeg = DEFAULT_FOV_DEG): number[3][3]
```

### `lib/matching.js`

```js
export const MATCH_RATIO = 0.75
/** Brute-force Hamming on binary descriptors with Lowe ratio test. Returns [{queryIdx, trainIdx, distance}] sorted ascending by distance. */
export function matchDescriptors(descA, descB, {ratio = MATCH_RATIO} = {}): Match[]
/** Hamming distance between two 32-byte descriptor views. */
export function hamming(a, b): number
```

Do NOT depend on cv.BFMatcher — implement popcount matching in plain JS so node/browser behave
identically.

### `lib/geometry.js`

```js
/**
 * Two-view geometry from 2D-2D matches.
 * kpsA/kpsB: arrays of {x,y}; K: 3x3.
 * Returns {R (3x3), t (3-vector, unit norm), inliers: number[] (indices into matches), essentialMask}
 * or null if estimation fails. Uses cv.findEssentialMat (RANSAC) + cv.recoverPose.
 * IMPORTANT: cheirality — fix sign/scale is NOT possible from essential alone (t is unit);
 * downstream triangulation must choose the sign giving positive depth in view A.
 */
export function estimateEssential(cv, kpsA, kpsB, matches, K): {R, t, inliers} | null

/**
 * Linear triangulation. poseA/poseB are T_world_cam (4x4). ptsA/ptsB: [{x,y}] pixel coords.
 * Returns Float64Array(n*3) world points, one per input pair (NaN rows for behind-camera).
 * Uses cv.triangulatePoints with projection matrices P = K·[R|t]^T per view.
 */
export function triangulate(cv, poseA, poseB, ptsA, ptsB, K): Float64Array

/**
 * PnP RANSAC. pts3d: [{x,y,z}] map-frame, pts2d: [{x,y}] pixels, K.
 * Returns {rvec, tvec, T_cam_obj (4x4: object/map -> camera), inliers: number[]} or null.
 * Uses cv.solvePnPRansac with EPNP + iterations=100, reprojectionError=4.0.
 */
export function solvePnPRansac(cv, pts3d, pts2d, K): {rvec, tvec, T_cam_obj, inliers} | null

/** Matrix utils: matMul, invertRigid (uses R^T, -R^T t), quatFromRotMat, rotMatFromQuat, poseFromRt(R,t), degBetween(v1,v2) */
export function matMul(A, B): number[4][4]
export function invertRigid(T): number[4][4]
export function poseFromRt(R, t): number[4][4]
export function quatFromRotMat(R): [x,y,z,w]
export function rotMatFromQuat(q): number[3][3]
export function degBetween(a, b): number
```

### `lib/map.js`

```js
/**
 * SpatialMap = {
 *   version: 1,
 *   K, width, height,          // intrinsics used at capture time
 *   keyframes: [{id, timestamp, T_map_head, T_head_camera, keypoints, descriptors}],
 *   landmarks: [{id, position: [x,y,z], descriptor: Uint8Array(32), observations: number}],
 * }
 * T_map_head for keyframe 0 is identity; later keyframes compose from the head-pose track the
 * caller supplies.
 */
export function createMap({width, height, K}): SpatialMap

/**
 * Add a keyframe. headPose = T_ref_head (builder's live reference space). The caller normalizes:
 * T_map_head = inverse(T_ref_head_kf0) · T_ref_head. Provide `setMapOrigin` helper for that.
 * Triangulates new landmarks against up to 3 previous keyframes with sufficient parallax
 * (MIN_PARALLAX_DEG); merges into existing landmarks when a new observation is within 5 cm and
 * its descriptor Hamming distance <= 48 (bump `observations`, keep first descriptor).
 * Returns {newLandmarks: count, newKeyframe: bool}
 */
export function addKeyframe(map, {timestamp, T_ref_head, T_head_camera, kps, descriptors}, helpers): {newLandmarks, newKeyframe}

/** Set/normalize origin so that current head pose becomes identity in map frame. */
export function setMapOrigin(map, T_ref_head_kf0): void

/** Serialize to Uint8Array (compact: JSON header + binary blobs; use DataView). Must round-trip exactly. */
export function serializeMap(map): Uint8Array
export function deserializeMap(bytes): SpatialMap
```

`helpers` = `{cv, estimateEssential, triangulate, matchDescriptors, MIN_PARALLAX_DEG}` injected so
map.js has no hard import of geometry.js (keeps unit tests hermetic). A thin convenience wrapper is
fine, but the signature above must exist.

### `lib/relocalize.js`

```js
export const MIN_RELOC_INLIERS = 12
/**
 * Localize a live frame against a map. frame = {kps, descriptors, K}.
 * Strategy: match frame descriptors against ALL map landmark descriptors (Hamming + ratio test
 * against best two map candidates per frame feature), collect 2D-3D pairs, solvePnPRansac.
 * Returns {T_map_camera: 4x4, inliers: number, matchedLandmarkIds: number[]} or null.
 */
export function relocalize(map, frame, {cv, minInliers = MIN_RELOC_INLIERS}): result | null

/** Convert T_map_camera + known T_head_camera -> T_map_head (for presence broadcasts). */
export function headPoseFromCameraPose(T_map_camera, T_head_camera): number[4][4]
```

### `lib/calibration.js`

```js
/**
 * Hand-eye rotation calibration from a motion sequence.
 * headPoses: T_ref_head per frame; cameraPoses: T_ref_cam per frame (from two-view geometry chain,
 * rotation-only accumulation is acceptable; translations are unit-scale and ignored).
 * Solves AX=XB on rotations (quaternion method) -> rotation part of T_head_camera.
 * Returns {R_head_camera (3x3), samples: number}. Returns null if < 10 valid samples or
 * rotational excitation (max angle between consecutive head rotations < 2 deg) is insufficient.
 */
export function estimateHandEyeRotation(headRotations, cameraRotations): {R_head_camera, samples} | null
```

Translation of `T_head_camera` defaults to `[0,0,0]` (documented assumption: camera ≈ head center);
the app exposes a manual override. Calibration is _optional refinement_ in the app, never a blocker.

## App (Worker B) requirements

CDNs (script tags / importmap in `index.html`):

- three.js `0.186.0` via importmap (`https://cdn.jsdelivr.net/npm/three@0.186.0/build/three.module.js`
  and addons path) — same pattern as `templates/00_basic`; look at that template.
- opencv: `<script src="https://docs.opencv.org/4.x/opencv.js">` (async global `cv`; poll or
  `cv.onRuntimeInitialized`).
- PeerJS `1.5.4` (`https://unpkg.com/peerjs@1.5.4/dist/peerjs.min.js`, global `Peer`).

### URL params

- `?room=NAME` — join this room (default: random 6-char).
- `?mode=build|relocalize|live` — start mode (default picker UI if absent).
- `?synthetic=1` — replace camera feed with an animated procedural textured scene rendered from a
  slowly moving virtual camera (canvas → same pipeline). This is the headless test path; it MUST
  work end-to-end (map building converges, a second tab relocalizes).
- `?fov=DEG` override.

### Modes / UI

1. **Build**: start camera (see capture.js), extract ORB at ≤10 Hz on downscaled (≤640px wide)
   frames, add keyframes per KEYFRAME\_\* thresholds, maintain SpatialMap. Show:
   - 2D overlay canvas of current ORB keypoints (Phase 0 visualization — colored by response
     strength).
   - 3D point cloud of map landmarks in a three.js canvas (Phase 1) — color each point by a hash
     of its descriptor so distinctive features stand out.
   - Stats: keyframes, landmarks, matches/frame, fps.
   - "Save map" (IndexedDB via store.js + announce to room).
2. **Relocalize**: load map (ask room host → IndexedDB fallback), run `relocalize` at ≤10 Hz.
   On success: show shared origin marker + inlier count; enter **Live**.
3. **Live**: broadcast presence `{type:'pose', peerId, label, T_map_head, ts}` at
   PRESENCE_INTERVAL_MS; render other devices as labeled avatars (capsule + name) at their pose in
   the 3D view (Phase 3 visualization) + a text list panel: "Devices relocalized to this map:
   [name ✓ inliers, ...]".
4. Desktop/no-camera fallback: synthetic mode also auto-activates if `getUserMedia` is denied.

### `app/capture.js`

- `startCamera({width:640})` → `{video, getFrame(): {data, width, height}, stop()}` via getUserMedia.
- `startSynthetic({width,height})` → same interface, rendering an animated textured room-ish scene
  (procedural: noise-textured quads at varied depths, camera translates/rotates smoothly) via
  OffscreenCanvas/regular canvas + 2D draw; frame data from `ctx.getImageData`.
- `startXRIfNeeded(renderer)` — try `renderer.xr.getSession()` / request `immersive-ar` with
  `local-floor`, expose `getHeadPose()` (from `renderer.xr.getCamera()` matrixWorld, converted to
  T_ref_head) or null on desktop; wrap in try/catch — if simultaneous camera+XR fails (expected
  risk on Android), fall back to camera-only mode with SfM-derived head poses (head pose ≈ camera
  pose, T_head_camera = identity) and say so in the UI.

### `app/net.js`

- PeerJS over the public cloud broker. Room model: mesh-star; first peer in room is **host**. Peer
  id: `${room}-${label}-${rand}`. Host relays messages between other peers.
- Messages (JSON over DataConnection, plus binary map as ArrayBuffer with a 1-byte type prefix 0x02):
  - `0x01` JSON: `{type:'hello', peerId, label, hasMap}` | `{type:'map-request', peerId}` |
    `{type:'map-data', from, name, bytes via 0x02}` | `{type:'pose', peerId, label, T_map_head, inliers, ts}` |
    `{type:'bye', peerId}`
- Reconnect with backoff; UI shows connection status. If host leaves, next peer takes over (basic).

### `app/store.js`

- IndexedDB wrapper: `saveMap(name, bytes)`, `loadMap(name)`, `listMaps()`.

### `app/viz.js`

- three.js scene: landmark point cloud (`THREE.Points`, size ~2px, per-point color), device avatars,
  shared-origin axis marker, and the 2D keypoint overlay painter. In XR mode, put the 3D content in
  world space (head-locked off); on desktop, orbit-style fixed camera.

### `app/main.js`

- Wiring + mode state machine + stats HUD (plain DOM, dark theme, big touch targets for phones).
- All imports of Worker A modules use relative paths `../lib/*.js` with the exact export names.

### `README.md`

- What it demonstrates, the 4 phases, how to run locally (`npx http-server` from repo root or
  `python3 -m http.server`), URL params, per-platform caveats (Quest: WebXR anchors exist but
  camera frames don't — hence app-level approach; Android Chrome: getUserMedia + ARCore camera
  exclusivity risk), and honest limitations (drift, no loop closure, identity extrinsics default).

## Tests (Worker A — colocated `*.test.js`, run by `npx vitest run demos/spatial-colocation`)

- `matching.test.js`: popcount correctness on known descriptor pairs; ratio test filters ambiguous.
- `orb.test.js`: extractOrb on a procedurally generated textured image returns >50 keypoints and
  descriptors of length n\*32; estimateIntrinsics math.
- `geometry.test.js`: (1) triangulate() recovers known 3D points from two known poses to <1e-3 m;
  (2) solvePnPRansac recovers a known pose from perfect 2D-3D correspondences (<0.5° rotation, <1 cm
  translation); (3) estimateEssential on a synthetic textured-plane render with known relative pose
  recovers R within 2° and inlier count > 30.
- `map.test.js`: build a map from a synthetic orbit sequence (rendered textured plane/box points);
  landmarks > 0; serialize→deserialize round-trip deep-equals; setMapOrigin normalizes kf0 to identity.
- `relocalize.test.js`: build map from synthetic sequence, relocalize a held-out view from a
  different "device" (slightly different K) → returns pose within 5°/5 cm of ground truth (or, if
  the synthetic generator is too coarse, assert inliers >= MIN_RELOC_INLIERS and pose within 15°).
- `calibration.test.js`: synthetic rotation sequences with known hand-eye rotation → recovery < 1°;
  insufficient excitation → null.

Test images: generate procedurally (no binary fixtures in repo). A good pattern: create a random
texture (deterministic seeded PRNG) on a 3D plane, project to two views using pure math + canvas-free
rasterization via `cv` drawing primitives, or simpler — build images with `cv.Mat` filled with
seeded noise + drawn shapes, apply `cv.warpPerspective` with a known homography for ORB smoke tests,
and use direct correspondence synthesis (skip rendering) for geometry tests.

## Definition of done

- `npx vitest run demos/spatial-colocation` green.
- `npm run lint` and `npm run format -- --check`... (integrator runs prettier --write on the new
  dir; lint scope is src/tools so demos are exempt but keep style tidy: 2-space, single quotes,
  semicolons per repo `.prettierrc.json`).
- Page loads from a static server with zero console errors; synthetic mode builds a map and a
  second tab relocalizes (integrator verifies headlessly).
- NO modifications anywhere outside the owned paths listed above (plus root package.json devDep).
