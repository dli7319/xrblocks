# SIM_SPEC.md — XR Blocks rebuild of the spatial-colocation demo (binding)

Companion to `IMPLEMENTATION_SPEC.md` (its module contracts for `lib/*` and the
wire protocol in `app/net.js` remain binding). This file defines the
**xrblocks-based simulator entry** that rebuilds the demo on the XR Blocks SDK so
the desktop simulator is a testbed for tracking stability.

## Worker C ownership (disjoint — do NOT touch anything else)

CREATE ONLY:

- `demos/spatial-colocation/sim.html`
- `demos/spatial-colocation/app/sim-main.js`
- `demos/spatial-colocation/app/sim-feed.js`
- `demos/spatial-colocation/app/sim-feature-room.js`
- `demos/spatial-colocation/lib/sim/*.test.js` (only if you add pure helpers)

DO NOT modify: `lib/*`, `app/{main,net,store,ui,viz,capture}.js`, `index.html`,
`README.md`, `test-transport.html`, `src/**`, root `package.json`, or anything
outside `demos/spatial-colocation/`. No git commands. `build/xrblocks.js` is
generated: if it is missing run `npm run build:sdk` in the worktree root
(read-only use of the repo build system is allowed; do not edit build output).

## Goal

The same 4-phase colocation pipeline (`lib/orb.js`, `matching.js`, `geometry.js`,
`map.js`, `relocalize.js`, `calibration.js` — exact contracts in
IMPLEMENTATION_SPEC.md) running as an XR Blocks application:

- head poses come from the XR Blocks camera (simulator virtual user on desktop,
  WebXR on device) instead of the synthetic trajectory;
- the ORB feed comes from the SDK's own rendered scene (so the pixels and the
  head poses describe the SAME world — that is what makes it a stability testbed);
- a world-locked **stability cube** is placed 1 m in front of the camera and must
  stay glued to the world as the virtual head moves;
- networking/presence identical to the existing demo (`app/net.js` protocol).

## SDK facts (verified against this checkout — re-verify, do not trust blindly)

- App shape (see `templates/00_basic/main.js` + `CONTEXT.md`): `import * as xb
from 'xrblocks'` (importmap entry `"xrblocks": "../../build/xrblocks.js"`;
  `three` 0.186.0 from jsdelivr exactly like `templates/00_basic/index.html`),
  `class X extends xb.Script { init() {...} update() {...} dispose() {...} }`,
  `xb.add(new X())`, `await xb.init(new xb.Options())`.
- `xb.core.renderer` exists (getter; available after init). `Core` also exposes
  scene/camera (`this.scene`, `this.camera` fields used in `Core.init` — likely
  reachable as `xb.core.scene` / `xb.core.camera`; VERIFY at runtime and in
  `src/core/Core.ts`, fall back to registry lookup if they are not public).
- `xb.user` is a scene Object3D (`xrSystemsGroup.add(this.user, ...)`) with
  `height` / `objectDistance` helpers.
- Simulator: `Options.enableSimulator`, `Options.simulator` (SimulatorOptions),
  `Options.enableAutomationMode(config)`; `?xrAutomation=1` triggers automation
  mode (see `src/core/Options.ts` ~line 287 — confirm the exact trigger).
  `?formFactor=desktop` forces the simulator (CONTEXT.md).
- `Core.stepFrame(dtMs)` exists for manual stepping (automation).
- For scripted virtual-head motion investigate, in this order:
  `src/addons/testing/**`, `src/simulator/userActions/**`
  (`SimulatorUserAction`), `src/simulator/SimulatorUser.ts`, and any
  `*.test.ts`/e2e tests that drive the simulator. Use the SANCTIONED mechanism
  if one exists. If none exists, moving `xb.user` (an Object3D) per `update()`
  is acceptable but must not fight `SimulatorUser`'s own writes (check update
  order / control mode; `SimulatorMode` values include USER, POSE, CONTROLLER,
  POINTER_LOCK, EDITOR — a mode with no user-control writes is ideal).

## Requirements

### `sim.html`

- importmap + CDNs as above; ALSO load `https://docs.opencv.org/4.x/opencv.js`
  and `https://unpkg.com/peerjs@1.5.4/dist/peerjs.min.js` (same as index.html).
- Reuse `app/ui.js` for panels/log/stats ONLY if its DOM contract fits (read it
  first; it needs ids like `#log`, `#stats`, `#st-kf`, `#device-list`,
  `#room-label`, pills, map select). If it does not fit, build a minimal
  equivalent INSIDE `sim-main.js` — do not edit `ui.js`.
- Dark theme, phone/desktop friendly. The SDK canvas is the main viewport;
  panels overlay it.

### `app/sim-feed.js`

- `createSceneFeed({renderer, scene, camera, width = 640})` → `{getFrame():
{data: Uint8ClampedArray, width, height}, kind: 'scene'|'synthetic', stop()}`.
- Renders `scene` from `camera` into a `THREE.WebGLRenderTarget` at ORB ticks
  (≤10 Hz), `readRenderTargetPixels` into RGBA bytes, restores
  `renderer.setRenderTarget(null)` afterwards. Flip rows if needed so image
  coords match OpenCV convention (origin top-left).
- Intrinsics from `camera.projectionMatrix.elements` (perspective): with render
  size w×h, `fx = (w/2) * m[0]`, `cx = (1 - m[8]) * w/2`, `fy = (h/2) * m[5]`,
  `cy = (1 + m[9]) * h/2` — VERIFY the signs on a real render by projecting a
  known world point and checking it lands in frame; export
  `intrinsicsFromProjection(proj, w, h)`.
- If the renderer has no sync `readRenderTargetPixels` (WebGPU), fall back to
  `app/capture.js`'s `startSynthetic` feed AND synthetic head poses (log a clear
  warning) so the page still degrades to the old testbed instead of breaking.

### `app/sim-feature-room.js`

- `addFeatureRoom(scene, {seed = 1})`: deterministic (seeded PRNG) textured
  quads/boxes (canvas noise + shapes textures) arranged around the origin at
  2–6 m, so ORB always has features and the stability cube visibly 'sticks' to
  a wall. `?features=0` disables. Keep poly-count/light (≤ 20 meshes).

### `app/sim-main.js`

- Boot: create `xb.Options()`; formFactor auto (simulator on desktop); if the
  URL has `?xrAutomation=1` (or whatever the Options trigger is) use
  `enableAutomationMode()`. Register one `xb.Script` subclass that owns
  everything; pipeline modules come from `../lib/*` with EXACTLY the contracts
  in IMPLEMENTATION_SPEC.md (`extractOrb`, `matchDescriptors`,
  `estimateIntrinsics`-compatible K, `createMap`, `setMapOrigin`, `addKeyframe`
  with the helpers bag, `relocalize`, `headPoseFromCameraPose`, `createMap`'s
  SpatialMap shape). Reuse `app/net.js` (`createNet`) and `app/store.js` as-is.
- Head pose each frame: `T_ref_head` = camera world matrix (4×4 nested
  `number[][]` per the conventions) — camera IS the head. Map-frame convention
  IDENTICAL to the raw demo (map frame = head frame at first keyframe;
  `setMapOrigin`).
- Modes + URL params: `?room=NAME&mode=build|relocalize|live&label=NAME&debug=1`
  with the same semantics as the raw demo (map request/retry, IndexedDB
  fallback, presence at 10 Hz in every mode, roster of relocalized devices).
- **Stability cube** (the headline feature):
  - 0.25 m cube, distinct color (e.g. orange emissive + edges), added to the SDK
    scene.
  - Placed once at startup (after the first valid head pose / first keyframe)
    and on demand via a `Place cube @1m` button: world position = camera world
    position + camera forward (−Z of camera world matrix) × 1.0 m. It is then
    WORLD-LOCKED (never follows the camera) — its apparent stability IS the
    measurement.
  - Its map-frame pose `T_map_cube` is derivable and exposed on
    `window.__scoloc.state.cube` (`{placed, worldPos, mapPos}`).
- `?autoSweep=1`: deterministic virtual-head sweep (slow circle + yaw scan over
  ~30 s) using whatever sanctioned mechanism you found, so stability is visible
  with zero user input and scriptable headlessly. Document exactly how it moves
  the user.
- `?debug=1`: `window.__scoloc = {state, net, placeCube, xb}`.
- Dispose: stop net, feed, remove listeners (`dispose()` hook).

## Verification (Worker C must do before reporting)

1. `node --check` every file you wrote.
2. `npm run build:sdk` completes (if you ran it) and `build/xrblocks.js` exists.
3. If you have browser tooling: load `sim.html` headlessly and confirm the page
   boots without console errors and `window.__scoloc` exists. (Full E2E —
   mapping + cube stability + presence — is the integrator's job; report any
   runtime issues you could not resolve.)

## Report back

Files written, which SDK facts verified vs corrected (exact symbols used for
scene/camera/renderer access and for scripted head motion), how `?autoSweep=1`
moves the virtual head, and any deviations from this spec with reasons.
