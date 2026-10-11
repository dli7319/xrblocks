# SIM_SPEC_V2.md — corrections to the XR Blocks sim entry (binding)

Amends `SIM_SPEC.md` per user corrections. The pipeline semantics from
`IMPLEMENTATION_SPEC.md` are unchanged (lib/\* contracts, map frame convention,
networking via `app/net.js`). This supersedes SIM_SPEC.md wherever it conflicts.

## Worker D ownership (disjoint)

MODIFY ONLY:

- `demos/spatial-colocation/sim.html`
- `demos/spatial-colocation/app/sim-main.js`
- `demos/spatial-colocation/app/sim-feed.js`
- `demos/spatial-colocation/sim-transport.html` (probe() must keep working)
- `demos/spatial-colocation/lib/sim/sim-feed.test.js` (only if exports change)

DELETE:

- `demos/spatial-colocation/app/sim-feature-room.js`
- `demos/spatial-colocation/lib/sim/sim-feature-room.test.js`
- `demos/spatial-colocation/lib/sim/sim-feature-room.test.ts`

DO NOT touch anything else (net.js, store.js, ui.js, main.js, viz.js, capture.js,
index.html, README.md, IMPLEMENTATION_SPEC.md, src/\*\*). No git commands.

## Correction 1 — ALL UI must use the XR Blocks UI framework (in-session)

The DOM panel UI (app/ui.js) must no longer drive the sim page: DOM is invisible
inside a WebXR session. Rebuild the sim's UI with the SDK UI components so it
renders in-world in both the desktop simulator and immersive sessions.

SDK facts (verified in this checkout):

- Pattern: `templates/01_spatial_ui/main.js` — read it first. Components are
  `xb.UICard`, `xb.UIText`, `xb.UIButton`, `xb.UIPanel`, `xb.UISlider`,
  `xb.UITextInput`, `xb.UIImage`, `xb.UIScrollView` (all exported via
  `src/ui/index.ts`). Layout is flex-style via `style: {flexDirection, gap,
padding, width...}`; sizes in meters (`size: {width: 0.62, height: 'auto'}`).
- `new xb.UICard({size, manipulation: true, edge: true, style, children})`,
  position it (`card.position.set(0, 1.45, -1.1)`), `this.add(card)` from the
  Script. Buttons: `new xb.UIButton({label, icon, style, onClick})`. Dynamic
  updates: `textNode.text = '...'`. The SDK's interaction pipeline drives
  onClick (mouse on desktop, rays/hands in XR) — no manual raycasting.

Required UI (single main UICard, world-anchored ~1 m ahead at ~1.45 m height,
`manipulation: true` so users can move it):

1. Status line (mode + badge text: 'Build: mapping…', 'relocalized ✓ N inliers',
   'waiting for map…', etc.).
2. Stats row: `kf · lm · matches · fps · inliers` — update text at ≤ 5 Hz from
   the same state the HUD used.
3. Buttons: Build / Relocalize / Live (mode switch), Place cube @1m, Save map,
   Load map. (Map naming: keep the auto-generated name; a `UITextInput` for a
   custom name is optional.)
4. Devices: one line per device ('Dev-xxxx (you) ✓ N inliers'); empty → 'none
   yet'.
5. Log tail: last ~6 log lines in a `UIText` (small font, monospace-ish).
   Keep `?debug=1` behavior. ALSO mirror every log line into a hidden DOM `#log`
   element (and stats into `#stats` with `#st-kf` etc. is NOT required) so
   `sim-transport.html`'s probe() and operators can still read logs — update
   sim-transport.html probe() accordingly if its selectors break. `app/ui.js` is
   used by the raw `index.html` demo and must remain untouched.

## Correction 2 — NO procedural noise textures

Delete the feature room entirely (files listed above). The simulator's own
environment (manifest scenes) is the visual world; ORB must work on whatever
the user's chosen simulator environment provides. Remove `?features=0` and all
`addFeatureRoom`/`disposeFeatureRoom` usage.

## Correction 3 — the ORB feed comes from the XR Blocks Simulator Camera

Stop rendering the scene to an offscreen render target. The feed is the SDK's
device camera stream — in the simulator that is fed by the Simulator Camera
(`XRDeviceCamera.simulatorCamera`), on devices by the real camera.

SDK facts (verified):

- `options.enableCamera()` (CameraOptions; `options.deviceCamera` config) makes
  `Core.init` create `new XRDeviceCamera(options.deviceCamera)` (Core.ts ~612),
  `registry.register` it (~614) and hold it on the core as `deviceCamera`.
  Access from app code via DI (`static dependencies = {deviceCamera:
XRDeviceCamera}`) or `xb.core.deviceCamera` (verify the field is public).
- `XRDeviceCamera extends VideoStream`; `get video()` returns the
  `HTMLVideoElement`, and it has a `texture` (THREE.Texture). Frame pixels for
  ORB: draw `video` into a small canvas (≤ 640 px wide, aspect from
  `video.videoWidth/Height`) at ≤ 10 Hz and `getImageData` — same
  `{data: Uint8ClampedArray, width, height}` shape the pipeline already uses.
  Handle `video.readyState < 2` (skip frame) and 0×0 video size.
- Intrinsics: prefer camera-provided parameters if
  `XRDeviceCameraDetails` / `CameraParameterUtils` expose them (read the
  sources); otherwise keep `intrinsicsFromProjection(camera.projectionMatrix)`
  as the fallback and say which one is used in the log once.
- `app/sim-feed.js` keeps its pure helpers (`intrinsicsFromProjection`,
  `matrixToNested`, `autoSweepPoseAt`, `SWEEP_*` constants) — its tests import
  them. Replace `createSceneFeed` with `createCameraFeed({deviceCamera})` (or
  adapt in place) returning the same `{getFrame(), kind, stop()}` interface,
  `kind: 'camera'`; keep a clearly-labeled synthetic fallback ONLY if the video
  never becomes ready (log a warning).

## Unchanged

Map/reloc/presence/cube/autoSweep semantics, `?debug=1` `window.__scoloc`
hook, chunked map transfer (net.js), URL params (`room`, `mode`, `label`,
`debug`, `autoSweep`, `xrAutomation`).

## Verification (Worker D)

1. `node --check` every touched file; `node demos/spatial-colocation/lib/sim/sim-feed.test.js` green.
2. `npx vitest run demos/spatial-colocation` green (after deleting the feature-room tests, count drops by 1 file).
3. If browser tooling available: load `sim.html?autoSweep=1&mode=build&debug=1`
   headlessly; confirm the UI card exists in the SDK scene, stats text updates,
   `__scoloc.state` feeds from the camera stream (`feed.kind === 'camera'` or
   documented fallback), no console errors. Report what you could not verify.

## Report back

Files changed/deleted, how frames flow from SimulatorCamera to ORB (exact
symbols), which intrinsics source is used, how the UI components are laid out,
and any deviations with reasons.
