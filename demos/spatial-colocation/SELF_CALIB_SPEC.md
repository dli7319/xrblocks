# SELF_CALIB_SPEC.md — online camera self-calibration (binding)

Replaces hardcoded camera parameters in the spatial-colocation demo with
**online self-calibration**: intrinsics `K` (fx, fy, cx, cy) and the
camera-to-head extrinsics `T_head_camera` (rotation + lever arm) are estimated
continuously from the live stream — metric head-pose motion + 2D-2D matches.
Prior/hardcoded values may serve as OPTIMIZATION INITIALIZATION ONLY; the
estimate must converge from data and the apps must display and USE the online
estimate. The simulator knows its true intrinsics and serves as the referee
for measuring estimation error.

## Worker E ownership (disjoint)

CREATE:
- `demos/spatial-colocation/lib/selfcalib.js`
- `demos/spatial-colocation/lib/selfcalib.test.js`
- `demos/spatial-colocation/lib/selfcalib.test.ts` (vitest shim, same pattern as
  `lib/sim/sim-feed.test.ts`)

MODIFY (wiring only — do not restructure these files):
- `demos/spatial-colocation/app/main.js`
- `demos/spatial-colocation/app/sim-main.js`

DO NOT touch: lib/{orb,matching,geometry,map,relocalize,cv-runtime,calibration}.js
(their tests included), app/{net,store,ui,viz,capture,sim-feed}.js, index.html,
sim.html, README.md, everything outside demos/spatial-colocation/. No git
commands.

## Model and math (implement exactly this)

Unknowns (10): `fx, fy, cx, cy`, extrinsics rotation as rotation-vector
(`rx, ry, rz`) and lever arm (`tx, ty, tz`) of `T_head_camera` (camera pose in
the head frame).

Per observation: a matched pair between frames i and j with
- head poses `T_ref_head_i`, `T_ref_head_j` (4x4, whatever convention the app
  uses — both apps use three/WebXR: camera forward = -Z),
- matched pixel coords `(u_i, v_i)`, `(u_j, v_j)`.

Camera pose in ref: `T_ref_cam = T_ref_head · T_head_camera`.
Relative: `T_cam_i_cam_j = inv(T_ref_cam_i) · T_ref_cam_j` = `R_ij, t_ij`.
Normalized bearings: `x = K^-1 [u, v, 1]^T` (K from current estimate).
Epipolar residual (Sampson-normalized): `r = (x_jᵀ E x_i) / sqrt(gx² + gy²)` with
`E = [t_ij]_× R_ij` and `gx, gy` the Sampson gradient terms; skip pairs where
the denominator is degenerate (< 1e-9) or either bearing has z <= 0.

**Convention note**: the apps' poses are three-convention (forward -Z, y up)
while pixels are image-convention (y down). Keep EVERYTHING in one convention
inside the calibrator: convert pixel coords to three-convention normalized
bearings at the boundary (`x = (u-cx)/fx`, `y = -(v-cy)/fy`, `z = -1`), or the
equivalent — but tests must pin it (a known-pose projection test).

Optimizer: damped Gauss-Newton (Levenberg) with NUMERIC Jacobians (central
differences) over the 10 parameters; solve the 10x10 normal system by Gaussian
elimination with pivoting. Per `step()`: up to 25 iterations, stop on small
parameter delta or residual plateau. Bound fx, fy to (0.2·w, 5·w) and cx, cy
to (-w, 2w) by clamping after each update.

Observability guard: a `translationDiversity` metric (RMS of lever-arm-agnostic
translation directions across pairs) gates the (tx, ty, tz) update — on
degenerate motion (pure rotation) keep the lever-arm fixed at its prior and set
`quality.leverArmConstrained = false`. Rotation diversity gate similar for K
(needs rotations about >= 2 distinct axes); expose `quality.constrained`.

## `lib/selfcalib.js` interface (exact)

```js
export class OnlineCalibrator {
  constructor({width, height, priorK, priorTHeadCamera}) // priorK: 3x3 nested, priorT: 4x4 nested
  addPair({T_ref_head_i, T_ref_head_j, pts_i, pts_j, matches})
    // pts_i/pts_j: [{x,y}...] keypoints; matches: [{queryIdx, trainIdx, ...}]
    // Stores into a ring buffer (cap ~400 pairs, drop oldest).
  step(): {iterations, rmsEpipolar, improved}   // runs LM on the buffer
  get K(): number[3][3]                          // current estimate
  get T_head_camera(): number[4][4]
  get quality(): {pairs, rmsEpipolar, constrained, leverArmConstrained, steps}
  reset(priorK?, priorTHeadCamera?)
}
export function KFromCalibrator(calibrator) // -> {fx, fy, cx, cy}
```

## Wiring

### `app/main.js` (raw demo)
- Create one `OnlineCalibrator` per source/camera setup (recreate on source
  toggle and on `ensureIntrinsics` dimension change). Prior K = the current
  `estimateIntrinsics` guess (INIT ONLY); prior T = the manual override value.
- Feed `addPair` from the per-frame `matchDescriptors(prevDesc, desc)` matches
  already computed (prev/current keypoints + `currentHeadPose` chain). When
  head poses come from the SfM chain (scale-ambiguous), feed them anyway — the
  epipolar constraint is scale-invariant.
- Every ~2 s of data: `step()`; when `quality.constrained`, set `state.K` and
  `state.T_head_camera` from the calibrator (this supersedes the identity
  default; the manual extrinsics UI sets the PRIOR and can lock the lever arm).
- UI: extend the stats/status area with a compact calib line, e.g.
  `calib: fx 512.3 fy 511.8 (rms 0.6px, 240 pairs, ✓)` vs `calib: gathering…`;
  use the existing hidden `#log`/ui.log for milestones
  ('online calibration converged: fx=… cy=… rms=…').

### `app/sim-main.js` (XR Blocks sim)
- Same wiring (pairs from consecutive processed frames). Prior K: initialize
  from `intrinsicsFromProjection` of `getDeviceCameraClipFromView` ONLY as the
  starting point — the ONLINE estimate must drive `state.K` once constrained.
- `?kSource=online` (default) | `provided` — `provided` pins the SDK-derived K
  and T (for A/B comparison). Log the active source once.
- UI card: append one calib line to the stats/status text
  (`calib ✓ fx 511.9 · rms 0.5px · 260 pairs` / `calib gathering…`).
- Ground-truth hook for the integrator: with `?debug=1`, expose
  `window.__scoloc.calib = calibrator` (so `calib.K` vs the provided K can be
  compared in the simulator).

## Tests (`lib/selfcalib.test.js`, node:assert style like `app/net.test.js`)

Synthetic generator: random 3D points (spread 1-6 m), TRUE K (e.g. fx=520,
fy=515, cx=319.5, cy=239.5 for 640x480), TRUE extrinsics (rotation ~5 deg,
lever arm ~2 cm), random head trajectory with BOTH translation and rotation
diversity (e.g. 25 frames, sinusoidal + noise), project with noise 0.5 px.

1. K recovery: fx/fy within 2%, cx/cy within 2 px of truth after `step()`
   convergence.
2. Extrinsics recovery: rotation within 2 deg, lever arm within 2 cm (loosen to
   5 cm with a comment if the synthetic geometry makes it weak; state the
   measured error in the test log).
3. Degenerate pure-rotation trajectory: `leverArmConstrained === false` and K
   still converges (rotation-driven self-calibration).
4. Prior is initialization only: start from a 20%-wrong prior; still converges
   to truth.
5. `reset()` clears the buffer and re-converges.

## Verification (Worker E)

`node --check` on touched files; `node demos/spatial-colocation/lib/selfcalib.test.js`
green; `npx vitest run demos/spatial-colocation` green (45 existing + new).
Browser smoke if available: sim.html?autoSweep=1&mode=build&debug=1 — calib
line turns '✓' with plausible fx/fy near the provided estimate, mapping still
converges. Report measured synthetic errors and anything unverified.

## Report back

Files written, measured recovery errors (K + extrinsics), exact convention
handling, how often `step()` runs in the apps, and any deviations.
