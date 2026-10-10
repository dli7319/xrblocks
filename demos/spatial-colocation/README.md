# Spatial Colocation Demo

Cross-device spatial colocation **without a persistent-anchors runtime** — app-level visual
mapping + relocalization. One static page runs four phases: feature mapping, a sparse 3D
landmark map, PnP relocalization against a map built on a _different device/session_, and
networked presence so every relocalized device appears in everyone else's view.

No build step: plain ES modules, three.js via importmap, OpenCV.js and PeerJS from CDNs.

## The four phases

| Phase | What happens                                                                                                    | Where                                                          |
| ----- | --------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| 0     | ORB feature extraction + cross-view Hamming matching on the (downscaled, ≤10 Hz) feed                           | `lib/orb.js`, `lib/matching.js`, overlay canvas in `viz.js`    |
| 1     | Calibration (camera→head extrinsics, optional hand-eye refinement) + triangulation of a sparse 3D landmark map  | `lib/calibration.js`, `lib/map.js`, 3D point cloud in `viz.js` |
| 2     | Map persistence (serialize/deserialize + IndexedDB) and PnP relocalization of a live frame against a stored map | `lib/map.js`, `lib/relocalize.js`, `app/store.js`              |
| 3     | Devices join a room over PeerJS; a relocalized device broadcasts its head pose; everyone renders the others     | `app/net.js`, avatars + device list in `viz.js` / `ui.js`      |

Map frame = the builder's head frame at the first keyframe (keyframe 0 pose = identity).

## Run it

Any static server works — the page needs no backend (PeerJS uses its public broker):

```bash
# from the repo root
npx http-server -c-1 -p 8080
# or
python3 -m http.server 8080
```

Open `http://localhost:8080/demos/spatial-colocation/`. WebXR and camera access require a
**secure context** — use `localhost` directly, or HTTPS (e.g. `npx http-server -S`) for phones.

### Headless / no-camera test path

```
http://localhost:8080/demos/spatial-colocation/?room=demo&mode=build&synthetic=1
```

`?synthetic=1` swaps the camera for an animated procedural textured scene rendered from a
slowly moving virtual camera (canvas → `getImageData`, same pipeline). The scene is a pure
function of (seed, elapsed time), so **every device renders the identical room** from a
different viewpoint — exactly what cross-device relocalization needs. It also activates
automatically when `getUserMedia` is denied.

Two-tab smoke test:

1. Tab A: `?room=demo&mode=build&synthetic=1` — watch keyframes/landmarks grow, then **Save map**
   (or just leave it running; the map is shareable after the first keyframe).
2. Tab B: `?room=demo&mode=relocalize&synthetic=1` — it requests the map from the room host
   (falls back to IndexedDB storage), relocalizes within a few seconds (`inliers ≥ 12`),
   shows the shared-origin marker and enters **Live**.
3. Both tabs now broadcast presence at 10 Hz; each renders the other's avatar at its pose in
   the map frame and lists it under _Devices relocalized to this map_.

`test-transport.html?room=NAME` is a debugging harness that loads both sides as iframes with
`?debug=1` and exposes `window.probe()` / `window.askMap()` — useful for verifying the
PeerJS map + presence path end to end from one tab.

## URL parameters

| Param                           | Effect                                                           |
| ------------------------------- | ---------------------------------------------------------------- |
| `?room=NAME`                    | Room to join (default: random 6-char id, shown in the header)    |
| `?mode=build\|relocalize\|live` | Start mode (default: picker tabs)                                |
| `?synthetic=1`                  | Procedural camera replacement (also the no-camera fallback)      |
| `?fov=DEG`                      | Vertical/horizontal FOV used for intrinsics (default 60)         |
| `?label=NAME`                   | Device label shown to peers (default: `Dev-xxxx`)                |
| `?debug=1`                      | Expose `window.__scoloc` ({state, net, …}) for in-page debugging |

## UI / modes

- **Build** — start the feed, sweep the view slowly over textured surfaces. ORB keypoints are
  drawn on the overlay (colored by keypoint size, blue weak → red strong); keyframes are
  gated on `KEYFRAME_MIN_MS/MOVE/ROT`; landmarks triangulate against up to 3 previous
  keyframes with sufficient parallax. Stats: keyframes, landmarks, matches/frame, proc fps.
  _Save map_ stores the serialized map in IndexedDB and announces it to the room.
- **Relocalize** — asks the room host for a map (`0x02` binary frame), falls back to IndexedDB
  after 3.5 s. Runs PnP (`solvePnPRansac`) against all landmark descriptors at ≤10 Hz; on
  success (`inliers ≥ 12`) it shows the shared-origin axes + inlier count and enters **Live**.
- **Live** — broadcasts `{type:'pose', peerId, label, T_map_head, inliers, ts}` every 100 ms
  and renders other devices as labeled capsules at their map poses. Between successful
  relocalizations the pose is propagated through the head-pose track.
- **Synthetic fallback** — source toggle button; auto-activates when camera access fails.

Networking is a mesh-star: the first peer claiming `${room}-host` on the public PeerJS broker
is the host and relays messages; if the host leaves, the next peer retries claiming the host
id (basic takeover). Messages are framed binary: `0x01`+UTF-8 JSON for control/pose,
`0x02`+header+bytes for map payloads.

## Platform caveats

- **Quest Browser** — WebXR immersive-ar gives head poses but _no camera frames_, which is the
  whole reason this demo maps at the app level. On Quest the demo runs on the synthetic (or a
  desktop) feed; you get the presence/visualization half, not the SLAM half.
- **Android Chrome (ARCore)** — camera and `immersive-ar` may be mutually exclusive on a given
  device/Chrome build: requesting both can kill the `getUserMedia` feed or fail the session.
  The app wraps the XR attempt in try/catch and falls back to camera-only mode with
  SfM-derived head poses (head pose ≈ camera pose); the header shows `xr: off`.
- **Desktop** — no XR: the 3D view is an orbit camera; head poses come from the SfM chain
  (camera-only) or the synthetic trajectory.
- Public PeerJS cloud broker and CDN assets are required; offline/network-blocked environments
  degrade to single-device + IndexedDB storage.

## Honest limitations

- **Scale drift in camera-only mode**: without a head tracker, the head-pose track is chained
  from essential-matrix translations (unit norm) scaled by a fixed heuristic
  (`SFM_STEP_M = 0.03` m/frame). Landmark geometry and broadcast poses are only as good as
  that chain — treat metric distances as approximate.
- **Identity extrinsics by default**: `T_head_camera` translation defaults to `[0,0,0]`
  (camera ≈ head center). Hand-eye rotation calibration is an _optional refinement_ (needs XR
  head poses + camera motion with ≥2° excitation); a manual translation override is exposed
  under `T_head→cam offset`.
- **No loop closure / no bundle adjustment**: drift accumulates over long build sessions and
  is never corrected; relocalization reuses first-observation descriptors.
- **Fragile textures**: ORB mapping needs textured, well-lit surfaces; plain walls produce no
  keyframes. Relocalization fails (returns no pose) when the current view doesn't match the
  mapped area — keep scanning.
- **Broker trust**: maps and poses travel unauthenticated over the public PeerJS broker within
  the room id; anyone who guesses the room name can join and inject poses/maps.

## Files

```
index.html          page shell, importmap, dark-theme UI
app/main.js         mode state machine, pipeline wiring, HUD
app/capture.js      camera / synthetic / XR frame sources
app/net.js          PeerJS mesh-star, framing, map + pose messages
app/store.js        IndexedDB map persistence
app/viz.js          three.js: point cloud, avatars, origin marker, 2D overlay
app/ui.js           DOM wiring
lib/*.js            environment-agnostic CV/geometry core (see IMPLEMENTATION_SPEC.md)
```
