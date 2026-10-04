# 001: Day→Night Loft Cycle

**Question:** Can a dynamic day–night cycle for the Simulator's baked Loft scenes
("Daytime Loft" / "Evening Loft") be rendered *without* rebaking or many
keyframes, while keeping the baked scene quality and making directional window
light move correctly?

**Approach:** sun-free baked base + real-time shadow-mapped sun.

- Base pass: the **evening GLB as-is** (unlit baked textures — they contain no
  sun, only artificial/ambient light), so all baked GI/AO/art direction is kept
  exactly.
- Sun pass: the same meshes rendered a second time with an additive Lambert
  material lit by one directional light whose shadow map is cut by the real
  window openings → window-shaped sun patches that **move and elongate** as the
  sun arc (azimuth/elevation/color/intensity vs `t`) changes. No double-lighting
  because the base carries no sun.
- Sky fill: a hemisphere light in the additive pass, faded/deepened by `t`.
- Window shades: the shade geometry exists **only** in the night GLB (merged
  into the room-shell meshes). It is split out at load (day/night vertex diff,
  ~5 mm world-space grid) into separate meshes and animated as rolling blinds
  (scale about the housing top), faded in over `t∈[0.42,0.52]`, rolled down over
  `t∈[0.5,0.82]`.
- Outside view: day/night sky-box textures blended on the sky sphere.
- Day-only rug: fades out with `t`. Smaller day/night differences (coffee table
  moved ~8 cm, desk prop, lamp re-tessellation) snap to the night state.

Run: serve the repo root and open `spikes/001-day-night-loft/index.html`.
Drive with `window.__frame(t, p)` (`t` = 0..1 day→night, `p` = camera drift).

## Results

- 22/23 meshes pair across the two GLBs; the shade splitter extracted exactly
  the window shades (office window 178 tris, living-room windows 326 tris) plus
  the day-only rug.
- Window-shaped sun patches confirmed at `t=0`; at `t=0.25` they elongate and
  move deeper into the room as the sun lowers; at `t=0.75` warm low sun + blinds
  nearly down; at `t=1.0` blinds fully down, dark exterior, interior lamp glow
  from the bake.
- 8 s / 240-frame cycle recorded: `daynight_loft_cycle.mp4` (in the session
  artifacts, not committed).
- No shader plumbing beyond one `onBeforeCompile` sky-blend patch; the sun pass
  reuses three.js's shadow-map machinery via `scene.overrideMaterial`.

## Verdict: VALIDATED

### What worked

- Moving, window-shaped directional light from one shadow-mapped directional —
  exactly the behavior keyframe blending cannot produce.
- Baked quality preserved: the base is the untouched evening bake.
- Shade splitting from the vertex diff is reliable and gives a real rolling-blind
  animation for the one genuine geometry state change.
- Cost is small: 23 meshes rendered twice + one shadow pass; no rebakes, no
  keyframe texture sets.

### What didn't

- The bake's own baked shadows coexist with the real-time sun shadows; some
  furniture shadows read slightly "double" at high sun. Tunable (bias) but not
  fully removable — the bake's shadows are part of the base.
- Small prop placement differences (8 cm table shift, desk item) can't be
  both-everything without crossfades; they snap to night state.
- Day endpoint is *relit*, not pixel-identical to the old Daytime Loft (the day
  bake remains the tuning reference for noon color/intensity).

### Surprises

- The two GLBs differ in geometry, not just textures: night-only roller shades
  (housing + fabric + weight bar) merged into the room shells, a day-only rug,
  and several moved props. Any plan must handle "state geometry", not only
  textures.
- The GLBs ship without normals (unlit export); `computeVertexNormals()` is
  sufficient for the sun pass on architectural geometry.
- Only the sky sphere has `COLOR_0` (a vertex-color gradient) — no AO bake hiding
  in vertex colors.

### Recommendation for the real build

- Use this architecture (baked base + additive shadow-mapped sun + sky fill) as
  the `SimulatorEnvironment` time-of-day mode; keep the two GLBs as the source
  assets and do the shade split at load.
- Tune the sun curve against the Daytime bake for the noon checkpoint; add 2–3
  additive point lights later if lamps should fade up dynamically.
- Optional: produce 1–2 extra bake keyframes (Blender round-trips the GLBs) and
  blend them as the *base* while the sun stays real-time — additive improvement,
  not a replacement.
