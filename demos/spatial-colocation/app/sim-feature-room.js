// app/sim-feature-room.js
/**
 * Deterministic, ORB-friendly "feature room" for the XR Blocks simulator
 * testbed: textured quads/boxes arranged around the origin at 2-6 m so the
 * scene always has corners to track and the world-locked stability cube
 * visibly 'sticks' to a wall as the virtual head moves.
 *
 * Determinism matters: the same seed builds the same room on every device, so
 * two tabs (builder + relocalizer) map the SAME world — exactly what
 * cross-device relocalization needs. `?features=0` disables the room.
 *
 * Pure helper `mulberry32()` is unit-tested in `lib/sim/*.test.js`.
 */

import * as THREE from 'three';

/** Hard cap on meshes so the room stays light (spec: <= 20). */
export const FEATURE_ROOM_MAX_MESHES = 20;
/** Wall panels: large quads forming an open box at `WALL_DISTANCE_M`. */
const WALL_COUNT = 4;
const WALL_DISTANCE_M = 6;
const WALL_SIZE_M = 9;
/** Scatter boxes between these radii/heights around the origin. */
const BOX_COUNT = 14;
const BOX_MIN_RADIUS_M = 2;
const BOX_MAX_RADIUS_M = 5.5;
const BOX_MIN_Y_M = 0.6;
const BOX_MAX_Y_M = 2.6;
const TEXTURE_SIZE = 256;

/**
 * Deterministic seeded PRNG (mulberry32). Returns a function yielding
 * floats in [0, 1).
 *
 * @param {number} seed 32-bit seed
 * @returns {() => number} deterministic uniform [0, 1) generator
 */
export function mulberry32(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Procedural ORB-friendly texture: dense noise (many corners at every scale)
 * plus high-contrast shapes (circles, rects, triangles, lines).
 * Browser-only (uses a 2D canvas).
 *
 * @param {() => number} random seeded PRNG
 * @returns {HTMLCanvasElement} 256x256 texture canvas
 */
function makeFeatureTexture(random) {
  const canvas = document.createElement('canvas');
  canvas.width = TEXTURE_SIZE;
  canvas.height = TEXTURE_SIZE;
  const ctx = canvas.getContext('2d');
  const image = ctx.createImageData(TEXTURE_SIZE, TEXTURE_SIZE);
  for (let i = 0; i < image.data.length; i += 4) {
    const v = 40 + Math.floor(random() * 180);
    image.data[i] = v;
    image.data[i + 1] = v;
    image.data[i + 2] = v;
    image.data[i + 3] = 255;
  }
  ctx.putImageData(image, 0, 0);
  const colors = [
    '#e8e8e8',
    '#181818',
    '#d33b2c',
    '#2c6fd3',
    '#e0b42c',
    '#2fae60',
  ];
  for (let i = 0; i < 10; i++) {
    const x = random() * TEXTURE_SIZE;
    const y = random() * TEXTURE_SIZE;
    const s = 12 + random() * 60;
    ctx.fillStyle = colors[Math.floor(random() * colors.length)];
    const kind = Math.floor(random() * 4);
    if (kind === 0) {
      ctx.beginPath();
      ctx.arc(x, y, s / 2, 0, Math.PI * 2);
      ctx.fill();
    } else if (kind === 1) {
      ctx.fillRect(x - s / 2, y - s / 2, s, s * (0.4 + random() * 0.8));
    } else if (kind === 2) {
      ctx.beginPath();
      ctx.moveTo(x, y - s / 2);
      ctx.lineTo(x + s / 2, y + s / 2);
      ctx.lineTo(x - s / 2, y + s / 2);
      ctx.closePath();
      ctx.fill();
    } else {
      ctx.strokeStyle = ctx.fillStyle;
      ctx.lineWidth = 3 + random() * 5;
      ctx.beginPath();
      ctx.moveTo(x - s, y - s / 2);
      ctx.lineTo(x + s, y + s / 2);
      ctx.stroke();
    }
  }
  return canvas;
}

/**
 * Build and add the deterministic feature room to `scene`.
 *
 * @param {THREE.Scene} scene SDK scene (`xb.scene` / `xb.core.scene`)
 * @param {{seed?: number}} [options] deterministic seed (default 1)
 * @returns {THREE.Group} the added group (remove + dispose on teardown)
 */
export function addFeatureRoom(scene, {seed = 1} = {}) {
  const random = mulberry32(seed);
  const group = new THREE.Group();
  group.name = 'Feature Room';

  const textures = [];
  for (let i = 0; i < 6; i++) {
    const texture = new THREE.CanvasTexture(makeFeatureTexture(random));
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.anisotropy = 4;
    textures.push(texture);
  }
  const pickTexture = () =>
    textures[Math.floor(random() * textures.length) % textures.length];

  // Floor: noise texture so looking down still yields features.
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(WALL_SIZE_M, WALL_SIZE_M),
    new THREE.MeshPhongMaterial({map: textures[0]})
  );
  floor.rotation.x = -Math.PI / 2;
  floor.name = 'Feature Floor';
  group.add(floor);

  // Four wall panels forming an open box at WALL_DISTANCE_M.
  for (let i = 0; i < WALL_COUNT; i++) {
    const angle = (i / WALL_COUNT) * Math.PI * 2;
    const panel = new THREE.Mesh(
      new THREE.PlaneGeometry(WALL_SIZE_M, WALL_SIZE_M),
      new THREE.MeshPhongMaterial({map: pickTexture()})
    );
    panel.name = `Feature Wall ${i}`;
    panel.position.set(
      WALL_DISTANCE_M * Math.sin(angle),
      WALL_SIZE_M / 2 - 1,
      WALL_DISTANCE_M * Math.cos(angle)
    );
    panel.rotation.y = angle + Math.PI;
    group.add(panel);
  }

  // Scatter boxes at 2-5.5 m, varied heights/orientations.
  for (let i = 0; i < BOX_COUNT; i++) {
    const angle = random() * Math.PI * 2;
    const radius =
      BOX_MIN_RADIUS_M + random() * (BOX_MAX_RADIUS_M - BOX_MIN_RADIUS_M);
    const sx = 0.8 + random() * 1.4;
    const sy = 0.8 + random() * 1.4;
    const sz = 0.8 + random() * 1.4;
    const box = new THREE.Mesh(
      new THREE.BoxGeometry(sx, sy, sz),
      new THREE.MeshPhongMaterial({map: pickTexture()})
    );
    box.name = `Feature Box ${i}`;
    box.position.set(
      radius * Math.sin(angle),
      BOX_MIN_Y_M + random() * (BOX_MAX_Y_M - BOX_MIN_Y_M),
      radius * Math.cos(angle)
    );
    box.rotation.y = random() * Math.PI * 2;
    group.add(box);
  }

  if (group.children.length > FEATURE_ROOM_MAX_MESHES) {
    throw new Error('addFeatureRoom: mesh budget exceeded');
  }
  scene.add(group);
  return group;
}

/**
 * Dispose every geometry/material/texture owned by a feature room group.
 *
 * @param {THREE.Group} group value returned by `addFeatureRoom`
 */
export function disposeFeatureRoom(group) {
  if (!group) return;
  const textures = new Set();
  group.traverse((obj) => {
    if (obj.geometry) obj.geometry.dispose();
    const material = obj.material;
    for (const m of Array.isArray(material) ? material : [material]) {
      if (!m) continue;
      if (m.map) textures.add(m.map);
      m.dispose();
    }
  });
  for (const texture of textures) texture.dispose();
  group.removeFromParent();
}
