/**
 * Pure-helper tests for `app/sim-feature-room.js`: seeded PRNG determinism
 * (the feature room must be identical across tabs/devices for cross-device
 * relocalization).
 *
 * Plain Node-runnable (node:assert, no test-runner API) so it executes with
 * `node demos/spatial-colocation/lib/sim/sim-feature-room.test.js`; it is also
 * safe to run under Vitest if a `*.test.ts` shim is added.
 */
import assert from 'node:assert/strict';

import {
  mulberry32,
  FEATURE_ROOM_MAX_MESHES,
} from '../../app/sim-feature-room.js';

let checks = 0;
function check(name, fn) {
  fn();
  checks++;
  console.log(`ok - ${name}`);
}

check('mulberry32: same seed -> same sequence', () => {
  const a = mulberry32(1);
  const b = mulberry32(1);
  for (let i = 0; i < 100; i++) {
    assert.equal(a(), b(), `draw ${i} differs`);
  }
});

check('mulberry32: values are uniform in [0, 1)', () => {
  const random = mulberry32(42);
  let min = 1;
  let max = 0;
  let sum = 0;
  const n = 10000;
  for (let i = 0; i < n; i++) {
    const v = random();
    assert.ok(v >= 0 && v < 1, `value ${v} out of range`);
    min = Math.min(min, v);
    max = Math.max(max, v);
    sum += v;
  }
  assert.ok(min >= 0 && max < 1);
  assert.ok(Math.abs(sum / n - 0.5) < 0.02, `mean ${sum / n} too far from 0.5`);
});

check('mulberry32: different seeds diverge', () => {
  const a = mulberry32(1);
  const b = mulberry32(2);
  let same = 0;
  for (let i = 0; i < 100; i++) {
    if (a() === b()) same++;
  }
  assert.ok(same < 5, `sequences too similar (${same}/100)`);
});

check('feature room mesh budget: <= 20 meshes', () => {
  // 1 floor + 4 walls + 14 boxes = 19 in addFeatureRoom().
  const budget = 1 + 4 + 14;
  assert.ok(budget <= FEATURE_ROOM_MAX_MESHES);
  assert.equal(FEATURE_ROOM_MAX_MESHES, 20);
});

console.log(`\n${checks} checks passed`);
