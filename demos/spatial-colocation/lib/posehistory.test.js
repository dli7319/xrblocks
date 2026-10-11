/**
 * PoseHistory tests (node:assert). Runs with
 * `node demos/spatial-colocation/lib/posehistory.test.js` and via the vitest
 * shim `posehistory.test.ts`.
 */
import assert from 'node:assert/strict';

import {PoseHistory} from './posehistory.js';

let checks = 0;
function check(name, fn) {
  fn();
  checks++;
  console.log(`ok - ${name}`);
}

check('nearest-sample picks the closest timestamp', () => {
  const h = new PoseHistory();
  h.push(100, 'a');
  h.push(200, 'b');
  h.push(300, 'c');
  assert.equal(h.sample(100), 'a');
  assert.equal(h.sample(149), 'a');
  assert.equal(h.sample(151), 'b');
  assert.equal(h.sample(301), 'c');
  assert.equal(h.sample(0), 'a');
});

check('out-of-order and stale pushes are dropped', () => {
  const h = new PoseHistory();
  h.push(200, 'b');
  h.push(100, 'stale'); // older than the tail: dropped
  h.push(200, 'dupe'); // equal timestamp: dropped
  assert.equal(h.sample(150), 'b');
  h.push(250, 'c');
  assert.equal(h.sample(249), 'c');
});

check('capacity bounds the ring', () => {
  const h = new PoseHistory({capacity: 3});
  for (let i = 0; i < 10; i++) h.push(i * 10, i);
  assert.equal(h.entries.length, 3);
  assert.equal(h.sample(90), 9);
});

check('empty history samples null', () => {
  assert.equal(new PoseHistory().sample(123), null);
});

console.log(`\n${checks} checks passed`);
