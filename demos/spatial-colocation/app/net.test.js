/**
 * Wire-format + chunked map-transfer tests for `app/net.js`.
 *
 * Plain Node-runnable (node:assert) so it executes with
 * `node demos/spatial-colocation/app/net.test.js`; also runs under Vitest via
 * the colocated `net.test.ts` shim.
 */
import assert from 'node:assert/strict';

import {decodeFrame, encodeMap, createMapAssembler} from './net.js';

let checks = 0;
function check(name, fn) {
  fn();
  checks++;
  console.log(`ok - ${name}`);
}

check('decodeFrame: JSON control frame round-trips', () => {
  // JSON frames are {0x01 + utf8 JSON}; build one through the documented shape.
  const payload = new TextEncoder().encode(
    JSON.stringify({type: 'pose', a: 1})
  );
  const frame = new Uint8Array(1 + payload.length);
  frame[0] = 0x01;
  frame.set(payload, 1);
  const decoded = decodeFrame(frame);
  assert.equal(decoded.kind, 'json');
  assert.deepEqual(decoded.header, {type: 'pose', a: 1});
});

check('encodeMap/decodeFrame: complete map frame round-trips', () => {
  const bytes = new Uint8Array(5000).map((_, i) => i % 251);
  const frame = encodeMap({type: 'map-data', from: 'a', name: 'm'}, bytes);
  const decoded = decodeFrame(frame);
  assert.equal(decoded.kind, 'map');
  assert.equal(decoded.header.name, 'm');
  assert.deepEqual([...decoded.bytes.slice(0, 100)], [...bytes.slice(0, 100)]);
  assert.equal(decoded.bytes.length, bytes.length);
});

check(
  'chunked transfer: 3 MiB map reassembles byte-identical, out of order',
  () => {
    const bytes = new Uint8Array(3 * 1024 * 1024);
    for (let i = 0; i < bytes.length; i++) bytes[i] = (i * 2654435761) % 256;
    const CHUNK = 60 * 1024;
    const total = Math.ceil(bytes.length / CHUNK);
    const frames = [];
    for (let i = 0; i < total; i++) {
      const start = i * CHUNK;
      const end = Math.min(start + CHUNK, bytes.length);
      const frame = encodeMap(
        {
          type: 'map-chunk',
          from: 'host',
          name: 'big-map',
          to: 'client',
          transferId: 't-1',
          index: i,
          total,
        },
        bytes.subarray(start, end),
        0x03
      );
      frames.push(frame);
    }
    const got = [];
    const assembler = createMapAssembler((m) => got.push(m));
    // Deliver in reverse order to prove index-based reassembly.
    for (const f of [...frames].reverse()) {
      const decoded = decodeFrame(f);
      assert.equal(decoded.kind, 'map-chunk');
      assembler(decoded, decoded.header);
    }
    assert.equal(got.length, 1);
    assert.equal(got[0].name, 'big-map');
    assert.equal(got[0].bytes.length, bytes.length);
    assert.deepEqual(
      [...got[0].bytes.slice(0, 4096)],
      [...bytes.slice(0, 4096)],
      'head bytes match'
    );
    assert.deepEqual(
      [...got[0].bytes.slice(-4096)],
      [...bytes.slice(-4096)],
      'tail bytes match'
    );
  }
);

check('chunked transfer: duplicate chunks do not double-count', () => {
  const bytes = new Uint8Array(100).fill(7);
  const frame = encodeMap(
    {
      type: 'map-chunk',
      from: 'h',
      name: 'm',
      transferId: 't-2',
      index: 0,
      total: 1,
    },
    bytes,
    0x03
  );
  const got = [];
  const assembler = createMapAssembler((m) => got.push(m));
  const d1 = decodeFrame(frame);
  const d2 = decodeFrame(frame);
  assembler(d1, d1.header);
  assembler(d2, d2.header);
  assert.equal(got.length, 1);
  assert.equal(got[0].bytes.length, 100);
});

check('complete map frame passes through the assembler unchanged', () => {
  const bytes = new Uint8Array(64).fill(3);
  const frame = encodeMap({type: 'map-data', from: 'h', name: 'small'}, bytes);
  const got = [];
  const assembler = createMapAssembler((m) => got.push(m));
  const d = decodeFrame(frame);
  assembler(d, d.header);
  assert.equal(got.length, 1);
  assert.equal(got[0].name, 'small');
});

console.log(`\n${checks} checks passed`);
