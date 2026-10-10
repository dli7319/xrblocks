/** @vitest-environment node */
/**
 * Brute-force descriptor matching: popcount correctness and the Lowe ratio
 * test.
 */
import {describe, expect, test} from 'vitest';
import {DESC_LEN} from './orb.js';
import {MATCH_RATIO, hamming, matchDescriptors} from './matching.js';

/** Deterministic LCG so every test run sees the same descriptors. */
function makeRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function randomRow(random) {
  const row = new Uint8Array(DESC_LEN);
  for (let i = 0; i < DESC_LEN; i++) row[i] = Math.floor(random() * 256);
  return row;
}

function stack(rows) {
  const out = new Uint8Array(rows.length * DESC_LEN);
  rows.forEach((row, i) => out.set(row, i * DESC_LEN));
  return out;
}

/** Copy `row` flipping exactly `count` distinct bits. */
function withFlippedBits(row, count, fromBit) {
  const copy = Uint8Array.from(row);
  for (let i = 0; i < count; i++) {
    const bit = (fromBit + i * 5) % (DESC_LEN * 8);
    copy[bit >> 3] ^= 1 << (bit & 7);
  }
  return copy;
}

/** Reference popcount straight from the bit string. */
function referenceHamming(a, b) {
  let distance = 0;
  for (let i = 0; i < DESC_LEN; i++) {
    let x = a[i] ^ b[i];
    while (x) {
      distance += x & 1;
      x >>= 1;
    }
  }
  return distance;
}

describe('matching', () => {
  test('hamming() counts differing bits on known pairs', () => {
    const zero = new Uint8Array(DESC_LEN);
    expect(hamming(zero, zero)).toBe(0);

    const b = new Uint8Array(DESC_LEN);
    b[0] = 0xff; // 8 bits
    b[1] = 0x0f; // 4 bits
    b[31] = 0x01; // 1 bit
    expect(hamming(zero, b)).toBe(13);
    expect(hamming(b, zero)).toBe(13);

    // Distance is symmetric and equals the reference popcount.
    const random = makeRandom(7);
    for (let trial = 0; trial < 20; trial++) {
      const a = randomRow(random);
      const c = randomRow(random);
      expect(hamming(a, c)).toBe(referenceHamming(a, c));
      expect(hamming(a, c)).toBe(hamming(c, a));
    }
  });

  test('matchDescriptors() pairs identical rows exactly', () => {
    const random = makeRandom(11);
    const queries = [
      randomRow(random),
      randomRow(random),
      randomRow(random),
      randomRow(random),
    ];
    const trains = [randomRow(random), randomRow(random), randomRow(random)];
    // Interleave exact copies among distractors.
    const trainRows = [
      trains[0],
      queries[0],
      trains[1],
      queries[1],
      trains[2],
      queries[2],
      queries[3],
    ];

    const matches = matchDescriptors(stack(queries), stack(trainRows));
    expect(matches).toHaveLength(4);
    for (const match of matches) {
      expect(match.distance).toBe(0);
      expect(Uint8Array.from(queries[match.queryIdx])).toEqual(
        trainRows[match.trainIdx]
      );
    }
    // Sorted ascending by distance.
    const distances = matches.map((m) => m.distance);
    expect(distances).toEqual([...distances].sort((a, b) => a - b));
  });

  test('ratio test keeps unambiguous matches and drops ambiguous ones', () => {
    const random = makeRandom(23);
    const query = randomRow(random);
    // Candidate 1: 0 bits away (perfect). Candidate 2: 96 bits away.
    const easy = [query, withFlippedBits(query, 96, 3)];
    const easyMatches = matchDescriptors(stack([query]), stack(easy));
    expect(easyMatches).toHaveLength(1);
    expect(easyMatches[0].trainIdx).toBe(0);
    expect(easyMatches[0].distance).toBe(0);

    // Candidates at distance 10 and 12 -> ratio 10/12 = 0.83 > 0.75: rejected.
    const ambiguousRows = [
      withFlippedBits(query, 10, 1),
      withFlippedBits(query, 12, 2),
    ];
    const ambiguousMatches = matchDescriptors(
      stack([query]),
      stack(ambiguousRows)
    );
    expect(ambiguousMatches).toHaveLength(0);

    // Same pair but below the ratio threshold (10 vs 40 bits): accepted.
    const clearRows = [
      withFlippedBits(query, 10, 1),
      withFlippedBits(query, 40, 7),
    ];
    const clearMatches = matchDescriptors(stack([query]), stack(clearRows));
    expect(clearMatches).toHaveLength(1);
    expect(clearMatches[0].distance).toBe(10);
  });

  test('custom ratio overrides the default and empty inputs return []', () => {
    expect(MATCH_RATIO).toBe(0.75);
    const random = makeRandom(31);
    const query = randomRow(random);
    const rows = [withFlippedBits(query, 10, 1), withFlippedBits(query, 12, 2)];
    expect(
      matchDescriptors(stack([query]), stack(rows), {ratio: 0.95})
    ).toHaveLength(1);
    expect(
      matchDescriptors(stack([query]), stack(rows), {ratio: 0.5})
    ).toHaveLength(0);

    expect(matchDescriptors(new Uint8Array(0), stack(rows))).toEqual([]);
    expect(matchDescriptors(stack([query]), new Uint8Array(0))).toEqual([]);
  });

  test('a lone train candidate always passes the ratio test', () => {
    const random = makeRandom(53);
    const query = randomRow(random);
    const alone = withFlippedBits(query, 60, 0);
    const matches = matchDescriptors(stack([query]), stack([alone]));
    expect(matches).toHaveLength(1);
    expect(matches[0].distance).toBe(60);
  });
});
