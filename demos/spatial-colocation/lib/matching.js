/**
 * Brute-force matching of binary (ORB) descriptors.
 *
 * Implemented in plain JS with 16-bit word-level popcount so Node and the
 * browser behave identically - no dependency on `cv.BFMatcher`. Descriptors are
 * packed into `Uint16Array` word views once per call so the inner loop is a
 * branch-light 16-iteration popcount with zero per-pair allocations.
 */

import {DESC_LEN} from './orb.js';

/** Lowe ratio test threshold. */
export const MATCH_RATIO = 0.75;

const POPCOUNT = new Uint8Array(256);
for (let i = 0; i < 256; i++) {
  let v = i;
  let count = 0;
  while (v) {
    count += v & 1;
    v >>= 1;
  }
  POPCOUNT[i] = count;
}

/** Popcount of every 16-bit value; enables word-level distance computation. */
const POPCOUNT16 = new Uint8Array(65536);
for (let i = 0; i < 65536; i++) {
  POPCOUNT16[i] = POPCOUNT[i & 0xff] + POPCOUNT[(i >>> 8) & 0xff];
}

/** Number of 16-bit words per descriptor. */
const WORDS = DESC_LEN >>> 1;

/**
 * Hamming distance between two 32-byte descriptor views.
 *
 * @param {Uint8Array|Uint8ClampedArray} a
 * @param {Uint8Array|Uint8ClampedArray} b
 * @returns {number} number of differing bits
 */
export function hamming(a, b) {
  const n = Math.min(a.length, b.length);
  let distance = 0;
  let i = 0;
  for (; i + 1 < n; i += 2) {
    const wa = a[i] | (a[i + 1] << 8);
    const wb = b[i] | (b[i + 1] << 8);
    distance += POPCOUNT16[wa ^ wb];
  }
  if (i < n) distance += POPCOUNT[a[i] ^ b[i]];
  return distance;
}

/**
 * Pack raw descriptor bytes into a `Uint16Array` word view for repeated
 * matching (e.g. one live frame against a cached landmark matrix).
 *
 * @param {Uint8Array|Uint8ClampedArray} bytes `n * DESC_LEN` bytes
 * @param {number} [count] descriptor count (default: derived from length)
 * @returns {Uint16Array} `count * WORDS` words
 */
export function packDescriptors(
  bytes,
  count = Math.floor(bytes.length / DESC_LEN)
) {
  const words = new Uint16Array(count * WORDS);
  for (let i = 0; i < count; i++) {
    let src = i * DESC_LEN;
    const dst = i * WORDS;
    for (let w = 0; w < WORDS; w++) {
      words[dst + w] = bytes[src] | (bytes[src + 1] << 8);
      src += 2;
    }
  }
  return words;
}

/**
 * Brute-force Hamming matching over packed word views with Lowe's ratio test.
 *
 * @param {Uint16Array} wordsA packed query words (`countA * WORDS`)
 * @param {Uint16Array} wordsB packed train words (`countB * WORDS`)
 * @param {number} countA
 * @param {number} countB
 * @param {{ratio?: number}} [options]
 * @returns {Array<{queryIdx: number, trainIdx: number, distance: number}>}
 */
export function matchDescriptorsWords(
  wordsA,
  wordsB,
  countA,
  countB,
  {ratio = MATCH_RATIO} = {}
) {
  const matches = [];
  if (!wordsA || !wordsB || countA === 0 || countB === 0) return matches;

  for (let i = 0; i < countA; i++) {
    const baseA = i * WORDS;
    let bestIdx = -1;
    let best = Infinity;
    let second = Infinity;
    for (let j = 0; j < countB; j++) {
      const baseB = j * WORDS;
      let distance = 0;
      for (let w = 0; w < WORDS; w++) {
        distance += POPCOUNT16[wordsA[baseA + w] ^ wordsB[baseB + w]];
      }
      if (distance < best) {
        second = best;
        best = distance;
        bestIdx = j;
      } else if (distance < second) {
        second = distance;
      }
    }
    if (bestIdx < 0) continue;
    // Lowe ratio test: keep only if the best match is clearly better than the
    // runner-up. A single train candidate (second = Infinity) always passes.
    if (best < ratio * second) {
      matches.push({queryIdx: i, trainIdx: bestIdx, distance: best});
    }
  }

  matches.sort((a, b) => {
    if (a.distance !== b.distance) return a.distance - b.distance;
    return a.queryIdx - b.queryIdx;
  });
  return matches;
}

/**
 * Brute-force Hamming matching with Lowe's ratio test.
 *
 * @param {Uint8Array} descA query descriptors, `n * DESC_LEN` bytes
 * @param {Uint8Array} descB train descriptors, `m * DESC_LEN` bytes
 * @param {{ratio?: number}} [options]
 * @returns {Array<{queryIdx: number, trainIdx: number, distance: number}>} sorted ascending by distance
 */
export function matchDescriptors(descA, descB, {ratio = MATCH_RATIO} = {}) {
  const matches = [];
  if (!descA || !descB) return matches;
  const countA = Math.floor(descA.length / DESC_LEN);
  const countB = Math.floor(descB.length / DESC_LEN);
  if (countA === 0 || countB === 0) return matches;
  return matchDescriptorsWords(
    packDescriptors(descA, countA),
    packDescriptors(descB, countB),
    countA,
    countB,
    {ratio}
  );
}
