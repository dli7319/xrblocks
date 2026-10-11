/** @vitest-environment node */
// Vitest only collects `*.test.ts` in this repo (see vitest.config.ts include);
// the real node:assert suite lives in `./selfcalib.test.js` and runs at import
// (same pattern as lib/sim/sim-feed.test.ts; the generous timeout only guards
// against slow CI — the suite takes ~2 s).
import {test} from 'vitest';

test('self-calibration synthetic recovery (node:assert suite)', async () => {
  await import('./selfcalib.test.js');
}, 60000);
