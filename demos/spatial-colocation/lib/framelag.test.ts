/** @vitest-environment node */
// Vitest only collects `*.test.ts` in this repo (see vitest.config.ts include);
// the real node:assert suite lives in `./framelag.test.js` and runs at import
// (same pattern as lib/selfcalib.test.ts).
import {test} from 'vitest';

test('frame-lag estimation recovery (node:assert suite)', async () => {
  await import('./framelag.test.js');
}, 60000);
