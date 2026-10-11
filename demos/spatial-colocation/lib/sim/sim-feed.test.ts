/** @vitest-environment node */
// Vitest only collects `*.test.ts` in this repo (see vitest.config.ts include);
// the real node:assert suite lives in `./sim-feed.test.js` and runs at import.
import {test} from 'vitest';

test('sim-feed pure helpers (node:assert suite)', async () => {
  await import('./sim-feed.test.js');
});
