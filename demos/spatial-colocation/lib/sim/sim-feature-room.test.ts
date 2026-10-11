/** @vitest-environment node */
// Vitest only collects `*.test.ts` in this repo (see vitest.config.ts include);
// the real node:assert suite lives in `./sim-feature-room.test.js` and runs at import.
import {test} from 'vitest';

test('sim-feature-room pure helpers (node:assert suite)', async () => {
  await import('./sim-feature-room.test.js');
});
