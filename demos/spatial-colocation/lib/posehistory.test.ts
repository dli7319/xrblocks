/** @vitest-environment node */
// Vitest only collects `*.test.ts` in this repo (see vitest.config.ts include);
// the real node:assert suite lives in `./posehistory.test.js` and runs at import.
import {test} from 'vitest';

test('PoseHistory (node:assert suite)', async () => {
  await import('./posehistory.test.js');
});
