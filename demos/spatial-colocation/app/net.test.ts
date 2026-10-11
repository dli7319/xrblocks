/** @vitest-environment node */
// Vitest only collects `*.test.ts` in this repo (see vitest.config.ts include);
// the real node:assert suite lives in `./net.test.js` and runs at import.
import {test} from 'vitest';

test('net wire format + chunked map transfer (node:assert suite)', async () => {
  await import('./net.test.js');
});
