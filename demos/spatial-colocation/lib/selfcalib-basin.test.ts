/** @vitest-environment node */
// Vitest only collects `*.test.ts` in this repo (see vitest.config.ts include);
// the real node:assert suite lives in `./selfcalib-basin.test.js`.
// Runs the reliable basin (±50% inits, must recover) — see that file's header
// for the measured full basin characterization.
import {test} from 'vitest';

process.env.BASIN_NEAR = process.env.BASIN_NEAR || '0.5';
process.env.BASIN_TRIALS = process.env.BASIN_TRIALS || '8';

test('OnlineCalibrator basin: ±50% inits recover the true parameters', async () => {
  await import('./selfcalib-basin.test.js');
}, 120000);
