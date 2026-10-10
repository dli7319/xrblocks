/**
 * OpenCV namespace loader.
 *
 * Browser: waits for the global `cv` published by the CDN `<script>` tag
 * (`https://docs.opencv.org/4.x/opencv.js`).
 * Node: loads `@techstark/opencv-js` (root devDependency).
 *
 * Two pitfalls are handled here:
 * 1. OpenCV.js publishes a *thenable* readiness shim as `cv.then`. A promise
 *    that resolves with such an object re-adopts it forever (the shim resolves
 *    with the very object that owns `.then`), starving the event loop. The cv
 *    object therefore never travels through a promise until the shim has been
 *    removed - intermediate steps wrap it in a plain holder object.
 * 2. Bundler module pipelines stall on the 13 MB CommonJS `opencv.js` bundle,
 *    so under Node it is loaded with `createRequire` (fast plain `require`).
 */

const READY_TIMEOUT_MS = 30000;
const POLL_INTERVAL_MS = 15;

/**
 * Detect node: `process.versions.node` present and no `window`.
 * @returns {boolean}
 */
export function isNode() {
  return (
    typeof process !== 'undefined' &&
    !!process.versions?.node &&
    typeof window === 'undefined'
  );
}

/**
 * Loads the cv namespace. Browser: waits for global `cv` (CDN script). Node:
 * imports @techstark/opencv-js.
 * @returns {Promise<object>} the ready OpenCV namespace (`cv`)
 */
export async function loadCv() {
  const globalScope = typeof globalThis !== 'undefined' ? globalThis : {};
  let holder = asHolder(globalScope.cv);

  if (!holder) {
    const canImport =
      typeof process !== 'undefined' && !!process.versions?.node;
    holder = canImport
      ? await importNodeCv()
      : await waitForGlobalCv(READY_TIMEOUT_MS);
  }

  // Promise exports (e.g. newer builds) resolve through a holder as well so a
  // thenable cv is never adopted by the promise machinery.
  if (holder.cv instanceof Promise) {
    holder = await holder.cv.then((value) => ({cv: value}));
  }

  await waitUntilReady(holder.cv, READY_TIMEOUT_MS);
  disarmThenable(holder.cv);
  return holder.cv;
}

/**
 * Load `@techstark/opencv-js` under Node. Returns a holder `{cv}` so the
 * (thenable) cv object never passes through a promise.
 *
 * @returns {Promise<{cv: object}>}
 */
async function importNodeCv() {
  const {createRequire} = await import('node:module');
  const requireCv = createRequire(import.meta.url);
  const holder = asHolder(requireCv('@techstark/opencv-js'));
  if (!holder) {
    throw new Error(
      'loadCv: @techstark/opencv-js did not export a cv namespace'
    );
  }
  return holder;
}

/**
 * Poll for the CDN global `cv`.
 *
 * @param {number} timeoutMs
 * @returns {Promise<{cv: object}>} holder - the cv object itself must not pass
 *   through a promise (see module docblock)
 */
async function waitForGlobalCv(timeoutMs) {
  const globalScope = typeof globalThis !== 'undefined' ? globalThis : {};
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const cv = globalScope.cv;
    if (cv && (typeof cv === 'object' || typeof cv === 'function')) {
      return {cv};
    }
    if (Date.now() > deadline) {
      throw new Error(
        `loadCv: global \`cv\` not found after ${timeoutMs} ms (is the OpenCV.js script tag present?)`
      );
    }
    await sleep(POLL_INTERVAL_MS);
  }
}

/**
 * Wrap a value that looks like the cv namespace into `{cv}`, unwrapping module
 * namespaces / `{default: ...}` along the way. Returns null when the input is
 * not a usable candidate.
 *
 * @param {any} value
 * @returns {{cv: object}|null}
 */
function asHolder(value) {
  if (!value || (typeof value !== 'object' && typeof value !== 'function')) {
    return null;
  }
  if (typeof value.Mat === 'function') return {cv: value};
  if (
    value.default &&
    (typeof value.default === 'object' || typeof value.default === 'function')
  ) {
    return {cv: value.default};
  }
  // A bare cv namespace before initialization has no `Mat` yet - keep it only
  // when it does not look like a module namespace.
  if (value.Mat !== undefined || typeof value.then === 'function') {
    return {cv: value};
  }
  if (typeof value.cv === 'object' || typeof value.cv === 'function') {
    return {cv: value.cv};
  }
  return null;
}

/**
 * Wait until the runtime finished initializing (`cv.Mat` is published).
 *
 * @param {object} cv
 * @param {number} timeoutMs
 */
async function waitUntilReady(cv, timeoutMs) {
  if (!cv) throw new Error('loadCv: OpenCV namespace is undefined');
  if (typeof cv.Mat === 'function') return;

  // Fast path: register on the readiness shim when the build provides one.
  if (typeof cv.then === 'function') {
    try {
      cv.then(() => {});
    } catch {
      /* ignore - polling below still works */
    }
  }

  const deadline = Date.now() + timeoutMs;
  while (typeof cv.Mat !== 'function') {
    if (Date.now() > deadline) {
      throw new Error(`loadCv: OpenCV runtime not ready after ${timeoutMs} ms`);
    }
    await sleep(POLL_INTERVAL_MS);
  }
}

/**
 * Remove the thenable readiness shim so `cv` can be returned through a promise
 * without the promise machinery re-adopting it forever.
 *
 * @param {object} cv
 */
function disarmThenable(cv) {
  if (typeof cv.then !== 'function') return;
  try {
    delete cv.then;
  } catch {
    /* fall through to the assignment fallback below */
  }
  if (typeof cv.then === 'function') {
    try {
      cv.then = undefined;
    } catch {
      /* non-writable: handled below */
    }
  }
  if (typeof cv.then === 'function') {
    throw new Error(
      'loadCv: OpenCV namespace exposes a non-removable `.then` shim; refusing to return it through a promise'
    );
  }
}

/**
 * @param {number} ms
 * @returns {Promise<void>}
 */
function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
