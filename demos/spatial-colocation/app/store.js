// app/store.js
/**
 * IndexedDB persistence for serialized SpatialMaps.
 * DB: 'spatial-colocation-demo', object store 'maps' keyed by {name, ts, bytes}.
 */

const DB_NAME = 'spatial-colocation-demo';
const STORE = 'maps';
let dbPromise = null;

function openDb() {
  if (!dbPromise) {
    dbPromise = new Promise((resolve, reject) => {
      const req = indexedDB.open(DB_NAME, 1);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains(STORE)) {
          db.createObjectStore(STORE, {keyPath: 'name'});
        }
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () =>
        reject(req.error || new Error('IndexedDB open failed'));
    });
  }
  return dbPromise;
}

function tx(db, mode) {
  return db.transaction(STORE, mode).objectStore(STORE);
}

function reqToPromise(req) {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () =>
      reject(req.error || new Error('IndexedDB request failed'));
  });
}

/** Store a serialized map. `bytes` is a Uint8Array (copied so callers can reuse buffers). */
export async function saveMap(name, bytes) {
  const db = await openDb();
  const record = {
    name,
    ts: Date.now(),
    bytes: bytes instanceof Uint8Array ? bytes.slice() : new Uint8Array(bytes),
  };
  await reqToPromise(tx(db, 'readwrite').put(record));
  return record;
}

/** Load a serialized map; returns Uint8Array or null when absent. */
export async function loadMap(name) {
  const db = await openDb();
  const rec = await reqToPromise(tx(db, 'readonly').get(name));
  if (!rec) return null;
  return rec.bytes instanceof Uint8Array
    ? rec.bytes
    : new Uint8Array(rec.bytes);
}

/** List stored maps, newest first: [{name, ts, size}]. */
export async function listMaps() {
  const db = await openDb();
  const recs = await reqToPromise(tx(db, 'readonly').getAll());
  return recs
    .map((r) => ({
      name: r.name,
      ts: r.ts || 0,
      size: r.bytes ? r.bytes.byteLength || r.bytes.length || 0 : 0,
    }))
    .sort((a, b) => b.ts - a.ts);
}

/** Remove a stored map. */
export async function deleteMap(name) {
  const db = await openDb();
  await reqToPromise(tx(db, 'readwrite').delete(name));
}
