// app/net.js
/**
 * PeerJS networking for the spatial-colocation demo.
 *
 * Room model: mesh-star. The first peer to open `${room}-host` on the public broker is the
 * host; everyone else connects to it directly and the host relays messages. If the host
 * disappears, the next peer retries claiming the host id (basic takeover).
 *
 * Wire format over every DataConnection (binaryType 'arraybuffer'):
 *   0x01 + UTF-8 JSON   control/pose messages
 *   0x02 + u32be headerLen + UTF-8 JSON header + raw bytes   serialized map blob
 *   0x03 + u32be headerLen + UTF-8 JSON header + raw bytes   map blob CHUNK
 *     (chunk headers carry {transferId, index, total}; maps above 128 KiB are
 *      split into 60 KiB chunks because Chrome drops SCTP messages >~1 MiB)
 * Headers carry {type, to?} — `to` routes a frame through the host to one peer; absent
 * means broadcast. Control message types: hello, roster, map-request, map-data, pose, bye.
 */

const TYPE_JSON = 0x01;
const TYPE_MAP = 0x02;
const TYPE_MAP_CHUNK = 0x03;
const HOST_SUFFIX = 'host';
const MAX_BACKOFF_MS = 8000;
// Chrome's SCTP data-channel message cap is ~1 MiB and drops larger messages
// silently, so maps above the threshold travel as 60 KiB chunks.
const MAP_CHUNK_BYTES = 60 * 1024;
const MAP_CHUNK_THRESHOLD = 128 * 1024;
const MAP_TRANSFER_TTL_MS = 60000;

function encodeJson(msg) {
  const payload = new TextEncoder().encode(JSON.stringify(msg));
  const out = new Uint8Array(1 + payload.length);
  out[0] = TYPE_JSON;
  out.set(payload, 1);
  return out;
}

export function encodeMap(header, bytes, type = TYPE_MAP) {
  const h = new TextEncoder().encode(JSON.stringify(header));
  const out = new Uint8Array(1 + 4 + h.length + bytes.length);
  out[0] = type;
  new DataView(out.buffer).setUint32(1, h.length, false);
  out.set(h, 5);
  out.set(bytes, 5 + h.length);
  return out;
}

function toU8(data) {
  if (data instanceof Uint8Array) return data;
  if (data instanceof ArrayBuffer) return new Uint8Array(data);
  if (ArrayBuffer.isView(data)) {
    return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  }
  return null;
}

/** Decode a frame of any of the supported shapes. Returns {kind, header, bytes?} | null. */
export function decodeFrame(data) {
  if (typeof data === 'string') {
    try {
      return {kind: 'json', header: JSON.parse(data)};
    } catch (err) {
      return null;
    }
  }
  const u8 = toU8(data);
  if (!u8 || u8.length < 1) return null;
  const dec = new TextDecoder();
  if (u8[0] === TYPE_JSON) {
    try {
      return {kind: 'json', header: JSON.parse(dec.decode(u8.subarray(1)))};
    } catch (err) {
      return null;
    }
  }
  if (u8[0] === TYPE_MAP || u8[0] === TYPE_MAP_CHUNK) {
    if (u8.length < 5) return null;
    const hlen = new DataView(
      u8.buffer,
      u8.byteOffset,
      u8.byteLength
    ).getUint32(1, false);
    if (hlen > u8.length - 5) return null;
    let header;
    try {
      header = JSON.parse(dec.decode(u8.subarray(5, 5 + hlen)));
    } catch (err) {
      return null;
    }
    return {
      kind: u8[0] === TYPE_MAP ? 'map' : 'map-chunk',
      header,
      bytes: u8.slice(5 + hlen),
    };
  }
  return null;
}

function slug(s) {
  return (
    String(s || 'dev')
      .replace(/[^a-zA-Z0-9_-]/g, '')
      .slice(0, 12) || 'dev'
  );
}

/**
 * Reassemble map transfers from complete (0x02) and chunked (0x03) frames.
 * Returns `(frame, msg) => void`; completed transfers call `onComplete({from,
 * name, bytes})`. Pure and stateful-per-instance — unit-tested directly.
 */
export function createMapAssembler(onComplete) {
  const transfers = new Map(); // transferId -> {from, name, parts, received, total, ts}
  const completed = new Map(); // transferId -> completion ts (drops late duplicates)
  return function pushMapFrame(frame, msg) {
    if (frame.kind === 'map') {
      onComplete({from: msg.from, name: msg.name, bytes: frame.bytes});
      return;
    }
    const id = msg.transferId;
    if (!id) return;
    const now = Date.now();
    if (completed.has(id)) return;
    let t = transfers.get(id);
    if (!t) {
      t = {
        from: msg.from,
        name: msg.name,
        parts: [],
        received: 0,
        total: msg.total || 0,
        ts: now,
      };
      transfers.set(id, t);
      for (const [tid, other] of transfers) {
        if (now - other.ts > MAP_TRANSFER_TTL_MS) transfers.delete(tid);
      }
      for (const [tid, ts] of completed) {
        if (now - ts > MAP_TRANSFER_TTL_MS) completed.delete(tid);
      }
    }
    t.ts = now;
    if (!t.parts[msg.index]) {
      t.parts[msg.index] = frame.bytes;
      t.received += 1;
    }
    if (t.total && t.received >= t.total) {
      transfers.delete(id);
      completed.set(id, now);
      let totalLen = 0;
      for (const p of t.parts) totalLen += p ? p.length : 0;
      const out = new Uint8Array(totalLen);
      let off = 0;
      for (const p of t.parts) {
        if (!p) continue;
        out.set(p, off);
        off += p.length;
      }
      onComplete({from: t.from, name: t.name, bytes: out});
    }
  };
}

/**
 * Create a room client. Returns an emitter-ish handle:
 *   on('status'|'peers'|'pose'|'map-request'|'map-data', cb)
 *   sendPose(T_map_head, inliers)      broadcast presence
 *   requestMap()                        ask the room for a map
 *   sendMap(bytes, name, toPeerId?)     send a serialized map (broadcast or routed)
 *   setHasMap(bool) + announce()        refresh our hello (hasMap flag)
 *   peerId, isHost, disconnect()
 */
export function createNet({room, label, hasMap = false} = {}) {
  const listeners = new Map();
  const roster = new Map(); // peerId -> {label, hasMap}
  const hostConns = new Map(); // host side: peerId -> DataConnection
  let peer = null;
  let hostConn = null;
  let isHost = false;
  let closed = false;
  let hostAttempts = 0;
  let claimTimer = 0;
  let myHasMap = !!hasMap;
  let gen = 0; // bumped on every teardown so stale connection handlers go quiet

  const hostPeerId = `${room}-${HOST_SUFFIX}`;
  const directId = `${room}-${slug(label)}-${Math.random().toString(36).slice(2, 6)}`;
  let myId = directId;

  function emit(event, payload) {
    const cbs = listeners.get(event);
    if (!cbs) return;
    for (const cb of cbs) {
      try {
        cb(payload);
      } catch (err) {
        console.error('[net] listener error', err);
      }
    }
  }

  function setStatus(status, detail) {
    emit('status', {status, detail: detail || '', isHost, peerId: myId});
  }

  function emitPeers() {
    const list = [{peerId: myId, label, hasMap: myHasMap, self: true}];
    for (const [pid, r] of roster) {
      if (pid !== myId)
        list.push({peerId: pid, label: r.label, hasMap: r.hasMap, self: false});
    }
    emit('peers', list);
  }

  /** Complete (0x02) or reassemble (0x03 chunk) a map transfer, then emit it. */
  const handleMapFrame = createMapAssembler((m) => emit('map-data', m));

  function teardownPeer() {
    gen++;
    try {
      if (hostConn) hostConn.close();
    } catch (err) {
      /* noop */
    }
    hostConn = null;
    for (const [, conn] of hostConns) {
      try {
        conn.close();
      } catch (err) {
        /* noop */
      }
    }
    hostConns.clear();
    roster.clear();
    try {
      if (peer) peer.destroy();
    } catch (err) {
      /* noop */
    }
    peer = null;
    isHost = false;
  }

  function schedule(fn, ms) {
    clearTimeout(claimTimer);
    claimTimer = setTimeout(fn, ms);
  }

  // ---- host side -----------------------------------------------------------

  function hostRelay(frame, header, exceptPid, buf) {
    for (const [pid, conn] of hostConns) {
      if (pid === exceptPid) continue;
      if (header && header.to && header.to !== pid) continue;
      try {
        conn.send(buf);
      } catch (err) {
        /* drop */
      }
    }
  }

  function hostBroadcastSelf() {
    const frame = encodeJson({
      type: 'hello',
      peerId: hostPeerId,
      label,
      hasMap: myHasMap,
      host: true,
    });
    for (const [, conn] of hostConns) {
      try {
        conn.send(frame);
      } catch (err) {
        /* drop */
      }
    }
  }

  function hostRemovePeer(pid) {
    const conn = hostConns.get(pid);
    if (conn) {
      try {
        conn.close();
      } catch (err) {
        /* noop */
      }
    }
    hostConns.delete(pid);
    if (roster.delete(pid)) {
      const bye = encodeJson({type: 'bye', peerId: pid});
      hostRelay(null, null, pid, bye);
      emitPeers();
    }
  }

  function hostHandle(conn, data) {
    const frame = decodeFrame(data);
    if (!frame) return;
    const msg = frame.header || {};
    const buf = toU8(data);
    if (msg.type === 'hello') {
      if (msg.peerId) {
        hostConns.set(msg.peerId, conn);
        roster.set(msg.peerId, {label: msg.label, hasMap: !!msg.hasMap});
        // Welcome: who we are + current roster snapshot.
        conn.send(
          encodeJson({
            type: 'hello',
            peerId: hostPeerId,
            label,
            hasMap: myHasMap,
            host: true,
          })
        );
        conn.send(
          encodeJson({
            type: 'roster',
            peers: Array.from(roster, ([pid, r]) => ({
              peerId: pid,
              label: r.label,
              hasMap: r.hasMap,
            })),
          })
        );
        emitPeers();
        hostRelay(frame, msg, msg.peerId, buf);
      }
      return;
    }
    if (msg.type === 'bye') {
      hostRemovePeer(msg.peerId);
      return;
    }
    if (
      (frame.kind === 'map' || frame.kind === 'map-chunk') &&
      (!msg.to || msg.to === hostPeerId || msg.to === myId)
    ) {
      handleMapFrame(frame, msg);
    } else if (msg.type === 'map-request') {
      emit('map-request', msg);
    } else if (msg.type === 'pose') {
      emit('pose', msg);
    }
    hostRelay(frame, msg, msg.peerId, buf);
  }

  // ---- client side ---------------------------------------------------------

  function handleFromHost(data) {
    const frame = decodeFrame(data);
    if (!frame) return;
    const msg = frame.header || {};
    if (msg.to && msg.to !== myId) return;
    if (frame.kind === 'map' || frame.kind === 'map-chunk') {
      handleMapFrame(frame, msg);
      return;
    }
    switch (msg.type) {
      case 'hello':
        if (msg.peerId && msg.peerId !== hostPeerId) {
          roster.set(msg.peerId, {label: msg.label, hasMap: !!msg.hasMap});
          emitPeers();
        }
        break;
      case 'roster':
        for (const p of msg.peers || []) {
          if (p.peerId !== myId)
            roster.set(p.peerId, {label: p.label, hasMap: !!p.hasMap});
        }
        emitPeers();
        break;
      case 'pose':
        emit('pose', msg);
        break;
      case 'map-request':
        emit('map-request', msg);
        break;
      case 'map-data':
        emit('map-data', {from: msg.from, name: msg.name, bytes: frame.bytes});
        break;
      case 'bye':
        roster.delete(msg.peerId);
        emitPeers();
        break;
      default:
        break;
    }
  }

  // ---- bootstrap -----------------------------------------------------------

  function bindPeerCommon(p) {
    p.on('disconnected', () => {
      try {
        p.reconnect();
      } catch (err) {
        /* noop */
      }
    });
  }

  function bootClaim() {
    if (closed) return;
    teardownPeer();
    setStatus('connecting', 'claiming host');
    peer = new Peer(hostPeerId, {debug: 0});
    bindPeerCommon(peer);
    peer.on('open', (id) => {
      if (closed) return;
      isHost = true;
      myId = id;
      hostAttempts = 0;
      setStatus('host', 'you are the room host');
      emitPeers();
    });
    peer.on('error', (err) => {
      if (closed) return;
      const type = (err && err.type) || String(err);
      if (type === 'unavailable-id') {
        isHost = false;
        bootJoin();
        return;
      }
      setStatus('error', String(type));
      schedule(bootClaim, Math.min(MAX_BACKOFF_MS, 1000 * 2 ** hostAttempts++));
    });
    peer.on('connection', (conn) => {
      conn.on('data', (data) => hostHandle(conn, data));
      conn.on('error', () => {
        /* close follows */
      });
    });
  }

  function connectHost() {
    if (closed || isHost || !peer) return;
    const myGen = gen;
    setStatus('connecting', 'connecting to host');
    hostConn = peer.connect(hostPeerId, {reliable: true});
    hostConn.on('open', () => {
      hostAttempts = 0;
      setStatus('joined', `joined ${room}`);
      hostConn.send(
        encodeJson({type: 'hello', peerId: myId, label, hasMap: myHasMap})
      );
    });
    hostConn.on('data', (data) => handleFromHost(data));
    hostConn.on('close', () => {
      if (myGen !== gen) return; // superseded by teardown/takeover
      hostConn = null;
      if (closed) return;
      hostAttempts++;
      if (hostAttempts > 4) {
        // Host looks gone: try to take over the host id (basic takeover).
        setStatus('reconnecting', 'host lost — claiming');
        bootClaim();
      } else {
        setStatus('reconnecting', 'host connection lost');
        schedule(
          connectHost,
          Math.min(MAX_BACKOFF_MS, 500 * 2 ** hostAttempts)
        );
      }
    });
    hostConn.on('error', () => {
      /* close follows */
    });
  }

  function bootJoin() {
    teardownPeer();
    peer = new Peer(directId, {debug: 0});
    bindPeerCommon(peer);
    peer.on('open', (id) => {
      if (closed) return;
      myId = id;
      hostAttempts = 0;
      connectHost();
    });
    peer.on('error', (err) => {
      if (closed) return;
      const type = (err && err.type) || String(err);
      if (type === 'peer-unavailable') {
        // Host id not on the broker (it left): try claiming it.
        schedule(bootClaim, 500 * 2 ** hostAttempts++);
        return;
      }
      setStatus('error', String(type));
      schedule(bootClaim, Math.min(MAX_BACKOFF_MS, 1000 * 2 ** hostAttempts++));
    });
  }

  function start() {
    if (typeof Peer === 'undefined') {
      setStatus('error', 'PeerJS not loaded');
      return;
    }
    bootClaim();
  }

  function sendJson(msg) {
    const buf = encodeJson(msg);
    if (isHost) {
      hostRelay(null, msg, null, buf);
    } else if (hostConn && hostConn.open) {
      hostConn.send(buf);
    }
  }

  start();

  return {
    on(event, cb) {
      if (!listeners.has(event)) listeners.set(event, []);
      listeners.get(event).push(cb);
      return this;
    },
    get peerId() {
      return myId;
    },
    get isHost() {
      return isHost;
    },
    get peerCount() {
      return roster.size;
    },
    sendPose(T_map_head, inliers) {
      sendJson({
        type: 'pose',
        peerId: myId,
        label,
        T_map_head,
        inliers: inliers || 0,
        ts: Date.now(),
      });
    },
    requestMap() {
      sendJson({type: 'map-request', peerId: myId});
    },
    sendMap(bytes, name, toPeerId) {
      const header = {type: 'map-data', from: myId, name: name || 'map'};
      if (toPeerId) header.to = toPeerId;
      const send = (buf) => {
        if (isHost) {
          hostRelay(null, header, null, buf);
        } else if (hostConn && hostConn.open) {
          hostConn.send(buf);
        }
      };
      if (bytes.length <= MAP_CHUNK_THRESHOLD) {
        send(encodeMap(header, bytes));
        return;
      }
      // Chunked: one SCTP message per 60 KiB slice, reassembled by the receiver.
      const transferId = `${myId}-${Date.now().toString(36)}-${Math.random()
        .toString(36)
        .slice(2, 8)}`;
      const total = Math.ceil(bytes.length / MAP_CHUNK_BYTES);
      for (let i = 0; i < total; i++) {
        const start = i * MAP_CHUNK_BYTES;
        const end = Math.min(start + MAP_CHUNK_BYTES, bytes.length);
        send(
          encodeMap(
            {...header, type: 'map-chunk', transferId, index: i, total},
            bytes.subarray(start, end),
            TYPE_MAP_CHUNK
          )
        );
      }
    },
    setHasMap(v) {
      myHasMap = !!v;
    },
    announce() {
      if (isHost) {
        hostBroadcastSelf();
      } else {
        sendJson({type: 'hello', peerId: myId, label, hasMap: myHasMap});
      }
      emitPeers();
    },
    disconnect() {
      closed = true;
      clearTimeout(claimTimer);
      try {
        sendJson({type: 'bye', peerId: myId});
      } catch (err) {
        /* noop */
      }
      teardownPeer();
    },
  };
}
