/**
 * Multi-View Graph-Cut Texture Mapping (Waechter et al., ECCV 2014) and
 * Connected-Component Mesh Filtering for XR Circle-to-Digitize.
 *
 * Pure TypeScript / TypedArray implementation with zero DOM or Three.js runtime
 * dependencies so it executes inside the Web Worker without affecting 72 FPS.
 */

import type {ExtractedTSDFMesh, Vec3Tuple} from './tsdf_volume.js';

export interface TextureKeyframe {
  id: number;
  rgba: Uint8ClampedArray;
  width: number;
  height: number;
  cameraBinaryMask: Uint8Array;
  /** Distance in pixels to nearest background pixel (0..32). */
  maskDistField: Uint8Array;
  /** 4x4 column-major matrix: World -> RGB Camera Clip Space. */
  rgbClipFromWorldMatrix: Float32Array;
  cameraPos: Vec3Tuple;
  cameraForward: Vec3Tuple;
  /** Optional depth buffer in meters for z-buffer occlusion testing. */
  depthData?: Float32Array;
  depthWidth?: number;
  depthHeight?: number;
  depthViewMatrix?: Float32Array;
  depthProjectionMatrix?: Float32Array;
  normDepthBufferFromNormViewMatrix?: Float32Array;
}

export interface GraphCutTexturedMesh {
  positions: Float32Array;
  normals: Float32Array;
  uvs: Float32Array;
  colors: Float32Array;
  triangleCount: number;
  atlasRgba: Uint8ClampedArray;
  atlasWidth: number;
  atlasHeight: number;
  chartCount: number;
}

export interface DualTriangleEdge {
  triA: number;
  triB: number;
  v0Index: number;
  v1Index: number;
  midX: number;
  midY: number;
  midZ: number;
}

export interface TriangleAdjacencyGraph {
  triangleCount: number;
  weldedVertexCount: number;
  /** Length `triangleCount * 3`: welded vertex index for each triangle corner. */
  triWeldedVertices: Int32Array;
  edges: DualTriangleEdge[];
}

/**
 * Fast Dinic s-t Max-Flow / Min-Cut solver over directed graphs using flat TypedArrays.
 */
export class DinicMaxFlowSolver {
  private readonly maxNodes: number;
  private readonly head: Int32Array;
  private readonly level: Int32Array;
  private readonly iter: Int32Array;
  private readonly queue: Int32Array;

  private to: Int32Array;
  private next: Int32Array;
  private cap: Float32Array;
  private edgeCount = 0;

  constructor(maxNodes: number, initialMaxEdges = maxNodes * 8) {
    this.maxNodes = maxNodes;
    this.head = new Int32Array(maxNodes).fill(-1);
    this.level = new Int32Array(maxNodes);
    this.iter = new Int32Array(maxNodes);
    this.queue = new Int32Array(maxNodes);

    const capEdges = Math.max(64, initialMaxEdges * 2);
    this.to = new Int32Array(capEdges);
    this.next = new Int32Array(capEdges);
    this.cap = new Float32Array(capEdges);
  }

  reset(): void {
    this.head.fill(-1);
    this.edgeCount = 0;
  }

  private ensureEdgeCapacity(needed: number): void {
    if (this.edgeCount + needed <= this.to.length) return;
    const newLen = Math.max(this.to.length * 2, this.edgeCount + needed);
    const newTo = new Int32Array(newLen);
    const newNext = new Int32Array(newLen);
    const newCap = new Float32Array(newLen);
    newTo.set(this.to);
    newNext.set(this.next);
    newCap.set(this.cap);
    this.to = newTo;
    this.next = newNext;
    this.cap = newCap;
  }

  addDirectedEdge(
    u: number,
    v: number,
    forwardCap: number,
    backwardCap = 0
  ): void {
    this.ensureEdgeCapacity(2);
    const e0 = this.edgeCount++;
    this.to[e0] = v;
    this.cap[e0] = forwardCap;
    this.next[e0] = this.head[u];
    this.head[u] = e0;

    const e1 = this.edgeCount++;
    this.to[e1] = u;
    this.cap[e1] = backwardCap;
    this.next[e1] = this.head[v];
    this.head[v] = e1;
  }

  private bfs(s: number, t: number): boolean {
    this.level.fill(-1);
    let qHead = 0;
    let qTail = 0;
    this.level[s] = 0;
    this.queue[qTail++] = s;

    while (qHead < qTail) {
      const v = this.queue[qHead++];
      const nextLevel = this.level[v] + 1;
      for (let e = this.head[v]; e !== -1; e = this.next[e]) {
        if (this.cap[e] > 1e-6) {
          const u = this.to[e];
          if (this.level[u] < 0) {
            this.level[u] = nextLevel;
            this.queue[qTail++] = u;
          }
        }
      }
    }
    return this.level[t] >= 0;
  }

  private dfs(v: number, t: number, pushed: number): number {
    if (pushed <= 1e-6) return 0;
    if (v === t) return pushed;

    for (let e = this.iter[v]; e !== -1; e = this.next[e]) {
      this.iter[v] = e;
      const u = this.to[e];
      const rem = this.cap[e];
      if (rem <= 1e-6 || this.level[v] + 1 !== this.level[u]) continue;

      const tr = this.dfs(u, t, Math.min(pushed, rem));
      if (tr <= 1e-6) continue;

      this.cap[e] -= tr;
      this.cap[e ^ 1] += tr;
      return tr;
    }
    return 0;
  }

  maxFlow(s: number, t: number): number {
    let flow = 0;
    const INF = 1e20;
    while (this.bfs(s, t)) {
      this.iter.set(this.head);
      while (true) {
        const pushed = this.dfs(s, t, INF);
        if (pushed <= 1e-6) break;
        flow += pushed;
      }
    }
    return flow;
  }

  /**
   * After `maxFlow(s, t)` finishes, returns true if `node` lies on the source side
   * of the minimum s-t cut.
   */
  isReachableFromSource(node: number): boolean {
    return node >= 0 && node < this.maxNodes && this.level[node] >= 0;
  }
}

/**
 * Computes a fast 2-pass Chamfer / Manhattan distance field (in pixels, capped at `maxDist`)
 * inside a binary mask (`1` = foreground, `0` = background).
 */
export function computeMaskDistanceField(
  mask: Uint8Array,
  width: number,
  height: number,
  maxDist = 32
): Uint8Array {
  const dist = new Uint8Array(width * height);
  for (let i = 0; i < mask.length; i++) {
    dist[i] = mask[i] ? maxDist : 0;
  }

  // Forward pass
  for (let y = 0; y < height; y++) {
    const row = y * width;
    for (let x = 0; x < width; x++) {
      const idx = row + x;
      let d = dist[idx];
      if (d === 0) continue;
      if (x === 0 || y === 0 || x === width - 1 || y === height - 1) {
        dist[idx] = 1;
        continue;
      }
      const up = dist[idx - width] + 1;
      const left = dist[idx - 1] + 1;
      if (up < d) d = up;
      if (left < d) d = left;
      dist[idx] = d;
    }
  }

  // Backward pass
  for (let y = height - 2; y >= 1; y--) {
    const row = y * width;
    for (let x = width - 2; x >= 1; x--) {
      const idx = row + x;
      let d = dist[idx];
      if (d === 0) continue;
      const down = dist[idx + width] + 1;
      const right = dist[idx + 1] + 1;
      if (down < d) d = down;
      if (right < d) d = right;
      dist[idx] = d;
    }
  }

  return dist;
}

/**
 * Welds coincident triangle vertices and constructs the dual triangle adjacency graph.
 */
export function buildTriangleAdjacencyGraph(
  positions: Float32Array
): TriangleAdjacencyGraph {
  const triangleCount = (positions.length / 9) | 0;
  const triWeldedVertices = new Int32Array(triangleCount * 3);
  const vertexMap = new Map<string, number>();
  const weldedCoords: number[] = [];

  const quantScale = 10000; // 0.1 mm precision
  for (let i = 0; i < triangleCount * 3; i++) {
    const px = positions[i * 3];
    const py = positions[i * 3 + 1];
    const pz = positions[i * 3 + 2];
    const qx = Math.round(px * quantScale);
    const qy = Math.round(py * quantScale);
    const qz = Math.round(pz * quantScale);
    const key = `${qx},${qy},${qz}`;
    let vIdx = vertexMap.get(key);
    if (vIdx === undefined) {
      vIdx = (weldedCoords.length / 3) | 0;
      vertexMap.set(key, vIdx);
      weldedCoords.push(px, py, pz);
    }
    triWeldedVertices[i] = vIdx;
  }

  const edges: DualTriangleEdge[] = [];
  const halfEdgeMap = new Map<string, number>();

  for (let t = 0; t < triangleCount; t++) {
    const base = t * 3;
    const v0 = triWeldedVertices[base];
    const v1 = triWeldedVertices[base + 1];
    const v2 = triWeldedVertices[base + 2];

    const pairs: [number, number][] = [
      [v0, v1],
      [v1, v2],
      [v2, v0],
    ];

    for (const [a, b] of pairs) {
      if (a === b) continue;
      const minV = a < b ? a : b;
      const maxV = a < b ? b : a;
      const edgeKey = `${minV}_${maxV}`;
      const existingTri = halfEdgeMap.get(edgeKey);
      if (existingTri === undefined) {
        halfEdgeMap.set(edgeKey, t);
      } else if (existingTri !== t) {
        const ax = weldedCoords[minV * 3];
        const ay = weldedCoords[minV * 3 + 1];
        const az = weldedCoords[minV * 3 + 2];
        const bx = weldedCoords[maxV * 3];
        const by = weldedCoords[maxV * 3 + 1];
        const bz = weldedCoords[maxV * 3 + 2];
        edges.push({
          triA: existingTri,
          triB: t,
          v0Index: minV,
          v1Index: maxV,
          midX: 0.5 * (ax + bx),
          midY: 0.5 * (ay + by),
          midZ: 0.5 * (az + bz),
        });
      }
    }
  }

  return {
    triangleCount,
    weldedVertexCount: (weldedCoords.length / 3) | 0,
    triWeldedVertices,
    edges,
  };
}

/**
 * Removes disconnected table/background shards by keeping only the primary connected
 * component of triangles closest to `targetCenter`.
 */
export function filterLargestConnectedMeshComponent(
  mesh: ExtractedTSDFMesh,
  targetCenter: Vec3Tuple
): ExtractedTSDFMesh {
  const triCount = mesh.triangleCount;
  if (triCount <= 1) return mesh;

  const adj = buildTriangleAdjacencyGraph(mesh.positions);
  const parent = new Int32Array(triCount);
  const rank = new Uint8Array(triCount);
  for (let i = 0; i < triCount; i++) parent[i] = i;

  const find = (x: number): number => {
    let root = x;
    while (parent[root] !== root) root = parent[root];
    let cur = x;
    while (cur !== root) {
      const nxt = parent[cur];
      parent[cur] = root;
      cur = nxt;
    }
    return root;
  };

  const union = (a: number, b: number): void => {
    const ra = find(a);
    const rb = find(b);
    if (ra === rb) return;
    if (rank[ra] < rank[rb]) {
      parent[ra] = rb;
    } else if (rank[ra] > rank[rb]) {
      parent[rb] = ra;
    } else {
      parent[rb] = ra;
      rank[ra]++;
    }
  };

  // Connect triangles that share any welded vertex
  const vertexFirstTri = new Int32Array(adj.weldedVertexCount).fill(-1);
  for (let t = 0; t < triCount; t++) {
    for (let c = 0; c < 3; c++) {
      const v = adj.triWeldedVertices[t * 3 + c];
      const first = vertexFirstTri[v];
      if (first === -1) {
        vertexFirstTri[v] = t;
      } else {
        union(first, t);
      }
    }
  }

  interface CompStats {
    count: number;
    sumX: number;
    sumY: number;
    sumZ: number;
  }
  const components = new Map<number, CompStats>();
  const pos = mesh.positions;

  for (let t = 0; t < triCount; t++) {
    const r = find(t);
    const base = t * 9;
    const cx = (pos[base] + pos[base + 3] + pos[base + 6]) / 3;
    const cy = (pos[base + 1] + pos[base + 4] + pos[base + 7]) / 3;
    const cz = (pos[base + 2] + pos[base + 5] + pos[base + 8]) / 3;
    let st = components.get(r);
    if (!st) {
      st = {count: 0, sumX: 0, sumY: 0, sumZ: 0};
      components.set(r, st);
    }
    st.count++;
    st.sumX += cx;
    st.sumY += cy;
    st.sumZ += cz;
  }

  if (components.size <= 1) {
    return mesh;
  }

  let bestRoot = -1;
  let bestScore = -Infinity;
  for (const [root, st] of components.entries()) {
    const mx = st.sumX / st.count;
    const my = st.sumY / st.count;
    const mz = st.sumZ / st.count;
    const dist = Math.hypot(
      mx - targetCenter.x,
      my - targetCenter.y,
      mz - targetCenter.z
    );
    const score = st.count / (1.0 + dist * 6.0);
    if (score > bestScore) {
      bestScore = score;
      bestRoot = root;
    }
  }

  const keptStats = components.get(bestRoot);
  if (!keptStats || keptStats.count === triCount) {
    return mesh;
  }

  const keptCount = keptStats.count;
  const outPositions = new Float32Array(keptCount * 9);
  const outNormals = new Float32Array(keptCount * 9);
  const outColors = new Float32Array(keptCount * 9);

  let dst = 0;
  for (let t = 0; t < triCount; t++) {
    if (find(t) !== bestRoot) continue;
    const src = t * 9;
    for (let k = 0; k < 9; k++) {
      outPositions[dst + k] = mesh.positions[src + k];
      outNormals[dst + k] = mesh.normals[src + k];
      outColors[dst + k] = mesh.colors[src + k];
    }
    dst += 9;
  }

  return {
    positions: outPositions,
    normals: outNormals,
    colors: outColors,
    triangleCount: keptCount,
  };
}

/**
 * Extrudes clean interior foreground RGB colors outward by `paddingPx` (default 32 px)
 * across the 2D SAM silhouette boundary using a linear-time multi-source BFS.
 *
 * First erodes the raw mask by 1 pixel to strip antialiased boundary pixels that mix
 * object and table/background colors, then flood-fills outward so any 3D mesh triangle
 * near the silhouette samples pure object color.
 */
export function padKeyframeForegroundRgba(
  rgba: Uint8ClampedArray,
  mask: Uint8Array,
  width: number,
  height: number,
  paddingPx = 32
): Uint8ClampedArray {
  const N = width * height;
  const out = new Uint8ClampedArray(rgba);

  // 1. Erode mask by 1px so mixed/antialiased boundary pixels are replaced by pure interior colors
  const seedMask = new Uint8Array(N);
  let interiorCount = 0;
  for (let y = 1; y < height - 1; y++) {
    const row = y * width;
    for (let x = 1; x < width - 1; x++) {
      const i = row + x;
      if (
        mask[i] &&
        mask[i - 1] &&
        mask[i + 1] &&
        mask[i - width] &&
        mask[i + width]
      ) {
        seedMask[i] = 1;
        interiorCount++;
      }
    }
  }

  const activeMask = interiorCount >= 16 ? seedMask : new Uint8Array(mask);
  const dist = new Uint16Array(N);
  const queue = new Int32Array(N);
  let head = 0;
  let tail = 0;

  for (let y = 0; y < height; y++) {
    const row = y * width;
    for (let x = 0; x < width; x++) {
      const i = row + x;
      if (!activeMask[i]) {
        dist[i] = 0xffff;
        continue;
      }
      dist[i] = 0;
      const isBoundary =
        (x > 0 && !activeMask[i - 1]) ||
        (x < width - 1 && !activeMask[i + 1]) ||
        (y > 0 && !activeMask[i - width]) ||
        (y < height - 1 && !activeMask[i + width]);
      if (isBoundary) {
        queue[tail++] = i;
      }
    }
  }

  // 2. Multi-source BFS outward up to `paddingPx` pixels
  while (head < tail) {
    const cur = queue[head++];
    const curDist = dist[cur];
    if (curDist >= paddingPx) continue;

    const nextDist = curDist + 1;
    const x = cur % width;
    const y = (cur / width) | 0;

    const neighbors: number[] = [];
    if (x > 0) neighbors.push(cur - 1);
    if (x < width - 1) neighbors.push(cur + 1);
    if (y > 0) neighbors.push(cur - width);
    if (y < height - 1) neighbors.push(cur + width);

    for (const nIdx of neighbors) {
      if (dist[nIdx] !== 0xffff) continue;
      dist[nIdx] = nextDist;

      // Average all already-resolved 4-neighbors at distance < nextDist
      const nx = nIdx % width;
      const ny = (nIdx / width) | 0;
      let rSum = 0;
      let gSum = 0;
      let bSum = 0;
      let count = 0;

      if (nx > 0 && dist[nIdx - 1] < nextDist) {
        const p = (nIdx - 1) * 4;
        rSum += out[p];
        gSum += out[p + 1];
        bSum += out[p + 2];
        count++;
      }
      if (nx < width - 1 && dist[nIdx + 1] < nextDist) {
        const p = (nIdx + 1) * 4;
        rSum += out[p];
        gSum += out[p + 1];
        bSum += out[p + 2];
        count++;
      }
      if (ny > 0 && dist[nIdx - width] < nextDist) {
        const p = (nIdx - width) * 4;
        rSum += out[p];
        gSum += out[p + 1];
        bSum += out[p + 2];
        count++;
      }
      if (ny < height - 1 && dist[nIdx + width] < nextDist) {
        const p = (nIdx + width) * 4;
        rSum += out[p];
        gSum += out[p + 1];
        bSum += out[p + 2];
        count++;
      }

      const dstP = nIdx * 4;
      if (count > 0) {
        out[dstP] = (rSum / count + 0.5) | 0;
        out[dstP + 1] = (gSum / count + 0.5) | 0;
        out[dstP + 2] = (bSum / count + 0.5) | 0;
        out[dstP + 3] = 255;
      } else {
        const srcP = cur * 4;
        out[dstP] = out[srcP];
        out[dstP + 1] = out[srcP + 1];
        out[dstP + 2] = out[srcP + 2];
        out[dstP + 3] = 255;
      }
      queue[tail++] = nIdx;
    }
  }

  return out;
}

function computeOutsideMaskDistanceField(
  mask: Uint8Array,
  width: number,
  height: number,
  maxDist = 16
): Uint8Array {
  const N = width * height;
  const dist = new Uint8Array(N);
  const queue = new Int32Array(N);
  let head = 0;
  let tail = 0;

  for (let y = 0; y < height; y++) {
    const row = y * width;
    for (let x = 0; x < width; x++) {
      const i = row + x;
      if (mask[i]) {
        dist[i] = 0;
        const isEdge =
          (x > 0 && !mask[i - 1]) ||
          (x < width - 1 && !mask[i + 1]) ||
          (y > 0 && !mask[i - width]) ||
          (y < height - 1 && !mask[i + width]);
        if (isEdge) {
          queue[tail++] = i;
        }
      } else {
        dist[i] = maxDist;
      }
    }
  }

  while (head < tail) {
    const cur = queue[head++];
    const d = dist[cur];
    if (d + 1 >= maxDist) continue;
    const nd = d + 1;
    const x = cur % width;
    const y = (cur / width) | 0;

    if (x > 0 && dist[cur - 1] > nd) {
      dist[cur - 1] = nd;
      queue[tail++] = cur - 1;
    }
    if (x < width - 1 && dist[cur + 1] > nd) {
      dist[cur + 1] = nd;
      queue[tail++] = cur + 1;
    }
    if (y > 0 && dist[cur - width] > nd) {
      dist[cur - width] = nd;
      queue[tail++] = cur - width;
    }
    if (y < height - 1 && dist[cur + width] > nd) {
      dist[cur + width] = nd;
      queue[tail++] = cur + width;
    }
  }

  return dist;
}

function projectPointToKeyframeUv(
  wx: number,
  wy: number,
  wz: number,
  clipFromWorld: Float32Array,
  outUv: Float32Array
): boolean {
  const cw =
    clipFromWorld[3] * wx +
    clipFromWorld[7] * wy +
    clipFromWorld[11] * wz +
    clipFromWorld[15];
  if (cw <= 1e-4) return false;
  const invW = 1.0 / cw;
  const ndcX =
    (clipFromWorld[0] * wx +
      clipFromWorld[4] * wy +
      clipFromWorld[8] * wz +
      clipFromWorld[12]) *
    invW;
  const ndcY =
    (clipFromWorld[1] * wx +
      clipFromWorld[5] * wy +
      clipFromWorld[9] * wz +
      clipFromWorld[13]) *
    invW;

  const u = 0.5 * (ndcX + 1.0);
  const v = 1.0 - 0.5 * (ndcY + 1.0);
  outUv[0] = u;
  outUv[1] = v;
  return u >= 0.005 && u <= 0.995 && v >= 0.005 && v <= 0.995;
}

function isDepthConsistentInKeyframe(
  wx: number,
  wy: number,
  wz: number,
  kf: TextureKeyframe
): boolean {
  const V = kf.depthViewMatrix;
  const P = kf.depthProjectionMatrix;
  const depthData = kf.depthData;
  const dw = kf.depthWidth ?? 0;
  const dh = kf.depthHeight ?? 0;
  if (!V || !P || !depthData || dw <= 0 || dh <= 0) {
    return true;
  }

  const vx = V[0] * wx + V[4] * wy + V[8] * wz + V[12];
  const vy = V[1] * wx + V[5] * wy + V[9] * wz + V[13];
  const vz = V[2] * wx + V[6] * wy + V[10] * wz + V[14];
  const zCam = -vz;
  if (zCam <= 0.05) return false;

  const cw = P[3] * vx + P[7] * vy + P[11] * vz + P[15];
  if (cw <= 1e-5) return false;
  const clipX = (P[0] * vx + P[4] * vy + P[8] * vz + P[12]) / cw;
  const clipY = (P[1] * vx + P[5] * vy + P[9] * vz + P[13]) / cw;
  if (clipX < -1.0 || clipX > 1.0 || clipY < -1.0 || clipY > 1.0) {
    return false;
  }

  const u = 0.5 * (clipX + 1.0);
  const v = 0.5 * (clipY + 1.0);
  let normBx = u;
  let normBy = 1.0 - v;
  const normM = kf.normDepthBufferFromNormViewMatrix;
  if (normM) {
    const topV = 1.0 - v;
    normBx = normM[0] * u + normM[4] * topV + normM[12];
    normBy = normM[1] * u + normM[5] * topV + normM[13];
  }

  const dx = Math.round(normBx * (dw - 1));
  const dy = Math.round(normBy * (dh - 1));
  if (dx < 0 || dx >= dw || dy < 0 || dy >= dh) {
    return false;
  }

  const dMeasured = depthData[dy * dw + dx];
  if (dMeasured <= 0) return true;
  // Reject if the keyframe depth map shows a closer occluder or a mismatched surface (> 5.5cm)
  return Math.abs(dMeasured - zCam) <= 0.055;
}

function sampleRgbaNormalized(
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
  u: number,
  v: number,
  outRgb: Float32Array
): void {
  const px = Math.max(0, Math.min(width - 1, Math.floor(u * width)));
  const py = Math.max(0, Math.min(height - 1, Math.floor(v * height)));
  const p = (py * width + px) * 4;
  outRgb[0] = rgba[p] / 255.0;
  outRgb[1] = rgba[p + 1] / 255.0;
  outRgb[2] = rgba[p + 2] / 255.0;
}

/**
 * Runs MRF Graph-Cut (Alpha-Expansion) view selection over the triangle mesh,
 * packs the active keyframes into a high-resolution UV texture atlas with 32px
 * silhouette foreground color extrusion, and performs seam color leveling.
 */
export function computeGraphCutTextureAtlas(
  mesh: ExtractedTSDFMesh,
  keyframes: TextureKeyframe[]
): GraphCutTexturedMesh {
  const triCount = mesh.triangleCount;
  const K = keyframes.length;

  if (triCount === 0 || K === 0) {
    const emptyAtlas = new Uint8ClampedArray(4 * 4 * 4).fill(255);
    return {
      positions: mesh.positions,
      normals: mesh.normals,
      uvs: new Float32Array(triCount * 6),
      colors: mesh.colors,
      triangleCount: triCount,
      atlasRgba: emptyAtlas,
      atlasWidth: 4,
      atlasHeight: 4,
      chartCount: 0,
    };
  }

  // Precompute 32px foreground-extruded RGBA and 16px outside-mask halo distance per keyframe
  const paddedRgbaList: Uint8ClampedArray[] = new Array(K);
  const outsideDistList: Uint8Array[] = new Array(K);
  for (let k = 0; k < K; k++) {
    const kf = keyframes[k];
    paddedRgbaList[k] = padKeyframeForegroundRgba(
      kf.rgba,
      kf.cameraBinaryMask,
      kf.width,
      kf.height,
      32
    );
    outsideDistList[k] = computeOutsideMaskDistanceField(
      kf.cameraBinaryMask,
      kf.width,
      kf.height,
      16
    );
  }

  const adj = buildTriangleAdjacencyGraph(mesh.positions);
  const dataCosts = new Float32Array(triCount * K);
  const tempUv0 = new Float32Array(2);
  const tempUv1 = new Float32Array(2);
  const tempUv2 = new Float32Array(2);
  const tempUvC = new Float32Array(2);

  const pos = mesh.positions;
  const norm = mesh.normals;
  const OCCLUDED_COST = 25.0;

  // 1. Evaluate Data Term E_data(i, k) for every triangle i and keyframe k
  for (let i = 0; i < triCount; i++) {
    const base = i * 9;
    const v0x = pos[base],
      v0y = pos[base + 1],
      v0z = pos[base + 2];
    const v1x = pos[base + 3],
      v1y = pos[base + 4],
      v1z = pos[base + 5];
    const v2x = pos[base + 6],
      v2y = pos[base + 7],
      v2z = pos[base + 8];

    const cx = (v0x + v1x + v2x) / 3;
    const cy = (v0y + v1y + v2y) / 3;
    const cz = (v0z + v1z + v2z) / 3;

    let nx = (norm[base] + norm[base + 3] + norm[base + 6]) / 3;
    let ny = (norm[base + 1] + norm[base + 4] + norm[base + 7]) / 3;
    let nz = (norm[base + 2] + norm[base + 5] + norm[base + 8]) / 3;
    const nLen = Math.hypot(nx, ny, nz);
    if (nLen > 1e-6) {
      nx /= nLen;
      ny /= nLen;
      nz /= nLen;
    }

    for (let k = 0; k < K; k++) {
      const kf = keyframes[k];
      const C = kf.rgbClipFromWorldMatrix;

      const ok0 = projectPointToKeyframeUv(v0x, v0y, v0z, C, tempUv0);
      const ok1 = projectPointToKeyframeUv(v1x, v1y, v1z, C, tempUv1);
      const ok2 = projectPointToKeyframeUv(v2x, v2y, v2z, C, tempUv2);
      const okC = projectPointToKeyframeUv(cx, cy, cz, C, tempUvC);

      if (!ok0 || !ok1 || !ok2 || !okC) {
        dataCosts[i * K + k] = OCCLUDED_COST;
        continue;
      }

      // Vector from triangle centroid to camera
      const dx = kf.cameraPos.x - cx;
      const dy = kf.cameraPos.y - cy;
      const dz = kf.cameraPos.z - cz;
      const dist = Math.hypot(dx, dy, dz);
      const cosTheta = dist > 1e-5 ? (nx * dx + ny * dy + nz * dz) / dist : 0.0;

      if (cosTheta <= 0.02) {
        dataCosts[i * K + k] = OCCLUDED_COST - cosTheta;
        continue;
      }

      // Check 3D depth consistency if keyframe carries a depth map
      if (!isDepthConsistentInKeyframe(cx, cy, cz, kf)) {
        dataCosts[i * K + k] = OCCLUDED_COST * 0.9 - cosTheta * 0.2;
        continue;
      }

      const W = kf.width;
      const H = kf.height;
      const pxC = Math.min(W - 1, Math.floor(tempUvC[0] * W));
      const pyC = Math.min(H - 1, Math.floor(tempUvC[1] * H));
      const px0 = Math.min(W - 1, Math.floor(tempUv0[0] * W));
      const py0 = Math.min(H - 1, Math.floor(tempUv0[1] * H));
      const px1 = Math.min(W - 1, Math.floor(tempUv1[0] * W));
      const py1 = Math.min(H - 1, Math.floor(tempUv1[1] * H));
      const px2 = Math.min(W - 1, Math.floor(tempUv2[0] * W));
      const py2 = Math.min(H - 1, Math.floor(tempUv2[1] * H));

      const outDistField = outsideDistList[k];
      const outDistC = outDistField[pyC * W + pxC];
      const maxOutDist = Math.max(
        outDistC,
        outDistField[py0 * W + px0],
        outDistField[py1 * W + px1],
        outDistField[py2 * W + px2]
      );

      // Only mark as occluded if triangle projects > 12px outside the 2D SAM mask.
      // Triangles within 12px of the silhouette safely sample the 32px BFS-extruded foreground colors!
      if (maxOutDist >= 12) {
        dataCosts[i * K + k] = OCCLUDED_COST * 0.85 - cosTheta * 0.5;
        continue;
      }

      const distEdge = kf.maskDistField[pyC * W + pxC] / 32.0;
      const haloPenalty = maxOutDist > 0 ? (maxOutDist / 12.0) * 0.35 : 0.0;

      // 2D projected triangle area reward (prefers closer / higher-res views)
      const area2D = Math.abs(
        (tempUv1[0] - tempUv0[0]) * (tempUv2[1] - tempUv0[1]) -
          (tempUv2[0] - tempUv0[0]) * (tempUv1[1] - tempUv0[1])
      );
      const areaWeight = Math.min(1.0, Math.sqrt(area2D * 800.0));

      // Slight preference for Keyframe 0 (user's primary circled view) to prevent unnecessary chart fragmentation
      const primaryBonus = k === 0 ? 0.08 : 0.0;

      const quality =
        Math.pow(cosTheta, 1.2) *
          (0.35 + 0.65 * distEdge) *
          (0.5 + 0.5 * areaWeight) +
        primaryBonus -
        haloPenalty;

      dataCosts[i * K + k] = Math.max(0.01, 1.5 - quality);
    }
  }

  // 2. Initialize labels with greedy best-view assignment
  const labels = new Int32Array(triCount);
  for (let i = 0; i < triCount; i++) {
    let bestK = 0;
    let bestCost = dataCosts[i * K];
    for (let k = 1; k < K; k++) {
      const c = dataCosts[i * K + k];
      if (c < bestCost) {
        bestCost = c;
        bestK = k;
      }
    }
    labels[i] = bestK;
  }

  // 3. Multi-Label Alpha-Expansion Graph-Cut Optimization (when K > 1)
  if (K > 1 && adj.edges.length > 0) {
    const numDualEdges = adj.edges.length;
    const maxNodes = triCount + numDualEdges + 2;
    const solver = new DinicMaxFlowSolver(
      maxNodes,
      triCount * 3 + numDualEdges * 4
    );

    const rgbA = new Float32Array(3);
    const rgbB = new Float32Array(3);
    const uvMidA = new Float32Array(2);
    const uvMidB = new Float32Array(2);
    const SMOOTH_WEIGHT = 1.45;

    const evalSmoothCost = (
      edge: DualTriangleEdge,
      labelA: number,
      labelB: number
    ): number => {
      if (labelA === labelB) return 0;
      const kfA = keyframes[labelA];
      const kfB = keyframes[labelB];
      const okA = projectPointToKeyframeUv(
        edge.midX,
        edge.midY,
        edge.midZ,
        kfA.rgbClipFromWorldMatrix,
        uvMidA
      );
      const okB = projectPointToKeyframeUv(
        edge.midX,
        edge.midY,
        edge.midZ,
        kfB.rgbClipFromWorldMatrix,
        uvMidB
      );
      if (!okA || !okB) {
        return SMOOTH_WEIGHT * 0.6;
      }
      sampleRgbaNormalized(
        paddedRgbaList[labelA],
        kfA.width,
        kfA.height,
        uvMidA[0],
        uvMidA[1],
        rgbA
      );
      sampleRgbaNormalized(
        paddedRgbaList[labelB],
        kfB.width,
        kfB.height,
        uvMidB[0],
        uvMidB[1],
        rgbB
      );
      const colorDiff = Math.hypot(
        rgbA[0] - rgbB[0],
        rgbA[1] - rgbB[1],
        rgbA[2] - rgbB[2]
      );
      return SMOOTH_WEIGHT * (0.15 + colorDiff);
    };

    for (let sweep = 0; sweep < 2; sweep++) {
      let changedInSweep = 0;
      for (let alpha = 0; alpha < K; alpha++) {
        solver.reset();
        const source = triCount + numDualEdges;
        const sink = source + 1;
        let auxNode = triCount;

        for (let i = 0; i < triCount; i++) {
          const curL = labels[i];
          const costCur = dataCosts[i * K + curL];
          const costAlpha = dataCosts[i * K + alpha];
          solver.addDirectedEdge(source, i, costCur, 0);
          solver.addDirectedEdge(i, sink, curL === alpha ? 1e12 : costAlpha, 0);
        }

        for (let eIdx = 0; eIdx < numDualEdges; eIdx++) {
          const edge = adj.edges[eIdx];
          const i = edge.triA;
          const j = edge.triB;
          const li = labels[i];
          const lj = labels[j];

          if (li === lj) {
            if (li !== alpha) {
              const w = evalSmoothCost(edge, li, alpha);
              solver.addDirectedEdge(i, j, w, w);
            }
          } else {
            const aux = auxNode++;
            const wpq = evalSmoothCost(edge, li, lj);
            const wpAlpha = evalSmoothCost(edge, li, alpha);
            const wqAlpha = evalSmoothCost(edge, lj, alpha);
            solver.addDirectedEdge(aux, sink, wpq, 0);
            solver.addDirectedEdge(i, aux, wpAlpha, wpAlpha);
            solver.addDirectedEdge(j, aux, wqAlpha, wqAlpha);
          }
        }

        solver.maxFlow(source, sink);

        for (let i = 0; i < triCount; i++) {
          if (labels[i] !== alpha && solver.isReachableFromSource(i)) {
            labels[i] = alpha;
            changedInSweep++;
          }
        }
      }
      if (changedInSweep === 0) break;
    }

    // Majority-vote smoothing pass to eliminate small 1-2 triangle sawtooth islands along seams
    const neighborLabels: number[][] = Array.from({length: triCount}, () => []);
    for (const edge of adj.edges) {
      neighborLabels[edge.triA].push(edge.triB);
      neighborLabels[edge.triB].push(edge.triA);
    }
    for (let pass = 0; pass < 2; pass++) {
      for (let i = 0; i < triCount; i++) {
        const nbrs = neighborLabels[i];
        if (nbrs.length >= 2) {
          const curL = labels[i];
          let sameCount = 0;
          let candidateL = -1;
          let candidateCount = 0;
          for (const n of nbrs) {
            const nl = labels[n];
            if (nl === curL) {
              sameCount++;
            } else if (nl === candidateL) {
              candidateCount++;
            } else if (candidateCount === 0) {
              candidateL = nl;
              candidateCount = 1;
            }
          }
          if (sameCount === 0 && candidateL >= 0 && candidateCount >= 2) {
            labels[i] = candidateL;
          }
        }
      }
    }
  }

  // 4. Pack Keyframes into a High-Resolution RGBA Texture Atlas with 32px Foreground Extrusion
  const tileW = keyframes[0].width;
  const tileH = keyframes[0].height;
  const gridCols = Math.ceil(Math.sqrt(K));
  const gridRows = Math.ceil(K / gridCols);
  const atlasWidth = gridCols * tileW;
  const atlasHeight = gridRows * tileH;
  const atlasRgba = new Uint8ClampedArray(atlasWidth * atlasHeight * 4);

  for (let k = 0; k < K; k++) {
    const paddedRgba = paddedRgbaList[k];
    const col = k % gridCols;
    const row = (k / gridCols) | 0;
    const offsetX = col * tileW;
    const offsetY = row * tileH;

    for (let y = 0; y < tileH; y++) {
      const srcRowStart = y * tileW * 4;
      const dstRowStart = ((offsetY + y) * atlasWidth + offsetX) * 4;
      atlasRgba.set(
        paddedRgba.subarray(srcRowStart, srcRowStart + tileW * 4),
        dstRowStart
      );
    }
  }

  // 5. Compute UVs and Seam-Leveled Vertex Colors
  const uvs = new Float32Array(triCount * 6);
  const leveledColors = new Float32Array(triCount * 9).fill(1.0);

  // Compute per-welded-vertex average RGB across all incident triangles to level seam exposure
  const vertSumR = new Float32Array(adj.weldedVertexCount);
  const vertSumG = new Float32Array(adj.weldedVertexCount);
  const vertSumB = new Float32Array(adj.weldedVertexCount);
  const vertCount = new Uint16Array(adj.weldedVertexCount);
  const cornerSampledRgb = new Float32Array(triCount * 9);
  const sampleRgb = new Float32Array(3);

  const usedLabels = new Set<number>();

  for (let i = 0; i < triCount; i++) {
    const k = labels[i];
    usedLabels.add(k);
    const kf = keyframes[k];
    const paddedRgba = paddedRgbaList[k];
    const col = k % gridCols;
    const row = (k / gridCols) | 0;
    const basePos = i * 9;
    const baseUv = i * 6;

    for (let c = 0; c < 3; c++) {
      const vx = pos[basePos + c * 3];
      const vy = pos[basePos + c * 3 + 1];
      const vz = pos[basePos + c * 3 + 2];

      projectPointToKeyframeUv(vx, vy, vz, kf.rgbClipFromWorldMatrix, tempUv0);
      const localU = Math.max(0.002, Math.min(0.998, tempUv0[0]));
      const localV = Math.max(0.002, Math.min(0.998, tempUv0[1]));

      uvs[baseUv + c * 2] = (col + localU) / gridCols;
      uvs[baseUv + c * 2 + 1] = (row + localV) / gridRows;

      sampleRgbaNormalized(
        paddedRgba,
        kf.width,
        kf.height,
        localU,
        localV,
        sampleRgb
      );
      cornerSampledRgb[basePos + c * 3] = sampleRgb[0];
      cornerSampledRgb[basePos + c * 3 + 1] = sampleRgb[1];
      cornerSampledRgb[basePos + c * 3 + 2] = sampleRgb[2];

      const wv = adj.triWeldedVertices[i * 3 + c];
      vertSumR[wv] += sampleRgb[0];
      vertSumG[wv] += sampleRgb[1];
      vertSumB[wv] += sampleRgb[2];
      vertCount[wv]++;
    }
  }

  // Seam leveling: compute gentle multiplicative color correction at shared seam vertices
  // so brightness/tint transitions smoothly across graph-cut chart boundaries
  for (let i = 0; i < triCount; i++) {
    const basePos = i * 9;
    for (let c = 0; c < 3; c++) {
      const wv = adj.triWeldedVertices[i * 3 + c];
      const cnt = vertCount[wv];
      if (cnt <= 1) {
        leveledColors[basePos + c * 3] = 1.0;
        leveledColors[basePos + c * 3 + 1] = 1.0;
        leveledColors[basePos + c * 3 + 2] = 1.0;
        continue;
      }
      const avgR = vertSumR[wv] / cnt;
      const avgG = vertSumG[wv] / cnt;
      const avgB = vertSumB[wv] / cnt;
      const curR = Math.max(0.08, cornerSampledRgb[basePos + c * 3]);
      const curG = Math.max(0.08, cornerSampledRgb[basePos + c * 3 + 1]);
      const curB = Math.max(0.08, cornerSampledRgb[basePos + c * 3 + 2]);

      leveledColors[basePos + c * 3] = Math.max(
        0.82,
        Math.min(1.18, avgR / curR)
      );
      leveledColors[basePos + c * 3 + 1] = Math.max(
        0.82,
        Math.min(1.18, avgG / curG)
      );
      leveledColors[basePos + c * 3 + 2] = Math.max(
        0.82,
        Math.min(1.18, avgB / curB)
      );
    }
  }

  return {
    positions: mesh.positions,
    normals: mesh.normals,
    uvs,
    colors: leveledColors,
    triangleCount: triCount,
    atlasRgba,
    atlasWidth,
    atlasHeight,
    chartCount: usedLabels.size,
  };
}
