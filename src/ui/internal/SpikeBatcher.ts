import * as THREE from 'three';
import {mergeGeometries} from 'three/addons/utils/BufferGeometryUtils.js';

/**
 * SPIKE ONLY (branch spike/draw-call-floor): aggressive draw-call batching for
 * the plain MeshBasicMaterial UI family (icon SVG outlines, text outlines,
 * small paint quads, textured quads). Enabled with ?spikeBatch=1.
 *
 * Strategy: every rebuild, re-census the effectively-visible MeshBasicMaterial
 * meshes, merge them per (texture, depthWrite, toneMapped, blending) bucket
 * into baked world-space geometry carrying per-source vertex colors (existing
 * SVG vertex colors × material color), then hide the sources by moving them to
 * layer 5 (the camera renders layer 0 + XR eye layers 1/2 only; HitRegistry's
 * raycaster is patched to also see layer 5, so hit testing keeps working).
 *
 * Paint order: the scene's transparent meshes all tie at z≈0.95 and three
 * resolves order via float noise + object id, with glass panels interleaved
 * between content meshes. A merged batch can only occupy one paint slot, so in
 * 'exact' mode (default) sources are split into runs delimited by non-batchable
 * transparent objects (panels/reticles) and the whole transparent list is
 * re-ordered via renderOrder = painter rank (frozen at rebuild; fine for
 * static layouts). 'coarse' mode (?spikeOrder=coarse) merges each bucket into
 * a single draw regardless of interleaving — fewer draws, tint diffs where
 * glass layers painted between merged sources.
 *
 * Known compromises (spike):
 * - UICardEdge, BackfaceLayer and reticle backing meshes are hidden outright
 *   (measured 0 px from the default front view; backfaces only paint when a
 *   card is viewed from behind).
 * - Any color/transform change rebuilds the affected buckets.
 * - renderOrder ranks freeze the rebuild-time painter order.
 */

const SKIP_SUBTREES = ['XR Blocks Systems', 'Reticles', 'MainScript'];

type Source = {
  mesh: THREE.Mesh;
  key: string;
};

type ClipSnapshot = {nx: number; ny: number; nz: number; c: number}[];

/**
 * uikit rewrites shared clipping planes per draw (card overflow clipping), so
 * the planes only hold a given mesh's real clip state during its own draw.
 * Snapshot them in onBeforeRender and use the snapshot for batching.
 */
function ensureClipCapture(mesh: THREE.Mesh): void {
  if (mesh.userData.spikeClipWrapped) return;
  mesh.userData.spikeClipWrapped = true;
  const orig = mesh.onBeforeRender.bind(mesh);
  mesh.onBeforeRender = function (
    renderer: THREE.WebGLRenderer,
    scene: THREE.Scene,
    camera: THREE.Camera,
    geometry: THREE.BufferGeometry,
    material: THREE.Material,
    group: THREE.Group
  ) {
    const planes = (material as THREE.Material).clippingPlanes;
    if (planes && planes.length > 0) {
      this.userData.spikeClip = planes.map((p) => ({
        nx: p.normal.x,
        ny: p.normal.y,
        nz: p.normal.z,
        c: p.constant,
      })) satisfies ClipSnapshot;
    }
    orig.call(this, renderer, scene, camera, geometry, material, group);
  };
}

function clipSnapshotOf(mesh: THREE.Mesh): ClipSnapshot | null {
  const snap = mesh.userData.spikeClip as ClipSnapshot | undefined;
  if (snap && snap.length > 0) return snap;
  return null;
}

type SeqEntry = {
  m: THREE.Mesh;
  z: number;
  src: Source | null;
};

export class SpikeBatcher {
  private batchMeshes: THREE.Mesh[] = [];
  private hiddenForGood = new Set<THREE.Object3D>();
  private signature = '';
  private savedRenderOrder = new Map<number, number>();
  private firstPass = true;

  constructor(
    private readonly scene: THREE.Scene,
    private readonly camera?: THREE.Camera
  ) {}

  update(): void {
    this.applyHideRules();
    const sources = this.collectSources();
    const sig = sources
      .map((s) => s.key)
      .sort()
      .join(';');
    // First pass only installs clip capture; sources must draw once so the
    // wrappers can snapshot uikit's per-draw clipping planes.
    if (this.firstPass) {
      this.firstPass = false;
      return;
    }
    if (sig !== this.signature) {
      this.rebuild(sources);
      this.signature = sig;
    }
  }

  dispose(): void {
    for (const batch of this.batchMeshes) {
      batch.geometry.dispose();
      (batch.material as THREE.Material).dispose();
      batch.removeFromParent();
    }
    this.batchMeshes = [];
    this.restoreSources();
    for (const o of this.hiddenForGood) o.visible = true;
    this.hiddenForGood.clear();
    for (const [id, order] of this.savedRenderOrder) {
      const o = this.scene.getObjectById(id);
      if (o) o.renderOrder = order;
    }
    this.savedRenderOrder.clear();
    this.signature = '';
  }

  /** Draw-nothing-at-idle meshes, measured 0 px on the default front view. */
  private applyHideRules(): void {
    this.scene.traverse((o) => {
      const mesh = o as THREE.Mesh;
      if (!mesh.isMesh) return;
      const mat = mesh.material as THREE.MeshBasicMaterial;
      const parentName = mesh.parent?.name ?? '';
      const isReticleBacking =
        parentName.startsWith('Reticle') && !!mat?.isMeshBasicMaterial;
      const isPanelShell =
        mesh.name === 'BackfaceLayer' || mesh.name === 'UICardEdge';
      if (isReticleBacking || isPanelShell) {
        // Re-forced every frame: reticle press states may re-show them.
        mesh.visible = false;
        this.hiddenForGood.add(mesh);
      }
    });
  }

  private restoreSources(): void {
    this.scene.traverse((o) => {
      if (o.userData.spikeSource) {
        o.layers.enable(0);
        o.layers.disable(5);
        delete o.userData.spikeSource;
      }
    });
  }

  private collectSources(): Source[] {
    const out: Source[] = [];
    const walk = (o: THREE.Object3D, parentVisible: boolean) => {
      const visible = parentVisible && o.visible;
      const mesh = o as THREE.Mesh;
      if (
        mesh.isMesh &&
        visible &&
        !mesh.userData.spikeBatched &&
        !this.hiddenForGood.has(mesh) &&
        (mesh.material as THREE.Material | undefined)?.type ===
          'MeshBasicMaterial' &&
        (mesh.material as THREE.Material).visible !== false &&
        // Materials with patched shaders (uikit MSDF font glyphs use
        // onBeforeCompile + per-instance atlas attributes) can't be baked
        // into a plain batched MeshBasicMaterial; keep drawing them directly.
        (mesh.material as THREE.Material).onBeforeCompile ===
          THREE.Material.prototype.onBeforeCompile
      ) {
        const mat = mesh.material as THREE.MeshBasicMaterial;
        if (mat.clippingPlanes && mat.clippingPlanes.length > 0) {
          ensureClipCapture(mesh);
        }
        out.push({
          mesh,
          key: [
            mesh.uuid,
            ...mesh.matrixWorld.elements.map((e) => Math.round(e * 1e4)),
            mat.color.getHexString(),
            mat.opacity,
            mat.map?.uuid ?? 'none',
            mat.depthWrite,
            mat.toneMapped,
            mat.blending,
            mat.side,
          ].join(','),
        });
      }
      if (SKIP_SUBTREES.includes(o.name)) return;
      for (const child of o.children) walk(child, visible);
    };
    walk(this.scene, true);
    if (new URLSearchParams(window.location.search).get('spikeDebug') != null) {
      console.log(
        `SPIKE_MATRIX ${out.map((s) => {
          const m = s.mesh;
          const t = m.matrixWorld.elements;
          const mt = m.matrix.elements;
          const pp = m.parent ? m.parent.matrixWorld.elements : null;
          return `${(m.name || m.parent?.name || '?').slice(0, 10)}:au${m.matrixAutoUpdate ? 1 : 0}${m.matrixWorldAutoUpdate ? 1 : 0}:mwT(${t[12].toFixed(2)},${t[13].toFixed(2)},${t[14].toFixed(2)})mT(${mt[12].toFixed(2)},${mt[13].toFixed(2)},${mt[14].toFixed(2)})pT(${pp ? pp[12].toFixed(2) + ',' + pp[13].toFixed(2) + ',' + pp[14].toFixed(2) : '-'})`;
        }).join(' ')}`
      );
    }
    // Painter order: keep current scene-graph order (matches uikit paint order).
    return out;
  }

  /** three.js painter order for transparent meshes: z asc, tie-break id. */
  private painterSeq(sources: Source[]): SeqEntry[] {
    const srcByUuid = new Map(sources.map((s) => [s.mesh.uuid, s]));
    const seq: SeqEntry[] = [];
    this.scene.traverse((o) => {
      const m = o as THREE.Mesh;
      if (!m.isMesh || !m.visible || this.hiddenForGood.has(m)) return;
      const mat = m.material as THREE.Material | undefined;
      if (!mat || Array.isArray(mat)) return;
      if (mat.transparent && mat.visible !== false) {
        seq.push({m, z: 0, src: srcByUuid.get(m.uuid) ?? null});
      }
    });
    if (this.camera) {
      const ps = new THREE.Matrix4().multiplyMatrices(
        this.camera.projectionMatrix,
        this.camera.matrixWorldInverse
      );
      const v = new THREE.Vector3();
      for (const e of seq) {
        v.setFromMatrixPosition(e.m.matrixWorld).applyMatrix4(ps);
        e.z = v.z;
      }
    }
    seq.sort((a, b) => a.z - b.z || a.m.id - b.m.id);
    return seq;
  }

  private rebuild(sources: Source[]): void {
    for (const batch of this.batchMeshes) {
      batch.geometry.dispose();
      (batch.material as THREE.Material).dispose();
      batch.removeFromParent();
    }
    this.batchMeshes = [];
    this.restoreSources();
    // The renderer updates matrixWorld inside its render call (XREffects does
    // it manually per-eye); baking before that reads stale/identity transforms
    // for recently-created meshes. Force current values first.
    this.scene.updateMatrixWorld();

    const params = new URLSearchParams(window.location.search);
    const debugColor = params.get('spikeDebug') != null;
    const coarse = params.get('spikeOrder') === 'coarse';

    if (params.get('spikeDebug') === 'hide') {
      // Diagnostic: hide all batch sources, build no batches. The pixel diff
      // vs baseline shows the sources' true paint contribution.
      for (const {mesh} of sources) {
        mesh.layers.disable(0);
        mesh.layers.enable(5);
        mesh.userData.spikeSource = true;
      }
      return;
    }

    const seq = this.painterSeq(sources);

    // Bucket sources by material state. `clippingPlanes` matter: uikit clips
    // oversized quads to card bounds, so buckets must not mix clip states.
    const clipId = new WeakMap<object, number>();
    let nextClipId = 0;
    const clipKeyOf = (_s: Source): string => 'none';
    const bucketKeyOf = (s: Source): string => {
      const mat = s.mesh.material as THREE.MeshBasicMaterial;
      return `${mat.map?.uuid ?? 'none'}|${mat.depthWrite}|${mat.toneMapped}|${mat.blending}|${clipKeyOf(s)}|${mat.colorWrite ? 1 : 0}|${mat.premultipliedAlpha ? 1 : 0}|${mat.blendSrc}|${mat.blendDst}|${mat.depthTest ? 1 : 0}|${mat.side}`;
    };
    const bucketState = new Map<
      string,
      {
        proto: THREE.MeshBasicMaterial;
        map: THREE.Texture | null;
        depthWrite: boolean;
        toneMapped: boolean;
        blending: THREE.Blending;
      }
    >();
    for (const s of sources) {
      const mat = s.mesh.material as THREE.MeshBasicMaterial;
      if (!bucketState.has(bucketKeyOf(s))) {
        bucketState.set(bucketKeyOf(s), {
          proto: mat,
          map: mat.map ?? null,
          depthWrite: mat.depthWrite,
          toneMapped: mat.toneMapped,
          blending: mat.blending,
        });
      }
    }

    // Split seq into segments at non-source transparent objects so batches do
    // not cross a panel/reticle paint slot (exact mode).
    const srcToBatch = new Map<string, THREE.Object3D>();
    type Seg = Map<string, Source[]>;
    const segs: Seg[] = [];
    const segIndexOfSeqIdx = new Map<number, number>();
    let cur: Seg = new Map();
    let seqIdx = 0;
    for (const e of seq) {
      if (e.src) {
        const key = bucketKeyOf(e.src);
        let arr = cur.get(key);
        if (!arr) {
          arr = [];
          cur.set(key, arr);
        }
        arr.push(e.src);
        segIndexOfSeqIdx.set(seqIdx, segs.length);
      } else if (!coarse) {
        if (cur.size > 0) {
          segs.push(cur);
          cur = new Map();
        }
      }
      seqIdx++;
    }
    if (cur.size > 0) segs.push(cur);
    if (coarse && segs.length > 1) {
      // Collapse to one segment spanning everything.
      const merged: Seg = new Map();
      for (const seg of segs) {
        for (const [k, arr] of seg) {
          const a = merged.get(k) ?? [];
          merged.set(k, a.concat(arr));
        }
      }
      segs.length = 0;
      segs.push(merged);
      segIndexOfSeqIdx.clear();
      seq.forEach((e, i) => {
        if (e.src) segIndexOfSeqIdx.set(i, 0);
      });
    }

    // Build one merged mesh per (segment, bucket).
    const srcSeqIdx = new Map<string, number>();
    seq.forEach((e, i) => {
      if (e.src) srcSeqIdx.set(e.src.mesh.uuid, i);
    });
    const batchesBySeg: {mesh: THREE.Mesh; firstSeqIdx: number}[][] = [];

    segs.forEach((seg, segIndex) => {
      const entries: {mesh: THREE.Mesh; firstSeqIdx: number}[] = [];
      for (const [key, arr] of seg) {
        const state = bucketState.get(key)!;
        const skip = params.get('spikeSkip');
        const skipMap = params.get('spikeSkipMap');
        if (
          (skip && key.includes(skip)) ||
          (skipMap === '1' && state.map) ||
          (skipMap === '0' && !state.map)
        ) {
          console.log(`SPIKE_SKIP ${key.slice(0, 22)} n=${arr.length}`);
          continue;
        }
        const pieces: THREE.BufferGeometry[] = [];
        let firstIdx = Infinity;
        for (const {mesh} of arr) {
          try {
            const piecesForMesh = bakePiece(mesh);
            for (const piece of piecesForMesh) {
              piece.computeBoundingBox();
              const pb = piece.boundingBox!;
              if (params.get('spikeDebug') != null) {
                console.log(
                  `SPIKE_PIECE ${(mesh.name || mesh.parent?.name || '?').slice(0, 12)} bb=(${(pb.max.x - pb.min.x).toFixed(3)},${(pb.max.y - pb.min.y).toFixed(3)},${(pb.max.z - pb.min.z).toFixed(3)}) at(${pb.min.x.toFixed(2)},${pb.min.y.toFixed(2)},${pb.min.z.toFixed(2)})`
                );
              }
              pieces.push(piece);
            }
          } catch (err) {
            console.log(
              `SPIKE_ERR bake ${(mesh.name || mesh.parent?.name || '?').slice(0, 20)}: ${String(err)}`
            );
          }
          firstIdx = Math.min(firstIdx, srcSeqIdx.get(mesh.uuid) ?? 0);
        }
        if (pieces.length === 0) continue;
        const merged = mergeGeometries(pieces, false);
        for (const piece of pieces) piece.dispose();
        if (!merged) {
          console.warn('SpikeBatcher: merge failed for bucket', key);
          continue;
        }
        console.log(
          `SPIKE_BATCH s${segIndex} ${key.slice(0, 22)} n=${arr.length} verts=${merged.getAttribute('position').count}`
        );
        // Copy the prototype material so colorWrite, blending etc. survive;
        // colors/opacity are baked into vertex colors.
        const material = new THREE.MeshBasicMaterial().copy(state.proto);
        material.vertexColors = true;
        material.color.setRGB(1, 1, 1);
        material.opacity = 1;
        material.transparent = true;
        material.map = state.map;
        if (debugColor) {
          const palette = [
            0xff2020, 0x20ff20, 0x2060ff, 0xffff20, 0xff20ff, 0x20ffff,
            0xff8000, 0x8000ff,
          ];
          material.vertexColors = false;
          material.map = null;
          material.color.setHex(
            palette[this.batchMeshes.length % palette.length]
          );
          // Keep real transparency/order so per-color masks match real paint.
          material.transparent = true;
          material.opacity = 1;
          material.depthTest = true;
        }
        const batchMesh = new THREE.Mesh(merged, material);
        batchMesh.name = `SpikeBatch:s${segIndex}:${key.slice(0, 18)}`;
        batchMesh.userData.spikeBatched = true;
        batchMesh.matrixAutoUpdate = false;
        batchMesh.frustumCulled = false;
        this.scene.add(batchMesh);
        this.batchMeshes.push(batchMesh);
        entries.push({mesh: batchMesh, firstSeqIdx: firstIdx});
        for (const {mesh} of arr) {
          srcToBatch.set(mesh.uuid, batchMesh);
          // Hide from every camera (layer 0 + XR eye layers 1/2) by moving the
          // source to layer 5; HitRegistry's raycaster also sees layer 5.
          // (spikeDebug=2 keeps sources visible for overpaint comparison.)
          if (params.get('spikeDebug') !== '2') {
            mesh.layers.disable(0);
            mesh.layers.enable(5);
          }
          mesh.userData.spikeSource = true;
        }
      }
      entries.sort((a, b) => a.firstSeqIdx - b.firstSeqIdx);
      batchesBySeg[segIndex] = entries;
    });

    // Freeze painter order = SCENE-GRAPH order (three's actual tie-break),
    // NOT the z-sorted seq: batches take the rank of their first source.
    const finalOrder: THREE.Object3D[] = [];
    const seen = new Set<THREE.Object3D>();
    this.scene.traverse((o) => {
      const m = o as THREE.Mesh;
      if (!m.isMesh || this.hiddenForGood.has(m)) return;
      const mat = m.material as THREE.Material | undefined;
      if (!mat || Array.isArray(mat)) return;
      if (!mat.transparent || mat.visible === false) return;
      if (m.userData.spikeBatched) {
        if (!seen.has(m)) {
          finalOrder.push(m);
          seen.add(m);
        }
        return;
      }
      const b = srcToBatch.get(m.uuid);
      if (b) {
        if (!seen.has(b)) {
          finalOrder.push(b);
          seen.add(b);
        }
      } else {
        finalOrder.push(m);
      }
    });
    finalOrder.forEach((o, rank) => {
      if (!this.savedRenderOrder.has(o.id)) {
        this.savedRenderOrder.set(o.id, o.renderOrder);
      }
      o.renderOrder = rank;
    });

    if (debugColor) {
      console.log(
        `SPIKE_ORDER_FINAL ${finalOrder.map((o, r) => `${r}:${o.name || o.parent?.name || '?'}#${o.id}`).join(' ')}`
      );
    }
  }
}

/** Bakes world transform + per-source color into mergeable geometry pieces. */
function bakePiece(mesh: THREE.Mesh): THREE.BufferGeometry[] {
  const geometry = mesh.geometry;
  if (!geometry.getAttribute('position')) return [];
  const mat = mesh.material as THREE.MeshBasicMaterial;
  const inst = mesh as THREE.InstancedMesh;
  const worlds: THREE.Matrix4[] = [];
  const instColors: (THREE.InstancedBufferAttribute | null)[] = [];
  if (inst.isInstancedMesh) {
    // Bake every instance (uikit renders glyph shadows etc. as instances of a
    // unit quad with tiny per-instance matrices).
    for (let i = 0; i < inst.count; i++) {
      const m = new THREE.Matrix4().fromArray(
        inst.instanceMatrix.array as unknown as number[],
        i * 16
      );
      worlds.push(new THREE.Matrix4().multiplyMatrices(mesh.matrixWorld, m));
      instColors.push(inst.instanceColor ? inst.instanceColor : null);
    }
  } else {
    worlds.push(mesh.matrixWorld);
    instColors.push(null);
  }

  const out: THREE.BufferGeometry[] = [];
  for (let w = 0; w < worlds.length; w++) {
    const world = worlds[w];
    const piece = geometry.clone();
    piece.applyMatrix4(world);
    // Normalize attributes so every piece merges cleanly (keep position, uv and
    // existing vertex colors).
    for (const name of Object.keys(piece.attributes)) {
      if (name !== 'position' && name !== 'uv' && name !== 'color') {
        piece.deleteAttribute(name);
      }
    }
    if (!piece.getAttribute('uv')) {
      const count = piece.getAttribute('position').count;
      piece.setAttribute(
        'uv',
        new THREE.BufferAttribute(new Float32Array(count * 2), 2)
      );
    }
    const windingMode = new URLSearchParams(window.location.search).get(
      'spikeDbgWinding'
    ); // 'flip' | 'none' | default auto
    const shouldReverse =
      windingMode === 'flip'
        ? true
        : windingMode === 'none'
          ? false
          : world.determinant() < 0;
    if (shouldReverse) {
      // The SVG y-flip makes three flip frontFace at draw time; baking the
      // matrix loses that, so reverse winding to keep identical culling.
      const index = piece.getIndex();
      if (index) {
        const arr = index.array;
        for (let i = 0; i < arr.length; i += 3) {
          const tmp = arr[i + 1];
          arr[i + 1] = arr[i + 2];
          arr[i + 2] = tmp;
        }
        index.needsUpdate = true;
      } else {
        // Non-indexed (MergedSvg outlines): swap vertices 1<->2 per triangle
        // across every attribute to reverse winding.
        const pos = piece.getAttribute('position');
        const triCount = Math.floor(pos.count / 3);
        for (const name of ['position', 'uv', 'color']) {
          const attr = piece.getAttribute(name);
          if (!attr) continue;
          const item = attr.itemSize;
          for (let t = 0; t < triCount; t++) {
            const a = (t * 3 + 1) * item;
            const b = (t * 3 + 2) * item;
            for (let k = 0; k < item; k++) {
              const tmp = attr.array[a + k];
              attr.array[a + k] = attr.array[b + k];
              attr.array[b + k] = tmp;
            }
          }
          attr.needsUpdate = true;
        }
      }
    }
    const count = piece.getAttribute('position').count;
    const colors = new Float32Array(count * 4);
    // Preserve source vertex colors (MergedSvg SVG fills carry them) and
    // per-instance colors (InstancedMesh glyph shadows).
    const src = piece.getAttribute('color');
    const instColor = instColors[w];
    const ir = instColor ? instColor.getX(w) : 1;
    const ig = instColor ? instColor.getY(w) : 1;
    const ib = instColor ? instColor.getZ(w) : 1;
    for (let i = 0; i < count; i++) {
      colors[i * 4] = (src ? src.getX(i) : 1) * ir * mat.color.r;
      colors[i * 4 + 1] = (src ? src.getY(i) : 1) * ig * mat.color.g;
      colors[i * 4 + 2] = (src ? src.getZ(i) : 1) * ib * mat.color.b;
      // Keep source alpha (MergedSvg holes are alpha-0 shapes over fills).
      colors[i * 4 + 3] =
        (src && src.itemSize >= 4 ? src.getW(i) : 1) * mat.opacity;
    }
    piece.setAttribute('color', new THREE.BufferAttribute(colors, 4));
    out.push(piece);
  }
  return out;
}
