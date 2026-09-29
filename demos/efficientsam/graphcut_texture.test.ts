import * as THREE from 'three';
import {describe, expect, it} from 'vitest';

import {
  DinicMaxFlowSolver,
  type TextureKeyframe,
  computeGraphCutTextureAtlas,
  computeMaskDistanceField,
  filterLargestConnectedMeshComponent,
} from './graphcut_texture';
import type {ExtractedTSDFMesh} from './tsdf_volume';

describe('Graph-Cut Multi-View Texture Mapping & Connected-Component Filtering', () => {
  it('computes exact s-t max-flow and min-cut partition with DinicMaxFlowSolver', () => {
    const solver = new DinicMaxFlowSolver(4);
    const s = 0;
    const a = 1;
    const b = 2;
    const t = 3;

    // s -> a (10), s -> b (2), a -> b (1), a -> t (3), b -> t (10)
    // Min-cut is (s->b: 2) + (a->b: 1) + (a->t: 3) = 6, leaving {s, a} on source side
    solver.addDirectedEdge(s, a, 10);
    solver.addDirectedEdge(s, b, 2);
    solver.addDirectedEdge(a, b, 1);
    solver.addDirectedEdge(a, t, 3);
    solver.addDirectedEdge(b, t, 10);

    const flow = solver.maxFlow(s, t);
    expect(flow).toBeCloseTo(6.0, 4);
    expect(solver.isReachableFromSource(a)).toBe(true);
    expect(solver.isReachableFromSource(b)).toBe(false);
  });

  it('filters out disconnected table/background shards and keeps only the main object component', () => {
    // Component 1: 2 connected triangles near origin (the object)
    // Component 2: 1 disconnected triangle far away (floating table shard)
    const positions = new Float32Array([
      // Tri 0
      0, 0, 0, 0.1, 0, 0, 0, 0.1, 0,
      // Tri 1 (shares edge (0.1,0,0)-(0,0.1,0) with Tri 0)
      0.1, 0, 0, 0.1, 0.1, 0, 0, 0.1, 0,
      // Tri 2 (disconnected shard at (1.5, -0.5, 1.5))
      1.5, -0.5, 1.5, 1.6, -0.5, 1.5, 1.5, -0.4, 1.5,
    ]);
    const normals = new Float32Array(27).fill(0);
    for (let i = 2; i < 27; i += 3) normals[i] = 1;
    const colors = new Float32Array(27).fill(1);

    const mesh: ExtractedTSDFMesh = {
      positions,
      normals,
      colors,
      triangleCount: 3,
    };

    const filtered = filterLargestConnectedMeshComponent(mesh, {
      x: 0.05,
      y: 0.05,
      z: 0,
    });
    expect(filtered.triangleCount).toBe(2);
    expect(filtered.positions.length).toBe(18);
  });

  it('assigns optimal front-facing views via alpha-expansion graph-cut and packs a UV texture atlas', () => {
    // Build a 2-face L-shaped corner mesh:
    // Face A (2 triangles) faces +Z
    // Face B (2 triangles) faces +X
    const positions = new Float32Array([
      // Face A (+Z normal): z = 0, x in [-0.1, 0], y in [-0.05, 0.05]
      -0.1, -0.05, 0, 0, -0.05, 0, 0, 0.05, 0, -0.1, -0.05, 0, 0, 0.05, 0, -0.1,
      0.05, 0,
      // Face B (+X normal, shares vertical edge at x=0, z=0): x = 0, z in [-0.1, 0]
      0,
      -0.05, 0, 0, -0.05, -0.1, 0, 0.05, -0.1, 0, -0.05, 0, 0, 0.05, -0.1, 0,
      0.05, 0,
    ]);
    const normals = new Float32Array([
      // Face A normals (+Z)
      0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1,
      // Face B normals (+X)
      1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
    ]);
    const colors = new Float32Array(36).fill(1.0);
    const mesh: ExtractedTSDFMesh = {
      positions,
      normals,
      colors,
      triangleCount: 4,
    };

    const makeKeyframe = (
      id: number,
      camPos: THREE.Vector3,
      rgb: [number, number, number]
    ): TextureKeyframe => {
      const W = 64;
      const H = 64;
      const cam = new THREE.PerspectiveCamera(60, 1.0, 0.05, 5.0);
      cam.position.copy(camPos);
      cam.lookAt(0, 0, -0.05);
      cam.updateMatrixWorld(true);
      cam.updateProjectionMatrix();

      const clipMat = new THREE.Matrix4().multiplyMatrices(
        cam.projectionMatrix,
        cam.matrixWorldInverse
      );
      const mask = new Uint8Array(W * H).fill(1);
      const distField = computeMaskDistanceField(mask, W, H, 32);
      const rgba = new Uint8ClampedArray(W * H * 4);
      for (let i = 0; i < W * H; i++) {
        rgba[i * 4] = rgb[0];
        rgba[i * 4 + 1] = rgb[1];
        rgba[i * 4 + 2] = rgb[2];
        rgba[i * 4 + 3] = 255;
      }

      const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(cam.quaternion);
      return {
        id,
        rgba,
        width: W,
        height: H,
        cameraBinaryMask: mask,
        maskDistField: distField,
        rgbClipFromWorldMatrix: new Float32Array(clipMat.elements),
        cameraPos: {x: camPos.x, y: camPos.y, z: camPos.z},
        cameraForward: {x: fwd.x, y: fwd.y, z: fwd.z},
      };
    };

    // Keyframe 0 looks from +Z (front-on to Face A)
    const kf0 = makeKeyframe(
      0,
      new THREE.Vector3(-0.05, 0, 0.5),
      [220, 40, 40]
    );
    // Keyframe 1 looks from +X (front-on to Face B)
    const kf1 = makeKeyframe(
      1,
      new THREE.Vector3(0.5, 0, -0.05),
      [40, 200, 80]
    );

    const result = computeGraphCutTextureAtlas(mesh, [kf0, kf1]);
    expect(result.triangleCount).toBe(4);
    expect(result.uvs.length).toBe(24);
    expect(result.chartCount).toBe(2);
    expect(result.atlasWidth).toBe(128);
    expect(result.atlasHeight).toBe(64);

    // Face A (triangles 0, 1) should map to tile 0 (u in [0, 0.5])
    expect(result.uvs[0]).toBeLessThan(0.5);
    expect(result.uvs[6]).toBeLessThan(0.5);
    // Face B (triangles 2, 3) should map to tile 1 (u in [0.5, 1.0])
    expect(result.uvs[12]).toBeGreaterThanOrEqual(0.5);
    expect(result.uvs[18]).toBeGreaterThanOrEqual(0.5);
  });
});
