import * as THREE from 'three';
import {describe, expect, it} from 'vitest';

import {
  ObjectTSDFVolume,
  fitSupportPlaneRansac,
  invertMatrix4ColMajor,
  seedVolumeFromMaskAndDepth,
} from './tsdf_volume';

/**
 * Synthesizes a depth buffer (`width x height`), binary mask (`maskW x maskH`),
 * and RGBA buffer for a sphere at `sphereCenter` with `sphereRadius` viewed from `camera`.
 */
function renderSyntheticSphereView(
  camera: THREE.PerspectiveCamera,
  sphereCenter: THREE.Vector3,
  sphereRadius: number,
  depthWidth = 80,
  depthHeight = 80,
  maskW = 128,
  maskH = 128
) {
  camera.updateMatrixWorld(true);
  camera.updateProjectionMatrix();

  const depthViewMatrix = new Float32Array(camera.matrixWorldInverse.elements);
  const depthProjectionMatrix = new Float32Array(
    camera.projectionMatrix.elements
  );
  const depthProjectionInverseMatrix = new Float32Array(
    camera.projectionMatrixInverse.elements
  );

  const clipFromWorld = new THREE.Matrix4().multiplyMatrices(
    camera.projectionMatrix,
    camera.matrixWorldInverse
  );
  const rgbClipFromWorldMatrix = new Float32Array(clipFromWorld.elements);

  const depthData = new Float32Array(depthWidth * depthHeight);
  const camOrigin = new THREE.Vector3();
  camera.getWorldPosition(camOrigin);

  const rayDir = new THREE.Vector3();
  const ndcPt = new THREE.Vector3();
  const viewPt = new THREE.Vector3();

  for (let dy = 0; dy < depthHeight; dy++) {
    const v = 1.0 - (dy + 0.5) / depthHeight;
    for (let dx = 0; dx < depthWidth; dx++) {
      const u = (dx + 0.5) / depthWidth;
      ndcPt.set(2.0 * u - 1.0, 2.0 * v - 1.0, -1.0);
      viewPt.copy(ndcPt).applyMatrix4(camera.projectionMatrixInverse);
      // Ray direction in world space
      rayDir.copy(viewPt).transformDirection(camera.matrixWorld).normalize();

      // Ray-sphere intersection
      const ocX = camOrigin.x - sphereCenter.x;
      const ocY = camOrigin.y - sphereCenter.y;
      const ocZ = camOrigin.z - sphereCenter.z;
      const b = ocX * rayDir.x + ocY * rayDir.y + ocZ * rayDir.z;
      const c = ocX * ocX + ocY * ocY + ocZ * ocZ - sphereRadius * sphereRadius;
      const disc = b * b - c;
      if (disc >= 0) {
        const t = -b - Math.sqrt(disc);
        if (t > 0.05) {
          const hitWorld = new THREE.Vector3()
            .copy(camOrigin)
            .addScaledVector(rayDir, t);
          const hitView = hitWorld.applyMatrix4(camera.matrixWorldInverse);
          depthData[dy * depthWidth + dx] = -hitView.z;
        }
      } else {
        // Background wall at 2.2m
        depthData[dy * depthWidth + dx] = 2.2;
      }
    }
  }

  const cameraBinaryMask = new Uint8Array(maskW * maskH);
  const rgbaData = new Uint8ClampedArray(maskW * maskH * 4);

  for (let my = 0; my < maskH; my++) {
    const v = 1.0 - (my + 0.5) / maskH;
    for (let mx = 0; mx < maskW; mx++) {
      const u = (mx + 0.5) / maskW;
      ndcPt.set(2.0 * u - 1.0, 2.0 * v - 1.0, -1.0);
      viewPt.copy(ndcPt).applyMatrix4(camera.projectionMatrixInverse);
      rayDir.copy(viewPt).transformDirection(camera.matrixWorld).normalize();

      const ocX = camOrigin.x - sphereCenter.x;
      const ocY = camOrigin.y - sphereCenter.y;
      const ocZ = camOrigin.z - sphereCenter.z;
      const b = ocX * rayDir.x + ocY * rayDir.y + ocZ * rayDir.z;
      const c = ocX * ocX + ocY * ocY + ocZ * ocZ - sphereRadius * sphereRadius;
      const idx = my * maskW + mx;
      if (b * b - c >= 0) {
        cameraBinaryMask[idx] = 1;
        rgbaData[idx * 4] = 240;
        rgbaData[idx * 4 + 1] = 100;
        rgbaData[idx * 4 + 2] = 50;
        rgbaData[idx * 4 + 3] = 255;
      }
    }
  }

  return {
    depthData,
    depthWidth,
    depthHeight,
    depthViewMatrix,
    depthProjectionMatrix,
    depthProjectionInverseMatrix,
    rgbClipFromWorldMatrix,
    cameraBinaryMask,
    rgbaData,
    maskW,
    maskH,
  };
}

describe('ObjectTSDFVolume & Circle-to-Digitize KinectFusion', () => {
  it('inverts a 4x4 column-major matrix accurately', () => {
    const cam = new THREE.PerspectiveCamera(60, 1.0, 0.05, 10.0);
    cam.position.set(0.3, 1.4, 0.8);
    cam.lookAt(0, 1.2, -0.5);
    cam.updateMatrixWorld(true);

    const m = new Float32Array(cam.matrixWorld.elements);
    const inv = new Float32Array(16);
    expect(invertMatrix4ColMajor(m, inv)).toBe(true);

    const expected = cam.matrixWorldInverse.elements;
    for (let i = 0; i < 16; i++) {
      expect(inv[i]).toBeCloseTo(expected[i], 4);
    }
  });

  it('seeds a tight 3D bounding volume from depth and binary mask while rejecting background wall', () => {
    const sphereCenter = new THREE.Vector3(0.05, 1.2, -0.55);
    const sphereRadius = 0.1;
    const cam = new THREE.PerspectiveCamera(60, 1.0, 0.05, 5.0);
    cam.position.set(0, 1.2, 0);
    cam.lookAt(sphereCenter);

    const view = renderSyntheticSphereView(cam, sphereCenter, sphereRadius);
    const seed = seedVolumeFromMaskAndDepth({
      depthData: view.depthData,
      depthWidth: view.depthWidth,
      depthHeight: view.depthHeight,
      rawValueToMeters: 1.0,
      depthViewMatrix: view.depthViewMatrix,
      depthProjectionInverseMatrix: view.depthProjectionInverseMatrix,
      rgbClipFromWorldMatrix: view.rgbClipFromWorldMatrix,
      cameraBinaryMask: view.cameraBinaryMask,
      maskWidth: view.maskW,
      maskHeight: view.maskH,
    });

    expect(seed).not.toBeNull();
    expect(seed!.sampleCount).toBeGreaterThan(20);
    expect(seed!.center.x).toBeCloseTo(sphereCenter.x, 1);
    expect(seed!.center.y).toBeCloseTo(sphereCenter.y, 1);
    expect(seed!.center.z).toBeCloseTo(sphereCenter.z, 0);
    expect(seed!.sizeMeters).toBeGreaterThanOrEqual(0.16);
    expect(seed!.sizeMeters).toBeLessThanOrEqual(0.45);
  });

  it('fuses multiple views into TSDF + RGB volume and extracts a valid Marching Cubes mesh', () => {
    const sphereCenter = new THREE.Vector3(0, 1.2, -0.5);
    const sphereRadius = 0.09;
    const volume = new ObjectTSDFVolume(
      {x: sphereCenter.x, y: sphereCenter.y, z: sphereCenter.z},
      0.28,
      48,
      sphereCenter.y - sphereRadius - 0.03
    );

    const angles = [0, Math.PI * 0.35, -Math.PI * 0.35, Math.PI * 0.75];
    for (const angle of angles) {
      const cam = new THREE.PerspectiveCamera(60, 1.0, 0.05, 5.0);
      cam.position.set(
        sphereCenter.x + Math.sin(angle) * 0.5,
        sphereCenter.y + 0.05,
        sphereCenter.z + Math.cos(angle) * 0.5
      );
      cam.lookAt(sphereCenter);

      const view = renderSyntheticSphereView(cam, sphereCenter, sphereRadius);
      const updated = volume.integrateFrame({
        depthData: view.depthData,
        depthWidth: view.depthWidth,
        depthHeight: view.depthHeight,
        rawValueToMeters: 1.0,
        depthViewMatrix: view.depthViewMatrix,
        depthProjectionMatrix: view.depthProjectionMatrix,
        rgbClipFromWorldMatrix: view.rgbClipFromWorldMatrix,
        rgbaData: view.rgbaData,
        rgbWidth: view.maskW,
        rgbHeight: view.maskH,
        cameraBinaryMask: view.cameraBinaryMask,
        maskWidth: view.maskW,
        maskHeight: view.maskH,
        carveOutsideMask: true,
      });

      expect(updated).toBeGreaterThan(100);
    }

    expect(volume.fusedFrameCount).toBe(angles.length);

    const mesh = volume.extractMesh(1.0);
    expect(mesh.triangleCount).toBeGreaterThan(100);
    expect(mesh.positions.length).toBe(mesh.triangleCount * 9);
    expect(mesh.normals.length).toBe(mesh.triangleCount * 9);
    expect(mesh.colors.length).toBe(mesh.triangleCount * 9);

    // Verify extracted vertices lie near the sphere surface and normals are unit length
    for (let i = 0; i < Math.min(90, mesh.positions.length); i += 3) {
      const vx = mesh.positions[i];
      const vy = mesh.positions[i + 1];
      const vz = mesh.positions[i + 2];
      const dist = Math.hypot(
        vx - sphereCenter.x,
        vy - sphereCenter.y,
        vz - sphereCenter.z
      );
      expect(dist).toBeGreaterThan(sphereRadius * 0.65);
      expect(dist).toBeLessThan(sphereRadius * 1.35);

      const nx = mesh.normals[i];
      const ny = mesh.normals[i + 1];
      const nz = mesh.normals[i + 2];
      expect(Math.hypot(nx, ny, nz)).toBeCloseTo(1.0, 2);

      // Verify fused RGB color matches the synthetic orange sphere (240, 100, 50)
      expect(mesh.colors[i]).toBeGreaterThan(0.7);
    }

    // Verify 3D bounding box projection generates a valid 2D prompt box
    const testCam = new THREE.PerspectiveCamera(60, 1.0, 0.05, 5.0);
    testCam.position.set(0, 1.2, 0);
    testCam.lookAt(sphereCenter);
    testCam.updateMatrixWorld(true);
    testCam.updateProjectionMatrix();
    const clipMat = new THREE.Matrix4().multiplyMatrices(
      testCam.projectionMatrix,
      testCam.matrixWorldInverse
    );
    const box2d = volume.projectBoundingBoxToCamera(
      new Float32Array(clipMat.elements),
      512,
      512
    );
    expect(box2d).not.toBeNull();
    expect(box2d!.cx).toBeCloseTo(256, 0);
    expect(box2d!.cy).toBeCloseTo(256, 0);
  });

  it('fits a supporting table plane from mask border points and clips table voxels', () => {
    const borderPts = [];
    for (let z = -0.7; z <= -0.3; z += 0.08) {
      for (let x = -0.2; x <= 0.2; x += 0.08) {
        // Horizontal tabletop at y = 0.75m
        borderPts.push({x, y: 0.75, z});
      }
    }
    const objectPts = [
      {x: 0, y: 0.8, z: -0.5},
      {x: 0.04, y: 0.82, z: -0.5},
      {x: -0.04, y: 0.79, z: -0.48},
      {x: 0, y: 0.84, z: -0.52},
      {x: 0.02, y: 0.81, z: -0.49},
      {x: -0.02, y: 0.83, z: -0.51},
    ];

    const plane = fitSupportPlaneRansac(borderPts, objectPts, {
      x: 0,
      y: 1.4,
      z: 0,
    });
    expect(plane).not.toBeNull();
    expect(plane!.ny).toBeCloseTo(1.0, 2);
    expect(plane!.d).toBeCloseTo(-0.75, 2);
  });
});
