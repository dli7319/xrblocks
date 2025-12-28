import * as THREE from 'three';

import {DepthOptions} from './DepthOptions';

export class DepthTextures {
  private float32Arrays: Float32Array[] = [];
  private uint8Arrays: Uint8Array[] = [];
  // Use a single array texture for all views.
  private dataTexture: THREE.DataArrayTexture | null = null;
  private nativeTextures: THREE.ExternalTexture[] = [];
  public depthData: XRCPUDepthInformation[] = [];

  constructor(private options: DepthOptions) {}

  private createDataDepthTextures(
    depthData: XRCPUDepthInformation
  ) {
    if (this.dataTexture) {
      this.dataTexture.dispose();
    }
    // We assume mostly 2 views for VR/AR.
    // If more views are needed, we should probably resize this dynamically.
    const depth = 2; // Default to 2 layers (left/right)

    if (this.options.useFloat32) {
      const size = depthData.width * depthData.height * depth;
      const typedArray = new Float32Array(size);
      const format = THREE.RedFormat;
      const type = THREE.FloatType;

      this.dataTexture = new THREE.DataArrayTexture(
        typedArray,
        depthData.width,
        depthData.height,
        depth
      );
      this.dataTexture.format = format;
      this.dataTexture.type = type;
    } else {
      const size = depthData.width * depthData.height * 2 * depth; // 2 bytes per pixel
      const typedArray = new Uint8Array(size);
      const format = THREE.RGFormat;
      const type = THREE.UnsignedByteType;

      this.dataTexture = new THREE.DataArrayTexture(
        typedArray,
        depthData.width,
        depthData.height,
        depth
      );
      this.dataTexture.format = format;
      this.dataTexture.type = type;
    }
    this.dataTexture.needsUpdate = true;
  }

  updateData(depthData: XRCPUDepthInformation, viewId: number) {
    if (
      !this.dataTexture ||
      this.dataTexture.image.width !== depthData.width ||
      this.dataTexture.image.height !== depthData.height
    ) {
      this.createDataDepthTextures(depthData);
    }

    const width = depthData.width;
    const height = depthData.height;
    const layerSize = width * height;

    if (this.options.useFloat32) {
      const textureData = this.dataTexture!.image.data as Float32Array;
      // Copy data to the correct layer offset
      textureData.set(new Float32Array(depthData.data), viewId * layerSize);
    } else {
      const textureData = this.dataTexture!.image.data as Uint8Array;
      // Copy data to the correct layer offset (2 bytes per pixel for Uint16 via Uint8 view)
      textureData.set(new Uint8Array(depthData.data), viewId * layerSize * 2);
    }

    this.dataTexture!.needsUpdate = true;
    this.depthData[viewId] = depthData;
  }

  updateNativeTexture(
    depthData: XRWebGLDepthInformation,
    renderer: THREE.WebGLRenderer,
    viewId: number
  ) {
    if (this.nativeTextures.length < viewId + 1) {
      this.nativeTextures[viewId] = new THREE.ExternalTexture(
        depthData.texture
      );
    } else {
      this.nativeTextures[viewId].sourceTexture = depthData.texture;
    }
    // fixed in newer revision of three
    const textureProperties = renderer.properties.get(
      this.nativeTextures[viewId]
    ) as {
      __webglTexture: WebGLTexture;
      __version: number;
    };
    textureProperties.__webglTexture = depthData.texture;
    textureProperties.__version = 1;
  }

  get(viewId: number) {
    if (this.dataTexture) {
      return this.dataTexture;
    }

    return this.nativeTextures[viewId];
  }
}
