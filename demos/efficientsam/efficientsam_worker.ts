import type {
  GraphCutTexturedMesh,
  TextureKeyframe,
} from './graphcut_texture.js';
import type {
  BoundingBox2D,
  ExtractedTSDFMesh,
  ObjectTSDFVolume,
  SupportPlane3D,
  Vec3Tuple,
  VolumeSeedParams,
  VolumeSeedResult,
} from './tsdf_volume.js';

export interface WorkerTsdfMeshPayload {
  positionsBuffer: ArrayBuffer;
  normalsBuffer: ArrayBuffer;
  uvsBuffer: ArrayBuffer;
  colorsBuffer: ArrayBuffer;
  atlasRgbaBuffer: ArrayBuffer;
  atlasWidth: number;
  atlasHeight: number;
  chartCount: number;
  triangleCount: number;
  fusedFrameCount: number;
  updatedVoxels: number;
  volumeCenter: Vec3Tuple;
  volumeSizeMeters: number;
  boundsMin: Vec3Tuple;
  boundsMax: Vec3Tuple;
  voxelSizeMm: number;
  projectedBox2D: BoundingBox2D | null;
  tsdfMs: number;
}

(() => {
  type Accelerator = 'webgpu' | 'wasm';

  interface LiteRtTensor {
    readonly deleted: boolean;
    data(): Promise<Float32Array | Int32Array | Uint8Array>;
    delete(): void;
  }

  interface LiteRtTensorConstructor {
    new (
      data: Float32Array | Int32Array | Uint8Array,
      shape: number[]
    ): LiteRtTensor;
  }

  interface LiteRtCompiledModel {
    readonly deleted: boolean;
    readonly isFullyAccelerated?: boolean;
    readonly options?: {accelerator?: Accelerator};
    run(inputs: LiteRtTensor[]): Promise<LiteRtTensor[]>;
    delete(): void;
  }

  interface LiteRtCoreModule {
    loadLiteRt(
      wasmPath: string,
      options?: {threads?: boolean; jspi?: boolean}
    ): Promise<{getWebGpuDevice(): unknown}>;
    loadAndCompile(
      modelPath: string,
      options?: {accelerator?: Accelerator}
    ): Promise<LiteRtCompiledModel>;
    supportsFeature(
      feature: 'relaxedSimd' | 'threads' | 'jspi'
    ): Promise<boolean>;
    Tensor: LiteRtTensorConstructor;
  }

  interface TsdfVolumeModule {
    ObjectTSDFVolume: new (
      center: Vec3Tuple,
      sizeMeters: number,
      resolution?: number,
      supportFloorY?: number,
      supportPlane?: SupportPlane3D | null,
      boundsMin?: Vec3Tuple,
      boundsMax?: Vec3Tuple
    ) => ObjectTSDFVolume;
    seedVolumeFromMaskAndDepth: (
      params: VolumeSeedParams
    ) => VolumeSeedResult | null;
    filterDepthDiscontinuities: (
      depthData: Float32Array | Uint16Array,
      width: number,
      height: number,
      rawValueToMeters: number,
      minDepthMeters?: number,
      maxDepthMeters?: number,
      edgeJumpThresholdMeters?: number
    ) => Float32Array;
    invertMatrix4ColMajor: (m: Float32Array, out?: Float32Array) => boolean;
  }

  interface GraphCutTextureModule {
    computeMaskDistanceField: (
      mask: Uint8Array,
      width: number,
      height: number,
      maxDist?: number
    ) => Uint8Array;
    filterLargestConnectedMeshComponent: (
      mesh: ExtractedTSDFMesh,
      targetCenter: Vec3Tuple
    ) => ExtractedTSDFMesh;
    computeGraphCutTextureAtlas: (
      mesh: ExtractedTSDFMesh,
      keyframes: TextureKeyframe[]
    ) => GraphCutTexturedMesh;
  }

  const LITERT_CORE_ESM_URL =
    'https://cdn.jsdelivr.net/npm/@litertjs/core@2.5.3/+esm';
  const LITERT_WASM_URL =
    'https://cdn.jsdelivr.net/npm/@litertjs/core@2.5.3/wasm/';

  const MODEL_IMG_SIZE = 512;
  const MASK_LOW_RES = 128;
  const MAX_POINTS = 6;
  const TSDF_RESOLUTION = 64;
  const MAX_TEXTURE_KEYFRAMES = 9;

  const ENCODER_MODEL_URL =
    'https://rawcdn.githack.com/xrblocks/proprietary-assets/21bcc2a3e5a44a05b778889244a212d33acaf119/tflite_models/efficientsam/efficientsam_ti_encoder.tflite';
  const DECODER_MODEL_URL =
    'https://rawcdn.githack.com/xrblocks/proprietary-assets/21bcc2a3e5a44a05b778889244a212d33acaf119/tflite_models/efficientsam/efficientsam_ti_decoder.tflite';

  let litertMod: LiteRtCoreModule | null = null;
  let tsdfMod: TsdfVolumeModule | null = null;
  let graphCutMod: GraphCutTextureModule | null = null;
  let encoderModel: LiteRtCompiledModel | null = null;
  let decoderModel: LiteRtCompiledModel | null = null;
  let encoderAccelerator: Accelerator = 'wasm';
  let decoderAccelerator: Accelerator = 'wasm';

  let activeTsdfVolume: ObjectTSDFVolume | null = null;
  let textureKeyframes: TextureKeyframe[] = [];
  let nextKeyframeId = 1;
  let pendingSeedCache: {
    cameraBinaryMask: Uint8Array;
    rgba: Uint8ClampedArray;
    rgbClipFromWorldMatrix: Float32Array;
  } | null = null;

  interface WorkerRequestMessage {
    id: number;
    type:
      | 'init'
      | 'xr_segment'
      | 'tsdf_integrate_depth'
      | 'tsdf_integrate_rgb_mask'
      | 'tsdf_reset';
    payload?: Record<string, unknown>;
  }

  async function ensureWorkerModules(): Promise<{
    tsdf: TsdfVolumeModule;
    graphCut: GraphCutTextureModule;
  }> {
    if (!tsdfMod) {
      const tsdfUrl = new URL('./tsdf_volume.js', self.location.href).href;
      tsdfMod = (await import(tsdfUrl)) as unknown as TsdfVolumeModule;
    }
    if (!graphCutMod) {
      const gcUrl = new URL('./graphcut_texture.js', self.location.href).href;
      graphCutMod = (await import(gcUrl)) as unknown as GraphCutTextureModule;
    }
    return {tsdf: tsdfMod, graphCut: graphCutMod};
  }

  async function compileWithWarmupFallback(
    litert: LiteRtCoreModule,
    modelUrl: string,
    preferredAccelerator: Accelerator,
    warmupShapes: number[][]
  ): Promise<{model: LiteRtCompiledModel; accelerator: Accelerator}> {
    const acceleratorsToTry: Accelerator[] =
      preferredAccelerator === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm'];

    for (const accelerator of acceleratorsToTry) {
      let model: LiteRtCompiledModel | null = null;
      const dummyInputs: LiteRtTensor[] = [];
      try {
        model = await litert.loadAndCompile(modelUrl, {accelerator});
        for (const shape of warmupShapes) {
          const size = shape.reduce((a, b) => a * b, 1);
          dummyInputs.push(new litert.Tensor(new Float32Array(size), shape));
        }
        const outputs = await model.run(dummyInputs);
        for (const out of outputs) {
          await out.data();
          if (!out.deleted) out.delete();
        }
        for (const inp of dummyInputs) {
          if (!inp.deleted) inp.delete();
        }
        const actualAccelerator: Accelerator =
          model.options?.accelerator === 'webgpu' ? 'webgpu' : 'wasm';
        console.info(
          `[EfficientSAM Worker] Compiled ${modelUrl} with accelerator='${actualAccelerator}' (fullyAccelerated=${Boolean(model.isFullyAccelerated)})`
        );
        return {model, accelerator: actualAccelerator};
      } catch (err) {
        console.warn(
          `[EfficientSAM Worker] LiteRT '${accelerator}' compile/warmup for ${modelUrl} fell back:`,
          err
        );
        for (const inp of dummyInputs) {
          if (!inp.deleted) inp.delete();
        }
        if (model && !model.deleted) {
          model.delete();
        }
      }
    }

    throw new Error(`Could not compile ${modelUrl} on any LiteRT accelerator.`);
  }

  async function handleInit(): Promise<{
    encoderAccelerator: Accelerator;
    decoderAccelerator: Accelerator;
    compileTimeMs: number;
  }> {
    const t0 = performance.now();
    await ensureWorkerModules();

    if (!litertMod) {
      litertMod = (await import(
        LITERT_CORE_ESM_URL
      )) as unknown as LiteRtCoreModule;
    }

    const jspi = await litertMod.supportsFeature('jspi').catch(() => false);

    // Ensure Emscripten inside the Worker resolves .wasm files from the LiteRT CDN
    // rather than relative to self.location.href (./build/).
    (self as unknown as {Module?: Record<string, unknown>}).Module = {
      locateFile: (path: string) => `${LITERT_WASM_URL}${path}`,
    };

    const liteRt = await litertMod.loadLiteRt(LITERT_WASM_URL, {jspi});
    const hasWebGpu = Boolean(liteRt.getWebGpuDevice());
    const preferred: Accelerator = hasWebGpu ? 'webgpu' : 'wasm';
    console.info(
      `[EfficientSAM Worker] LiteRT 2.5.3 initialized (hasWebGpu=${hasWebGpu}, jspi=${jspi})`
    );

    const encRes = await compileWithWarmupFallback(
      litertMod,
      ENCODER_MODEL_URL,
      preferred,
      [[1, 3, MODEL_IMG_SIZE, MODEL_IMG_SIZE]]
    );
    encoderModel = encRes.model;
    encoderAccelerator = encRes.accelerator;

    const decRes = await compileWithWarmupFallback(
      litertMod,
      DECODER_MODEL_URL,
      preferred,
      [
        [1, 256, 32, 32],
        [1, MAX_POINTS, 2],
        [1, MAX_POINTS],
      ]
    );
    decoderModel = decRes.model;
    decoderAccelerator = decRes.accelerator;

    const compileTimeMs = performance.now() - t0;
    return {
      encoderAccelerator,
      decoderAccelerator,
      compileTimeMs,
    };
  }

  function rgbaToPlanarFloat32(
    rgba: Uint8ClampedArray | Uint8Array
  ): Float32Array {
    const hw = MODEL_IMG_SIZE * MODEL_IMG_SIZE;
    const inputFloat32 = new Float32Array(3 * hw);
    const inv255 = 1.0 / 255.0;
    for (let i = 0; i < hw; i++) {
      const idx = i * 4;
      inputFloat32[i] = rgba[idx] * inv255;
      inputFloat32[hw + i] = rgba[idx + 1] * inv255;
      inputFloat32[2 * hw + i] = rgba[idx + 2] * inv255;
    }
    return inputFloat32;
  }

  async function runSamInferenceOnRgba(
    rgba: Uint8ClampedArray,
    pts: Float32Array,
    lbls: Float32Array
  ): Promise<{
    cameraBinaryMask: Uint8Array;
    fgCount: number;
    camMinX: number;
    camMinY: number;
    camMaxX: number;
    camMaxY: number;
    encoderMs: number;
    decoderMs: number;
    totalMs: number;
    bestIou: number;
  }> {
    if (!litertMod || !encoderModel || !decoderModel) {
      throw new Error('LiteRT models are not initialized yet.');
    }

    const inputFloat32 = rgbaToPlanarFloat32(rgba);

    let inputTensor: LiteRtTensor | null = null;
    let ptsTensor: LiteRtTensor | null = null;
    let lblsTensor: LiteRtTensor | null = null;
    let encOutputs: LiteRtTensor[] = [];
    let decOutputs: LiteRtTensor[] = [];
    let encoderMs = 0;
    let decoderMs = 0;
    let masksLogits: Float32Array;
    let ious: Float32Array;

    try {
      const tEnc0 = performance.now();
      inputTensor = new litertMod.Tensor(inputFloat32, [
        1,
        3,
        MODEL_IMG_SIZE,
        MODEL_IMG_SIZE,
      ]);
      encOutputs = await encoderModel.run([inputTensor]);
      const embTensor = encOutputs[0];
      encoderMs = performance.now() - tEnc0;

      const tDec0 = performance.now();
      ptsTensor = new litertMod.Tensor(pts, [1, MAX_POINTS, 2]);
      lblsTensor = new litertMod.Tensor(lbls, [1, MAX_POINTS]);

      decOutputs = await decoderModel.run([embTensor, ptsTensor, lblsTensor]);
      masksLogits = new Float32Array(
        (await decOutputs[0].data()) as Float32Array
      );
      ious = new Float32Array((await decOutputs[1].data()) as Float32Array);
      decoderMs = performance.now() - tDec0;
    } finally {
      for (const t of [
        inputTensor,
        ptsTensor,
        lblsTensor,
        ...encOutputs,
        ...decOutputs,
      ]) {
        if (t && !t.deleted) {
          t.delete();
        }
      }
    }

    const totalMs = encoderMs + decoderMs;

    let bestIdx = 0;
    if (ious[1] > ious[bestIdx]) bestIdx = 1;
    if (ious[2] > ious[bestIdx]) bestIdx = 2;
    const bestIou = ious[bestIdx];

    const W = MODEL_IMG_SIZE;
    const H = MODEL_IMG_SIZE;
    const maskOffset = bestIdx * MASK_LOW_RES * MASK_LOW_RES;
    const cameraBinaryMask = new Uint8Array(W * H);
    const scaleX = MASK_LOW_RES / W;
    const scaleY = MASK_LOW_RES / H;

    let fgCount = 0;
    let camMinX = W;
    let camMinY = H;
    let camMaxX = 0;
    let camMaxY = 0;

    for (let y = 0; y < H; y++) {
      const sy = (y + 0.5) * scaleY - 0.5;
      const y0 = Math.max(0, Math.min(MASK_LOW_RES - 1, Math.floor(sy)));
      const y1 = Math.max(0, Math.min(MASK_LOW_RES - 1, y0 + 1));
      const wy = sy - y0;
      const row0 = maskOffset + y0 * MASK_LOW_RES;
      const row1 = maskOffset + y1 * MASK_LOW_RES;

      for (let x = 0; x < W; x++) {
        const sx = (x + 0.5) * scaleX - 0.5;
        const x0 = Math.max(0, Math.min(MASK_LOW_RES - 1, Math.floor(sx)));
        const x1 = Math.max(0, Math.min(MASK_LOW_RES - 1, x0 + 1));
        const wx = sx - x0;

        const v00 = masksLogits[row0 + x0];
        const v01 = masksLogits[row0 + x1];
        const v10 = masksLogits[row1 + x0];
        const v11 = masksLogits[row1 + x1];

        const val =
          (1 - wy) * ((1 - wx) * v00 + wx * v01) +
          wy * ((1 - wx) * v10 + wx * v11);

        if (val >= 0.0) {
          cameraBinaryMask[y * W + x] = 1;
          fgCount++;
          if (x < camMinX) camMinX = x;
          if (y < camMinY) camMinY = y;
          if (x > camMaxX) camMaxX = x;
          if (y > camMaxY) camMaxY = y;
        }
      }
    }

    return {
      cameraBinaryMask,
      fgCount,
      camMinX,
      camMinY,
      camMaxX,
      camMaxY,
      encoderMs,
      decoderMs,
      totalMs,
      bestIou,
    };
  }

  function extractCutoutFromMask(
    rgba: Uint8ClampedArray,
    cameraBinaryMask: Uint8Array,
    fgCount: number,
    camMinX: number,
    camMinY: number,
    camMaxX: number,
    camMaxY: number
  ): {cutoutRgbaBuffer: ArrayBuffer | null; cropW: number; cropH: number} {
    const W = MODEL_IMG_SIZE;
    let cutoutRgbaBuffer: ArrayBuffer | null = null;
    let cropW = 0;
    let cropH = 0;
    if (fgCount > 0 && camMaxX >= camMinX && camMaxY >= camMinY) {
      cropW = Math.max(1, camMaxX - camMinX + 1);
      cropH = Math.max(1, camMaxY - camMinY + 1);
      const cutoutRgba = new Uint8ClampedArray(cropW * cropH * 4);
      for (let cy = 0; cy < cropH; cy++) {
        const srcY = camMinY + cy;
        for (let cx = 0; cx < cropW; cx++) {
          const srcX = camMinX + cx;
          const srcIdx = srcY * W + srcX;
          if (cameraBinaryMask[srcIdx]) {
            const srcP = srcIdx * 4;
            const dstP = (cy * cropW + cx) * 4;
            cutoutRgba[dstP] = rgba[srcP];
            cutoutRgba[dstP + 1] = rgba[srcP + 1];
            cutoutRgba[dstP + 2] = rgba[srcP + 2];
            cutoutRgba[dstP + 3] = 255;
          }
        }
      }
      cutoutRgbaBuffer = cutoutRgba.buffer;
    }
    return {cutoutRgbaBuffer, cropW, cropH};
  }

  function decodeDepthBuffer(
    depthBuffer: ArrayBuffer,
    depthFormat: string
  ): Float32Array | Uint16Array {
    return depthFormat === 'uint16' || depthFormat === 'luminance-alpha'
      ? new Uint16Array(depthBuffer)
      : new Float32Array(depthBuffer);
  }

  function extractCameraPoseFromViewMatrix(
    depthViewMatrix: Float32Array,
    invertMatrix4: (m: Float32Array, out?: Float32Array) => boolean
  ): {cameraPos: Vec3Tuple; cameraForward: Vec3Tuple} {
    const worldFromView = new Float32Array(16);
    if (invertMatrix4(depthViewMatrix, worldFromView)) {
      const fx = -worldFromView[8];
      const fy = -worldFromView[9];
      const fz = -worldFromView[10];
      const len = Math.hypot(fx, fy, fz) || 1.0;
      return {
        cameraPos: {
          x: worldFromView[12],
          y: worldFromView[13],
          z: worldFromView[14],
        },
        cameraForward: {x: fx / len, y: fy / len, z: fz / len},
      };
    }
    return {
      cameraPos: {x: 0, y: 1.5, z: 0},
      cameraForward: {x: 0, y: 0, z: -1},
    };
  }

  function addOrUpdateTextureKeyframe(
    rgba: Uint8ClampedArray,
    cameraBinaryMask: Uint8Array,
    rgbClipFromWorldMatrix: Float32Array,
    depthData: Float32Array | Uint16Array,
    depthWidth: number,
    depthHeight: number,
    rawValueToMeters: number,
    depthViewMatrix: Float32Array,
    depthProjectionMatrix: Float32Array,
    normDepthBufferFromNormViewMatrix: Float32Array | undefined,
    tsdf: TsdfVolumeModule,
    graphCut: GraphCutTextureModule
  ): void {
    const {cameraPos, cameraForward} = extractCameraPoseFromViewMatrix(
      depthViewMatrix,
      tsdf.invertMatrix4ColMajor
    );
    const maskDistField = graphCut.computeMaskDistanceField(
      cameraBinaryMask,
      MODEL_IMG_SIZE,
      MODEL_IMG_SIZE,
      32
    );
    const cleanDepth = tsdf.filterDepthDiscontinuities(
      depthData,
      depthWidth,
      depthHeight,
      rawValueToMeters
    );

    const newKf: TextureKeyframe = {
      id: nextKeyframeId++,
      rgba: new Uint8ClampedArray(rgba),
      width: MODEL_IMG_SIZE,
      height: MODEL_IMG_SIZE,
      cameraBinaryMask: new Uint8Array(cameraBinaryMask),
      maskDistField,
      rgbClipFromWorldMatrix: new Float32Array(rgbClipFromWorldMatrix),
      cameraPos,
      cameraForward,
      depthData: cleanDepth,
      depthWidth,
      depthHeight,
      depthViewMatrix: new Float32Array(depthViewMatrix),
      depthProjectionMatrix: new Float32Array(depthProjectionMatrix),
      normDepthBufferFromNormViewMatrix: normDepthBufferFromNormViewMatrix
        ? new Float32Array(normDepthBufferFromNormViewMatrix)
        : undefined,
    };

    if (textureKeyframes.length === 0) {
      textureKeyframes.push(newKf);
      return;
    }

    // Check if an existing keyframe has a very similar viewing direction (< 7 deg)
    let closestIdx = -1;
    let maxDot = -1.0;
    for (let i = 0; i < textureKeyframes.length; i++) {
      const kf = textureKeyframes[i];
      const dot =
        kf.cameraForward.x * cameraForward.x +
        kf.cameraForward.y * cameraForward.y +
        kf.cameraForward.z * cameraForward.z;
      if (dot > maxDot) {
        maxDot = dot;
        closestIdx = i;
      }
    }

    // cos(7 deg) ~= 0.9925 — never overwrite Keyframe 0 (user's primary circled view)
    if (maxDot > 0.9925) {
      if (closestIdx > 0) {
        textureKeyframes[closestIdx] = newKf;
      }
      return;
    }

    if (textureKeyframes.length < MAX_TEXTURE_KEYFRAMES) {
      textureKeyframes.push(newKf);
    } else if (closestIdx > 0) {
      // Replace the most redundant non-initial keyframe
      textureKeyframes[closestIdx] = newKf;
    }
  }

  function buildTexturedMeshPayload(
    volume: ObjectTSDFVolume,
    updatedVoxels: number,
    rgbClipFromWorldMatrix: Float32Array | undefined,
    graphCut: GraphCutTextureModule,
    t0: number
  ): WorkerTsdfMeshPayload {
    const rawExtracted = volume.extractMesh(1.0);
    const cleanMesh = graphCut.filterLargestConnectedMeshComponent(
      rawExtracted,
      volume.center
    );
    const textured = graphCut.computeGraphCutTextureAtlas(
      cleanMesh,
      textureKeyframes
    );

    const projectedBox2D = rgbClipFromWorldMatrix
      ? volume.projectBoundingBoxToCamera(
          rgbClipFromWorldMatrix,
          MODEL_IMG_SIZE,
          MODEL_IMG_SIZE
        )
      : null;
    const tsdfMs = performance.now() - t0;

    return {
      positionsBuffer: textured.positions.buffer,
      normalsBuffer: textured.normals.buffer,
      uvsBuffer: textured.uvs.buffer,
      colorsBuffer: textured.colors.buffer,
      atlasRgbaBuffer: textured.atlasRgba.buffer,
      atlasWidth: textured.atlasWidth,
      atlasHeight: textured.atlasHeight,
      chartCount: textured.chartCount,
      triangleCount: textured.triangleCount,
      fusedFrameCount: volume.fusedFrameCount,
      updatedVoxels,
      volumeCenter: volume.center,
      volumeSizeMeters: volume.sizeMeters,
      boundsMin: volume.boundsMin,
      boundsMax: volume.boundsMax,
      voxelSizeMm: volume.voxelSize * 1000,
      projectedBox2D,
      tsdfMs,
    };
  }

  async function seedAndIntegrateInitialVolume(
    payload: Record<string, unknown>,
    cameraBinaryMask: Uint8Array,
    rgba: Uint8ClampedArray,
    rgbClipFromWorldMatrix: Float32Array
  ): Promise<WorkerTsdfMeshPayload | null> {
    const depthBuffer = payload.depthBuffer as ArrayBuffer | undefined;
    const depthViewMatrixBuffer = payload.depthViewMatrixBuffer as
      | ArrayBuffer
      | undefined;
    const depthProjectionMatrixBuffer = payload.depthProjectionMatrixBuffer as
      | ArrayBuffer
      | undefined;
    const depthProjectionInverseMatrixBuffer =
      payload.depthProjectionInverseMatrixBuffer as ArrayBuffer | undefined;

    if (
      !depthBuffer ||
      !depthViewMatrixBuffer ||
      !depthProjectionMatrixBuffer ||
      !depthProjectionInverseMatrixBuffer
    ) {
      return null;
    }

    const {tsdf, graphCut} = await ensureWorkerModules();

    const t0 = performance.now();
    const depthWidth = Number(payload.depthWidth ?? 160);
    const depthHeight = Number(payload.depthHeight ?? 160);
    const rawValueToMeters = Number(payload.rawValueToMeters ?? 1.0);
    const depthFormat = String(payload.depthFormat ?? 'float32');
    const depthData = decodeDepthBuffer(depthBuffer, depthFormat);

    const depthViewMatrix = new Float32Array(depthViewMatrixBuffer);
    const depthProjectionMatrix = new Float32Array(depthProjectionMatrixBuffer);
    const depthProjectionInverseMatrix = new Float32Array(
      depthProjectionInverseMatrixBuffer
    );
    const normDepthBufferFromNormViewMatrix =
      payload.normDepthBufferFromNormViewMatrixBuffer instanceof ArrayBuffer
        ? new Float32Array(payload.normDepthBufferFromNormViewMatrixBuffer)
        : undefined;

    const seed = tsdf.seedVolumeFromMaskAndDepth({
      depthData,
      depthWidth,
      depthHeight,
      rawValueToMeters,
      depthViewMatrix,
      depthProjectionInverseMatrix,
      normDepthBufferFromNormViewMatrix,
      rgbClipFromWorldMatrix,
      cameraBinaryMask,
      maskWidth: MODEL_IMG_SIZE,
      maskHeight: MODEL_IMG_SIZE,
    });

    if (!seed) {
      return null;
    }

    activeTsdfVolume = new tsdf.ObjectTSDFVolume(
      seed.center,
      seed.sizeMeters,
      TSDF_RESOLUTION,
      seed.supportFloorY,
      seed.supportPlane,
      seed.boundsMin,
      seed.boundsMax
    );
    textureKeyframes = [];
    pendingSeedCache = null;

    addOrUpdateTextureKeyframe(
      rgba,
      cameraBinaryMask,
      rgbClipFromWorldMatrix,
      depthData,
      depthWidth,
      depthHeight,
      rawValueToMeters,
      depthViewMatrix,
      depthProjectionMatrix,
      normDepthBufferFromNormViewMatrix,
      tsdf,
      graphCut
    );

    const updatedVoxels = activeTsdfVolume.integrateFrame({
      depthData,
      depthWidth,
      depthHeight,
      rawValueToMeters,
      depthViewMatrix,
      depthProjectionMatrix,
      normDepthBufferFromNormViewMatrix,
      rgbClipFromWorldMatrix,
      rgbaData: rgba,
      rgbWidth: MODEL_IMG_SIZE,
      rgbHeight: MODEL_IMG_SIZE,
      cameraBinaryMask,
      maskWidth: MODEL_IMG_SIZE,
      maskHeight: MODEL_IMG_SIZE,
      carveOutsideMask: true,
    });

    return buildTexturedMeshPayload(
      activeTsdfVolume,
      updatedVoxels,
      rgbClipFromWorldMatrix,
      graphCut,
      t0
    );
  }

  async function handleXrSegment(payload: Record<string, unknown>): Promise<{
    quadOverlayBuffer: ArrayBuffer;
    cutoutRgbaBuffer: ArrayBuffer | null;
    cropW: number;
    cropH: number;
    quadMinY: number;
    fgCount: number;
    encoderMs: number;
    decoderMs: number;
    totalMs: number;
    bestIou: number;
    tsdfMesh: WorkerTsdfMeshPayload | null;
  }> {
    const rgbaBuffer = payload.rgbaBuffer as ArrayBuffer;
    const pts = new Float32Array(payload.ptsBuffer as ArrayBuffer);
    const lbls = new Float32Array(payload.lblsBuffer as ArrayBuffer);
    const quadToClipElements = new Float32Array(
      payload.quadToClipBuffer as ArrayBuffer
    );
    const quadSizeMeters = Number(payload.quadSizeMeters ?? 0.8);

    const rgba = new Uint8ClampedArray(rgbaBuffer);
    const samRes = await runSamInferenceOnRgba(rgba, pts, lbls);
    const {
      cameraBinaryMask,
      fgCount,
      camMinX,
      camMinY,
      camMaxX,
      camMaxY,
      encoderMs,
      decoderMs,
      totalMs,
      bestIou,
    } = samRes;

    const W = MODEL_IMG_SIZE;
    const H = MODEL_IMG_SIZE;

    // Project cameraBinaryMask onto circleQuad's 512x512 UV space
    const quadBinaryMask = new Uint8Array(W * H);
    let quadMinY = H;
    const e = quadToClipElements;

    for (let qy = 0; qy < H; qy++) {
      const ly = (0.5 - (qy + 0.5) / H) * quadSizeMeters;
      const rowOffset = qy * W;
      for (let qx = 0; qx < W; qx++) {
        const lx = ((qx + 0.5) / W - 0.5) * quadSizeMeters;
        const w = e[3] * lx + e[7] * ly + e[15];
        if (w <= 1e-5) continue;
        const ndcX = (e[0] * lx + e[4] * ly + e[12]) / w;
        const ndcY = (e[1] * lx + e[5] * ly + e[13]) / w;
        const camX = Math.floor((ndcX + 1.0) * 0.5 * W);
        const camY = Math.floor((1.0 - (ndcY + 1.0) * 0.5) * H);
        if (camX >= 0 && camX < W && camY >= 0 && camY < H) {
          if (cameraBinaryMask[camY * W + camX]) {
            quadBinaryMask[rowOffset + qx] = 1;
            if (qy < quadMinY) quadMinY = qy;
          }
        }
      }
    }

    // Build RGBA overlay for the 3D quad
    const quadOverlayRgba = new Uint8ClampedArray(W * H * 4);
    for (let y = 0; y < H; y++) {
      const rowOffset = y * W;
      for (let x = 0; x < W; x++) {
        const i = rowOffset + x;
        if (!quadBinaryMask[i]) continue;

        const p = i * 4;
        const isEdge =
          x > 0 &&
          x < W - 1 &&
          y > 0 &&
          y < H - 1 &&
          (!quadBinaryMask[i - 1] ||
            !quadBinaryMask[i + 1] ||
            !quadBinaryMask[i - W] ||
            !quadBinaryMask[i + W]);

        if (isEdge) {
          quadOverlayRgba[p] = 224;
          quadOverlayRgba[p + 1] = 242;
          quadOverlayRgba[p + 2] = 254;
          quadOverlayRgba[p + 3] = 250;
        } else {
          quadOverlayRgba[p] = 56;
          quadOverlayRgba[p + 1] = 189;
          quadOverlayRgba[p + 2] = 248;
          quadOverlayRgba[p + 3] = 115;
        }
      }
    }

    const {cutoutRgbaBuffer, cropW, cropH} = extractCutoutFromMask(
      rgba,
      cameraBinaryMask,
      fgCount,
      camMinX,
      camMinY,
      camMaxX,
      camMaxY
    );

    // Seed & fuse Frame 0 into ObjectTSDFVolume if depth is available
    let tsdfMesh: WorkerTsdfMeshPayload | null = null;
    if (
      fgCount > 16 &&
      payload.rgbClipFromWorldMatrixBuffer instanceof ArrayBuffer
    ) {
      const rgbClipFromWorldMatrix = new Float32Array(
        payload.rgbClipFromWorldMatrixBuffer
      );
      tsdfMesh = await seedAndIntegrateInitialVolume(
        payload,
        cameraBinaryMask,
        rgba,
        rgbClipFromWorldMatrix
      );
      if (!tsdfMesh) {
        // Cache mask + RGB so the very next depth frame can lazily seed the volume
        pendingSeedCache = {
          cameraBinaryMask,
          rgba: new Uint8ClampedArray(rgba),
          rgbClipFromWorldMatrix,
        };
      }
    }

    return {
      quadOverlayBuffer: quadOverlayRgba.buffer,
      cutoutRgbaBuffer,
      cropW,
      cropH,
      quadMinY,
      fgCount,
      encoderMs,
      decoderMs,
      totalMs,
      bestIou,
      tsdfMesh,
    };
  }

  async function handleTsdfIntegrateDepth(
    payload: Record<string, unknown>
  ): Promise<WorkerTsdfMeshPayload | null> {
    // Only used when lazily seeding Frame 0 if depth wasn't ready on the exact circle release frame
    if (!activeTsdfVolume && pendingSeedCache) {
      return seedAndIntegrateInitialVolume(
        payload,
        pendingSeedCache.cameraBinaryMask,
        pendingSeedCache.rgba,
        pendingSeedCache.rgbClipFromWorldMatrix
      );
    }
    return null;
  }

  async function handleTsdfIntegrateRgbMask(
    payload: Record<string, unknown>
  ): Promise<{
    tsdfMesh: WorkerTsdfMeshPayload | null;
    cutoutRgbaBuffer: ArrayBuffer | null;
    cropW: number;
    cropH: number;
    bestIou: number;
    totalSamMs: number;
  } | null> {
    if (!activeTsdfVolume && pendingSeedCache) {
      const tsdfMesh = await seedAndIntegrateInitialVolume(
        payload,
        pendingSeedCache.cameraBinaryMask,
        pendingSeedCache.rgba,
        pendingSeedCache.rgbClipFromWorldMatrix
      );
      return tsdfMesh
        ? {
            tsdfMesh,
            cutoutRgbaBuffer: null,
            cropW: 0,
            cropH: 0,
            bestIou: 1.0,
            totalSamMs: 0,
          }
        : null;
    }

    if (!activeTsdfVolume) {
      return null;
    }

    const rgbaBuffer = payload.rgbaBuffer as ArrayBuffer | undefined;
    const rgbClipFromWorldMatrixBuffer =
      payload.rgbClipFromWorldMatrixBuffer as ArrayBuffer | undefined;
    const depthBuffer = payload.depthBuffer as ArrayBuffer | undefined;
    const depthViewMatrixBuffer = payload.depthViewMatrixBuffer as
      | ArrayBuffer
      | undefined;
    const depthProjectionMatrixBuffer = payload.depthProjectionMatrixBuffer as
      | ArrayBuffer
      | undefined;

    if (
      !rgbaBuffer ||
      !rgbClipFromWorldMatrixBuffer ||
      !depthBuffer ||
      !depthViewMatrixBuffer ||
      !depthProjectionMatrixBuffer
    ) {
      return null;
    }

    const {tsdf, graphCut} = await ensureWorkerModules();

    const rgbClipFromWorldMatrix = new Float32Array(
      rgbClipFromWorldMatrixBuffer
    );
    const box2d = activeTsdfVolume.projectBoundingBoxToCamera(
      rgbClipFromWorldMatrix,
      MODEL_IMG_SIZE,
      MODEL_IMG_SIZE
    );
    if (!box2d) {
      return null;
    }

    // Require the object center to be comfortably within the camera frame
    if (
      box2d.cx < 40 ||
      box2d.cx > MODEL_IMG_SIZE - 40 ||
      box2d.cy < 40 ||
      box2d.cy > MODEL_IMG_SIZE - 40
    ) {
      return null;
    }

    // Build automatic 2D bounding-box + center prompt from the projected 3D volume
    const pts = new Float32Array(MAX_POINTS * 2).fill(-1.0);
    const lbls = new Float32Array(MAX_POINTS).fill(-1.0);
    pts[0] = box2d.x1;
    pts[1] = box2d.y1;
    lbls[0] = 2.0;
    pts[2] = box2d.x2;
    pts[3] = box2d.y2;
    lbls[1] = 3.0;
    pts[4] = box2d.cx;
    pts[5] = box2d.cy;
    lbls[2] = 1.0;

    const rgba = new Uint8ClampedArray(rgbaBuffer);
    const samRes = await runSamInferenceOnRgba(rgba, pts, lbls);

    if (samRes.bestIou < 0.65 || samRes.fgCount < 48) {
      return null;
    }

    const t0 = performance.now();
    const depthWidth = Number(payload.depthWidth ?? 160);
    const depthHeight = Number(payload.depthHeight ?? 160);
    const rawValueToMeters = Number(payload.rawValueToMeters ?? 1.0);
    const depthFormat = String(payload.depthFormat ?? 'float32');
    const depthData = decodeDepthBuffer(depthBuffer, depthFormat);

    const depthViewMatrix = new Float32Array(depthViewMatrixBuffer);
    const depthProjectionMatrix = new Float32Array(depthProjectionMatrixBuffer);
    const normDepthBufferFromNormViewMatrix =
      payload.normDepthBufferFromNormViewMatrixBuffer instanceof ArrayBuffer
        ? new Float32Array(payload.normDepthBufferFromNormViewMatrixBuffer)
        : undefined;

    const updatedVoxels = activeTsdfVolume.integrateFrame({
      depthData,
      depthWidth,
      depthHeight,
      rawValueToMeters,
      depthViewMatrix,
      depthProjectionMatrix,
      normDepthBufferFromNormViewMatrix,
      rgbClipFromWorldMatrix,
      rgbaData: rgba,
      rgbWidth: MODEL_IMG_SIZE,
      rgbHeight: MODEL_IMG_SIZE,
      cameraBinaryMask: samRes.cameraBinaryMask,
      maskWidth: MODEL_IMG_SIZE,
      maskHeight: MODEL_IMG_SIZE,
      carveOutsideMask: true,
    });

    if (updatedVoxels < 36) {
      return null;
    }

    // Only admit this view into the Graph-Cut texture keyframe bank if it genuinely
    // observed and updated the 3D object's surface depth!
    addOrUpdateTextureKeyframe(
      rgba,
      samRes.cameraBinaryMask,
      rgbClipFromWorldMatrix,
      depthData,
      depthWidth,
      depthHeight,
      rawValueToMeters,
      depthViewMatrix,
      depthProjectionMatrix,
      normDepthBufferFromNormViewMatrix,
      tsdf,
      graphCut
    );

    const tsdfMesh = buildTexturedMeshPayload(
      activeTsdfVolume,
      updatedVoxels,
      rgbClipFromWorldMatrix,
      graphCut,
      t0
    );

    const {cutoutRgbaBuffer, cropW, cropH} = extractCutoutFromMask(
      rgba,
      samRes.cameraBinaryMask,
      samRes.fgCount,
      samRes.camMinX,
      samRes.camMinY,
      samRes.camMaxX,
      samRes.camMaxY
    );

    return {
      tsdfMesh,
      cutoutRgbaBuffer,
      cropW,
      cropH,
      bestIou: samRes.bestIou,
      totalSamMs: samRes.totalMs,
    };
  }

  self.addEventListener(
    'message',
    async (event: MessageEvent<WorkerRequestMessage>) => {
      const {id, type, payload = {}} = event.data;
      try {
        if (type === 'init') {
          const result = await handleInit();
          self.postMessage({id, ok: true, result});
        } else if (type === 'xr_segment') {
          const result = await handleXrSegment(payload);
          const transferList: Transferable[] = [result.quadOverlayBuffer];
          if (result.cutoutRgbaBuffer) {
            transferList.push(result.cutoutRgbaBuffer);
          }
          if (result.tsdfMesh) {
            transferList.push(
              result.tsdfMesh.positionsBuffer,
              result.tsdfMesh.normalsBuffer,
              result.tsdfMesh.uvsBuffer,
              result.tsdfMesh.colorsBuffer,
              result.tsdfMesh.atlasRgbaBuffer
            );
          }
          self.postMessage({id, ok: true, result}, {transfer: transferList});
        } else if (type === 'tsdf_integrate_depth') {
          const result = await handleTsdfIntegrateDepth(payload);
          const transferList: Transferable[] = [];
          if (result) {
            transferList.push(
              result.positionsBuffer,
              result.normalsBuffer,
              result.uvsBuffer,
              result.colorsBuffer,
              result.atlasRgbaBuffer
            );
          }
          self.postMessage({id, ok: true, result}, {transfer: transferList});
        } else if (type === 'tsdf_integrate_rgb_mask') {
          const result = await handleTsdfIntegrateRgbMask(payload);
          const transferList: Transferable[] = [];
          if (result?.tsdfMesh) {
            transferList.push(
              result.tsdfMesh.positionsBuffer,
              result.tsdfMesh.normalsBuffer,
              result.tsdfMesh.uvsBuffer,
              result.tsdfMesh.colorsBuffer,
              result.tsdfMesh.atlasRgbaBuffer
            );
          }
          if (result?.cutoutRgbaBuffer) {
            transferList.push(result.cutoutRgbaBuffer);
          }
          self.postMessage({id, ok: true, result}, {transfer: transferList});
        } else if (type === 'tsdf_reset') {
          activeTsdfVolume = null;
          textureKeyframes = [];
          pendingSeedCache = null;
          self.postMessage({id, ok: true, result: {reset: true}});
        } else {
          throw new Error(`Unknown worker command: ${String(type)}`);
        }
      } catch (err) {
        const error = err instanceof Error ? err.message : String(err);
        self.postMessage({id, ok: false, error});
      }
    }
  );
})();
