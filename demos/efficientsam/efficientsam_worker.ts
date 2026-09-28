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
      modelPathOrBytes: string | Uint8Array,
      options?: {accelerator?: Accelerator}
    ): Promise<LiteRtCompiledModel>;
    supportsFeature(
      feature: 'relaxedSimd' | 'threads' | 'jspi'
    ): Promise<boolean>;
    Tensor: LiteRtTensorConstructor;
  }

  const LITERT_CORE_ESM_URL =
    'https://cdn.jsdelivr.net/npm/@litertjs/core@2.5.3/+esm';
  const LITERT_WASM_URL =
    'https://cdn.jsdelivr.net/npm/@litertjs/core@2.5.3/wasm/';

  const MODEL_IMG_SIZE = 512;
  const MASK_LOW_RES = 128;
  const MAX_POINTS = 6;

  const ENCODER_MODEL_URL =
    'https://rawcdn.githack.com/xrblocks/proprietary-assets/21bcc2a3e5a44a05b778889244a212d33acaf119/tflite_models/efficientsam/efficientsam_ti_encoder.tflite';
  const DECODER_MODEL_URL =
    'https://rawcdn.githack.com/xrblocks/proprietary-assets/21bcc2a3e5a44a05b778889244a212d33acaf119/tflite_models/efficientsam/efficientsam_ti_decoder.tflite';

  const MOGE_HF_BASE =
    'https://huggingface.co/litert-community/MoGe-2-LiteRT/resolve/main/';
  const MOGE_MODEL_URLS: Record<Accelerator, string> = {
    webgpu: `${MOGE_HF_BASE}moge_fp16.tflite`,
    wasm: `${MOGE_HF_BASE}moge.tflite`,
  };
  const MOGE_MODEL_SIZES_MB: Record<Accelerator, number> = {
    webgpu: 71,
    wasm: 136,
  };
  const MOGE_CACHE_NAME = 'xrblocks-photo-to-3d-v1';
  const MOGE_SIZE = 448;
  const MOGE_MASK_THRESHOLD = 0.5;

  let litertMod: LiteRtCoreModule | null = null;
  let hasWebGpuSupport = false;
  let encoderModel: LiteRtCompiledModel | null = null;
  let decoderModel: LiteRtCompiledModel | null = null;
  let encoderAccelerator: Accelerator = 'wasm';
  let decoderAccelerator: Accelerator = 'wasm';

  let mogeModel: LiteRtCompiledModel | null = null;
  let mogeAccelerator: Accelerator = 'wasm';
  let mogeInitPromise: Promise<{
    mogeAccelerator: Accelerator;
    mogeWarmupMs: number;
  }> | null = null;

  interface WorkerRequestMessage {
    id: number;
    type: 'init' | 'xr_segment' | 'init_moge' | 'xr_moge_twin';
    payload?: Record<string, unknown>;
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
    hasWebGpuSupport = hasWebGpu;
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

  async function handleXrSegment(payload: Record<string, unknown>): Promise<{
    quadOverlayBuffer: ArrayBuffer;
    cutoutRgbaBuffer: ArrayBuffer | null;
    cameraBinaryMaskBuffer: ArrayBuffer;
    cropW: number;
    cropH: number;
    quadMinY: number;
    fgCount: number;
    encoderMs: number;
    decoderMs: number;
    totalMs: number;
    bestIou: number;
  }> {
    if (!litertMod || !encoderModel || !decoderModel) {
      throw new Error('LiteRT models are not initialized yet.');
    }

    const rgbaBuffer = payload.rgbaBuffer as ArrayBuffer;
    const pts = new Float32Array(payload.ptsBuffer as ArrayBuffer);
    const lbls = new Float32Array(payload.lblsBuffer as ArrayBuffer);
    const quadToClipElements = new Float32Array(
      payload.quadToClipBuffer as ArrayBuffer
    );
    const quadSizeMeters = Number(payload.quadSizeMeters ?? 0.8);

    const rgba = new Uint8ClampedArray(rgbaBuffer);
    const inputFloat32 = rgbaToPlanarFloat32(rgba);

    // 1. Run LiteRT Encoder & 2. Run LiteRT Decoder with guaranteed tensor cleanup
    let inputTensor: Tensor | null = null;
    let ptsTensor: Tensor | null = null;
    let lblsTensor: Tensor | null = null;
    let encOutputs: Tensor[] = [];
    let decOutputs: Tensor[] = [];
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

    // 3. Bilinearly upsample 128x128 mask logits -> 512x512 camera binary mask
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

    // 4. Project cameraBinaryMask onto circleQuad's 512x512 UV space
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

    // 5. Build RGBA overlay for the 3D quad
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

    // 6. Extract cropped RGBA cutout of the segmented object in camera space
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

    return {
      quadOverlayBuffer: quadOverlayRgba.buffer,
      cutoutRgbaBuffer,
      cameraBinaryMaskBuffer: cameraBinaryMask.buffer,
      cropW,
      cropH,
      quadMinY,
      fgCount,
      encoderMs,
      decoderMs,
      totalMs,
      bestIou,
    };
  }

  async function fetchCachedModelInWorker(
    url: string,
    expectedMb: number,
    onProgress?: (status: string) => void
  ): Promise<Uint8Array> {
    let cache: Cache | null = null;
    if ('caches' in self) {
      try {
        cache = await self.caches.open(MOGE_CACHE_NAME);
        const hit = await cache.match(url);
        if (hit) {
          return new Uint8Array(await hit.arrayBuffer());
        }
      } catch {
        cache = null;
      }
    }

    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch ${url} (HTTP ${response.status})`);
    }
    const total = Number(response.headers.get('Content-Length')) || 0;
    if (!response.body) {
      const bytes = new Uint8Array(await response.arrayBuffer());
      if (cache) {
        await cache.put(url, new Response(bytes.slice())).catch(() => {});
      }
      return bytes;
    }

    const reader = response.body.getReader();
    const chunks: Uint8Array[] = [];
    let received = 0;
    for (;;) {
      const {done, value} = await reader.read();
      if (done) break;
      if (value) {
        chunks.push(value);
        received += value.byteLength;
        const mb = (received / 1048576).toFixed(0);
        const pct = total ? ` (${Math.round((100 * received) / total)}%)` : '';
        onProgress?.(
          `Downloading MoGe-2 3D model… ${mb}/${expectedMb} MB${pct}`
        );
      }
    }
    const combined = new Uint8Array(received);
    let offset = 0;
    for (const chunk of chunks) {
      combined.set(chunk, offset);
      offset += chunk.byteLength;
    }
    if (cache) {
      await cache.put(url, new Response(combined.slice())).catch(() => {});
    }
    return combined;
  }

  function unitLengthError(map: Float32Array): number {
    const pixels = map.length / 3;
    const step = Math.max(1, Math.floor(pixels / 5000));
    let sum = 0;
    let n = 0;
    for (let p = 0; p < pixels; p += step) {
      const x = map[p * 3];
      const y = map[p * 3 + 1];
      const z = map[p * 3 + 2];
      const len = Math.sqrt(x * x + y * y + z * z);
      if (!Number.isFinite(len)) continue;
      sum += Math.abs(len - 1);
      n++;
    }
    return n ? sum / n : Infinity;
  }

  function resolveMogeOutputs(buffers: Float32Array[]): {
    points: Float32Array;
    mask: Float32Array;
    scale: number;
  } {
    const plane = MOGE_SIZE * MOGE_SIZE;
    const big = buffers.filter((b) => b.length === plane * 3);
    const mask = buffers.find((b) => b.length === plane);
    const scaleBuf = buffers.find((b) => b.length === 1);
    if (big.length < 2 || !mask) {
      throw new Error('Unexpected MoGe-2 output tensor shapes.');
    }
    const points =
      unitLengthError(big[0]) >= unitLengthError(big[1]) ? big[0] : big[1];
    const scale =
      scaleBuf && Number.isFinite(scaleBuf[0]) && scaleBuf[0] > 0
        ? scaleBuf[0]
        : 1.0;
    return {points, mask, scale};
  }

  async function runMogeRaw(nchw: Float32Array): Promise<{
    points: Float32Array;
    mask: Float32Array;
    scale: number;
  }> {
    if (!litertMod || !mogeModel) {
      throw new Error('MoGe-2 model is not initialized.');
    }
    let inputTensor: LiteRtTensor | null = null;
    let outputs: LiteRtTensor[] = [];
    try {
      inputTensor = new litertMod.Tensor(nchw, [1, 3, MOGE_SIZE, MOGE_SIZE]);
      outputs = await mogeModel.run([inputTensor]);
      const buffers: Float32Array[] = [];
      for (const out of outputs) {
        buffers.push(new Float32Array((await out.data()) as Float32Array));
      }
      return resolveMogeOutputs(buffers);
    } finally {
      for (const t of [inputTensor, ...outputs]) {
        if (t && !t.deleted) {
          t.delete();
        }
      }
    }
  }

  async function handleInitMoge(): Promise<{
    mogeAccelerator: Accelerator;
    mogeWarmupMs: number;
    compileTimeMs: number;
  }> {
    if (mogeModel) {
      return {mogeAccelerator, mogeWarmupMs: 0, compileTimeMs: 0};
    }
    if (mogeInitPromise) {
      const res = await mogeInitPromise;
      return {...res, compileTimeMs: res.mogeWarmupMs};
    }

    mogeInitPromise = (async () => {
      if (!litertMod) {
        await handleInit();
      }
      const acceleratorsToTry: Accelerator[] = hasWebGpuSupport
        ? ['webgpu', 'wasm']
        : ['wasm'];

      for (const accel of acceleratorsToTry) {
        let candidate: LiteRtCompiledModel | null = null;
        try {
          const bytes = await fetchCachedModelInWorker(
            MOGE_MODEL_URLS[accel],
            MOGE_MODEL_SIZES_MB[accel],
            (status) => {
              self.postMessage({type: 'moge_status', status});
            }
          );
          self.postMessage({
            type: 'moge_status',
            status: `Compiling MoGe-2 (${accel})…`,
          });
          candidate = await litertMod!.loadAndCompile(bytes, {
            accelerator: accel,
          });
          mogeModel = candidate;
          mogeAccelerator = accel;

          self.postMessage({
            type: 'moge_status',
            status: `Warming up MoGe-2 (${accel})…`,
          });
          const tWarm0 = performance.now();
          const dummy = new Float32Array(3 * MOGE_SIZE * MOGE_SIZE).fill(0.5);
          await runMogeRaw(dummy);
          const mogeWarmupMs = performance.now() - tWarm0;
          return {
            mogeAccelerator,
            mogeWarmupMs,
            compileTimeMs: mogeWarmupMs,
          };
        } catch (err) {
          console.warn(
            `[EfficientSAM Worker] MoGe-2 '${accel}' init failed, falling back:`,
            err
          );
          if (candidate && !candidate.deleted) {
            candidate.delete();
          }
          mogeModel = null;
        }
      }
      throw new Error('Could not initialize MoGe-2 on any LiteRT accelerator.');
    })();

    try {
      const res = await mogeInitPromise;
      return {...res, compileTimeMs: res.mogeWarmupMs};
    } catch (err) {
      mogeInitPromise = null;
      throw err;
    }
  }

  function median(values: number[]): number {
    const sorted = values.slice().sort((a, b) => a - b);
    return sorted[sorted.length >> 1] ?? 0;
  }

  async function handleMoGeTwin(payload: Record<string, unknown>): Promise<{
    positionsBuffer: ArrayBuffer;
    uvsBuffer: ArrayBuffer;
    indicesBuffer: ArrayBuffer;
    worldOrigin: [number, number, number];
    bboxTopWorld: [number, number, number];
    vertexCount: number;
    triangleCount: number;
    mogeMs: number;
  }> {
    if (!mogeModel) {
      await handleInitMoge();
    }

    const rgba = new Uint8ClampedArray(payload.rgbaBuffer as ArrayBuffer);
    const maskArrayBuffer = (payload.maskBuffer ??
      payload.cameraBinaryMaskBuffer) as ArrayBuffer;
    const samMask512 = new Uint8Array(maskArrayBuffer);
    const clipFromView = payload.clipFromViewBuffer
      ? new Float32Array(payload.clipFromViewBuffer as ArrayBuffer)
      : null;
    const worldFromView = payload.worldFromViewBuffer
      ? new Float32Array(payload.worldFromViewBuffer as ArrayBuffer)
      : null;
    const anchorDepthMeters = Number(payload.anchorDepthMeters ?? 0);

    // 1. Downsample 512x512 camera RGBA and EfficientSAM binary mask to 448x448
    const SIZE = MOGE_SIZE;
    const plane = SIZE * SIZE;
    const nchw = new Float32Array(3 * plane);
    const samMask448 = new Uint8Array(plane);
    const scaleRatio = MODEL_IMG_SIZE / SIZE;
    const inv255 = 1.0 / 255.0;

    for (let y = 0; y < SIZE; y++) {
      const srcY = Math.min(MODEL_IMG_SIZE - 1, Math.floor(y * scaleRatio));
      for (let x = 0; x < SIZE; x++) {
        const srcX = Math.min(MODEL_IMG_SIZE - 1, Math.floor(x * scaleRatio));
        const srcIdx = srcY * MODEL_IMG_SIZE + srcX;
        const dstIdx = y * SIZE + x;
        const srcP = srcIdx * 4;

        nchw[dstIdx] = rgba[srcP] * inv255;
        nchw[plane + dstIdx] = rgba[srcP + 1] * inv255;
        nchw[2 * plane + dstIdx] = rgba[srcP + 2] * inv255;

        samMask448[dstIdx] = samMask512[srcIdx];
      }
    }

    // 2. Erode SAM mask by 1px to trim mixed boundary pixels while preserving thin legs/structures
    const erodedMask448 = new Uint8Array(plane);
    let erodedCount = 0;
    for (let y = 1; y < SIZE - 1; y++) {
      for (let x = 1; x < SIZE - 1; x++) {
        const i = y * SIZE + x;
        if (
          samMask448[i] &&
          samMask448[i - 1] &&
          samMask448[i + 1] &&
          samMask448[i - SIZE] &&
          samMask448[i + SIZE]
        ) {
          erodedMask448[i] = 1;
          erodedCount++;
        }
      }
    }
    const activeMask = erodedCount >= 32 ? erodedMask448 : samMask448;

    // 3. Run MoGe-2 inference
    const tMoge0 = performance.now();
    const {points, mask: mogeMask, scale: mogeScale} = await runMogeRaw(nchw);
    const mogeMs = performance.now() - tMoge0;

    // 4. Collect valid MoGe forward depths inside the segmented object mask
    const rawDepthGrid = new Float32Array(plane);
    const depthSamples: number[] = [];
    for (let pass = 0; pass < 2; pass++) {
      const requireConfidence = pass === 0;
      for (let i = 0; i < plane; i++) {
        if (!activeMask[i]) continue;
        if (requireConfidence && mogeMask[i] <= MOGE_MASK_THRESHOLD) continue;
        const z = points[i * 3 + 2];
        if (!Number.isFinite(z) || z <= 1e-4) continue;
        rawDepthGrid[i] = z;
        depthSamples.push(z);
      }
      if (depthSamples.length >= 24) break;
      rawDepthGrid.fill(0);
      depthSamples.length = 0;
    }

    if (depthSamples.length === 0) {
      throw new Error('No valid 3D surface found inside the segmented mask.');
    }

    const medianMogeDepth = Math.max(median(depthSamples), 1e-5);
    const trueAnchorDepth =
      anchorDepthMeters > 0.1 && Number.isFinite(anchorDepthMeters)
        ? anchorDepthMeters
        : Math.max(0.35, Math.min(8.0, medianMogeDepth * mogeScale));

    // 5. Convert MoGe relative depth to metric depth anchored at trueAnchorDepth,
    // clamping depth excursions so silhouette bleed never stretches triangles into the background.
    let depthGrid = new Float32Array(plane);
    for (let i = 0; i < plane; i++) {
      const z = rawDepthGrid[i];
      if (z <= 0) continue;
      const ratio = Math.max(0.76, Math.min(1.28, z / medianMogeDepth));
      depthGrid[i] = trueAnchorDepth * ratio;
    }

    // 6. Smooth the metric depth map (2 passes of 3x3 neighbor averaging inside the mask)
    for (let iter = 0; iter < 2; iter++) {
      const nextGrid = new Float32Array(plane);
      for (let y = 0; y < SIZE; y++) {
        for (let x = 0; x < SIZE; x++) {
          const i = y * SIZE + x;
          const centerD = depthGrid[i];
          if (centerD <= 0) continue;
          let sum = centerD * 2.0;
          let weight = 2.0;
          const y0 = Math.max(0, y - 1);
          const y1 = Math.min(SIZE - 1, y + 1);
          const x0 = Math.max(0, x - 1);
          const x1 = Math.min(SIZE - 1, x + 1);
          for (let ny = y0; ny <= y1; ny++) {
            for (let nx = x0; nx <= x1; nx++) {
              const nd = depthGrid[ny * SIZE + nx];
              if (nd > 0 && Math.abs(nd - centerD) < 0.18 * trueAnchorDepth) {
                sum += nd;
                weight += 1.0;
              }
            }
          }
          nextGrid[i] = sum / weight;
        }
      }
      depthGrid = nextGrid;
    }

    // 7. Unproject every valid mask pixel into 3D world space using the camera's exact intrinsics & pose
    const fx =
      clipFromView && Math.abs(clipFromView[0]) > 1e-5 ? clipFromView[0] : 1.0;
    const fy =
      clipFromView && Math.abs(clipFromView[5]) > 1e-5 ? clipFromView[5] : 1.0;
    const m =
      worldFromView && worldFromView.length === 16
        ? worldFromView
        : new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1.6, 0, 1]);

    const vertexIndexMap = new Int32Array(plane).fill(-1);
    const worldPositions: number[] = [];
    const uvs: number[] = [];
    let minX = Infinity;
    let minY = Infinity;
    let minZ = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    let maxZ = -Infinity;

    for (let y = 0; y < SIZE; y++) {
      const v = (y + 0.5) / SIZE;
      const ndcY = 1.0 - 2.0 * v;
      for (let x = 0; x < SIZE; x++) {
        const i = y * SIZE + x;
        const d = depthGrid[i];
        if (d <= 0) continue;

        const u = (x + 0.5) / SIZE;
        const ndcX = 2.0 * u - 1.0;
        // Shift 1.5 cm toward the camera so the in-place mesh sits cleanly in front of the real surface
        const zMetric = Math.max(0.1, d - 0.015);

        const vx = (ndcX / fx) * zMetric;
        const vy = (ndcY / fy) * zMetric;
        const vz = -zMetric;

        const wx = m[0] * vx + m[4] * vy + m[8] * vz + m[12];
        const wy = m[1] * vx + m[5] * vy + m[9] * vz + m[13];
        const wz = m[2] * vx + m[6] * vy + m[10] * vz + m[14];

        const vIdx = worldPositions.length / 3;
        vertexIndexMap[i] = vIdx;
        worldPositions.push(wx, wy, wz);
        uvs.push(u, 1.0 - v);

        if (wx < minX) minX = wx;
        if (wy < minY) minY = wy;
        if (wz < minZ) minZ = wz;
        if (wx > maxX) maxX = wx;
        if (wy > maxY) maxY = wy;
        if (wz > maxZ) maxZ = wz;
      }
    }

    const vertexCount = worldPositions.length / 3;
    if (vertexCount < 3) {
      throw new Error('Insufficient vertices to build 3D Digital Twin mesh.');
    }

    // 8. Triangulate adjacent 2x2 grid cells inside the segmented mask
    const indices: number[] = [];
    const maxStepJump = 0.14 * trueAnchorDepth;
    const canConnect = (idxA: number, idxB: number): boolean => {
      return Math.abs(depthGrid[idxA] - depthGrid[idxB]) <= maxStepJump;
    };

    for (let y = 0; y < SIZE - 1; y++) {
      const row = y * SIZE;
      const nextRow = (y + 1) * SIZE;
      for (let x = 0; x < SIZE - 1; x++) {
        const p00 = row + x;
        const p10 = row + x + 1;
        const p01 = nextRow + x;
        const p11 = nextRow + x + 1;

        const i00 = vertexIndexMap[p00];
        const i10 = vertexIndexMap[p10];
        const i01 = vertexIndexMap[p01];
        const i11 = vertexIndexMap[p11];

        const has00 = i00 >= 0;
        const has10 = i10 >= 0;
        const has01 = i01 >= 0;
        const has11 = i11 >= 0;

        if (has00 && has10 && has01 && has11) {
          if (
            canConnect(p00, p01) &&
            canConnect(p00, p10) &&
            canConnect(p01, p10)
          ) {
            indices.push(i00, i01, i10);
          }
          if (
            canConnect(p10, p01) &&
            canConnect(p10, p11) &&
            canConnect(p01, p11)
          ) {
            indices.push(i10, i01, i11);
          }
        } else if (has00 && has01 && has10) {
          if (
            canConnect(p00, p01) &&
            canConnect(p00, p10) &&
            canConnect(p01, p10)
          ) {
            indices.push(i00, i01, i10);
          }
        } else if (has10 && has01 && has11) {
          if (
            canConnect(p10, p01) &&
            canConnect(p10, p11) &&
            canConnect(p01, p11)
          ) {
            indices.push(i10, i01, i11);
          }
        } else if (has00 && has01 && has11) {
          if (
            canConnect(p00, p01) &&
            canConnect(p01, p11) &&
            canConnect(p00, p11)
          ) {
            indices.push(i00, i01, i11);
          }
        } else if (has00 && has11 && has10) {
          if (
            canConnect(p00, p11) &&
            canConnect(p11, p10) &&
            canConnect(p00, p10)
          ) {
            indices.push(i00, i11, i10);
          }
        }
      }
    }

    const triangleCount = indices.length / 3;
    if (triangleCount === 0) {
      throw new Error('Could not triangulate segmented 3D surface.');
    }

    // 9. Express mesh vertices relative to the object's true world-space bottom-center origin
    const originX = 0.5 * (minX + maxX);
    const originY = minY;
    const originZ = 0.5 * (minZ + maxZ);

    const positionsFloat32 = new Float32Array(worldPositions.length);
    for (let i = 0; i < vertexCount; i++) {
      positionsFloat32[i * 3] = worldPositions[i * 3] - originX;
      positionsFloat32[i * 3 + 1] = worldPositions[i * 3 + 1] - originY;
      positionsFloat32[i * 3 + 2] = worldPositions[i * 3 + 2] - originZ;
    }
    const uvsFloat32 = new Float32Array(uvs);
    const indicesUint32 = new Uint32Array(indices);

    return {
      positionsBuffer: positionsFloat32.buffer,
      uvsBuffer: uvsFloat32.buffer,
      indicesBuffer: indicesUint32.buffer,
      worldOrigin: [originX, originY, originZ],
      bboxTopWorld: [originX, maxY, originZ],
      vertexCount,
      triangleCount,
      mogeMs,
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
          const transferList: Transferable[] = [
            result.quadOverlayBuffer,
            result.cameraBinaryMaskBuffer,
          ];
          if (result.cutoutRgbaBuffer) {
            transferList.push(result.cutoutRgbaBuffer);
          }
          self.postMessage({id, ok: true, result}, {transfer: transferList});
        } else if (type === 'init_moge') {
          const result = await handleInitMoge();
          self.postMessage({id, ok: true, result});
        } else if (type === 'xr_moge_twin') {
          const result = await handleMoGeTwin(payload);
          self.postMessage(
            {id, ok: true, result},
            {
              transfer: [
                result.positionsBuffer,
                result.uvsBuffer,
                result.indicesBuffer,
              ],
            }
          );
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
