/**
 * Dedicated Web Worker for all XR perception compute:
 * 1. YOLOE-11s (365-class open-vocabulary object detection via ONNX Runtime WebGPU/WASM)
 * 2. EfficientSAM-Ti (image encoder & mask decoder via LiteRT WebGPU/WASM)
 * 3. Direct 128x128 mask selection, 64-ray contour extraction, and 32x32 foreground mask generation
 *    (eliminating the 512x512 bilinear upsampling bottleneck)
 * 4. Single-pass 48x48 buffer-indexed CV template & rotation tracking
 */

import {
  Tensor as LiteRtTensor,
  loadAndCompile,
  loadLiteRt,
  supportsFeature,
} from 'https://cdn.jsdelivr.net/npm/@litertjs/core@2.5.3/+esm';

const LITERT_WASM_URL =
  'https://cdn.jsdelivr.net/npm/@litertjs/core@2.5.3/wasm/';
const ORT_ESM_URL =
  'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.webgpu.min.mjs';
const ORT_WASM_DIR =
  'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';

const MODEL_CACHE_NAME = 'xrblocks-litert-models-v1';

const SAM_IMG_SIZE = 512;
const SAM_MASK_LOW_RES = 128;
const SAM_EMBED_DIM = 256;
const SAM_EMBED_GRID = 32;
const SAM_MAX_POINTS = 6;

const ENCODER_MODEL_URL =
  'https://rawcdn.githack.com/xrblocks/proprietary-assets/15e67828b08e0b05160cc8d497106bd600312517/tflite_models/efficientsam/efficientsam_ti_encoder.tflite';
const DECODER_MODEL_URL =
  'https://rawcdn.githack.com/xrblocks/proprietary-assets/15e67828b08e0b05160cc8d497106bd600312517/tflite_models/efficientsam/efficientsam_ti_decoder.tflite';

const YOLOE_IMG_SIZE = 640;
const YOLOE_NUM_ANCHORS = 8400;
const YOLOE_MODEL_URL =
  'https://huggingface.co/wide-video/yoloe-11-Objects365-v1.0.0/resolve/main/yoloe-11s-seg-object365.onnx';

const CONF_SURVIVAL_THRESHOLD = 0.14;

const OBJECTS365_NAMES: readonly string[] = [
  'Person',
  'Sneakers',
  'Chair',
  'Other Shoes',
  'Hat',
  'Car',
  'Lamp',
  'Glasses',
  'Bottle',
  'Desk',
  'Cup',
  'Street Lights',
  'Cabinet/shelf',
  'Handbag/Satchel',
  'Bracelet',
  'Plate',
  'Picture/Frame',
  'Helmet',
  'Book',
  'Gloves',
  'Storage box',
  'Boat',
  'Leather Shoes',
  'Flower',
  'Bench',
  'Potted Plant',
  'Bowl/Basin',
  'Flag',
  'Pillow',
  'Boots',
  'Vase',
  'Microphone',
  'Necklace',
  'Ring',
  'SUV',
  'Wine Glass',
  'Belt',
  'Monitor/TV',
  'Backpack',
  'Umbrella',
  'Traffic Light',
  'Speaker',
  'Watch',
  'Tie',
  'Trash bin Can',
  'Slippers',
  'Bicycle',
  'Stool',
  'Barrel/bucket',
  'Van',
  'Couch',
  'Sandals',
  'Basket',
  'Drum',
  'Pen/Pencil',
  'Bus',
  'Wild Bird',
  'High Heels',
  'Motorcycle',
  'Guitar',
  'Carpet',
  'Cell Phone',
  'Bread',
  'Camera',
  'Canned',
  'Truck',
  'Traffic cone',
  'Cymbal',
  'Lifesaver',
  'Towel',
  'Stuffed Toy',
  'Candle',
  'Sailboat',
  'Laptop',
  'Awning',
  'Bed',
  'Faucet',
  'Tent',
  'Horse',
  'Mirror',
  'Power outlet',
  'Sink',
  'Apple',
  'Air Conditioner',
  'Knife',
  'Hockey Stick',
  'Paddle',
  'Pickup Truck',
  'Fork',
  'Traffic Sign',
  'Balloon',
  'Tripod',
  'Dog',
  'Spoon',
  'Clock',
  'Pot',
  'Cow',
  'Cake',
  'Dining Table',
  'Sheep',
  'Hanger',
  'Blackboard/Whiteboard',
  'Napkin',
  'Other Fish',
  'Orange/Tangerine',
  'Toiletry',
  'Keyboard',
  'Tomato',
  'Lantern',
  'Machinery Vehicle',
  'Fan',
  'Green Vegetables',
  'Banana',
  'Baseball Glove',
  'Airplane',
  'Mouse',
  'Train',
  'Pumpkin',
  'Soccer',
  'Skiboard',
  'Luggage',
  'Nightstand',
  'Tea pot',
  'Telephone',
  'Trolley',
  'Head Phone',
  'Sports Car',
  'Stop Sign',
  'Dessert',
  'Scooter',
  'Stroller',
  'Crane',
  'Remote',
  'Refrigerator',
  'Oven',
  'Lemon',
  'Duck',
  'Baseball Bat',
  'Surveillance Camera',
  'Cat',
  'Jug',
  'Broccoli',
  'Piano',
  'Pizza',
  'Elephant',
  'Skateboard',
  'Surfboard',
  'Gun',
  'Skating and Skiing shoes',
  'Gas stove',
  'Donut',
  'Bow Tie',
  'Carrot',
  'Toilet',
  'Kite',
  'Strawberry',
  'Other Balls',
  'Shovel',
  'Pepper',
  'Computer Box',
  'Toilet Paper',
  'Cleaning Products',
  'Chopsticks',
  'Microwave',
  'Pigeon',
  'Baseball',
  'Cutting/chopping Board',
  'Coffee Table',
  'Side Table',
  'Scissors',
  'Marker',
  'Pie',
  'Ladder',
  'Snowboard',
  'Cookies',
  'Radiator',
  'Fire Hydrant',
  'Basketball',
  'Zebra',
  'Grape',
  'Giraffe',
  'Potato',
  'Sausage',
  'Tricycle',
  'Violin',
  'Egg',
  'Fire Extinguisher',
  'Candy',
  'Fire Truck',
  'Billiards',
  'Converter',
  'Bathtub',
  'Wheelchair',
  'Golf Club',
  'Briefcase',
  'Cucumber',
  'Cigar/Cigarette',
  'Paint Brush',
  'Pear',
  'Heavy Truck',
  'Hamburger',
  'Extractor',
  'Extension Cord',
  'Tong',
  'Tennis Racket',
  'Folder',
  'American Football',
  'earphone',
  'Mask',
  'Kettle',
  'Tennis',
  'Ship',
  'Swing',
  'Coffee Machine',
  'Slide',
  'Carriage',
  'Onion',
  'Green beans',
  'Projector',
  'Frisbee',
  'Washing Machine/Drying Machine',
  'Chicken',
  'Printer',
  'Watermelon',
  'Saxophone',
  'Tissue',
  'Toothbrush',
  'Ice cream',
  'Hot air balloon',
  'Cello',
  'French Fries',
  'Scale',
  'Trophy',
  'Cabbage',
  'Hot dog',
  'Blender',
  'Peach',
  'Rice',
  'Wallet/Purse',
  'Volleyball',
  'Deer',
  'Goose',
  'Tape',
  'Tablet',
  'Cosmetics',
  'Trumpet',
  'Pineapple',
  'Golf Ball',
  'Ambulance',
  'Parking meter',
  'Mango',
  'Key',
  'Hurdle',
  'Fishing Rod',
  'Medal',
  'Flute',
  'Brush',
  'Penguin',
  'Megaphone',
  'Corn',
  'Lettuce',
  'Garlic',
  'Swan',
  'Helicopter',
  'Green Onion',
  'Sandwich',
  'Nuts',
  'Speed Limit Sign',
  'Induction Cooker',
  'Broom',
  'Trombone',
  'Plum',
  'Rickshaw',
  'Goldfish',
  'Kiwi fruit',
  'Router/modem',
  'Poker Card',
  'Toaster',
  'Shrimp',
  'Sushi',
  'Cheese',
  'Notepaper',
  'Cherry',
  'Pliers',
  'CD',
  'Pasta',
  'Hammer',
  'Cue',
  'Avocado',
  'Hamimelon',
  'Flask',
  'Mushroom',
  'Screwdriver',
  'Soap',
  'Recorder',
  'Bear',
  'Eggplant',
  'Board Eraser',
  'Coconut',
  'Tape Measure/Ruler',
  'Pig',
  'Showerhead',
  'Globe',
  'Chips',
  'Steak',
  'Crosswalk Sign',
  'Stapler',
  'Camel',
  'Formula 1',
  'Pomegranate',
  'Dishwasher',
  'Crab',
  'Hoverboard',
  'Meat ball',
  'Rice Cooker',
  'Tuba',
  'Calculator',
  'Papaya',
  'Antelope',
  'Parrot',
  'Seal',
  'Butterfly',
  'Dumbbell',
  'Donkey',
  'Lion',
  'Urinal',
  'Dolphin',
  'Electric Drill',
  'Hair Dryer',
  'Egg tart',
  'Jellyfish',
  'Treadmill',
  'Lighter',
  'Grapefruit',
  'Game board',
  'Mop',
  'Radish',
  'Baozi',
  'Target',
  'French',
  'Spring Rolls',
  'Monkey',
  'Rabbit',
  'Pencil Case',
  'Yak',
  'Red Cabbage',
  'Binoculars',
  'Asparagus',
  'Barbell',
  'Scallop',
  'Noddles',
  'Comb',
  'Dumpling',
  'Oyster',
  'Table Teniis paddle',
  'Cosmetics Brush/Eyeliner Pencil',
  'Chainsaw',
  'Eraser',
  'Lobster',
  'Durian',
  'Okra',
  'Lipstick',
  'Cosmetics Mirror',
  'Curling',
  'Table Tennis',
];

export interface BoundingBox2D {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface ContourProfile {
  mcx: number;
  mcy: number;
  bcx: number;
  bcy: number;
  bw: number;
  bh: number;
  radii: Float32Array;
}

interface OrtTensor {
  data: Float32Array;
  dims: readonly number[];
  dispose?: () => void;
}

interface OrtSession {
  run(feeds: Record<string, OrtTensor>): Promise<Record<string, OrtTensor>>;
}

interface OrtModule {
  env: {
    wasm: {
      wasmPaths: string;
      numThreads?: number;
    };
    logLevel?: string;
  };
  Tensor: new (
    type: 'float32',
    data: Float32Array,
    dims: readonly number[]
  ) => OrtTensor;
  InferenceSession: {
    create(
      buffer: ArrayBuffer,
      options?: {executionProviders?: string[]}
    ): Promise<OrtSession>;
  };
}

interface CompiledLiteRtModel {
  run(inputs: unknown[]): Promise<
    Array<{
      data(): Promise<Float32Array | ArrayBufferView>;
      deleted?: boolean;
      delete(): void;
    }>
  >;
  options?: {accelerator?: string};
  isFullyAccelerated?: boolean;
  deleted?: boolean;
  delete(): void;
}

// --- Pre-allocated Canvases & TypedArray Buffers (Zero per-frame GC) ---
const canvas512 = new OffscreenCanvas(SAM_IMG_SIZE, SAM_IMG_SIZE);
const ctx512 = canvas512.getContext('2d', {willReadFrequently: true})!;

const canvas640 = new OffscreenCanvas(YOLOE_IMG_SIZE, YOLOE_IMG_SIZE);
const ctx640 = canvas640.getContext('2d', {willReadFrequently: true})!;

const canvas32 = new OffscreenCanvas(32, 32);
const ctx32 = canvas32.getContext('2d', {willReadFrequently: true})!;

const canvas48 = new OffscreenCanvas(48, 48);
const ctx48 = canvas48.getContext('2d', {willReadFrequently: true})!;

const yoloePlanarBuffer = new Float32Array(3 * YOLOE_IMG_SIZE * YOLOE_IMG_SIZE);
const samPlanarBuffer = new Float32Array(3 * SAM_IMG_SIZE * SAM_IMG_SIZE);
const samPtsBuffer = new Float32Array(SAM_MAX_POINTS * 2);
const samLblsBuffer = new Float32Array(SAM_MAX_POINTS);

const bestScoresBuffer = new Float32Array(YOLOE_NUM_ANCHORS);
const bestClassesBuffer = new Int16Array(YOLOE_NUM_ANCHORS);

const patch48Gray = new Uint8ClampedArray(48 * 48);
const cur32Gray = new Uint8ClampedArray(32 * 32);

let ortModule: OrtModule | null = null;
let yoloeSession: OrtSession | null = null;
let yoloeProvider = 'WASM';

let encoderModel: CompiledLiteRtModel | null = null;
let decoderModel: CompiledLiteRtModel | null = null;
let encoderAccelerator = 'wasm';
let decoderAccelerator = 'wasm';

let cachedEmbeddings: Float32Array | null = null;
let currentFrameId = 0;
let encodedFrameId = -1;
let lastEncoderMs = 0;

// --- Fast 32x32 CV Template + Rotation Tracker State inside Worker ---
interface WorkerFastTracker {
  trackId: number;
  templateData: Uint8ClampedArray;
  refTemplate: Uint8ClampedArray;
  fgMask: Uint8Array;
  anchorBox: BoundingBox2D;
  offsetX: number;
  offsetY: number;
  activeAngle: number;
}

let fastTracker: WorkerFastTracker | null = null;

async function fetchCachedModelBytes(url: string): Promise<ArrayBuffer> {
  if (typeof caches !== 'undefined') {
    try {
      const cache = await caches.open(MODEL_CACHE_NAME);
      const hit = await cache.match(url);
      if (hit) {
        return await hit.arrayBuffer();
      }
      const res = await fetch(url, {mode: 'cors'});
      if (res.ok) {
        await cache.put(url, res.clone());
        return await res.arrayBuffer();
      }
    } catch {
      // Fall through to direct fetch
    }
  }
  const response = await fetch(url, {mode: 'cors'});
  if (!response.ok) {
    throw new Error(`Failed to fetch model ${url}: HTTP ${response.status}`);
  }
  return await response.arrayBuffer();
}

async function preloadLiteRtWasmModule(jspi: boolean): Promise<void> {
  const relaxedSimd = await supportsFeature('relaxedSimd').catch(() => false);
  const wasmFileName = !relaxedSimd
    ? 'litert_wasm_compat_internal.js'
    : jspi
      ? 'litert_wasm_jspi_internal.js'
      : 'litert_wasm_internal.js';
  const cjsShim = {exports: {}};
  const workerGlobal = self as unknown as Record<string, unknown>;
  workerGlobal.module = cjsShim;
  workerGlobal.exports = cjsShim.exports;
  workerGlobal.Module = {
    locateFile: (path: string) => `${LITERT_WASM_URL}${path}`,
  };
  const imported = (await import(`${LITERT_WASM_URL}${wasmFileName}`)) as {
    default?: unknown;
  };
  workerGlobal.ModuleFactory = imported.default ?? cjsShim.exports;
  delete workerGlobal.module;
  delete workerGlobal.exports;
  workerGlobal.importScripts = () => {};
}

async function compileLiteRtModelWithFallback(
  modelUrl: string,
  preferredAccelerator: 'webgpu' | 'wasm',
  warmupShapes: number[][]
): Promise<{model: CompiledLiteRtModel; accelerator: string}> {
  const modelBytes = new Uint8Array(await fetchCachedModelBytes(modelUrl));
  const accelerators: Array<'webgpu' | 'wasm'> =
    preferredAccelerator === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm'];

  for (const accelerator of accelerators) {
    let model: CompiledLiteRtModel | null = null;
    const dummyInputs: Array<{deleted?: boolean; delete(): void}> = [];
    try {
      const compiled = (await loadAndCompile(modelBytes, {
        accelerator,
      })) as unknown as CompiledLiteRtModel;
      model = compiled;
      for (const shape of warmupShapes) {
        const size = shape.reduce((a, b) => a * b, 1);
        dummyInputs.push(new LiteRtTensor(new Float32Array(size), shape));
      }
      const outputs = await compiled.run(dummyInputs);
      for (const out of outputs) {
        await out.data();
        if (!out.deleted) out.delete();
      }
      for (const inp of dummyInputs) {
        if (!inp.deleted) inp.delete();
      }
      const actualAcc =
        compiled.options?.accelerator === 'webgpu' ? 'webgpu' : 'wasm';
      return {model: compiled, accelerator: actualAcc};
    } catch (err) {
      console.warn(
        `[PerceptionWorker] LiteRT '${accelerator}' fallback for ${modelUrl}:`,
        err
      );
      for (const inp of dummyInputs) {
        if (!inp.deleted) inp.delete();
      }
      if (model && !model.deleted) model.delete();
    }
  }
  throw new Error(`Could not compile LiteRT model: ${modelUrl}`);
}

async function initSamInWorker(): Promise<void> {
  const jspi = await supportsFeature('jspi').catch(() => false);
  await preloadLiteRtWasmModule(jspi);
  await loadLiteRt(LITERT_WASM_URL, {jspi});
  // Run LiteRT SAM Encoder & Decoder on WASM SIMD inside the DedicatedWorker thread
  // so SAM never submits monolithic 265ms GPUTasks to Chrome's shared CrGpuMain thread.
  const preferred = 'wasm' as const;

  const encRes = await compileLiteRtModelWithFallback(
    ENCODER_MODEL_URL,
    preferred,
    [[1, 3, SAM_IMG_SIZE, SAM_IMG_SIZE]]
  );
  encoderModel = encRes.model;
  encoderAccelerator = encRes.accelerator;

  const decRes = await compileLiteRtModelWithFallback(
    DECODER_MODEL_URL,
    preferred,
    [
      [1, SAM_EMBED_DIM, SAM_EMBED_GRID, SAM_EMBED_GRID],
      [1, SAM_MAX_POINTS, 2],
      [1, SAM_MAX_POINTS],
    ]
  );
  decoderModel = decRes.model;
  decoderAccelerator = decRes.accelerator;
}

async function initYoloeInWorker(): Promise<void> {
  const imported = (await import(ORT_ESM_URL)) as {
    default?: OrtModule;
  } & OrtModule;
  const ort: OrtModule = imported.default ?? imported;
  ort.env.wasm.wasmPaths = ORT_WASM_DIR;
  ort.env.wasm.numThreads = 1;
  ort.env.logLevel = 'error';
  ortModule = ort;

  const modelBytes = await fetchCachedModelBytes(YOLOE_MODEL_URL);
  for (const ep of ['webgpu', 'wasm']) {
    try {
      yoloeSession = await ort.InferenceSession.create(modelBytes, {
        executionProviders: [ep],
      });
      yoloeProvider = ep === 'webgpu' ? 'WebGPU' : 'WASM';
      return;
    } catch (err) {
      console.warn(`[PerceptionWorker] YOLOE-11s ${ep} fallback:`, err);
    }
  }
  throw new Error('Could not initialize YOLOE-11s ONNX session in worker.');
}

async function handleInit() {
  const t0 = performance.now();
  await Promise.all([initSamInWorker(), initYoloeInWorker()]);
  return {
    yoloeProvider,
    encoderAccelerator,
    decoderAccelerator,
    initMs: performance.now() - t0,
  };
}

function fillPlanarFloat32(
  rgba: Uint8ClampedArray,
  size: number,
  out: Float32Array
): void {
  const hw = size * size;
  const inv255 = 1 / 255;
  for (let i = 0; i < hw; i++) {
    const p = i * 4;
    out[i] = rgba[p] * inv255;
    out[hw + i] = rgba[p + 1] * inv255;
    out[2 * hw + i] = rgba[p + 2] * inv255;
  }
}

function drawFrameToWorkerCanvases(payload: {
  bitmap?: ImageBitmap;
  rgbaBuffer?: ArrayBuffer;
}): void {
  if (payload.bitmap) {
    const bmp = payload.bitmap;
    const sw = bmp.width;
    const sh = bmp.height;
    // Center-square crop to match XRDeviceCamera / getDeviceCameraClipFromView
    const side = Math.min(sw, sh);
    const sx = (sw - side) * 0.5;
    const sy = (sh - side) * 0.5;

    ctx512.drawImage(bmp, sx, sy, side, side, 0, 0, SAM_IMG_SIZE, SAM_IMG_SIZE);
    ctx640.drawImage(
      bmp,
      sx,
      sy,
      side,
      side,
      0,
      0,
      YOLOE_IMG_SIZE,
      YOLOE_IMG_SIZE
    );
    bmp.close();
    currentFrameId++;
  } else if (payload.rgbaBuffer) {
    const clamped = new Uint8ClampedArray(payload.rgbaBuffer);
    const imgData = new ImageData(clamped, SAM_IMG_SIZE, SAM_IMG_SIZE);
    ctx512.putImageData(imgData, 0, 0);
    ctx640.drawImage(canvas512, 0, 0, YOLOE_IMG_SIZE, YOLOE_IMG_SIZE);
    currentFrameId++;
  }
}

function computeBoxIou(a: BoundingBox2D, b: BoundingBox2D): number {
  const ix1 = Math.max(a.x1, b.x1);
  const iy1 = Math.max(a.y1, b.y1);
  const ix2 = Math.min(a.x2, b.x2);
  const iy2 = Math.min(a.y2, b.y2);
  const iw = Math.max(0, ix2 - ix1);
  const ih = Math.max(0, iy2 - iy1);
  const inter = iw * ih;
  if (inter <= 0) return 0;
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / Math.max(1e-5, areaA + areaB - inter);
}

async function runYoloeOnCanvas640(): Promise<{
  detections: Array<{
    classId: number;
    className: string;
    score: number;
    box: BoundingBox2D;
    boxArea: number;
  }>;
  yoloeMs: number;
}> {
  if (!yoloeSession || !ortModule) {
    return {detections: [], yoloeMs: 0};
  }
  const t0 = performance.now();
  const rgba640 = ctx640.getImageData(
    0,
    0,
    YOLOE_IMG_SIZE,
    YOLOE_IMG_SIZE
  ).data;
  fillPlanarFloat32(rgba640, YOLOE_IMG_SIZE, yoloePlanarBuffer);

  const inputTensor = new ortModule.Tensor('float32', yoloePlanarBuffer, [
    1,
    3,
    YOLOE_IMG_SIZE,
    YOLOE_IMG_SIZE,
  ]);

  let outputs: Record<string, OrtTensor>;
  try {
    outputs = await yoloeSession.run({images: inputTensor});
  } catch (err) {
    if (yoloeProvider === 'WebGPU') {
      console.warn(
        '[PerceptionWorker] YOLOE WebGPU run failed, falling back to WASM:',
        err
      );
      const modelBytes = await fetchCachedModelBytes(YOLOE_MODEL_URL);
      yoloeSession = await ortModule.InferenceSession.create(modelBytes, {
        executionProviders: ['wasm'],
      });
      yoloeProvider = 'WASM';
      outputs = await yoloeSession.run({images: inputTensor});
    } else {
      throw err;
    }
  }

  const outTensor = outputs.output0 ?? Object.values(outputs)[0];
  const data = outTensor.data;
  const dims = outTensor.dims;
  const numRows = dims[1] ?? 369;
  const numAnchors = Math.min(YOLOE_NUM_ANCHORS, dims[2] ?? YOLOE_NUM_ANCHORS);
  const numClasses = Math.min(OBJECTS365_NAMES.length, numRows - 4);
  const scaleTo512 = SAM_IMG_SIZE / YOLOE_IMG_SIZE;

  // Stride-1 class-major scan to avoid 3.06M L1/L2 cache misses
  bestScoresBuffer.fill(CONF_SURVIVAL_THRESHOLD);
  bestClassesBuffer.fill(-1);
  for (let c = 0; c < numClasses; c++) {
    const rowOffset = (4 + c) * numAnchors;
    for (let a = 0; a < numAnchors; a++) {
      const score = data[rowOffset + a];
      if (score > bestScoresBuffer[a]) {
        bestScoresBuffer[a] = score;
        bestClassesBuffer[a] = c;
      }
    }
  }

  const candidates: Array<{
    classId: number;
    className: string;
    score: number;
    box: BoundingBox2D;
    boxArea: number;
  }> = [];

  for (let a = 0; a < numAnchors; a++) {
    const bestClass = bestClassesBuffer[a];
    if (bestClass < 0) continue;
    const bestScore = bestScoresBuffer[a];

    const cx = data[0 * numAnchors + a] * scaleTo512;
    const cy = data[1 * numAnchors + a] * scaleTo512;
    const w = data[2 * numAnchors + a] * scaleTo512;
    const h = data[3 * numAnchors + a] * scaleTo512;

    const x1 = Math.max(0, Math.min(SAM_IMG_SIZE, cx - w * 0.5));
    const y1 = Math.max(0, Math.min(SAM_IMG_SIZE, cy - h * 0.5));
    const x2 = Math.max(0, Math.min(SAM_IMG_SIZE, cx + w * 0.5));
    const y2 = Math.max(0, Math.min(SAM_IMG_SIZE, cy + h * 0.5));
    const bw = x2 - x1;
    const bh = y2 - y1;
    if (bw < 8 || bh < 8) continue;

    candidates.push({
      classId: bestClass,
      className: OBJECTS365_NAMES[bestClass] || `Class ${bestClass}`,
      score: bestScore,
      box: {x1, y1, x2, y2},
      boxArea: bw * bh,
    });
  }

  candidates.sort((a, b) => b.score - a.score);
  const nmsDetections: typeof candidates = [];
  for (const cand of candidates) {
    let suppressed = false;
    for (const kept of nmsDetections) {
      const iou = computeBoxIou(cand.box, kept.box);
      if ((cand.classId === kept.classId && iou > 0.45) || iou > 0.72) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) {
      nmsDetections.push(cand);
      if (nmsDetections.length >= 24) break;
    }
  }

  return {
    detections: nmsDetections,
    yoloeMs: performance.now() - t0,
  };
}

async function runSamEncoderOnCanvas512(): Promise<number> {
  if (!encoderModel) return 0;
  const t0 = performance.now();
  const rgba512 = ctx512.getImageData(0, 0, SAM_IMG_SIZE, SAM_IMG_SIZE).data;
  fillPlanarFloat32(rgba512, SAM_IMG_SIZE, samPlanarBuffer);

  let inputTensor: {deleted?: boolean; delete(): void} | null = null;
  let encOutputs: Array<{
    data(): Promise<Float32Array | ArrayBufferView>;
    deleted?: boolean;
    delete(): void;
  }> = [];
  try {
    inputTensor = new LiteRtTensor(samPlanarBuffer, [
      1,
      3,
      SAM_IMG_SIZE,
      SAM_IMG_SIZE,
    ]);
    encOutputs = await encoderModel.run([inputTensor]);
    const raw = await encOutputs[0].data();
    cachedEmbeddings =
      raw instanceof Float32Array
        ? new Float32Array(raw)
        : new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
    encodedFrameId = currentFrameId;
  } finally {
    if (inputTensor && !inputTensor.deleted) inputTensor.delete();
    for (const out of encOutputs) {
      if (out && !out.deleted) out.delete();
    }
  }
  lastEncoderMs = performance.now() - t0;
  return lastEncoderMs;
}

/**
 * Selects the best candidate mask from EfficientSAM-Ti's 3 hierarchical outputs
 * directly on the 128x128 low-res logit grid.
 */
function selectBestCandidateMask128(
  masksLogits: Float32Array,
  ious: Float32Array,
  box: BoundingBox2D
): number {
  const planeStride = SAM_MASK_LOW_RES * SAM_MASK_LOW_RES;
  const numChannels = Math.max(1, Math.floor(masksLogits.length / planeStride));

  const bx1 = Math.max(
    0,
    Math.floor((box.x1 / SAM_IMG_SIZE) * SAM_MASK_LOW_RES)
  );
  const by1 = Math.max(
    0,
    Math.floor((box.y1 / SAM_IMG_SIZE) * SAM_MASK_LOW_RES)
  );
  const bx2 = Math.min(
    SAM_MASK_LOW_RES,
    Math.ceil((box.x2 / SAM_IMG_SIZE) * SAM_MASK_LOW_RES)
  );
  const by2 = Math.min(
    SAM_MASK_LOW_RES,
    Math.ceil((box.y2 / SAM_IMG_SIZE) * SAM_MASK_LOW_RES)
  );
  const boxArea = Math.max(1, (bx2 - bx1) * (by2 - by1));

  let best = 0;
  let bestScore = -Infinity;
  for (let c = 0; c < numChannels; c++) {
    const off = c * planeStride;
    let inside = 0;
    let total = 0;
    let mMinX = SAM_MASK_LOW_RES;
    let mMinY = SAM_MASK_LOW_RES;
    let mMaxX = 0;
    let mMaxY = 0;

    for (let y = 0; y < SAM_MASK_LOW_RES; y++) {
      const row = off + y * SAM_MASK_LOW_RES;
      const yIn = y >= by1 && y < by2;
      for (let x = 0; x < SAM_MASK_LOW_RES; x++) {
        if (masksLogits[row + x] >= 0.0) {
          total++;
          if (yIn && x >= bx1 && x < bx2) {
            inside++;
            if (x < mMinX) mMinX = x;
            if (y < mMinY) mMinY = y;
            if (x + 1 > mMaxX) mMaxX = x + 1;
            if (y + 1 > mMaxY) mMaxY = y + 1;
          }
        }
      }
    }
    if (total === 0 || inside === 0) continue;

    const containment = inside / total;
    const fillRatio = inside / boxArea;
    const interW = Math.max(0, Math.min(bx2, mMaxX) - Math.max(bx1, mMinX));
    const interH = Math.max(0, Math.min(by2, mMaxY) - Math.max(by1, mMinY));
    const maskBoxIou = (interW * interH) / boxArea;

    const subPartPenalty = maskBoxIou < 0.42 || fillRatio < 0.18 ? -0.65 : 0.0;
    const iou = ious[c] ?? 0;

    const score =
      containment * 1.15 +
      maskBoxIou * 1.1 +
      Math.min(0.65, fillRatio) * 0.55 +
      0.15 * iou +
      subPartPenalty;

    if (score > bestScore) {
      bestScore = score;
      best = c;
    }
  }
  return best;
}

/**
 * Bilinearly samples a single point `(px, py)` in `512x512` coordinates directly from
 * the `128x128` (`SAM_MASK_LOW_RES`) logit plane without allocating or upsampling a 512x512 buffer.
 */
function sampleLogitAt512(
  masksLogits: Float32Array,
  channelOffset: number,
  px512: number,
  py512: number
): number {
  const scale = SAM_MASK_LOW_RES / SAM_IMG_SIZE; // 0.25
  const sx = (px512 + 0.5) * scale - 0.5;
  const sy = (py512 + 0.5) * scale - 0.5;
  const x0 = Math.max(0, Math.min(SAM_MASK_LOW_RES - 1, Math.floor(sx)));
  const y0 = Math.max(0, Math.min(SAM_MASK_LOW_RES - 1, Math.floor(sy)));
  const x1 = Math.min(SAM_MASK_LOW_RES - 1, x0 + 1);
  const y1 = Math.min(SAM_MASK_LOW_RES - 1, y0 + 1);
  const wx = Math.max(0, sx - x0);
  const wy = Math.max(0, sy - y0);

  const row0 = channelOffset + y0 * SAM_MASK_LOW_RES;
  const row1 = channelOffset + y1 * SAM_MASK_LOW_RES;
  const v00 = masksLogits[row0 + x0];
  const v01 = masksLogits[row0 + x1];
  const v10 = masksLogits[row1 + x0];
  const v11 = masksLogits[row1 + x1];

  return (
    (1 - wy) * ((1 - wx) * v00 + wx * v01) + wy * ((1 - wx) * v10 + wx * v11)
  );
}

/**
 * Extracts both the 64-ray `ContourProfile` (in 512x512 coordinates) and a compact
 * `32x32` foreground binary mask (`fgMask32`) directly from the `128x128` logit plane.
 * Cost: ~0.15ms (vs 1,256.7ms for full 512x512 upsampling in the trace).
 */
function extractContourAndFgMaskFrom128Logits(
  masksLogits: Float32Array,
  channelIdx: number,
  box: BoundingBox2D
): {contourProfile?: ContourProfile; fgMask32: Uint8Array} {
  const channelOffset = channelIdx * SAM_MASK_LOW_RES * SAM_MASK_LOW_RES;
  const scaleTo128 = SAM_MASK_LOW_RES / SAM_IMG_SIZE; // 0.25
  const invScaleTo512 = SAM_IMG_SIZE / SAM_MASK_LOW_RES; // 4.0

  const lx1 = Math.max(0, Math.floor((box.x1 - 8) * scaleTo128));
  const ly1 = Math.max(0, Math.floor((box.y1 - 8) * scaleTo128));
  const lx2 = Math.min(
    SAM_MASK_LOW_RES - 1,
    Math.ceil((box.x2 + 8) * scaleTo128)
  );
  const ly2 = Math.min(
    SAM_MASK_LOW_RES - 1,
    Math.ceil((box.y2 + 8) * scaleTo128)
  );

  let sumX = 0;
  let sumY = 0;
  let count = 0;
  for (let y = ly1; y <= ly2; y++) {
    const row = channelOffset + y * SAM_MASK_LOW_RES;
    for (let x = lx1; x <= lx2; x++) {
      if (masksLogits[row + x] >= 0.0) {
        sumX += (x + 0.5) * invScaleTo512;
        sumY += (y + 0.5) * invScaleTo512;
        count++;
      }
    }
  }

  const bcx = (box.x1 + box.x2) * 0.5;
  const bcy = (box.y1 + box.y2) * 0.5;
  const bw = Math.max(8, box.x2 - box.x1);
  const bh = Math.max(8, box.y2 - box.y1);

  // Build 32x32 foreground mask inside `box` for depth sampling, ray hit-testing, and CV tracking
  const fgMask32 = new Uint8Array(32 * 32);
  let fg32Count = 0;
  for (let ty = 0; ty < 32; ty++) {
    const py = box.y1 + ((ty + 0.5) / 32) * bh;
    for (let tx = 0; tx < 32; tx++) {
      const px = box.x1 + ((tx + 0.5) / 32) * bw;
      if (sampleLogitAt512(masksLogits, channelOffset, px, py) >= 0.0) {
        fgMask32[ty * 32 + tx] = 1;
        fg32Count++;
      }
    }
  }

  if (fg32Count < 24) {
    for (let ty = 3; ty < 29; ty++) {
      const ny = (ty - 15.5) / 12.5;
      for (let tx = 3; tx < 29; tx++) {
        const nx = (tx - 15.5) / 12.5;
        if (nx * nx + ny * ny <= 1.0) {
          fgMask32[ty * 32 + tx] = 1;
        }
      }
    }
  }

  if (count < 2) {
    return {contourProfile: undefined, fgMask32};
  }

  const mcx = sumX / count;
  const mcy = sumY / count;
  const numRays = 64;
  const maxR = Math.hypot(bw, bh) * 0.78;
  const radii = new Float32Array(numRays);

  const bx1 = Math.max(0, box.x1 - 8);
  const by1 = Math.max(0, box.y1 - 8);
  const bx2 = Math.min(SAM_IMG_SIZE - 1, box.x2 + 8);
  const by2 = Math.min(SAM_IMG_SIZE - 1, box.y2 + 8);

  for (let i = 0; i < numRays; i++) {
    const angle = (i / numRays) * Math.PI * 2;
    const dx = Math.cos(angle);
    const dy = Math.sin(angle);
    let furthestR = 0;
    for (let r = 2; r <= maxR; r += 2.0) {
      const px = mcx + dx * r;
      const py = mcy + dy * r;
      if (px < bx1 || px > bx2 || py < by1 || py > by2) break;
      if (sampleLogitAt512(masksLogits, channelOffset, px, py) >= 0.0) {
        furthestR = r;
      }
    }
    radii[i] = furthestR;
  }

  for (let i = 0; i < numRays; i++) {
    if (radii[i] === 0) {
      const prev = radii[(i - 1 + numRays) % numRays];
      const next = radii[(i + 1) % numRays];
      radii[i] = (prev + next) * 0.5 || Math.min(bw, bh) * 0.25;
    }
  }

  return {
    contourProfile: {mcx, mcy, bcx, bcy, bw, bh, radii},
    fgMask32,
  };
}

function syncWorkerFastTrackerFromCanvas512(
  trackId: number,
  box: BoundingBox2D,
  fgMask32: Uint8Array,
  resetOffsets: boolean
): void {
  const x1 = Math.max(0, Math.floor(box.x1));
  const y1 = Math.max(0, Math.floor(box.y1));
  const x2 = Math.min(SAM_IMG_SIZE, Math.ceil(box.x2));
  const y2 = Math.min(SAM_IMG_SIZE, Math.ceil(box.y2));
  const w = Math.max(8, x2 - x1);
  const h = Math.max(8, y2 - y1);

  ctx32.clearRect(0, 0, 32, 32);
  ctx32.drawImage(canvas512, x1, y1, w, h, 0, 0, 32, 32);
  const rgba = ctx32.getImageData(0, 0, 32, 32).data;

  if (!fastTracker || fastTracker.trackId !== trackId || resetOffsets) {
    const gray = new Uint8ClampedArray(1024);
    for (let i = 0; i < 1024; i++) {
      const p = i * 4;
      gray[i] =
        (rgba[p] * 0.299 + rgba[p + 1] * 0.587 + rgba[p + 2] * 0.114) | 0;
    }
    fastTracker = {
      trackId,
      templateData: gray,
      refTemplate: new Uint8ClampedArray(gray),
      fgMask: new Uint8Array(fgMask32),
      anchorBox: {...box},
      offsetX: 0,
      offsetY: 0,
      activeAngle: 0,
    };
  } else {
    for (let i = 0; i < 1024; i++) {
      const p = i * 4;
      const g =
        (rgba[p] * 0.299 + rgba[p + 1] * 0.587 + rgba[p + 2] * 0.114) | 0;
      fastTracker.refTemplate[i] = g;
      fastTracker.templateData[i] = g;
    }
    fastTracker.fgMask.set(fgMask32);
    fastTracker.anchorBox = {...box};
    fastTracker.offsetX *= 0.25;
    fastTracker.offsetY *= 0.25;
    fastTracker.activeAngle *= 0.4;
  }
}

async function decodeSamForBox(box: BoundingBox2D): Promise<{
  contourProfile?: ContourProfile;
  fgMask32: Uint8Array;
  samIou: number;
  decoderMs: number;
  encoderMs: number;
}> {
  if (!decoderModel) {
    return {
      contourProfile: undefined,
      fgMask32: new Uint8Array(1024).fill(1),
      samIou: 0,
      decoderMs: 0,
      encoderMs: 0,
    };
  }

  // Lazily encode the current 512x512 frame only when a SAM mask is actually requested
  if (!cachedEmbeddings || encodedFrameId !== currentFrameId) {
    await runSamEncoderOnCanvas512();
  }

  if (!cachedEmbeddings) {
    return {
      contourProfile: undefined,
      fgMask32: new Uint8Array(1024).fill(1),
      samIou: 0,
      decoderMs: 0,
      encoderMs: lastEncoderMs,
    };
  }

  samPtsBuffer.fill(-1);
  samLblsBuffer.fill(-1);
  samPtsBuffer[0] = (box.x1 + box.x2) * 0.5;
  samPtsBuffer[1] = (box.y1 + box.y2) * 0.5;
  samLblsBuffer[0] = 1.0;

  samPtsBuffer[2] = box.x1;
  samPtsBuffer[3] = box.y1;
  samLblsBuffer[1] = 2.0;

  samPtsBuffer[4] = box.x2;
  samPtsBuffer[5] = box.y2;
  samLblsBuffer[2] = 3.0;

  let embTensor: {deleted?: boolean; delete(): void} | null = null;
  let ptsTensor: {deleted?: boolean; delete(): void} | null = null;
  let lblsTensor: {deleted?: boolean; delete(): void} | null = null;
  let decOutputs: Array<{
    data(): Promise<Float32Array | ArrayBufferView>;
    deleted?: boolean;
    delete(): void;
  }> = [];

  const t0 = performance.now();
  let masksLogits: Float32Array;
  let ious: Float32Array;

  try {
    embTensor = new LiteRtTensor(cachedEmbeddings, [
      1,
      SAM_EMBED_DIM,
      SAM_EMBED_GRID,
      SAM_EMBED_GRID,
    ]);
    ptsTensor = new LiteRtTensor(samPtsBuffer, [1, SAM_MAX_POINTS, 2]);
    lblsTensor = new LiteRtTensor(samLblsBuffer, [1, SAM_MAX_POINTS]);

    decOutputs = await decoderModel.run([embTensor, ptsTensor, lblsTensor]);
    const raw0 = await decOutputs[0].data();
    const raw1 = decOutputs[1]
      ? await decOutputs[1].data()
      : new Float32Array([0, 0, 0]);
    const asF32 = (arr: Float32Array | ArrayBufferView): Float32Array =>
      arr instanceof Float32Array
        ? arr
        : new Float32Array(arr.buffer, arr.byteOffset, arr.byteLength / 4);
    const out0 = asF32(raw0);
    const out1 = asF32(raw1);
    [masksLogits, ious] =
      out0.length < out1.length ? [out1, out0] : [out0, out1];
  } finally {
    for (const t of [embTensor, ptsTensor, lblsTensor, ...decOutputs]) {
      if (t && !t.deleted) t.delete();
    }
  }
  const decoderMs = performance.now() - t0;

  const bestIdx = selectBestCandidateMask128(masksLogits, ious, box);
  const {contourProfile, fgMask32} = extractContourAndFgMaskFrom128Logits(
    masksLogits,
    bestIdx,
    box
  );

  return {
    contourProfile,
    fgMask32,
    samIou: ious[bestIdx] ?? 0,
    decoderMs,
    encoderMs: lastEncoderMs,
  };
}

async function handleDetectFrame(payload: {
  bitmap?: ImageBitmap;
  rgbaBuffer?: ArrayBuffer;
}) {
  drawFrameToWorkerCanvases(payload);
  const yoloeRes = await runYoloeOnCanvas640();
  return {
    detections: yoloeRes.detections,
    yoloeMs: yoloeRes.yoloeMs,
    encoderMs: lastEncoderMs,
  };
}

async function handleSegmentTracks(payload: {
  requests: Array<{
    trackId: number;
    box: BoundingBox2D;
    isLocked: boolean;
  }>;
}) {
  const results: Array<{
    trackId: number;
    box: BoundingBox2D;
    contourProfile?: ContourProfile;
    fgMask32: Uint8Array;
    samIou: number;
    decoderMs: number;
    encoderMs: number;
  }> = [];

  for (const req of payload.requests) {
    const decoded = await decodeSamForBox(req.box);
    if (req.isLocked) {
      syncWorkerFastTrackerFromCanvas512(
        req.trackId,
        req.box,
        decoded.fgMask32,
        false
      );
    }
    results.push({
      trackId: req.trackId,
      box: req.box,
      contourProfile: decoded.contourProfile,
      fgMask32: decoded.fgMask32,
      samIou: decoded.samIou,
      decoderMs: decoded.decoderMs,
      encoderMs: decoded.encoderMs,
    });
  }

  return {results, encoderMs: lastEncoderMs};
}

async function handleLockTrack(payload: {
  trackId: number;
  box: BoundingBox2D;
  fgMask32?: Uint8Array;
  needsSamDecode: boolean;
}) {
  let contourProfile: ContourProfile | undefined;
  let fgMask32 = payload.fgMask32;
  let samIou = 0;
  let decoderMs = 0;

  if (payload.needsSamDecode || !fgMask32) {
    const decoded = await decodeSamForBox(payload.box);
    contourProfile = decoded.contourProfile;
    fgMask32 = decoded.fgMask32;
    samIou = decoded.samIou;
    decoderMs = decoded.decoderMs;
  }

  syncWorkerFastTrackerFromCanvas512(
    payload.trackId,
    payload.box,
    fgMask32,
    true
  );

  return {
    trackId: payload.trackId,
    contourProfile,
    fgMask32,
    samIou,
    decoderMs,
    encoderMs: lastEncoderMs,
  };
}

function estimateRotationAngle32(
  curGray: Uint8ClampedArray,
  refGray: Uint8ClampedArray,
  fgMask: Uint8Array,
  currentActiveAngle: number
): number {
  const evalAngleSad = (theta: number): number => {
    const cosT = Math.cos(theta);
    const sinT = Math.sin(theta);
    let diffSum = 0;
    let validCount = 0;

    for (let y = 3; y < 29; y++) {
      const dy = y - 15.5;
      const row = y * 32;
      for (let x = 3; x < 29; x++) {
        const idx = row + x;
        if (!fgMask[idx]) continue;
        const dx = x - 15.5;
        const rx = 15.5 + dx * cosT - dy * sinT;
        const ry = 15.5 + dx * sinT + dy * cosT;
        const ix = rx | 0;
        const iy = ry | 0;
        if (ix < 1 || ix >= 30 || iy < 1 || iy >= 30) continue;

        const fx = rx - ix;
        const fy = ry - iy;
        const r0 = iy * 32 + ix;
        const r1 = r0 + 32;
        const val =
          (1 - fy) * ((1 - fx) * curGray[r0] + fx * curGray[r0 + 1]) +
          fy * ((1 - fx) * curGray[r1] + fx * curGray[r1 + 1]);

        const d = val - refGray[idx];
        diffSum += d < 0 ? -d : d;
        validCount++;
      }
    }
    return validCount >= 16 ? diffSum / validCount : Infinity;
  };

  const sad0 = evalAngleSad(0);
  if (!Number.isFinite(sad0)) return 0;

  let bestTheta = 0;
  let bestCost = sad0;
  const stepRad = (3.5 * Math.PI) / 180;

  for (let k = -7; k <= 7; k++) {
    if (k === 0) continue;
    const theta = k * stepRad;
    const rawSad = evalAngleSad(theta);
    const regPenalty =
      sad0 * (0.065 + 0.16 * Math.abs(theta)) +
      sad0 * 0.05 * Math.abs(theta - currentActiveAngle);
    const cost = rawSad + regPenalty;
    if (cost < bestCost) {
      bestCost = cost;
      bestTheta = theta;
    }
  }

  if (Math.abs(bestTheta) < 0.045) return 0;
  return bestTheta;
}

/**
 * Single-pass 48x48 buffer-indexed CV template + rotation tracker inside the worker.
 * Calls `getImageData(0, 0, 48, 48)` ONCE per frame (instead of 82 `drawImage` + `getImageData` calls).
 */
function handleTrackPatch(payload: {
  trackId: number;
  patchBitmap?: ImageBitmap;
  egoBox: BoundingBox2D;
  isBorderClamped: boolean;
}) {
  if (!fastTracker || fastTracker.trackId !== payload.trackId) {
    if (payload.patchBitmap) payload.patchBitmap.close();
    return {
      trackId: payload.trackId,
      offsetX: 0,
      offsetY: 0,
      activeAngle: 0,
    };
  }

  const egoBox = payload.egoBox;
  const anchorW = Math.max(
    8,
    fastTracker.anchorBox.x2 - fastTracker.anchorBox.x1
  );
  const anchorH = Math.max(
    8,
    fastTracker.anchorBox.y2 - fastTracker.anchorBox.y1
  );
  const boxW = Math.max(8, egoBox.x2 - egoBox.x1);
  const boxH = Math.max(8, egoBox.y2 - egoBox.y1);
  const scaleRatio = Math.hypot(boxW, boxH) / Math.hypot(anchorW, anchorH);

  if (
    payload.isBorderClamped ||
    scaleRatio < 0.72 ||
    scaleRatio > 1.38 ||
    !payload.patchBitmap
  ) {
    if (payload.patchBitmap) payload.patchBitmap.close();
    fastTracker.offsetX *= 0.85;
    fastTracker.offsetY *= 0.85;
    fastTracker.activeAngle *= 0.9;
    return {
      trackId: fastTracker.trackId,
      offsetX: fastTracker.offsetX,
      offsetY: fastTracker.offsetY,
      activeAngle: fastTracker.activeAngle,
    };
  }

  ctx48.clearRect(0, 0, 48, 48);
  ctx48.drawImage(payload.patchBitmap, 0, 0, 48, 48);
  payload.patchBitmap.close();

  const rgba48 = ctx48.getImageData(0, 0, 48, 48).data;
  for (let i = 0; i < 48 * 48; i++) {
    const p = i * 4;
    patch48Gray[i] =
      (rgba48[p] * 0.299 + rgba48[p + 1] * 0.587 + rgba48[p + 2] * 0.114) | 0;
  }

  const tData = fastTracker.templateData;
  const fgMask = fastTracker.fgMask;

  // Search offsets in 48x48 template coordinates: center (8, 8) corresponds to zero shift
  let bestGx = 0;
  let bestGy = 0;
  let minDiff = Infinity;
  let zeroDiff = Infinity;

  for (let gy = -6; gy <= 6; gy += 1) {
    const startY = 8 + gy;
    for (let gx = -6; gx <= 6; gx += 1) {
      const startX = 8 + gx;
      let diff = 0;
      let fgCount = 0;

      for (let ty = 0; ty < 32; ty++) {
        const patchRow = (startY + ty) * 48 + startX;
        const tmplRow = ty * 32;
        for (let tx = 0; tx < 32; tx++) {
          const idx = tmplRow + tx;
          if (!fgMask[idx]) continue;
          const d = patch48Gray[patchRow + tx] - tData[idx];
          diff += d < 0 ? -d : d;
          fgCount++;
        }
      }
      if (fgCount > 0) {
        diff /= fgCount;
      }
      if (gx === 0 && gy === 0) {
        zeroDiff = diff;
      }
      if (diff < minDiff) {
        minDiff = diff;
        bestGx = gx;
        bestGy = gy;
      }
    }
  }

  if (minDiff > zeroDiff * 0.92) {
    bestGx = 0;
    bestGy = 0;
  }

  const dx512 = bestGx * (boxW / 32.0);
  const dy512 = bestGy * (boxH / 32.0);

  fastTracker.offsetX = Math.max(
    -14,
    Math.min(14, fastTracker.offsetX * 0.92 + dx512 * 0.35)
  );
  fastTracker.offsetY = Math.max(
    -14,
    Math.min(14, fastTracker.offsetY * 0.92 + dy512 * 0.35)
  );

  // Extract winning 32x32 window from patch48Gray and adapt template + estimate rotation
  const winStartX = 8 + bestGx;
  const winStartY = 8 + bestGy;
  const adaptRate = 0.05;
  for (let ty = 0; ty < 32; ty++) {
    const patchRow = (winStartY + ty) * 48 + winStartX;
    const tmplRow = ty * 32;
    for (let tx = 0; tx < 32; tx++) {
      const idx = tmplRow + tx;
      const g = patch48Gray[patchRow + tx];
      cur32Gray[idx] = g;
      tData[idx] = (tData[idx] * (1 - adaptRate) + g * adaptRate) | 0;
    }
  }

  const targetAngle = estimateRotationAngle32(
    cur32Gray,
    fastTracker.refTemplate,
    fastTracker.fgMask,
    fastTracker.activeAngle
  );
  fastTracker.activeAngle = fastTracker.activeAngle * 0.78 + targetAngle * 0.22;
  if (Math.abs(fastTracker.activeAngle) < 0.018) {
    fastTracker.activeAngle = 0;
  }

  return {
    trackId: fastTracker.trackId,
    offsetX: fastTracker.offsetX,
    offsetY: fastTracker.offsetY,
    activeAngle: fastTracker.activeAngle,
  };
}

// Serialize async worker commands so WebGPU / LiteRT calls never overlap
let workerQueue: Promise<void> = Promise.resolve();

self.addEventListener('message', (event: MessageEvent) => {
  const {
    id,
    type,
    payload = {},
  } = event.data as {
    id: number;
    type: string;
    payload?: Record<string, unknown>;
  };

  // Synchronous fast path for `track_patch` or `unlock_track` if queue is idle
  if (type === 'unlock_track') {
    fastTracker = null;
    self.postMessage({id, ok: true, result: {}});
    return;
  }

  if (type === 'track_patch') {
    try {
      const result = handleTrackPatch(
        payload as {
          trackId: number;
          patchBitmap?: ImageBitmap;
          egoBox: BoundingBox2D;
          isBorderClamped: boolean;
        }
      );
      self.postMessage({id, ok: true, result});
    } catch (err) {
      const error = err instanceof Error ? err.message : String(err);
      self.postMessage({id, ok: false, error});
    }
    return;
  }

  workerQueue = workerQueue
    .then(async () => {
      try {
        if (type === 'init') {
          const result = await handleInit();
          self.postMessage({id, ok: true, result});
        } else if (type === 'detect_frame') {
          const result = await handleDetectFrame(
            payload as {bitmap?: ImageBitmap; rgbaBuffer?: ArrayBuffer}
          );
          self.postMessage({id, ok: true, result});
        } else if (type === 'segment_tracks') {
          const result = await handleSegmentTracks(
            payload as {
              requests: Array<{
                trackId: number;
                box: BoundingBox2D;
                isLocked: boolean;
              }>;
            }
          );
          self.postMessage({id, ok: true, result});
        } else if (type === 'lock_track') {
          const result = await handleLockTrack(
            payload as {
              trackId: number;
              box: BoundingBox2D;
              fgMask32?: Uint8Array;
              needsSamDecode: boolean;
            }
          );
          self.postMessage({id, ok: true, result});
        } else {
          throw new Error(`Unknown worker command: ${type}`);
        }
      } catch (err) {
        const error = err instanceof Error ? err.message : String(err);
        self.postMessage({id, ok: false, error});
      }
    })
    .catch(() => {});
});
