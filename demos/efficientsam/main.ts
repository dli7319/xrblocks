import * as THREE from 'three';
import {GLTFExporter} from 'three/addons/exporters/GLTFExporter.js';
import * as xb from 'xrblocks';
import type {Accelerator} from '@litertjs/core';
import type {WorkerTsdfMeshPayload} from './efficientsam_worker.js';

const MODEL_IMG_SIZE = 512;
const MAX_POINTS = 6;
const QUAD_DISTANCE_METERS = 0.3;
const QUAD_SIZE_METERS = 0.8;
const MOUSE_QUAD_DISTANCE_METERS = 1.0;
const MOUSE_QUAD_SIZE_METERS = 1.4;

const MIN_KEYFRAME_TRANSLATION_METERS = 0.018;
const MIN_KEYFRAME_ROTATION_RAD = THREE.MathUtils.degToRad(2.5);
const DEPTH_FUSION_INTERVAL_MS = 70;
const RGB_MASK_FUSION_INTERVAL_MS = 220;

interface CirclePoint {
  x: number;
  y: number;
  u: number;
  v: number;
  worldPoint: THREE.Vector3;
}

interface BoundingBox2D {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

interface CirclePromptInfo {
  pts: Float32Array;
  lbls: Float32Array;
  box: BoundingBox2D | null;
  center: {x: number; y: number};
}

interface CameraCaptureResult {
  imageData: ImageData;
  clipFromWorld: THREE.Matrix4;
}

interface DepthTransferPayload {
  depthBuffer: ArrayBuffer;
  depthWidth: number;
  depthHeight: number;
  rawValueToMeters: number;
  depthFormat: string;
  depthViewMatrixBuffer: ArrayBuffer;
  depthProjectionMatrixBuffer: ArrayBuffer;
  depthProjectionInverseMatrixBuffer: ArrayBuffer;
  normDepthBufferFromNormViewMatrixBuffer: ArrayBuffer;
  cameraPos: THREE.Vector3;
  cameraQuat: THREE.Quaternion;
}

interface WorkerInitResult {
  encoderAccelerator: Accelerator;
  decoderAccelerator: Accelerator;
  compileTimeMs: number;
}

interface WorkerXrSegmentResult {
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
}

interface WorkerRgbMaskIntegrateResult {
  tsdfMesh: WorkerTsdfMeshPayload | null;
  cutoutRgbaBuffer: ArrayBuffer | null;
  cropW: number;
  cropH: number;
  bestIou: number;
  totalSamMs: number;
}

/**
 * XR Circle to Digitize Script powered by XR Blocks (v0.20.0+), LiteRT 2.5.3 EfficientSAM-Ti,
 * Object-Centric KinectFusion (WebXR Depth TSDF + RANSAC Table Removal + Marching Cubes),
 * and Multi-View Graph-Cut Texture Mapping.
 *
 * All CPU/GPU-intensive LiteRT compilation, image preprocessing, inference, mask reprojection,
 * RANSAC table plane subtraction, TSDF volumetric integration, Marching Cubes surface extraction,
 * and Graph-Cut UV texture atlas synthesis run inside `efficientsam_worker.js` so the 72 FPS
 * WebXR render loop never stalls.
 */
export class XRCircleToSearchScript extends xb.Script {
  private worker: Worker | null = null;
  private nextWorkerReqId = 1;
  private readonly pendingWorkerRequests = new Map<
    number,
    {
      resolve: (value: unknown) => void;
      reject: (reason?: unknown) => void;
    }
  >();

  private encoderAccelerator: Accelerator = 'wasm';
  private decoderAccelerator: Accelerator = 'wasm';
  private modelsReady = false;
  private isSegmenting = false;
  private isIntegratingTsdf = false;
  private compileTimeMs = 0;

  private activeController: xb.InteractionSource['controller'] | null = null;
  private isDrawingCircle = false;
  private circlePath: CirclePoint[] = [];
  private readonly raycaster = new THREE.Raycaster();
  private readonly tempRay = new THREE.Ray();

  private readonly quadCanvas: HTMLCanvasElement;
  private readonly quadCtx: CanvasRenderingContext2D;
  private readonly quadTexture: THREE.CanvasTexture;

  private readonly captureCanvas: HTMLCanvasElement;
  private readonly captureCtx: CanvasRenderingContext2D;
  private readonly cutoutCanvas: HTMLCanvasElement;
  private readonly atlasCanvas: HTMLCanvasElement;
  private readonly atlasCtx: CanvasRenderingContext2D;
  private atlasTexture: THREE.CanvasTexture;

  private circleQuad: THREE.Mesh<
    THREE.PlaneGeometry,
    THREE.MeshBasicMaterial
  > | null = null;

  // Object-Centric KinectFusion live 3D reconstruction state
  private isScanningTsdf = false;
  private isPoppedOut = false;
  private fusedFrameCount = 0;
  private triangleCount = 0;
  private chartCount = 0;
  private voxelSizeMm = 0;
  private lastDepthFusionMs = 0;
  private lastRgbMaskFusionMs = 0;
  private readonly lastFusedCameraPos = new THREE.Vector3(NaN, NaN, NaN);
  private readonly lastFusedCameraQuat = new THREE.Quaternion(
    NaN,
    NaN,
    NaN,
    NaN
  );
  private readonly currentVolumeCenter = new THREE.Vector3();
  private currentVolumeSize = 0.25;

  private tsdfMeshGroup: THREE.Group | null = null;
  private tsdfLiveMesh: THREE.Mesh<
    THREE.BufferGeometry,
    THREE.MeshStandardMaterial
  > | null = null;
  private tsdfBoundingBox: THREE.LineSegments<
    THREE.EdgesGeometry,
    THREE.LineBasicMaterial
  > | null = null;

  private hudCard: xb.UICard | null = null;
  private hudStatusText: xb.UIText | null = null;
  private hudMetricsText: xb.UIText | null = null;
  private hudTsdfMetricsText: xb.UIText | null = null;
  private hudCutoutPlaceholder: xb.UIPanel | null = null;
  private hudCutoutImage: xb.UIImage | null = null;
  private telemetryBadgeCard: xb.UICard | null = null;
  private telemetryBadgeText: xb.UIText | null = null;

  private readonly targetDevice: string;

  private readonly domStatus: HTMLElement | null;
  private readonly domMetricTotal: HTMLElement | null;
  private readonly domMetricSplit: HTMLElement | null;
  private readonly domMetricIou: HTMLElement | null;
  private readonly domMetricTsdf: HTMLElement | null;

  constructor() {
    super();

    this.targetDevice =
      typeof navigator !== 'undefined' &&
      /OculusBrowser|Quest/i.test(navigator.userAgent)
        ? 'quest3'
        : 'galaxyxr';

    this.quadCanvas = document.createElement('canvas');
    this.quadCanvas.width = MODEL_IMG_SIZE;
    this.quadCanvas.height = MODEL_IMG_SIZE;
    this.quadCtx = this.quadCanvas.getContext('2d', {
      willReadFrequently: true,
    })!;
    this.quadTexture = new THREE.CanvasTexture(this.quadCanvas);
    this.quadTexture.colorSpace = THREE.SRGBColorSpace;
    this.quadTexture.minFilter = THREE.LinearFilter;
    this.quadTexture.magFilter = THREE.LinearFilter;

    this.captureCanvas = document.createElement('canvas');
    this.captureCanvas.width = MODEL_IMG_SIZE;
    this.captureCanvas.height = MODEL_IMG_SIZE;
    this.captureCtx = this.captureCanvas.getContext('2d', {
      willReadFrequently: true,
    })!;

    this.cutoutCanvas = document.createElement('canvas');
    this.cutoutCanvas.width = 320;
    this.cutoutCanvas.height = 320;

    this.atlasCanvas = document.createElement('canvas');
    this.atlasCanvas.width = MODEL_IMG_SIZE;
    this.atlasCanvas.height = MODEL_IMG_SIZE;
    this.atlasCtx = this.atlasCanvas.getContext('2d')!;
    this.atlasTexture = new THREE.CanvasTexture(this.atlasCanvas);
    this.atlasTexture.flipY = false;
    this.atlasTexture.colorSpace = THREE.SRGBColorSpace;
    this.atlasTexture.minFilter = THREE.LinearFilter;
    this.atlasTexture.magFilter = THREE.LinearFilter;
    this.atlasTexture.generateMipmaps = false;

    this.domStatus = document.getElementById('xr-hud-status');
    this.domMetricTotal = document.getElementById('xr-metric-total');
    this.domMetricSplit = document.getElementById('xr-metric-split');
    this.domMetricIou = document.getElementById('xr-metric-iou');
    this.domMetricTsdf = document.getElementById('xr-metric-tsdf');

    const clearBtn = document.getElementById('xr-clear-btn');
    if (clearBtn) {
      clearBtn.addEventListener('click', (e: MouseEvent) => {
        e.stopPropagation();
        this.clearQuadOverlay();
      });
    }

    const popOutBtn = document.getElementById('xr-popout-btn');
    if (popOutBtn) {
      popOutBtn.addEventListener('click', (e: MouseEvent) => {
        e.stopPropagation();
        this.finishAndPopOut3DModel();
      });
    }

    const exportBtn = document.getElementById('xr-export-btn');
    if (exportBtn) {
      exportBtn.addEventListener('click', (e: MouseEvent) => {
        e.stopPropagation();
        this.exportDigitizedMeshGlb();
      });
    }
  }

  override async init(): Promise<void> {
    // 1. Ensure scene has directional + ambient lighting so the vertex-colored 3D TSDF mesh is shaded clearly
    const ambientLight = new THREE.AmbientLight(0xffffff, 1.1);
    const dirLight = new THREE.DirectionalLight(0xffffff, 1.4);
    dirLight.position.set(0.8, 2.2, 1.2);
    this.add(ambientLight);
    this.add(dirLight);

    // 2. Build the 30cm invisible raycast quad
    this.createInvisibleCircleQuad();

    // 3. Build the live TSDF 3D Mesh + Bounding Volume visualizer
    this.createTsdfVisualizerObjects();

    // 4. Build XR Blocks Spatial HUD Card & Floating Telemetry Badge (v0.20.0 UICard API)
    this.createSpatialHudCard();

    // 5. Initialize LiteRT and compile EfficientSAM-Ti models in Worker
    await this.initLiteRtModels();
  }

  /**
   * Creates the invisible quad that spawns 30 cm in front of the user's pinching hand.
   * When cleared, its CanvasTexture is 100% transparent (`rgba(0,0,0,0)`), making
   * the quad invisible while still allowing the XR Blocks Reticle to raycast and
   * glide across its surface.
   */
  private createInvisibleCircleQuad(): void {
    this.quadCtx.clearRect(0, 0, MODEL_IMG_SIZE, MODEL_IMG_SIZE);
    this.quadTexture.needsUpdate = true;

    const geometry = new THREE.PlaneGeometry(
      QUAD_SIZE_METERS,
      QUAD_SIZE_METERS
    );
    const material = new THREE.MeshBasicMaterial({
      map: this.quadTexture,
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      toneMapped: false,
    });

    this.circleQuad = new THREE.Mesh(geometry, material);
    this.circleQuad.name = 'CircleToSearchQuad30cm';
    this.circleQuad.renderOrder = 500;
    this.circleQuad.visible = false;
    this.circleQuad.xb = {pointerEvents: 'none', reticleMode: 'auto'};
    this.add(this.circleQuad);
  }

  /**
   * Creates the live `THREE.Mesh` and wireframe `THREE.LineSegments` bounding box
   * that display the ongoing multi-view KinectFusion TSDF reconstruction in world space.
   */
  private createTsdfVisualizerObjects(): void {
    this.tsdfMeshGroup = new THREE.Group();
    this.tsdfMeshGroup.name = 'KinectFusionTSDFGroup';
    this.tsdfMeshGroup.visible = false;
    this.add(this.tsdfMeshGroup);

    const meshGeo = new THREE.BufferGeometry();
    const meshMat = new THREE.MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.48,
      metalness: 0.08,
      side: THREE.DoubleSide,
    });
    this.tsdfLiveMesh = new THREE.Mesh(meshGeo, meshMat);
    this.tsdfLiveMesh.name = 'DigitizedObjectMesh';
    this.tsdfMeshGroup.add(this.tsdfLiveMesh);

    const boxEdges = new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1));
    const boxMat = new THREE.LineBasicMaterial({
      color: 0x38bdf8,
      transparent: true,
      opacity: 0.75,
    });
    this.tsdfBoundingBox = new THREE.LineSegments(boxEdges, boxMat);
    this.tsdfBoundingBox.name = 'TSDFVolumeWireframe';
    this.tsdfBoundingBox.visible = false;
    this.add(this.tsdfBoundingBox);
  }

  /**
   * Builds the XR Blocks Spatial UI Card (`xb.UICard`) to the left of the user's main view
   * and the floating telemetry pill (`xb.UICard`) that appears above segmented masks.
   */
  private createSpatialHudCard(): void {
    const userHeight = xb.user?.height || 1.6;

    this.hudStatusText = new xb.UIText({
      text: 'Pinch & circle an object to segment & 3D digitize',
      style: {fontSize: 14},
    });

    this.hudMetricsText = new xb.UIText({
      text: 'Compiling EfficientSAM-Ti on LiteRT...',
      style: {fontSize: 12},
    });

    this.hudTsdfMetricsText = new xb.UIText({
      text: '3D TSDF Scan: Waiting for circle seed...',
      style: {fontSize: 12, color: '#38bdf8'},
    });

    this.hudCutoutPlaceholder = new xb.UIPanel({
      style: {
        width: '100%',
        height: 180,
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 6,
        borderRadius: 12,
        backgroundColor: '#0f172a',
        borderColor: '#38bdf8',
        borderWidth: 1.5,
      },
      children: [
        new xb.UIIcon({
          icon: 'view_in_ar',
          style: {fontSize: 28, color: '#38bdf8'},
        }),
        new xb.UIText({
          text: 'Circle-to-Digitize (KinectFusion)',
          style: {fontSize: 15, fontWeight: 'bold', color: '#e2e8f0'},
        }),
        new xb.UIText({
          text: 'Circle once, then move around object to fuse 3D mesh',
          style: {fontSize: 12, color: '#94a3b8'},
        }),
      ],
    });

    this.hudCutoutImage = new xb.UIImage({
      src: 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7',
      style: {
        width: '100%',
        height: 180,
        objectFit: 'contain',
        borderRadius: 12,
        display: 'none',
      },
    });

    const popOutBtn = new xb.UIButton({
      label: 'Pop Out 3D',
      icon: 'deployed_code',
      onClick: () => {
        this.finishAndPopOut3DModel();
      },
    });

    const exportBtn = new xb.UIButton({
      label: 'Export .GLB',
      icon: 'download',
      onClick: () => {
        this.exportDigitizedMeshGlb();
      },
    });

    const clearBtn = new xb.UIButton({
      label: 'Clear',
      icon: 'delete',
      onClick: () => {
        this.clearQuadOverlay();
      },
    });

    const card = new xb.UICard({
      size: {width: 0.56, height: 'auto'},
      manipulation: true,
      edge: true,
      style: {
        flexDirection: 'column',
        gap: 9,
        padding: 16,
      },
      children: [
        new xb.UIPanel({
          style: {
            flexDirection: 'row',
            alignItems: 'center',
            gap: 8,
          },
          children: [
            new xb.UIIcon({icon: 'view_in_ar'}),
            new xb.UIText({
              text: 'XR Circle to Digitize (SAM + TSDF)',
              style: {fontSize: 17},
            }),
          ],
        }),
        this.hudStatusText,
        this.hudMetricsText,
        this.hudTsdfMetricsText,
        this.hudCutoutPlaceholder,
        this.hudCutoutImage,
        new xb.UIPanel({
          style: {
            flexDirection: 'row',
            justifyContent: 'flex-start',
            gap: 8,
          },
          children: [popOutBtn, exportBtn, clearBtn],
        }),
      ],
    });

    card.position.set(-0.92, userHeight - 0.05, -1.55);
    card.rotation.y = 0.38;
    this.add(card);
    this.hudCard = card;

    this.telemetryBadgeText = new xb.UIText({
      text: 'Segmented',
      style: {
        fontSize: 13,
        fontWeight: 'bold',
        color: '#f8fafc',
        whiteSpace: 'nowrap',
      },
    });

    this.telemetryBadgeCard = new xb.UICard({
      size: {width: 0.54, height: 'auto'},
      appearance: 'surface',
      visible: false,
      pointerEvents: 'none',
      reticleMode: 'hidden',
      style: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 8,
        paddingTop: 8,
        paddingBottom: 8,
        paddingLeft: 16,
        paddingRight: 16,
        borderRadius: 999,
        backgroundColor: '#0f172a',
        opacity: 0.92,
        borderColor: '#38bdf8',
        borderWidth: 1.5,
      },
      children: [
        new xb.UIIcon({
          icon: 'auto_awesome',
          style: {fontSize: 16, color: '#38bdf8'},
        }),
        this.telemetryBadgeText,
      ],
    });
    this.add(this.telemetryBadgeCard);
  }

  private ensureWorker(): Worker {
    if (!this.worker) {
      this.worker = new Worker(
        new URL('./efficientsam_worker.js', import.meta.url)
      );
      this.worker.addEventListener('message', (event: MessageEvent) => {
        const {id, ok, result, error} = event.data as {
          id: number;
          ok: boolean;
          result?: unknown;
          error?: string;
        };
        const pending = this.pendingWorkerRequests.get(id);
        if (!pending) return;
        this.pendingWorkerRequests.delete(id);
        if (ok) {
          pending.resolve(result);
        } else {
          pending.reject(new Error(error || 'Worker request failed'));
        }
      });
      const rejectAllPending = (err: Error) => {
        for (const [, pending] of this.pendingWorkerRequests) {
          pending.reject(err);
        }
        this.pendingWorkerRequests.clear();
        this.isSegmenting = false;
        this.isIntegratingTsdf = false;
      };
      this.worker.addEventListener('error', (event: ErrorEvent) => {
        rejectAllPending(
          new Error(event.message || 'EfficientSAM Web Worker error')
        );
      });
      this.worker.addEventListener('messageerror', () => {
        rejectAllPending(
          new Error('EfficientSAM Web Worker message deserialization error')
        );
      });
    }
    return this.worker;
  }

  private callWorker<T>(
    type:
      | 'init'
      | 'xr_segment'
      | 'tsdf_integrate_depth'
      | 'tsdf_integrate_rgb_mask'
      | 'tsdf_reset',
    payload: Record<string, unknown> = {},
    transfer: Transferable[] = []
  ): Promise<T> {
    const worker = this.ensureWorker();
    const id = this.nextWorkerReqId++;
    return new Promise<T>((resolve, reject) => {
      this.pendingWorkerRequests.set(id, {
        resolve: resolve as (value: unknown) => void,
        reject,
      });
      worker.postMessage({id, type, payload}, transfer);
    });
  }

  private async initLiteRtModels(): Promise<void> {
    try {
      this.updateStatusText(
        'Compiling EfficientSAM-Ti in Web Worker (LiteRT 2.5.3)...'
      );
      const res = await this.callWorker<WorkerInitResult>('init');
      this.encoderAccelerator = res.encoderAccelerator;
      this.decoderAccelerator = res.decoderAccelerator;
      this.compileTimeMs = res.compileTimeMs;
      this.modelsReady = true;

      const accelLabel =
        this.encoderAccelerator === 'webgpu'
          ? this.decoderAccelerator === 'webgpu'
            ? 'WebGPU'
            : 'WebGPU + WASM'
          : 'WASM XNNPACK';

      this.updateStatusText(
        `Ready (${accelLabel} Worker) — Circle any object to digitize in 3D!`
      );
      if (this.hudMetricsText) {
        this.hudMetricsText.text = `LiteRT 2.5.3 (${accelLabel} Worker) | Compile: ${this.compileTimeMs.toFixed(0)} ms`;
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      console.error('Failed to initialize LiteRT worker models:', err);
      this.updateStatusText(`Model load error: ${message}`);
    }
  }

  private updateStatusText(msg: string): void {
    if (this.domStatus) {
      this.domStatus.innerHTML = msg;
    }
    if (this.hudStatusText) {
      this.hudStatusText.text = msg.replace(/<[^>]*>/g, '');
    }
  }

  private updateTsdfMetricsDisplay(tsdfMs?: number): void {
    const chartLabel =
      this.chartCount > 0 ? ` · ${this.chartCount} charts` : '';
    const summary =
      this.fusedFrameCount > 0
        ? `${this.fusedFrameCount} views${chartLabel} · ${this.triangleCount} tris (${this.voxelSizeMm.toFixed(1)}mm)`
        : 'Waiting for depth...';
    if (this.domMetricTsdf) {
      this.domMetricTsdf.textContent = summary;
    }
    if (this.hudTsdfMetricsText) {
      const msSuffix =
        typeof tsdfMs === 'number'
          ? ` | Fuse+GraphCut: ${tsdfMs.toFixed(1)} ms`
          : '';
      this.hudTsdfMetricsText.text = `3D TSDF Scan: ${summary}${msSuffix}`;
    }
  }

  private isDescendantOf(
    obj: THREE.Object3D | undefined,
    ancestor: THREE.Object3D | null
  ): boolean {
    if (!obj || !ancestor) return false;
    let cur: THREE.Object3D | null = obj;
    while (cur) {
      if (cur === ancestor) return true;
      cur = cur.parent;
    }
    return false;
  }

  private getControllerIndex(
    controller: xb.InteractionSource['controller'] | null
  ): number {
    if (!controller) return -1;
    const list = xb.user?.controllers || xb.core?.input?.controllers;
    if (Array.isArray(list)) {
      const idx = list.indexOf(controller);
      if (idx >= 0) return idx;
    }
    return typeof controller.userData?.id === 'number'
      ? controller.userData.id
      : -1;
  }

  private getControllerRay(
    controller: xb.InteractionSource['controller'],
    targetRay: THREE.Ray = this.tempRay
  ): THREE.Ray {
    controller.updateMatrixWorld(true);
    controller.getWorldPosition(targetRay.origin);
    const quat = new THREE.Quaternion();
    controller.getWorldQuaternion(quat);
    targetRay.direction.set(0, 0, -1).applyQuaternion(quat).normalize();
    if (targetRay.direction.lengthSq() > 0) {
      return targetRay;
    }
    const idx = this.getControllerIndex(controller);
    if (idx >= 0 && xb.user?.getRay) {
      xb.user.getRay(idx, targetRay);
    }
    return targetRay;
  }

  private isMouseController(
    controller: THREE.Object3D | null | undefined
  ): boolean {
    if (!controller) return false;
    return (
      controller === xb.core?.input?.mouseController ||
      (typeof xb.MouseController === 'function' &&
        controller instanceof xb.MouseController) ||
      (controller as {type?: string}).type === 'MouseController'
    );
  }

  /**
   * Called globally when any hand pinch or controller click begins.
   * Spawns the invisible quad 30 cm in front of the pinching hand
   * (or 1 m in front of the camera for MouseController)
   * so the XR Blocks Reticle can raycast and draw a circle on it.
   */
  override onSelectStart(event: xb.SelectEvent): void {
    const controller = event?.source?.controller;
    if (!controller || !this.circleQuad) {
      return;
    }

    // Do not spawn the circle quad if the user is interacting with the HUD card
    if (
      this.hudCard &&
      (xb.user?.isPointingAt?.(this.hudCard) ||
        xb.user?.isSelectingAt?.(this.hudCard) ||
        this.isDescendantOf(event?.target, this.hudCard) ||
        this.isDescendantOf(event?.surface, this.hudCard))
    ) {
      return;
    }

    if (this.isDrawingCircle || this.isSegmenting) {
      return;
    }

    // Starting a new circle resets any previous TSDF scan
    this.resetTsdfScanState();

    this.activeController = controller;
    this.isDrawingCircle = true;
    this.circlePath = [];
    if (this.telemetryBadgeCard) {
      this.telemetryBadgeCard.visible = false;
    }

    const isMouse =
      this.isMouseController(controller) ||
      Boolean(xb.core?.simulator && !xb.core?.renderer?.xr?.isPresenting);
    const quadDistance = isMouse
      ? MOUSE_QUAD_DISTANCE_METERS
      : QUAD_DISTANCE_METERS;
    const quadSize = isMouse ? MOUSE_QUAD_SIZE_METERS : QUAD_SIZE_METERS;

    // 1. Compute hand/controller ray origin and forward direction
    const ray = this.getControllerRay(controller);

    // 2. Place the invisible quad in front of the pinching hand (30cm) or mouse camera (1m)
    const quadCenter = ray.origin
      .clone()
      .addScaledVector(ray.direction, quadDistance);
    this.circleQuad.position.copy(quadCenter);
    this.circleQuad.scale.setScalar(quadSize / QUAD_SIZE_METERS);

    // Orient quad toward the viewer camera so the quad is perpendicular to line-of-sight
    const eyePos = new THREE.Vector3();
    xb.core.camera.getWorldPosition(eyePos);
    this.circleQuad.lookAt(eyePos);
    this.circleQuad.visible = true;
    this.circleQuad.xb = {pointerEvents: 'auto', reticleMode: 'auto'};
    this.circleQuad.updateMatrixWorld(true);

    // 3. Clear quad canvas so the quad starts invisible (except for reticle & stroke)
    this.quadCtx.clearRect(0, 0, MODEL_IMG_SIZE, MODEL_IMG_SIZE);
    this.quadTexture.needsUpdate = true;

    // 4. Sample the initial intersection on the newly placed quad
    this.sampleReticleOnQuad(controller);
    this.updateStatusText(
      isMouse
        ? 'Drawing circle on 1m quad with XR Blocks Reticle...'
        : 'Drawing circle on 30cm quad with XR Blocks Reticle...'
    );
  }

  /**
   * Called each frame while the user holds the pinch.
   */
  override onSelecting(event: xb.SelectEvent): void {
    const controller = event?.source?.controller;
    if (!controller || !this.isDrawingCircle) return;
    if (controller !== this.activeController) return;

    this.sampleReticleOnQuad(controller);
  }

  /**
   * Called when the user releases their pinch.
   * Triggers device camera capture, LiteRT EfficientSAM segmentation, and Frame 0 TSDF seeding.
   */
  override onSelectEnd(event: xb.SelectEndEvent): void {
    const controller = event?.source?.controller;
    if (!this.isDrawingCircle || controller !== this.activeController) {
      return;
    }

    this.isDrawingCircle = false;
    this.activeController = null;
    if (this.circleQuad) {
      this.circleQuad.xb = {pointerEvents: 'none', reticleMode: 'auto'};
    }

    if (this.circlePath.length === 0) {
      if (this.circleQuad) {
        this.circleQuad.visible = false;
      }
      return;
    }

    void this.executeCircleToSearch();
  }

  /**
   * Raycasts the active controller ray against the 30cm invisible quad's current
   * world transform and appends the UV + 3D world intersection to the Circle to Search path.
   */
  private sampleReticleOnQuad(
    controller: xb.InteractionSource['controller'],
    eventIntersection?: THREE.Intersection
  ): void {
    if (!this.circleQuad || !this.circleQuad.visible) return;

    let intersection: THREE.Intersection | null = null;

    // 1. Always raycast directly against this.circleQuad's current matrixWorld first.
    const ray = this.getControllerRay(controller);
    this.raycaster.ray.copy(ray);
    const hits = this.raycaster.intersectObject(this.circleQuad, false);
    if (hits.length > 0 && hits[0].uv) {
      intersection = hits[0];
      if (controller.reticle) {
        controller.reticle.visible = true;
        controller.reticle.position.copy(intersection.point);
      }
    }

    // 2. Fallback to XR Blocks Interaction intersection if direct raycast missed
    if (
      !intersection &&
      eventIntersection &&
      eventIntersection.object === this.circleQuad &&
      eventIntersection.uv
    ) {
      intersection = eventIntersection;
    }

    if (!intersection && xb.user?.getIntersectionAt) {
      const idx = this.getControllerIndex(controller);
      const hit = xb.user.getIntersectionAt(this.circleQuad, idx);
      if (hit && hit.uv) {
        intersection = hit;
      }
    }

    if (
      !intersection &&
      controller.reticle?.intersection?.object === this.circleQuad &&
      controller.reticle.intersection.uv
    ) {
      intersection = controller.reticle.intersection;
    }

    if (!intersection || !intersection.uv) return;

    const u = THREE.MathUtils.clamp(intersection.uv.x, 0, 1);
    const v = THREE.MathUtils.clamp(intersection.uv.y, 0, 1);
    const x = u * MODEL_IMG_SIZE;
    const y = (1.0 - v) * MODEL_IMG_SIZE;

    const lastPt = this.circlePath[this.circlePath.length - 1];
    if (lastPt && Math.hypot(x - lastPt.x, y - lastPt.y) < 2.0) {
      return;
    }

    this.circlePath.push({
      x,
      y,
      u,
      v,
      worldPoint: intersection.point.clone(),
    });
    this.renderCircleStrokeOnQuad(false);
  }

  /**
   * Renders the glowing Google "Circle to Search" stroke onto the 30cm quad's CanvasTexture.
   */
  private renderCircleStrokeOnQuad(
    showFinalBox = false,
    promptInfo: CirclePromptInfo | null = null
  ): void {
    const ctx = this.quadCtx;
    ctx.clearRect(0, 0, MODEL_IMG_SIZE, MODEL_IMG_SIZE);

    if (this.circlePath.length === 0) {
      this.quadTexture.needsUpdate = true;
      return;
    }

    ctx.save();
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    // Outer cyan glow pass
    ctx.beginPath();
    ctx.moveTo(this.circlePath[0].x, this.circlePath[0].y);
    for (let i = 1; i < this.circlePath.length; i++) {
      ctx.lineTo(this.circlePath[i].x, this.circlePath[i].y);
    }
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.55)';
    ctx.lineWidth = 16;
    ctx.shadowColor = 'rgba(59, 130, 246, 0.95)';
    ctx.shadowBlur = 18;
    ctx.stroke();

    // Inner bright core pass
    ctx.beginPath();
    ctx.moveTo(this.circlePath[0].x, this.circlePath[0].y);
    for (let i = 1; i < this.circlePath.length; i++) {
      ctx.lineTo(this.circlePath[i].x, this.circlePath[i].y);
    }
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.96)';
    ctx.lineWidth = 5.5;
    ctx.shadowColor = 'rgba(168, 85, 247, 0.85)';
    ctx.shadowBlur = 8;
    ctx.stroke();

    // Sparkle particles along the drawn trail
    const step = Math.max(1, Math.floor(this.circlePath.length / 10));
    for (let i = 0; i < this.circlePath.length; i += step) {
      const pt = this.circlePath[i];
      const offsetX = Math.sin(i * 2.3) * 8;
      const offsetY = Math.cos(i * 1.9) * 8;
      ctx.beginPath();
      ctx.arc(pt.x + offsetX, pt.y + offsetY, 2.8, 0, Math.PI * 2);
      ctx.fillStyle = i % 2 === 0 ? '#e0f2fe' : '#f5d0fe';
      ctx.fill();
    }

    // Current reticle tip ring
    const tip = this.circlePath[this.circlePath.length - 1];
    ctx.beginPath();
    ctx.arc(tip.x, tip.y, 8, 0, Math.PI * 2);
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2.5;
    ctx.stroke();

    if (showFinalBox && promptInfo?.box) {
      const {x1, y1, x2, y2} = promptInfo.box;
      ctx.setLineDash([7, 5]);
      ctx.strokeStyle = 'rgba(56, 189, 248, 0.75)';
      ctx.lineWidth = 2;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    }

    ctx.restore();
    this.quadTexture.needsUpdate = true;
  }

  /**
   * Captures the current `160x160` WebXR / Simulator depth buffer and camera matrices
   * as Transferable ArrayBuffers (< 0.15 ms copy).
   */
  private captureDepthPayload(): DepthTransferPayload | null {
    const depth = xb.core?.depth;
    if (
      !depth ||
      !depth.depthArray[0] ||
      depth.width <= 0 ||
      depth.height <= 0
    ) {
      return null;
    }

    const srcArray = depth.depthArray[0];
    if (srcArray.length === 0) return null;

    const clonedDepth =
      srcArray instanceof Float32Array
        ? new Float32Array(srcArray)
        : new Uint16Array(srcArray);

    const viewMat =
      depth.depthViewMatrices[0] ?? xb.core.camera.matrixWorldInverse;
    const projMat =
      depth.depthProjectionMatrices[0] ?? xb.core.camera.projectionMatrix;
    const projInvMat =
      depth.depthProjectionInverseMatrices[0] ??
      xb.core.camera.projectionMatrixInverse;
    const normMat =
      depth.normDepthBufferFromNormViewMatrices[0] ?? new THREE.Matrix4();

    const cameraPos = new THREE.Vector3();
    const cameraQuat = new THREE.Quaternion();
    if (depth.depthCameraPositions[0] && depth.depthCameraRotations[0]) {
      cameraPos.copy(depth.depthCameraPositions[0]);
      cameraQuat.copy(depth.depthCameraRotations[0]);
    } else {
      xb.core.camera.getWorldPosition(cameraPos);
      xb.core.camera.getWorldQuaternion(cameraQuat);
    }

    return {
      depthBuffer: clonedDepth.buffer,
      depthWidth: depth.width,
      depthHeight: depth.height,
      rawValueToMeters: depth.rawValueToMeters || 1.0,
      depthFormat:
        srcArray instanceof Uint16Array
          ? 'uint16'
          : (depth.depthDataFormat ?? 'float32'),
      depthViewMatrixBuffer: new Float32Array(viewMat.elements).buffer,
      depthProjectionMatrixBuffer: new Float32Array(projMat.elements).buffer,
      depthProjectionInverseMatrixBuffer: new Float32Array(projInvMat.elements)
        .buffer,
      normDepthBufferFromNormViewMatrixBuffer: new Float32Array(
        normMat.elements
      ).buffer,
      cameraPos,
      cameraQuat,
    };
  }

  /**
   * Computes the current RGB camera `clipFromWorld` matrix without reading back pixels.
   */
  private getRgbClipFromWorldMatrix(): THREE.Matrix4 {
    const renderCamera = xb.core.camera as THREE.PerspectiveCamera;
    renderCamera.updateMatrixWorld(true);

    const deviceCamera = xb.core.deviceCamera;
    const xrCameras = xb.core.renderer.xr.isPresenting
      ? (xb.core.renderer.xr.getCamera() as THREE.WebXRArrayCamera)
      : null;

    if (deviceCamera) {
      const cameraParams = xb.getCameraParametersSnapshot(
        renderCamera,
        xrCameras,
        deviceCamera,
        this.targetDevice
      );
      if (cameraParams) {
        return cameraParams.worldFromClip.clone().invert();
      }
    }

    return new THREE.Matrix4().multiplyMatrices(
      renderCamera.projectionMatrix,
      renderCamera.matrixWorldInverse
    );
  }

  /**
   * Captures the 512x512 RGB frame and synchronized `clipFromWorld` matrix.
   *
   * - In the Desktop Simulator, reads synchronously from `SimulatorCamera.canvas`
   *   (which contains only `simulatorScene` rendered at the exact current camera pose),
   *   avoiding Chrome's frame throttling on detached `<video>` elements.
   * - On WebXR headsets, awaits `deviceCamera.waitForFreshFrame()` before capturing
   *   and samples `clipFromWorld` immediately upon frame resolution.
   */
  private async captureCameraImage(): Promise<CameraCaptureResult> {
    const simCamCanvas = (
      xb.core.simulator?.simulatorCamera as
        | {canvas?: HTMLCanvasElement}
        | undefined
    )?.canvas;

    // 1. Desktop Simulator path: synchronous 0-lag read from SimulatorCamera's 512x512 canvas
    if (simCamCanvas && !xb.core.renderer.xr.isPresenting) {
      const clipFromWorld = this.getRgbClipFromWorldMatrix();
      this.captureCtx.clearRect(0, 0, MODEL_IMG_SIZE, MODEL_IMG_SIZE);
      this.captureCtx.drawImage(
        simCamCanvas,
        0,
        0,
        MODEL_IMG_SIZE,
        MODEL_IMG_SIZE
      );
      return {
        imageData: this.captureCtx.getImageData(
          0,
          0,
          MODEL_IMG_SIZE,
          MODEL_IMG_SIZE
        ),
        clipFromWorld,
      };
    }

    // 2. WebXR Device Camera path: wait for fresh frame and sample pose at capture time
    const deviceCamera = xb.core.deviceCamera;
    if (deviceCamera) {
      try {
        await deviceCamera.waitForFreshFrame?.(120);
      } catch {
        // Best-effort freshness wait
      }
      const snapshot = await deviceCamera.captureSnapshot({
        width: MODEL_IMG_SIZE,
        height: MODEL_IMG_SIZE,
        outputFormat: 'imageData',
      });
      if (snapshot instanceof ImageData) {
        const clipFromWorld = this.getRgbClipFromWorldMatrix();
        return {imageData: snapshot, clipFromWorld};
      }
    }

    // 3. Fallback if SimulatorCamera canvas is available
    if (simCamCanvas) {
      const clipFromWorld = this.getRgbClipFromWorldMatrix();
      this.captureCtx.clearRect(0, 0, MODEL_IMG_SIZE, MODEL_IMG_SIZE);
      this.captureCtx.drawImage(
        simCamCanvas,
        0,
        0,
        MODEL_IMG_SIZE,
        MODEL_IMG_SIZE
      );
      return {
        imageData: this.captureCtx.getImageData(
          0,
          0,
          MODEL_IMG_SIZE,
          MODEL_IMG_SIZE
        ),
        clipFromWorld,
      };
    }

    throw new Error('Device camera image is not available yet.');
  }

  /**
   * Projects the 3D circle path points drawn on the 30cm quad into [0..512] camera
   * image coordinates using `clipFromWorld` and builds the EfficientSAM prompt:
   * - Freehand loop / stroke: Bounding Box [labels 2, 3] + Centroid FG Point [label 1]
   * - Quick tap: Single FG Point [label 1]
   */
  private buildPromptFromCircle(
    clipFromWorld: THREE.Matrix4
  ): CirclePromptInfo {
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    let sumX = 0;
    let sumY = 0;

    const tempNdc = new THREE.Vector3();
    for (const pt of this.circlePath) {
      tempNdc.copy(pt.worldPoint).applyMatrix4(clipFromWorld);
      const camU = (tempNdc.x + 1.0) * 0.5;
      const camV = 1.0 - (tempNdc.y + 1.0) * 0.5;
      const camX = THREE.MathUtils.clamp(
        camU * MODEL_IMG_SIZE,
        0,
        MODEL_IMG_SIZE
      );
      const camY = THREE.MathUtils.clamp(
        camV * MODEL_IMG_SIZE,
        0,
        MODEL_IMG_SIZE
      );

      if (camX < minX) minX = camX;
      if (camY < minY) minY = camY;
      if (camX > maxX) maxX = camX;
      if (camY > maxY) maxY = camY;
      sumX += camX;
      sumY += camY;
    }

    const n = this.circlePath.length;
    const span = Math.hypot(maxX - minX, maxY - minY);

    const pts = new Float32Array(MAX_POINTS * 2).fill(-1.0);
    const lbls = new Float32Array(MAX_POINTS).fill(-1.0);

    if (span < 14) {
      const cx = THREE.MathUtils.clamp(sumX / n, 0, MODEL_IMG_SIZE);
      const cy = THREE.MathUtils.clamp(sumY / n, 0, MODEL_IMG_SIZE);
      pts[0] = cx;
      pts[1] = cy;
      lbls[0] = 1.0;
      return {pts, lbls, box: null, center: {x: cx, y: cy}};
    }

    const pad = Math.max(4, span * 0.04);
    const x1 = THREE.MathUtils.clamp(minX - pad, 0, MODEL_IMG_SIZE);
    const y1 = THREE.MathUtils.clamp(minY - pad, 0, MODEL_IMG_SIZE);
    const x2 = THREE.MathUtils.clamp(maxX + pad, 0, MODEL_IMG_SIZE);
    const y2 = THREE.MathUtils.clamp(maxY + pad, 0, MODEL_IMG_SIZE);
    const cx = THREE.MathUtils.clamp((x1 + x2) * 0.5, 0, MODEL_IMG_SIZE);
    const cy = THREE.MathUtils.clamp((y1 + y2) * 0.5, 0, MODEL_IMG_SIZE);

    // Box top-left (label 2), Box bottom-right (label 3), Center FG point (label 1)
    pts[0] = x1;
    pts[1] = y1;
    lbls[0] = 2.0;

    pts[2] = x2;
    pts[3] = y2;
    lbls[1] = 3.0;

    pts[4] = cx;
    pts[5] = cy;
    lbls[2] = 1.0;

    return {
      pts,
      lbls,
      box: {x1, y1, x2, y2},
      center: {x: cx, y: cy},
    };
  }

  /**
   * Runs EfficientSAM-Ti Encoder + Decoder + mask reprojection + Frame 0 TSDF seeding
   * inside `efficientsam_worker.js` so the main WebXR thread never blocks.
   */
  private async executeCircleToSearch(): Promise<void> {
    if (!this.modelsReady) {
      this.updateStatusText(
        'LiteRT models are still compiling in worker, please wait...'
      );
      return;
    }

    this.isSegmenting = true;
    this.updateStatusText(
      'Segmenting & seeding 3D TSDF volume in Web Worker...'
    );

    try {
      // 1. Capture the 512x512 RGB image directly from the device camera
      const {imageData: cameraImageData, clipFromWorld} =
        await this.captureCameraImage();

      // 2. Capture the current 160x160 WebXR Depth frame for Frame 0 TSDF seeding
      const depthPayload = this.captureDepthPayload();

      // 3. Build Camera-Space Circle Prompt & quadToClip matrix
      const promptInfo = this.buildPromptFromCircle(clipFromWorld);
      const quadToClip = new THREE.Matrix4();
      if (this.circleQuad) {
        this.circleQuad.updateMatrixWorld(true);
        quadToClip.multiplyMatrices(clipFromWorld, this.circleQuad.matrixWorld);
      }
      const quadToClipElements = new Float32Array(quadToClip.elements);
      const rgbClipFromWorldElements = new Float32Array(clipFromWorld.elements);

      const rgbaBuffer = cameraImageData.data.buffer.slice(0);
      const ptsBuffer = promptInfo.pts.buffer.slice(0);
      const lblsBuffer = promptInfo.lbls.buffer.slice(0);
      const quadToClipBuffer = quadToClipElements.buffer;
      const rgbClipFromWorldMatrixBuffer = rgbClipFromWorldElements.buffer;

      const workerPayload: Record<string, unknown> = {
        rgbaBuffer,
        ptsBuffer,
        lblsBuffer,
        quadToClipBuffer,
        rgbClipFromWorldMatrixBuffer,
        quadSizeMeters: QUAD_SIZE_METERS,
      };
      const transferList: Transferable[] = [
        rgbaBuffer,
        ptsBuffer,
        lblsBuffer,
        quadToClipBuffer,
        rgbClipFromWorldMatrixBuffer,
      ];

      if (depthPayload) {
        workerPayload.depthBuffer = depthPayload.depthBuffer;
        workerPayload.depthWidth = depthPayload.depthWidth;
        workerPayload.depthHeight = depthPayload.depthHeight;
        workerPayload.rawValueToMeters = depthPayload.rawValueToMeters;
        workerPayload.depthFormat = depthPayload.depthFormat;
        workerPayload.depthViewMatrixBuffer =
          depthPayload.depthViewMatrixBuffer;
        workerPayload.depthProjectionMatrixBuffer =
          depthPayload.depthProjectionMatrixBuffer;
        workerPayload.depthProjectionInverseMatrixBuffer =
          depthPayload.depthProjectionInverseMatrixBuffer;
        workerPayload.normDepthBufferFromNormViewMatrixBuffer =
          depthPayload.normDepthBufferFromNormViewMatrixBuffer;

        transferList.push(
          depthPayload.depthBuffer,
          depthPayload.depthViewMatrixBuffer,
          depthPayload.depthProjectionMatrixBuffer,
          depthPayload.depthProjectionInverseMatrixBuffer,
          depthPayload.normDepthBufferFromNormViewMatrixBuffer
        );
      }

      // 4. Run preprocessing + LiteRT Encoder + Decoder + TSDF seeding in Web Worker
      const workerResult = await this.callWorker<WorkerXrSegmentResult>(
        'xr_segment',
        workerPayload,
        transferList
      );

      if (depthPayload && workerResult.tsdfMesh) {
        this.lastFusedCameraPos.copy(depthPayload.cameraPos);
        this.lastFusedCameraQuat.copy(depthPayload.cameraQuat);
        const now = performance.now();
        this.lastDepthFusionMs = now;
        this.lastRgbMaskFusionMs = now;
      }

      // 5. Present the precomputed RGBA quad overlay, cutout, and live 3D TSDF mesh
      this.presentWorkerSegmentationResult(workerResult);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      console.error('XR Circle to Digitize segmentation failed:', err);
      this.updateStatusText(`Segmentation error: ${message}`);
    } finally {
      this.isSegmenting = false;
    }
  }

  /**
   * Applies the precomputed RGBA overlay, cropped 2D cutout, and Frame 0 TSDF 3D mesh
   * returned by the Web Worker.
   */
  private presentWorkerSegmentationResult(res: WorkerXrSegmentResult): void {
    const W = MODEL_IMG_SIZE;
    const H = MODEL_IMG_SIZE;

    const quadOverlay = new ImageData(
      new Uint8ClampedArray(res.quadOverlayBuffer),
      W,
      H
    );
    this.quadCtx.putImageData(quadOverlay, 0, 0);

    // Draw the circle trail on top of the quad
    if (this.circlePath.length > 1) {
      this.quadCtx.save();
      this.quadCtx.beginPath();
      this.quadCtx.moveTo(this.circlePath[0].x, this.circlePath[0].y);
      for (let i = 1; i < this.circlePath.length; i++) {
        this.quadCtx.lineTo(this.circlePath[i].x, this.circlePath[i].y);
      }
      this.quadCtx.strokeStyle = 'rgba(255, 255, 255, 0.85)';
      this.quadCtx.lineWidth = 3.5;
      this.quadCtx.shadowColor = 'rgba(56, 189, 248, 0.9)';
      this.quadCtx.shadowBlur = 10;
      this.quadCtx.stroke();
      this.quadCtx.restore();
    }

    this.quadTexture.needsUpdate = true;

    // Update and position the native XR Blocks Spatial UI telemetry pill card above the mask
    if (this.telemetryBadgeCard && this.telemetryBadgeText && this.circleQuad) {
      this.telemetryBadgeText.text = `Segmented in ${res.totalMs.toFixed(1)} ms · Scanning 3D TSDF...`;
      const badgePixelY = Math.max(
        36,
        (res.quadMinY < H ? res.quadMinY : 96) - 36
      );
      const v = 1.0 - badgePixelY / H;
      const localBadgePos = new THREE.Vector3(
        0,
        (v - 0.5) * QUAD_SIZE_METERS,
        0.01
      );
      this.circleQuad.updateMatrixWorld(true);
      localBadgePos.applyMatrix4(this.circleQuad.matrixWorld);
      this.telemetryBadgeCard.position.copy(localBadgePos);
      this.telemetryBadgeCard.quaternion.copy(this.circleQuad.quaternion);
      this.telemetryBadgeCard.scale.setScalar(this.circleQuad.scale.x * 0.55);
      this.telemetryBadgeCard.visible = true;
    }

    if (res.cutoutRgbaBuffer && res.cropW > 0 && res.cropH > 0) {
      this.updateCutoutPreview(res.cutoutRgbaBuffer, res.cropW, res.cropH);
    }

    const coveragePct = ((res.fgCount / (W * H)) * 100).toFixed(1);
    if (this.domMetricTotal) {
      this.domMetricTotal.textContent = `${res.totalMs.toFixed(1)} ms`;
    }
    if (this.domMetricSplit) {
      this.domMetricSplit.textContent = `${res.encoderMs.toFixed(0)} / ${res.decoderMs.toFixed(1)} ms`;
    }
    if (this.domMetricIou) {
      this.domMetricIou.textContent = `${res.bestIou.toFixed(3)} (${coveragePct}%)`;
    }

    if (this.hudMetricsText) {
      this.hudMetricsText.text = `SAM: ${res.totalMs.toFixed(1)}ms (Enc ${res.encoderMs.toFixed(0)}ms / Dec ${res.decoderMs.toFixed(1)}ms) | IoU: ${res.bestIou.toFixed(2)}`;
    }

    if (res.fgCount > 16) {
      this.isScanningTsdf = true;
      this.isPoppedOut = false;
      if (res.tsdfMesh) {
        this.applyTsdfMeshPayload(res.tsdfMesh);
        this.updateStatusText(
          `<strong>3D Digitizing!</strong> Move around object to fuse views (${this.fusedFrameCount} views · ${this.triangleCount} tris).`
        );
      } else {
        this.updateTsdfMetricsDisplay();
        this.updateStatusText(
          `<strong>Segmented!</strong> Waiting for WebXR Depth to seed 3D TSDF volume...`
        );
      }
    } else {
      this.updateStatusText(
        `<strong>Segmented!</strong> Pinch & circle again anywhere in XR.`
      );
    }
  }

  private updateCutoutPreview(
    cutoutRgbaBuffer: ArrayBuffer,
    cropW: number,
    cropH: number
  ): void {
    const cutoutDataUrl = this.buildCutoutDataUrlFromCrop(
      cutoutRgbaBuffer,
      cropW,
      cropH
    );
    if (this.hudCutoutImage) {
      this.hudCutoutImage.src = cutoutDataUrl;
      this.hudCutoutImage.style.display = 'flex';
    }
    if (this.hudCutoutPlaceholder) {
      this.hudCutoutPlaceholder.style.display = 'none';
    }
  }

  /**
   * Zero-copy main-thread `BufferGeometry` + Graph-Cut UV Atlas Texture update
   * using the `Float32Array` and `Uint8Array` buffers transferred from the Web Worker.
   */
  private applyTsdfMeshPayload(mesh: WorkerTsdfMeshPayload): void {
    this.fusedFrameCount = mesh.fusedFrameCount;
    this.triangleCount = mesh.triangleCount;
    this.chartCount = mesh.chartCount ?? 0;
    this.voxelSizeMm = mesh.voxelSizeMm;
    this.currentVolumeCenter.set(
      mesh.volumeCenter.x,
      mesh.volumeCenter.y,
      mesh.volumeCenter.z
    );
    this.currentVolumeSize = mesh.volumeSizeMeters;

    if (this.tsdfBoundingBox && !this.isPoppedOut) {
      if (mesh.boundsMin && mesh.boundsMax) {
        const bMin = mesh.boundsMin;
        const bMax = mesh.boundsMax;
        this.tsdfBoundingBox.position.set(
          (bMin.x + bMax.x) * 0.5,
          (bMin.y + bMax.y) * 0.5,
          (bMin.z + bMax.z) * 0.5
        );
        this.tsdfBoundingBox.scale.set(
          Math.max(0.02, bMax.x - bMin.x),
          Math.max(0.02, bMax.y - bMin.y),
          Math.max(0.02, bMax.z - bMin.z)
        );
      } else {
        this.tsdfBoundingBox.position.copy(this.currentVolumeCenter);
        this.tsdfBoundingBox.scale.setScalar(this.currentVolumeSize);
      }
      this.tsdfBoundingBox.visible = true;
    }

    if (this.tsdfLiveMesh && this.tsdfMeshGroup) {
      const positions = new Float32Array(mesh.positionsBuffer);
      const normals = new Float32Array(mesh.normalsBuffer);
      const uvs = new Float32Array(mesh.uvsBuffer);
      const colors = new Float32Array(mesh.colorsBuffer);

      const geom = new THREE.BufferGeometry();
      if (positions.length > 0) {
        geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geom.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
        if (uvs.length > 0) {
          geom.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
        }
        geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        geom.computeBoundingSphere();
      }

      if (
        mesh.atlasRgbaBuffer &&
        mesh.atlasRgbaBuffer.byteLength > 0 &&
        mesh.atlasWidth > 0 &&
        mesh.atlasHeight > 0 &&
        uvs.length > 0
      ) {
        const sizeChanged =
          this.atlasCanvas.width !== mesh.atlasWidth ||
          this.atlasCanvas.height !== mesh.atlasHeight;
        if (sizeChanged) {
          this.atlasCanvas.width = mesh.atlasWidth;
          this.atlasCanvas.height = mesh.atlasHeight;
        }
        this.atlasCtx.putImageData(
          new ImageData(
            new Uint8ClampedArray(mesh.atlasRgbaBuffer),
            mesh.atlasWidth,
            mesh.atlasHeight
          ),
          0,
          0
        );
        if (sizeChanged) {
          this.atlasTexture.dispose();
          this.atlasTexture = new THREE.CanvasTexture(this.atlasCanvas);
          this.atlasTexture.flipY = false;
          this.atlasTexture.colorSpace = THREE.SRGBColorSpace;
          this.atlasTexture.minFilter = THREE.LinearFilter;
          this.atlasTexture.magFilter = THREE.LinearFilter;
          this.atlasTexture.generateMipmaps = false;
        }
        this.atlasTexture.needsUpdate = true;

        const mat = this.tsdfLiveMesh.material;
        if (mat.map !== this.atlasTexture || mat.vertexColors !== true) {
          mat.map = this.atlasTexture;
          mat.vertexColors = true;
          mat.color.setHex(0xffffff);
          mat.needsUpdate = true;
        }
      } else {
        const mat = this.tsdfLiveMesh.material;
        if (mat.map !== null || mat.vertexColors !== true) {
          mat.map = null;
          mat.vertexColors = true;
          mat.color.setHex(0xffffff);
          mat.needsUpdate = true;
        }
      }

      this.tsdfLiveMesh.geometry.dispose();
      this.tsdfLiveMesh.geometry = geom;

      if (!this.isPoppedOut) {
        this.tsdfMeshGroup.position.set(0, 0, 0);
        this.tsdfMeshGroup.rotation.set(0, 0, 0);
        this.tsdfMeshGroup.scale.setScalar(1.0);
        this.tsdfLiveMesh.position.set(0, 0, 0);
      }
      this.tsdfMeshGroup.visible = mesh.triangleCount > 0;
    }

    if (
      this.telemetryBadgeCard &&
      this.telemetryBadgeText &&
      this.isScanningTsdf
    ) {
      const chartLabel =
        this.chartCount > 0 ? ` · ${this.chartCount} charts` : '';
      this.telemetryBadgeText.text = `Scanning 3D: ${this.fusedFrameCount} views${chartLabel} · ${this.triangleCount} tris (${mesh.tsdfMs.toFixed(1)}ms)`;
    }

    this.updateTsdfMetricsDisplay(mesh.tsdfMs);
  }

  /**
   * Streams a fast depth-only keyframe (`102 KB`) to the Worker for initial Frame 0
   * TSDF seeding if WebXR depth warmed up after `onSelectEnd`.
   */
  private async integrateDepthKeyframe(
    depthPayload: DepthTransferPayload
  ): Promise<void> {
    if (this.isIntegratingTsdf || this.isSegmenting) return;
    this.isIntegratingTsdf = true;
    this.lastDepthFusionMs = performance.now();

    try {
      const clipFromWorld = this.getRgbClipFromWorldMatrix();
      const rgbClipFromWorldMatrixBuffer = new Float32Array(
        clipFromWorld.elements
      ).buffer;

      const res = await this.callWorker<WorkerTsdfMeshPayload | null>(
        'tsdf_integrate_depth',
        {
          depthBuffer: depthPayload.depthBuffer,
          depthWidth: depthPayload.depthWidth,
          depthHeight: depthPayload.depthHeight,
          rawValueToMeters: depthPayload.rawValueToMeters,
          depthFormat: depthPayload.depthFormat,
          depthViewMatrixBuffer: depthPayload.depthViewMatrixBuffer,
          depthProjectionMatrixBuffer: depthPayload.depthProjectionMatrixBuffer,
          depthProjectionInverseMatrixBuffer:
            depthPayload.depthProjectionInverseMatrixBuffer,
          normDepthBufferFromNormViewMatrixBuffer:
            depthPayload.normDepthBufferFromNormViewMatrixBuffer,
          rgbClipFromWorldMatrixBuffer,
        },
        [
          depthPayload.depthBuffer,
          depthPayload.depthViewMatrixBuffer,
          depthPayload.depthProjectionMatrixBuffer,
          depthPayload.depthProjectionInverseMatrixBuffer,
          depthPayload.normDepthBufferFromNormViewMatrixBuffer,
          rgbClipFromWorldMatrixBuffer,
        ]
      );

      if (res && this.isScanningTsdf) {
        this.lastFusedCameraPos.copy(depthPayload.cameraPos);
        this.lastFusedCameraQuat.copy(depthPayload.cameraQuat);
        this.applyTsdfMeshPayload(res);

        // Once the user starts moving around to scan, fade out the 2D circle quad so they have an unobstructed view of the 3D mesh
        if (this.fusedFrameCount >= 2 && this.circleQuad?.visible) {
          this.circleQuad.visible = false;
        }

        this.updateStatusText(
          `<strong>3D Digitizing!</strong> Move around object (${this.fusedFrameCount} views · ${this.triangleCount} tris).`
        );
      }
    } catch (err) {
      console.warn('TSDF depth keyframe integration skipped:', err);
    } finally {
      this.isIntegratingTsdf = false;
    }
  }

  /**
   * Runs automatic 3D-to-2D bounding-box projection + EfficientSAM segmentation
   * + support-plane clipping + Graph-Cut multi-view texturing in the Worker for
   * every new viewpoint so background/table surfaces never leak into the 3D mesh.
   */
  private async integrateRgbMaskKeyframe(
    depthPayload: DepthTransferPayload
  ): Promise<void> {
    if (this.isIntegratingTsdf || this.isSegmenting) return;
    this.isIntegratingTsdf = true;
    const now = performance.now();
    this.lastRgbMaskFusionMs = now;
    this.lastDepthFusionMs = now;

    try {
      const {imageData, clipFromWorld} = await this.captureCameraImage();
      const rgbaBuffer = imageData.data.buffer.slice(0);
      const rgbClipFromWorldMatrixBuffer = new Float32Array(
        clipFromWorld.elements
      ).buffer;

      const res = await this.callWorker<WorkerRgbMaskIntegrateResult | null>(
        'tsdf_integrate_rgb_mask',
        {
          rgbaBuffer,
          rgbClipFromWorldMatrixBuffer,
          depthBuffer: depthPayload.depthBuffer,
          depthWidth: depthPayload.depthWidth,
          depthHeight: depthPayload.depthHeight,
          rawValueToMeters: depthPayload.rawValueToMeters,
          depthFormat: depthPayload.depthFormat,
          depthViewMatrixBuffer: depthPayload.depthViewMatrixBuffer,
          depthProjectionMatrixBuffer: depthPayload.depthProjectionMatrixBuffer,
          normDepthBufferFromNormViewMatrixBuffer:
            depthPayload.normDepthBufferFromNormViewMatrixBuffer,
        },
        [
          rgbaBuffer,
          rgbClipFromWorldMatrixBuffer,
          depthPayload.depthBuffer,
          depthPayload.depthViewMatrixBuffer,
          depthPayload.depthProjectionMatrixBuffer,
          depthPayload.normDepthBufferFromNormViewMatrixBuffer,
        ]
      );

      if (res?.tsdfMesh && this.isScanningTsdf) {
        this.lastFusedCameraPos.copy(depthPayload.cameraPos);
        this.lastFusedCameraQuat.copy(depthPayload.cameraQuat);
        this.applyTsdfMeshPayload(res.tsdfMesh);

        if (this.fusedFrameCount >= 2 && this.circleQuad?.visible) {
          this.circleQuad.visible = false;
        }

        if (res.cutoutRgbaBuffer && res.cropW > 0 && res.cropH > 0) {
          this.updateCutoutPreview(res.cutoutRgbaBuffer, res.cropW, res.cropH);
        }

        this.updateStatusText(
          `<strong>3D Digitizing!</strong> Move around object (${this.fusedFrameCount} views · ${this.chartCount} charts · ${this.triangleCount} tris).`
        );
      }
    } catch (err) {
      console.warn('TSDF RGB+Mask keyframe integration skipped:', err);
    } finally {
      this.isIntegratingTsdf = false;
    }
  }

  /**
   * Stops scanning and pops out the reconstructed 3D mesh onto a rotating turntable
   * in front of the user for inspection.
   */
  public finishAndPopOut3DModel(): void {
    if (!this.tsdfLiveMesh || !this.tsdfMeshGroup || this.triangleCount === 0) {
      this.updateStatusText(
        'No 3D mesh reconstructed yet — pinch & circle an object first!'
      );
      return;
    }

    this.isScanningTsdf = false;
    this.isPoppedOut = true;

    if (this.circleQuad) {
      this.circleQuad.visible = false;
      this.circleQuad.xb = {pointerEvents: 'none', reticleMode: 'auto'};
    }
    if (this.tsdfBoundingBox) {
      this.tsdfBoundingBox.visible = false;
    }
    if (this.telemetryBadgeCard) {
      this.telemetryBadgeCard.visible = false;
    }

    // Center the mesh geometry around (0, 0, 0) inside tsdfMeshGroup so it rotates around its own centroid
    this.tsdfLiveMesh.geometry.computeBoundingBox();
    const bbox = this.tsdfLiveMesh.geometry.boundingBox;
    if (bbox) {
      const center = new THREE.Vector3();
      bbox.getCenter(center);
      this.tsdfLiveMesh.position.copy(center).multiplyScalar(-1);
    }

    // Place the popped-out 3D model 0.65m in front of the user's current view
    const camPos = new THREE.Vector3();
    const camQuat = new THREE.Quaternion();
    xb.core.camera.getWorldPosition(camPos);
    xb.core.camera.getWorldQuaternion(camQuat);
    const forward = new THREE.Vector3(0, 0, -1)
      .applyQuaternion(camQuat)
      .normalize();

    this.tsdfMeshGroup.position.copy(camPos).addScaledVector(forward, 0.65);
    this.tsdfMeshGroup.rotation.set(0, 0, 0);
    this.tsdfMeshGroup.scale.setScalar(1.15);
    this.tsdfMeshGroup.visible = true;

    this.updateStatusText(
      `<strong>3D Model Popped Out!</strong> (${this.fusedFrameCount} views · ${this.chartCount} charts · ${this.triangleCount} tris) — Click Export .GLB to save.`
    );
  }

  /**
   * Exports the digitized Graph-Cut UV-textured `THREE.Mesh` as a binary `.glb` file.
   */
  public exportDigitizedMeshGlb(): void {
    if (!this.tsdfLiveMesh || this.triangleCount === 0) {
      this.updateStatusText(
        'No 3D mesh to export yet — pinch & circle an object first!'
      );
      return;
    }

    // Clone geometry centered at origin for clean GLB export
    const exportGeom = this.tsdfLiveMesh.geometry.clone();
    exportGeom.computeBoundingBox();
    if (exportGeom.boundingBox) {
      const center = new THREE.Vector3();
      exportGeom.boundingBox.getCenter(center);
      exportGeom.translate(-center.x, -center.y, -center.z);
    }
    const exportMesh = new THREE.Mesh(
      exportGeom,
      this.tsdfLiveMesh.material.clone()
    );
    exportMesh.name = 'CircleToDigitizeMesh';

    const exporter = new GLTFExporter();
    exporter.parse(
      exportMesh,
      (result) => {
        exportGeom.dispose();
        if (result instanceof ArrayBuffer) {
          const blob = new Blob([result], {type: 'model/gltf-binary'});
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = `circle_to_digitize_${this.fusedFrameCount}views.glb`;
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
          URL.revokeObjectURL(url);
          this.updateStatusText(
            `<strong>Exported .GLB!</strong> (${this.triangleCount} triangles from ${this.fusedFrameCount} fused views)`
          );
        }
      },
      (err) => {
        exportGeom.dispose();
        console.error('GLTFExporter failed:', err);
        this.updateStatusText('Failed to export .GLB file.');
      },
      {binary: true}
    );
  }

  /**
   * Builds a 320x320 card preview DataURL from the pre-cropped RGBA buffer computed in the Web Worker.
   */
  private buildCutoutDataUrlFromCrop(
    cutoutRgbaBuffer: ArrayBuffer,
    cropW: number,
    cropH: number
  ): string {
    const cropCanvas = document.createElement('canvas');
    cropCanvas.width = cropW;
    cropCanvas.height = cropH;
    const cropCtx = cropCanvas.getContext('2d')!;
    cropCtx.putImageData(
      new ImageData(new Uint8ClampedArray(cutoutRgbaBuffer), cropW, cropH),
      0,
      0
    );

    const ctx = this.cutoutCanvas.getContext('2d')!;
    ctx.clearRect(0, 0, 320, 320);

    ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
    ctx.fillRect(0, 0, 320, 320);

    const pad = 24;
    const scale = Math.min(
      (320 - pad * 2) / Math.max(16, cropW),
      (320 - pad * 2) / Math.max(16, cropH)
    );
    const drawW = cropW * scale;
    const drawH = cropH * scale;
    const dx = (320 - drawW) / 2;
    const dy = (320 - drawH) / 2;

    ctx.save();
    ctx.shadowColor = 'rgba(56, 189, 248, 0.75)';
    ctx.shadowBlur = 14;
    ctx.drawImage(cropCanvas, 0, 0, cropW, cropH, dx, dy, drawW, drawH);
    ctx.restore();

    ctx.strokeStyle = 'rgba(56, 189, 248, 0.5)';
    ctx.lineWidth = 3;
    ctx.strokeRect(6, 6, 308, 308);

    return this.cutoutCanvas.toDataURL('image/png');
  }

  private resetTsdfScanState(): void {
    this.isScanningTsdf = false;
    this.isPoppedOut = false;
    this.fusedFrameCount = 0;
    this.triangleCount = 0;
    this.chartCount = 0;
    this.voxelSizeMm = 0;
    this.lastFusedCameraPos.set(NaN, NaN, NaN);
    this.lastFusedCameraQuat.set(NaN, NaN, NaN, NaN);

    if (this.tsdfMeshGroup) {
      this.tsdfMeshGroup.visible = false;
    }
    if (this.tsdfLiveMesh) {
      this.tsdfLiveMesh.geometry.dispose();
      this.tsdfLiveMesh.geometry = new THREE.BufferGeometry();
      this.tsdfLiveMesh.material.map = null;
      this.tsdfLiveMesh.material.vertexColors = true;
      this.tsdfLiveMesh.material.needsUpdate = true;
    }
    if (this.tsdfBoundingBox) {
      this.tsdfBoundingBox.visible = false;
    }
    if (this.modelsReady) {
      void this.callWorker('tsdf_reset').catch(() => {});
    }
    this.updateTsdfMetricsDisplay();
  }

  /**
   * Hides and clears the 30cm circle quad and resets the 3D TSDF volume.
   */
  public clearQuadOverlay(): void {
    this.circlePath = [];
    this.isDrawingCircle = false;
    this.activeController = null;
    this.quadCtx.clearRect(0, 0, MODEL_IMG_SIZE, MODEL_IMG_SIZE);
    this.quadTexture.needsUpdate = true;
    if (this.circleQuad) {
      this.circleQuad.visible = false;
      this.circleQuad.xb = {pointerEvents: 'none', reticleMode: 'auto'};
    }
    if (this.telemetryBadgeCard) {
      this.telemetryBadgeCard.visible = false;
    }
    if (this.hudCutoutImage) {
      this.hudCutoutImage.style.display = 'none';
    }
    if (this.hudCutoutPlaceholder) {
      this.hudCutoutPlaceholder.style.display = 'flex';
    }
    this.resetTsdfScanState();
    this.updateStatusText(
      'Cleared — Pinch & circle any object to segment & 3D digitize'
    );
  }

  /**
   * Per-frame update loop (72+ FPS).
   */
  override update(): void {
    // Keep the idle reticle at 1m for Simulator/MouseController and 30cm for XR device controllers
    const isSimulatorOrMouse =
      Boolean(xb.core?.simulator && !xb.core?.renderer?.xr?.isPresenting) ||
      xb.core?.input?.mouseController?.userData?.connected === true ||
      !xb.core?.renderer?.xr?.isPresenting;
    const targetDistance = isSimulatorOrMouse
      ? MOUSE_QUAD_DISTANCE_METERS
      : QUAD_DISTANCE_METERS;

    if (xb.core?.options?.reticles) {
      xb.core.options.reticles.defaultRenderDistance = targetDistance;
    }
    const coreWithReticleOptions = xb.core as unknown as {
      reticleOptions?: {defaultRenderDistance: number};
    };
    if (coreWithReticleOptions?.reticleOptions) {
      coreWithReticleOptions.reticleOptions.defaultRenderDistance =
        targetDistance;
    }

    // Automatic fallback if a WebXR headset provides GPU depth instead of CPU depth
    const depth = xb.core?.depth;
    if (
      depth &&
      depth.gpuDepthData.length > 0 &&
      !depth.depthArray[0] &&
      !depth.options.depthMesh.enabled
    ) {
      depth.options.depthMesh.enabled = true;
    }

    // Ensure continuous reticle sampling while drawing a circle
    if (this.isDrawingCircle && this.activeController) {
      this.sampleReticleOnQuad(this.activeController);
      return;
    }

    // Gently rotate popped-out 3D model on a turntable
    if (this.isPoppedOut && this.tsdfMeshGroup?.visible) {
      this.tsdfMeshGroup.rotation.y += 0.012;
      return;
    }

    // Continuous multi-view KinectFusion TSDF scanning when camera moves
    if (
      this.isScanningTsdf &&
      !this.isSegmenting &&
      !this.isIntegratingTsdf &&
      this.modelsReady
    ) {
      if (
        this.hudCard &&
        (xb.user?.isPointingAt?.(this.hudCard) ||
          xb.user?.isSelectingAt?.(this.hudCard))
      ) {
        return;
      }

      const now = performance.now();
      const needsInitialSeed = this.fusedFrameCount === 0;
      const requiredIntervalMs = needsInitialSeed
        ? DEPTH_FUSION_INTERVAL_MS
        : RGB_MASK_FUSION_INTERVAL_MS;
      if (now - this.lastDepthFusionMs < requiredIntervalMs) {
        return;
      }

      const depthPayload = this.captureDepthPayload();
      if (!depthPayload) {
        return;
      }

      const posDelta = Number.isNaN(this.lastFusedCameraPos.x)
        ? Infinity
        : depthPayload.cameraPos.distanceTo(this.lastFusedCameraPos);
      const rotDelta = Number.isNaN(this.lastFusedCameraQuat.x)
        ? Infinity
        : depthPayload.cameraQuat.angleTo(this.lastFusedCameraQuat);

      if (
        needsInitialSeed ||
        posDelta >= MIN_KEYFRAME_TRANSLATION_METERS ||
        rotDelta >= MIN_KEYFRAME_ROTATION_RAD
      ) {
        if (needsInitialSeed) {
          void this.integrateDepthKeyframe(depthPayload);
        } else {
          void this.integrateRgbMaskKeyframe(depthPayload);
        }
      }
    }
  }

  override dispose(): void {
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }
    for (const [, pending] of this.pendingWorkerRequests) {
      pending.reject(new Error('XRCircleToSearchScript disposed'));
    }
    this.pendingWorkerRequests.clear();
    this.isSegmenting = false;
    this.isIntegratingTsdf = false;
    this.quadTexture.dispose();
    this.atlasTexture.dispose();
    if (this.circleQuad) {
      this.circleQuad.geometry.dispose();
      this.circleQuad.material.dispose();
      this.circleQuad = null;
    }
    if (this.tsdfLiveMesh) {
      this.tsdfLiveMesh.geometry.dispose();
      this.tsdfLiveMesh.material.dispose();
      this.tsdfLiveMesh = null;
    }
    if (this.tsdfBoundingBox) {
      this.tsdfBoundingBox.geometry.dispose();
      this.tsdfBoundingBox.material.dispose();
      this.tsdfBoundingBox = null;
    }
    super.dispose();
  }
}

document.addEventListener('DOMContentLoaded', () => {
  const options = new xb.Options();
  options.enableReticles();
  options.reticles.defaultRenderDistance = MOUSE_QUAD_DISTANCE_METERS;
  options.enableControllers();
  options.controllers.visualizeRays = false;
  options.enableHands();
  options.enableCamera('environment');

  // Enable WebXR / Simulator Depth in cpu-optimized mode for 72 FPS Worker TSDF fusion
  options.enableDepth();
  options.depth.usagePreference = ['cpu-optimized'];
  options.depth.depthMesh.enabled = false;

  options.hands.enabled = true;
  options.hands.visualization = false;
  options.hands.visualizeJoints = false;
  options.hands.visualizeMeshes = false;

  xb.add(new XRCircleToSearchScript());
  void xb.init(options);
});
