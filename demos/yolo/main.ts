import * as THREE from 'three';
import * as xb from 'xrblocks';

const MODEL_IMG_SIZE = 512;
const QUAD_CANVAS_SIZE = 384;
// Expansion factor around the unprojected 3D bounding box so padded white outlines never clip
const QUAD_EXPANSION = 1.65;

// --- Multi-Stage Hysteresis & 3D Tracking Constants ---
const CONF_BIRTH_THRESHOLD = 0.28; // High Schmitt-trigger threshold required to spawn a new 3D track
const CONF_INSTANT_CONFIRM = 0.45; // High-confidence detections confirm immediately on first hit
const MIN_HITS_TO_CONFIRM = 2; // Hits required before a tentative track becomes visible
const MAX_MISSED_IN_FRUSTUM = 4; // Inference cycles a confirmed track coasts in-frustum before fading
const OUT_OF_FRUSTUM_TTL_MS = 15000; // Keep out-of-frustum 3D tracks in world memory for 15s when looking away
const MAX_ACTIVE_TRACKS = 18;
const DEADZONE_POS_METERS = 0.018; // 1.8cm 3D spatial deadzone to suppress depth sensor jitter
const FAST_MOVE_POS_METERS = 0.11; // 11cm threshold where 3D position filter ramps to high gain
const CAMERA_STORAGE_KEY = 'xrblocks:efficientsam:camera';

interface Point2D {
  x: number;
  y: number;
}

interface BoundingBox2D {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

interface ContourProfile {
  mcx: number;
  mcy: number;
  bcx: number;
  bcy: number;
  bw: number;
  bh: number;
  radii: Float32Array;
}

/**
 * Frozen snapshot of the device camera's matrices at a specific capture instant,
 * augmented with inverse matrices (`viewFromWorld`, `clipFromWorld`) and basis vectors
 * so 3D world tracks can be projected back into 2D camera space for ego-motion-compensated matching.
 */
interface ExtendedCameraParams extends xb.CameraParametersSnapshot {
  viewFromWorld: THREE.Matrix4;
  clipFromWorld: THREE.Matrix4;
  cameraOrigin: THREE.Vector3;
  cameraRight: THREE.Vector3;
  cameraUp: THREE.Vector3;
}

interface CandidateDetection3D {
  classId: number;
  className: string;
  score: number;
  box: BoundingBox2D;
  boxArea: number;
  worldCenter: THREE.Vector3;
  worldWidth: number;
  worldHeight: number;
  depthInMeters: number;
}

type TrackLifecycleState = 'tentative' | 'confirmed' | 'coasting' | 'expiring';

/**
 * Pooled 3D billboard quad node with dirty-checked CanvasTexture uploads
 * and zero scene-graph hitMesh draw call overhead.
 */
class DetectionQuadNode extends THREE.Group {
  readonly visualMesh: THREE.Mesh;
  readonly canvas: HTMLCanvasElement;
  readonly ctx: CanvasRenderingContext2D;
  readonly texture: THREE.CanvasTexture;
  inUse = false;
  lastRenderKey = '';

  constructor(sharedGeometry: THREE.PlaneGeometry) {
    super();

    this.canvas = document.createElement('canvas');
    this.canvas.width = QUAD_CANVAS_SIZE;
    this.canvas.height = QUAD_CANVAS_SIZE;
    this.ctx = this.canvas.getContext('2d')!;

    this.texture = new THREE.CanvasTexture(this.canvas);
    this.texture.colorSpace = THREE.SRGBColorSpace;
    this.texture.minFilter = THREE.LinearFilter;
    this.texture.magFilter = THREE.LinearFilter;
    this.texture.generateMipmaps = false;

    const visMat = new THREE.MeshBasicMaterial({
      map: this.texture,
      transparent: true,
      depthWrite: false,
      depthTest: false,
      side: THREE.DoubleSide,
    });
    this.visualMesh = new THREE.Mesh(sharedGeometry, visMat);
    this.visualMesh.raycast = () => {};
    this.visualMesh.renderOrder = 950;
    this.add(this.visualMesh);
    this.visible = false;
  }

  resetForPool(): void {
    this.inUse = false;
    this.visible = false;
    this.lastRenderKey = '';
  }
}

/**
 * Persistent 3D world object track with multi-stage hysteresis,
 * class-vote stabilization, and pooled 3D billboard quad node.
 */
class WorldObjectTrack {
  readonly trackId: number;
  readonly node: DetectionQuadNode;

  classId: number;
  className: string;
  score: number;
  readonly classVotes = new Map<number, {className: string; weight: number}>();

  // Filtered 3D world-space state
  readonly worldCenter = new THREE.Vector3();
  worldWidth: number;
  worldHeight: number;
  depthInMeters: number;

  // Most recent 2D observation in the capture camera frame
  lastCaptureBox: BoundingBox2D;

  // Worker-computed 64-ray contour & compact 32x32 foreground mask (replaces 512x512 binaryMask)
  fgMask32?: Uint8Array;
  maskBox?: BoundingBox2D;
  contourProfile?: ContourProfile;
  samIou?: number;
  samDecoderMs?: number;
  needsSamUpdate = true;

  // Lifecycle & temporal hysteresis state
  state: TrackLifecycleState;
  hits = 1;
  missedFrames = 0;
  lastSeenTimeMs: number;
  visualOpacity = 0;

  constructor(
    trackId: number,
    cand: CandidateDetection3D,
    pooledNode: DetectionQuadNode,
    nowMs: number
  ) {
    this.trackId = trackId;
    this.classId = cand.classId;
    this.className = cand.className;
    this.score = cand.score;
    this.classVotes.set(cand.classId, {
      className: cand.className,
      weight: cand.score,
    });

    this.worldCenter.copy(cand.worldCenter);
    this.worldWidth = cand.worldWidth;
    this.worldHeight = cand.worldHeight;
    this.depthInMeters = cand.depthInMeters;

    this.lastCaptureBox = {...cand.box};

    this.state = cand.score >= CONF_INSTANT_CONFIRM ? 'confirmed' : 'tentative';
    this.lastSeenTimeMs = nowMs;

    this.node = pooledNode;
    this.node.inUse = true;
    this.node.lastRenderKey = '';
    this.node.position.copy(this.worldCenter);
  }

  recordClassVote(classId: number, className: string, score: number): void {
    for (const [cid, entry] of this.classVotes.entries()) {
      entry.weight *= 0.82;
      if (entry.weight < 0.05) {
        this.classVotes.delete(cid);
      }
    }
    const existing = this.classVotes.get(classId);
    if (existing) {
      existing.weight += score;
    } else {
      this.classVotes.set(classId, {className, weight: score});
    }

    let bestId = this.classId;
    let bestName = this.className;
    let bestWeight = -1;
    for (const [cid, entry] of this.classVotes.entries()) {
      if (entry.weight > bestWeight) {
        bestWeight = entry.weight;
        bestId = cid;
        bestName = entry.className;
      }
    }
    this.classId = bestId;
    this.className = bestName;
  }
}

interface WorkerPendingRequest {
  resolve: (value: unknown) => void;
  reject: (reason?: unknown) => void;
}

/**
 * WebXR + Desktop Simulator Semantic Object Tracker with:
 * - Dedicated Web Worker (`perception_worker.js`) for all perception compute:
 *   YOLOE-11s, EfficientSAM-Ti, direct 128x128 contour & 32x32 mask extraction,
 *   and single-pass 48x48 CV template/rotation tracking
 * - Zero-copy `ImageBitmap` frame transfers (no synchronous main-thread video readback)
 * - O(1) parallax-compensated OpenXR depth buffer lookups (no triangle-mesh raycasting)
 * - Pooled 3D `DetectionQuadNode` billboards with dirty-gated `CanvasTexture` uploads
 */
export class YoloSamXRScript extends xb.Script {
  private worker: Worker | null = null;
  private workerReady = false;
  private nextWorkerMsgId = 1;
  private readonly pendingWorkerRequests = new Map<
    number,
    WorkerPendingRequest
  >();

  private lastEncoderMs = 0;
  private lastYoloeMs = 0;

  private nextTrackId = 1;
  private readonly tracks: WorldObjectTrack[] = [];
  private readonly nodePool: DetectionQuadNode[] = [];
  private hoveredTrack: WorldObjectTrack | null = null;
  private lockedTrack: WorldObjectTrack | null = null;

  // Latest CV tracker offsets & angle
  private trackerOffsetX = 0;
  private trackerOffsetY = 0;
  private activeAngle = 0;
  private lastLockedSamTimeMs = 0;

  private readonly captureCanvas: HTMLCanvasElement;
  private readonly captureCtx: CanvasRenderingContext2D;

  private outlineSmoothness = 7;
  private outlinePadding = 17;
  private animatedOutlinePoints: Point2D[] | null = null;

  private isInferenceLoopRunning = false;
  private targetDevice = 'galaxyxr';

  private readonly sharedPlaneGeo = new THREE.PlaneGeometry(1, 1);
  private readonly tempRay = new THREE.Ray();
  private readonly tempOrigin = new THREE.Vector3();
  private readonly tempDir = new THREE.Vector3();
  private readonly tempCenterDir = new THREE.Vector3();
  private readonly tempLeft = new THREE.Vector3();
  private readonly tempRight = new THREE.Vector3();
  private readonly tempTop = new THREE.Vector3();
  private readonly tempBottom = new THREE.Vector3();
  private readonly tempCamPos = new THREE.Vector3();
  private readonly tempTargetPos = new THREE.Vector3();
  private readonly tempViewPt = new THREE.Vector3();
  private readonly tempClipPt = new THREE.Vector3();
  private readonly tempCorner = new THREE.Vector3();
  private readonly tempDepthWorldPt = new THREE.Vector3();
  private readonly tempDepthClipPt = new THREE.Vector3();
  private readonly tempPlaneNormal = new THREE.Vector3();
  private readonly tempPlane = new THREE.Plane();
  private readonly tempHitPt = new THREE.Vector3();
  private readonly tempLocalVec = new THREE.Vector3();

  private statusEl: HTMLElement | null = null;
  private metricObjectEl: HTMLElement | null = null;
  private metricLatencyEl: HTMLElement | null = null;
  private metricDepthEl: HTMLElement | null = null;
  private domCameraBtn: HTMLElement | null = null;

  private hudCard: xb.UICard | null = null;
  private hudStatusText: xb.UIText | null = null;
  private hudCameraBtn: xb.UIButton | null = null;

  constructor() {
    super();

    const urlCamera =
      typeof window !== 'undefined'
        ? new URLSearchParams(window.location.search)
            .get('camera')
            ?.toLowerCase()
        : null;
    let storedCamera: string | null = null;
    try {
      storedCamera =
        localStorage.getItem(CAMERA_STORAGE_KEY)?.toLowerCase() ?? null;
    } catch {
      storedCamera = null;
    }
    const initialCamera = urlCamera || storedCamera;
    if (
      initialCamera === 'aura' ||
      initialCamera === 'galaxyxr' ||
      initialCamera === 'quest3'
    ) {
      this.targetDevice = initialCamera;
      this.persistCameraDevice(initialCamera);
    } else {
      this.targetDevice = xb.detectDeviceCameraTarget();
    }

    this.captureCanvas = document.createElement('canvas');
    this.captureCanvas.width = MODEL_IMG_SIZE;
    this.captureCanvas.height = MODEL_IMG_SIZE;
    this.captureCtx = this.captureCanvas.getContext('2d', {
      willReadFrequently: true,
    })!;

    // Pre-allocate pooled 3D billboard quad nodes to avoid mid-session shader/texture compilation
    for (let i = 0; i < MAX_ACTIVE_TRACKS; i++) {
      const node = new DetectionQuadNode(this.sharedPlaneGeo);
      this.nodePool.push(node);
      this.add(node);
    }
  }

  override init(): void {
    xb.core.depth?.resumeDepth(this);
    this.createSpatialHudCard();
    this.bindHudControls();
    void this.initWorkerAndStartLoop();
  }

  private getCameraButtonLabel(): string {
    if (this.targetDevice === 'aura') return 'Camera: Aura';
    if (this.targetDevice === 'quest3') return 'Camera: Quest 3';
    return 'Camera: Galaxy XR';
  }

  private persistCameraDevice(device: string): void {
    try {
      localStorage.setItem(CAMERA_STORAGE_KEY, device);
    } catch {
      // Ignore storage errors (e.g. private mode / sandboxed iframe).
    }
  }

  /**
   * Switches between Galaxy XR and Aura camera parameters in XR.
   */
  public toggleCameraDevice(): void {
    this.targetDevice = this.targetDevice === 'aura' ? 'galaxyxr' : 'aura';
    this.persistCameraDevice(this.targetDevice);
    const label = this.getCameraButtonLabel();
    if (this.hudCameraBtn) {
      this.hudCameraBtn.label = label;
    }
    if (this.domCameraBtn) {
      this.domCameraBtn.textContent = label;
    }
    this.setStatus(
      `Switched to <strong>${this.targetDevice === 'aura' ? 'Aura' : 'Galaxy XR'}</strong> camera parameters.`
    );
  }

  private createSpatialHudCard(): void {
    const userHeight = xb.user?.height || 1.6;

    this.hudStatusText = new xb.UIText({
      text: 'Loading YOLOE-11s (365 classes) & EfficientSAM-Ti...',
      style: {fontSize: 14},
    });

    const clearBtn = new xb.UIButton({
      label: 'Unlock / Clear',
      icon: 'delete',
      onClick: () => {
        this.clearLockedTrack();
      },
    });

    this.hudCameraBtn = new xb.UIButton({
      label: this.getCameraButtonLabel(),
      icon: 'photo_camera',
      onClick: () => {
        this.toggleCameraDevice();
      },
    });

    const card = new xb.UICard({
      size: {width: 0.54, height: 'auto'},
      manipulation: true,
      edge: true,
      style: {
        flexDirection: 'column',
        gap: 10,
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
            new xb.UIIcon({icon: 'search'}),
            new xb.UIText({
              text: 'XR YOLOE-11s × SAM Tracker',
              style: {fontSize: 17},
            }),
          ],
        }),
        this.hudStatusText,
        new xb.UIPanel({
          style: {
            flexDirection: 'row',
            justifyContent: 'flex-start',
            gap: 8,
          },
          children: [clearBtn, this.hudCameraBtn],
        }),
      ],
    });

    card.position.set(-0.92, userHeight - 0.05, -1.55);
    card.rotation.y = 0.38;
    this.add(card);
    this.hudCard = card;
  }

  private acquirePooledNode(): DetectionQuadNode | null {
    for (const node of this.nodePool) {
      if (!node.inUse) {
        return node;
      }
    }
    return null;
  }

  private bindHudControls(): void {
    this.statusEl = document.getElementById('xr-hud-status');
    this.metricObjectEl = document.getElementById('xr-metric-object');
    this.metricLatencyEl = document.getElementById('xr-metric-latency');
    this.metricDepthEl = document.getElementById('xr-metric-depth');
    this.domCameraBtn = document.getElementById('xr-camera-btn');

    if (this.domCameraBtn) {
      this.domCameraBtn.textContent = this.getCameraButtonLabel();
      this.domCameraBtn.addEventListener('click', (e: MouseEvent) => {
        e.stopPropagation();
        this.toggleCameraDevice();
      });
    }

    const smoothnessSlider = document.getElementById(
      'xr-smoothness-slider'
    ) as HTMLInputElement | null;
    const smoothnessVal = document.getElementById('xr-smoothness-val');
    if (smoothnessSlider) {
      smoothnessSlider.addEventListener('input', () => {
        this.outlineSmoothness = Number(smoothnessSlider.value);
        if (smoothnessVal) {
          smoothnessVal.textContent = `${this.outlineSmoothness}`;
        }
      });
    }

    const paddingSlider = document.getElementById(
      'xr-padding-slider'
    ) as HTMLInputElement | null;
    const paddingVal = document.getElementById('xr-padding-val');
    if (paddingSlider) {
      paddingSlider.addEventListener('input', () => {
        this.outlinePadding = Number(paddingSlider.value);
        if (paddingVal) {
          paddingVal.textContent = `${this.outlinePadding}px`;
        }
      });
    }

    const clearBtn = document.getElementById('xr-clear-btn');
    if (clearBtn) {
      clearBtn.addEventListener('click', () => {
        this.clearLockedTrack();
      });
    }
  }

  private setStatus(html: string): void {
    if (this.statusEl) {
      this.statusEl.innerHTML = html;
    }
    if (this.hudStatusText) {
      this.hudStatusText.text = html
        .replace(/<[^>]*>/g, '')
        .replace(/&amp;/g, '&');
    }
  }

  private callWorker<T>(
    type: string,
    payload: Record<string, unknown> = {},
    transfer: Transferable[] = []
  ): Promise<T> {
    if (!this.worker) {
      return Promise.reject(new Error('Perception worker not initialized.'));
    }
    const id = this.nextWorkerMsgId++;
    return new Promise<T>((resolve, reject) => {
      this.pendingWorkerRequests.set(id, {
        resolve: resolve as (v: unknown) => void,
        reject,
      });
      this.worker!.postMessage({id, type, payload}, transfer);
    });
  }

  private clearLockedTrack(): void {
    this.lockedTrack = null;
    this.trackerOffsetX = 0;
    this.trackerOffsetY = 0;
    this.animatedOutlinePoints = null;
    this.activeAngle = 0;
    if (this.worker && this.workerReady) {
      void this.callWorker('unlock_track').catch(() => {});
    }
    if (this.metricObjectEl) {
      this.metricObjectEl.textContent = 'None';
    }
    this.setStatus(
      `<strong>Unlocked.</strong> Point at any faint YOLOE-11s bounding box and <strong>pinch</strong> to track its white outline curve.`
    );
  }

  private async initWorkerAndStartLoop(): Promise<void> {
    this.setStatus(
      `Loading <strong>YOLOE-11s (365 classes)</strong> &amp; <strong>EfficientSAM-Ti</strong> in Web Worker...`
    );

    try {
      this.worker = new Worker('./build/perception_worker.js', {
        type: 'module',
      });
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
          pending.reject(new Error(error ?? 'Worker error'));
        }
      });

      const initRes = await this.callWorker<{
        yoloeProvider: string;
        encoderAccelerator: string;
        decoderAccelerator: string;
        initMs: number;
      }>('init');

      this.workerReady = true;
      this.setStatus(
        `<strong>Worker Ready</strong> (YOLOE ${initRes.yoloeProvider} · SAM ${initRes.encoderAccelerator.toUpperCase()}). Scanning camera feed...`
      );
      void this.runContinuousDetectionLoop();
    } catch (err) {
      console.error('Failed to initialize perception worker:', err);
      const msg = err instanceof Error ? err.message : String(err);
      this.setStatus(`<strong>Worker Error:</strong> ${msg}`);
    }
  }

  /**
   * Snapshots the current camera matrices and precomputes inverse/projection helpers
   * so detections are unprojected using the exact camera pose from when the frame was captured.
   */
  private captureExtendedCameraParams(): ExtendedCameraParams | null {
    const camera = xb.core.camera as THREE.PerspectiveCamera | undefined;
    const deviceCamera = xb.core.deviceCamera;
    const renderer = xb.core.renderer;
    if (!camera || !deviceCamera || !renderer) return null;

    const xrCameras = renderer.xr.isPresenting ? renderer.xr.getCamera() : null;
    const snap = xb.getCameraParametersSnapshot(
      camera,
      xrCameras,
      deviceCamera,
      this.targetDevice
    );
    if (!snap) return null;

    const clipFromView = snap.clipFromView.clone();
    const viewFromClip = snap.viewFromClip.clone();
    const worldFromView = snap.worldFromView.clone();
    const worldFromClip = snap.worldFromClip.clone();
    const viewFromWorld = worldFromView.clone().invert();
    const clipFromWorld = new THREE.Matrix4().multiplyMatrices(
      clipFromView,
      viewFromWorld
    );

    const cameraOrigin = new THREE.Vector3().setFromMatrixPosition(
      worldFromView
    );
    const cameraRight = new THREE.Vector3()
      .setFromMatrixColumn(worldFromView, 0)
      .normalize();
    const cameraUp = new THREE.Vector3()
      .setFromMatrixColumn(worldFromView, 1)
      .normalize();

    return {
      clipFromView,
      viewFromClip,
      worldFromView,
      worldFromClip,
      viewFromWorld,
      clipFromWorld,
      cameraOrigin,
      cameraRight,
      cameraUp,
    };
  }

  /**
   * Captures the latest camera frame paired with the frozen `ExtendedCameraParams`
   * at the instant of capture. Downsamples to 512x512 RGBA `ArrayBuffer` once per
   * detection cycle so the worker never triggers cross-thread GPU ImageBitmap readbacks.
   */
  private async captureFrameForWorkerWithPose(): Promise<{
    pose: ExtendedCameraParams;
    bitmap?: ImageBitmap;
    rgbaBuffer?: ArrayBuffer;
  } | null> {
    const deviceCamera = xb.core.deviceCamera;
    if (!deviceCamera) return null;

    const poseBefore = this.captureExtendedCameraParams();
    const video = deviceCamera.video;
    if (
      video &&
      video.readyState >= video.HAVE_CURRENT_DATA &&
      video.videoWidth > 0
    ) {
      const vw = video.videoWidth;
      const vh = video.videoHeight;
      const side = Math.min(vw, vh);
      const sx = (vw - side) * 0.5;
      const sy = (vh - side) * 0.5;
      this.captureCtx.drawImage(
        video,
        sx,
        sy,
        side,
        side,
        0,
        0,
        MODEL_IMG_SIZE,
        MODEL_IMG_SIZE
      );
      const imgData = this.captureCtx.getImageData(
        0,
        0,
        MODEL_IMG_SIZE,
        MODEL_IMG_SIZE
      );
      const poseAfter = this.captureExtendedCameraParams() ?? poseBefore;
      if (poseAfter) {
        return {pose: poseAfter, rgbaBuffer: imgData.data.buffer};
      }
    }

    const snapshot = await deviceCamera.captureSnapshot({
      width: MODEL_IMG_SIZE,
      height: MODEL_IMG_SIZE,
      outputFormat: 'imageData',
    });
    const poseAfter = this.captureExtendedCameraParams() ?? poseBefore;
    if (snapshot instanceof ImageData && poseAfter) {
      return {pose: poseAfter, rgbaBuffer: snapshot.data.buffer};
    }
    return null;
  }

  /**
   * Background inference loop that dispatches frames to `perception_worker`
   * and updates persistent 3D tracks on the main thread with O(1) depth lookups.
   */
  private async runContinuousDetectionLoop(): Promise<void> {
    if (this.isInferenceLoopRunning) return;
    this.isInferenceLoopRunning = true;

    while (this.isInferenceLoopRunning) {
      try {
        const frame = await this.captureFrameForWorkerWithPose();
        if (frame && this.workerReady) {
          const captureParams = frame.pose;
          const transfer: Transferable[] = [];
          if (frame.bitmap) transfer.push(frame.bitmap);
          else if (frame.rgbaBuffer) transfer.push(frame.rgbaBuffer);

          const detectRes = await this.callWorker<{
            detections: Array<{
              classId: number;
              className: string;
              score: number;
              box: BoundingBox2D;
              boxArea: number;
            }>;
            yoloeMs: number;
            encoderMs: number;
          }>(
            'detect_frame',
            {
              bitmap: frame.bitmap,
              rgbaBuffer: frame.rgbaBuffer,
            },
            transfer
          );

          this.lastYoloeMs = detectRes.yoloeMs;
          if (detectRes.encoderMs > 0) {
            this.lastEncoderMs = detectRes.encoderMs;
          }

          // Unproject all candidate 2D boxes to 3D world space using O(1) depth lookups
          const candidates3D: CandidateDetection3D[] = [];
          for (const raw of detectRes.detections) {
            const {worldCenter, worldWidth, worldHeight, depthInMeters} =
              this.sampleDetectionDepthAndWorldBox(
                raw.box,
                captureParams,
                undefined,
                this.lockedTrack?.depthInMeters
              );
            candidates3D.push({
              classId: raw.classId,
              className: raw.className,
              score: raw.score,
              box: raw.box,
              boxArea: raw.boxArea,
              worldCenter,
              worldWidth,
              worldHeight,
              depthInMeters,
            });
          }

          const nowMs = performance.now();
          // Match 3D candidates to persistent WorldObjectTracks with multi-stage hysteresis
          const matchedTrackSet = this.updateWorldTracksWithHysteresis(
            candidates3D,
            captureParams,
            nowMs
          );

          // Update 3D poses for all matched tracks using O(1) depth lookups
          for (const trk of matchedTrackSet) {
            const refined = this.sampleDetectionDepthAndWorldBox(
              trk.lastCaptureBox,
              captureParams,
              trk.fgMask32,
              trk.depthInMeters
            );
            this.applyFiltered3DUpdate(trk, refined, 1.0, trk.lastCaptureBox);
          }

          // Only run SAM segmentation for the locked track (unselected tracks only render faint boxes)
          const locked = this.lockedTrack;
          if (
            locked &&
            (!locked.contourProfile ||
              (locked.needsSamUpdate &&
                nowMs - this.lastLockedSamTimeMs >= 2500))
          ) {
            locked.needsSamUpdate = false;
            this.lastLockedSamTimeMs = nowMs;

            const segRes = await this.callWorker<{
              results: Array<{
                trackId: number;
                box: BoundingBox2D;
                contourProfile?: ContourProfile;
                fgMask32: Uint8Array;
                samIou: number;
                decoderMs: number;
                encoderMs: number;
              }>;
              encoderMs: number;
            }>('segment_tracks', {
              requests: [
                {
                  trackId: locked.trackId,
                  box: locked.lastCaptureBox,
                  isLocked: true,
                },
              ],
            });

            if (segRes.encoderMs > 0) {
              this.lastEncoderMs = segRes.encoderMs;
            }

            for (const item of segRes.results) {
              const trk = this.tracks.find((t) => t.trackId === item.trackId);
              if (!trk) continue;

              if (item.contourProfile) {
                if (trk.contourProfile && trk === this.lockedTrack) {
                  const oldRadii = trk.contourProfile.radii;
                  const newRadii = item.contourProfile.radii;
                  const oldSpan = Math.max(
                    8,
                    Math.hypot(trk.contourProfile.bw, trk.contourProfile.bh)
                  );
                  const newSpan = Math.max(
                    8,
                    Math.hypot(item.contourProfile.bw, item.contourProfile.bh)
                  );
                  for (let i = 0; i < newRadii.length; i++) {
                    const oldNorm = oldRadii[i] / oldSpan;
                    const newNorm = newRadii[i] / newSpan;
                    if (newNorm < oldNorm * 0.72) {
                      newRadii[i] = oldNorm * 0.85 * newSpan;
                    }
                  }
                }
                trk.contourProfile = item.contourProfile;
              }
              trk.fgMask32 = item.fgMask32;
              trk.maskBox = {...item.box};
              trk.samIou = item.samIou;
              trk.samDecoderMs = item.decoderMs;

              if (matchedTrackSet.has(trk)) {
                const refined = this.sampleDetectionDepthAndWorldBox(
                  trk.lastCaptureBox,
                  captureParams,
                  trk.fgMask32,
                  trk.depthInMeters
                );
                this.applyFiltered3DUpdate(
                  trk,
                  refined,
                  1.0,
                  trk.lastCaptureBox
                );
              }
            }
          }

          if (this.metricLatencyEl) {
            const samMs = this.lockedTrack?.samDecoderMs ?? 0;
            this.metricLatencyEl.textContent = `${this.lastYoloeMs.toFixed(0)} / ${(this.lastEncoderMs + samMs).toFixed(0)} ms`;
          }

          if (this.lockedTrack && this.metricObjectEl) {
            this.metricObjectEl.textContent = `${this.lockedTrack.className} (${(this.lockedTrack.score * 100).toFixed(0)}%)`;
          } else if (!this.lockedTrack) {
            const count = this.tracks.filter(
              (t) => t.state === 'confirmed' || t.state === 'coasting'
            ).length;
            this.setStatus(
              count > 0
                ? `Found <strong>${count} YOLOE-11s object${count === 1 ? '' : 's'}</strong>. Point &amp; <strong>pinch</strong> any faint box to track its white outline curve.`
                : `Scanning scene with <strong>YOLOE-11s (365 classes)</strong>...`
            );
          }
        }
      } catch (err) {
        console.warn('Detection loop iteration warning:', err);
      }

      await new Promise((resolve) => setTimeout(resolve, 60));
    }
  }

  private computeBoxIou(a: BoundingBox2D, b: BoundingBox2D): number {
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

  /**
   * Projects a 3D world point into 512x512 camera image coordinates `[0, 512]`
   * for a given `ExtendedCameraParams`.
   */
  private projectWorldPointToImage(
    worldPt: THREE.Vector3,
    cameraParams: ExtendedCameraParams
  ): {x: number; y: number; inFront: boolean; inFrustum: boolean} {
    this.tempViewPt.copy(worldPt).applyMatrix4(cameraParams.viewFromWorld);
    if (this.tempViewPt.z >= -0.08) {
      return {x: -1000, y: -1000, inFront: false, inFrustum: false};
    }

    this.tempClipPt.copy(worldPt).applyMatrix4(cameraParams.clipFromWorld);
    const u = 0.5 * (this.tempClipPt.x + 1.0);
    const v = 1.0 - 0.5 * (this.tempClipPt.y + 1.0);
    const inFrustum = u >= 0.03 && u <= 0.97 && v >= 0.03 && v <= 0.97;
    return {
      x: u * MODEL_IMG_SIZE,
      y: v * MODEL_IMG_SIZE,
      inFront: true,
      inFrustum,
    };
  }

  /**
   * Projects a 3D `WorldObjectTrack` into the 2D image space `[0, 512]` of `cameraParams`.
   */
  private projectTrackTo2DBox(
    track: WorldObjectTrack,
    cameraParams: ExtendedCameraParams
  ): {
    box: BoundingBox2D;
    inFront: boolean;
    inFrustum: boolean;
    isBorderClamped: boolean;
  } | null {
    const centerProj = this.projectWorldPointToImage(
      track.worldCenter,
      cameraParams
    );
    if (!centerProj.inFront) return null;

    const halfW = track.worldWidth * 0.5;
    const halfH = track.worldHeight * 0.5;

    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;

    const signs = [
      [-1, -1],
      [1, -1],
      [1, 1],
      [-1, 1],
    ];
    for (const [sx, sy] of signs) {
      this.tempCorner
        .copy(track.worldCenter)
        .addScaledVector(cameraParams.cameraRight, sx * halfW)
        .addScaledVector(cameraParams.cameraUp, sy * halfH);
      const p = this.projectWorldPointToImage(this.tempCorner, cameraParams);
      if (!p.inFront) continue;
      if (p.x < minX) minX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.x > maxX) maxX = p.x;
      if (p.y > maxY) maxY = p.y;
    }

    if (!Number.isFinite(minX) || !Number.isFinite(maxX)) return null;

    const isBorderClamped =
      minX < 8 ||
      minY < 8 ||
      maxX > MODEL_IMG_SIZE - 8 ||
      maxY > MODEL_IMG_SIZE - 8;

    return {
      box: {
        x1: Math.max(0, Math.min(MODEL_IMG_SIZE, minX)),
        y1: Math.max(0, Math.min(MODEL_IMG_SIZE, minY)),
        x2: Math.max(0, Math.min(MODEL_IMG_SIZE, maxX)),
        y2: Math.max(0, Math.min(MODEL_IMG_SIZE, maxY)),
      },
      inFront: true,
      inFrustum: centerProj.inFrustum,
      isBorderClamped,
    };
  }

  /**
   * Computes the unit ray direction in world space for a normalized RGB UV `(u, v)` (`[0, 1]`, top-left origin).
   */
  private getRayDirectionForUv(
    u: number,
    v: number,
    cameraParams: ExtendedCameraParams,
    targetDir: THREE.Vector3
  ): THREE.Vector3 {
    this.tempOrigin.copy(cameraParams.cameraOrigin);
    return targetDir
      .set(2 * u - 1, 2 * (1.0 - v) - 1, -1)
      .applyMatrix4(cameraParams.worldFromClip)
      .sub(this.tempOrigin)
      .normalize();
  }

  /**
   * Unprojects a normalized RGB UV `(u, v)` onto the billboard plane at `centerDepth`
   * perpendicular to `centerRayDir`, eliminating `cos(theta)` spherical shrinkage when close to an object.
   */
  private unprojectUvToBillboardPlane(
    u: number,
    v: number,
    centerDepth: number,
    centerRayDir: THREE.Vector3,
    cameraParams: ExtendedCameraParams,
    target: THREE.Vector3
  ): THREE.Vector3 {
    this.getRayDirectionForUv(u, v, cameraParams, this.tempDir);
    const cosAngle = Math.max(0.35, this.tempDir.dot(centerRayDir));
    const planarRayDist = centerDepth / cosAngle;
    return target
      .copy(cameraParams.cameraOrigin)
      .addScaledVector(this.tempDir, planarRayDist);
  }

  /**
   * O(1) parallax-compensated OpenXR depth lookup for a normalized RGB UV `(su, sv)`.
   * Replaces `xb.transformRgbUvToWorld` triangle-mesh raycasting (`1,226.7ms` in trace -> `< 0.01ms`).
   */
  private sampleDepthFastAtRgbUv(
    su: number,
    sv: number,
    cameraParams: ExtendedCameraParams,
    estDepth: number
  ): number {
    const depth = xb.core.depth;
    if (!depth || !depth.depthArray[0]) return 0;

    if (
      depth.depthViewMatrices.length > 0 &&
      depth.depthProjectionMatrices.length > 0
    ) {
      this.getRayDirectionForUv(su, sv, cameraParams, this.tempDir);
      let d = estDepth;
      for (let iter = 0; iter < 2; iter++) {
        this.tempDepthWorldPt
          .copy(cameraParams.cameraOrigin)
          .addScaledVector(this.tempDir, d);
        this.tempDepthClipPt
          .copy(this.tempDepthWorldPt)
          .applyMatrix4(depth.depthViewMatrices[0])
          .applyMatrix4(depth.depthProjectionMatrices[0]);
        const du = 0.5 * (this.tempDepthClipPt.x + 1.0);
        const dvBottom = 0.5 * (this.tempDepthClipPt.y + 1.0);
        if (du < 0.01 || du > 0.99 || dvBottom < 0.01 || dvBottom > 0.99) {
          break;
        }
        const sampled = depth.getDepth(du, dvBottom);
        if (!Number.isFinite(sampled) || sampled <= 0.1 || sampled >= 10.0) {
          return 0;
        }
        d = sampled;
      }
      return d > 0.1 && d < 10.0 ? d : 0;
    }

    const directD = depth.getDepth(su, 1.0 - sv);
    return Number.isFinite(directD) && directD > 0.1 && directD < 10.0
      ? directD
      : 0;
  }

  /**
   * Multi-point foreground depth sampling using O(1) OpenXR depth buffer lookups
   * filtered by the worker's 32x32 SAM foreground mask (`fgMask32`).
   */
  private sampleDetectionDepthAndWorldBox(
    box: BoundingBox2D,
    cameraParams: ExtendedCameraParams,
    fgMask32?: Uint8Array,
    fallbackDepth?: number
  ): {
    worldCenter: THREE.Vector3;
    worldWidth: number;
    worldHeight: number;
    depthInMeters: number;
  } {
    const u1 = box.x1 / MODEL_IMG_SIZE;
    const v1 = box.y1 / MODEL_IMG_SIZE;
    const u2 = box.x2 / MODEL_IMG_SIZE;
    const v2 = box.y2 / MODEL_IMG_SIZE;
    const uc = (u1 + u2) * 0.5;
    const vc = (v1 + v2) * 0.5;
    const uw = u2 - u1;
    const uh = v2 - v1;

    const sampleOffsets = [
      [0, 0],
      [-0.16, 0],
      [0.16, 0],
      [0, -0.16],
      [0, 0.16],
      [-0.12, -0.12],
      [0.12, -0.12],
      [-0.12, 0.12],
      [0.12, 0.12],
    ];

    const defaultDepth =
      fallbackDepth && Number.isFinite(fallbackDepth) && fallbackDepth > 0.1
        ? fallbackDepth
        : 1.45;

    const validDepths: number[] = [];

    for (const [ox, oy] of sampleOffsets) {
      if (fgMask32 && (ox !== 0 || oy !== 0)) {
        const tx = Math.min(31, Math.max(0, Math.floor((0.5 + ox) * 32)));
        const ty = Math.min(31, Math.max(0, Math.floor((0.5 + oy) * 32)));
        if (fgMask32[ty * 32 + tx] === 0) {
          continue;
        }
      }

      const su = Math.max(0.01, Math.min(0.99, uc + ox * uw));
      const sv = Math.max(0.01, Math.min(0.99, vc + oy * uh));
      const d = this.sampleDepthFastAtRgbUv(su, sv, cameraParams, defaultDepth);
      if (d > 0) {
        validDepths.push(d);
      }
    }

    let finalDepth = defaultDepth;
    if (validDepths.length > 0) {
      validDepths.sort((a, b) => a - b);
      const idx = Math.min(
        validDepths.length - 1,
        Math.floor(validDepths.length * 0.35)
      );
      finalDepth = validDepths[idx];
    }

    this.getRayDirectionForUv(uc, vc, cameraParams, this.tempCenterDir);
    const worldCenter = new THREE.Vector3()
      .copy(cameraParams.cameraOrigin)
      .addScaledVector(this.tempCenterDir, finalDepth);

    this.unprojectUvToBillboardPlane(
      u1,
      vc,
      finalDepth,
      this.tempCenterDir,
      cameraParams,
      this.tempLeft
    );
    this.unprojectUvToBillboardPlane(
      u2,
      vc,
      finalDepth,
      this.tempCenterDir,
      cameraParams,
      this.tempRight
    );
    this.unprojectUvToBillboardPlane(
      uc,
      v1,
      finalDepth,
      this.tempCenterDir,
      cameraParams,
      this.tempTop
    );
    this.unprojectUvToBillboardPlane(
      uc,
      v2,
      finalDepth,
      this.tempCenterDir,
      cameraParams,
      this.tempBottom
    );

    const worldWidth = Math.max(0.04, this.tempLeft.distanceTo(this.tempRight));
    const worldHeight = Math.max(
      0.04,
      this.tempTop.distanceTo(this.tempBottom)
    );

    return {
      worldCenter,
      worldWidth,
      worldHeight,
      depthInMeters: finalDepth,
    };
  }

  /**
   * Applies a 3D spatial deadzone + adaptive-gain filter to update a track's 3D pose
   * while protecting physical 3D dimensions from shrinking when clipped by the camera border.
   */
  private applyFiltered3DUpdate(
    track: WorldObjectTrack,
    measurement: {
      worldCenter: THREE.Vector3;
      worldWidth: number;
      worldHeight: number;
      depthInMeters: number;
    },
    gainMultiplier = 1.0,
    captureBox?: BoundingBox2D
  ): void {
    const delta3D = track.worldCenter.distanceTo(measurement.worldCenter);
    let posAlpha: number;
    if (delta3D <= DEADZONE_POS_METERS) {
      posAlpha = 0.06;
    } else if (delta3D >= FAST_MOVE_POS_METERS) {
      posAlpha = 0.48;
    } else {
      const t =
        (delta3D - DEADZONE_POS_METERS) /
        (FAST_MOVE_POS_METERS - DEADZONE_POS_METERS);
      posAlpha = 0.06 + t * (0.48 - 0.06);
    }
    posAlpha = Math.min(1.0, posAlpha * gainMultiplier);

    const box = captureBox ?? track.lastCaptureBox;
    const isBorderClipped =
      box.x1 <= 16 ||
      box.y1 <= 16 ||
      box.x2 >= MODEL_IMG_SIZE - 16 ||
      box.y2 >= MODEL_IMG_SIZE - 16;

    track.worldCenter.lerp(measurement.worldCenter, posAlpha);
    track.depthInMeters =
      track.depthInMeters * (1 - posAlpha) +
      measurement.depthInMeters * posAlpha;

    let targetW = measurement.worldWidth;
    let targetH = measurement.worldHeight;
    if (track.hits > 2 && isBorderClipped) {
      targetW = Math.max(track.worldWidth, targetW);
      targetH = Math.max(track.worldHeight, targetH);
    } else if (track === this.lockedTrack && track.hits > 2) {
      targetW = Math.max(track.worldWidth * 0.94, targetW);
      targetH = Math.max(track.worldHeight * 0.94, targetH);
    }

    const relW =
      Math.abs(targetW - track.worldWidth) / Math.max(0.04, track.worldWidth);
    const relH =
      Math.abs(targetH - track.worldHeight) / Math.max(0.04, track.worldHeight);
    const sizeAlpha = Math.max(relW, relH) < 0.05 ? 0.06 : 0.24;

    track.worldWidth += (targetW - track.worldWidth) * sizeAlpha;
    track.worldHeight += (targetH - track.worldHeight) * sizeAlpha;
  }

  /**
   * 1-to-1 Global 3D + Ego-Compensated 2D Matching and Multi-Stage Hysteresis.
   */
  private updateWorldTracksWithHysteresis(
    candidates: CandidateDetection3D[],
    captureParams: ExtendedCameraParams,
    nowMs: number
  ): Set<WorldObjectTrack> {
    interface MatchPair {
      trackIdx: number;
      candIdx: number;
      affinity: number;
    }

    const pairs: MatchPair[] = [];
    const matchedTrackObjects = new Set<WorldObjectTrack>();

    for (let tIdx = 0; tIdx < this.tracks.length; tIdx++) {
      const track = this.tracks[tIdx];
      if (track.state === 'expiring') continue;

      const proj = this.projectTrackTo2DBox(track, captureParams);
      const trackDiag3D = Math.hypot(track.worldWidth, track.worldHeight);
      const gate3D = Math.max(0.42, trackDiag3D * 1.75);

      for (let cIdx = 0; cIdx < candidates.length; cIdx++) {
        const cand = candidates[cIdx];
        const dist3D = track.worldCenter.distanceTo(cand.worldCenter);
        const reprojIou = proj ? this.computeBoxIou(proj.box, cand.box) : 0;

        let centerDistNorm2D = 1.0;
        if (proj) {
          const tCx = (proj.box.x1 + proj.box.x2) * 0.5;
          const tCy = (proj.box.y1 + proj.box.y2) * 0.5;
          const cCx = (cand.box.x1 + cand.box.x2) * 0.5;
          const cCy = (cand.box.y1 + cand.box.y2) * 0.5;
          centerDistNorm2D = Math.min(
            1.0,
            Math.hypot(cCx - tCx, cCy - tCy) / 180.0
          );
        }

        const sameClass =
          track.classId === cand.classId || track.className === cand.className;

        if (!sameClass && (reprojIou < 0.5 || dist3D > 0.24)) {
          continue;
        }

        const passes3DGate = dist3D <= gate3D;
        const passes2DGate = reprojIou >= 0.2 || centerDistNorm2D < 0.38;
        if (!passes3DGate && !passes2DGate) continue;
        if (dist3D > gate3D * 1.9 && reprojIou < 0.32) continue;

        const score3D = Math.max(0, 1.0 - dist3D / gate3D);
        const score2D = reprojIou * 0.75 + (1.0 - centerDistNorm2D) * 0.25;
        const classBonus = sameClass ? 4.0 : 0.6;
        const lockedBonus = track === this.lockedTrack ? 1.5 : 0.0;
        const confirmedBonus = track.state === 'confirmed' ? 0.6 : 0.0;

        const affinity =
          classBonus +
          score3D * 4.8 +
          score2D * 5.5 +
          cand.score * 1.5 +
          lockedBonus +
          confirmedBonus;

        pairs.push({trackIdx: tIdx, candIdx: cIdx, affinity});
      }
    }

    pairs.sort((a, b) => b.affinity - a.affinity);
    const matchedTracks = new Set<number>();
    const matchedCands = new Set<number>();

    for (const pair of pairs) {
      if (matchedTracks.has(pair.trackIdx) || matchedCands.has(pair.candIdx)) {
        continue;
      }
      matchedTracks.add(pair.trackIdx);
      matchedCands.add(pair.candIdx);

      const track = this.tracks[pair.trackIdx];
      const cand = candidates[pair.candIdx];
      matchedTrackObjects.add(track);

      track.hits++;
      track.missedFrames = 0;
      track.lastSeenTimeMs = nowMs;
      track.score = track.score * 0.35 + cand.score * 0.65;
      track.recordClassVote(cand.classId, cand.className, cand.score);

      const boxShift = Math.hypot(
        cand.box.x1 - track.lastCaptureBox.x1,
        cand.box.y1 - track.lastCaptureBox.y1
      );
      const sizeShift = Math.hypot(
        cand.box.x2 -
          cand.box.x1 -
          (track.lastCaptureBox.x2 - track.lastCaptureBox.x1),
        cand.box.y2 -
          cand.box.y1 -
          (track.lastCaptureBox.y2 - track.lastCaptureBox.y1)
      );
      if (boxShift > 6 || sizeShift > 8 || !track.contourProfile) {
        track.needsSamUpdate = true;
      }
      track.lastCaptureBox = {...cand.box};

      if (
        track.state === 'tentative' &&
        (track.hits >= MIN_HITS_TO_CONFIRM ||
          cand.score >= CONF_INSTANT_CONFIRM ||
          track === this.lockedTrack)
      ) {
        track.state = 'confirmed';
      } else if (track.state === 'coasting') {
        track.state = 'confirmed';
      }
    }

    for (let tIdx = 0; tIdx < this.tracks.length; tIdx++) {
      if (matchedTracks.has(tIdx)) continue;
      const track = this.tracks[tIdx];
      if (track.state === 'expiring') continue;

      if (track === this.lockedTrack) {
        track.lastSeenTimeMs = nowMs;
        continue;
      }

      const proj = this.projectTrackTo2DBox(track, captureParams);
      const isInsideFrustum = proj ? proj.inFrustum : false;

      if (!isInsideFrustum) {
        if (nowMs - track.lastSeenTimeMs > OUT_OF_FRUSTUM_TTL_MS) {
          track.state = 'expiring';
        }
        continue;
      }

      track.missedFrames++;
      if (track.state === 'tentative' && track.missedFrames >= 2) {
        track.state = 'expiring';
      } else if (track.state === 'confirmed' || track.state === 'coasting') {
        track.state = 'coasting';
        if (track.missedFrames > MAX_MISSED_IN_FRUSTUM) {
          track.state = 'expiring';
        }
      }
    }

    for (let cIdx = 0; cIdx < candidates.length; cIdx++) {
      if (matchedCands.has(cIdx)) continue;
      const cand = candidates[cIdx];
      if (cand.score < CONF_BIRTH_THRESHOLD) continue;
      if (this.tracks.length >= MAX_ACTIVE_TRACKS) break;

      let overlapsExisting = false;
      for (const existing of this.tracks) {
        if (existing.state === 'expiring') continue;
        const d3 = existing.worldCenter.distanceTo(cand.worldCenter);
        const proj = this.projectTrackTo2DBox(existing, captureParams);
        const iou2 = proj ? this.computeBoxIou(proj.box, cand.box) : 0;
        if (
          (existing.classId === cand.classId && (d3 < 0.25 || iou2 > 0.38)) ||
          iou2 > 0.68
        ) {
          overlapsExisting = true;
          break;
        }
      }
      if (overlapsExisting) continue;

      const pooledNode = this.acquirePooledNode();
      if (!pooledNode) break;

      const newTrack = new WorldObjectTrack(
        this.nextTrackId++,
        cand,
        pooledNode,
        nowMs
      );
      this.tracks.push(newTrack);
    }

    return matchedTrackObjects;
  }

  /**
   * Builds normalized contour control points around `profile`.
   * Performs smoothing in Cartesian `(x, y)` space so straight edges stay straight,
   * and applies `this.outlinePadding` via aspect-preserving horizontal/vertical margin
   * expansion around `(mcx, mcy)` so padding never bows edges or distorts the object shape.
   */
  private buildSmoothContourPoints(profile: ContourProfile): Point2D[] {
    const {mcx, mcy, bcx, bcy, bw, bh, radii} = profile;
    const numRays = radii.length;
    const passes = Math.max(0, Math.round(this.outlineSmoothness));
    const boxSpan = Math.hypot(bw, bh);
    const scaleInvariantPad = this.outlinePadding * (boxSpan / 220.0);

    let xs = new Float32Array(numRays);
    let ys = new Float32Array(numRays);
    let hw0 = 4;
    let hh0 = 4;
    for (let i = 0; i < numRays; i++) {
      const angle = (i / numRays) * Math.PI * 2;
      const r = Math.max(2, radii[i]);
      const dx = Math.cos(angle) * r;
      const dy = Math.sin(angle) * r;
      xs[i] = mcx + dx;
      ys[i] = mcy + dy;
      const ax = Math.abs(dx);
      const ay = Math.abs(dy);
      if (ax > hw0) hw0 = ax;
      if (ay > hh0) hh0 = ay;
    }

    for (let pass = 0; pass < passes; pass++) {
      const nextX = new Float32Array(numRays);
      const nextY = new Float32Array(numRays);
      for (let i = 0; i < numRays; i++) {
        const prev = (i - 1 + numRays) % numRays;
        const next = (i + 1) % numRays;
        nextX[i] = 0.25 * xs[prev] + 0.5 * xs[i] + 0.25 * xs[next];
        nextY[i] = 0.25 * ys[prev] + 0.5 * ys[i] + 0.25 * ys[next];
      }
      xs = nextX;
      ys = nextY;
    }

    let hw = 4;
    let hh = 4;
    for (let i = 0; i < numRays; i++) {
      const ax = Math.abs(xs[i] - mcx);
      const ay = Math.abs(ys[i] - mcy);
      if (ax > hw) hw = ax;
      if (ay > hh) hh = ay;
    }

    const scaleX = Math.max(0.1, (hw0 + scaleInvariantPad) / hw);
    const scaleY = Math.max(0.1, (hh0 + scaleInvariantPad) / hh);

    const points: Point2D[] = [];
    for (let i = 0; i < numRays; i++) {
      const px = mcx + (xs[i] - mcx) * scaleX;
      const py = mcy + (ys[i] - mcy) * scaleY;
      points.push({
        x: (px - bcx) / bw,
        y: (py - bcy) / bh,
      });
    }
    return points;
  }

  private getAnimatedOutlinePoints(targetPts: Point2D[]): Point2D[] {
    if (
      !this.animatedOutlinePoints ||
      this.animatedOutlinePoints.length !== targetPts.length
    ) {
      this.animatedOutlinePoints = targetPts.map((p) => ({x: p.x, y: p.y}));
      return this.animatedOutlinePoints;
    }

    const alpha = 0.22;
    const maxStep = 0.08;
    for (let i = 0; i < targetPts.length; i++) {
      const cur = this.animatedOutlinePoints[i];
      const tgt = targetPts[i];
      const dx = tgt.x - cur.x;
      const dy = tgt.y - cur.y;
      cur.x += Math.max(-maxStep, Math.min(maxStep, dx * alpha));
      cur.y += Math.max(-maxStep, Math.min(maxStep, dy * alpha));
    }
    return this.animatedOutlinePoints;
  }

  /**
   * Renders either:
   * - The smooth animated white outline curve (`#ffffff` + soft outer glow) when `track === this.lockedTrack`
   * - Or a dirty-checked faint bounding box (`rgba(255, 255, 255, 0.18)`) for unselected confirmed tracks,
   *   avoiding redundant per-frame `texSubImage2D` uploads when state hasn't changed.
   */
  private drawQuadCanvas(track: WorldObjectTrack): void {
    const node = track.node;
    const ctx = node.ctx;
    const C = QUAD_CANVAS_SIZE;

    if (track.visualOpacity <= 0.01) {
      if (node.lastRenderKey !== 'empty') {
        ctx.clearRect(0, 0, C, C);
        node.texture.needsUpdate = true;
        node.lastRenderKey = 'empty';
      }
      return;
    }

    const maxWorldDim = Math.max(track.worldWidth, track.worldHeight, 0.04);
    const baseBoxSpan = C / QUAD_EXPANSION;
    const sw = baseBoxSpan * (track.worldWidth / maxWorldDim);
    const sh = baseBoxSpan * (track.worldHeight / maxWorldDim);

    const isLocked = track === this.lockedTrack;
    const isHovered = track === this.hoveredTrack;

    if (!isLocked) {
      const opQuant = Math.round(track.visualOpacity * 20);
      const swQuant = Math.round(sw * 0.5);
      const shQuant = Math.round(sh * 0.5);
      const key = `box:${isHovered ? 1 : 0}:${opQuant}:${swQuant}:${shQuant}`;
      if (node.lastRenderKey === key) {
        return;
      }
      node.lastRenderKey = key;
    } else {
      node.lastRenderKey = 'locked';
    }

    ctx.clearRect(0, 0, C, C);
    ctx.save();
    ctx.globalAlpha = track.visualOpacity;
    ctx.translate(C * 0.5, C * 0.5);

    if (isLocked) {
      const padScale = (this.outlinePadding / 220.0) * baseBoxSpan;
      ctx.save();
      ctx.rotate(this.activeAngle);

      if (track.contourProfile) {
        const targetPts = this.buildSmoothContourPoints(track.contourProfile);
        const animPts = this.getAnimatedOutlinePoints(targetPts);
        this.drawSmoothWhiteOutline(ctx, animPts, sw, sh);
      } else {
        const rw = Math.max(12, sw + padScale * 2);
        const rh = Math.max(12, sh + padScale * 2);
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.35)';
        ctx.lineWidth = 6.5;
        ctx.beginPath();
        ctx.roundRect(-rw * 0.5, -rh * 0.5, rw, rh, Math.min(rw, rh) * 0.28);
        ctx.stroke();

        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2.6;
        ctx.beginPath();
        ctx.roundRect(-rw * 0.5, -rh * 0.5, rw, rh, Math.min(rw, rh) * 0.28);
        ctx.stroke();
      }
      ctx.restore();

      this.drawLabelBadge(
        ctx,
        `${track.className} ${(track.score * 100).toFixed(0)}%`,
        0,
        -sh * 0.5 - padScale - 18
      );
    } else {
      ctx.strokeStyle = isHovered
        ? 'rgba(255, 255, 255, 0.42)'
        : 'rgba(255, 255, 255, 0.18)';
      ctx.lineWidth = isHovered ? 1.8 : 1.3;
      ctx.beginPath();
      ctx.roundRect(-sw * 0.5, -sh * 0.5, sw, sh, 8);
      ctx.stroke();
    }

    ctx.restore();
    node.texture.needsUpdate = true;
  }

  private drawSmoothWhiteOutline(
    ctx: CanvasRenderingContext2D,
    pts: Point2D[],
    sw: number,
    sh: number
  ): void {
    if (pts.length < 3) return;
    ctx.save();
    ctx.globalAlpha *= 0.96;
    ctx.beginPath();
    const n = pts.length;
    const p0x = pts[0].x * sw;
    const p0y = pts[0].y * sh;
    const pLastX = pts[n - 1].x * sw;
    const pLastY = pts[n - 1].y * sh;
    ctx.moveTo((pLastX + p0x) * 0.5, (pLastY + p0y) * 0.5);
    for (let i = 0; i < n; i++) {
      const cur = pts[i];
      const next = pts[(i + 1) % n];
      const curX = cur.x * sw;
      const curY = cur.y * sh;
      const midX = (curX + next.x * sw) * 0.5;
      const midY = (curY + next.y * sh) * 0.5;
      ctx.quadraticCurveTo(curX, curY, midX, midY);
    }
    ctx.closePath();

    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.35)';
    ctx.lineWidth = 6.5;
    ctx.stroke();

    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2.6;
    ctx.stroke();
    ctx.restore();
  }

  private drawLabelBadge(
    ctx: CanvasRenderingContext2D,
    text: string,
    cx: number,
    cy: number
  ): void {
    ctx.save();
    ctx.font = '600 13px -apple-system, BlinkMacSystemFont, sans-serif';
    const textW = ctx.measureText(text).width;
    const pillW = textW + 18;
    const pillH = 24;
    const rx = cx - pillW * 0.5;
    const ry = Math.max(-QUAD_CANVAS_SIZE * 0.48, cy - pillH * 0.5);

    ctx.fillStyle = 'rgba(255, 255, 255, 0.95)';
    ctx.beginPath();
    ctx.roundRect(rx, ry, pillW, pillH, 999);
    ctx.fill();

    ctx.fillStyle = '#0f172a';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, cx, ry + pillH * 0.5);
    ctx.restore();
  }

  /**
   * Direct mathematical ray-billboard intersection (zero `THREE.Raycaster` / `hitMesh` overhead).
   */
  private updateHoveredTrackFromRays(
    liveParams: ExtendedCameraParams | null
  ): void {
    const user = xb.core.user;
    if (!user || !liveParams || this.tracks.length === 0) {
      this.hoveredTrack = null;
      return;
    }

    let bestMaskHit: WorldObjectTrack | null = null;
    let bestMaskArea = Infinity;
    let bestBoxHit: WorldObjectTrack | null = null;
    let bestBoxArea = Infinity;

    const numControllers = user.controllers?.length ?? 0;
    for (let c = 0; c < numControllers; c++) {
      this.tempRay.direction.set(0, 0, 0);
      user.getRay(c, this.tempRay);
      if (this.tempRay.direction.lengthSq() < 1e-6) continue;

      for (const trk of this.tracks) {
        if (!trk.node.visible || trk.visualOpacity <= 0.15) continue;

        this.tempPlaneNormal
          .copy(this.tempCamPos)
          .sub(trk.node.position)
          .normalize();
        this.tempPlane.setFromNormalAndCoplanarPoint(
          this.tempPlaneNormal,
          trk.node.position
        );

        const hitPt = this.tempRay.intersectPlane(
          this.tempPlane,
          this.tempHitPt
        );
        if (!hitPt) continue;

        this.tempLocalVec.copy(hitPt).sub(trk.node.position);
        const localX = this.tempLocalVec.dot(liveParams.cameraRight);
        const localY = this.tempLocalVec.dot(liveParams.cameraUp);

        const halfW = trk.worldWidth * 0.5;
        const halfH = trk.worldHeight * 0.5;
        if (Math.abs(localX) > halfW || Math.abs(localY) > halfH) continue;

        const uBox = localX / Math.max(0.02, trk.worldWidth) + 0.5;
        const vBox = 0.5 - localY / Math.max(0.02, trk.worldHeight);
        const tx = Math.min(31, Math.max(0, Math.floor(uBox * 32)));
        const ty = Math.min(31, Math.max(0, Math.floor(vBox * 32)));

        const worldArea = trk.worldWidth * trk.worldHeight;
        if (trk.fgMask32 && trk.fgMask32[ty * 32 + tx] === 1) {
          if (worldArea < bestMaskArea) {
            bestMaskArea = worldArea;
            bestMaskHit = trk;
          }
        }
        if (worldArea < bestBoxArea) {
          bestBoxArea = worldArea;
          bestBoxHit = trk;
        }
      }
    }

    this.hoveredTrack = bestMaskHit ?? bestBoxHit;
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

  /**
   * Pinch (`onSelectStart`) locks onto the hovered `WorldObjectTrack`
   * and initializes the worker's CV template/rotation tracker.
   */
  override onSelectStart(event: xb.SelectEvent): void {
    if (
      this.hudCard &&
      (xb.user?.isPointingAt?.(this.hudCard) ||
        xb.user?.isSelectingAt?.(this.hudCard) ||
        this.isDescendantOf(event?.target, this.hudCard) ||
        this.isDescendantOf(event?.surface, this.hudCard))
    ) {
      return;
    }

    const liveParams = this.captureExtendedCameraParams();
    this.updateHoveredTrackFromRays(liveParams);
    const candidate = this.hoveredTrack;
    if (candidate) {
      this.lockedTrack = candidate;
      candidate.state = 'confirmed';
      candidate.missedFrames = 0;
      this.animatedOutlinePoints = null;
      this.trackerOffsetX = 0;
      this.trackerOffsetY = 0;
      this.activeAngle = 0;

      if (this.workerReady) {
        void this.callWorker<{
          trackId: number;
          contourProfile?: ContourProfile;
          fgMask32: Uint8Array;
          samIou: number;
          encoderMs: number;
          decoderMs: number;
        }>('lock_track', {
          trackId: candidate.trackId,
          box: candidate.maskBox ?? candidate.lastCaptureBox,
          fgMask32: candidate.fgMask32,
          needsSamDecode: !candidate.contourProfile,
        })
          .then((res) => {
            if (this.lockedTrack === candidate) {
              if (res.contourProfile) {
                candidate.contourProfile = res.contourProfile;
                this.lastLockedSamTimeMs = performance.now();
              }
              candidate.fgMask32 = res.fgMask32;
              if (res.encoderMs > 0) {
                this.lastEncoderMs = res.encoderMs;
              }
              if (res.decoderMs > 0) {
                candidate.samIou = res.samIou;
                candidate.samDecoderMs = res.decoderMs;
              }
              if (this.metricLatencyEl) {
                this.metricLatencyEl.textContent = `${this.lastYoloeMs.toFixed(0)}ms + ${this.lastEncoderMs.toFixed(0)}ms + ${candidate.samDecoderMs.toFixed(1)}ms`;
              }
            }
          })
          .catch(() => {});
      }

      if (this.metricObjectEl) {
        this.metricObjectEl.textContent = `${candidate.className} (${(candidate.score * 100).toFixed(0)}%)`;
      }
      this.setStatus(
        `🔒 <strong>Tracking ${candidate.className} (${(candidate.score * 100).toFixed(0)}%)</strong> — Smooth white outline curve locked at OpenXR depth.`
      );
    } else if (this.lockedTrack) {
      this.clearLockedTrack();
    }
  }

  override update(): void {
    const camera = xb.core.camera as THREE.PerspectiveCamera | undefined;
    if (!camera) return;

    const liveParams = this.captureExtendedCameraParams();
    camera.getWorldPosition(this.tempCamPos);

    // Update lifecycle opacity, 3D world pose, and dirty-checked canvas rendering for all persistent tracks
    for (let i = this.tracks.length - 1; i >= 0; i--) {
      const trk = this.tracks[i];
      const isLocked = trk === this.lockedTrack;

      let targetOpacity = 0;
      if (isLocked || trk.state === 'confirmed') {
        targetOpacity = 1.0;
      } else if (trk.state === 'coasting') {
        targetOpacity = 0.6;
      } else if (trk.state === 'tentative' || trk.state === 'expiring') {
        targetOpacity = 0.0;
      }

      const fadeStep = targetOpacity > trk.visualOpacity ? 0.14 : 0.08;
      trk.visualOpacity +=
        Math.sign(targetOpacity - trk.visualOpacity) *
        Math.min(fadeStep, Math.abs(targetOpacity - trk.visualOpacity));

      if (trk.state === 'expiring' && trk.visualOpacity <= 0.01) {
        trk.node.resetForPool();
        this.tracks.splice(i, 1);
        continue;
      }

      const node = trk.node;
      node.visible = trk.visualOpacity > 0.01;
      if (!node.visible) continue;

      this.tempTargetPos.copy(trk.worldCenter);
      if (isLocked && liveParams) {
        const egoProj = this.projectTrackTo2DBox(trk, liveParams);
        const boxW = egoProj
          ? Math.max(24, egoProj.box.x2 - egoProj.box.x1)
          : 160;
        const boxH = egoProj
          ? Math.max(24, egoProj.box.y2 - egoProj.box.y1)
          : 160;
        const worldDx = (this.trackerOffsetX / boxW) * trk.worldWidth;
        const worldDy = -(this.trackerOffsetY / boxH) * trk.worldHeight;
        this.tempTargetPos
          .addScaledVector(liveParams.cameraRight, worldDx)
          .addScaledVector(liveParams.cameraUp, worldDy);
      }

      node.position.lerp(this.tempTargetPos, 0.45);
      node.lookAt(this.tempCamPos);

      const isotropicSpan =
        Math.max(trk.worldWidth, trk.worldHeight) * QUAD_EXPANSION;
      node.visualMesh.scale.set(isotropicSpan, isotropicSpan, 1);

      this.drawQuadCanvas(trk);

      if (isLocked && this.metricDepthEl) {
        this.metricDepthEl.textContent = `${trk.depthInMeters.toFixed(2)} m`;
      }
    }

    this.updateHoveredTrackFromRays(liveParams);

    if (!this.lockedTrack && this.hoveredTrack && this.metricDepthEl) {
      this.metricDepthEl.textContent = `${this.hoveredTrack.depthInMeters.toFixed(2)} m`;
    }
  }
}

const options = new xb.Options();
options.enableCamera();
options.deviceCamera.willCaptureFrequently = false;
options.deviceCamera.videoConstraints = {
  width: {ideal: 1280},
  height: {ideal: 720},
  facingMode: 'environment',
};
options.enableReticles();
options.reticles.projectOnDepthMesh = false;
options.reticles.defaultRenderDistance = 6;
options.enableControllers();
options.enableHands();
options.enableDepth();
options.depth.depthMesh.enabled = false;
options.depth.depthMesh.updateFullResolutionGeometry = false;
options.depth.depthTexture.enabled = false;
options.depth.matchDepthView = false;
options.xrButton.showEnterSimulatorButton = true;
options.simulator.defaultMode = xb.SimulatorMode.USER;
options.setAppTitle('XR YOLOE-11s × SAM Semantic Tracker');

document.addEventListener('DOMContentLoaded', () => {
  xb.add(new YoloSamXRScript());
  void xb.init(options);
});
