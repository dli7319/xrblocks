import * as THREE from "three";
import * as xb from "xrblocks";
//#region demos/yolo/main.ts
const MODEL_IMG_SIZE = 512;
const QUAD_CANVAS_SIZE = 384;
const QUAD_EXPANSION = 1.65;
const CONF_BIRTH_THRESHOLD = .28;
const CONF_INSTANT_CONFIRM = .45;
const MIN_HITS_TO_CONFIRM = 2;
const MAX_MISSED_IN_FRUSTUM = 4;
const OUT_OF_FRUSTUM_TTL_MS = 15e3;
const MAX_ACTIVE_TRACKS = 18;
const DEADZONE_POS_METERS = .018;
const FAST_MOVE_POS_METERS = .11;
const CAMERA_STORAGE_KEY = "xrblocks:efficientsam:camera";
/**
* Pooled 3D billboard quad node with dirty-checked CanvasTexture uploads
* and zero scene-graph hitMesh draw call overhead.
*/
var DetectionQuadNode = class extends THREE.Group {
	visualMesh;
	canvas;
	ctx;
	texture;
	inUse = false;
	lastRenderKey = "";
	constructor(sharedGeometry) {
		super();
		this.canvas = document.createElement("canvas");
		this.canvas.width = QUAD_CANVAS_SIZE;
		this.canvas.height = QUAD_CANVAS_SIZE;
		this.ctx = this.canvas.getContext("2d");
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
			side: THREE.DoubleSide
		});
		this.visualMesh = new THREE.Mesh(sharedGeometry, visMat);
		this.visualMesh.raycast = () => {};
		this.visualMesh.renderOrder = 950;
		this.add(this.visualMesh);
		this.visible = false;
	}
	resetForPool() {
		this.inUse = false;
		this.visible = false;
		this.lastRenderKey = "";
	}
};
/**
* Persistent 3D world object track with multi-stage hysteresis,
* class-vote stabilization, and pooled 3D billboard quad node.
*/
var WorldObjectTrack = class {
	trackId;
	node;
	classId;
	className;
	score;
	classVotes = /* @__PURE__ */ new Map();
	worldCenter = new THREE.Vector3();
	worldWidth;
	worldHeight;
	depthInMeters;
	lastCaptureBox;
	fgMask32;
	maskBox;
	contourProfile;
	samIou;
	samDecoderMs;
	needsSamUpdate = true;
	state;
	hits = 1;
	missedFrames = 0;
	lastSeenTimeMs;
	visualOpacity = 0;
	constructor(trackId, cand, pooledNode, nowMs) {
		this.trackId = trackId;
		this.classId = cand.classId;
		this.className = cand.className;
		this.score = cand.score;
		this.classVotes.set(cand.classId, {
			className: cand.className,
			weight: cand.score
		});
		this.worldCenter.copy(cand.worldCenter);
		this.worldWidth = cand.worldWidth;
		this.worldHeight = cand.worldHeight;
		this.depthInMeters = cand.depthInMeters;
		this.lastCaptureBox = { ...cand.box };
		this.state = cand.score >= CONF_INSTANT_CONFIRM ? "confirmed" : "tentative";
		this.lastSeenTimeMs = nowMs;
		this.node = pooledNode;
		this.node.inUse = true;
		this.node.lastRenderKey = "";
		this.node.position.copy(this.worldCenter);
	}
	recordClassVote(classId, className, score) {
		for (const [cid, entry] of this.classVotes.entries()) {
			entry.weight *= .82;
			if (entry.weight < .05) this.classVotes.delete(cid);
		}
		const existing = this.classVotes.get(classId);
		if (existing) existing.weight += score;
		else this.classVotes.set(classId, {
			className,
			weight: score
		});
		let bestId = this.classId;
		let bestName = this.className;
		let bestWeight = -1;
		for (const [cid, entry] of this.classVotes.entries()) if (entry.weight > bestWeight) {
			bestWeight = entry.weight;
			bestId = cid;
			bestName = entry.className;
		}
		this.classId = bestId;
		this.className = bestName;
	}
};
/**
* WebXR + Desktop Simulator Semantic Object Tracker with:
* - Dedicated Web Worker (`perception_worker.js`) for all perception compute:
*   YOLOE-11s, EfficientSAM-Ti, direct 128x128 contour & 32x32 mask extraction,
*   and single-pass 48x48 CV template/rotation tracking
* - Zero-copy `ImageBitmap` frame transfers (no synchronous main-thread video readback)
* - O(1) parallax-compensated OpenXR depth buffer lookups (no triangle-mesh raycasting)
* - Pooled 3D `DetectionQuadNode` billboards with dirty-gated `CanvasTexture` uploads
*/
var YoloSamXRScript = class extends xb.Script {
	worker = null;
	workerReady = false;
	nextWorkerMsgId = 1;
	pendingWorkerRequests = /* @__PURE__ */ new Map();
	lastEncoderMs = 0;
	lastYoloeMs = 0;
	nextTrackId = 1;
	tracks = [];
	nodePool = [];
	hoveredTrack = null;
	lockedTrack = null;
	trackerOffsetX = 0;
	trackerOffsetY = 0;
	activeAngle = 0;
	lastLockedSamTimeMs = 0;
	captureCanvas;
	captureCtx;
	outlineSmoothness = 7;
	outlinePadding = 17;
	animatedOutlinePoints = null;
	isInferenceLoopRunning = false;
	targetDevice = "galaxyxr";
	sharedPlaneGeo = new THREE.PlaneGeometry(1, 1);
	tempRay = new THREE.Ray();
	tempOrigin = new THREE.Vector3();
	tempDir = new THREE.Vector3();
	tempCenterDir = new THREE.Vector3();
	tempLeft = new THREE.Vector3();
	tempRight = new THREE.Vector3();
	tempTop = new THREE.Vector3();
	tempBottom = new THREE.Vector3();
	tempCamPos = new THREE.Vector3();
	tempTargetPos = new THREE.Vector3();
	tempViewPt = new THREE.Vector3();
	tempClipPt = new THREE.Vector3();
	tempCorner = new THREE.Vector3();
	tempDepthWorldPt = new THREE.Vector3();
	tempDepthClipPt = new THREE.Vector3();
	tempPlaneNormal = new THREE.Vector3();
	tempPlane = new THREE.Plane();
	tempHitPt = new THREE.Vector3();
	tempLocalVec = new THREE.Vector3();
	statusEl = null;
	metricObjectEl = null;
	metricLatencyEl = null;
	metricDepthEl = null;
	domCameraBtn = null;
	hudCard = null;
	hudStatusText = null;
	hudCameraBtn = null;
	constructor() {
		super();
		const urlCamera = typeof window !== "undefined" ? new URLSearchParams(window.location.search).get("camera")?.toLowerCase() : null;
		let storedCamera = null;
		try {
			storedCamera = localStorage.getItem(CAMERA_STORAGE_KEY)?.toLowerCase() ?? null;
		} catch {
			storedCamera = null;
		}
		const initialCamera = urlCamera || storedCamera;
		if (initialCamera === "aura" || initialCamera === "galaxyxr" || initialCamera === "quest3") {
			this.targetDevice = initialCamera;
			this.persistCameraDevice(initialCamera);
		} else this.targetDevice = xb.detectDeviceCameraTarget();
		this.captureCanvas = document.createElement("canvas");
		this.captureCanvas.width = MODEL_IMG_SIZE;
		this.captureCanvas.height = MODEL_IMG_SIZE;
		this.captureCtx = this.captureCanvas.getContext("2d", { willReadFrequently: true });
		for (let i = 0; i < MAX_ACTIVE_TRACKS; i++) {
			const node = new DetectionQuadNode(this.sharedPlaneGeo);
			this.nodePool.push(node);
			this.add(node);
		}
	}
	init() {
		xb.core.depth?.resumeDepth(this);
		this.createSpatialHudCard();
		this.bindHudControls();
		this.initWorkerAndStartLoop();
	}
	getCameraButtonLabel() {
		if (this.targetDevice === "aura") return "Camera: Aura";
		if (this.targetDevice === "quest3") return "Camera: Quest 3";
		return "Camera: Galaxy XR";
	}
	persistCameraDevice(device) {
		try {
			localStorage.setItem(CAMERA_STORAGE_KEY, device);
		} catch {}
	}
	/**
	* Switches between Galaxy XR and Aura camera parameters in XR.
	*/
	toggleCameraDevice() {
		this.targetDevice = this.targetDevice === "aura" ? "galaxyxr" : "aura";
		this.persistCameraDevice(this.targetDevice);
		const label = this.getCameraButtonLabel();
		if (this.hudCameraBtn) this.hudCameraBtn.label = label;
		if (this.domCameraBtn) this.domCameraBtn.textContent = label;
		this.setStatus(`Switched to <strong>${this.targetDevice === "aura" ? "Aura" : "Galaxy XR"}</strong> camera parameters.`);
	}
	createSpatialHudCard() {
		const userHeight = xb.user?.height || 1.6;
		this.hudStatusText = new xb.UIText({
			text: "Loading YOLOE-11s (365 classes) & EfficientSAM-Ti...",
			style: { fontSize: 14 }
		});
		const clearBtn = new xb.UIButton({
			label: "Unlock / Clear",
			icon: "delete",
			onClick: () => {
				this.clearLockedTrack();
			}
		});
		this.hudCameraBtn = new xb.UIButton({
			label: this.getCameraButtonLabel(),
			icon: "photo_camera",
			onClick: () => {
				this.toggleCameraDevice();
			}
		});
		const card = new xb.UICard({
			size: {
				width: .54,
				height: "auto"
			},
			manipulation: true,
			edge: true,
			style: {
				flexDirection: "column",
				gap: 10,
				padding: 16
			},
			children: [
				new xb.UIPanel({
					style: {
						flexDirection: "row",
						alignItems: "center",
						gap: 8
					},
					children: [new xb.UIIcon({ icon: "search" }), new xb.UIText({
						text: "XR YOLOE-11s × SAM Tracker",
						style: { fontSize: 17 }
					})]
				}),
				this.hudStatusText,
				new xb.UIPanel({
					style: {
						flexDirection: "row",
						justifyContent: "flex-start",
						gap: 8
					},
					children: [clearBtn, this.hudCameraBtn]
				})
			]
		});
		card.position.set(-.92, userHeight - .05, -1.55);
		card.rotation.y = .38;
		this.add(card);
		this.hudCard = card;
	}
	acquirePooledNode() {
		for (const node of this.nodePool) if (!node.inUse) return node;
		return null;
	}
	bindHudControls() {
		this.statusEl = document.getElementById("xr-hud-status");
		this.metricObjectEl = document.getElementById("xr-metric-object");
		this.metricLatencyEl = document.getElementById("xr-metric-latency");
		this.metricDepthEl = document.getElementById("xr-metric-depth");
		this.domCameraBtn = document.getElementById("xr-camera-btn");
		if (this.domCameraBtn) {
			this.domCameraBtn.textContent = this.getCameraButtonLabel();
			this.domCameraBtn.addEventListener("click", (e) => {
				e.stopPropagation();
				this.toggleCameraDevice();
			});
		}
		const smoothnessSlider = document.getElementById("xr-smoothness-slider");
		const smoothnessVal = document.getElementById("xr-smoothness-val");
		if (smoothnessSlider) smoothnessSlider.addEventListener("input", () => {
			this.outlineSmoothness = Number(smoothnessSlider.value);
			if (smoothnessVal) smoothnessVal.textContent = `${this.outlineSmoothness}`;
		});
		const paddingSlider = document.getElementById("xr-padding-slider");
		const paddingVal = document.getElementById("xr-padding-val");
		if (paddingSlider) paddingSlider.addEventListener("input", () => {
			this.outlinePadding = Number(paddingSlider.value);
			if (paddingVal) paddingVal.textContent = `${this.outlinePadding}px`;
		});
		const clearBtn = document.getElementById("xr-clear-btn");
		if (clearBtn) clearBtn.addEventListener("click", () => {
			this.clearLockedTrack();
		});
	}
	setStatus(html) {
		if (this.statusEl) this.statusEl.innerHTML = html;
		if (this.hudStatusText) this.hudStatusText.text = html.replace(/<[^>]*>/g, "").replace(/&amp;/g, "&");
	}
	callWorker(type, payload = {}, transfer = []) {
		if (!this.worker) return Promise.reject(/* @__PURE__ */ new Error("Perception worker not initialized."));
		const id = this.nextWorkerMsgId++;
		return new Promise((resolve, reject) => {
			this.pendingWorkerRequests.set(id, {
				resolve,
				reject
			});
			this.worker.postMessage({
				id,
				type,
				payload
			}, transfer);
		});
	}
	clearLockedTrack() {
		this.lockedTrack = null;
		this.trackerOffsetX = 0;
		this.trackerOffsetY = 0;
		this.animatedOutlinePoints = null;
		this.activeAngle = 0;
		if (this.worker && this.workerReady) this.callWorker("unlock_track").catch(() => {});
		if (this.metricObjectEl) this.metricObjectEl.textContent = "None";
		this.setStatus(`<strong>Unlocked.</strong> Point at any faint YOLOE-11s bounding box and <strong>pinch</strong> to track its white outline curve.`);
	}
	async initWorkerAndStartLoop() {
		this.setStatus(`Loading <strong>YOLOE-11s (365 classes)</strong> &amp; <strong>EfficientSAM-Ti</strong> in Web Worker...`);
		try {
			this.worker = new Worker("./build/perception_worker.js", { type: "module" });
			this.worker.addEventListener("message", (event) => {
				const { id, ok, result, error } = event.data;
				const pending = this.pendingWorkerRequests.get(id);
				if (!pending) return;
				this.pendingWorkerRequests.delete(id);
				if (ok) pending.resolve(result);
				else pending.reject(new Error(error ?? "Worker error"));
			});
			const initRes = await this.callWorker("init");
			this.workerReady = true;
			this.setStatus(`<strong>Worker Ready</strong> (YOLOE ${initRes.yoloeProvider} · SAM ${initRes.encoderAccelerator.toUpperCase()}). Scanning camera feed...`);
			this.runContinuousDetectionLoop();
		} catch (err) {
			console.error("Failed to initialize perception worker:", err);
			const msg = err instanceof Error ? err.message : String(err);
			this.setStatus(`<strong>Worker Error:</strong> ${msg}`);
		}
	}
	/**
	* Snapshots the current camera matrices and precomputes inverse/projection helpers
	* so detections are unprojected using the exact camera pose from when the frame was captured.
	*/
	captureExtendedCameraParams() {
		const camera = xb.core.camera;
		const deviceCamera = xb.core.deviceCamera;
		const renderer = xb.core.renderer;
		if (!camera || !deviceCamera || !renderer) return null;
		const xrCameras = renderer.xr.isPresenting ? renderer.xr.getCamera() : null;
		const snap = xb.getCameraParametersSnapshot(camera, xrCameras, deviceCamera, this.targetDevice);
		if (!snap) return null;
		const clipFromView = snap.clipFromView.clone();
		const viewFromClip = snap.viewFromClip.clone();
		const worldFromView = snap.worldFromView.clone();
		const worldFromClip = snap.worldFromClip.clone();
		const viewFromWorld = worldFromView.clone().invert();
		return {
			clipFromView,
			viewFromClip,
			worldFromView,
			worldFromClip,
			viewFromWorld,
			clipFromWorld: new THREE.Matrix4().multiplyMatrices(clipFromView, viewFromWorld),
			cameraOrigin: new THREE.Vector3().setFromMatrixPosition(worldFromView),
			cameraRight: new THREE.Vector3().setFromMatrixColumn(worldFromView, 0).normalize(),
			cameraUp: new THREE.Vector3().setFromMatrixColumn(worldFromView, 1).normalize()
		};
	}
	/**
	* Captures the latest camera frame paired with the frozen `ExtendedCameraParams`
	* at the instant of capture. Downsamples to 512x512 RGBA `ArrayBuffer` once per
	* detection cycle so the worker never triggers cross-thread GPU ImageBitmap readbacks.
	*/
	async captureFrameForWorkerWithPose() {
		const deviceCamera = xb.core.deviceCamera;
		if (!deviceCamera) return null;
		const poseBefore = this.captureExtendedCameraParams();
		const video = deviceCamera.video;
		if (video && video.readyState >= video.HAVE_CURRENT_DATA && video.videoWidth > 0) {
			const vw = video.videoWidth;
			const vh = video.videoHeight;
			const side = Math.min(vw, vh);
			const sx = (vw - side) * .5;
			const sy = (vh - side) * .5;
			this.captureCtx.drawImage(video, sx, sy, side, side, 0, 0, MODEL_IMG_SIZE, MODEL_IMG_SIZE);
			const imgData = this.captureCtx.getImageData(0, 0, MODEL_IMG_SIZE, MODEL_IMG_SIZE);
			const poseAfter = this.captureExtendedCameraParams() ?? poseBefore;
			if (poseAfter) return {
				pose: poseAfter,
				rgbaBuffer: imgData.data.buffer
			};
		}
		const snapshot = await deviceCamera.captureSnapshot({
			width: MODEL_IMG_SIZE,
			height: MODEL_IMG_SIZE,
			outputFormat: "imageData"
		});
		const poseAfter = this.captureExtendedCameraParams() ?? poseBefore;
		if (snapshot instanceof ImageData && poseAfter) return {
			pose: poseAfter,
			rgbaBuffer: snapshot.data.buffer
		};
		return null;
	}
	/**
	* Background inference loop that dispatches frames to `perception_worker`
	* and updates persistent 3D tracks on the main thread with O(1) depth lookups.
	*/
	async runContinuousDetectionLoop() {
		if (this.isInferenceLoopRunning) return;
		this.isInferenceLoopRunning = true;
		while (this.isInferenceLoopRunning) {
			try {
				const frame = await this.captureFrameForWorkerWithPose();
				if (frame && this.workerReady) {
					const captureParams = frame.pose;
					const transfer = [];
					if (frame.bitmap) transfer.push(frame.bitmap);
					else if (frame.rgbaBuffer) transfer.push(frame.rgbaBuffer);
					const detectRes = await this.callWorker("detect_frame", {
						bitmap: frame.bitmap,
						rgbaBuffer: frame.rgbaBuffer
					}, transfer);
					this.lastYoloeMs = detectRes.yoloeMs;
					if (detectRes.encoderMs > 0) this.lastEncoderMs = detectRes.encoderMs;
					const candidates3D = [];
					for (const raw of detectRes.detections) {
						const { worldCenter, worldWidth, worldHeight, depthInMeters } = this.sampleDetectionDepthAndWorldBox(raw.box, captureParams, void 0, this.lockedTrack?.depthInMeters);
						candidates3D.push({
							classId: raw.classId,
							className: raw.className,
							score: raw.score,
							box: raw.box,
							boxArea: raw.boxArea,
							worldCenter,
							worldWidth,
							worldHeight,
							depthInMeters
						});
					}
					const nowMs = performance.now();
					const matchedTrackSet = this.updateWorldTracksWithHysteresis(candidates3D, captureParams, nowMs);
					for (const trk of matchedTrackSet) {
						const refined = this.sampleDetectionDepthAndWorldBox(trk.lastCaptureBox, captureParams, trk.fgMask32, trk.depthInMeters);
						this.applyFiltered3DUpdate(trk, refined, 1, trk.lastCaptureBox);
					}
					const locked = this.lockedTrack;
					if (locked && (!locked.contourProfile || locked.needsSamUpdate && nowMs - this.lastLockedSamTimeMs >= 2500)) {
						locked.needsSamUpdate = false;
						this.lastLockedSamTimeMs = nowMs;
						const segRes = await this.callWorker("segment_tracks", { requests: [{
							trackId: locked.trackId,
							box: locked.lastCaptureBox,
							isLocked: true
						}] });
						if (segRes.encoderMs > 0) this.lastEncoderMs = segRes.encoderMs;
						for (const item of segRes.results) {
							const trk = this.tracks.find((t) => t.trackId === item.trackId);
							if (!trk) continue;
							if (item.contourProfile) {
								if (trk.contourProfile && trk === this.lockedTrack) {
									const oldRadii = trk.contourProfile.radii;
									const newRadii = item.contourProfile.radii;
									const oldSpan = Math.max(8, Math.hypot(trk.contourProfile.bw, trk.contourProfile.bh));
									const newSpan = Math.max(8, Math.hypot(item.contourProfile.bw, item.contourProfile.bh));
									for (let i = 0; i < newRadii.length; i++) {
										const oldNorm = oldRadii[i] / oldSpan;
										if (newRadii[i] / newSpan < oldNorm * .72) newRadii[i] = oldNorm * .85 * newSpan;
									}
								}
								trk.contourProfile = item.contourProfile;
							}
							trk.fgMask32 = item.fgMask32;
							trk.maskBox = { ...item.box };
							trk.samIou = item.samIou;
							trk.samDecoderMs = item.decoderMs;
							if (matchedTrackSet.has(trk)) {
								const refined = this.sampleDetectionDepthAndWorldBox(trk.lastCaptureBox, captureParams, trk.fgMask32, trk.depthInMeters);
								this.applyFiltered3DUpdate(trk, refined, 1, trk.lastCaptureBox);
							}
						}
					}
					if (this.metricLatencyEl) {
						const samMs = this.lockedTrack?.samDecoderMs ?? 0;
						this.metricLatencyEl.textContent = `${this.lastYoloeMs.toFixed(0)} / ${(this.lastEncoderMs + samMs).toFixed(0)} ms`;
					}
					if (this.lockedTrack && this.metricObjectEl) this.metricObjectEl.textContent = `${this.lockedTrack.className} (${(this.lockedTrack.score * 100).toFixed(0)}%)`;
					else if (!this.lockedTrack) {
						const count = this.tracks.filter((t) => t.state === "confirmed" || t.state === "coasting").length;
						this.setStatus(count > 0 ? `Found <strong>${count} YOLOE-11s object${count === 1 ? "" : "s"}</strong>. Point &amp; <strong>pinch</strong> any faint box to track its white outline curve.` : `Scanning scene with <strong>YOLOE-11s (365 classes)</strong>...`);
					}
				}
			} catch (err) {
				console.warn("Detection loop iteration warning:", err);
			}
			await new Promise((resolve) => setTimeout(resolve, 60));
		}
	}
	computeBoxIou(a, b) {
		const ix1 = Math.max(a.x1, b.x1);
		const iy1 = Math.max(a.y1, b.y1);
		const ix2 = Math.min(a.x2, b.x2);
		const iy2 = Math.min(a.y2, b.y2);
		const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
		if (inter <= 0) return 0;
		const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
		const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
		return inter / Math.max(1e-5, areaA + areaB - inter);
	}
	/**
	* Projects a 3D world point into 512x512 camera image coordinates `[0, 512]`
	* for a given `ExtendedCameraParams`.
	*/
	projectWorldPointToImage(worldPt, cameraParams) {
		this.tempViewPt.copy(worldPt).applyMatrix4(cameraParams.viewFromWorld);
		if (this.tempViewPt.z >= -.08) return {
			x: -1e3,
			y: -1e3,
			inFront: false,
			inFrustum: false
		};
		this.tempClipPt.copy(worldPt).applyMatrix4(cameraParams.clipFromWorld);
		const u = .5 * (this.tempClipPt.x + 1);
		const v = 1 - .5 * (this.tempClipPt.y + 1);
		const inFrustum = u >= .03 && u <= .97 && v >= .03 && v <= .97;
		return {
			x: u * MODEL_IMG_SIZE,
			y: v * MODEL_IMG_SIZE,
			inFront: true,
			inFrustum
		};
	}
	/**
	* Projects a 3D `WorldObjectTrack` into the 2D image space `[0, 512]` of `cameraParams`.
	*/
	projectTrackTo2DBox(track, cameraParams) {
		const centerProj = this.projectWorldPointToImage(track.worldCenter, cameraParams);
		if (!centerProj.inFront) return null;
		const halfW = track.worldWidth * .5;
		const halfH = track.worldHeight * .5;
		let minX = Infinity;
		let minY = Infinity;
		let maxX = -Infinity;
		let maxY = -Infinity;
		for (const [sx, sy] of [
			[-1, -1],
			[1, -1],
			[1, 1],
			[-1, 1]
		]) {
			this.tempCorner.copy(track.worldCenter).addScaledVector(cameraParams.cameraRight, sx * halfW).addScaledVector(cameraParams.cameraUp, sy * halfH);
			const p = this.projectWorldPointToImage(this.tempCorner, cameraParams);
			if (!p.inFront) continue;
			if (p.x < minX) minX = p.x;
			if (p.y < minY) minY = p.y;
			if (p.x > maxX) maxX = p.x;
			if (p.y > maxY) maxY = p.y;
		}
		if (!Number.isFinite(minX) || !Number.isFinite(maxX)) return null;
		const isBorderClamped = minX < 8 || minY < 8 || maxX > 504 || maxY > 504;
		return {
			box: {
				x1: Math.max(0, Math.min(MODEL_IMG_SIZE, minX)),
				y1: Math.max(0, Math.min(MODEL_IMG_SIZE, minY)),
				x2: Math.max(0, Math.min(MODEL_IMG_SIZE, maxX)),
				y2: Math.max(0, Math.min(MODEL_IMG_SIZE, maxY))
			},
			inFront: true,
			inFrustum: centerProj.inFrustum,
			isBorderClamped
		};
	}
	/**
	* Computes the unit ray direction in world space for a normalized RGB UV `(u, v)` (`[0, 1]`, top-left origin).
	*/
	getRayDirectionForUv(u, v, cameraParams, targetDir) {
		this.tempOrigin.copy(cameraParams.cameraOrigin);
		return targetDir.set(2 * u - 1, 2 * (1 - v) - 1, -1).applyMatrix4(cameraParams.worldFromClip).sub(this.tempOrigin).normalize();
	}
	/**
	* Unprojects a normalized RGB UV `(u, v)` onto the billboard plane at `centerDepth`
	* perpendicular to `centerRayDir`, eliminating `cos(theta)` spherical shrinkage when close to an object.
	*/
	unprojectUvToBillboardPlane(u, v, centerDepth, centerRayDir, cameraParams, target) {
		this.getRayDirectionForUv(u, v, cameraParams, this.tempDir);
		const planarRayDist = centerDepth / Math.max(.35, this.tempDir.dot(centerRayDir));
		return target.copy(cameraParams.cameraOrigin).addScaledVector(this.tempDir, planarRayDist);
	}
	/**
	* O(1) parallax-compensated OpenXR depth lookup for a normalized RGB UV `(su, sv)`.
	* Replaces `xb.transformRgbUvToWorld` triangle-mesh raycasting (`1,226.7ms` in trace -> `< 0.01ms`).
	*/
	sampleDepthFastAtRgbUv(su, sv, cameraParams, estDepth) {
		const depth = xb.core.depth;
		if (!depth || !depth.depthArray[0]) return 0;
		if (depth.depthViewMatrices.length > 0 && depth.depthProjectionMatrices.length > 0) {
			this.getRayDirectionForUv(su, sv, cameraParams, this.tempDir);
			let d = estDepth;
			for (let iter = 0; iter < 2; iter++) {
				this.tempDepthWorldPt.copy(cameraParams.cameraOrigin).addScaledVector(this.tempDir, d);
				this.tempDepthClipPt.copy(this.tempDepthWorldPt).applyMatrix4(depth.depthViewMatrices[0]).applyMatrix4(depth.depthProjectionMatrices[0]);
				const du = .5 * (this.tempDepthClipPt.x + 1);
				const dvBottom = .5 * (this.tempDepthClipPt.y + 1);
				if (du < .01 || du > .99 || dvBottom < .01 || dvBottom > .99) break;
				const sampled = depth.getDepth(du, dvBottom);
				if (!Number.isFinite(sampled) || sampled <= .1 || sampled >= 10) return 0;
				d = sampled;
			}
			return d > .1 && d < 10 ? d : 0;
		}
		const directD = depth.getDepth(su, 1 - sv);
		return Number.isFinite(directD) && directD > .1 && directD < 10 ? directD : 0;
	}
	/**
	* Multi-point foreground depth sampling using O(1) OpenXR depth buffer lookups
	* filtered by the worker's 32x32 SAM foreground mask (`fgMask32`).
	*/
	sampleDetectionDepthAndWorldBox(box, cameraParams, fgMask32, fallbackDepth) {
		const u1 = box.x1 / MODEL_IMG_SIZE;
		const v1 = box.y1 / MODEL_IMG_SIZE;
		const u2 = box.x2 / MODEL_IMG_SIZE;
		const v2 = box.y2 / MODEL_IMG_SIZE;
		const uc = (u1 + u2) * .5;
		const vc = (v1 + v2) * .5;
		const uw = u2 - u1;
		const uh = v2 - v1;
		const sampleOffsets = [
			[0, 0],
			[-.16, 0],
			[.16, 0],
			[0, -.16],
			[0, .16],
			[-.12, -.12],
			[.12, -.12],
			[-.12, .12],
			[.12, .12]
		];
		const defaultDepth = fallbackDepth && Number.isFinite(fallbackDepth) && fallbackDepth > .1 ? fallbackDepth : 1.45;
		const validDepths = [];
		for (const [ox, oy] of sampleOffsets) {
			if (fgMask32 && (ox !== 0 || oy !== 0)) {
				const tx = Math.min(31, Math.max(0, Math.floor((.5 + ox) * 32)));
				if (fgMask32[Math.min(31, Math.max(0, Math.floor((.5 + oy) * 32))) * 32 + tx] === 0) continue;
			}
			const su = Math.max(.01, Math.min(.99, uc + ox * uw));
			const sv = Math.max(.01, Math.min(.99, vc + oy * uh));
			const d = this.sampleDepthFastAtRgbUv(su, sv, cameraParams, defaultDepth);
			if (d > 0) validDepths.push(d);
		}
		let finalDepth = defaultDepth;
		if (validDepths.length > 0) {
			validDepths.sort((a, b) => a - b);
			finalDepth = validDepths[Math.min(validDepths.length - 1, Math.floor(validDepths.length * .35))];
		}
		this.getRayDirectionForUv(uc, vc, cameraParams, this.tempCenterDir);
		const worldCenter = new THREE.Vector3().copy(cameraParams.cameraOrigin).addScaledVector(this.tempCenterDir, finalDepth);
		this.unprojectUvToBillboardPlane(u1, vc, finalDepth, this.tempCenterDir, cameraParams, this.tempLeft);
		this.unprojectUvToBillboardPlane(u2, vc, finalDepth, this.tempCenterDir, cameraParams, this.tempRight);
		this.unprojectUvToBillboardPlane(uc, v1, finalDepth, this.tempCenterDir, cameraParams, this.tempTop);
		this.unprojectUvToBillboardPlane(uc, v2, finalDepth, this.tempCenterDir, cameraParams, this.tempBottom);
		return {
			worldCenter,
			worldWidth: Math.max(.04, this.tempLeft.distanceTo(this.tempRight)),
			worldHeight: Math.max(.04, this.tempTop.distanceTo(this.tempBottom)),
			depthInMeters: finalDepth
		};
	}
	/**
	* Applies a 3D spatial deadzone + adaptive-gain filter to update a track's 3D pose
	* while protecting physical 3D dimensions from shrinking when clipped by the camera border.
	*/
	applyFiltered3DUpdate(track, measurement, gainMultiplier = 1, captureBox) {
		const delta3D = track.worldCenter.distanceTo(measurement.worldCenter);
		let posAlpha;
		if (delta3D <= DEADZONE_POS_METERS) posAlpha = .06;
		else if (delta3D >= FAST_MOVE_POS_METERS) posAlpha = .48;
		else posAlpha = .06 + (delta3D - DEADZONE_POS_METERS) / .092 * .42;
		posAlpha = Math.min(1, posAlpha * gainMultiplier);
		const box = captureBox ?? track.lastCaptureBox;
		const isBorderClipped = box.x1 <= 16 || box.y1 <= 16 || box.x2 >= 496 || box.y2 >= 496;
		track.worldCenter.lerp(measurement.worldCenter, posAlpha);
		track.depthInMeters = track.depthInMeters * (1 - posAlpha) + measurement.depthInMeters * posAlpha;
		let targetW = measurement.worldWidth;
		let targetH = measurement.worldHeight;
		if (track.hits > 2 && isBorderClipped) {
			targetW = Math.max(track.worldWidth, targetW);
			targetH = Math.max(track.worldHeight, targetH);
		} else if (track === this.lockedTrack && track.hits > 2) {
			targetW = Math.max(track.worldWidth * .94, targetW);
			targetH = Math.max(track.worldHeight * .94, targetH);
		}
		const relW = Math.abs(targetW - track.worldWidth) / Math.max(.04, track.worldWidth);
		const relH = Math.abs(targetH - track.worldHeight) / Math.max(.04, track.worldHeight);
		const sizeAlpha = Math.max(relW, relH) < .05 ? .06 : .24;
		track.worldWidth += (targetW - track.worldWidth) * sizeAlpha;
		track.worldHeight += (targetH - track.worldHeight) * sizeAlpha;
	}
	/**
	* 1-to-1 Global 3D + Ego-Compensated 2D Matching and Multi-Stage Hysteresis.
	*/
	updateWorldTracksWithHysteresis(candidates, captureParams, nowMs) {
		const pairs = [];
		const matchedTrackObjects = /* @__PURE__ */ new Set();
		for (let tIdx = 0; tIdx < this.tracks.length; tIdx++) {
			const track = this.tracks[tIdx];
			if (track.state === "expiring") continue;
			const proj = this.projectTrackTo2DBox(track, captureParams);
			const trackDiag3D = Math.hypot(track.worldWidth, track.worldHeight);
			const gate3D = Math.max(.42, trackDiag3D * 1.75);
			for (let cIdx = 0; cIdx < candidates.length; cIdx++) {
				const cand = candidates[cIdx];
				const dist3D = track.worldCenter.distanceTo(cand.worldCenter);
				const reprojIou = proj ? this.computeBoxIou(proj.box, cand.box) : 0;
				let centerDistNorm2D = 1;
				if (proj) {
					const tCx = (proj.box.x1 + proj.box.x2) * .5;
					const tCy = (proj.box.y1 + proj.box.y2) * .5;
					const cCx = (cand.box.x1 + cand.box.x2) * .5;
					const cCy = (cand.box.y1 + cand.box.y2) * .5;
					centerDistNorm2D = Math.min(1, Math.hypot(cCx - tCx, cCy - tCy) / 180);
				}
				const sameClass = track.classId === cand.classId || track.className === cand.className;
				if (!sameClass && (reprojIou < .5 || dist3D > .24)) continue;
				if (!(dist3D <= gate3D) && !(reprojIou >= .2 || centerDistNorm2D < .38)) continue;
				if (dist3D > gate3D * 1.9 && reprojIou < .32) continue;
				const score3D = Math.max(0, 1 - dist3D / gate3D);
				const score2D = reprojIou * .75 + (1 - centerDistNorm2D) * .25;
				const classBonus = sameClass ? 4 : .6;
				const lockedBonus = track === this.lockedTrack ? 1.5 : 0;
				const confirmedBonus = track.state === "confirmed" ? .6 : 0;
				const affinity = classBonus + score3D * 4.8 + score2D * 5.5 + cand.score * 1.5 + lockedBonus + confirmedBonus;
				pairs.push({
					trackIdx: tIdx,
					candIdx: cIdx,
					affinity
				});
			}
		}
		pairs.sort((a, b) => b.affinity - a.affinity);
		const matchedTracks = /* @__PURE__ */ new Set();
		const matchedCands = /* @__PURE__ */ new Set();
		for (const pair of pairs) {
			if (matchedTracks.has(pair.trackIdx) || matchedCands.has(pair.candIdx)) continue;
			matchedTracks.add(pair.trackIdx);
			matchedCands.add(pair.candIdx);
			const track = this.tracks[pair.trackIdx];
			const cand = candidates[pair.candIdx];
			matchedTrackObjects.add(track);
			track.hits++;
			track.missedFrames = 0;
			track.lastSeenTimeMs = nowMs;
			track.score = track.score * .35 + cand.score * .65;
			track.recordClassVote(cand.classId, cand.className, cand.score);
			const boxShift = Math.hypot(cand.box.x1 - track.lastCaptureBox.x1, cand.box.y1 - track.lastCaptureBox.y1);
			const sizeShift = Math.hypot(cand.box.x2 - cand.box.x1 - (track.lastCaptureBox.x2 - track.lastCaptureBox.x1), cand.box.y2 - cand.box.y1 - (track.lastCaptureBox.y2 - track.lastCaptureBox.y1));
			if (boxShift > 6 || sizeShift > 8 || !track.contourProfile) track.needsSamUpdate = true;
			track.lastCaptureBox = { ...cand.box };
			if (track.state === "tentative" && (track.hits >= MIN_HITS_TO_CONFIRM || cand.score >= CONF_INSTANT_CONFIRM || track === this.lockedTrack)) track.state = "confirmed";
			else if (track.state === "coasting") track.state = "confirmed";
		}
		for (let tIdx = 0; tIdx < this.tracks.length; tIdx++) {
			if (matchedTracks.has(tIdx)) continue;
			const track = this.tracks[tIdx];
			if (track.state === "expiring") continue;
			if (track === this.lockedTrack) {
				track.lastSeenTimeMs = nowMs;
				continue;
			}
			const proj = this.projectTrackTo2DBox(track, captureParams);
			if (!(proj ? proj.inFrustum : false)) {
				if (nowMs - track.lastSeenTimeMs > OUT_OF_FRUSTUM_TTL_MS) track.state = "expiring";
				continue;
			}
			track.missedFrames++;
			if (track.state === "tentative" && track.missedFrames >= 2) track.state = "expiring";
			else if (track.state === "confirmed" || track.state === "coasting") {
				track.state = "coasting";
				if (track.missedFrames > MAX_MISSED_IN_FRUSTUM) track.state = "expiring";
			}
		}
		for (let cIdx = 0; cIdx < candidates.length; cIdx++) {
			if (matchedCands.has(cIdx)) continue;
			const cand = candidates[cIdx];
			if (cand.score < CONF_BIRTH_THRESHOLD) continue;
			if (this.tracks.length >= MAX_ACTIVE_TRACKS) break;
			let overlapsExisting = false;
			for (const existing of this.tracks) {
				if (existing.state === "expiring") continue;
				const d3 = existing.worldCenter.distanceTo(cand.worldCenter);
				const proj = this.projectTrackTo2DBox(existing, captureParams);
				const iou2 = proj ? this.computeBoxIou(proj.box, cand.box) : 0;
				if (existing.classId === cand.classId && (d3 < .25 || iou2 > .38) || iou2 > .68) {
					overlapsExisting = true;
					break;
				}
			}
			if (overlapsExisting) continue;
			const pooledNode = this.acquirePooledNode();
			if (!pooledNode) break;
			const newTrack = new WorldObjectTrack(this.nextTrackId++, cand, pooledNode, nowMs);
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
	buildSmoothContourPoints(profile) {
		const { mcx, mcy, bcx, bcy, bw, bh, radii } = profile;
		const numRays = radii.length;
		const passes = Math.max(0, Math.round(this.outlineSmoothness));
		const boxSpan = Math.hypot(bw, bh);
		const scaleInvariantPad = this.outlinePadding * (boxSpan / 220);
		let xs = new Float32Array(numRays);
		let ys = new Float32Array(numRays);
		let hw0 = 4;
		let hh0 = 4;
		for (let i = 0; i < numRays; i++) {
			const angle = i / numRays * Math.PI * 2;
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
				nextX[i] = .25 * xs[prev] + .5 * xs[i] + .25 * xs[next];
				nextY[i] = .25 * ys[prev] + .5 * ys[i] + .25 * ys[next];
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
		const scaleX = Math.max(.1, (hw0 + scaleInvariantPad) / hw);
		const scaleY = Math.max(.1, (hh0 + scaleInvariantPad) / hh);
		const points = [];
		for (let i = 0; i < numRays; i++) {
			const px = mcx + (xs[i] - mcx) * scaleX;
			const py = mcy + (ys[i] - mcy) * scaleY;
			points.push({
				x: (px - bcx) / bw,
				y: (py - bcy) / bh
			});
		}
		return points;
	}
	getAnimatedOutlinePoints(targetPts) {
		if (!this.animatedOutlinePoints || this.animatedOutlinePoints.length !== targetPts.length) {
			this.animatedOutlinePoints = targetPts.map((p) => ({
				x: p.x,
				y: p.y
			}));
			return this.animatedOutlinePoints;
		}
		const alpha = .22;
		const maxStep = .08;
		for (let i = 0; i < targetPts.length; i++) {
			const cur = this.animatedOutlinePoints[i];
			const tgt = targetPts[i];
			const dx = tgt.x - cur.x;
			const dy = tgt.y - cur.y;
			cur.x += Math.max(-.08, Math.min(maxStep, dx * alpha));
			cur.y += Math.max(-.08, Math.min(maxStep, dy * alpha));
		}
		return this.animatedOutlinePoints;
	}
	/**
	* Renders either:
	* - The smooth animated white outline curve (`#ffffff` + soft outer glow) when `track === this.lockedTrack`
	* - Or a dirty-checked faint bounding box (`rgba(255, 255, 255, 0.18)`) for unselected confirmed tracks,
	*   avoiding redundant per-frame `texSubImage2D` uploads when state hasn't changed.
	*/
	drawQuadCanvas(track) {
		const node = track.node;
		const ctx = node.ctx;
		const C = QUAD_CANVAS_SIZE;
		if (track.visualOpacity <= .01) {
			if (node.lastRenderKey !== "empty") {
				ctx.clearRect(0, 0, C, C);
				node.texture.needsUpdate = true;
				node.lastRenderKey = "empty";
			}
			return;
		}
		const maxWorldDim = Math.max(track.worldWidth, track.worldHeight, .04);
		const baseBoxSpan = C / QUAD_EXPANSION;
		const sw = baseBoxSpan * (track.worldWidth / maxWorldDim);
		const sh = baseBoxSpan * (track.worldHeight / maxWorldDim);
		const isLocked = track === this.lockedTrack;
		const isHovered = track === this.hoveredTrack;
		if (!isLocked) {
			const opQuant = Math.round(track.visualOpacity * 20);
			const swQuant = Math.round(sw * .5);
			const shQuant = Math.round(sh * .5);
			const key = `box:${isHovered ? 1 : 0}:${opQuant}:${swQuant}:${shQuant}`;
			if (node.lastRenderKey === key) return;
			node.lastRenderKey = key;
		} else node.lastRenderKey = "locked";
		ctx.clearRect(0, 0, C, C);
		ctx.save();
		ctx.globalAlpha = track.visualOpacity;
		ctx.translate(C * .5, C * .5);
		if (isLocked) {
			const padScale = this.outlinePadding / 220 * baseBoxSpan;
			ctx.save();
			ctx.rotate(this.activeAngle);
			if (track.contourProfile) {
				const targetPts = this.buildSmoothContourPoints(track.contourProfile);
				const animPts = this.getAnimatedOutlinePoints(targetPts);
				this.drawSmoothWhiteOutline(ctx, animPts, sw, sh);
			} else {
				const rw = Math.max(12, sw + padScale * 2);
				const rh = Math.max(12, sh + padScale * 2);
				ctx.strokeStyle = "rgba(255, 255, 255, 0.35)";
				ctx.lineWidth = 6.5;
				ctx.beginPath();
				ctx.roundRect(-rw * .5, -rh * .5, rw, rh, Math.min(rw, rh) * .28);
				ctx.stroke();
				ctx.strokeStyle = "#ffffff";
				ctx.lineWidth = 2.6;
				ctx.beginPath();
				ctx.roundRect(-rw * .5, -rh * .5, rw, rh, Math.min(rw, rh) * .28);
				ctx.stroke();
			}
			ctx.restore();
			this.drawLabelBadge(ctx, `${track.className} ${(track.score * 100).toFixed(0)}%`, 0, -sh * .5 - padScale - 18);
		} else {
			ctx.strokeStyle = isHovered ? "rgba(255, 255, 255, 0.42)" : "rgba(255, 255, 255, 0.18)";
			ctx.lineWidth = isHovered ? 1.8 : 1.3;
			ctx.beginPath();
			ctx.roundRect(-sw * .5, -sh * .5, sw, sh, 8);
			ctx.stroke();
		}
		ctx.restore();
		node.texture.needsUpdate = true;
	}
	drawSmoothWhiteOutline(ctx, pts, sw, sh) {
		if (pts.length < 3) return;
		ctx.save();
		ctx.globalAlpha *= .96;
		ctx.beginPath();
		const n = pts.length;
		const p0x = pts[0].x * sw;
		const p0y = pts[0].y * sh;
		const pLastX = pts[n - 1].x * sw;
		const pLastY = pts[n - 1].y * sh;
		ctx.moveTo((pLastX + p0x) * .5, (pLastY + p0y) * .5);
		for (let i = 0; i < n; i++) {
			const cur = pts[i];
			const next = pts[(i + 1) % n];
			const curX = cur.x * sw;
			const curY = cur.y * sh;
			const midX = (curX + next.x * sw) * .5;
			const midY = (curY + next.y * sh) * .5;
			ctx.quadraticCurveTo(curX, curY, midX, midY);
		}
		ctx.closePath();
		ctx.lineJoin = "round";
		ctx.lineCap = "round";
		ctx.strokeStyle = "rgba(255, 255, 255, 0.35)";
		ctx.lineWidth = 6.5;
		ctx.stroke();
		ctx.strokeStyle = "#ffffff";
		ctx.lineWidth = 2.6;
		ctx.stroke();
		ctx.restore();
	}
	drawLabelBadge(ctx, text, cx, cy) {
		ctx.save();
		ctx.font = "600 13px -apple-system, BlinkMacSystemFont, sans-serif";
		const pillW = ctx.measureText(text).width + 18;
		const pillH = 24;
		const rx = cx - pillW * .5;
		const ry = Math.max(-184.32, cy - pillH * .5);
		ctx.fillStyle = "rgba(255, 255, 255, 0.95)";
		ctx.beginPath();
		ctx.roundRect(rx, ry, pillW, pillH, 999);
		ctx.fill();
		ctx.fillStyle = "#0f172a";
		ctx.textAlign = "center";
		ctx.textBaseline = "middle";
		ctx.fillText(text, cx, ry + pillH * .5);
		ctx.restore();
	}
	/**
	* Direct mathematical ray-billboard intersection (zero `THREE.Raycaster` / `hitMesh` overhead).
	*/
	updateHoveredTrackFromRays(liveParams) {
		const user = xb.core.user;
		if (!user || !liveParams || this.tracks.length === 0) {
			this.hoveredTrack = null;
			return;
		}
		let bestMaskHit = null;
		let bestMaskArea = Infinity;
		let bestBoxHit = null;
		let bestBoxArea = Infinity;
		const numControllers = user.controllers?.length ?? 0;
		for (let c = 0; c < numControllers; c++) {
			this.tempRay.direction.set(0, 0, 0);
			user.getRay(c, this.tempRay);
			if (this.tempRay.direction.lengthSq() < 1e-6) continue;
			for (const trk of this.tracks) {
				if (!trk.node.visible || trk.visualOpacity <= .15) continue;
				this.tempPlaneNormal.copy(this.tempCamPos).sub(trk.node.position).normalize();
				this.tempPlane.setFromNormalAndCoplanarPoint(this.tempPlaneNormal, trk.node.position);
				const hitPt = this.tempRay.intersectPlane(this.tempPlane, this.tempHitPt);
				if (!hitPt) continue;
				this.tempLocalVec.copy(hitPt).sub(trk.node.position);
				const localX = this.tempLocalVec.dot(liveParams.cameraRight);
				const localY = this.tempLocalVec.dot(liveParams.cameraUp);
				const halfW = trk.worldWidth * .5;
				const halfH = trk.worldHeight * .5;
				if (Math.abs(localX) > halfW || Math.abs(localY) > halfH) continue;
				const uBox = localX / Math.max(.02, trk.worldWidth) + .5;
				const vBox = .5 - localY / Math.max(.02, trk.worldHeight);
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
	isDescendantOf(obj, ancestor) {
		if (!obj || !ancestor) return false;
		let cur = obj;
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
	onSelectStart(event) {
		if (this.hudCard && (xb.user?.isPointingAt?.(this.hudCard) || xb.user?.isSelectingAt?.(this.hudCard) || this.isDescendantOf(event?.target, this.hudCard) || this.isDescendantOf(event?.surface, this.hudCard))) return;
		const liveParams = this.captureExtendedCameraParams();
		this.updateHoveredTrackFromRays(liveParams);
		const candidate = this.hoveredTrack;
		if (candidate) {
			this.lockedTrack = candidate;
			candidate.state = "confirmed";
			candidate.missedFrames = 0;
			this.animatedOutlinePoints = null;
			this.trackerOffsetX = 0;
			this.trackerOffsetY = 0;
			this.activeAngle = 0;
			if (this.workerReady) this.callWorker("lock_track", {
				trackId: candidate.trackId,
				box: candidate.maskBox ?? candidate.lastCaptureBox,
				fgMask32: candidate.fgMask32,
				needsSamDecode: !candidate.contourProfile
			}).then((res) => {
				if (this.lockedTrack === candidate) {
					if (res.contourProfile) {
						candidate.contourProfile = res.contourProfile;
						this.lastLockedSamTimeMs = performance.now();
					}
					candidate.fgMask32 = res.fgMask32;
					if (res.encoderMs > 0) this.lastEncoderMs = res.encoderMs;
					if (res.decoderMs > 0) {
						candidate.samIou = res.samIou;
						candidate.samDecoderMs = res.decoderMs;
					}
					if (this.metricLatencyEl) this.metricLatencyEl.textContent = `${this.lastYoloeMs.toFixed(0)}ms + ${this.lastEncoderMs.toFixed(0)}ms + ${candidate.samDecoderMs.toFixed(1)}ms`;
				}
			}).catch(() => {});
			if (this.metricObjectEl) this.metricObjectEl.textContent = `${candidate.className} (${(candidate.score * 100).toFixed(0)}%)`;
			this.setStatus(`🔒 <strong>Tracking ${candidate.className} (${(candidate.score * 100).toFixed(0)}%)</strong> — Smooth white outline curve locked at OpenXR depth.`);
		} else if (this.lockedTrack) this.clearLockedTrack();
	}
	update() {
		const camera = xb.core.camera;
		if (!camera) return;
		const liveParams = this.captureExtendedCameraParams();
		camera.getWorldPosition(this.tempCamPos);
		for (let i = this.tracks.length - 1; i >= 0; i--) {
			const trk = this.tracks[i];
			const isLocked = trk === this.lockedTrack;
			let targetOpacity = 0;
			if (isLocked || trk.state === "confirmed") targetOpacity = 1;
			else if (trk.state === "coasting") targetOpacity = .6;
			else if (trk.state === "tentative" || trk.state === "expiring") targetOpacity = 0;
			const fadeStep = targetOpacity > trk.visualOpacity ? .14 : .08;
			trk.visualOpacity += Math.sign(targetOpacity - trk.visualOpacity) * Math.min(fadeStep, Math.abs(targetOpacity - trk.visualOpacity));
			if (trk.state === "expiring" && trk.visualOpacity <= .01) {
				trk.node.resetForPool();
				this.tracks.splice(i, 1);
				continue;
			}
			const node = trk.node;
			node.visible = trk.visualOpacity > .01;
			if (!node.visible) continue;
			this.tempTargetPos.copy(trk.worldCenter);
			if (isLocked && liveParams) {
				const egoProj = this.projectTrackTo2DBox(trk, liveParams);
				const boxW = egoProj ? Math.max(24, egoProj.box.x2 - egoProj.box.x1) : 160;
				const boxH = egoProj ? Math.max(24, egoProj.box.y2 - egoProj.box.y1) : 160;
				const worldDx = this.trackerOffsetX / boxW * trk.worldWidth;
				const worldDy = -(this.trackerOffsetY / boxH) * trk.worldHeight;
				this.tempTargetPos.addScaledVector(liveParams.cameraRight, worldDx).addScaledVector(liveParams.cameraUp, worldDy);
			}
			node.position.lerp(this.tempTargetPos, .45);
			node.lookAt(this.tempCamPos);
			const isotropicSpan = Math.max(trk.worldWidth, trk.worldHeight) * QUAD_EXPANSION;
			node.visualMesh.scale.set(isotropicSpan, isotropicSpan, 1);
			this.drawQuadCanvas(trk);
			if (isLocked && this.metricDepthEl) this.metricDepthEl.textContent = `${trk.depthInMeters.toFixed(2)} m`;
		}
		this.updateHoveredTrackFromRays(liveParams);
		if (!this.lockedTrack && this.hoveredTrack && this.metricDepthEl) this.metricDepthEl.textContent = `${this.hoveredTrack.depthInMeters.toFixed(2)} m`;
	}
};
const options = new xb.Options();
options.enableCamera();
options.deviceCamera.willCaptureFrequently = false;
options.deviceCamera.videoConstraints = {
	width: { ideal: 1280 },
	height: { ideal: 720 },
	facingMode: "environment"
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
options.setAppTitle("XR YOLOE-11s × SAM Semantic Tracker");
document.addEventListener("DOMContentLoaded", () => {
	xb.add(new YoloSamXRScript());
	xb.init(options);
});
//#endregion
export { YoloSamXRScript };
