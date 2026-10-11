// app/sim-main.js
/**
 * Spatial Colocation — XR Blocks rebuild (desktop simulator testbed).
 *
 * The same 4-phase colocation pipeline as the raw demo
 * (`lib/{orb,matching,geometry,map,relocalize}.js` — contracts per
 * IMPLEMENTATION_SPEC.md) running as an XR Blocks application:
 *
 * - head poses come from the XB camera (`xb.core.camera` world matrix; the
 *   simulator virtual user on desktop, WebXR on device) — camera IS the head,
 *   so `T_head_camera` is identity;
 * - the ORB feed comes from the SDK's device camera stream
 *   (`app/sim-feed.js` -> `XRDeviceCamera.video`, fed in the simulator by
 *   `SimulatorCamera` and on devices by the real camera), so the pipeline runs
 *   on the pixels a real deployment would see;
 * - ALL UI lives in a world-anchored `xb.UICard` (XR Blocks UI framework) —
 *   DOM is invisible inside a WebXR session, so the status/stats/buttons/
 *   devices/log card renders in both the desktop simulator and immersive
 *   sessions. Every log line is additionally mirrored into a hidden DOM `#log`
 *   element for `sim-transport.html`'s probe() and operators;
 * - a world-locked stability cube (0.25 m, orange emissive) is placed 1 m in
 *   front of the camera and must stay glued to the world as the virtual head
 *   moves — its apparent stability IS the measurement;
 * - networking/presence is identical to the raw demo (`app/net.js` protocol).
 *
 * Map-frame convention IDENTICAL to the raw demo: the map frame is the
 * builder's head frame at the first keyframe (`setMapOrigin`).
 *
 * URL params: `?room=NAME&mode=build|relocalize|live&label=NAME&debug=1`,
 * `?autoSweep=1`, `?fov=DEG`, `?xrAutomation=1`.
 *
 * `?autoSweep=1` moves the virtual head with the sanctioned simulator
 * "journey" mechanism (`Simulator.simulatorUser.loadJourney` + a
 * `SimulatorUserAction`-shaped step, see `startAutoSweep()`), falling back to
 * per-`update()` camera writes when the simulator is not running. Either way
 * the motion is `autoSweepPoseAt()` from `app/sim-feed.js`: one slow circle
 * (0.5 m radius, 30 s per lap) at 1.5 m height with a +/-60 deg yaw scan
 * (15 s) and +/-5 deg pitch nod (9 s) around the outward direction.
 */

import * as THREE from 'three';
import * as xb from 'xrblocks';

import {loadCv} from '../lib/cv-runtime.js';
import {
  extractOrb,
  estimateIntrinsics,
  MAX_FEATURES,
  DEFAULT_FOV_DEG,
  MIN_PARALLAX_DEG,
  KEYFRAME_MIN_MS,
  KEYFRAME_MIN_MOVE_M,
  KEYFRAME_MIN_ROT_DEG,
  PRESENCE_INTERVAL_MS,
} from '../lib/orb.js';
import {matchDescriptors} from '../lib/matching.js';
import {
  estimateEssential,
  triangulate,
  invertRigid,
  matMul,
} from '../lib/geometry.js';
import {
  createMap,
  addKeyframe,
  setMapOrigin,
  serializeMap,
  deserializeMap,
} from '../lib/map.js';
import {
  relocalize,
  headPoseFromCameraPose,
  MIN_RELOC_INLIERS,
} from '../lib/relocalize.js';
import {createNet} from './net.js';
import * as store from './store.js';
import {
  createCameraFeed,
  matrixToNested,
  intrinsicsFromProjection,
  autoSweepPoseAt,
  FEED_WIDTH,
} from './sim-feed.js';

// Local timing constants — copied exactly from app/main.js (raw demo).
const PROCESS_INTERVAL_MS = 100; // feature extraction <= 10 Hz
const RELOC_MIN_INTERVAL_MS = 250; // PnP search <= 4 Hz
const POSE_TTL_MS = 4000;
const MAP_FALLBACK_MS = 3500;
const MAP_RETRY_MS = 5000;
const DEVICE_RENDER_INTERVAL_MS = 200;
const SWEEP_JOURNEY_WAIT_MS = 5000;

// UI update cadences.
const STATS_TEXT_INTERVAL_MS = 200; // stats text <= 5 Hz
const LOG_TAIL_LINES = 6;
const DOM_LOG_MAX_NODES = 200;

// Stability cube.
const CUBE_SIZE_M = 0.25;
const CUBE_DISTANCE_M = 1.0;

const IDENTITY = [
  [1, 0, 0, 0],
  [0, 1, 0, 0],
  [0, 0, 1, 0],
  [0, 0, 0, 1],
];

const MODE_TITLES = {
  idle: 'Idle',
  build: 'Build',
  relocalize: 'Relocalize',
  live: 'Live',
};

function randId(n) {
  return Math.random()
    .toString(36)
    .slice(2, 2 + n);
}

function clone4(T) {
  return T.map((row) => row.slice());
}

function rot3(T) {
  return [
    [T[0][0], T[0][1], T[0][2]],
    [T[1][0], T[1][1], T[1][2]],
    [T[2][0], T[2][1], T[2][2]],
  ];
}

/** Rotation angle (deg) between two 4x4 poses. */
function rotationDegBetween(Ta, Tb) {
  const Ra = rot3(Ta);
  const Rb = rot3(Tb);
  let tr = 0;
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) tr += Ra[i][j] * Rb[i][j];
  }
  const c = Math.min(1, Math.max(-1, (tr - 1) / 2));
  return (Math.acos(c) * 180) / Math.PI;
}

function translationDist(Ta, Tb) {
  return Math.hypot(
    Ta[0][3] - Tb[0][3],
    Ta[1][3] - Tb[1][3],
    Ta[2][3] - Tb[2][3]
  );
}

/**
 * Apply an `autoSweepPoseAt()` pose to the XB camera (the virtual head).
 *
 * @param {THREE.Camera} camera XB camera
 * @param {number} tSec seconds since the sweep started
 */
function applySweepPose(camera, tSec) {
  const pose = autoSweepPoseAt(tSec);
  camera.position.set(pose.position[0], pose.position[1], pose.position[2]);
  camera.quaternion.setFromEuler(
    new THREE.Euler(pose.pitch, pose.yaw, 0, 'YXZ')
  );
  camera.updateMatrixWorld(true);
}

// ---- in-session UI (XR Blocks UI framework) ---------------------------------

/**
 * The sim's single main `UICard`, built with the XR Blocks UI components so it
 * renders in-world in the desktop simulator AND inside immersive WebXR
 * sessions (DOM panels would be invisible there). Layout: status line, stats
 * row (`kf · lm · matches · fps · inliers`, text updated at <= 5 Hz), mode +
 * action buttons, per-device presence lines, and a 6-line log tail. The card
 * is world-anchored ~1 m ahead at ~1.45 m with `manipulation: true` so users
 * can move it. Every log line is also mirrored into the hidden DOM `#log`
 * element kept for `sim-transport.html`'s probe() and operators.
 */
class SimCardUI {
  constructor({room, label, onMode, onPlaceCube, onSaveMap, onLoadMap}) {
    this.mode = 'idle';
    this.badge = '';
    this.stats = {
      keyframes: 0,
      landmarks: 0,
      matches: 0,
      fps: 0,
      inliers: null,
    };
    this.lastStatsWrite = 0;
    this.logLines = [];
    this.domLog = document.getElementById('log');

    this.statusText = new xb.UIText({
      text: MODE_TITLES.idle,
      style: {
        fontSize: 18,
        fontWeight: 'bold',
        width: '100%',
        whiteSpace: 'pre-line',
      },
    });
    this.statsText = new xb.UIText({
      text: this.formatStats(),
      style: {fontSize: 15, opacity: 0.85, width: '100%'},
    });
    this.netText = new xb.UIText({
      text: 'net: …',
      style: {fontSize: 13, opacity: 0.7, width: '100%'},
    });
    this.devicesText = new xb.UIText({
      text: 'none yet',
      style: {
        fontSize: 14,
        width: '100%',
        whiteSpace: 'pre-line',
        lineHeight: 1.35,
      },
    });
    this.logText = new xb.UIText({
      text: '',
      style: {
        fontSize: 12,
        opacity: 0.7,
        width: '100%',
        whiteSpace: 'pre-line',
        lineHeight: 1.3,
      },
    });

    const modeButton = (title, mode) =>
      new xb.UIButton({
        label: title,
        style: {flexGrow: 1},
        onClick: () => onMode(mode),
      });

    this.card = new xb.UICard({
      size: {width: 0.62, height: 'auto'},
      manipulation: true,
      edge: true,
      style: {flexDirection: 'column', gap: 10, padding: 18},
      children: [
        new xb.UIText({
          text: 'Spatial Colocation · sim',
          style: {
            fontSize: 26,
            fontWeight: 'bold',
            textAlign: 'center',
            width: '100%',
          },
        }),
        new xb.UIText({
          text: `${room} · ${label}`,
          style: {
            fontSize: 13,
            opacity: 0.7,
            textAlign: 'center',
            width: '100%',
          },
        }),
        this.statusText,
        this.statsText,
        new xb.UIPanel({
          style: {width: '100%', flexDirection: 'row', gap: 8},
          children: [
            modeButton('Build', 'build'),
            modeButton('Relocalize', 'relocalize'),
            modeButton('Live', 'live'),
          ],
        }),
        new xb.UIPanel({
          style: {width: '100%', flexDirection: 'row', gap: 8},
          children: [
            new xb.UIButton({
              label: 'Place cube @1m',
              style: {flexGrow: 1},
              onClick: onPlaceCube,
            }),
            new xb.UIButton({
              label: 'Save map',
              style: {flexGrow: 1},
              onClick: onSaveMap,
            }),
            new xb.UIButton({
              label: 'Load map',
              style: {flexGrow: 1},
              onClick: onLoadMap,
            }),
          ],
        }),
        this.netText,
        new xb.UIText({
          text: 'Devices relocalized to this map',
          style: {fontSize: 13, opacity: 0.7, width: '100%'},
        }),
        this.devicesText,
        new xb.UIText({
          text: 'Log',
          style: {fontSize: 13, opacity: 0.7, width: '100%'},
        }),
        this.logText,
      ],
    });
  }

  /** Compose the status line: mode + badge text. */
  refreshStatus() {
    const title = MODE_TITLES[this.mode] || this.mode;
    this.statusText.text = this.badge ? `${title}: ${this.badge}` : title;
  }

  setMode(mode) {
    this.mode = mode;
    this.refreshStatus();
  }

  setBadge(badge) {
    this.badge = badge;
    this.refreshStatus();
  }

  /** Merge stats and rewrite the stats text at <= 5 Hz. */
  setStats(delta) {
    Object.assign(this.stats, delta);
    const now = performance.now();
    if (now - this.lastStatsWrite < STATS_TEXT_INTERVAL_MS) return;
    this.lastStatsWrite = now;
    this.statsText.text = this.formatStats();
  }

  formatStats() {
    const s = this.stats;
    return (
      `kf ${s.keyframes} · lm ${s.landmarks} · matches ${s.matches} · ` +
      `fps ${s.fps} · inliers ${s.inliers ?? '—'}`
    );
  }

  /** One line per device ('Dev-xxxx (you) ✓ N inliers'); empty → 'none yet'. */
  setDevices(list) {
    this.devicesText.text = list.length
      ? list
          .map(
            (d) =>
              `${d.label || d.peerId}${d.self ? ' (you)' : ''} ✓ ${d.inliers} inliers`
          )
          .join('\n')
      : 'none yet';
  }

  setNet({status, detail, isHost, peerCount}) {
    this.netInfo = {status, detail, isHost, peerCount};
    this.refreshNetText();
  }

  /** Feed source (camera vs synthetic fallback) + runtime surface. */
  setFeed(kind, surface) {
    const source = kind === 'camera' ? 'device camera' : 'synthetic fallback';
    this.feedInfo = `source: ${source} · xr: ${surface}`;
    this.refreshNetText();
  }

  refreshNetText() {
    const n = this.netInfo;
    let netPart = 'net: …';
    if (n) {
      const role = n.isHost ? ' · host' : '';
      const peers =
        typeof n.peerCount === 'number' ? ` · ${n.peerCount} peers` : '';
      netPart = `net: ${n.status}${role}${peers}${n.detail ? ` · ${n.detail}` : ''}`;
    }
    this.netText.text = this.feedInfo
      ? `${netPart} · ${this.feedInfo}`
      : netPart;
  }

  /** Log a line to the card tail (last LOG_TAIL_LINES) and mirror it to DOM. */
  log(line, level = '') {
    const text = level ? `${level}: ${line}` : line;
    this.logLines.push(text);
    while (this.logLines.length > LOG_TAIL_LINES) this.logLines.shift();
    this.logText.text = this.logLines.join('\n');
    if (this.domLog) {
      this.domLog.appendChild(document.createTextNode(`${text}\n`));
      while (this.domLog.childNodes.length > DOM_LOG_MAX_NODES) {
        this.domLog.removeChild(this.domLog.firstChild);
      }
    }
  }
}

// ---- URL params --------------------------------------------------------------

const params = new URLSearchParams(location.search);
const room =
  (params.get('room') || randId(6))
    .replace(/[^a-zA-Z0-9_-]/g, '')
    .slice(0, 24) || randId(6);
const startMode = params.get('mode');
const label = (params.get('label') || `Sim-${randId(4)}`).slice(0, 24);
const wantDebug = params.has('debug');
const wantAutoSweep = params.get('autoSweep') === '1';
const fovOverride = params.get('fov') ? parseFloat(params.get('fov')) : null;
const wantAutomation = params.get('xrAutomation') === '1';

// ---- the owning Script -------------------------------------------------------

class SimMain extends xb.Script {
  name = 'Spatial Colocation Sim';

  constructor() {
    super();
    this.engineCamera = null;
    this.deviceCamera = null;
    this.feed = null;
    this.net = null;
    this.ui = null;
    this.cv = null;
    this.storedMaps = [];
    this.helpers = {
      cv: null,
      estimateEssential,
      triangulate,
      matchDescriptors,
      MIN_PARALLAX_DEG,
    };
    this.cube = null;
    this.avatarGroup = null;
    this.avatars = new Map(); // peerId -> {group, label, lastSeen}
    this.mapRetryTimer = 0;
    this.sweep = {t0: 0, journeyActive: false, timer: 0};
    this.onUnload = () => this.dispose();
    this.state = {
      mode: 'idle',
      K: null,
      KFrameW: 0,
      KSource: null,
      map: null,
      mapName: null,
      hasMap: false,
      T_head_camera: clone4(IDENTITY), // camera IS the head
      lastKf: null, // {ts, T}
      kf0Pose: null, // builder's T_ref_head at the first keyframe
      prevKps: null,
      prevDesc: null,
      T_map_head: null,
      T_map_ref: null, // T_map_ref paired with T_map_head (map->ref, fixed while tracking holds)
      relocRef: null, // {T_map_head, T_ref_head} — propagate between relocs
      lastReloc: null, // {inliers, ts}
      matches: 0,
      poses: new Map(), // peerId -> {label, T_map_head, inliers, ts}
      lastBroadcast: 0,
      lastMapReq: 0,
      lastProcess: 0,
      lastRelocStep: 0,
      lastDeviceRender: 0,
      fps: 0,
      fpsSamples: [],
      headPose: null,
      feedKind: null,
      cube: {placed: false, worldPos: null, mapPos: null},
    };
  }

  // ---- scene setup (runs during xb.init) ------------------------------------

  init() {
    this.add(new THREE.HemisphereLight(0xffffff, 0x666666, 3));
    const sun = new THREE.DirectionalLight(0xffffff, 1.5);
    sun.position.set(3, 6, 4);
    this.add(sun);

    // In-session UI card: world-anchored ~1 m ahead at ~1.45 m, movable.
    this.ui = new SimCardUI({
      room,
      label,
      onMode: (mode) => this.setMode(mode),
      onPlaceCube: () => this.placeCube(),
      onSaveMap: () => this.saveMapNow(),
      onLoadMap: () => this.loadLatestMap(),
    });
    this.ui.card.position.set(0, 1.45, -1.1);
    this.add(this.ui.card);

    // World-locked stability cube: placed once (first valid head pose) and on
    // demand; it NEVER follows the camera afterwards.
    const cube = new THREE.Mesh(
      new THREE.BoxGeometry(CUBE_SIZE_M, CUBE_SIZE_M, CUBE_SIZE_M),
      new THREE.MeshStandardMaterial({
        color: 0xff8c1a,
        emissive: 0xff6600,
        emissiveIntensity: 0.6,
        metalness: 0.1,
        roughness: 0.5,
      })
    );
    cube.name = 'Stability Cube';
    cube.visible = false;
    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(cube.geometry),
      new THREE.LineBasicMaterial({color: 0xffe08a})
    );
    cube.add(edges);
    this.cube = cube;
    xb.scene.add(cube);

    this.avatarGroup = new THREE.Group();
    this.avatarGroup.name = 'Colocation Avatars';
    xb.scene.add(this.avatarGroup);

    // Triangulated map landmarks as a 3D point cloud. ORB detections are 2D,
    // but every map landmark holds a 3D map-frame position (that is what PnP
    // relocalization consumes). The cloud lives in MAP frame; the group carries
    // the paired map->ref transform, so while tracking holds the points stay
    // glued to the world exactly like the stability cube.
    this.landmarkGroup = new THREE.Group();
    this.landmarkGroup.name = 'Map Landmarks';
    this.landmarkGroup.matrixAutoUpdate = false;
    xb.scene.add(this.landmarkGroup);
    this.landmarkPoints = null;
    this.landmarkCount = -1;
  }

  /**
   * Post-engine boot (called after `await xb.init()` resolves): device-camera
   * feed creation, OpenCV, networking, UI state.
   */
  async postInit() {
    this.engineCamera = xb.core.camera;
    if (fovOverride && Number.isFinite(fovOverride) && fovOverride > 0) {
      this.engineCamera.fov = fovOverride;
      this.engineCamera.updateProjectionMatrix();
    }

    // The ORB feed is the SDK device-camera stream: in the simulator it is fed
    // by SimulatorCamera (XRDeviceCamera.simulatorCamera), on devices by the
    // real camera.
    this.deviceCamera = xb.core.deviceCamera ?? null;
    this.feed = createCameraFeed({
      deviceCamera: this.deviceCamera,
      width: FEED_WIDTH,
    });
    this.state.feedKind = this.feed.kind;
    this.ui.log(
      this.feed.kind === 'camera'
        ? 'ORB feed: XR Blocks device camera (XRDeviceCamera video stream)'
        : 'ORB feed: synthetic fallback (no device camera)',
      this.feed.kind === 'camera' ? '' : 'warn'
    );
    this.ui.setFeed(
      this.feed.kind,
      xb.core.simulatorRunning || xb.core.simulator ? 'simulator' : 'device'
    );

    this.net = createNet({room, label, hasMap: false});
    this.wireNet();

    try {
      this.cv = await loadCv();
      this.ui.log('OpenCV ready');
    } catch (err) {
      this.ui.log(`loadCv failed: ${err.message} — polling global cv`, 'warn');
      this.cv = await new Promise((resolve, reject) => {
        const t0 = performance.now();
        const timer = setInterval(() => {
          const g = globalThis.cv;
          if (g && g.Mat) {
            clearInterval(timer);
            resolve(g);
          } else if (performance.now() - t0 > 30000) {
            clearInterval(timer);
            reject(new Error('cv never became available'));
          }
        }, 100);
      });
    }
    this.helpers.cv = this.cv;

    this.ui.log(
      `room=${room} label=${label} autoSweep=${wantAutoSweep ? 1 : 0}`
    );
    if (['build', 'relocalize', 'live'].includes(startMode)) {
      this.setMode(startMode);
    } else {
      this.ui.setBadge('pick a mode: Build · Relocalize · Live');
    }
    await this.refreshMapList();
    window.addEventListener('beforeunload', this.onUnload);

    if (wantAutoSweep) {
      this.startAutoSweep();
      this.ui.log('autoSweep: scripted virtual-head sweep active (30 s loop)');
    }
    if (wantDebug) {
      window.__scoloc = {
        state: this.state,
        net: this.net,
        placeCube: () => this.placeCube(),
        xb,
      };
    }
  }

  // ---- networking (identical protocol to the raw demo) ----------------------

  wireNet() {
    const net = this.net;
    net.on('status', ({status, detail, isHost}) => {
      this.ui.setNet({status, detail, isHost, peerCount: net.peerCount});
      this.ui.log(
        `net: ${status}${detail ? ` — ${detail}` : ''}`,
        status === 'error' ? 'error' : ''
      );
      // The peer may connect after Relocalize asked for a map — ask again.
      if (status === 'joined' || status === 'host') this.requestMapSoon();
    });
    net.on('peers', (peers) => {
      this.ui.setNet({
        status: net.isHost ? 'host' : 'joined',
        peerCount: Math.max(0, peers.length - 1),
      });
    });
    net.on('pose', (msg) => {
      if (!msg || !msg.peerId || msg.peerId === net.peerId) return;
      this.state.poses.set(msg.peerId, {
        label: msg.label,
        T_map_head: msg.T_map_head,
        inliers: msg.inliers,
        ts: performance.now(),
      });
    });
    net.on('map-request', (msg) => {
      if (!this.state.hasMap || !this.state.map) return;
      try {
        const bytes = serializeMap(this.state.map);
        net.sendMap(bytes, this.state.mapName || `${room}-map`, msg.peerId);
        this.ui.log(`sent map (${bytes.length} B) to ${msg.peerId}`);
      } catch (err) {
        this.ui.log(`serialize failed: ${err.message}`, 'error');
      }
    });
    net.on('map-data', ({from, name, bytes}) => {
      try {
        const m = deserializeMap(bytes);
        this.adoptMap(m, name);
        this.ui.log(
          `map "${name}" received from ${from} (${m.landmarks?.length || 0} landmarks)`
        );
        this.ui.setBadge(`map: ${name}`);
      } catch (err) {
        this.ui.log(`bad map from ${from}: ${err.message}`, 'error');
      }
    });
  }

  // ---- mode machine ---------------------------------------------------------

  /** Ask the room for a map (rate-limited; used on mode entry and net connect). */
  requestMapSoon() {
    if (this.state.mode !== 'relocalize' || this.state.map) return;
    const now = performance.now();
    if (now - this.state.lastMapReq < 2000) return;
    this.state.lastMapReq = now;
    this.net.requestMap();
  }

  setMode(mode) {
    if (!['build', 'relocalize', 'live'].includes(mode)) return;
    this.state.mode = mode;
    this.ui.setMode(mode);
    if (mode === 'build') {
      this.ui.setBadge('mapping…');
      this.ui.log('Build: ORB -> keyframes -> triangulated landmark map');
    } else if (mode === 'relocalize') {
      this.acquireMapForReloc();
      this.ui.log('Relocalize: PnP against the stored map');
    } else if (mode === 'live') {
      if (!this.state.map) {
        this.ui.setBadge('no map yet — Relocalize first');
        this.ui.log(
          'Live without a map: broadcast starts after relocalization',
          'warn'
        );
      } else {
        this.ui.setBadge(
          this.state.T_map_head
            ? 'broadcasting head pose'
            : 'waiting for a pose'
        );
        this.ui.log('Live: broadcasting presence at 10 Hz');
      }
    }
  }

  // ---- autoSweep (scripted virtual head) ------------------------------------

  /**
   * `?autoSweep=1`: deterministic virtual-head sweep with zero user input.
   *
   * Primary mechanism — the simulator's sanctioned scripted-user API: a
   * `SimulatorUserAction`-shaped journey played through
   * `xb.core.simulator.simulatorUser.loadJourney()` (src/simulator/SimulatorUser.ts;
   * `WalkTowardsPanelAction` shows actions move the head by writing the camera
   * position/quaternion). The journey is stoppable via `stopJourney()`.
   *
   * Fallback — if the simulator runtime is not up within a few seconds (e.g.
   * autoSweep without the simulator running), the same `applySweepPose()` runs
   * every `update()`. The simulator's control modes only apply input-driven
   * movement and with no input their per-frame movement pass is a zero-delta
   * copy (`navMesh.enabled` defaults to false), so scripted writes are not
   * fought. Both writers share one t0, so a handover is seamless.
   */
  startAutoSweep() {
    this.sweep.t0 = performance.now();
    const t0 = this.sweep.t0;
    const sweep = this.sweep;
    const camera = this.engineCamera;

    const tryJourney = () => {
      const simulator = xb.core.simulator;
      const simulatorUser = simulator && simulator.simulatorUser;
      if (!simulatorUser || typeof simulatorUser.loadJourney !== 'function') {
        return false;
      }
      try {
        simulatorUser.loadJourney([
          {
            async init() {},
            async play({simulatorUser: user, journeyId, waitFrame}) {
              sweep.journeyActive = true;
              while (user.isOnJourneyId(journeyId)) {
                applySweepPose(camera, (performance.now() - t0) / 1000);
                await waitFrame.waitFrame();
              }
              sweep.journeyActive = false;
            },
          },
        ]);
        this.ui.log('autoSweep: driving the head via SimulatorUser journey');
        return true;
      } catch (err) {
        this.ui.log(`autoSweep journey failed: ${err.message}`, 'warn');
        return false;
      }
    };

    if (!tryJourney()) {
      // The simulator runtime chunk loads lazily — give it a moment.
      const startedAt = performance.now();
      this.sweep.timer = window.setInterval(() => {
        if (this.sweep.journeyActive || tryJourney()) {
          window.clearInterval(this.sweep.timer);
          this.sweep.timer = 0;
        } else if (performance.now() - startedAt > SWEEP_JOURNEY_WAIT_MS) {
          window.clearInterval(this.sweep.timer);
          this.sweep.timer = 0;
          this.ui.log(
            'autoSweep: simulator unavailable — writing the camera from update()',
            'warn'
          );
        }
      }, 250);
    }

    // Hidden tabs must not drive the virtual user (and the sweep work starves
    // the event loop badly enough that a backgrounded room host stops answering
    // PeerJS connection offers — clients hang at 'connecting to host').
    this.sweep.tryJourney = tryJourney;
    this.onVisibilityChange = () => {
      if (document.hidden) this.pauseSweep();
      else this.resumeSweep();
    };
    document.addEventListener('visibilitychange', this.onVisibilityChange);
  }

  /** Stop all sweep work (journey + fallback timer). */
  pauseSweep() {
    const sweep = this.sweep;
    if (sweep.timer) {
      window.clearInterval(sweep.timer);
      sweep.timer = 0;
    }
    if (sweep.journeyActive) {
      const simulatorUser =
        xb.core.simulator && xb.core.simulator.simulatorUser;
      try {
        if (simulatorUser) simulatorUser.stopJourney();
      } catch (err) {
        /* noop */
      }
      sweep.journeyActive = false;
    }
    sweep.paused = true;
  }

  /** Restart the sweep when the tab becomes visible again. */
  resumeSweep() {
    const sweep = this.sweep;
    if (!sweep.paused) return;
    sweep.paused = false;
    if (sweep.tryJourney) sweep.tryJourney();
  }

  // ---- frame processing -----------------------------------------------------

  /** Head pose `T_ref_head` — camera world matrix; the camera IS the head. */
  currentHeadPose() {
    if (this.feed && this.feed.kind === 'synthetic') {
      const T = this.feed.getHeadPose && this.feed.getHeadPose();
      if (T) return T;
    }
    const camera = this.engineCamera;
    camera.updateMatrixWorld(true);
    return matrixToNested(camera.matrixWorld);
  }

  ensureIntrinsics(frame) {
    if (this.state.K && this.state.KFrameW === frame.width) return this.state.K;
    let source;
    if (this.feed && this.feed.kind === 'synthetic') {
      // Synthetic fallback renders with the raw demo's 60 deg camera.
      this.state.K = estimateIntrinsics(
        frame.width,
        frame.height,
        DEFAULT_FOV_DEG
      );
      source = 'synthetic feed (estimateIntrinsics, 60 deg)';
    } else {
      try {
        // Camera-provided parameters: the device-camera clip matrix — in the
        // simulator the SimulatorCamera center-crop frustum
        // (CameraUtils.getDeviceCameraClipFromView), on devices the
        // CameraParameterUtils device profiles. This matches the pixels the
        // camera stream actually delivers.
        const clip = xb.getDeviceCameraClipFromView(
          this.engineCamera,
          this.deviceCamera,
          xb.detectDeviceCameraTarget()
        );
        this.state.K = intrinsicsFromProjection(
          clip,
          frame.width,
          frame.height
        );
        source =
          'device-camera clip matrix (CameraUtils.getDeviceCameraClipFromView)';
      } catch (err) {
        this.state.K = intrinsicsFromProjection(
          this.engineCamera.projectionMatrix,
          frame.width,
          frame.height
        );
        source = `render-camera projectionMatrix (fallback: ${err.message})`;
      }
    }
    this.state.KFrameW = frame.width;
    this.state.KSource = source;
    this.ui.log(`intrinsics: ${source}`);
    return this.state.K;
  }

  keyframeGate(now, T) {
    if (!this.state.lastKf) return true;
    if (now - this.state.lastKf.ts < KEYFRAME_MIN_MS) return false;
    return (
      translationDist(T, this.state.lastKf.T) >= KEYFRAME_MIN_MOVE_M ||
      rotationDegBetween(T, this.state.lastKf.T) >= KEYFRAME_MIN_ROT_DEG
    );
  }

  /**
   * Record our map-frame head pose together with its paired map->ref
   * transform (`T_map_ref = T_map_head · inv(T_ref_head)`). Using the pair is
   * what keeps derived poses (cube, avatars) consistent even while the head
   * moves between keyframes.
   */
  setMapHead(T_map_head, headPose) {
    this.state.T_map_head = T_map_head;
    this.state.T_map_ref = matMul(T_map_head, invertRigid(headPose));
  }

  buildStep(now, headPose, kps, descriptors, frame) {
    const state = this.state;
    if (!state.map) {
      state.map = createMap({
        width: frame.width,
        height: frame.height,
        K: state.K,
      });
      state.kf0Pose = null;
      this.ui.log(`map created (${frame.width}x${frame.height})`);
    }
    if (!this.keyframeGate(now, headPose)) return;
    const fresh = state.map.keyframes.length === 0;
    if (fresh) setMapOrigin(state.map, headPose);
    let res;
    try {
      res = addKeyframe(
        state.map,
        {
          timestamp: now,
          T_ref_head: headPose,
          T_head_camera: state.T_head_camera,
          kps,
          descriptors,
        },
        this.helpers
      );
    } catch (err) {
      this.ui.log(`addKeyframe error: ${err.message}`, 'error');
      return;
    }
    if (!res || res.newKeyframe === false) return;
    state.lastKf = {ts: now, T: headPose};
    if (fresh && !state.kf0Pose) {
      state.kf0Pose = headPose;
    } else if (state.kf0Pose) {
      // Own pose in map frame (map frame = head frame at the first keyframe).
      this.setMapHead(matMul(invertRigid(state.kf0Pose), headPose), headPose);
    }
    if (!state.hasMap) {
      state.hasMap = true;
      this.net.setHasMap(true);
      this.net.announce();
      this.ui.log(
        `map is now shareable — ${state.map.landmarks.length} landmarks`
      );
    }
  }

  /**
   * Rebuild the 3D landmark cloud when the map grows or changes (cheap count
   * check; rebuilds happen at keyframe/adopt rate). Points are map-frame.
   */
  updateLandmarkPoints() {
    const map = this.state.map;
    const count =
      map && Array.isArray(map.landmarks) ? map.landmarks.length : 0;
    if (count === this.landmarkCount) return;
    this.landmarkCount = count;
    if (this.landmarkPoints) {
      this.landmarkGroup.remove(this.landmarkPoints);
      this.landmarkPoints.geometry.dispose();
      this.landmarkPoints.material.dispose();
      this.landmarkPoints = null;
    }
    if (!count) return;
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const lm = map.landmarks[i];
      positions[3 * i] = lm.position[0];
      positions[3 * i + 1] = lm.position[1];
      positions[3 * i + 2] = lm.position[2];
      const [r, g, b] = this.descriptorColor(lm.descriptor);
      colors[3 * i] = r;
      colors[3 * i + 1] = g;
      colors[3 * i + 2] = b;
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    this.landmarkPoints = new THREE.Points(
      geometry,
      new THREE.PointsMaterial({
        size: 0.012,
        vertexColors: true,
        sizeAttenuation: true,
      })
    );
    this.landmarkPoints.name = 'ORB Landmarks';
    this.landmarkGroup.add(this.landmarkPoints);
  }

  /** Deterministic bright color from a 32-byte ORB descriptor. */
  descriptorColor(desc) {
    let h = 0;
    const d = desc || [];
    for (let i = 0; i < d.length; i++) h = (h * 31 + d[i]) >>> 0;
    const c = new THREE.Color().setHSL((h % 360) / 360, 0.65, 0.6);
    return [c.r, c.g, c.b];
  }

  /** Pose the cloud group with the paired map->ref transform (hidden untracked). */
  updateLandmarkGroupPose() {
    const T_map_ref = this.state.T_map_ref;
    if (!T_map_ref) {
      this.landmarkGroup.visible = false;
      return;
    }
    this.landmarkGroup.visible = true;
    const T_ref_map = invertRigid(T_map_ref);
    this.landmarkGroup.matrix.set(
      T_ref_map[0][0],
      T_ref_map[0][1],
      T_ref_map[0][2],
      T_ref_map[0][3],
      T_ref_map[1][0],
      T_ref_map[1][1],
      T_ref_map[1][2],
      T_ref_map[1][3],
      T_ref_map[2][0],
      T_ref_map[2][1],
      T_ref_map[2][2],
      T_ref_map[2][3],
      0,
      0,
      0,
      1
    );
    this.landmarkGroup.matrixWorldNeedsUpdate = true;
  }

  liveRelocalize() {
    const state = this.state;
    if (!state.map) return null;
    try {
      return relocalize(
        state.map,
        {kps: state.prevKps, descriptors: state.prevDesc, K: state.K},
        {cv: this.cv}
      );
    } catch (err) {
      return null;
    }
  }

  relocStep(headPose) {
    // The relocalization solve is heavy on large maps; cap it below frame rate.
    const now = performance.now();
    if (now - this.state.lastRelocStep < RELOC_MIN_INTERVAL_MS) return;
    this.state.lastRelocStep = now;
    const r = this.liveRelocalize();
    if (r) {
      this.state.lastReloc = {inliers: r.inliers, ts: now};
      const T_map_head = headPoseFromCameraPose(
        r.T_map_camera,
        this.state.T_head_camera
      );
      this.setMapHead(T_map_head, headPose);
      this.state.relocRef = {T_map_head, T_ref_head: headPose};
      this.ui.setStats({inliers: r.inliers});
      this.ui.setBadge(`relocalized ✓ ${r.inliers} inliers`);
      this.ui.log(
        `relocalized ✓ ${r.inliers} inliers (need >= ${MIN_RELOC_INLIERS})`
      );
      if (this.state.mode === 'relocalize') this.setMode('live');
    } else {
      this.ui.setBadge('no pose yet — keep scanning the mapped area');
    }
  }

  liveStep(now, headPose) {
    const state = this.state;
    const r = this.liveRelocalize();
    if (r) {
      const T_map_head = headPoseFromCameraPose(
        r.T_map_camera,
        state.T_head_camera
      );
      this.setMapHead(T_map_head, headPose);
      state.relocRef = {T_map_head, T_ref_head: headPose};
      state.lastReloc = {inliers: r.inliers, ts: now};
    } else if (state.relocRef && headPose) {
      // Propagate between successful relocalizations via the head-pose track.
      this.setMapHead(
        matMul(
          state.relocRef.T_map_head,
          matMul(invertRigid(state.relocRef.T_ref_head), headPose)
        ),
        headPose
      );
    }
    if (state.T_map_head) {
      this.ui.setStats({
        inliers: state.lastReloc ? state.lastReloc.inliers : null,
      });
    }
  }

  /** Presence roster + labeled avatars for relocalized devices. */
  renderDevices(now) {
    const list = [];
    if (this.state.T_map_head) {
      list.push({
        peerId: this.net.peerId,
        label,
        T_map_head: this.state.T_map_head,
        inliers: this.state.lastReloc ? this.state.lastReloc.inliers : 0,
        self: true,
      });
    }
    for (const [pid, p] of this.state.poses) {
      if (now - p.ts > POSE_TTL_MS) {
        this.state.poses.delete(pid);
        continue;
      }
      list.push({
        peerId: pid,
        label: p.label,
        T_map_head: p.T_map_head,
        inliers: p.inliers,
        self: false,
      });
    }
    this.ui.setDevices(list);
    this.updateAvatars(list);
  }

  /** Place labeled avatars at each remote device's map-frame head pose. */
  updateAvatars(list) {
    const T_map_ref = this.state.T_map_ref;
    const seen = new Set();
    if (T_map_ref) {
      // avatar pose in ref: T_ref_peer = inv(T_map_ref) · T_map_head_peer.
      const T_ref_map = invertRigid(T_map_ref);
      for (const d of list) {
        if (d.self || !d.T_map_head) continue;
        seen.add(d.peerId);
        const T_ref_peer = matMul(T_ref_map, d.T_map_head);
        const avatar = this.getAvatar(d.peerId, d.label);
        avatar.position.set(
          T_ref_peer[0][3],
          T_ref_peer[1][3],
          T_ref_peer[2][3]
        );
        avatar.visible = true;
      }
    }
    for (const [pid, avatar] of this.avatars) {
      if (!seen.has(pid)) avatar.visible = false;
    }
  }

  getAvatar(peerId, peerLabel) {
    let avatar = this.avatars.get(peerId);
    if (avatar) return avatar;
    avatar = new THREE.Group();
    const capsule = new THREE.Mesh(
      new THREE.CapsuleGeometry(0.18, 0.5, 4, 12),
      new THREE.MeshStandardMaterial({color: 0x38bdf8, roughness: 0.6})
    );
    capsule.position.y = 0.25;
    avatar.add(capsule);
    const labelCanvas = document.createElement('canvas');
    labelCanvas.width = 256;
    labelCanvas.height = 64;
    const ctx = labelCanvas.getContext('2d');
    ctx.fillStyle = 'rgba(10, 12, 16, 0.75)';
    ctx.fillRect(0, 0, 256, 64);
    ctx.fillStyle = '#e6edf5';
    ctx.font = '28px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(String(peerLabel || peerId).slice(0, 16), 128, 32);
    const sprite = new THREE.Sprite(
      new THREE.SpriteMaterial({
        map: new THREE.CanvasTexture(labelCanvas),
        depthTest: false,
      })
    );
    sprite.position.y = 1.15;
    sprite.scale.set(0.8, 0.2, 1);
    avatar.add(sprite);
    avatar.visible = false;
    this.avatarGroup.add(avatar);
    this.avatars.set(peerId, avatar);
    return avatar;
  }

  processFrame(frame, now) {
    const state = this.state;
    this.ensureIntrinsics(frame);
    const {keypoints, descriptors} = extractOrb(this.cv, frame, {
      maxFeatures: MAX_FEATURES,
    });

    let matchesList = [];
    if (state.prevDesc) {
      try {
        matchesList = matchDescriptors(state.prevDesc, descriptors);
      } catch (err) {
        matchesList = [];
      }
    }
    state.matches = matchesList.length;

    const headPose = this.currentHeadPose();
    state.headPose = headPose;
    state.prevKps = keypoints;
    state.prevDesc = descriptors;

    // The stability cube is placed once at startup after the first valid head
    // pose (and on demand via the button / window.__scoloc.placeCube).
    if (!state.cube.placed) this.placeCube();

    if (state.mode === 'build') {
      this.buildStep(now, headPose, keypoints, descriptors, frame);
    } else if (state.mode === 'relocalize') {
      this.relocStep(headPose);
    } else if (state.mode === 'live') {
      this.liveStep(now, headPose);
    }
    this.updateCubeMapPos();
    this.updateLandmarkPoints();
    this.updateLandmarkGroupPose();

    // Presence and the device roster run in EVERY mode: builders broadcast as
    // soon as they have a map-frame pose.
    if (state.T_map_head && now - state.lastBroadcast >= PRESENCE_INTERVAL_MS) {
      this.net.sendPose(
        state.T_map_head,
        state.lastReloc ? state.lastReloc.inliers : 0
      );
      state.lastBroadcast = now;
    }
    if (now - state.lastDeviceRender >= DEVICE_RENDER_INTERVAL_MS) {
      this.renderDevices(now);
      state.lastDeviceRender = now;
    }

    state.fpsSamples.push(now);
    while (state.fpsSamples.length && now - state.fpsSamples[0] > 1000) {
      state.fpsSamples.shift();
    }
    state.fps = state.fpsSamples.length;

    this.ui.setStats({
      keyframes: state.map ? state.map.keyframes.length : 0,
      landmarks: state.map ? state.map.landmarks.length : 0,
      matches: state.matches,
      fps: state.fps,
    });
  }

  // ---- stability cube -------------------------------------------------------

  /**
   * Place the world-locked stability cube 1 m in front of the camera:
   * world position = camera world position + camera forward (-Z of the camera
   * world matrix) × 1.0 m. From then on it never follows the camera — its
   * apparent stability under head motion IS the measurement.
   */
  placeCube() {
    const camera = this.engineCamera;
    if (!camera || !this.cube) return;
    camera.updateMatrixWorld(true);
    const e = camera.matrixWorld.elements;
    const position = new THREE.Vector3(e[12], e[13], e[14]);
    const forward = new THREE.Vector3(-e[8], -e[9], -e[10]).normalize();
    position.addScaledVector(forward, CUBE_DISTANCE_M);
    this.cube.position.copy(position);
    this.cube.quaternion.setFromRotationMatrix(camera.matrixWorld);
    this.cube.visible = true;
    this.state.cube.placed = true;
    this.state.cube.worldPos = [position.x, position.y, position.z];
    this.updateCubeMapPos();
    if (this.ui) {
      this.ui.log(
        `stability cube placed @1m: world (${position.x.toFixed(2)}, ` +
          `${position.y.toFixed(2)}, ${position.z.toFixed(2)})`
      );
    }
  }

  /**
   * Expose the cube's map-frame position:
   * `T_map_cube = T_map_ref · T_ref_cube` with the paired map->ref transform
   * (constant while tracking holds — drift here IS the stability measurement).
   */
  updateCubeMapPos() {
    const state = this.state;
    if (!state.cube.placed || !state.T_map_ref) return;
    const T_ref_cube = matrixToNested(this.cube.matrixWorld);
    const T_map_cube = matMul(state.T_map_ref, T_ref_cube);
    state.cube.mapPos = [T_map_cube[0][3], T_map_cube[1][3], T_map_cube[2][3]];
  }

  // ---- map persistence ------------------------------------------------------

  adoptMap(m, name) {
    const state = this.state;
    state.map = m;
    state.mapName = name || state.mapName;
    state.hasMap = true;
    state.lastKf = null;
    state.kf0Pose = null;
    state.T_map_head = null;
    state.T_map_ref = null;
    state.relocRef = null;
    state.lastReloc = null;
    state.cube.mapPos = null;
    this.ui.setStats({
      keyframes: m.keyframes?.length || 0,
      landmarks: m.landmarks?.length || 0,
      inliers: null,
    });
    this.net.setHasMap(true);
  }

  async saveMapNow() {
    const state = this.state;
    if (!state.map) {
      this.ui.log('nothing to save yet', 'warn');
      return;
    }
    // Auto-generated name (the card has no custom-name input).
    const name = `${room}-${label}-${new Date()
      .toISOString()
      .slice(11, 19)
      .replace(/:/g, '')}`;
    try {
      const bytes = serializeMap(state.map);
      await store.saveMap(name, bytes);
      state.mapName = name;
      state.hasMap = true;
      this.net.setHasMap(true);
      this.net.announce();
      await this.refreshMapList();
      this.ui.log(
        `saved map "${name}" (${bytes.length} B) + announced to room`
      );
    } catch (err) {
      this.ui.log(`save failed: ${err.message}`, 'error');
    }
  }

  async loadMapByName(name) {
    try {
      const bytes = await store.loadMap(name);
      if (!bytes) {
        this.ui.log(`map "${name}" not in IndexedDB`, 'warn');
        return false;
      }
      const m = deserializeMap(bytes);
      this.adoptMap(m, name);
      this.ui.log(
        `loaded map "${name}" (${m.landmarks?.length || 0} landmarks)`
      );
      this.ui.setBadge(`map: ${name}`);
      return true;
    } catch (err) {
      this.ui.log(`load failed: ${err.message}`, 'error');
      return false;
    }
  }

  /** "Load map" button: load the newest stored map (auto-generated names). */
  async loadLatestMap() {
    try {
      const maps = await store.listMaps();
      if (!maps.length) {
        this.ui.log('no stored maps yet — build and save one first', 'warn');
        return;
      }
      await this.loadMapByName(maps[0].name);
      if (maps.length > 1) {
        this.ui.log(
          `${maps.length} maps stored — loaded newest "${maps[0].name}"`
        );
      }
    } catch (err) {
      this.ui.log(`load failed: ${err.message}`, 'error');
    }
  }

  async refreshMapList() {
    try {
      this.storedMaps = await store.listMaps();
    } catch (err) {
      this.storedMaps = [];
      this.ui.log(`IndexedDB unavailable: ${err.message}`, 'warn');
    }
  }

  acquireMapForReloc() {
    const state = this.state;
    state.map = null;
    state.mapName = null;
    state.T_map_head = null;
    state.T_map_ref = null;
    state.relocRef = null;
    state.lastReloc = null;
    state.lastKf = null;
    state.kf0Pose = null;
    state.cube.mapPos = null;
    this.ui.setStats({inliers: null, keyframes: 0, landmarks: 0});
    this.ui.setBadge('asking room for map…');
    state.lastMapReq = performance.now();
    this.net.requestMap();
    window.setTimeout(async () => {
      if (state.map || state.mode !== 'relocalize') return;
      const maps = await store.listMaps().catch(() => []);
      if (maps.length) {
        const ok = await this.loadMapByName(maps[0].name);
        if (ok) this.ui.setBadge(`map: ${maps[0].name} (from storage)`);
      }
      if (state.map || state.mode !== 'relocalize') return;
      this.ui.setBadge('waiting for map…');
      this.ui.log('no map in room or storage yet — retrying the room', 'warn');
      if (!this.mapRetryTimer) {
        this.mapRetryTimer = window.setInterval(() => {
          if (state.map || state.mode !== 'relocalize') {
            window.clearInterval(this.mapRetryTimer);
            this.mapRetryTimer = 0;
            return;
          }
          this.requestMapSoon();
        }, MAP_RETRY_MS);
      }
    }, MAP_FALLBACK_MS);
  }

  // ---- frame loop + teardown -----------------------------------------------

  update() {
    const now = performance.now();
    if (document.hidden) return; // background tab: keep the event loop free
    if (wantAutoSweep && !this.sweep.journeyActive && this.engineCamera) {
      applySweepPose(this.engineCamera, (now - this.sweep.t0) / 1000);
    }
    if (!this.cv || !this.feed || !this.net) return;
    if (now - this.state.lastProcess < PROCESS_INTERVAL_MS) return;
    this.state.lastProcess = now;
    this.state.feedKind = this.feed.kind;
    const frame = this.feed.getFrame();
    if (!frame) return;
    try {
      this.processFrame(frame, now);
    } catch (err) {
      this.ui.log(
        `frame error: ${err && err.message ? err.message : err}`,
        'error'
      );
    }
  }

  dispose() {
    window.removeEventListener('beforeunload', this.onUnload);
    if (this.onVisibilityChange) {
      document.removeEventListener('visibilitychange', this.onVisibilityChange);
    }
    if (this.sweep.timer) {
      window.clearInterval(this.sweep.timer);
      this.sweep.timer = 0;
    }
    if (this.mapRetryTimer) {
      window.clearInterval(this.mapRetryTimer);
      this.mapRetryTimer = 0;
    }
    const simulatorUser = xb.core.simulator && xb.core.simulator.simulatorUser;
    if (simulatorUser && this.sweep.journeyActive) {
      try {
        simulatorUser.stopJourney();
      } catch (err) {
        /* noop */
      }
    }
    if (this.feed) {
      this.feed.stop();
      this.feed = null;
    }
    if (this.net) {
      this.net.disconnect();
      this.net = null;
    }
    if (this.cube) {
      this.cube.geometry.dispose();
      this.cube.material.dispose();
      this.cube.children.forEach((c) => {
        c.geometry.dispose();
        c.material.dispose();
      });
      this.cube.removeFromParent();
      this.cube = null;
    }
    for (const avatar of this.avatars.values()) {
      avatar.traverse((obj) => {
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) {
          if (obj.material.map) obj.material.map.dispose();
          obj.material.dispose();
        }
      });
    }
    this.avatars.clear();
    if (this.avatarGroup) {
      this.avatarGroup.removeFromParent();
      this.avatarGroup = null;
    }
    if (this.ui && this.ui.card) {
      this.ui.card.removeFromParent();
    }
    this.ui = null;
  }
}

// ---- boot -------------------------------------------------------------------

const main = new SimMain();
xb.add(main);

const options = new xb.Options();
options.canvas = document.getElementById('xb-canvas');
// ORB feed = the SDK device camera (simulator camera in the simulator).
options.enableCamera();
if (wantAutomation) {
  // The Options constructor auto-triggers on ?xrAutomation=1 too; call it
  // explicitly so the intent (and the automation preset) is unambiguous.
  options.enableAutomationMode();
}

(async () => {
  try {
    await xb.init(options);
    await main.postInit();
  } catch (err) {
    console.error('[sim-main] init failed', err);
    const log = document.getElementById('log');
    if (log) {
      log.appendChild(
        document.createTextNode(
          `init failed: ${err && err.message ? err.message : err}\n`
        )
      );
    }
  }
})();
