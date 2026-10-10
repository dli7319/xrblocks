// app/viz.js
/**
 * three.js visualization for the spatial-colocation demo:
 *  - landmark point cloud (THREE.Points, per-point color hashed from its descriptor)
 *  - other-device avatars (capsule + name label) placed at their broadcast map pose
 *  - shared-origin axis marker (shown once a device relocalized to the map)
 *  - 2D ORB keypoint overlay painter on a separate 2D canvas
 *
 * Map-frame content lives under `mapRoot`; in XR mode main.js poses mapRoot by T_ref_map
 * so the map is rendered in world space, on desktop the orbit camera views it directly.
 */
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';

function hashDescriptor(desc) {
  let h = 0;
  if (desc) {
    for (let i = 0; i < desc.length; i++) {
      h = (Math.imul(h, 31) + desc[i]) >>> 0;
    }
  }
  return h;
}

function labelTexture(label) {
  const pad = 8;
  const font = '600 28px system-ui, sans-serif';
  const meas = document.createElement('canvas').getContext('2d');
  meas.font = font;
  const w = Math.ceil(meas.measureText(label).width) + pad * 2;
  const h = 44;
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  const g = c.getContext('2d');
  g.font = font;
  g.fillStyle = 'rgba(16, 20, 28, 0.85)';
  g.strokeStyle = 'rgba(120, 200, 255, 0.9)';
  g.lineWidth = 2;
  const r = 10;
  g.beginPath();
  g.roundRect(1, 1, w - 2, h - 2, r);
  g.fill();
  g.stroke();
  g.fillStyle = '#e8f2ff';
  g.textBaseline = 'middle';
  g.fillText(label, pad, h / 2 + 1);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return {tex, aspect: w / h};
}

export class Viz {
  constructor({container, overlay}) {
    this.container = container;
    this.overlay = overlay;
    this.overlayCtx = overlay ? overlay.getContext('2d') : null;

    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      powerPreference: 'high-performance',
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.xr.enabled = true;
    container.appendChild(this.renderer.domElement);
    this.renderer.domElement.style.display = 'block';
    this.renderer.domElement.style.width = '100%';
    this.renderer.domElement.style.height = '100%';

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0d1015);
    this.scene.fog = new THREE.Fog(0x0d1015, 6, 20);

    this.camera = new THREE.PerspectiveCamera(60, 1, 0.05, 60);
    this.camera.position.set(0, 0.8, 2.6);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 0, -1.5);
    this.controls.enableDamping = true;
    this.controls.maxDistance = 15;

    this.scene.add(new THREE.HemisphereLight(0xbfd8ff, 0x1a2027, 1.6));
    const dir = new THREE.DirectionalLight(0xffffff, 1.2);
    dir.position.set(2, 4, 1);
    this.scene.add(dir);

    // Map-frame root (posed from T_ref_map in XR; identity on desktop).
    this.mapRoot = new THREE.Group();
    this.scene.add(this.mapRoot);

    this.points = null;
    this.pointsCount = -1;
    this.avatarGroup = new THREE.Group();
    this.mapRoot.add(this.avatarGroup);
    this.avatars = new Map(); // peerId -> {group, sprite, label}
    this._framed = false;

    this.origin = this.buildOriginMarker();
    this.origin.visible = false;
    this.mapRoot.add(this.origin);

    this._onResize = () => this.resize();
    this.resizeObserver = new ResizeObserver(this._onResize);
    this.resizeObserver.observe(container);
    this.resize();

    this.renderer.setAnimationLoop(() => {
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    });
  }

  buildOriginMarker() {
    const g = new THREE.Group();
    g.add(new THREE.AxesHelper(0.35));
    const ring = new THREE.Mesh(
      new THREE.RingGeometry(0.08, 0.1, 48),
      new THREE.MeshBasicMaterial({
        color: 0x66ccff,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.9,
      })
    );
    ring.rotation.x = -Math.PI / 2;
    g.add(ring);
    const {tex, aspect} = labelTexture('map origin');
    const sprite = new THREE.Sprite(
      new THREE.SpriteMaterial({map: tex, transparent: true, depthTest: false})
    );
    sprite.scale.set(0.22 * aspect, 0.22, 1);
    sprite.position.set(0, 0.22, 0);
    g.add(sprite);
    return g;
  }

  resize() {
    const w = Math.max(1, this.container.clientWidth);
    const h = Math.max(1, this.container.clientHeight);
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  /** Rebuild the landmark point cloud. `landmarks` = [{position: [x,y,z], descriptor}]. */
  setLandmarks(landmarks) {
    const list = landmarks || [];
    if (list.length === this.pointsCount) return;
    this.pointsCount = list.length;
    const pos = new Float32Array(list.length * 3);
    const col = new Float32Array(list.length * 3);
    const c = new THREE.Color();
    for (let i = 0; i < list.length; i++) {
      const p = list[i].position || [0, 0, 0];
      pos[i * 3] = p[0];
      pos[i * 3 + 1] = p[1];
      pos[i * 3 + 2] = p[2];
      const hue = (hashDescriptor(list[i].descriptor) % 360) / 360;
      c.setHSL(hue, 0.85, 0.6);
      col[i * 3] = c.r;
      col[i * 3 + 1] = c.g;
      col[i * 3 + 2] = c.b;
    }
    if (this.points) {
      this.mapRoot.remove(this.points);
      this.points.geometry.dispose();
      this.points.material.dispose();
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    this.points = new THREE.Points(
      geo,
      new THREE.PointsMaterial({
        size: 3,
        sizeAttenuation: false,
        vertexColors: true,
      })
    );
    this.mapRoot.add(this.points);
    if (!this._framed && list.length > 0) {
      this.frameLandmarks(list);
      this._framed = true;
    }
  }

  /** Orbit the desktop camera around the cloud bbox (first set only). */
  frameLandmarks(list) {
    const box = new THREE.Box3();
    const v = new THREE.Vector3();
    for (const l of list) {
      const p = l.position || [0, 0, 0];
      box.expandByPoint(v.set(p[0], p[1], p[2]));
    }
    if (box.isEmpty()) return;
    const center = box.getCenter(new THREE.Vector3());
    const size = Math.max(0.5, box.getSize(new THREE.Vector3()).length());
    this.controls.target.copy(center);
    this.camera.position.set(
      center.x + size * 0.5,
      center.y + size * 0.45,
      center.z + size * 1.1
    );
    this.controls.update();
  }

  /** Pose all map-frame content from T_ref_map (4x4 nested). */
  setMapPose(T) {
    this.mapRoot.matrixAutoUpdate = false;
    if (!T) {
      this.mapRoot.matrix.identity();
      return;
    }
    this.mapRoot.matrix.set(
      T[0][0],
      T[0][1],
      T[0][2],
      T[0][3],
      T[1][0],
      T[1][1],
      T[1][2],
      T[1][3],
      T[2][0],
      T[2][1],
      T[2][2],
      T[2][3],
      T[3][0],
      T[3][1],
      T[3][2],
      T[3][3]
    );
  }

  setOriginVisible(v) {
    this.origin.visible = !!v;
  }

  setXrActive(active) {
    this.controls.enabled = !active;
  }

  /** Upsert one device avatar. devices: [{peerId, label, T_map_head, inliers, self}]. */
  setDevices(devices) {
    const seen = new Set();
    for (const d of devices || []) {
      if (!d.T_map_head) continue;
      seen.add(d.peerId);
      let a = this.avatars.get(d.peerId);
      if (!a) {
        const group = new THREE.Group();
        const body = new THREE.Mesh(
          new THREE.CapsuleGeometry(0.13, 0.45, 6, 14),
          new THREE.MeshStandardMaterial({
            color: d.self ? 0x33bbff : 0xffb454,
            roughness: 0.5,
          })
        );
        body.position.y = 0.35;
        group.add(body);
        const {tex, aspect} = labelTexture(
          `${d.label || d.peerId}${d.self ? ' (you)' : ''}`
        );
        const sprite = new THREE.Sprite(
          new THREE.SpriteMaterial({
            map: tex,
            transparent: true,
            depthTest: false,
          })
        );
        sprite.scale.set(0.3 * aspect, 0.3, 1);
        sprite.position.y = 0.95;
        group.add(sprite);
        this.avatarGroup.add(group);
        a = {group, sprite, texture: tex, label: d.label};
        this.avatars.set(d.peerId, a);
      } else if (a.label !== d.label) {
        a.label = d.label;
        a.sprite.material.map?.dispose();
        const {tex, aspect} = labelTexture(
          `${d.label}${d.self ? ' (you)' : ''}`
        );
        a.sprite.material.map = tex;
        a.sprite.scale.set(0.3 * aspect, 0.3, 1);
      }
      const T = d.T_map_head;
      const m = new THREE.Matrix4().set(
        T[0][0],
        T[0][1],
        T[0][2],
        T[0][3],
        T[1][0],
        T[1][1],
        T[1][2],
        T[1][3],
        T[2][0],
        T[2][1],
        T[2][2],
        T[2][3],
        T[3][0],
        T[3][1],
        T[3][2],
        T[3][3]
      );
      a.group.position.setFromMatrixPosition(m);
      a.group.quaternion.setFromRotationMatrix(m);
    }
    for (const [pid, a] of this.avatars) {
      if (!seen.has(pid)) {
        this.avatarGroup.remove(a.group);
        a.texture?.dispose();
        this.avatars.delete(pid);
      }
    }
  }

  /**
   * Paint the current frame's ORB keypoints on the overlay canvas.
   * Keypoints: [{x, y, angle, size, octave}]; colored by size (proxy for response
   * strength — lib/orb.js does not expose response).
   */
  drawKeypoints(keypoints, width, height) {
    const ctx = this.overlayCtx;
    if (!ctx) return;
    if (this.overlay.width !== width || this.overlay.height !== height) {
      this.overlay.width = width;
      this.overlay.height = height;
    }
    ctx.clearRect(0, 0, width, height);
    const kps = keypoints || [];
    if (!kps.length) return;
    let min = Infinity;
    let max = -Infinity;
    for (const kp of kps) {
      if (kp.size < min) min = kp.size;
      if (kp.size > max) max = kp.size;
    }
    const span = max - min || 1;
    ctx.lineWidth = 1;
    for (const kp of kps) {
      const s = (kp.size - min) / span; // 0 (weak) -> 1 (strong)
      const hue = 210 - 210 * s; // blue -> red
      ctx.strokeStyle = `hsla(${hue}, 90%, 60%, 0.85)`;
      const r = Math.min(6, Math.max(1.5, (kp.size || 6) / 2));
      ctx.beginPath();
      ctx.arc(kp.x, kp.y, r, 0, Math.PI * 2);
      ctx.stroke();
      if (kp.angle !== undefined) {
        ctx.beginPath();
        ctx.moveTo(kp.x, kp.y);
        ctx.lineTo(
          kp.x + Math.cos(kp.angle) * r * 2,
          kp.y + Math.sin(kp.angle) * r * 2
        );
        ctx.stroke();
      }
    }
  }

  dispose() {
    this.renderer.setAnimationLoop(null);
    this.resizeObserver.disconnect();
    this.controls.dispose();
    this.renderer.dispose();
  }
}
