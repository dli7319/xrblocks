import {css, html, LitElement} from 'lit';
import * as THREE from 'three';
import * as xb from 'xrblocks';

type HandSide = 'left' | 'right';
type GlbSource = 'existing' | 'uploaded';

function roundNum(value: number): number {
  return Number(value.toFixed(6));
}

function getDisplayedJoints(
  bones: Array<THREE.Object3D | undefined>
): xb.SimulatorHandPoseJoints {
  return xb.HAND_JOINT_NAMES.map((_, i) => {
    const bone = bones[i];
    if (!bone) {
      return {t: [0, 0, 0], r: [0, 0, 0, 1], s: [1, 1, 1]};
    }
    return {
      t: bone.position.toArray().map(roundNum),
      r: bone.quaternion.toArray().map(roundNum),
      s: bone.scale.toArray().map(roundNum),
    };
  });
}

function formatJointsJson(joints: xb.SimulatorHandPoseJoints): string {
  const rows = joints.map(
    (j) =>
      `  {\n    "t": [${j.t.join(', ')}],\n    "r": [${j.r.join(', ')}],\n    "s": [${(j.s ?? [1, 1, 1]).join(', ')}]\n  }`
  );
  return `[\n${rows.join(',\n')}\n]`;
}

function parseCustomJointsInput(
  rawText: string,
  fallback: xb.SimulatorHandPoseJoints
): xb.SimulatorHandPoseJoints {
  const normalized = rawText
    .trim()
    .replace(/^export\s+const\s+\w+\s*=\s*/, '')
    .replace(/^const\s+\w+\s*=\s*/, '')
    .replace(/\s+as\s+const\s*;?\s*$/, '')
    .replace(/;\s*$/, '')
    .replace(/([{,]\s*)(t|r|s)(\s*:)/g, '$1"$2"$3')
    .replace(/,\s*([}\]])/g, '$1');

  const parsed: unknown = JSON.parse(normalized);

  if (Array.isArray(parsed)) {
    if (
      parsed.length >= xb.HAND_JOINT_NAMES.length * 7 &&
      parsed.every((n) => typeof n === 'number')
    ) {
      const nums = parsed as number[];
      return xb.HAND_JOINT_NAMES.map((_, i) => {
        const base = i * 7;
        return {
          t: [nums[base], nums[base + 1], nums[base + 2]].map(roundNum),
          r: [
            nums[base + 3],
            nums[base + 4],
            nums[base + 5],
            nums[base + 6],
          ].map(roundNum),
          s: [1, 1, 1],
        };
      });
    }

    return xb.HAND_JOINT_NAMES.map((_, i) => {
      const item = (parsed[i] ?? {}) as {
        t?: number[];
        r?: number[];
        s?: number[];
      };
      const fb = fallback[i] ?? {t: [0, 0, 0], r: [0, 0, 0, 1], s: [1, 1, 1]};
      return {
        t:
          Array.isArray(item.t) && item.t.length >= 3
            ? [Number(item.t[0]), Number(item.t[1]), Number(item.t[2])].map(
                roundNum
              )
            : [...fb.t],
        r:
          Array.isArray(item.r) && item.r.length >= 4
            ? [
                Number(item.r[0]),
                Number(item.r[1]),
                Number(item.r[2]),
                Number(item.r[3]),
              ].map(roundNum)
            : [...fb.r],
        s:
          Array.isArray(item.s) && item.s.length >= 3
            ? [Number(item.s[0]), Number(item.s[1]), Number(item.s[2])].map(
                roundNum
              )
            : [...(fb.s ?? [1, 1, 1])],
      };
    });
  }

  if (parsed && typeof parsed === 'object') {
    const obj = parsed as Record<
      string,
      {t?: number[]; r?: number[]; s?: number[]}
    >;
    return xb.HAND_JOINT_NAMES.map((jointName, i) => {
      const item = obj[jointName] ?? {};
      const fb = fallback[i] ?? {t: [0, 0, 0], r: [0, 0, 0, 1], s: [1, 1, 1]};
      return {
        t:
          Array.isArray(item.t) && item.t.length >= 3
            ? [Number(item.t[0]), Number(item.t[1]), Number(item.t[2])].map(
                roundNum
              )
            : [...fb.t],
        r:
          Array.isArray(item.r) && item.r.length >= 4
            ? [
                Number(item.r[0]),
                Number(item.r[1]),
                Number(item.r[2]),
                Number(item.r[3]),
              ].map(roundNum)
            : [...fb.r],
        s:
          Array.isArray(item.s) && item.s.length >= 3
            ? [Number(item.s[0]), Number(item.s[1]), Number(item.s[2])].map(
                roundNum
              )
            : [...(fb.s ?? [1, 1, 1])],
      };
    });
  }

  throw new Error('Expected an array of 25 {t, r, s} joints.');
}

class HandDemoScene extends xb.Script {
  constructor(private onFrameSync: () => void) {
    super();
  }

  override init() {
    this.add(new THREE.HemisphereLight(0xaaaaaa, 0x666666, 3));
  }

  override update() {
    this.onFrameSync();
  }
}

export class HandJointsPanel extends LitElement {
  static properties = {
    leftSource: {type: String},
    rightSource: {type: String},
    leftUploadedName: {type: String},
    rightUploadedName: {type: String},
    selectedHand: {type: String},
    activePoseLabel: {type: String},
    jointsText: {type: String},
    status: {type: String},
  };

  static styles = css`
    :host {
      position: fixed;
      top: 12px;
      right: 12px;
      width: 360px;
      max-height: calc(100vh - 24px);
      display: flex;
      flex-direction: column;
      gap: 10px;
      padding: 14px;
      border-radius: 12px;
      background: rgba(15, 23, 42, 0.88);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(148, 163, 184, 0.25);
      box-shadow: 0 12px 32px rgba(0, 0, 0, 0.45);
      color: #f1f5f9;
      font-family:
        system-ui,
        -apple-system,
        sans-serif;
      font-size: 13px;
      z-index: 1000;
      box-sizing: border-box;
      overflow-y: auto;
      user-select: none;
    }

    .drag-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      cursor: grab;
      padding-bottom: 4px;
      touch-action: none;
    }

    .drag-header:active {
      cursor: grabbing;
    }

    h2 {
      margin: 0;
      font-size: 14px;
      font-weight: 700;
    }

    .drag-grip {
      font-size: 14px;
      color: #64748b;
      letter-spacing: 1px;
    }

    .section {
      display: flex;
      flex-direction: column;
      gap: 6px;
      padding-top: 8px;
      border-top: 1px solid rgba(148, 163, 184, 0.18);
    }

    .section-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .pose-pill {
      font-size: 11px;
      padding: 2px 7px;
      border-radius: 999px;
      background: rgba(56, 189, 248, 0.16);
      border: 1px solid rgba(56, 189, 248, 0.35);
      color: #38bdf8;
      font-weight: 600;
    }

    .row {
      display: flex;
      gap: 6px;
      align-items: center;
    }

    button,
    label.upload-btn {
      flex: 1;
      padding: 6px 8px;
      border-radius: 6px;
      border: 1px solid rgba(148, 163, 184, 0.3);
      background: rgba(30, 41, 59, 0.9);
      color: #e2e8f0;
      font: inherit;
      font-size: 12px;
      font-weight: 600;
      text-align: center;
      cursor: pointer;
    }

    button:hover:not(:disabled),
    label.upload-btn:hover {
      border-color: #38bdf8;
      color: #fff;
    }

    button.active {
      background: rgba(56, 189, 248, 0.22);
      border-color: #38bdf8;
      color: #38bdf8;
    }

    button:disabled {
      opacity: 0.45;
      cursor: not-allowed;
    }

    label.upload-btn input {
      display: none;
    }

    textarea {
      width: 100%;
      height: 220px;
      box-sizing: border-box;
      resize: vertical;
      border-radius: 6px;
      border: 1px solid rgba(148, 163, 184, 0.3);
      background: rgba(10, 15, 28, 0.9);
      color: #e2e8f0;
      padding: 8px;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 11px;
      line-height: 1.4;
      white-space: pre;
      user-select: text;
    }

    .status {
      font-size: 11.5px;
      color: #94a3b8;
    }
  `;

  declare leftSource: GlbSource;
  declare rightSource: GlbSource;
  declare leftUploadedName: string;
  declare rightUploadedName: string;
  declare selectedHand: HandSide;
  declare activePoseLabel: string;
  declare jointsText: string;
  declare status: string;

  private uploadedUrls: Partial<Record<HandSide, string>> = {};
  private isEditingTextarea = false;
  private dragging = false;
  private dragOffsetX = 0;
  private dragOffsetY = 0;

  constructor() {
    super();
    this.leftSource = 'existing';
    this.rightSource = 'existing';
    this.leftUploadedName = '';
    this.rightUploadedName = '';
    this.selectedHand = 'left';
    this.activePoseLabel = 'Relaxed';
    this.jointsText = '[]';
    this.status = 'Ready';
  }

  override connectedCallback(): void {
    super.connectedCallback();
    const simHands = xb.core.simulator?.hands;
    const ctrlState = xb.core.simulator?.controls?.simulatorControllerState;
    if (ctrlState) {
      this.selectedHand =
        ctrlState.currentControllerIndex === 0 ? 'left' : 'right';
    }
    if (simHands) {
      simHands.onHandednessChanged = (handedness) => {
        this.selectedHand = handedness;
        this.isEditingTextarea = false;
        this.syncFromSimulatorHands();
      };
      simHands.handPosePanelElement?.addEventListener(
        'simulatorHandPoseChangeRequest',
        this.onHandPosePanelRequest
      );
    }
    this.syncFromSimulatorHands();
  }

  override disconnectedCallback(): void {
    super.disconnectedCallback();
    const simHands = xb.core.simulator?.hands;
    simHands?.handPosePanelElement?.removeEventListener(
      'simulatorHandPoseChangeRequest',
      this.onHandPosePanelRequest
    );
    if (this.uploadedUrls.left) URL.revokeObjectURL(this.uploadedUrls.left);
    if (this.uploadedUrls.right) URL.revokeObjectURL(this.uploadedUrls.right);
  }

  private onHandPosePanelRequest = (): void => {
    this.isEditingTextarea = false;
    this.syncFromSimulatorHands();
  };

  syncFromSimulatorHands(): void {
    const simHands = xb.core.simulator?.hands;
    const ctrlState = xb.core.simulator?.controls?.simulatorControllerState;
    if (!simHands) return;

    if (ctrlState) {
      const activeHand: HandSide =
        ctrlState.currentControllerIndex === 0 ? 'left' : 'right';
      if (this.selectedHand !== activeHand) {
        this.selectedHand = activeHand;
      }
    }

    const pose =
      this.selectedHand === 'left'
        ? simHands.leftHandPose
        : simHands.rightHandPose;
    this.activePoseLabel = pose
      ? xb.SIMULATOR_HAND_POSE_NAMES[pose]
      : 'Custom Joints';

    if (!this.isEditingTextarea) {
      const bones =
        this.selectedHand === 'left'
          ? simHands.leftHandBones
          : simHands.rightHandBones;
      const formatted = formatJointsJson(getDisplayedJoints(bones));
      if (this.jointsText !== formatted) {
        this.jointsText = formatted;
      }
    }
  }

  private selectHand(side: HandSide): void {
    this.selectedHand = side;
    this.isEditingTextarea = false;
    const simHands = xb.core.simulator?.hands;
    const ctrlState = xb.core.simulator?.controls?.simulatorControllerState;
    if (ctrlState && simHands) {
      ctrlState.currentControllerIndex = side === 'left' ? 0 : 1;
      simHands.updateHandPosePanel();
    }
    this.syncFromSimulatorHands();
  }

  private onDragStart(event: PointerEvent): void {
    if (event.button !== 0) return;
    const rect = this.getBoundingClientRect();
    this.dragging = true;
    this.dragOffsetX = event.clientX - rect.left;
    this.dragOffsetY = event.clientY - rect.top;
    this.style.left = `${rect.left}px`;
    this.style.top = `${rect.top}px`;
    this.style.right = 'auto';
    (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
    event.stopPropagation();
  }

  private onDragMove(event: PointerEvent): void {
    if (!this.dragging) return;
    const maxX = Math.max(0, window.innerWidth - this.offsetWidth);
    const maxY = Math.max(0, window.innerHeight - this.offsetHeight);
    const nextX = Math.min(maxX, Math.max(0, event.clientX - this.dragOffsetX));
    const nextY = Math.min(maxY, Math.max(0, event.clientY - this.dragOffsetY));
    this.style.left = `${nextX}px`;
    this.style.top = `${nextY}px`;
    event.stopPropagation();
  }

  private onDragEnd(event: PointerEvent): void {
    if (!this.dragging) return;
    this.dragging = false;
    const target = event.currentTarget as HTMLElement;
    if (target.hasPointerCapture(event.pointerId)) {
      target.releasePointerCapture(event.pointerId);
    }
    event.stopPropagation();
  }

  private async switchHandGlb(
    side: HandSide,
    source: GlbSource
  ): Promise<void> {
    const simHands = xb.core.simulator?.hands;
    if (!simHands) return;

    const customUrl = this.uploadedUrls[side];
    if (source === 'uploaded' && !customUrl) return;

    if (side === 'left') {
      await simHands.loadLeftHandMesh(
        source === 'uploaded' ? customUrl : undefined
      );
      this.leftSource = source;
    } else {
      await simHands.loadRightHandMesh(
        source === 'uploaded' ? customUrl : undefined
      );
      this.rightSource = source;
    }

    if (this.selectedHand === side) {
      this.isEditingTextarea = false;
      this.syncFromSimulatorHands();
    }
    this.status = `Switched ${side} hand to ${source} GLB`;
  }

  private async onUploadGlb(side: HandSide, event: Event): Promise<void> {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;

    try {
      const prevUrl = this.uploadedUrls[side];
      if (prevUrl) {
        URL.revokeObjectURL(prevUrl);
      }
      const objectUrl = URL.createObjectURL(file);
      this.uploadedUrls[side] = objectUrl;

      if (side === 'left') {
        this.leftUploadedName = file.name;
      } else {
        this.rightUploadedName = file.name;
      }

      await this.switchHandGlb(side, 'uploaded');
      this.status = `Loaded ${file.name} (${side})`;
    } catch (error) {
      this.status = `Upload error: ${
        error instanceof Error ? error.message : String(error)
      }`;
    } finally {
      input.value = '';
    }
  }

  private async flipHandGlb(side: HandSide): Promise<void> {
    if (!this.uploadedUrls[side]) return;
    const current = side === 'left' ? this.leftSource : this.rightSource;
    const next: GlbSource = current === 'existing' ? 'uploaded' : 'existing';
    await this.switchHandGlb(side, next);
  }

  private loadDisplayedJointsIntoEditor(): void {
    this.isEditingTextarea = false;
    this.syncFromSimulatorHands();
  }

  private applyCustomJoints(): void {
    const simHands = xb.core.simulator?.hands;
    if (!simHands) return;

    try {
      const bones =
        this.selectedHand === 'left'
          ? simHands.leftHandBones
          : simHands.rightHandBones;
      const joints = parseCustomJointsInput(
        this.jointsText,
        getDisplayedJoints(bones)
      );
      if (this.selectedHand === 'left') {
        simHands.setLeftHandJoints(joints);
      } else {
        simHands.setRightHandJoints(joints);
      }
      simHands.syncHandJoints();
      this.isEditingTextarea = false;
      this.syncFromSimulatorHands();
      this.status = `Applied custom joints to ${this.selectedHand} hand`;
    } catch (error) {
      this.status = `Invalid JSON: ${
        error instanceof Error ? error.message : String(error)
      }`;
    }
  }

  private stopKeys(event: Event): void {
    event.stopPropagation();
  }

  private renderHandSection(side: HandSide) {
    const source = side === 'left' ? this.leftSource : this.rightSource;
    const uploadedName =
      side === 'left' ? this.leftUploadedName : this.rightUploadedName;
    const label = side === 'left' ? 'Left Hand GLB' : 'Right Hand GLB';

    return html`
      <div class="section">
        <strong>${label}</strong>
        <div class="row">
          <button
            type="button"
            class=${source === 'existing' ? 'active' : ''}
            @click=${() => this.switchHandGlb(side, 'existing')}
          >
            Existing (${side}.glb)
          </button>
          <button
            type="button"
            class=${source === 'uploaded' ? 'active' : ''}
            ?disabled=${!uploadedName}
            @click=${() => this.switchHandGlb(side, 'uploaded')}
          >
            ${uploadedName || 'Uploaded (None)'}
          </button>
          <button
            type="button"
            ?disabled=${!uploadedName}
            @click=${() => this.flipHandGlb(side)}
          >
            Flip
          </button>
        </div>
        <div class="row">
          <label class="upload-btn">
            Upload ${side === 'left' ? 'Left' : 'Right'} .glb
            <input
              type="file"
              accept=".glb,.gltf"
              @change=${(e: Event) => this.onUploadGlb(side, e)}
            />
          </label>
        </div>
      </div>
    `;
  }

  render() {
    return html`
      <div
        class="drag-header"
        @pointerdown=${this.onDragStart}
        @pointermove=${this.onDragMove}
        @pointerup=${this.onDragEnd}
        @pointercancel=${this.onDragEnd}
      >
        <h2>Simulator Hands GLB & Joints</h2>
        <span class="drag-grip" title="Drag to move panel">⋮⋮</span>
      </div>

      ${this.renderHandSection('left')} ${this.renderHandSection('right')}

      <div class="section">
        <div class="section-header">
          <strong>Custom Hand Joints</strong>
          <span class="pose-pill">${this.activePoseLabel}</span>
        </div>
        <div class="row">
          <button
            type="button"
            class=${this.selectedHand === 'left' ? 'active' : ''}
            @click=${() => this.selectHand('left')}
          >
            Left Hand
          </button>
          <button
            type="button"
            class=${this.selectedHand === 'right' ? 'active' : ''}
            @click=${() => this.selectHand('right')}
          >
            Right Hand
          </button>
          <button type="button" @click=${this.loadDisplayedJointsIntoEditor}>
            Load Current
          </button>
        </div>
        <textarea
          .value=${this.jointsText}
          @focus=${() => {
            this.isEditingTextarea = true;
          }}
          @blur=${() => {
            if (!this.jointsText.trim()) {
              this.isEditingTextarea = false;
            }
          }}
          @input=${(e: Event) => {
            this.isEditingTextarea = true;
            this.jointsText = (e.target as HTMLTextAreaElement).value;
          }}
          @keydown=${this.stopKeys}
          @keyup=${this.stopKeys}
          @keypress=${this.stopKeys}
          spellcheck="false"
        ></textarea>
        <div class="row">
          <button type="button" @click=${this.applyCustomJoints}>
            Apply Custom Joints
          </button>
        </div>
      </div>

      <div class="status">${this.status}</div>
    `;
  }
}

customElements.define('hand-joints-panel', HandJointsPanel);

async function start() {
  const options = new xb.Options();
  options.enableReticles();
  options.enableHands();
  options.setAppTitle('Simulator Hand GLB & Joints');
  options.hands.visualization = true;
  options.hands.visualizeJoints = true;
  options.hands.visualizeMeshes = false;
  options.simulator.defaultMode = xb.SimulatorMode.POSE;

  let panel: HandJointsPanel | undefined;
  xb.add(
    new HandDemoScene(() => {
      panel?.syncFromSimulatorHands();
    })
  );
  await xb.init(options);

  panel = document.createElement('hand-joints-panel') as HandJointsPanel;
  document.body.appendChild(panel);
}

document.addEventListener('DOMContentLoaded', start);
