import * as THREE from 'three';

import {Input} from '../Input';
import {Handedness} from '../Hands';
import {HAND_JOINT_NAMES} from '../components/HandJointNames.js';
import {User} from '../../core/User';
import {Script} from '../../core/Script';
import {GestureEventDetail, GestureEventType} from './GestureEvents';
import {
  BuiltInGestureName,
  GestureRecognitionOptions,
  GestureConfiguration,
} from './GestureRecognitionOptions';
import {
  GestureDetector,
  GestureDetectorMap,
  HandContext,
  HandLabel,
  JointPositions,
} from './GestureTypes';
import {heuristicDetectors} from './providers/HeuristicGestureDetectors';

type ActiveGestureState = {
  confidence: number;
  data?: Record<string, unknown>;
};

const HAND_INDEX_TO_LABEL: Record<number, HandLabel> = {
  [Handedness.LEFT]: 'left',
  [Handedness.RIGHT]: 'right',
};

const JOINT_TEMP_POOL =
    new Map<HandLabel, Map<string, THREE.Vector3>>();

type GestureScriptEvent = THREE.Event&{
  type: GestureEventType,
  target: GestureRecognition,
  detail: GestureEventDetail,
};

interface GestureRecognitionEventMap extends THREE.Object3DEventMap {
  gesturestart: GestureScriptEvent;
  gestureupdate: GestureScriptEvent;
  gestureend: GestureScriptEvent;
}

export class GestureRecognition extends Script<GestureRecognitionEventMap> {
  static dependencies = {
    input: Input,
    user: User,
    options: GestureRecognitionOptions,
  };

  private options!: GestureRecognitionOptions;
  private user!: User;
  private input!: Input;
  private activeGestures: Record<HandLabel, Map<string, ActiveGestureState>> =
      {left: new Map(), right: new Map()};
  private lastEvaluation = 0;
  private detectors = new Map<BuiltInGestureName, GestureDetector>();
  private activeProvider: string|null = null;
  private providerWarned = false;

  async init(
      {options, user, input}: {
        options: GestureRecognitionOptions,
        user: User,
        input: Input
      }) {
    this.options = options;
    this.user = user;
    this.input = input;
    this.configureProvider(true);
    if (!this.options.enabled) {
      console.info(
          'GestureRecognition initialized but disabled. Call options.enableGestures() to activate.');
    }
  }

  update() {
    if (!this.options.enabled) return;
    if (!this.user.hands?.isValid?.()) return;

    this.configureProvider();

    const now = performance.now();
    const interval = this.activeProvider === 'heuristics' ?
        0 :
        this.options.updateIntervalMs;
    if (interval > 0 && (now - this.lastEvaluation) < interval) {
      return;
    }
    this.lastEvaluation = now;

    this.evaluateHand(Handedness.LEFT);
    this.evaluateHand(Handedness.RIGHT);
  }

  private configureProvider(force = false) {
    const provider = this.options.provider;
    if (!force && provider === this.activeProvider) return;

    this.detectors.clear();
    switch (provider) {
      case 'heuristics':
        this.assignDetectors(heuristicDetectors);
        this.providerWarned = false;
        break;
      case 'mediapipe':
      case 'tfjs':
        this.assignDetectors(heuristicDetectors);
        if (!this.providerWarned) {
          console.warn(
              `GestureRecognition: provider '${provider}' is not yet implemented; falling back to heuristics.`);
          this.providerWarned = true;
        }
        break;
      default:
        this.assignDetectors(heuristicDetectors);
        if (!this.providerWarned) {
          console.warn(
              `GestureRecognition: provider '${provider}' is unknown; falling back to heuristics.`);
          this.providerWarned = true;
        }
        break;
    }
    this.activeProvider = provider;
  }

  private assignDetectors(detectors: GestureDetectorMap) {
    for (const [name, detector] of Object.entries(detectors)) {
      if (!detector) continue;
      this.detectors.set(name as BuiltInGestureName, detector);
    }
  }

  private evaluateHand(handedness: Handedness) {
    const handLabel = HAND_INDEX_TO_LABEL[handedness];
    const activeMap = this.activeGestures[handLabel];
    if (!handLabel) return;

    const context = this.buildHandContext(handedness, handLabel);
    if (!context) {
      for (const [name] of activeMap.entries()) {
        this.emitGesture('gestureend', {name, hand: handLabel, confidence: 0});
      }
      activeMap.clear();
      return;
    }

    const processed = new Set<string>();
    for (const [name, config] of Object.entries(this.options.gestures)) {
      const gestureName = name as BuiltInGestureName;
      if (!config?.enabled) continue;
      const detector = this.detectors.get(gestureName);
      if (!detector) continue;

      const result = detector(context, config as GestureConfiguration);
      const isActive =
          result && result.confidence >= this.options.minimumConfidence;
      processed.add(gestureName);
      const previousState = activeMap.get(gestureName);

      if (isActive) {
        const detail: GestureEventDetail = {
          name: gestureName,
          hand: handLabel,
          confidence: THREE.MathUtils.clamp(result.confidence, 0, 1),
          data: result.data,
        };
        if (!previousState) {
          activeMap.set(gestureName,
                        {confidence: detail.confidence, data: detail.data});
          this.emitGesture('gesturestart', detail);
        } else {
          previousState.confidence = detail.confidence;
          previousState.data = detail.data;
          this.emitGesture('gestureupdate', detail);
        }
      } else if (previousState) {
        activeMap.delete(gestureName);
        this.emitGesture(
            'gestureend',
            {name: gestureName, hand: handLabel, confidence: 0.0});
      }
    }

    for (const name of Array.from(activeMap.keys())) {
      if (!processed.has(name)) {
        activeMap.delete(name);
        this.emitGesture(
            'gestureend', {name, hand: handLabel, confidence: 0.0});
      }
    }
  }

  private buildHandContext(
      handedness: Handedness, handLabel: HandLabel): HandContext|null {
    if (!this.user.hands) return null;
    const hand = this.user.hands.hands[handedness];
    if (!hand?.joints) return null;

    let jointCache = JOINT_TEMP_POOL.get(handLabel);
    if (!jointCache) {
      jointCache = new Map();
      JOINT_TEMP_POOL.set(handLabel, jointCache);
    }
    const joints = jointCache as JointPositions;
    joints.clear();

    for (const jointName of HAND_JOINT_NAMES) {
      const joint = hand.joints[jointName];
      if (!joint) continue;
      let vector = joints.get(jointName);
      if (!vector) {
        vector = new THREE.Vector3();
        joints.set(jointName, vector);
      }
      vector.setFromMatrixPosition(joint.matrixWorld);
    }

    if (!joints.size) return null;
    return {
      handedness,
      handLabel,
      joints,
    };
  }

  private emitGesture(type: GestureEventType, detail: GestureEventDetail) {
    const event: GestureScriptEvent = {type, detail, target: this};
    this.dispatchEvent(event);
  }
}
