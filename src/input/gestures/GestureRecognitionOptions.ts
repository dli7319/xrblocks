import {deepMerge} from '../../utils/OptionsUtils';
import {DeepPartial, DeepReadonly} from '../../utils/Types';

export type GestureProvider = 'heuristics' | 'mediapipe' | 'tfjs';

export type BuiltInGestureName =
  | 'pinch'
  | 'open-palm'
  | 'fist'
  | 'thumbs-up'
  | 'point'
  | 'spread';

export type GestureConfiguration = {
  enabled: boolean;
  /**
   * Optional override for gesture-specific score thresholds. For distance based
   * gestures this is treated as a maximum distance; for confidence based
   * gestures it is treated as a minimum score.
   */
  threshold?: number;
};

export type GestureConfigurations = Partial<
  Record<BuiltInGestureName, Partial<GestureConfiguration>>
>;

export class GestureRecognitionOptions {
  /** Master switch for the gesture recognition block. */
  enabled = false;

  /**
   * Backing provider that extracts gesture information.
   *  - 'heuristics': WebXR joint heuristics only (no external ML dependency).
   *  - 'mediapipe': MediaPipe Hands running via Web APIs / wasm.
   *  - 'tfjs': TensorFlow.js hand-pose-detection models.
   */
  provider: GestureProvider = 'heuristics';

  /**
   * Minimum confidence score to emit gesture events. Different providers map to
   * different score domains so this value is normalised to [0-1].
   */
  minimumConfidence = 0.6;

  /**
   * Optional throttle window for expensive providers.
   */
  updateIntervalMs = 33;

  /**
   * Default gesture catalogue.
   */
  gestures: Record<BuiltInGestureName, GestureConfiguration> = {
    pinch: {enabled: true, threshold: 0.025},
    'open-palm': {enabled: true},
    fist: {enabled: true},
    'thumbs-up': {enabled: true},
    point: {enabled: false},
    spread: {enabled: false, threshold: 0.04},
  };

  constructor(options?: DeepReadonly<DeepPartial<GestureRecognitionOptions>>) {
    deepMerge(this, options);
    if (options?.gestures) {
      for (const [name, config] of Object.entries(options.gestures)) {
        const gestureName = name as BuiltInGestureName;
        this.gestures[gestureName] = deepMerge(
          {...this.gestures[gestureName]},
          config
        ) as GestureConfiguration;
      }
    }
  }

  enable() {
    this.enabled = true;
    return this;
  }

  /**
   * Convenience helper to toggle specific gestures.
   */
  setGestureEnabled(name: BuiltInGestureName, enabled: boolean) {
    this.gestures[name] ??= {enabled};
    this.gestures[name].enabled = enabled;
    return this;
  }
}
