import * as xb from 'xrblocks';

const GESTURE_NAMES = [
  'pinch',
  'open-palm',
  'fist',
  'thumbs-up',
  'point',
  'spread',
];

const HAND_LABELS = {
  left: 'Left Hand',
  right: 'Right Hand',
};

function formatGestureName(name) {
  return name.replace(/-/g, ' ').replace(/\b\w/g, (char) => char.toUpperCase());
}

export class GestureDashboard extends xb.Script {
  init() {
    this.container = document.getElementById('gesture-dashboard');
    if (!this.container) {
      console.warn('[GestureDashboard] Missing #gesture-dashboard container.');
      return;
    }

    this.handStates = {
      left: new Map(),
      right: new Map(),
    };
    this.rows = {
      left: new Map(),
      right: new Map(),
    };

    this.buildPanels();
    this.attachToGestureRecognition();
  }

  buildPanels() {
    for (const hand of Object.keys(HAND_LABELS)) {
      const panel = document.createElement('div');
      panel.className = 'gesture-panel';

      const title = document.createElement('h2');
      title.textContent = HAND_LABELS[hand];
      panel.appendChild(title);

      for (const gesture of GESTURE_NAMES) {
        const row = document.createElement('div');
        row.className = 'gesture-row';

        const label = document.createElement('span');
        label.className = 'gesture-label';
        label.textContent = formatGestureName(gesture);

        const value = document.createElement('span');
        value.className = 'gesture-confidence';
        value.textContent = '—';

        row.append(label, value);
        panel.appendChild(row);
        this.rows[hand].set(gesture, {row, value});
      }

      this.container.appendChild(panel);
    }
  }

  attachToGestureRecognition() {
    const gestures = xb.core.gestureRecognition;
    if (!gestures) {
      console.warn(
          '[GestureDashboard] GestureRecognition is not available yet. ' +
          'Ensure options.enableGestures() is called before xb.init().');
      return;
    }

    this.gestureRecognition = gestures;
    this._onGestureStart = this.handleGestureStart.bind(this);
    this._onGestureUpdate = this.handleGestureUpdate.bind(this);
    this._onGestureEnd = this.handleGestureEnd.bind(this);

    gestures.addEventListener('gesturestart', this._onGestureStart);
    gestures.addEventListener('gestureupdate', this._onGestureUpdate);
    gestures.addEventListener('gestureend', this._onGestureEnd);
  }

  handleGestureStart(event) {
    const {hand, name, confidence} = event.detail;
    this.handStates[hand].set(name, confidence);
    this.renderHand(hand);
  }

  handleGestureUpdate(event) {
    const {hand, name, confidence} = event.detail;
    if (!this.handStates[hand].has(name)) return;
    this.handStates[hand].set(name, confidence);
    this.renderHand(hand);
  }

  handleGestureEnd(event) {
    const {hand, name} = event.detail;
    this.handStates[hand].delete(name);
    this.renderHand(hand);
  }

  renderHand(hand) {
    for (const gesture of GESTURE_NAMES) {
      const rowElements = this.rows[hand].get(gesture);
      if (!rowElements) continue;
      const confidence = this.handStates[hand].get(gesture);

      if (confidence !== undefined) {
        rowElements.row.classList.add('active');
        rowElements.value.textContent = confidence.toFixed(2);
      } else {
        rowElements.row.classList.remove('active');
        rowElements.value.textContent = '—';
      }
    }
  }

  dispose() {
    if (!this.gestureRecognition) return;
    this.gestureRecognition.removeEventListener(
        'gesturestart', this._onGestureStart);
    this.gestureRecognition.removeEventListener(
        'gestureupdate', this._onGestureUpdate);
    this.gestureRecognition.removeEventListener(
        'gestureend', this._onGestureEnd);
  }
}
