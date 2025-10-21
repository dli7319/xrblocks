import 'xrblocks/addons/simulator/SimulatorAddons.js';

import * as xb from 'xrblocks';

import {GestureDashboard} from './GestureDashboard.js';

const options = new xb.Options();
options.enableReticles();
options.enableGestures();

options.gestures.minimumConfidence = 0.55;
options.gestures.setGestureEnabled('point', true);
options.gestures.setGestureEnabled('spread', true);

options.hands.enabled = true;
options.hands.visualization = true;
options.hands.visualizeJoints = true;
options.hands.visualizeMeshes = true;

options.simulator.defaultMode = xb.SimulatorMode.POSE;

function start() {
  xb.add(new GestureDashboard());
  xb.init(options);
}

document.addEventListener('DOMContentLoaded', () => {
  start();
});
