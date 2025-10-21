import * as THREE from 'three';

import {GestureDetectorMap, HandContext} from '../GestureTypes';
import {GestureConfiguration} from '../GestureRecognitionOptions';

const DEFAULT_PINCH_THRESHOLD = 0.03;
const DEFAULT_FIST_THRESHOLD = 0.06;
const DEFAULT_OPEN_PALM_THRESHOLD = 0.075;
const DEFAULT_POINT_THRESHOLD = 0.07;
const DEFAULT_SPREAD_THRESHOLD = 0.04;

export const heuristicDetectors: GestureDetectorMap = {
  'pinch': computePinch,
  'open-palm': computeOpenPalm,
  'fist': computeFist,
  'thumbs-up': computeThumbsUp,
  'point': computePoint,
  'spread': computeSpread,
};

function computePinch(context: HandContext, config: GestureConfiguration) {
  const thumb = getJoint(context, 'thumb-tip');
  const index = getJoint(context, 'index-finger-tip');
  if (!thumb || !index) return undefined;

  const threshold = config.threshold ?? DEFAULT_PINCH_THRESHOLD;
  const distance = thumb.distanceTo(index);
  const confidence =
      1 - THREE.MathUtils.clamp(distance / (threshold * 1.5), 0, 1);
  if (distance > threshold) return {confidence: confidence * 0.5};
  return {
    confidence: THREE.MathUtils.clamp(confidence, 0, 1),
    data: {distance},
  };
}

function computeOpenPalm(context: HandContext, config: GestureConfiguration) {
  const wrist = getJoint(context, 'wrist');
  if (!wrist) return undefined;

  const fingerTips = getFingerTips(context);
  if (fingerTips.length === 0) return undefined;

  const distances = fingerTips.map(tip => tip.distanceTo(wrist));
  const averageDistance =
      distances.reduce((sum, val) => sum + val, 0) / distances.length;

  const palmWidth = getPalmWidth(context);
  const baseThreshold =
      palmWidth ? palmWidth * 0.9 : DEFAULT_OPEN_PALM_THRESHOLD;
  const threshold = config.threshold ?? baseThreshold;

  const confidence = THREE.MathUtils.clamp(
      (averageDistance - threshold) / (threshold * 0.75), 0, 1);
  return {confidence, data: {averageDistance}};
}

function computeFist(context: HandContext, config: GestureConfiguration) {
  const wrist = getJoint(context, 'wrist');
  if (!wrist) return undefined;
  const fingerTips = getFingerTips(context);
  if (fingerTips.length === 0) return undefined;

  const distances = fingerTips.map(tip => tip.distanceTo(wrist));
  const averageDistance =
      distances.reduce((sum, val) => sum + val, 0) / distances.length;

  const palmWidth = getPalmWidth(context);
  const baseThreshold =
      palmWidth ? palmWidth * 0.55 : DEFAULT_FIST_THRESHOLD;
  const threshold = config.threshold ?? baseThreshold;

  const proximityScore = THREE.MathUtils.clamp(
      (threshold - averageDistance) / (threshold * 0.9), 0, 1);

  const neighbors = getAdjacentFingerDistances(context);
  const spreadReference =
      palmWidth ? palmWidth * 0.55 : DEFAULT_OPEN_PALM_THRESHOLD;
  const clusteringScore = THREE.MathUtils.clamp(
      (spreadReference - neighbors.average) / spreadReference, 0, 1);

  const confidence = (proximityScore * 0.7) + (clusteringScore * 0.3);
  return {
    confidence: THREE.MathUtils.clamp(confidence, 0, 1),
    data: {
      averageDistance,
      neighborAverage: neighbors.average,
      proximityScore,
      clusteringScore,
    },
  };
}

function computeThumbsUp(context: HandContext, config: GestureConfiguration) {
  const wrist = getJoint(context, 'wrist');
  const thumbTip = getJoint(context, 'thumb-tip');
  const fingerTips = getFingerTips(context).filter(tip => tip !== thumbTip);
  if (!wrist || !thumbTip || fingerTips.length === 0) return undefined;

  const thumbDistance = thumbTip.distanceTo(wrist);
  const palmWidth = getPalmWidth(context);
  const extendedThreshold =
      config.threshold ?? (palmWidth ? palmWidth * 0.9 : DEFAULT_OPEN_PALM_THRESHOLD * 0.9);

  const thumbDirection = new THREE.Vector3().subVectors(thumbTip, wrist);
  const thumbVerticalScore =
      THREE.MathUtils.clamp(thumbDirection.length() === 0 ?
                                0 :
                                thumbDirection.normalize().dot(
                                    getPalmNormal(context) ?? new THREE.Vector3()),
                            0, 1);

  const otherDistances = fingerTips.map(tip => tip.distanceTo(wrist));
  const averageOther =
      otherDistances.reduce((sum, val) => sum + val, 0) / otherDistances.length;
  const curledThreshold =
      palmWidth ? palmWidth * 0.55 : DEFAULT_FIST_THRESHOLD * 1.1;

  const thumbExtendedScore = THREE.MathUtils.clamp(
      (thumbDistance - extendedThreshold) / (extendedThreshold * 0.6), 0, 1);
  const othersCurledScore = THREE.MathUtils.clamp(
      (curledThreshold - averageOther) / (curledThreshold * 0.9), 0, 1);

  const confidence = (thumbExtendedScore * 0.5) +
      (othersCurledScore * 0.3) + (thumbVerticalScore * 0.2);
  return {
    confidence: THREE.MathUtils.clamp(confidence, 0, 1),
    data: {thumbDistance, averageOther, thumbVerticalScore},
  };
}

function computePoint(context: HandContext, config: GestureConfiguration) {
  const wrist = getJoint(context, 'wrist');
  const indexTip = getJoint(context, 'index-finger-tip');
  const thumbTip = getJoint(context, 'thumb-tip');
  if (!wrist || !indexTip) return undefined;

  const otherTips = getFingerTips(context)
                        .filter(tip => tip !== indexTip && tip !== thumbTip);
  if (!otherTips.length) return undefined;

  const indexDistance = indexTip.distanceTo(wrist);
  const averageOthers =
      otherTips.reduce((sum, tip) => sum + tip.distanceTo(wrist), 0) /
      otherTips.length;

  const palmWidth = getPalmWidth(context);
  const extendedThreshold =
      config.threshold ?? (palmWidth ? palmWidth * 0.9 : DEFAULT_POINT_THRESHOLD);
  const curledThreshold =
      palmWidth ? palmWidth * 0.55 : DEFAULT_FIST_THRESHOLD * 1.1;

  const indexScore = THREE.MathUtils.clamp(
      (indexDistance - extendedThreshold) / (extendedThreshold * 0.6), 0, 1);
  const othersScore = THREE.MathUtils.clamp(
      (curledThreshold - averageOthers) / (curledThreshold * 0.9), 0, 1);

  const confidence = (indexScore * 0.7) + (othersScore * 0.3);
  return {
    confidence: THREE.MathUtils.clamp(confidence, 0, 1),
    data: {indexDistance, averageOthers},
  };
}

function computeSpread(context: HandContext, config: GestureConfiguration) {
  const thumb = getJoint(context, 'thumb-tip');
  const index = getJoint(context, 'index-finger-tip');
  const middle = getJoint(context, 'middle-finger-tip');
  const ring = getJoint(context, 'ring-finger-tip');
  const pinky = getJoint(context, 'pinky-finger-tip');
  if (!thumb || !index || !middle || !ring || !pinky) return undefined;

  const pairs: [THREE.Vector3, THREE.Vector3][] = [
    [thumb, index],
    [index, middle],
    [middle, ring],
    [ring, pinky],
  ];
  const distances = pairs.map(([a, b]) => a.distanceTo(b));
  const average = distances.reduce((sum, v) => sum + v, 0) / distances.length;

  const palmWidth = getPalmWidth(context);
  const baseThreshold =
      palmWidth ? palmWidth * 0.45 : DEFAULT_SPREAD_THRESHOLD;
  const threshold = config.threshold ?? baseThreshold;

  const confidence = THREE.MathUtils.clamp(
      (average - threshold) / (threshold * 0.6), 0, 1);
  return {confidence, data: {averageDistance: average}};
}

function getJoint(context: HandContext, jointName: string) {
  return context.joints.get(jointName);
}

function getFingerTips(context: HandContext) {
  const tips: THREE.Vector3[] = [];
  const names = [
    'thumb-tip',
    'index-finger-tip',
    'middle-finger-tip',
    'ring-finger-tip',
    'pinky-finger-tip',
  ];
  for (const name of names) {
    const joint = getJoint(context, name);
    if (joint) tips.push(joint);
  }
  return tips;
}

function getPalmWidth(context: HandContext) {
  const indexBase = getJoint(context, 'index-finger-metacarpal');
  const pinkyBase = getJoint(context, 'pinky-finger-metacarpal');
  if (!indexBase || !pinkyBase) return null;
  return indexBase.distanceTo(pinkyBase);
}

function getPalmNormal(context: HandContext) {
  const wrist = getJoint(context, 'wrist');
  const indexBase = getJoint(context, 'index-finger-metacarpal');
  const pinkyBase = getJoint(context, 'pinky-finger-metacarpal');
  if (!wrist || !indexBase || !pinkyBase) return null;

  const u = new THREE.Vector3().subVectors(indexBase, wrist);
  const v = new THREE.Vector3().subVectors(pinkyBase, wrist);
  if (u.lengthSq() === 0 || v.lengthSq() === 0) return null;

  const normal = new THREE.Vector3().crossVectors(u, v);
  if (normal.lengthSq() === 0) return null;
  if (context.handLabel === 'left') normal.multiplyScalar(-1);
  return normal.normalize();
}

function getAdjacentFingerDistances(context: HandContext) {
  const index = getJoint(context, 'index-finger-tip');
  const middle = getJoint(context, 'middle-finger-tip');
  const ring = getJoint(context, 'ring-finger-tip');
  const pinky = getJoint(context, 'pinky-finger-tip');
  if (!index || !middle || !ring || !pinky) {
    return {average: Infinity};
  }
  const distances = [
    index.distanceTo(middle),
    middle.distanceTo(ring),
    ring.distanceTo(pinky),
  ];
  const average = distances.reduce((sum, value) => sum + value, 0) /
      distances.length;
  return {average};
}
