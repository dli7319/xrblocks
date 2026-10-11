/**
 * Ring buffer of head poses keyed by wall-clock time. Video frames lag the
 * clock (capture + element latency), so a frame must be paired with the pose
 * it was CAPTURED under, not the pose at processing time — otherwise every
 * geometric constraint (mapping, calibration) sees motion-smeared pairs.
 */
export class PoseHistory {
  constructor({capacity = 256} = {}) {
    this.capacity = capacity;
    this.entries = []; // {tMs, T}, sorted ascending by tMs
  }

  /** Record a pose. Stale/out-of-order samples are dropped. */
  push(tMs, T) {
    if (!T || !Number.isFinite(tMs)) return;
    const e = this.entries;
    if (e.length && tMs <= e[e.length - 1].tMs) return;
    e.push({tMs, T});
    if (e.length > this.capacity) e.shift();
  }

  /**
   * Pose nearest to `tMs` (samples arrive at 10-60 Hz, so nearest is at most
   * half a sample off — well under a pixel of reprojection error at UI-sweep
   * speeds). Returns null when the history is empty.
   */
  sample(tMs) {
    const e = this.entries;
    if (!e.length) return null;
    let lo = 0;
    let hi = e.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (e[mid].tMs < tMs) lo = mid + 1;
      else hi = mid;
    }
    const after = e[lo];
    const before = lo > 0 ? e[lo - 1] : after;
    return (after.tMs - tMs < tMs - before.tMs ? after : before).T;
  }

  clear() {
    this.entries.length = 0;
  }
}
