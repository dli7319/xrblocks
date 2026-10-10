// app/ui.js
/**
 * DOM wiring for the spatial-colocation HUD: dark theme, phone-friendly (big touch
 * targets). All markup lives in index.html; this module only binds handlers and pushes
 * state into elements.
 */

const MODES = ['build', 'relocalize', 'live'];
const LOG_MAX_LINES = 200;

export class UI {
  /**
   * @param {object} cb callbacks:
   *   onMode(mode), onSourceToggle(), onXrEnable(), onSaveMap(), onLoadMap(name),
   *   onExtrinsics({tx, ty, tz})
   */
  constructor(cb = {}) {
    this.cb = cb;
    this.$ = (id) => document.getElementById(id);
    this.tabs = Array.from(document.querySelectorAll('#tabs .tab'));
    this.el = {
      pillNet: this.$('pill-net'),
      pillSource: this.$('pill-source'),
      pillXr: this.$('pill-xr'),
      stKf: this.$('st-kf'),
      stLm: this.$('st-lm'),
      stMt: this.$('st-mt'),
      stFps: this.$('st-fps'),
      stInl: this.$('st-inl'),
      feedBadge: this.$('feed-badge'),
      vizBadge: this.$('viz-badge'),
      deviceList: this.$('device-list'),
      log: this.$('log'),
      mapSelect: this.$('sel-maps'),
      mapName: this.$('in-mapname'),
      btnSource: this.$('btn-source'),
      btnXr: this.$('btn-xr'),
      btnSave: this.$('btn-save'),
      btnLoad: this.$('btn-load'),
      extrStatus: this.$('extr-status'),
    };

    for (const tab of this.tabs) {
      tab.addEventListener('click', () => {
        const mode = tab.dataset.mode;
        if (MODES.includes(mode)) this.cb.onMode?.(mode);
      });
    }
    this.el.btnSource?.addEventListener('click', () =>
      this.cb.onSourceToggle?.()
    );
    this.el.btnXr?.addEventListener('click', () => this.cb.onXrEnable?.());
    this.el.btnSave?.addEventListener('click', () => this.cb.onSaveMap?.());
    this.el.btnLoad?.addEventListener('click', () => {
      const name = this.el.mapSelect?.value;
      if (name) this.cb.onLoadMap?.(name);
    });
    this.$('btn-extr')?.addEventListener('click', () => {
      this.cb.onExtrinsics?.({
        tx: parseFloat(this.$('in-tx')?.value || '0'),
        ty: parseFloat(this.$('in-ty')?.value || '0'),
        tz: parseFloat(this.$('in-tz')?.value || '0'),
      });
    });
  }

  setMode(mode) {
    for (const tab of this.tabs) {
      tab.classList.toggle('active', tab.dataset.mode === mode);
    }
  }

  setNet({status, detail, isHost, peerCount}) {
    const pill = this.el.pillNet;
    if (!pill) return;
    const map = {
      connecting: ['net …', 'warn'],
      joined: [
        `net ✓ ${peerCount ?? 0} peer${peerCount === 1 ? '' : 's'}`,
        'ok',
      ],
      host: [`net ★ host${peerCount ? ` · ${peerCount}` : ''}`, 'ok'],
      reconnecting: ['net ↻', 'warn'],
      error: ['net ✗', 'err'],
    };
    const [text, cls] = map[status] || [`net ${status}`, 'warn'];
    pill.textContent = text;
    pill.className = `pill ${cls}`;
    pill.title = `${status}${detail ? ` — ${detail}` : ''}${isHost ? ' (host)' : ''}`;
  }

  setSource(kind, note) {
    const pill = this.el.pillSource;
    if (!pill) return;
    pill.textContent =
      kind === 'synthetic' ? 'source: synthetic' : 'source: camera';
    pill.className = `pill ${kind === 'synthetic' ? 'warn' : 'ok'}`;
    pill.title = note || '';
  }

  setXr(text, level) {
    const pill = this.el.pillXr;
    if (!pill) return;
    pill.textContent = `xr: ${text}`;
    pill.className = `pill ${level || ''}`;
  }

  showXrButton(show) {
    if (this.el.btnXr) this.el.btnXr.hidden = !show;
  }

  setSourceButton(label) {
    if (this.el.btnSource) this.el.btnSource.textContent = label;
  }

  setStats({keyframes, landmarks, matches, fps, inliers} = {}) {
    if (keyframes !== undefined && this.el.stKf)
      this.el.stKf.textContent = String(keyframes);
    if (landmarks !== undefined && this.el.stLm)
      this.el.stLm.textContent = String(landmarks);
    if (matches !== undefined && this.el.stMt)
      this.el.stMt.textContent = String(matches);
    if (fps !== undefined && this.el.stFps)
      this.el.stFps.textContent = Number(fps).toFixed(1);
    if (inliers !== undefined && this.el.stInl) {
      this.el.stInl.textContent =
        inliers === null || inliers === undefined ? '—' : String(inliers);
    }
  }

  /** Devices list: "Devices relocalized to this map: [name ✓ inliers, ...]". */
  setDevices(list) {
    const ul = this.el.deviceList;
    if (!ul) return;
    ul.textContent = '';
    if (!list.length) {
      const li = document.createElement('li');
      li.className = 'empty';
      li.textContent = 'none yet';
      ul.appendChild(li);
      return;
    }
    for (const d of list) {
      const li = document.createElement('li');
      // Everyone in this list broadcasts a map-frame pose — they are localized
      // by construction (builders track their own pose, others relocalized).
      li.textContent = `${d.label || d.peerId}${d.self ? ' (you)' : ''} ✓ ${
        d.inliers ? `${d.inliers} inliers` : 'tracking'
      }`;
      li.className = d.self ? 'self' : '';
      ul.appendChild(li);
    }
  }

  setMapList(maps) {
    const sel = this.el.mapSelect;
    if (!sel) return;
    sel.textContent = '';
    const none = document.createElement('option');
    none.value = '';
    none.textContent = maps.length ? 'stored maps…' : 'no stored maps';
    sel.appendChild(none);
    for (const m of maps) {
      const opt = document.createElement('option');
      opt.value = m.name;
      const kb = Math.round((m.size || 0) / 102.4) / 10;
      opt.textContent = `${m.name} (${kb} kB)`;
      sel.appendChild(opt);
    }
  }

  get mapName() {
    return (this.el.mapName?.value || '').trim();
  }

  setBadge(text) {
    if (this.el.feedBadge) this.el.feedBadge.textContent = text || '';
    if (this.el.vizBadge) this.el.vizBadge.textContent = text || '';
  }

  setExtrinsicsStatus(text, level) {
    if (this.el.extrStatus) {
      this.el.extrStatus.textContent = text || '';
      this.el.extrStatus.className = level || '';
    }
  }

  log(text, level = '') {
    const pre = this.el.log;
    if (!pre) return;
    const t = new Date().toISOString().slice(11, 19);
    const line = document.createElement('span');
    line.className = level;
    line.textContent = `[${t}] ${text}\n`;
    pre.appendChild(line);
    while (pre.childNodes.length > LOG_MAX_LINES)
      pre.removeChild(pre.firstChild);
    pre.scrollTop = pre.scrollHeight;
    if (level === 'error') console.error(text);
  }
}
