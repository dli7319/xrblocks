import * as xb from 'xrblocks';
import {Keyboard} from 'xrblocks/addons/virtualkeyboard/index.js';
import {A2UISurface} from './a2ui-renderer.js';
import {generateWithAI, offlineGenerate} from './a2ui-generate.js';

const PANEL_DISTANCE_METERS = 0.95;

/**
 * A2UI generative UI: type a UI request, generate an A2UI document (a2ui.org
 * v0.9.1 message stream), and render it live into the XR Blocks spatial UI.
 * Generated buttons send A2UI action messages back into the app, which answers
 * with updateDataModel/updateComponents messages (the updatable-document loop).
 */
class A2UIGenerative extends xb.Script {
  static dependencies = {
    ai: xb.AI,
    aiOptions: xb.AIOptions,
  };

  init({ai} = {}) {
    this.ai = ai;

    this.status = new xb.UIText({
      text: 'Describe the UI to generate, then press Generate.',
      style: {fontSize: 18},
    });
    this.request = new xb.UITextInput({
      ariaLabel: 'UI request',
      value: 'a dashboard with three stat cards',
      placeholder: 'e.g. a feedback form with a rating slider',
      style: {height: 48, fontSize: 22},
      onFocus: () => (this.keyboard.input = this.request),
      onSubmit: () => this.generate(),
    });
    this.mountain = new xb.UIScrollView({
      ariaLabel: 'Generated UI',
      style: {height: 360, gap: 12},
    });
    this.keyboard = new Keyboard({input: this.request, open: false});

    this.surface = new A2UISurface({
      container: this.mountain,
      xb,
      onAction: (message) => this.handleAction(message),
      onLog: (text) => this.setStatus(text),
    });

    const keyboardToggle = new xb.UIButton({
      label: 'Show keyboard',
      onClick: () => {
        this.keyboard.open = !this.keyboard.open;
        keyboardToggle.label = this.keyboard.open ? 'Hide keyboard' : 'Show keyboard';
        keyboardToggle.xb.preserveTextFocus = true;
      },
    });
    keyboardToggle.xb.preserveTextFocus = true;

    const children = [
      new xb.UIExitButton(),
      new xb.UIText({text: 'A2UI generative UI', style: {fontSize: 32}}),
      this.request,
      new xb.UIPanel({
        style: {flexDirection: 'row', gap: 12},
        children: [
          new xb.UIButton({label: 'Generate', onClick: () => this.generate()}),
          keyboardToggle,
        ],
      }),
      this.status,
      this.mountain,
      this.keyboard,
    ];
    this.panel = new xb.UICard({
      size: {width: 0.95, height: 'auto'},
      manipulation: true,
      edge: true,
      style: {flexDirection: 'column', gap: 12, padding: 20},
      children,
    });
    this.panel.position.set(0, 1.4, -PANEL_DISTANCE_METERS);
    this.add(this.panel);
  }

  async generate() {
    const request = (this.request.value ?? '').trim();
    if (!request) {
      this.setStatus('Type a UI request first.');
      return;
    }
    this.setStatus('Generating A2UI document\u2026');
    let doc;
    if (this.ai) {
      try {
        doc = await generateWithAI(request, this.ai);
      } catch (error) {
        console.warn('[a2ui] AI generation failed, using offline generator:', error);
        this.setStatus('AI unavailable \u2014 offline generator used.');
      }
    }
    if (!doc) doc = offlineGenerate(request);
    for (const message of doc.messages) this.surface.apply(message);
    this.setStatus(`Rendered "${doc.name}" (${doc.messages.length} A2UI messages).`);
  }

  /** Handle a client->server A2UI action message (see client_to_server.json). */
  handleAction(message) {
    console.log('[a2ui action]', message);
    const {name, context} = message.action;
    this.setStatus(`Action "${name}" with context ${JSON.stringify(context)}`);

    // Demonstrate the updatable-document loop: answer actions with new A2UI
    // messages (what an agent/backend would stream back).
    if (name === 'add_metric') {
      const items = structuredClone(this.surface.dataModel.items ?? []);
      items.push({title: `Metric ${items.length + 1}`, value: `${(items.length + 1) * 7} ms`});
      this.surface.apply({
        version: 'v0.9',
        updateDataModel: {surfaceId: this.surface.surfaceId, path: '/items', value: items},
      });
    } else if (name === 'add_item') {
      const items = structuredClone(this.surface.dataModel.items ?? []);
      items.push({title: `Item ${items.length + 1}`, detail: 'Added from an A2UI action'});
      this.surface.apply({
        version: 'v0.9',
        updateDataModel: {surfaceId: this.surface.surfaceId, path: '/items', value: items},
      });
    } else if (name === 'refresh') {
      this.surface.apply({
        version: 'v0.9',
        updateDataModel: {
          surfaceId: this.surface.surfaceId,
          path: '/lastAction',
          value: `refreshed at ${new Date().toLocaleTimeString()}`,
        },
      });
    }
  }

  setStatus(text) {
    this.status.text = text;
  }
}

const options = new xb.Options();
options.enableHands();
options.enableAI();
const script = new A2UIGenerative();
xb.add(script);
await xb.init(options);
if (new URLSearchParams(location.search).has('debug')) window.__a2uiSample = script;
