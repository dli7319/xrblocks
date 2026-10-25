// A2UI (a2ui.org, v0.9.1 message protocol) -> XR Blocks spatial UI renderer.
//
// A2UI is a stream of envelope messages: createSurface / updateComponents /
// updateDataModel / deleteSurface. Components are a FLAT adjacency list keyed
// by id (exactly one "root"); children reference ids, and lists can bind to
// data-model arrays via templates ({componentId, path}). This renderer keeps
// the surface state and repaints the tree into an xb.UI container on change.
//
// Spec: https://a2ui.org/specification/v0.9.1-a2ui/ (schemas in the a2ui repo
// under specification/v0_9_1/json/, Basic Catalog under catalogs/basic/).

const TEXT_SIZES = {
  h1: 44,
  h2: 36,
  h3: 30,
  h4: 26,
  h5: 22,
  body: 20,
  caption: 16,
};

const JUSTIFY = {
  start: 'flex-start',
  center: 'center',
  end: 'flex-end',
  spaceBetween: 'space-between',
  spaceAround: 'space-around',
  spaceEvenly: 'space-evenly',
};

const ALIGN = {
  start: 'flex-start',
  center: 'center',
  end: 'flex-end',
  stretch: 'stretch',
};

const ICON_GLYPHS = {star: '\u2605', heart: '\u2665', check: '\u2713', warning: '\u26a0', info: '\u24d8'};

/** Resolve a DynamicValue: literal, {"path": "/json/pointer"}, or {"call": ...}. */
function resolveValue(value, model, scope) {
  if (value == null) return '';
  if (typeof value !== 'object') return value;
  if ('path' in value) {
    const path = value.path;
    const source = path.startsWith('/') ? model : scope;
    if (source == null) return '';
    const result = readPointer(source, path.startsWith('/') ? path : '/' + path);
    return result == null ? '' : result;
  }
  if ('call' in value) return ''; // catalog functions are out of scope for this sample
  return '';
}

/** Minimal JSON-Pointer reader ("" or "/" = whole document). */
function readPointer(doc, pointer) {
  if (!pointer || pointer === '/') return doc;
  let current = doc;
  for (const raw of pointer.split('/').slice(1)) {
    const key = raw.replace(/~1/g, '/').replace(/~0/g, '~');
    if (current == null) return undefined;
    current = current[key];
  }
  return current;
}

/** JSON-Pointer writer: replace-at-path (A2UI updateDataModel semantics). */
function writePointer(doc, pointer, value) {
  if (!pointer || pointer === '/') return value;
  const parts = pointer
    .split('/')
    .slice(1)
    .map((p) => p.replace(/~1/g, '/').replace(/~0/g, '~'));
  let current = doc;
  for (const part of parts.slice(0, -1)) {
    if (current[part] == null) current[part] = {};
    current = current[part];
  }
  const last = parts[parts.length - 1];
  if (value === undefined) delete current[last];
  else current[last] = value;
  return doc;
}

/** Simple Markdown subset -> plain text (the spec allows simple markdown). */
function stripMarkdown(text) {
  return String(text)
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    .replace(/\*([^*]+)\*/g, '$1')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/^#{1,6}\s+/gm, '');
}

export class A2UISurface {
  /**
   * @param {{container: any, xb: any, onAction?: (msg: object) => void,
   *          onLog?: (text: string) => void}} options
   */
  constructor({container, xb, onAction, onLog}) {
    this.container = container;
    this.xb = xb;
    this.onAction = onAction ?? (() => {});
    this.onLog = onLog ?? (() => {});
    this.components = new Map();
    this.dataModel = {};
    this.surfaceId = 'surface';
    this.sendDataModel = false;
  }

  /** Apply one A2UI envelope message ({version, <type>: {...}}). */
  apply(message) {
    if (message.createSurface) {
      const m = message.createSurface;
      this.surfaceId = m.surfaceId ?? this.surfaceId;
      this.sendDataModel = Boolean(m.sendDataModel);
      this.components.clear();
      this.dataModel = m.dataModel ?? {};
      this.render();
    } else if (message.updateComponents) {
      const m = message.updateComponents;
      for (const component of m.components ?? []) {
        this.components.set(component.id, component);
      }
      this.render();
    } else if (message.updateDataModel) {
      const m = message.updateDataModel;
      this.dataModel = writePointer(structuredClone(this.dataModel), m.path, m.value);
      this.render();
    } else if (message.deleteSurface) {
      this.components.clear();
      this.dataModel = {};
      this.container.clear?.();
    }
  }

  /** Rebuild the UI tree under the container (cheap; fine at this scale). */
  render() {
    this.container.clear?.();
    const root = this.components.get('root');
    if (!root) return; // progressive rendering: buffer until "root" arrives
    const node = this.renderComponent(root, {});
    if (node) this.container.add(node);
  }

  renderComponent(component, scope) {
    const {xb} = this;
    const model = this.dataModel;
    const style = {};
    if (component.weight != null) style.flexGrow = component.weight;

    switch (component.component) {
      case 'Column':
      case 'Row': {
        style.flexDirection = component.component === 'Row' ? 'row' : 'column';
        style.gap = 12;
        if (component.justify) style.justifyContent = JUSTIFY[component.justify] ?? component.justify;
        if (component.align) style.alignItems = ALIGN[component.align] ?? component.align;
        return new xb.UIPanel({style, children: this.renderChildren(component, scope)});
      }
      case 'List': {
        style.flexDirection = component.direction === 'horizontal' ? 'row' : 'column';
        style.gap = 12;
        return new xb.UIScrollView({
          style: {height: 320, ...style},
          children: this.renderChildren(component, scope),
        });
      }
      case 'Card':
        // A2UI Card wraps one child with elevation; UICard requires fixed
        // numeric sizes, so use a styled UIPanel for flexible widths.
        return new xb.UIPanel({
          style: {
            flexDirection: 'column',
            gap: 8,
            padding: 14,
            backgroundColor: '#3a4150',
            borderRadius: 12,
            flexShrink: 0,
          },
          children: [this.renderChild(component.child, scope)].filter(Boolean),
        });
      case 'Text': {
        const variant = component.variant ?? 'body';
        return new xb.UIText({
          text: stripMarkdown(String(resolveValue(component.text, model, scope))),
          style: {fontSize: TEXT_SIZES[variant] ?? TEXT_SIZES.body, whiteSpace: 'pre-line'},
        });
      }
      case 'Button': {
        const label = this.buttonLabel(component, scope);
        return new xb.UIButton({
          label,
          variant: component.variant === 'primary' ? 'primary' : 'default',
          onClick: () => this.dispatchAction(component),
        });
      }
      case 'TextField': {
        const input = new xb.UITextInput({
          ariaLabel: String(resolveValue(component.label, model, scope)),
          placeholder: String(resolveValue(component.label, model, scope)),
          value: String(resolveValue(component.value, model, scope)),
          multiline: component.variant === 'longText',
          style: {height: component.variant === 'longText' ? 96 : 48, fontSize: 20},
          onInput: (value) => {
            // Two-way binding: user edits write back to the surface data model.
            if (component.value?.path) {
              this.dataModel = writePointer(structuredClone(this.dataModel), component.value.path, value);
            }
          },
        });
        return input;
      }
      case 'Slider': {
        const slider = new xb.UISlider({
          ariaLabel: String(resolveValue(component.label, model, scope)),
          value: Number(resolveValue(component.value, model, scope)) || 0,
          min: component.min ?? 0,
          max: component.max ?? 100,
          style,
          onInput: (value) => {
            if (component.value?.path) {
              this.dataModel = writePointer(structuredClone(this.dataModel), component.value.path, value);
            }
          },
        });
        return slider;
      }
      case 'Image':
        return new xb.UIImage({
          src: String(resolveValue(component.url, model, scope)),
          ariaLabel: String(resolveValue(component.description, model, scope)),
          style: {height: 160, ...style},
        });
      case 'Icon': {
        const glyph = ICON_GLYPHS[component.name] ?? '\u25cf';
        return new xb.UIText({text: glyph, style: {fontSize: 28}});
      }
      case 'Divider':
        return new xb.UIPanel({
          style: {
            height: component.axis === 'vertical' ? '100%' : 2,
            width: component.axis === 'vertical' ? 2 : '100%',
            backgroundColor: '#565e6b',
            flexShrink: 0,
          },
        });
      default: {
        // Renderer contract: unknown components should surface an error rather
        // than vanish silently (spec: VALIDATION_FAILED).
        this.onLog(`Unsupported A2UI component "${component.component}" (${component.id})`);
        return new xb.UIText({
          text: `[unsupported: ${component.component}]`,
          style: {fontSize: 16, color: '#d9a441'},
        });
      }
    }
  }

  renderChildren(parent, scope) {
    const children = [];
    if (Array.isArray(parent.children)) {
      for (const id of parent.children) {
        const node = this.renderChild(id, scope);
        if (node) children.push(node);
      }
    } else if (parent.children && typeof parent.children === 'object') {
      // ChildList template: {componentId, path} - one subtree per data element,
      // with relative bindings scoped to each element.
      const items = readPointer(this.dataModel, parent.children.path);
      if (Array.isArray(items)) {
        items.forEach((item, index) => {
          const node = this.renderChild(parent.children.componentId, {...item, '@index': index}, item);
          if (node) children.push(node);
        });
      }
    }
    return children;
  }

  renderChild(id, scope) {
    const component = this.components.get(id);
    if (!component) {
      // Progressive rendering: skip references that have not arrived yet.
      return new xb.UIText({text: '\u2026', style: {fontSize: 16, color: '#8a93a3'}});
    }
    return this.renderComponent(component, scope);
  }

  buttonLabel(component, scope) {
    const child = component.child ? this.components.get(component.child) : null;
    if (child?.component === 'Text') {
      return stripMarkdown(String(resolveValue(child.text, this.dataModel, scope)));
    }
    return 'Button';
  }

  dispatchAction(component) {
    const action = component.action;
    if (!action) return;
    if (action.event) {
      const context = {};
      for (const [key, value] of Object.entries(action.event.context ?? {})) {
        context[key] = resolveValue(value, this.dataModel, {});
      }
      // client_to_server.json action message shape.
      const message = {
        version: 'v0.9.1',
        action: {
          name: action.event.name,
          surfaceId: this.surfaceId,
          sourceComponentId: component.id,
          timestamp: new Date().toISOString(),
          context,
        },
      };
      if (this.sendDataModel) {
        message.a2uiClientDataModel = {surfaces: {[this.surfaceId]: this.dataModel}};
      }
      this.onAction(message);
    } else if (action.functionCall) {
      this.onLog(`Local function call "${action.functionCall.call}" (not executed in this sample)`);
    }
  }
}
