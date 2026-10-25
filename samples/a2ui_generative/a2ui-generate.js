// Request text -> A2UI message stream ({name, description, messages: [...]}).
//
// Two generators produce the same artifact:
//  - generateWithAI: asks a model for A2UI JSON only (needs the xb AI facade).
//  - offlineGenerate: keyword-driven templates, so the sample runs without keys.

const CATALOG_ID = 'https://a2ui.org/specification/v0_9/catalogs/basic/catalog.json';

const SYSTEM_PROMPT = `You generate A2UI documents (a2ui.org v0.9.1, Basic Catalog).
Answer with ONLY one JSON object, no prose, no code fences:
{"name": string, "description": string, "messages": [envelope, ...]}
Each envelope is exactly {"version": "v0.9", "<type>": {...}} with one of
createSurface / updateComponents / updateDataModel.
Rules:
- createSurface: {"surfaceId": "generated", "catalogId": "${CATALOG_ID}", "sendDataModel": true}
- updateComponents carries a FLAT component list; exactly one component has
  "id": "root". Containers (Column/Row/Card/List) reference children by id:
  "children": ["id1", "id2"] or "child": "id". Buttons need "child" (a Text
  component) and "action": {"event": {"name": "...", "context": {}}}.
- Components: Text{text, variant: h1|h2|h3|body|caption}, Column/Row{children,
  justify, align}, Card{child}, List{children, direction}, Button{child,
  variant: primary|default, action}, TextField{label, value:{path}, variant:
  shortText|longText|number|obscured}, Slider{label, value:{path}, min, max},
  Image{url, description}, Divider{axis}.
- Bind dynamic content with {"path": "/json/pointer"} and add an
  updateDataModel envelope providing that data (e.g. /items array for a List
  template: "children": {"componentId": "item_tpl", "path": "/items"}).
- Keep it small: under 25 components. Use text only from the user request.`;

function envelope(type, body) {
  return {version: 'v0.9', [type]: body};
}

function text(id, value, variant = 'body') {
  return {id, component: 'Text', text: value, variant};
}

/** Parse a model reply that is a JSON object, tolerating code fences. */
function parseReply(reply) {
  const cleaned = reply.trim().replace(/^```(?:json)?\s*/i, '').replace(/```\s*$/, '').trim();
  const start = cleaned.indexOf('{');
  const end = cleaned.lastIndexOf('}');
  if (start < 0 || end < start) throw new Error('Model did not return JSON');
  return JSON.parse(cleaned.slice(start, end + 1));
}

export async function generateWithAI(request, ai) {
  const response = await ai.query({
    type: 'multiPart',
    parts: [{text: `${SYSTEM_PROMPT}\n\nUser request: ${request}`}],
  });
  const raw = typeof response === 'string' ? response : response?.text;
  if (!raw) throw new Error('AI returned an empty response');
  return parseReply(raw);
}

/** Offline fallback: a small template family covering the demo flows. */
export function offlineGenerate(request) {
  const r = request.toLowerCase();

  if (/(dashboard|stat|metric|kpi)/.test(r)) {
    return {
      name: 'Stat dashboard',
      description: 'Generated dashboard of stat cards',
      messages: [
        envelope('createSurface', {
          surfaceId: 'generated',
          catalogId: CATALOG_ID,
          sendDataModel: true,
        }),
        envelope('updateDataModel', {surfaceId: 'generated', path: '/', value: {
          items: [
            {title: 'Active users', value: '1,284'},
            {title: 'Latency', value: '42 ms'},
            {title: 'Uptime', value: '99.98%'},
          ],
          lastAction: 'none',
        }}),
        envelope('updateComponents', {
          surfaceId: 'generated',
          components: [
            {id: 'root', component: 'Column', children: ['heading', 'stats', 'actions'], gap: 12},
            text('heading', 'Operations dashboard', 'h2'),
            {
              id: 'stats',
              component: 'Row',
              children: {componentId: 'stat_card', path: '/items'},
              justify: 'spaceBetween',
            },
            {
              id: 'stat_card',
              component: 'Card',
              child: 'stat_column',
            },
            {
              id: 'stat_column',
              component: 'Column',
              children: ['stat_title', 'stat_value'],
            },
            {id: 'stat_title', component: 'Text', text: {path: 'title'}, variant: 'caption'},
            {id: 'stat_value', component: 'Text', text: {path: 'value'}, variant: 'h3'},
            {
              id: 'actions',
              component: 'Row',
              children: ['add_button', 'refresh_button'],
            },
            {
              id: 'add_button',
              component: 'Button',
              child: 'add_label',
              variant: 'primary',
              action: {event: {name: 'add_metric', context: {lastAction: {path: '/lastAction'}}}},
            },
            text('add_label', 'Add metric'),
            {
              id: 'refresh_button',
              component: 'Button',
              child: 'refresh_label',
              action: {event: {name: 'refresh', context: {}}},
            },
            text('refresh_label', 'Refresh'),
          ],
        }),
      ],
    };
  }

  if (/(form|feedback|survey|contact|input|login)/.test(r)) {
    return {
      name: 'Feedback form',
      description: 'Generated form with fields and a rating slider',
      messages: [
        envelope('createSurface', {
          surfaceId: 'generated',
          catalogId: CATALOG_ID,
          sendDataModel: true,
        }),
        envelope('updateDataModel', {surfaceId: 'generated', path: '/', value: {
          name: '',
          message: '',
          rating: 3,
        }}),
        envelope('updateComponents', {
          surfaceId: 'generated',
          components: [
            {id: 'root', component: 'Column', children: ['heading', 'name_field', 'message_field', 'rating_slider', 'submit'], gap: 12},
            text('heading', 'Send us feedback', 'h2'),
            {
              id: 'name_field',
              component: 'TextField',
              label: 'Your name',
              value: {path: '/name'},
              variant: 'shortText',
            },
            {
              id: 'message_field',
              component: 'TextField',
              label: 'Message',
              value: {path: '/message'},
              variant: 'longText',
            },
            {
              id: 'rating_slider',
              component: 'Slider',
              label: 'Rating',
              value: {path: '/rating'},
              min: 0,
              max: 5,
            },
            {
              id: 'submit',
              component: 'Button',
              child: 'submit_label',
              variant: 'primary',
              action: {
                event: {
                  name: 'submit_feedback',
                  context: {
                    name: {path: '/name'},
                    message: {path: '/message'},
                    rating: {path: '/rating'},
                  },
                },
              },
            },
            text('submit_label', 'Submit feedback'),
          ],
        }),
      ],
    };
  }

  if (/(list|card|menu|catalog|playlist|items)/.test(r)) {
    return {
      name: 'Item list',
      description: 'Generated card list with an add action',
      messages: [
        envelope('createSurface', {
          surfaceId: 'generated',
          catalogId: CATALOG_ID,
          sendDataModel: true,
        }),
        envelope('updateDataModel', {surfaceId: 'generated', path: '/', value: {
          items: [
            {title: 'Espresso', detail: 'Double shot, 18 g'},
            {title: 'Cappuccino', detail: 'Oat milk, cinnamon'},
            {title: 'Cold brew', detail: '16 h steep'},
          ],
        }}),
        envelope('updateComponents', {
          surfaceId: 'generated',
          components: [
            {id: 'root', component: 'Column', children: ['heading', 'items', 'add'], gap: 12},
            text('heading', 'Today\u2019s picks', 'h2'),
            {
              id: 'items',
              component: 'List',
              children: {componentId: 'item_card', path: '/items'},
              direction: 'vertical',
            },
            {id: 'item_card', component: 'Card', child: 'item_column'},
            {id: 'item_column', component: 'Column', children: ['item_title', 'item_detail']},
            {id: 'item_title', component: 'Text', text: {path: 'title'}, variant: 'h4'},
            {id: 'item_detail', component: 'Text', text: {path: 'detail'}, variant: 'caption'},
            {
              id: 'add',
              component: 'Button',
              child: 'add_label',
              variant: 'primary',
              action: {event: {name: 'add_item', context: {}}},
            },
            text('add_label', 'Add item'),
          ],
        }),
      ],
    };
  }

  // Default: a small intro card that echoes the request.
  return {
    name: 'Generated intro',
    description: 'Default generated panel',
    messages: [
      envelope('createSurface', {
        surfaceId: 'generated',
        catalogId: CATALOG_ID,
        sendDataModel: true,
      }),
      envelope('updateDataModel', {surfaceId: 'generated', path: '/', value: {request}}),
      envelope('updateComponents', {
        surfaceId: 'generated',
        components: [
          {id: 'root', component: 'Column', children: ['heading', 'body', 'divider', 'cta'], gap: 12},
          text('heading', 'Generated UI', 'h2'),
          {id: 'body', component: 'Text', text: {path: '/request'}, variant: 'body'},
          {id: 'divider', component: 'Divider', axis: 'horizontal'},
          {
            id: 'cta',
            component: 'Button',
            child: 'cta_label',
            variant: 'primary',
            action: {event: {name: 'primary_action', context: {request: {path: '/request'}}}},
          },
          text('cta_label', 'Looks good'),
        ],
      }),
    ],
  };
}
