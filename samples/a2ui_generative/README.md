# A2UI Generative UI

Type a UI request and have the application generate new UI at runtime as an
[A2UI](https://a2ui.org) document, rendered live into the XR Blocks spatial UI
system.

This sample investigates an A2UI → XR Blocks bridge: an LLM (or the built-in
offline generator) emits an A2UI JSON document, and `a2ui-renderer.js` maps it
onto `xb.UI*` components inside the scene.

## How it works

1. Type a request in the input (e.g. "a dashboard with three stat cards", "a
   feedback form with a rating slider").
2. **Generate** builds an A2UI document:
   - with AI enabled (`?key=` / `keys.json`, see the AI facade) the request goes
     to a model that answers with A2UI JSON only;
   - without a key, a built-in offline generator produces documents from
     templates so the sample always runs.
3. `a2ui-renderer.js` renders the document into the mounted panel — text,
   buttons, cards, inputs, sliders — and wires A2UI actions back into the app
   (buttons update the document / echo to the assistant).

## Run

```bash
npm run build:sdk   # from the repo root, generates build/xrblocks.js
npm run serve       # then open /samples/a2ui_generative/
```

Use the desktop simulator ("Enter Simulator") to type via mouse or the virtual
keyboard.
