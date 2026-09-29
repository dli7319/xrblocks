# XR Circle to Digitize | EfficientSAM × KinectFusion × XR Blocks

Interactive **WebXR / XR Blocks (`v0.20.0+`)** Circle-to-Digitize demo combining **LiteRT `EfficientSAM-Ti`** segmentation ([yformer/EfficientSAM](https://github.com/yformer/EfficientSAM)) with **Object-Centric KinectFusion** (`64³` TSDF + RGB volumetric fusion and block-accelerated Marching Cubes) running 100% client-side inside a dedicated Web Worker (`build/efficientsam_worker.js` + `build/tsdf_volume.js`) at **72+ FPS**.

## Features

- **Circle Once to Seed 3D Object Volume (`main.ts` / `tsdf_volume.ts`)**:
  - Pinch in XR (`30 cm` quad) or click-drag in the XR Blocks Desktop Simulator (`1.0 m` quad) to draw a circle around any physical or virtual object.
  - Releasing the pinch runs `EfficientSAM-Ti` in the Web Worker and back-projects the masked `160×160` WebXR depth pixels (`xb.core.depth`) into world space, filtering depth percentiles (`[15th, 85th]`) to anchor a tight `64³` **ObjectTSDFVolume** around the target object.
- **Automatic Multi-View TSDF + RGB Fusion (No Re-Circling Required)**:
  - As you move your head or walk around the object, `update()` automatically streams `160×160` WebXR depth keyframes (~15 Hz) to the Worker for projective TSDF fusion and block-accelerated **Marching Cubes** surface extraction.
  - Every ~420 ms, the 3D volume's bounding box is automatically projected into the current `512×512` RGB camera view to prompt `EfficientSAM-Ti` in the background — blending RGB vertex colors and carving away background silhouettes (visual-hull space carving).
- **Pop Out 3D Turntable & `.GLB` Export**:
  - Click **Pop Out 3D** (on the Spatial `xb.UICard` or DOM HUD) to place the reconstructed vertex-colored 3D mesh on a rotating turntable in front of you, or click **Export .GLB** to download a binary `.glb` mesh via Three.js `GLTFExporter`.

## Building & Running Locally

From the repository root:

```bash
npm run dev
```

This builds the SDK and demo TypeScript files (`main.ts` -> `build/main.js`, `efficientsam_worker.ts` -> `build/efficientsam_worker.js`, and `tsdf_volume.ts` -> `build/tsdf_volume.js`) and starts the local server.

Then open:

- `http://127.0.0.1:8080/demos/efficientsam/`

## Re-exporting the LiteRT `.tflite` Models

```bash
python3 export_tflite.py \
  --repo-dir /path/to/EfficientSAM \
  --output-dir ./models \
  --img-size 512
```
