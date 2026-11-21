import { describeError, fetchCachedModel, runModel } from "xrblocks/addons/litert/index.js";
import { SAM_EMBED_DIM, SAM_EMBED_GRID, SAM_IMG_SIZE, SAM_MASK_LOW_RES, SAM_MAX_POINTS, getSam, samSerialize, snapshotToPlanarFloat32 } from "xrblocks/addons/objects3d/masks/SamMask.js";
//#region demos/yolo/desktop.ts
const MODEL_IMG_SIZE = SAM_IMG_SIZE;
const MASK_LOW_RES = SAM_MASK_LOW_RES;
const MAX_POINTS = SAM_MAX_POINTS;
const YOLOE_IMG_SIZE = 640;
const YOLOE_NUM_ANCHORS = 8400;
const YOLOE_MODEL_URL = "https://huggingface.co/wide-video/yoloe-11-Objects365-v1.0.0/resolve/main/yoloe-11s-seg-object365.onnx";
/**
* All 365 semantic object categories baked into `yoloe-11s-seg-object365.onnx`.
*/
const OBJECTS365_NAMES = [
	"Person",
	"Sneakers",
	"Chair",
	"Other Shoes",
	"Hat",
	"Car",
	"Lamp",
	"Glasses",
	"Bottle",
	"Desk",
	"Cup",
	"Street Lights",
	"Cabinet/shelf",
	"Handbag/Satchel",
	"Bracelet",
	"Plate",
	"Picture/Frame",
	"Helmet",
	"Book",
	"Gloves",
	"Storage box",
	"Boat",
	"Leather Shoes",
	"Flower",
	"Bench",
	"Potted Plant",
	"Bowl/Basin",
	"Flag",
	"Pillow",
	"Boots",
	"Vase",
	"Microphone",
	"Necklace",
	"Ring",
	"SUV",
	"Wine Glass",
	"Belt",
	"Monitor/TV",
	"Backpack",
	"Umbrella",
	"Traffic Light",
	"Speaker",
	"Watch",
	"Tie",
	"Trash bin Can",
	"Slippers",
	"Bicycle",
	"Stool",
	"Barrel/bucket",
	"Van",
	"Couch",
	"Sandals",
	"Basket",
	"Drum",
	"Pen/Pencil",
	"Bus",
	"Wild Bird",
	"High Heels",
	"Motorcycle",
	"Guitar",
	"Carpet",
	"Cell Phone",
	"Bread",
	"Camera",
	"Canned",
	"Truck",
	"Traffic cone",
	"Cymbal",
	"Lifesaver",
	"Towel",
	"Stuffed Toy",
	"Candle",
	"Sailboat",
	"Laptop",
	"Awning",
	"Bed",
	"Faucet",
	"Tent",
	"Horse",
	"Mirror",
	"Power outlet",
	"Sink",
	"Apple",
	"Air Conditioner",
	"Knife",
	"Hockey Stick",
	"Paddle",
	"Pickup Truck",
	"Fork",
	"Traffic Sign",
	"Balloon",
	"Tripod",
	"Dog",
	"Spoon",
	"Clock",
	"Pot",
	"Cow",
	"Cake",
	"Dining Table",
	"Sheep",
	"Hanger",
	"Blackboard/Whiteboard",
	"Napkin",
	"Other Fish",
	"Orange/Tangerine",
	"Toiletry",
	"Keyboard",
	"Tomato",
	"Lantern",
	"Machinery Vehicle",
	"Fan",
	"Green Vegetables",
	"Banana",
	"Baseball Glove",
	"Airplane",
	"Mouse",
	"Train",
	"Pumpkin",
	"Soccer",
	"Skiboard",
	"Luggage",
	"Nightstand",
	"Tea pot",
	"Telephone",
	"Trolley",
	"Head Phone",
	"Sports Car",
	"Stop Sign",
	"Dessert",
	"Scooter",
	"Stroller",
	"Crane",
	"Remote",
	"Refrigerator",
	"Oven",
	"Lemon",
	"Duck",
	"Baseball Bat",
	"Surveillance Camera",
	"Cat",
	"Jug",
	"Broccoli",
	"Piano",
	"Pizza",
	"Elephant",
	"Skateboard",
	"Surfboard",
	"Gun",
	"Skating and Skiing shoes",
	"Gas stove",
	"Donut",
	"Bow Tie",
	"Carrot",
	"Toilet",
	"Kite",
	"Strawberry",
	"Other Balls",
	"Shovel",
	"Pepper",
	"Computer Box",
	"Toilet Paper",
	"Cleaning Products",
	"Chopsticks",
	"Microwave",
	"Pigeon",
	"Baseball",
	"Cutting/chopping Board",
	"Coffee Table",
	"Side Table",
	"Scissors",
	"Marker",
	"Pie",
	"Ladder",
	"Snowboard",
	"Cookies",
	"Radiator",
	"Fire Hydrant",
	"Basketball",
	"Zebra",
	"Grape",
	"Giraffe",
	"Potato",
	"Sausage",
	"Tricycle",
	"Violin",
	"Egg",
	"Fire Extinguisher",
	"Candy",
	"Fire Truck",
	"Billiards",
	"Converter",
	"Bathtub",
	"Wheelchair",
	"Golf Club",
	"Briefcase",
	"Cucumber",
	"Cigar/Cigarette",
	"Paint Brush",
	"Pear",
	"Heavy Truck",
	"Hamburger",
	"Extractor",
	"Extension Cord",
	"Tong",
	"Tennis Racket",
	"Folder",
	"American Football",
	"earphone",
	"Mask",
	"Kettle",
	"Tennis",
	"Ship",
	"Swing",
	"Coffee Machine",
	"Slide",
	"Carriage",
	"Onion",
	"Green beans",
	"Projector",
	"Frisbee",
	"Washing Machine/Drying Machine",
	"Chicken",
	"Printer",
	"Watermelon",
	"Saxophone",
	"Tissue",
	"Toothbrush",
	"Ice cream",
	"Hot-air balloon",
	"Cello",
	"French Fries",
	"Scale",
	"Trophy",
	"Cabbage",
	"Hot dog",
	"Blender",
	"Peach",
	"Rice",
	"Wallet/Purse",
	"Volleyball",
	"Deer",
	"Goose",
	"Tape",
	"Tablet",
	"Cosmetics",
	"Trumpet",
	"Pineapple",
	"Golf Ball",
	"Ambulance",
	"Parking meter",
	"Mango",
	"Key",
	"Hurdle",
	"Fishing Rod",
	"Medal",
	"Flute",
	"Brush",
	"Penguin",
	"Megaphone",
	"Corn",
	"Lettuce",
	"Garlic",
	"Swan",
	"Helicopter",
	"Green Onion",
	"Sandwich",
	"Nuts",
	"Speed Limit Sign",
	"Induction Cooker",
	"Broom",
	"Trombone",
	"Plum",
	"Rickshaw",
	"Goldfish",
	"Kiwi fruit",
	"Router/modem",
	"Poker Card",
	"Toaster",
	"Shrimp",
	"Sushi",
	"Cheese",
	"Notepaper",
	"Cherry",
	"Pliers",
	"CD",
	"Pasta",
	"Hammer",
	"Cue",
	"Avocado",
	"Hamimelon",
	"Flask",
	"Mushroom",
	"Screwdriver",
	"Soap",
	"Recorder",
	"Bear",
	"Eggplant",
	"Board Eraser",
	"Coconut",
	"Tape Measure/Ruler",
	"Pig",
	"Showerhead",
	"Globe",
	"Chips",
	"Steak",
	"Crosswalk Sign",
	"Stapler",
	"Camel",
	"Formula 1",
	"Pomegranate",
	"Dishwasher",
	"Crab",
	"Hoverboard",
	"Meat ball",
	"Rice Cooker",
	"Tuba",
	"Calculator",
	"Papaya",
	"Antelope",
	"Parrot",
	"Seal",
	"Butterfly",
	"Dumbbell",
	"Donkey",
	"Lion",
	"Urinal",
	"Dolphin",
	"Electric Drill",
	"Hair Dryer",
	"Egg tart",
	"Jellyfish",
	"Treadmill",
	"Lighter",
	"Grapefruit",
	"Game board",
	"Mop",
	"Radish",
	"Baozi",
	"Target",
	"French",
	"Spring Rolls",
	"Monkey",
	"Rabbit",
	"Pencil Case",
	"Yak",
	"Red Cabbage",
	"Binoculars",
	"Asparagus",
	"Barbell",
	"Scallop",
	"Noddles",
	"Comb",
	"Dumpling",
	"Oyster",
	"Table Tennis paddle",
	"Cosmetics Brush/Eyeliner Pencil",
	"Chainsaw",
	"Eraser",
	"Lobster",
	"Durian",
	"Okra",
	"Lipstick",
	"Cosmetics Mirror",
	"Curling",
	"Table Tennis"
];
const PRESET_URLS = {
	livingroom: "../../samples/advanced/ballpit/textures/background.webp",
	dogs: "https://raw.githubusercontent.com/yformer/EfficientSAM/main/figs/examples/dogs.jpg",
	ballpit: "../../assets/ballpit-demo.webp"
};
var EfficientSamDesktopApp = class {
	samModels = null;
	samReady = false;
	yoloeSession = null;
	yoloeProvider = "wasm";
	yoloeReady = false;
	lastYoloeMs = 0;
	isSegmenting = false;
	pendingRerun = false;
	cachedEmbeddings = null;
	currentImageVersion = 0;
	encodedImageVersion = -1;
	yoloeImageVersion = -1;
	lastEncoderMs = 0;
	rawYoloeDetections = [];
	activeYoloeDetections = [];
	hoveredDetection = null;
	yoloeConfThreshold = .2;
	showAllYoloeBoxes = true;
	hoverFallbackToSam = true;
	mode = "hover";
	candidateChoice = "auto";
	activePointLabel = 1;
	includeBboxCenter = true;
	fitContain = false;
	showMask = true;
	showOutline = true;
	showPrompt = true;
	maskOpacity = .8;
	outlineSmoothness = 7;
	outlinePadding = 17;
	loadedSourceImage = null;
	webcamVideo = null;
	isVideoMode = false;
	videoAnimationFrameId = null;
	activeSmoothBox = null;
	activeTargetBox = null;
	fastTracker = null;
	activeAngle = 0;
	cameraFacingMode = "user";
	baseCanvas;
	baseCtx;
	baseImageData;
	maskCanvas;
	maskCtx;
	hasMask = false;
	manualContourProfile = void 0;
	animatedOutlinePoints = null;
	outlineAnimFrameId = null;
	displayCanvas;
	displayCtx;
	cutoutCanvas;
	cutoutCtx;
	lastCutoutCropCanvas = null;
	hoverPoint = null;
	hoverLocked = false;
	isDragging = false;
	dragStart = null;
	currentBox = null;
	circlePath = [];
	circleDerivedBox = null;
	promptPoints = [];
	statusBanner;
	accelBadge;
	yoloeBadge;
	stageHint;
	detectedChipsEl;
	metricTotal;
	metricSplit;
	metricIou;
	metricArea;
	promptDebug;
	pointsCountEl;
	undoPtBtn;
	downloadCutoutBtn;
	constructor() {
		this.baseCanvas = document.createElement("canvas");
		this.baseCanvas.width = MODEL_IMG_SIZE;
		this.baseCanvas.height = MODEL_IMG_SIZE;
		this.baseCtx = this.baseCanvas.getContext("2d", { willReadFrequently: true });
		this.baseImageData = this.baseCtx.createImageData(MODEL_IMG_SIZE, MODEL_IMG_SIZE);
		this.maskCanvas = document.createElement("canvas");
		this.maskCanvas.width = MODEL_IMG_SIZE;
		this.maskCanvas.height = MODEL_IMG_SIZE;
		this.maskCtx = this.maskCanvas.getContext("2d");
		this.displayCanvas = document.getElementById("interactive-canvas");
		this.displayCtx = this.displayCanvas.getContext("2d");
		this.cutoutCanvas = document.getElementById("cutout-canvas");
		this.cutoutCtx = this.cutoutCanvas.getContext("2d");
		this.statusBanner = document.getElementById("status-banner");
		this.accelBadge = document.getElementById("accel-badge");
		this.yoloeBadge = document.getElementById("yoloe-badge");
		this.stageHint = document.getElementById("stage-hint");
		this.detectedChipsEl = document.getElementById("detected-chips");
		this.metricTotal = document.getElementById("metric-total");
		this.metricSplit = document.getElementById("metric-split");
		this.metricIou = document.getElementById("metric-iou");
		this.metricArea = document.getElementById("metric-area");
		this.promptDebug = document.getElementById("prompt-debug");
		this.pointsCountEl = document.getElementById("points-count");
		this.undoPtBtn = document.getElementById("undo-pt-btn");
		this.downloadCutoutBtn = document.getElementById("download-cutout-btn");
		this.bindUiEvents();
		this.clearCutoutPreview();
		this.loadPresetImage("livingroom");
		this.initAllModels();
	}
	async initAllModels() {
		await Promise.all([this.initSamModels(), this.initYoloeModel()]);
		await this.analyzeCurrentImage("initial scene");
	}
	async initSamModels() {
		try {
			const t0 = performance.now();
			const models = await getSam();
			this.samModels = models;
			this.samReady = true;
			const compileTimeMs = performance.now() - t0;
			const accelLabel = models.encoderAccelerator === "webgpu" ? models.decoderAccelerator === "webgpu" ? "WebGPU" : "WebGPU+WASM" : "WASM";
			this.accelBadge.textContent = `SAM-Ti: ${accelLabel} (${compileTimeMs.toFixed(0)} ms)`;
		} catch (err) {
			const msg = describeError(err);
			console.error("Failed to initialize EfficientSAM-Ti:", err);
			this.accelBadge.textContent = "SAM-Ti: Error";
			this.setStatus(`<strong>SAM Error:</strong> ${msg}`);
		}
	}
	async initYoloeModel() {
		try {
			const t0 = performance.now();
			const ort = window.ort;
			if (!ort) throw new Error("onnxruntime-web (window.ort) failed to load.");
			ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";
			ort.env.logLevel = "error";
			this.yoloeBadge.textContent = "YOLOE-11s: Caching 38 MB ONNX...";
			const modelBytes = await fetchCachedModel(YOLOE_MODEL_URL);
			this.yoloeBadge.textContent = "YOLOE-11s: Compiling...";
			let session = null;
			let provider = "WASM";
			for (const ep of ["webgpu", "wasm"]) try {
				session = await ort.InferenceSession.create(modelBytes, { executionProviders: [ep] });
				provider = ep === "webgpu" ? "WebGPU" : "WASM";
				break;
			} catch (err) {
				console.warn(`YOLOE-11s ${ep} init fell back:`, err);
			}
			if (!session) throw new Error("Could not create YOLOE-11s ONNX session.");
			this.yoloeSession = session;
			this.yoloeProvider = provider;
			this.yoloeReady = true;
			const initMs = performance.now() - t0;
			this.yoloeBadge.textContent = `YOLOE-11s (365 cls): ${provider} (${initMs.toFixed(0)} ms)`;
		} catch (err) {
			console.error("Failed to initialize YOLOE-11s:", err);
			this.yoloeBadge.textContent = "YOLOE-11s: Unavailable";
		}
	}
	/**
	* Runs both EfficientSAM-Ti image encoding and YOLOE-11s semantic object detection
	* on the current 512x512 image, then precomputes SAM masks for the top YOLOE detections.
	*/
	async analyzeCurrentImage(label) {
		const targetVersion = this.currentImageVersion;
		try {
			this.setStatus(`Analyzing <strong>${label}</strong> with <strong>YOLOE-11s (Objects365)</strong> + <strong>EfficientSAM-Ti</strong>...`);
			await Promise.all([this.samReady ? this.ensureImageEncoded() : Promise.resolve(null), this.yoloeReady ? this.runYoloeDetectionOnCurrentImage() : Promise.resolve()]);
			if (targetVersion !== this.currentImageVersion) return;
			await this.filterAndPrecomputeYoloeMasks(targetVersion);
			if (targetVersion !== this.currentImageVersion) return;
			const count = this.activeYoloeDetections.length;
			this.setStatus(`<strong>Ready!</strong> YOLOE-11s found <strong>${count} semantic objects</strong> (${this.lastYoloeMs.toFixed(0)} ms) · SAM encoded (${this.lastEncoderMs.toFixed(0)} ms). Hover any object!`);
			this.renderCanvas();
		} catch (err) {
			this.setStatus(`<strong>Analysis Error:</strong> ${describeError(err)}`);
		}
	}
	async ensureImageEncoded() {
		if (this.cachedEmbeddings && this.encodedImageVersion === this.currentImageVersion) return this.cachedEmbeddings;
		if (!this.samModels) this.samModels = await getSam();
		const targetVersion = this.currentImageVersion;
		const snapshot = this.baseImageData;
		const { encoder } = this.samModels;
		return samSerialize(async () => {
			const tEnc0 = performance.now();
			const inputFloat32 = snapshotToPlanarFloat32(snapshot, MODEL_IMG_SIZE);
			const [embOut] = await runModel(encoder, [{
				data: inputFloat32,
				shape: [
					1,
					3,
					MODEL_IMG_SIZE,
					MODEL_IMG_SIZE
				]
			}]);
			const embeddings = embOut instanceof Float32Array ? new Float32Array(embOut) : new Float32Array(embOut.buffer, embOut.byteOffset, embOut.byteLength / 4);
			this.lastEncoderMs = performance.now() - tEnc0;
			this.cachedEmbeddings = embeddings;
			this.encodedImageVersion = targetVersion;
			return embeddings;
		});
	}
	/**
	* Runs YOLOE-11s (`[1, 3, 640, 640]` -> `[1, 369, 8400]`) and performs NMS across the
	* 365 Objects365 semantic classes.
	*/
	async runYoloeDetectionOnCurrentImage() {
		if (!this.yoloeSession) return;
		const ort = window.ort;
		if (!ort) return;
		const targetVersion = this.currentImageVersion;
		const t0 = performance.now();
		const input640 = snapshotToPlanarFloat32(this.baseImageData, YOLOE_IMG_SIZE);
		const inputTensor = new ort.Tensor("float32", input640, [
			1,
			3,
			YOLOE_IMG_SIZE,
			YOLOE_IMG_SIZE
		]);
		let outputs;
		try {
			outputs = await this.yoloeSession.run({ images: inputTensor });
		} catch (err) {
			if (this.yoloeProvider === "WebGPU") {
				console.warn("YOLOE WebGPU run failed, falling back to WASM:", err);
				const modelBytes = await fetchCachedModel(YOLOE_MODEL_URL);
				this.yoloeSession = await ort.InferenceSession.create(modelBytes, { executionProviders: ["wasm"] });
				this.yoloeProvider = "WASM";
				this.yoloeBadge.textContent = "YOLOE-11s (365 cls): WASM";
				outputs = await this.yoloeSession.run({ images: inputTensor });
			} else throw err;
		}
		if (targetVersion !== this.currentImageVersion) return;
		const outTensor = outputs.output0 ?? Object.values(outputs)[0];
		const data = outTensor.data;
		const dims = outTensor.dims;
		const numRows = dims[1] ?? 369;
		const numAnchors = dims[2] ?? YOLOE_NUM_ANCHORS;
		const numClasses = Math.min(OBJECTS365_NAMES.length, numRows - 4);
		const scaleTo512 = MODEL_IMG_SIZE / YOLOE_IMG_SIZE;
		const minRawScore = .1;
		const candidates = [];
		for (let a = 0; a < numAnchors; a++) {
			let bestClass = 0;
			let bestScore = -1;
			for (let c = 0; c < numClasses; c++) {
				const score = data[(4 + c) * numAnchors + a];
				if (score > bestScore) {
					bestScore = score;
					bestClass = c;
				}
			}
			if (bestScore < minRawScore) continue;
			const cx = data[0 * numAnchors + a] * scaleTo512;
			const cy = data[1 * numAnchors + a] * scaleTo512;
			const w = data[2 * numAnchors + a] * scaleTo512;
			const h = data[3 * numAnchors + a] * scaleTo512;
			const x1 = Math.max(0, Math.min(MODEL_IMG_SIZE, cx - w * .5));
			const y1 = Math.max(0, Math.min(MODEL_IMG_SIZE, cy - h * .5));
			const x2 = Math.max(0, Math.min(MODEL_IMG_SIZE, cx + w * .5));
			const y2 = Math.max(0, Math.min(MODEL_IMG_SIZE, cy + h * .5));
			const bw = x2 - x1;
			const bh = y2 - y1;
			if (bw < 6 || bh < 6) continue;
			candidates.push({
				id: 0,
				classId: bestClass,
				className: OBJECTS365_NAMES[bestClass] || `Class ${bestClass}`,
				score: bestScore,
				box: {
					x1,
					y1,
					x2,
					y2
				},
				boxArea: bw * bh
			});
		}
		candidates.sort((a, b) => b.score - a.score);
		const nmsDetections = [];
		for (const cand of candidates) {
			let suppressed = false;
			for (const kept of nmsDetections) {
				const iou = this.computeBoxIou(cand.box, kept.box);
				if (cand.classId === kept.classId && iou > .45 || iou > .75) {
					suppressed = true;
					break;
				}
			}
			if (!suppressed) {
				cand.id = nmsDetections.length + 1;
				nmsDetections.push(cand);
				if (nmsDetections.length >= 35) break;
			}
		}
		this.lastYoloeMs = performance.now() - t0;
		this.rawYoloeDetections = nmsDetections;
		this.yoloeImageVersion = targetVersion;
	}
	computeBoxIou(a, b) {
		const ix1 = Math.max(a.x1, b.x1);
		const iy1 = Math.max(a.y1, b.y1);
		const ix2 = Math.min(a.x2, b.x2);
		const iy2 = Math.min(a.y2, b.y2);
		const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
		if (inter <= 0) return 0;
		const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
		const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
		return inter / Math.max(1e-5, areaA + areaB - inter);
	}
	updateObjectTracking() {
		if (!this.hoveredDetection) return;
		const prev = this.hoveredDetection;
		let bestMatch = null;
		let bestScore = -Infinity;
		const refBox = this.fastTracker?.lastBox ?? prev.box;
		const refCx = (refBox.x1 + refBox.x2) * .5;
		const refCy = (refBox.y1 + refBox.y2) * .5;
		for (const det of this.activeYoloeDetections) {
			const classMatch = det.classId === prev.classId || det.className === prev.className ? 5 : 0;
			const iou = this.computeBoxIou(det.box, refBox);
			const detCx = (det.box.x1 + det.box.x2) * .5;
			const detCy = (det.box.y1 + det.box.y2) * .5;
			const dist = Math.hypot(detCx - refCx, detCy - refCy);
			const distScore = Math.max(0, 1 - dist / 250);
			const score = classMatch + iou * 6 + distScore * 3;
			if (score > bestScore) {
				bestScore = score;
				bestMatch = det;
			}
		}
		if (bestMatch && bestScore > .3) {
			if (prev.smoothBox) bestMatch.smoothBox = {
				x1: prev.smoothBox.x1,
				y1: prev.smoothBox.y1,
				x2: prev.smoothBox.x2,
				y2: prev.smoothBox.y2
			};
			if (!bestMatch.overlayData && prev.overlayData) {
				bestMatch.overlayData = prev.overlayData;
				bestMatch.maskBox = prev.maskBox;
				bestMatch.contourProfile = prev.contourProfile;
				bestMatch.binaryMask = prev.binaryMask;
				bestMatch.needsSamUpdate = true;
			}
			this.hoveredDetection = bestMatch;
			if (bestMatch.overlayData) {
				this.maskCtx.putImageData(bestMatch.overlayData, 0, 0);
				this.hasMask = true;
			}
			if (this.fastTracker) {
				const fuseAlpha = .4;
				const oldBox = this.fastTracker.lastBox;
				const oldW = oldBox.x2 - oldBox.x1;
				const oldH = oldBox.y2 - oldBox.y1;
				const newW = bestMatch.box.x2 - bestMatch.box.x1;
				const newH = bestMatch.box.y2 - bestMatch.box.y1;
				const blendedW = oldW * .6 + newW * fuseAlpha;
				const blendedH = oldH * .6 + newH * fuseAlpha;
				const oldCx = (oldBox.x1 + oldBox.x2) * .5;
				const oldCy = (oldBox.y1 + oldBox.y2) * .5;
				const newCx = (bestMatch.box.x1 + bestMatch.box.x2) * .5;
				const newCy = (bestMatch.box.y1 + bestMatch.box.y2) * .5;
				const blendedCx = oldCx * .6 + newCx * fuseAlpha;
				const blendedCy = oldCy * .6 + newCy * fuseAlpha;
				this.fastTracker.lastBox = {
					x1: blendedCx - blendedW * .5,
					y1: blendedCy - blendedH * .5,
					x2: blendedCx + blendedW * .5,
					y2: blendedCy + blendedH * .5
				};
			} else this.initFastTracker(bestMatch);
		}
	}
	getSmoothBox(det) {
		if (!det.smoothBox) det.smoothBox = {
			x1: det.box.x1,
			y1: det.box.y1,
			x2: det.box.x2,
			y2: det.box.y2
		};
		else {
			const maxStep = 8;
			det.smoothBox.x1 += Math.max(-8, Math.min(maxStep, det.box.x1 - det.smoothBox.x1));
			det.smoothBox.y1 += Math.max(-8, Math.min(maxStep, det.box.y1 - det.smoothBox.y1));
			det.smoothBox.x2 += Math.max(-8, Math.min(maxStep, det.box.x2 - det.smoothBox.x2));
			det.smoothBox.y2 += Math.max(-8, Math.min(maxStep, det.box.y2 - det.smoothBox.y2));
		}
		return det.smoothBox;
	}
	/**
	* Builds a 32x32 foreground weight mask for rotational template matching
	* using the SAM binary mask (or an inner elliptical region if SAM isn't ready yet).
	*/
	buildTrackerFgMask(det, box) {
		const fg = /* @__PURE__ */ new Uint8Array(1024);
		const bw = Math.max(8, box.x2 - box.x1);
		const bh = Math.max(8, box.y2 - box.y1);
		let count = 0;
		if (det.binaryMask) for (let ty = 2; ty < 30; ty++) {
			const sy = Math.min(MODEL_IMG_SIZE - 1, Math.max(0, Math.floor(box.y1 + (ty + .5) / 32 * bh)));
			for (let tx = 2; tx < 30; tx++) {
				const sx = Math.min(MODEL_IMG_SIZE - 1, Math.max(0, Math.floor(box.x1 + (tx + .5) / 32 * bw)));
				if (det.binaryMask[sy * MODEL_IMG_SIZE + sx]) {
					fg[ty * 32 + tx] = 1;
					count++;
				}
			}
		}
		if (count < 24) for (let ty = 3; ty < 29; ty++) {
			const ny = (ty - 15.5) / 12.5;
			for (let tx = 3; tx < 29; tx++) {
				const nx = (tx - 15.5) / 12.5;
				if (nx * nx + ny * ny <= 1) fg[ty * 32 + tx] = 1;
			}
		}
		return fg;
	}
	/**
	* Estimates object rotation angle (in radians) relative to `refTemplate`
	* by evaluating foreground-masked SAD over candidate angles in [-25°, +25°].
	* Strong regularization toward 0° prevents random drift/jitter when upright.
	*/
	estimateRotationAngle(curGray, refGray, fgMask) {
		const evalAngleSad = (theta) => {
			const cosT = Math.cos(theta);
			const sinT = Math.sin(theta);
			let diffSum = 0;
			let validCount = 0;
			for (let y = 3; y < 29; y++) {
				const dy = y - 15.5;
				const row = y * 32;
				for (let x = 3; x < 29; x++) {
					const idx = row + x;
					if (!fgMask[idx]) continue;
					const dx = x - 15.5;
					const rx = 15.5 + dx * cosT - dy * sinT;
					const ry = 15.5 + dx * sinT + dy * cosT;
					const ix = rx | 0;
					const iy = ry | 0;
					if (ix < 1 || ix >= 30 || iy < 1 || iy >= 30) continue;
					const fx = rx - ix;
					const fy = ry - iy;
					const r0 = iy * 32 + ix;
					const r1 = r0 + 32;
					const d = (1 - fy) * ((1 - fx) * curGray[r0] + fx * curGray[r0 + 1]) + fy * ((1 - fx) * curGray[r1] + fx * curGray[r1 + 1]) - refGray[idx];
					diffSum += d < 0 ? -d : d;
					validCount++;
				}
			}
			return validCount >= 16 ? diffSum / validCount : Infinity;
		};
		const sad0 = evalAngleSad(0);
		if (!Number.isFinite(sad0)) return 0;
		let bestTheta = 0;
		let bestCost = sad0;
		const stepRad = 3.5 * Math.PI / 180;
		for (let k = -7; k <= 7; k++) {
			if (k === 0) continue;
			const theta = k * stepRad;
			const cost = evalAngleSad(theta) + (sad0 * (.065 + .16 * Math.abs(theta)) + sad0 * .05 * Math.abs(theta - this.activeAngle));
			if (cost < bestCost) {
				bestCost = cost;
				bestTheta = theta;
			}
		}
		if (Math.abs(bestTheta) < .045) return 0;
		return bestTheta;
	}
	initFastTracker(det) {
		const box = det.box;
		const x1 = Math.max(0, Math.floor(box.x1));
		const y1 = Math.max(0, Math.floor(box.y1));
		const x2 = Math.min(MODEL_IMG_SIZE, Math.ceil(box.x2));
		const y2 = Math.min(MODEL_IMG_SIZE, Math.ceil(box.y2));
		const w = Math.max(8, x2 - x1);
		const h = Math.max(8, y2 - y1);
		const tc = document.createElement("canvas");
		tc.width = 32;
		tc.height = 32;
		const tctx = tc.getContext("2d");
		tctx.drawImage(this.baseCanvas, x1, y1, w, h, 0, 0, 32, 32);
		const imgData = tctx.getImageData(0, 0, 32, 32);
		this.activeAngle = 0;
		const gray = /* @__PURE__ */ new Uint8ClampedArray(1024);
		for (let i = 0; i < 1024; i++) {
			const p = i * 4;
			gray[i] = imgData.data[p] * .299 + imgData.data[p + 1] * .587 + imgData.data[p + 2] * .114 | 0;
		}
		const fgMask = this.buildTrackerFgMask(det, box);
		this.fastTracker = {
			templateData: gray,
			refTemplate: new Uint8ClampedArray(gray),
			fgMask,
			lastBox: {
				x1: box.x1,
				y1: box.y1,
				x2: box.x2,
				y2: box.y2
			}
		};
	}
	/**
	* Synchronizes the rotational reference template and foreground mask
	* at the exact moment a new SAM mask is applied to `maskCanvas`.
	*/
	syncTrackerReferenceWithSamMask(det) {
		if (!this.fastTracker) {
			this.initFastTracker(det);
			return;
		}
		const box = this.fastTracker.lastBox;
		const x1 = Math.max(0, Math.floor(box.x1));
		const y1 = Math.max(0, Math.floor(box.y1));
		const x2 = Math.min(MODEL_IMG_SIZE, Math.ceil(box.x2));
		const y2 = Math.min(MODEL_IMG_SIZE, Math.ceil(box.y2));
		const w = Math.max(8, x2 - x1);
		const h = Math.max(8, y2 - y1);
		const tc = document.createElement("canvas");
		tc.width = 32;
		tc.height = 32;
		const tctx = tc.getContext("2d");
		tctx.drawImage(this.baseCanvas, x1, y1, w, h, 0, 0, 32, 32);
		const data = tctx.getImageData(0, 0, 32, 32).data;
		for (let i = 0; i < 1024; i++) {
			const p = i * 4;
			const g = data[p] * .299 + data[p + 1] * .587 + data[p + 2] * .114 | 0;
			this.fastTracker.refTemplate[i] = g;
			this.fastTracker.templateData[i] = g;
		}
		this.fastTracker.fgMask = this.buildTrackerFgMask(det, det.maskBox ?? det.box);
		this.activeAngle = 0;
	}
	updateFastTracker(det) {
		if (!this.fastTracker) return null;
		const tData = this.fastTracker.templateData;
		const curW = MODEL_IMG_SIZE;
		const curH = MODEL_IMG_SIZE;
		const oldBox = this.fastTracker.lastBox;
		const oldW = Math.max(8, oldBox.x2 - oldBox.x1);
		const oldH = Math.max(8, oldBox.y2 - oldBox.y1);
		const searchRange = 12;
		const step = 2;
		let bestDx = 0;
		let bestDy = 0;
		let minDiff = Infinity;
		const tc = document.createElement("canvas");
		tc.width = 32;
		tc.height = 32;
		const tctx = tc.getContext("2d");
		for (let dy = -12; dy <= searchRange; dy += step) for (let dx = -12; dx <= searchRange; dx += step) {
			const nx1 = Math.max(0, Math.min(curW - oldW, oldBox.x1 + dx));
			const ny1 = Math.max(0, Math.min(curH - oldH, oldBox.y1 + dy));
			tctx.clearRect(0, 0, 32, 32);
			tctx.drawImage(this.baseCanvas, nx1, ny1, oldW, oldH, 0, 0, 32, 32);
			const candData = tctx.getImageData(0, 0, 32, 32).data;
			let diff = 0;
			for (let i = 0; i < 1024; i++) {
				const p = i * 4;
				const d = (candData[p] * .299 + candData[p + 1] * .587 + candData[p + 2] * .114 | 0) - tData[i];
				diff += d < 0 ? -d : d;
				if (diff >= minDiff) break;
			}
			if (diff < minDiff) {
				minDiff = diff;
				bestDx = dx;
				bestDy = dy;
			}
		}
		const targetW = Math.max(8, det.box.x2 - det.box.x1);
		const targetH = Math.max(8, det.box.y2 - det.box.y1);
		const sizeAlpha = .3;
		const newW = oldW * .7 + targetW * sizeAlpha;
		const newH = oldH * .7 + targetH * sizeAlpha;
		const finalX1 = Math.max(0, Math.min(curW - newW, oldBox.x1 + bestDx));
		const finalY1 = Math.max(0, Math.min(curH - newH, oldBox.y1 + bestDy));
		tctx.clearRect(0, 0, 32, 32);
		tctx.drawImage(this.baseCanvas, finalX1, finalY1, newW, newH, 0, 0, 32, 32);
		const adaptedData = tctx.getImageData(0, 0, 32, 32).data;
		const curGray = /* @__PURE__ */ new Uint8ClampedArray(1024);
		const adaptRate = .1;
		for (let i = 0; i < 1024; i++) {
			const p = i * 4;
			const g = adaptedData[p] * .299 + adaptedData[p + 1] * .587 + adaptedData[p + 2] * .114 | 0;
			curGray[i] = g;
			tData[i] = tData[i] * .9 + g * adaptRate | 0;
		}
		const targetAngle = this.estimateRotationAngle(curGray, this.fastTracker.refTemplate, this.fastTracker.fgMask);
		this.activeAngle = this.activeAngle * .72 + targetAngle * .28;
		if (Math.abs(this.activeAngle) < .015) this.activeAngle = 0;
		const newBox = {
			x1: finalX1,
			y1: finalY1,
			x2: Math.max(newW, Math.min(curW, finalX1 + newW)),
			y2: Math.max(newH, Math.min(curH, finalY1 + newH))
		};
		this.fastTracker.lastBox = newBox;
		return newBox;
	}
	getGlobalSmoothBox(det) {
		if (det === this.hoveredDetection) {
			const target = this.updateFastTracker(det) ?? det.box;
			if (target) {
				if (!this.activeTargetBox) this.activeTargetBox = {
					x1: target.x1,
					y1: target.y1,
					x2: target.x2,
					y2: target.y2
				};
				else {
					const targetAlpha = .7;
					this.activeTargetBox.x1 += (target.x1 - this.activeTargetBox.x1) * targetAlpha;
					this.activeTargetBox.y1 += (target.y1 - this.activeTargetBox.y1) * targetAlpha;
					this.activeTargetBox.x2 += (target.x2 - this.activeTargetBox.x2) * targetAlpha;
					this.activeTargetBox.y2 += (target.y2 - this.activeTargetBox.y2) * targetAlpha;
				}
			}
			if (!this.activeSmoothBox && this.activeTargetBox) this.activeSmoothBox = {
				x1: this.activeTargetBox.x1,
				y1: this.activeTargetBox.y1,
				x2: this.activeTargetBox.x2,
				y2: this.activeTargetBox.y2
			};
			if (this.activeSmoothBox && this.activeTargetBox) {
				const maxStep = 40;
				const dx1 = this.activeTargetBox.x1 - this.activeSmoothBox.x1;
				const dy1 = this.activeTargetBox.y1 - this.activeSmoothBox.y1;
				const dx2 = this.activeTargetBox.x2 - this.activeSmoothBox.x2;
				const dy2 = this.activeTargetBox.y2 - this.activeSmoothBox.y2;
				this.activeSmoothBox.x1 += Math.max(-40, Math.min(maxStep, dx1));
				this.activeSmoothBox.y1 += Math.max(-40, Math.min(maxStep, dy1));
				this.activeSmoothBox.x2 += Math.max(-40, Math.min(maxStep, dx2));
				this.activeSmoothBox.y2 += Math.max(-40, Math.min(maxStep, dy2));
				return this.activeSmoothBox;
			}
		}
		return this.getSmoothBox(det);
	}
	/**
	* Filters `rawYoloeDetections` by `yoloeConfThreshold`, updates the UI chips,
	* and precomputes the EfficientSAM-Ti mask for each active detection so hover is 0 ms.
	*/
	async filterAndPrecomputeYoloeMasks(expectedImageVersion = this.currentImageVersion) {
		this.activeYoloeDetections = this.rawYoloeDetections.filter((d) => d.score >= this.yoloeConfThreshold).slice(0, 18);
		this.updateObjectTracking();
		this.renderDetectionChips();
		this.renderCanvas();
		if (!this.samReady) return;
		const embeddings = await this.ensureImageEncoded();
		if (expectedImageVersion !== this.currentImageVersion) return;
		for (const det of this.activeYoloeDetections) {
			if (expectedImageVersion !== this.currentImageVersion) return;
			if (!det.binaryMask || det.needsSamUpdate) {
				det.needsSamUpdate = false;
				await this.computeSamMaskForDetection(det, embeddings);
				if (det === this.hoveredDetection && det.overlayData) {
					this.maskCtx.putImageData(det.overlayData, 0, 0);
					this.hasMask = true;
					this.syncTrackerReferenceWithSamMask(det);
					this.renderCanvas();
				}
			}
		}
		if (this.hoverLocked && this.hoveredDetection) this.presentYoloeDetection(this.hoveredDetection);
		this.renderCanvas();
	}
	async computeSamMaskForDetection(det, embeddings) {
		if (!this.samModels) return;
		const { decoder } = this.samModels;
		const pts = new Float32Array(MAX_POINTS * 2).fill(-1);
		const lbls = new Float32Array(MAX_POINTS).fill(-1);
		const { x1, y1, x2, y2 } = det.box;
		pts[0] = (x1 + x2) * .5;
		pts[1] = (y1 + y2) * .5;
		lbls[0] = 1;
		pts[2] = x1;
		pts[3] = y1;
		lbls[1] = 2;
		pts[4] = x2;
		pts[5] = y2;
		lbls[2] = 3;
		const { masksLogits, ious, decoderMs } = await samSerialize(async () => {
			const tDec0 = performance.now();
			const outputs = await runModel(decoder, [
				{
					data: embeddings,
					shape: [
						1,
						SAM_EMBED_DIM,
						SAM_EMBED_GRID,
						SAM_EMBED_GRID
					]
				},
				{
					data: pts,
					shape: [
						1,
						MAX_POINTS,
						2
					]
				},
				{
					data: lbls,
					shape: [1, MAX_POINTS]
				}
			]);
			const decMs = performance.now() - tDec0;
			const asFloat32 = (arr) => arr instanceof Float32Array ? new Float32Array(arr) : new Float32Array(arr.buffer, arr.byteOffset, arr.byteLength / 4);
			const out0 = asFloat32(outputs[0]);
			const out1 = outputs[1] ? asFloat32(outputs[1]) : new Float32Array([
				0,
				0,
				0
			]);
			const [logits, iouScores] = out0.length < out1.length ? [out1, out0] : [out0, out1];
			return {
				masksLogits: logits,
				ious: iouScores,
				decoderMs: decMs
			};
		});
		const bestIdx = this.selectBestCandidateMask(masksLogits, ious, det.box, []);
		const { binaryMask, overlayData, cutoutBuffer, cropW, cropH, fgCount } = this.upsampleAndBuildMaskOverlay(masksLogits, bestIdx);
		det.binaryMask = binaryMask;
		det.overlayData = overlayData;
		det.maskBox = {
			x1: det.box.x1,
			y1: det.box.y1,
			x2: det.box.x2,
			y2: det.box.y2
		};
		det.contourProfile = this.extractContourProfile(binaryMask, det.maskBox);
		det.cutoutBuffer = cutoutBuffer;
		det.cropW = cropW;
		det.cropH = cropH;
		det.fgCount = fgCount;
		det.samIou = ious[bestIdx] ?? 0;
		det.samDecoderMs = decoderMs;
		det.pts = pts;
		det.lbls = lbls;
	}
	/**
	* Extracts a 64-ray polar radial profile around the binary segmentation mask
	* so smoothness and radial padding can be adjusted dynamically at 60fps.
	*/
	extractContourProfile(binaryMask, box) {
		const W = MODEL_IMG_SIZE;
		const H = MODEL_IMG_SIZE;
		const bx1 = Math.max(0, Math.floor(box.x1 - 8));
		const by1 = Math.max(0, Math.floor(box.y1 - 8));
		const bx2 = Math.min(W - 1, Math.ceil(box.x2 + 8));
		const by2 = Math.min(H - 1, Math.ceil(box.y2 + 8));
		let sumX = 0;
		let sumY = 0;
		let count = 0;
		for (let y = by1; y <= by2; y++) {
			const row = y * W;
			for (let x = bx1; x <= bx2; x++) if (binaryMask[row + x]) {
				sumX += x;
				sumY += y;
				count++;
			}
		}
		const bcx = (box.x1 + box.x2) * .5;
		const bcy = (box.y1 + box.y2) * .5;
		const bw = Math.max(8, box.x2 - box.x1);
		const bh = Math.max(8, box.y2 - box.y1);
		if (count < 10) return void 0;
		const mcx = sumX / count;
		const mcy = sumY / count;
		const numRays = 64;
		const maxR = Math.hypot(bw, bh) * .78;
		const radii = new Float32Array(numRays);
		for (let i = 0; i < numRays; i++) {
			const angle = i / numRays * Math.PI * 2;
			const dx = Math.cos(angle);
			const dy = Math.sin(angle);
			let furthestR = 0;
			for (let r = 1; r <= maxR; r += 1.5) {
				const px = Math.round(mcx + dx * r);
				const py = Math.round(mcy + dy * r);
				if (px < bx1 || px > bx2 || py < by1 || py > by2) break;
				if (binaryMask[py * W + px]) furthestR = r;
			}
			radii[i] = furthestR;
		}
		for (let i = 0; i < numRays; i++) if (radii[i] === 0) {
			const prev = radii[(i - 1 + numRays) % numRays];
			const next = radii[(i + 1) % numRays];
			radii[i] = (prev + next) * .5 || Math.min(bw, bh) * .25;
		}
		return {
			mcx,
			mcy,
			bcx,
			bcy,
			bw,
			bh,
			radii
		};
	}
	/**
	* Computes normalized [-0.5, 0.5] contour points from a `ContourProfile`
	* using the current `outlineSmoothness` and `outlinePadding` slider values.
	* Performs smoothing in Cartesian `(x, y)` space so straight edges stay straight,
	* and applies `this.outlinePadding` via aspect-preserving horizontal/vertical margin
	* expansion around `(mcx, mcy)` so padding never bows edges or distorts the object shape.
	*/
	buildSmoothContourPoints(profile) {
		const { mcx, mcy, bcx, bcy, bw, bh, radii } = profile;
		const numRays = radii.length;
		const passes = Math.max(0, Math.round(this.outlineSmoothness));
		const pad = this.outlinePadding;
		let xs = new Float32Array(numRays);
		let ys = new Float32Array(numRays);
		let hw0 = 4;
		let hh0 = 4;
		for (let i = 0; i < numRays; i++) {
			const angle = i / numRays * Math.PI * 2;
			const r = Math.max(2, radii[i]);
			const dx = Math.cos(angle) * r;
			const dy = Math.sin(angle) * r;
			xs[i] = mcx + dx;
			ys[i] = mcy + dy;
			const ax = Math.abs(dx);
			const ay = Math.abs(dy);
			if (ax > hw0) hw0 = ax;
			if (ay > hh0) hh0 = ay;
		}
		for (let pass = 0; pass < passes; pass++) {
			const nextX = new Float32Array(numRays);
			const nextY = new Float32Array(numRays);
			for (let i = 0; i < numRays; i++) {
				const prev = (i - 1 + numRays) % numRays;
				const next = (i + 1) % numRays;
				nextX[i] = .25 * xs[prev] + .5 * xs[i] + .25 * xs[next];
				nextY[i] = .25 * ys[prev] + .5 * ys[i] + .25 * ys[next];
			}
			xs = nextX;
			ys = nextY;
		}
		let hw = 4;
		let hh = 4;
		for (let i = 0; i < numRays; i++) {
			const ax = Math.abs(xs[i] - mcx);
			const ay = Math.abs(ys[i] - mcy);
			if (ax > hw) hw = ax;
			if (ay > hh) hh = ay;
		}
		const scaleX = Math.max(.1, (hw0 + pad) / hw);
		const scaleY = Math.max(.1, (hh0 + pad) / hh);
		const points = [];
		for (let i = 0; i < numRays; i++) {
			const px = mcx + (xs[i] - mcx) * scaleX;
			const py = mcy + (ys[i] - mcy) * scaleY;
			points.push({
				x: (px - bcx) / bw,
				y: (py - bcy) / bh
			});
		}
		return points;
	}
	renderDetectionChips() {
		this.detectedChipsEl.innerHTML = "";
		if (this.activeYoloeDetections.length === 0) {
			const empty = document.createElement("span");
			empty.style.fontSize = "0.75rem";
			empty.style.color = "#64748b";
			empty.textContent = this.yoloeReady ? "No YOLOE-11s objects above current confidence threshold (try lowering the slider)." : "Waiting for YOLOE-11s model...";
			this.detectedChipsEl.appendChild(empty);
			return;
		}
		for (const det of this.activeYoloeDetections) {
			const btn = document.createElement("button");
			btn.type = "button";
			btn.className = "det-chip" + (this.hoveredDetection === det ? " active" : "");
			btn.textContent = `${det.className} ${(det.score * 100).toFixed(0)}%`;
			btn.addEventListener("pointerenter", () => {
				if (this.mode === "hover" && !this.hoverLocked) this.presentYoloeDetection(det);
			});
			btn.addEventListener("click", () => {
				if (this.mode !== "hover") this.setMode("hover");
				this.hoverLocked = true;
				this.presentYoloeDetection(det);
			});
			this.detectedChipsEl.appendChild(btn);
		}
	}
	/**
	* Finds the most specific YOLOE-11s semantic object at pixel `(x, y)`:
	* 1. Prefers detections whose precomputed SAM binary mask is `1` at `(x, y)` (smallest area first).
	* 2. Falls back to detections whose YOLOE bounding box contains `(x, y)` (smallest box area first).
	*/
	findYoloeDetectionAtPoint(pt) {
		const px = Math.max(0, Math.min(MODEL_IMG_SIZE - 1, Math.floor(pt.x)));
		const idx = Math.max(0, Math.min(MODEL_IMG_SIZE - 1, Math.floor(pt.y))) * MODEL_IMG_SIZE + px;
		let bestByMask = null;
		let bestMaskArea = Infinity;
		for (const det of this.activeYoloeDetections) if (det.binaryMask && det.binaryMask[idx] === 1) {
			const area = det.fgCount ?? det.boxArea;
			if (area < bestMaskArea) {
				bestMaskArea = area;
				bestByMask = det;
			}
		}
		if (bestByMask) return bestByMask;
		let bestByBox = null;
		let bestBoxArea = Infinity;
		for (const det of this.activeYoloeDetections) {
			const { x1, y1, x2, y2 } = det.box;
			if (pt.x >= x1 && pt.x <= x2 && pt.y >= y1 && pt.y <= y2) {
				if (det.boxArea < bestBoxArea) {
					bestBoxArea = det.boxArea;
					bestByBox = det;
				}
			}
		}
		return bestByBox;
	}
	presentYoloeDetection(det) {
		if (this.hoveredDetection !== det) {
			this.activeSmoothBox = {
				x1: det.box.x1,
				y1: det.box.y1,
				x2: det.box.x2,
				y2: det.box.y2
			};
			this.activeTargetBox = {
				x1: det.box.x1,
				y1: det.box.y1,
				x2: det.box.x2,
				y2: det.box.y2
			};
			this.initFastTracker(det);
		}
		this.hoveredDetection = det;
		this.activeTargetBox = det.box;
		this.updateChipHighlight();
		if (det.overlayData) {
			this.maskCtx.putImageData(det.overlayData, 0, 0);
			this.hasMask = true;
			this.renderCanvas();
			const fgCount = det.fgCount ?? 0;
			const areaPct = (fgCount / (MODEL_IMG_SIZE * MODEL_IMG_SIZE) * 100).toFixed(1);
			this.metricTotal.textContent = `${det.className} (${(det.score * 100).toFixed(0)}%)`;
			this.metricSplit.textContent = `${this.lastYoloeMs.toFixed(0)} / ${this.lastEncoderMs.toFixed(0)} / ${(det.samDecoderMs ?? 0).toFixed(1)} ms`;
			this.metricIou.textContent = `${(det.samIou ?? 0).toFixed(3)} · Conf ${(det.score * 100).toFixed(0)}%`;
			this.metricArea.textContent = `${areaPct}% (${fgCount.toLocaleString()} px)`;
			this.updateCutoutPreview(det.cutoutBuffer ?? null, det.cropW ?? 0, det.cropH ?? 0);
			if (det.pts && det.lbls) {
				const header = `YOLOE-11s Class: ${det.className} (id=${det.classId}, conf=${(det.score * 100).toFixed(1)}%)\nBox: [${det.box.x1.toFixed(0)}, ${det.box.y1.toFixed(0)}, ${det.box.x2.toFixed(0)}, ${det.box.y2.toFixed(0)}]\n\n`;
				this.promptDebug.textContent = header + this.formatPromptDebug(det.pts, det.lbls);
			}
			const lockTag = this.hoverLocked ? "🔒 Locked (click to unlock)" : "0 ms Semantic Hover (click to lock)";
			this.setStatus(`<strong>${det.className} (${(det.score * 100).toFixed(0)}%)</strong> — SAM IoU ${(det.samIou ?? 0).toFixed(3)} · ${lockTag}`);
		} else if (this.samReady && this.cachedEmbeddings) this.computeSamMaskForDetection(det, this.cachedEmbeddings).then(() => {
			if (this.hoveredDetection === det) this.presentYoloeDetection(det);
		});
	}
	updateChipHighlight() {
		this.detectedChipsEl.querySelectorAll(".det-chip").forEach((chip, idx) => {
			chip.classList.toggle("active", this.activeYoloeDetections[idx] === this.hoveredDetection);
		});
	}
	setStatus(html) {
		this.statusBanner.innerHTML = html;
	}
	bindUiEvents() {
		const presetBtns = document.querySelectorAll(".preset-btn");
		presetBtns.forEach((btn) => {
			btn.addEventListener("click", () => {
				this.stopVideoMode();
				const webcamBtn = document.getElementById("webcam-btn");
				if (webcamBtn) webcamBtn.classList.remove("pos-active");
				presetBtns.forEach((b) => b.classList.remove("active"));
				btn.classList.add("active");
				const preset = btn.dataset.preset || "livingroom";
				this.loadPresetImage(preset);
			});
		});
		const webcamBtn = document.getElementById("webcam-btn");
		const cameraFlipBtn = document.getElementById("camera-flip-btn");
		const webcamVideo = document.getElementById("webcam-video");
		if (cameraFlipBtn && webcamBtn && webcamVideo) cameraFlipBtn.addEventListener("click", async () => {
			this.cameraFacingMode = this.cameraFacingMode === "user" ? "environment" : "user";
			cameraFlipBtn.textContent = this.cameraFacingMode === "user" ? "🔄 Front (Mirrored)" : "🔄 Rear";
			if (this.isVideoMode) {
				this.stopVideoMode();
				try {
					presetBtns.forEach((b) => b.classList.remove("active"));
					const stream = await navigator.mediaDevices.getUserMedia({ video: {
						width: { ideal: 512 },
						height: { ideal: 512 },
						facingMode: this.cameraFacingMode
					} });
					webcamVideo.srcObject = stream;
					await webcamVideo.play();
					this.startVideoSource(webcamVideo);
					webcamBtn.classList.add("pos-active");
					webcamBtn.textContent = "⏹️ Stop Live Webcam";
				} catch (err) {
					alert(`Could not switch camera: ${describeError(err)}`);
				}
			}
		});
		if (webcamBtn && webcamVideo) webcamBtn.addEventListener("click", async () => {
			if (this.isVideoMode) {
				this.stopVideoMode();
				webcamBtn.classList.remove("pos-active");
				webcamBtn.textContent = "📹 Start Live Webcam";
				this.setStatus("Webcam stopped.");
				return;
			}
			try {
				presetBtns.forEach((b) => b.classList.remove("active"));
				const stream = await navigator.mediaDevices.getUserMedia({ video: {
					width: { ideal: 512 },
					height: { ideal: 512 },
					facingMode: this.cameraFacingMode
				} });
				webcamVideo.srcObject = stream;
				await webcamVideo.play();
				this.startVideoSource(webcamVideo);
				webcamBtn.classList.add("pos-active");
				webcamBtn.textContent = "⏹️ Stop Live Webcam";
			} catch (err) {
				alert(`Could not start webcam: ${describeError(err)}`);
			}
		});
		const fileInput = document.getElementById("file-input");
		fileInput.addEventListener("change", () => {
			const file = fileInput.files?.[0];
			if (file) {
				this.stopVideoMode();
				if (webcamBtn) webcamBtn.classList.remove("pos-active");
				presetBtns.forEach((b) => b.classList.remove("active"));
				if (file.type.startsWith("video/")) {
					const url = URL.createObjectURL(file);
					webcamVideo.srcObject = null;
					webcamVideo.src = url;
					webcamVideo.loop = true;
					webcamVideo.play();
					this.startVideoSource(webcamVideo);
				} else this.loadUploadedFile(file);
				fileInput.value = "";
			}
		});
		const uploadZone = document.getElementById("upload-zone");
		const canvasWrapper = document.getElementById("canvas-wrapper");
		for (const dropTarget of [uploadZone, canvasWrapper]) {
			dropTarget.addEventListener("dragover", (e) => {
				e.preventDefault();
				uploadZone.classList.add("dragover");
			});
			dropTarget.addEventListener("dragleave", () => {
				uploadZone.classList.remove("dragover");
			});
			dropTarget.addEventListener("drop", (e) => {
				e.preventDefault();
				uploadZone.classList.remove("dragover");
				const file = e.dataTransfer?.files?.[0];
				if (file && file.type.startsWith("image/")) {
					presetBtns.forEach((b) => b.classList.remove("active"));
					this.loadUploadedFile(file);
				}
			});
		}
		window.addEventListener("paste", (e) => {
			const items = e.clipboardData?.items;
			if (!items) return;
			for (let i = 0; i < items.length; i++) if (items[i].type.startsWith("image/")) {
				const file = items[i].getAsFile();
				if (file) {
					presetBtns.forEach((b) => b.classList.remove("active"));
					this.loadUploadedFile(file);
					break;
				}
			}
		});
		const fitToggle = document.getElementById("fit-contain-toggle");
		fitToggle.addEventListener("change", () => {
			this.fitContain = fitToggle.checked;
			if (this.loadedSourceImage) {
				this.drawSourceToBaseCanvas(this.loadedSourceImage);
				this.clearPrompts();
				this.analyzeCurrentImage("resized image");
			}
		});
		document.querySelectorAll(".mode-btn").forEach((btn) => {
			btn.addEventListener("click", () => {
				const nextMode = btn.dataset.mode || "hover";
				this.setMode(nextMode);
			});
		});
		const confSlider = document.getElementById("yoloe-conf-slider");
		const confLabel = document.getElementById("yoloe-conf-label");
		confSlider.addEventListener("input", () => {
			const pct = Number(confSlider.value);
			confLabel.textContent = `${pct}%`;
			this.yoloeConfThreshold = pct / 100;
			this.filterAndPrecomputeYoloeMasks();
		});
		const showAllBoxesToggle = document.getElementById("show-all-boxes-toggle");
		showAllBoxesToggle.addEventListener("change", () => {
			this.showAllYoloeBoxes = showAllBoxesToggle.checked;
			this.renderCanvas();
		});
		const hoverFallbackToggle = document.getElementById("hover-fallback-sam-toggle");
		hoverFallbackToggle.addEventListener("change", () => {
			this.hoverFallbackToSam = hoverFallbackToggle.checked;
		});
		document.querySelectorAll("#mask-candidate-row .btn").forEach((btn) => {
			btn.addEventListener("click", () => {
				const val = btn.dataset.candidate;
				this.setCandidateChoice(val === "0" ? 0 : val === "1" ? 1 : val === "2" ? 2 : "auto");
			});
		});
		const bboxCenterToggle = document.getElementById("bbox-center-toggle");
		bboxCenterToggle.addEventListener("change", () => {
			this.includeBboxCenter = bboxCenterToggle.checked;
			this.renderCanvas();
			if (this.mode === "bbox" && this.currentBox) this.runSegmentationForCurrentPrompt();
		});
		const posBtn = document.getElementById("pt-pos-btn");
		const negBtn = document.getElementById("pt-neg-btn");
		posBtn.addEventListener("click", () => {
			this.activePointLabel = 1;
			posBtn.classList.add("pos-active");
			negBtn.classList.remove("neg-active");
		});
		negBtn.addEventListener("click", () => {
			this.activePointLabel = 0;
			negBtn.classList.add("neg-active");
			posBtn.classList.remove("pos-active");
		});
		this.undoPtBtn.addEventListener("click", () => {
			if (this.promptPoints.length > 0) {
				this.promptPoints.pop();
				this.updatePointsUi();
				if (this.promptPoints.length > 0) {
					this.renderCanvas();
					this.runSegmentationForCurrentPrompt();
				} else this.clearPrompts();
			}
		});
		const showMaskToggle = document.getElementById("show-mask-toggle");
		showMaskToggle.addEventListener("change", () => {
			this.showMask = showMaskToggle.checked;
			this.renderCanvas();
		});
		const showOutlineToggle = document.getElementById("show-outline-toggle");
		if (showOutlineToggle) showOutlineToggle.addEventListener("change", () => {
			this.showOutline = showOutlineToggle.checked;
			this.renderCanvas();
		});
		const showPromptToggle = document.getElementById("show-prompt-toggle");
		showPromptToggle.addEventListener("change", () => {
			this.showPrompt = showPromptToggle.checked;
			this.renderCanvas();
		});
		const opacitySlider = document.getElementById("mask-opacity-slider");
		opacitySlider.addEventListener("input", () => {
			this.maskOpacity = Number(opacitySlider.value) / 100;
			this.renderCanvas();
		});
		const outlineSmoothnessSlider = document.getElementById("outline-smoothness-slider");
		const outlineSmoothnessLabel = document.getElementById("outline-smoothness-label");
		if (outlineSmoothnessSlider) outlineSmoothnessSlider.addEventListener("input", () => {
			this.outlineSmoothness = Number(outlineSmoothnessSlider.value);
			if (outlineSmoothnessLabel) outlineSmoothnessLabel.textContent = `${this.outlineSmoothness}`;
			this.renderCanvas();
		});
		const outlinePaddingSlider = document.getElementById("outline-padding-slider");
		const outlinePaddingLabel = document.getElementById("outline-padding-label");
		if (outlinePaddingSlider) outlinePaddingSlider.addEventListener("input", () => {
			this.outlinePadding = Number(outlinePaddingSlider.value);
			if (outlinePaddingLabel) outlinePaddingLabel.textContent = `${this.outlinePadding}px`;
			this.renderCanvas();
		});
		document.getElementById("clear-btn").addEventListener("click", () => {
			this.clearPrompts();
		});
		this.downloadCutoutBtn.addEventListener("click", () => {
			if (!this.lastCutoutCropCanvas) return;
			const a = document.createElement("a");
			a.href = this.lastCutoutCropCanvas.toDataURL("image/png");
			a.download = "yoloe-sam-cutout.png";
			a.click();
		});
		this.displayCanvas.addEventListener("contextmenu", (e) => e.preventDefault());
		this.displayCanvas.addEventListener("pointerdown", (e) => this.onPointerDown(e));
		this.displayCanvas.addEventListener("pointermove", (e) => this.onPointerMove(e));
		this.displayCanvas.addEventListener("pointerleave", () => {
			if (this.mode === "hover" && !this.hoverLocked) {
				this.hoveredDetection = null;
				this.hoverPoint = null;
				this.hasMask = false;
				this.updateChipHighlight();
				this.renderCanvas();
			}
		});
		window.addEventListener("pointerup", (e) => this.onPointerUp(e));
	}
	setCandidateChoice(choice) {
		this.candidateChoice = choice;
		document.querySelectorAll("#mask-candidate-row .btn").forEach((btn) => {
			const val = btn.dataset.candidate;
			const isMatch = choice === "auto" && val === "auto" || String(choice) === val;
			btn.classList.toggle("pos-active", isMatch);
		});
		this.runSegmentationForCurrentPrompt();
	}
	setMode(nextMode) {
		this.mode = nextMode;
		document.querySelectorAll(".mode-btn").forEach((btn) => {
			btn.classList.toggle("active", btn.dataset.mode === nextMode);
		});
		document.getElementById("hover-controls").style.display = nextMode === "hover" ? "flex" : "none";
		document.getElementById("candidate-controls").style.display = nextMode === "points" ? "flex" : "none";
		document.getElementById("bbox-controls").style.display = nextMode === "bbox" ? "flex" : "none";
		document.getElementById("points-controls").style.display = nextMode === "points" ? "flex" : "none";
		if (nextMode === "hover") this.stageHint.textContent = "Hover over any YOLOE-11s semantic object to highlight its mask & label (click to lock/unlock).";
		else if (nextMode === "bbox") this.stageHint.textContent = "Click and drag a bounding box around any object to segment it.";
		else if (nextMode === "circle") this.stageHint.textContent = "Draw a freehand circle/loop around any object (or tap) to segment it.";
		else this.stageHint.textContent = "Left-click to add Positive (+) points, Right-click or Shift+click for Negative (−) points (up to 6).";
		this.clearPrompts();
	}
	getCanvasPoint(e) {
		const rect = this.displayCanvas.getBoundingClientRect();
		const scaleX = MODEL_IMG_SIZE / Math.max(1, rect.width);
		const scaleY = MODEL_IMG_SIZE / Math.max(1, rect.height);
		return {
			x: Math.max(0, Math.min(MODEL_IMG_SIZE, (e.clientX - rect.left) * scaleX)),
			y: Math.max(0, Math.min(MODEL_IMG_SIZE, (e.clientY - rect.top) * scaleY))
		};
	}
	onPointerDown(e) {
		const pt = this.getCanvasPoint(e);
		if (this.mode === "hover") {
			this.hoverLocked = !this.hoverLocked;
			this.hoverPoint = pt;
			const hit = this.findYoloeDetectionAtPoint(pt);
			if (hit) this.presentYoloeDetection(hit);
			else if (this.hoverFallbackToSam) {
				this.hoveredDetection = null;
				this.updateChipHighlight();
				this.renderCanvas();
				this.runSegmentationForCurrentPrompt();
			}
			return;
		}
		if (this.mode === "points") {
			const existingIdx = this.promptPoints.findIndex((p) => Math.hypot(p.x - pt.x, p.y - pt.y) < 11);
			if (existingIdx >= 0 && !e.shiftKey && e.button === 0) {
				this.promptPoints.splice(existingIdx, 1);
				this.updatePointsUi();
				if (this.promptPoints.length > 0) {
					this.renderCanvas();
					this.runSegmentationForCurrentPrompt();
				} else this.clearPrompts();
				return;
			}
			if (this.promptPoints.length >= MAX_POINTS) {
				this.setStatus(`Maximum of <strong>${MAX_POINTS} points</strong> reached. Click an existing point or Undo to remove one.`);
				return;
			}
			const isNeg = e.button === 2 || e.shiftKey || this.activePointLabel === 0;
			this.promptPoints.push({
				x: pt.x,
				y: pt.y,
				label: isNeg ? 0 : 1
			});
			this.updatePointsUi();
			this.renderCanvas();
			this.runSegmentationForCurrentPrompt();
			return;
		}
		this.isDragging = true;
		this.dragStart = pt;
		if (this.mode === "bbox") {
			this.currentBox = {
				x1: pt.x,
				y1: pt.y,
				x2: pt.x,
				y2: pt.y
			};
			this.renderCanvas();
		} else if (this.mode === "circle") {
			this.circlePath = [pt];
			this.circleDerivedBox = null;
			this.renderCanvas();
		}
	}
	onPointerMove(e) {
		const pt = this.getCanvasPoint(e);
		if (this.mode === "hover") {
			if (this.hoverLocked) return;
			this.hoverPoint = pt;
			const hit = this.findYoloeDetectionAtPoint(pt);
			if (hit) {
				if (this.hoveredDetection !== hit || !this.hasMask) this.presentYoloeDetection(hit);
				else this.renderCanvas();
				return;
			}
			this.hoveredDetection = null;
			this.updateChipHighlight();
			if (this.hoverFallbackToSam) {
				this.renderCanvas();
				this.runSegmentationForCurrentPrompt();
			} else {
				this.hasMask = false;
				this.renderCanvas();
			}
			return;
		}
		if (!this.isDragging) return;
		if (this.mode === "bbox" && this.dragStart) {
			this.currentBox = {
				x1: Math.min(this.dragStart.x, pt.x),
				y1: Math.min(this.dragStart.y, pt.y),
				x2: Math.max(this.dragStart.x, pt.x),
				y2: Math.max(this.dragStart.y, pt.y)
			};
			this.renderCanvas();
		} else if (this.mode === "circle") {
			const last = this.circlePath[this.circlePath.length - 1];
			if (!last || Math.hypot(pt.x - last.x, pt.y - last.y) >= 2) {
				this.circlePath.push(pt);
				this.renderCanvas();
			}
		}
	}
	onPointerUp(_e) {
		if (!this.isDragging) return;
		this.isDragging = false;
		if (this.mode === "bbox" && this.currentBox) {
			const w = this.currentBox.x2 - this.currentBox.x1;
			const h = this.currentBox.y2 - this.currentBox.y1;
			if (w < 4 && h < 4) {
				const cx = this.currentBox.x1;
				const cy = this.currentBox.y1;
				this.currentBox = {
					x1: Math.max(0, cx - 20),
					y1: Math.max(0, cy - 20),
					x2: Math.min(MODEL_IMG_SIZE, cx + 20),
					y2: Math.min(MODEL_IMG_SIZE, cy + 20)
				};
			}
			this.renderCanvas();
			this.runSegmentationForCurrentPrompt();
		} else if (this.mode === "circle" && this.circlePath.length > 0) {
			this.renderCanvas();
			this.runSegmentationForCurrentPrompt();
		}
	}
	updatePointsUi() {
		this.pointsCountEl.textContent = `${this.promptPoints.length} / ${MAX_POINTS} pts`;
		this.undoPtBtn.disabled = this.promptPoints.length === 0;
	}
	clearPrompts() {
		this.hoverPoint = null;
		this.hoverLocked = false;
		this.hoveredDetection = null;
		this.isDragging = false;
		this.dragStart = null;
		this.currentBox = null;
		this.circlePath = [];
		this.circleDerivedBox = null;
		this.promptPoints = [];
		this.hasMask = false;
		this.maskCtx.clearRect(0, 0, MODEL_IMG_SIZE, MODEL_IMG_SIZE);
		this.updatePointsUi();
		this.updateChipHighlight();
		this.promptDebug.textContent = "No active prompt.";
		this.clearCutoutPreview();
		this.renderCanvas();
	}
	buildPromptTensors() {
		const pts = new Float32Array(MAX_POINTS * 2).fill(-1);
		const lbls = new Float32Array(MAX_POINTS).fill(-1);
		if (this.mode === "hover") {
			if (!this.hoverPoint) return null;
			pts[0] = this.hoverPoint.x;
			pts[1] = this.hoverPoint.y;
			lbls[0] = 1;
			return {
				pts,
				lbls,
				box: null
			};
		}
		if (this.mode === "bbox") {
			if (!this.currentBox) return null;
			const { x1, y1, x2, y2 } = this.currentBox;
			if (this.includeBboxCenter) {
				pts[0] = (x1 + x2) * .5;
				pts[1] = (y1 + y2) * .5;
				lbls[0] = 1;
				pts[2] = x1;
				pts[3] = y1;
				lbls[1] = 2;
				pts[4] = x2;
				pts[5] = y2;
				lbls[2] = 3;
			} else {
				pts[0] = x1;
				pts[1] = y1;
				lbls[0] = 2;
				pts[2] = x2;
				pts[3] = y2;
				lbls[1] = 3;
			}
			return {
				pts,
				lbls,
				box: this.currentBox
			};
		}
		if (this.mode === "circle") {
			if (this.circlePath.length === 0) return null;
			let minX = Infinity;
			let minY = Infinity;
			let maxX = -Infinity;
			let maxY = -Infinity;
			let sumX = 0;
			let sumY = 0;
			for (const p of this.circlePath) {
				if (p.x < minX) minX = p.x;
				if (p.y < minY) minY = p.y;
				if (p.x > maxX) maxX = p.x;
				if (p.y > maxY) maxY = p.y;
				sumX += p.x;
				sumY += p.y;
			}
			const n = this.circlePath.length;
			const span = Math.hypot(maxX - minX, maxY - minY);
			if (span < 14) {
				const cx = sumX / n;
				const cy = sumY / n;
				pts[0] = cx;
				pts[1] = cy;
				lbls[0] = 1;
				this.circleDerivedBox = null;
				return {
					pts,
					lbls,
					box: null
				};
			}
			const pad = Math.max(4, span * .04);
			const x1 = Math.max(0, minX - pad);
			const y1 = Math.max(0, minY - pad);
			const x2 = Math.min(MODEL_IMG_SIZE, maxX + pad);
			const y2 = Math.min(MODEL_IMG_SIZE, maxY + pad);
			const cx = (x1 + x2) * .5;
			const cy = (y1 + y2) * .5;
			pts[0] = x1;
			pts[1] = y1;
			lbls[0] = 2;
			pts[2] = x2;
			pts[3] = y2;
			lbls[1] = 3;
			pts[4] = cx;
			pts[5] = cy;
			lbls[2] = 1;
			this.circleDerivedBox = {
				x1,
				y1,
				x2,
				y2
			};
			return {
				pts,
				lbls,
				box: this.circleDerivedBox
			};
		}
		if (this.mode === "points") {
			if (this.promptPoints.length === 0) return null;
			for (let i = 0; i < this.promptPoints.length && i < MAX_POINTS; i++) {
				pts[i * 2] = this.promptPoints[i].x;
				pts[i * 2 + 1] = this.promptPoints[i].y;
				lbls[i] = this.promptPoints[i].label;
			}
			return {
				pts,
				lbls,
				box: null
			};
		}
		return null;
	}
	formatPromptDebug(pts, lbls, ious, bestIdx) {
		const labelName = (l) => {
			if (l === 1) return "1.0 (Pos FG)";
			if (l === 0) return "0.0 (Neg BG)";
			if (l === 2) return "2.0 (Box TL)";
			if (l === 3) return "3.0 (Box BR)";
			return "-1.0 (Pad)";
		};
		const lines = ["Slot | Coord (x, y)       | Label"];
		lines.push("-----+--------------------+--------------");
		for (let i = 0; i < MAX_POINTS; i++) {
			const x = pts[i * 2].toFixed(1).padStart(6);
			const y = pts[i * 2 + 1].toFixed(1).padStart(6);
			lines.push(`  #${i} | (${x}, ${y}) | ${labelName(lbls[i])}`);
		}
		if (ious && ious.length > 0) {
			lines.push("");
			const iouStr = Array.from(ious).map((v, idx) => `${idx === bestIdx ? "*" : ""}m${idx}=${v.toFixed(3)}`).join("  ");
			lines.push(`IoUs: ${iouStr}`);
		}
		return lines.join("\n");
	}
	async runSegmentationForCurrentPrompt() {
		const built = this.buildPromptTensors();
		if (!built) return;
		this.promptDebug.textContent = this.formatPromptDebug(built.pts, built.lbls);
		if (!this.samReady) {
			this.pendingRerun = true;
			this.setStatus("Waiting for LiteRT SAM initialization...");
			return;
		}
		if (this.isSegmenting) {
			this.pendingRerun = true;
			return;
		}
		this.isSegmenting = true;
		try {
			const embeddings = await this.ensureImageEncoded();
			const { decoder } = this.samModels;
			const { masksLogits, ious, decoderMs } = await samSerialize(async () => {
				const tDec0 = performance.now();
				const outputs = await runModel(decoder, [
					{
						data: embeddings,
						shape: [
							1,
							SAM_EMBED_DIM,
							SAM_EMBED_GRID,
							SAM_EMBED_GRID
						]
					},
					{
						data: built.pts,
						shape: [
							1,
							MAX_POINTS,
							2
						]
					},
					{
						data: built.lbls,
						shape: [1, MAX_POINTS]
					}
				]);
				const decMs = performance.now() - tDec0;
				const asFloat32 = (arr) => arr instanceof Float32Array ? new Float32Array(arr) : new Float32Array(arr.buffer, arr.byteOffset, arr.byteLength / 4);
				const out0 = asFloat32(outputs[0]);
				const out1 = outputs[1] ? asFloat32(outputs[1]) : new Float32Array([
					0,
					0,
					0
				]);
				const [logits, iouScores] = out0.length < out1.length ? [out1, out0] : [out0, out1];
				return {
					masksLogits: logits,
					ious: iouScores,
					decoderMs: decMs
				};
			});
			const bestIdx = this.mode === "points" && this.candidateChoice !== "auto" ? this.candidateChoice : this.selectBestCandidateMask(masksLogits, ious, built.box, this.mode === "points" ? this.promptPoints : this.mode === "hover" && this.hoverPoint ? [{
				x: this.hoverPoint.x,
				y: this.hoverPoint.y,
				label: 1
			}] : []);
			const bestIou = ious[bestIdx] ?? 0;
			this.promptDebug.textContent = this.formatPromptDebug(built.pts, built.lbls, ious, bestIdx);
			const { binaryMask, overlayData, cutoutBuffer, cropW, cropH, fgCount } = this.upsampleAndBuildMaskOverlay(masksLogits, bestIdx);
			const fullBox = {
				x1: 0,
				y1: 0,
				x2: MODEL_IMG_SIZE,
				y2: MODEL_IMG_SIZE
			};
			this.manualContourProfile = this.extractContourProfile(binaryMask, fullBox);
			this.maskCtx.putImageData(overlayData, 0, 0);
			this.hasMask = true;
			this.renderCanvas();
			const areaPct = (fgCount / (MODEL_IMG_SIZE * MODEL_IMG_SIZE) * 100).toFixed(1);
			this.metricTotal.textContent = this.mode === "hover" ? "1-Pt Fallback" : `${decoderMs.toFixed(1)} ms`;
			this.metricSplit.textContent = `${this.lastYoloeMs.toFixed(0)} / ${this.lastEncoderMs.toFixed(0)} / ${decoderMs.toFixed(1)} ms`;
			this.metricIou.textContent = `${bestIou.toFixed(3)} (M${bestIdx})`;
			this.metricArea.textContent = `${areaPct}% (${fgCount.toLocaleString()} px)`;
			this.updateCutoutPreview(cutoutBuffer, cropW, cropH);
			this.setStatus(`<strong>Decoded in ${decoderMs.toFixed(1)} ms</strong> (Encoder ${this.lastEncoderMs.toFixed(0)} ms · IoU ${bestIou.toFixed(3)} [M${bestIdx}])`);
		} catch (err) {
			const msg = describeError(err);
			console.error("Segmentation failed:", err);
			this.setStatus(`<strong>Segmentation Error:</strong> ${msg}`);
		} finally {
			this.isSegmenting = false;
			if (this.pendingRerun) {
				this.pendingRerun = false;
				this.runSegmentationForCurrentPrompt();
			}
		}
	}
	selectBestCandidateMask(masksLogits, ious, box, points) {
		const planeStride = MASK_LOW_RES * MASK_LOW_RES;
		const numChannels = Math.max(1, Math.floor(masksLogits.length / planeStride));
		if (box) {
			const bx1 = Math.max(0, Math.floor(box.x1 / MODEL_IMG_SIZE * MASK_LOW_RES));
			const by1 = Math.max(0, Math.floor(box.y1 / MODEL_IMG_SIZE * MASK_LOW_RES));
			const bx2 = Math.min(MASK_LOW_RES, Math.ceil(box.x2 / MODEL_IMG_SIZE * MASK_LOW_RES));
			const by2 = Math.min(MASK_LOW_RES, Math.ceil(box.y2 / MODEL_IMG_SIZE * MASK_LOW_RES));
			let best = 0;
			let bestScore = -Infinity;
			for (let c = 0; c < numChannels; c++) {
				const off = c * planeStride;
				let inside = 0;
				let total = 0;
				for (let y = 0; y < MASK_LOW_RES; y++) {
					const row = off + y * MASK_LOW_RES;
					const yIn = y >= by1 && y < by2;
					for (let x = 0; x < MASK_LOW_RES; x++) if (masksLogits[row + x] >= 0) {
						total++;
						if (yIn && x >= bx1 && x < bx2) inside++;
					}
				}
				if (total === 0) continue;
				const score = inside / total + .15 * (ious[c] ?? 0);
				if (score > bestScore) {
					bestScore = score;
					best = c;
				}
			}
			return best;
		}
		if (points.length > 0) {
			let best = 0;
			let bestScore = -Infinity;
			for (let c = 0; c < numChannels; c++) {
				const off = c * planeStride;
				let ptScore = 0;
				for (const p of points) {
					const lx = Math.max(0, Math.min(MASK_LOW_RES - 1, Math.floor(p.x / MODEL_IMG_SIZE * MASK_LOW_RES)));
					const logit = masksLogits[off + Math.max(0, Math.min(MASK_LOW_RES - 1, Math.floor(p.y / MODEL_IMG_SIZE * MASK_LOW_RES))) * MASK_LOW_RES + lx];
					if (p.label === 1) ptScore += logit >= 0 ? 1 : -1;
					else ptScore += logit < 0 ? 1 : -1.5;
				}
				const score = ptScore + (ious[c] ?? 0);
				if (score > bestScore) {
					bestScore = score;
					best = c;
				}
			}
			return best;
		}
		let bestIdx = 0;
		for (let c = 1; c < numChannels; c++) if ((ious[c] ?? 0) > (ious[bestIdx] ?? 0)) bestIdx = c;
		return bestIdx;
	}
	upsampleAndBuildMaskOverlay(masksLogits, channelIdx) {
		const W = MODEL_IMG_SIZE;
		const H = MODEL_IMG_SIZE;
		const maskOffset = channelIdx * MASK_LOW_RES * MASK_LOW_RES;
		const binaryMask = new Uint8Array(W * H);
		const scaleX = MASK_LOW_RES / W;
		const scaleY = MASK_LOW_RES / H;
		let fgCount = 0;
		let minX = W;
		let minY = H;
		let maxX = 0;
		let maxY = 0;
		for (let y = 0; y < H; y++) {
			const sy = (y + .5) * scaleY - .5;
			const y0 = Math.max(0, Math.min(MASK_LOW_RES - 1, Math.floor(sy)));
			const y1 = Math.max(0, Math.min(MASK_LOW_RES - 1, y0 + 1));
			const wy = sy - y0;
			const row0 = maskOffset + y0 * MASK_LOW_RES;
			const row1 = maskOffset + y1 * MASK_LOW_RES;
			for (let x = 0; x < W; x++) {
				const sx = (x + .5) * scaleX - .5;
				const x0 = Math.max(0, Math.min(MASK_LOW_RES - 1, Math.floor(sx)));
				const x1 = Math.max(0, Math.min(MASK_LOW_RES - 1, x0 + 1));
				const wx = sx - x0;
				const v00 = masksLogits[row0 + x0];
				const v01 = masksLogits[row0 + x1];
				const v10 = masksLogits[row1 + x0];
				const v11 = masksLogits[row1 + x1];
				if ((1 - wy) * ((1 - wx) * v00 + wx * v01) + wy * ((1 - wx) * v10 + wx * v11) >= 0) {
					binaryMask[y * W + x] = 1;
					fgCount++;
					if (x < minX) minX = x;
					if (y < minY) minY = y;
					if (x > maxX) maxX = x;
					if (y > maxY) maxY = y;
				}
			}
		}
		const overlayRgba = new Uint8ClampedArray(W * H * 4);
		for (let y = 0; y < H; y++) {
			const rowOffset = y * W;
			for (let x = 0; x < W; x++) {
				const i = rowOffset + x;
				if (!binaryMask[i]) continue;
				const p = i * 4;
				overlayRgba[p] = 56;
				overlayRgba[p + 1] = 189;
				overlayRgba[p + 2] = 248;
				overlayRgba[p + 3] = 128;
			}
		}
		let cutoutBuffer = null;
		let cropW = 0;
		let cropH = 0;
		const srcRgba = this.baseImageData.data;
		if (fgCount > 0 && maxX >= minX && maxY >= minY) {
			cropW = Math.max(1, maxX - minX + 1);
			cropH = Math.max(1, maxY - minY + 1);
			const cutoutRgba = new Uint8ClampedArray(cropW * cropH * 4);
			for (let cy = 0; cy < cropH; cy++) {
				const srcY = minY + cy;
				for (let cx = 0; cx < cropW; cx++) {
					const srcX = minX + cx;
					const srcIdx = srcY * W + srcX;
					if (binaryMask[srcIdx]) {
						const srcP = srcIdx * 4;
						const dstP = (cy * cropW + cx) * 4;
						cutoutRgba[dstP] = srcRgba[srcP];
						cutoutRgba[dstP + 1] = srcRgba[srcP + 1];
						cutoutRgba[dstP + 2] = srcRgba[srcP + 2];
						cutoutRgba[dstP + 3] = 255;
					}
				}
			}
			cutoutBuffer = cutoutRgba.buffer;
		}
		return {
			binaryMask,
			overlayData: new ImageData(overlayRgba, W, H),
			cutoutBuffer,
			cropW,
			cropH,
			fgCount
		};
	}
	/**
	* Smoothly interpolates each of the 64 contour control points toward `targetPts`
	* so the white bounding outline morphs organically across SAM updates and slider changes.
	*/
	getAnimatedOutlinePoints(targetPts) {
		if (!this.animatedOutlinePoints || this.animatedOutlinePoints.length !== targetPts.length) {
			this.animatedOutlinePoints = targetPts.map((p) => ({
				x: p.x,
				y: p.y
			}));
			return this.animatedOutlinePoints;
		}
		const alpha = .22;
		const maxStep = .08;
		let maxDelta = 0;
		for (let i = 0; i < targetPts.length; i++) {
			const cur = this.animatedOutlinePoints[i];
			const tgt = targetPts[i];
			const dx = tgt.x - cur.x;
			const dy = tgt.y - cur.y;
			const stepX = Math.max(-.08, Math.min(maxStep, dx * alpha));
			const stepY = Math.max(-.08, Math.min(maxStep, dy * alpha));
			cur.x += stepX;
			cur.y += stepY;
			const rem = Math.max(Math.abs(tgt.x - cur.x), Math.abs(tgt.y - cur.y));
			if (rem > maxDelta) maxDelta = rem;
		}
		if (!this.isVideoMode && maxDelta > .0015 && this.outlineAnimFrameId === null) this.outlineAnimFrameId = requestAnimationFrame(() => {
			this.outlineAnimFrameId = null;
			this.renderCanvas();
		});
		return this.animatedOutlinePoints;
	}
	drawSmoothOutline(ctx, pts, sw, sh) {
		if (pts.length < 3) return;
		ctx.save();
		ctx.globalAlpha = .95;
		ctx.beginPath();
		const n = pts.length;
		const p0x = pts[0].x * sw;
		const p0y = pts[0].y * sh;
		const pLastX = pts[n - 1].x * sw;
		const pLastY = pts[n - 1].y * sh;
		ctx.moveTo((pLastX + p0x) * .5, (pLastY + p0y) * .5);
		for (let i = 0; i < n; i++) {
			const cur = pts[i];
			const next = pts[(i + 1) % n];
			const curX = cur.x * sw;
			const curY = cur.y * sh;
			const midX = (curX + next.x * sw) * .5;
			const midY = (curY + next.y * sh) * .5;
			ctx.quadraticCurveTo(curX, curY, midX, midY);
		}
		ctx.closePath();
		ctx.lineJoin = "round";
		ctx.lineCap = "round";
		ctx.strokeStyle = "rgba(255, 255, 255, 0.35)";
		ctx.lineWidth = 6;
		ctx.stroke();
		ctx.strokeStyle = "#ffffff";
		ctx.lineWidth = 2.4;
		ctx.stroke();
		ctx.restore();
	}
	renderCanvas() {
		const ctx = this.displayCtx;
		ctx.clearRect(0, 0, MODEL_IMG_SIZE, MODEL_IMG_SIZE);
		ctx.drawImage(this.baseCanvas, 0, 0);
		const activeSmooth = this.hoveredDetection && this.hoveredDetection.box ? this.getGlobalSmoothBox(this.hoveredDetection) : null;
		if ((this.showMask || this.showOutline) && this.hasMask) {
			ctx.save();
			if (this.hoveredDetection && activeSmooth) {
				const det = this.hoveredDetection;
				const srcBox = det.maskBox ?? det.box;
				const cx = (activeSmooth.x1 + activeSmooth.x2) * .5;
				const cy = (activeSmooth.y1 + activeSmooth.y2) * .5;
				const sw = activeSmooth.x2 - activeSmooth.x1;
				const sh = activeSmooth.y2 - activeSmooth.y1;
				ctx.save();
				ctx.translate(cx, cy);
				ctx.rotate(this.activeAngle);
				if (this.showMask) {
					ctx.globalAlpha = this.maskOpacity;
					ctx.drawImage(this.maskCanvas, Math.floor(srcBox.x1), Math.floor(srcBox.y1), Math.max(1, Math.floor(srcBox.x2 - srcBox.x1)), Math.max(1, Math.floor(srcBox.y2 - srcBox.y1)), -sw * .5, -sh * .5, sw, sh);
				}
				if (this.showOutline && det.contourProfile) {
					const targetPts = this.buildSmoothContourPoints(det.contourProfile);
					const animPts = this.getAnimatedOutlinePoints(targetPts);
					this.drawSmoothOutline(ctx, animPts, sw, sh);
				}
				ctx.restore();
			} else {
				if (this.showMask) {
					ctx.globalAlpha = this.maskOpacity;
					ctx.drawImage(this.maskCanvas, 0, 0);
				}
				if (this.showOutline && this.manualContourProfile) {
					const targetPts = this.buildSmoothContourPoints(this.manualContourProfile);
					const animPts = this.getAnimatedOutlinePoints(targetPts);
					ctx.save();
					ctx.translate(MODEL_IMG_SIZE * .5, MODEL_IMG_SIZE * .5);
					this.drawSmoothOutline(ctx, animPts, MODEL_IMG_SIZE, MODEL_IMG_SIZE);
					ctx.restore();
				}
			}
			ctx.restore();
		}
		if (!this.showPrompt) return;
		ctx.save();
		if (this.mode === "hover") {
			if (this.showAllYoloeBoxes) for (const det of this.activeYoloeDetections) {
				if (det === this.hoveredDetection) continue;
				const { x1, y1, x2, y2 } = this.getSmoothBox(det);
				ctx.strokeStyle = "rgba(192, 132, 252, 0.45)";
				ctx.lineWidth = 1.5;
				ctx.setLineDash([4, 4]);
				ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
				ctx.setLineDash([]);
			}
			if (this.hoveredDetection && activeSmooth) {
				const { x1, y1, x2, y2 } = activeSmooth;
				ctx.strokeStyle = this.hoverLocked ? "#facc15" : "#c084fc";
				ctx.lineWidth = 2.5;
				ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
				const labelText = `${this.hoveredDetection.className} ${(this.hoveredDetection.score * 100).toFixed(0)}%`;
				ctx.font = "bold 12px -apple-system, BlinkMacSystemFont, sans-serif";
				const pillW = ctx.measureText(labelText).width + 14;
				const pillH = 22;
				const px = Math.max(4, Math.min(MODEL_IMG_SIZE - pillW - 4, x1));
				const py = y1 > 26 ? y1 - 24 : y1 + 4;
				ctx.fillStyle = this.hoverLocked ? "rgba(234, 179, 8, 0.95)" : "rgba(126, 34, 206, 0.92)";
				ctx.beginPath();
				ctx.roundRect(px, py, pillW, pillH, 6);
				ctx.fill();
				ctx.fillStyle = this.hoverLocked ? "#0f172a" : "#ffffff";
				ctx.textBaseline = "middle";
				ctx.fillText(labelText, px + 7, py + pillH * .5);
			} else if (this.hoverPoint && this.hoverFallbackToSam) {
				const { x, y } = this.hoverPoint;
				ctx.beginPath();
				ctx.arc(x, y, 10, 0, Math.PI * 2);
				ctx.strokeStyle = this.hoverLocked ? "#facc15" : "#38bdf8";
				ctx.lineWidth = 2.2;
				ctx.stroke();
			}
		} else if (this.mode === "bbox" && this.currentBox) {
			const { x1, y1, x2, y2 } = this.currentBox;
			const w = x2 - x1;
			const h = y2 - y1;
			ctx.fillStyle = "rgba(56, 189, 248, 0.12)";
			ctx.fillRect(x1, y1, w, h);
			ctx.strokeStyle = "#38bdf8";
			ctx.lineWidth = 2.5;
			ctx.setLineDash([7, 5]);
			ctx.strokeRect(x1, y1, w, h);
			ctx.setLineDash([]);
			this.drawHandle(ctx, x1, y1, "#38bdf8", "TL");
			this.drawHandle(ctx, x2, y2, "#38bdf8", "BR");
			if (this.includeBboxCenter) this.drawPointMarker(ctx, (x1 + x2) * .5, (y1 + y2) * .5, 1);
		} else if (this.mode === "circle" && this.circlePath.length > 0) {
			ctx.lineCap = "round";
			ctx.lineJoin = "round";
			ctx.beginPath();
			ctx.moveTo(this.circlePath[0].x, this.circlePath[0].y);
			for (let i = 1; i < this.circlePath.length; i++) ctx.lineTo(this.circlePath[i].x, this.circlePath[i].y);
			ctx.strokeStyle = "rgba(56, 189, 248, 0.6)";
			ctx.lineWidth = 14;
			ctx.shadowColor = "rgba(59, 130, 246, 0.95)";
			ctx.shadowBlur = 16;
			ctx.stroke();
			ctx.beginPath();
			ctx.moveTo(this.circlePath[0].x, this.circlePath[0].y);
			for (let i = 1; i < this.circlePath.length; i++) ctx.lineTo(this.circlePath[i].x, this.circlePath[i].y);
			ctx.strokeStyle = "rgba(255, 255, 255, 0.96)";
			ctx.lineWidth = 4.5;
			ctx.shadowColor = "rgba(168, 85, 247, 0.85)";
			ctx.shadowBlur = 8;
			ctx.stroke();
			ctx.shadowBlur = 0;
			const step = Math.max(1, Math.floor(this.circlePath.length / 10));
			for (let i = 0; i < this.circlePath.length; i += step) {
				const pt = this.circlePath[i];
				const ox = Math.sin(i * 2.3) * 7;
				const oy = Math.cos(i * 1.9) * 7;
				ctx.beginPath();
				ctx.arc(pt.x + ox, pt.y + oy, 2.5, 0, Math.PI * 2);
				ctx.fillStyle = i % 2 === 0 ? "#e0f2fe" : "#f5d0fe";
				ctx.fill();
			}
			if (!this.isDragging && this.circleDerivedBox) {
				const { x1, y1, x2, y2 } = this.circleDerivedBox;
				ctx.setLineDash([6, 5]);
				ctx.strokeStyle = "rgba(56, 189, 248, 0.8)";
				ctx.lineWidth = 1.8;
				ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
			}
		} else if (this.mode === "points") for (let i = 0; i < this.promptPoints.length; i++) {
			const pt = this.promptPoints[i];
			this.drawPointMarker(ctx, pt.x, pt.y, pt.label, i + 1);
		}
		ctx.restore();
	}
	drawHandle(ctx, x, y, color, tag) {
		ctx.fillStyle = color;
		ctx.beginPath();
		ctx.arc(x, y, 5, 0, Math.PI * 2);
		ctx.fill();
		ctx.strokeStyle = "#0f172a";
		ctx.lineWidth = 1.5;
		ctx.stroke();
		ctx.font = "bold 10px system-ui, sans-serif";
		ctx.fillStyle = "#e0f2fe";
		ctx.fillText(tag, x + 7, y - 6);
	}
	drawPointMarker(ctx, x, y, label, index) {
		const isPos = label === 1;
		const fill = isPos ? "#22c55e" : "#ef4444";
		ctx.save();
		ctx.shadowColor = "rgba(0, 0, 0, 0.65)";
		ctx.shadowBlur = 6;
		ctx.beginPath();
		ctx.arc(x, y, 9, 0, Math.PI * 2);
		ctx.fillStyle = fill;
		ctx.fill();
		ctx.lineWidth = 2.2;
		ctx.strokeStyle = "#ffffff";
		ctx.stroke();
		ctx.shadowBlur = 0;
		ctx.fillStyle = "#ffffff";
		ctx.font = "bold 13px system-ui, sans-serif";
		ctx.textAlign = "center";
		ctx.textBaseline = "middle";
		ctx.fillText(isPos ? "+" : "−", x, y + .5);
		if (index !== void 0) {
			ctx.font = "bold 10px system-ui, sans-serif";
			ctx.fillStyle = "#f8fafc";
			ctx.fillText(String(index), x + 13, y - 9);
		}
		ctx.restore();
	}
	clearCutoutPreview() {
		this.lastCutoutCropCanvas = null;
		this.downloadCutoutBtn.disabled = true;
		const ctx = this.cutoutCtx;
		ctx.clearRect(0, 0, 320, 320);
		ctx.fillStyle = "#64748b";
		ctx.font = "600 13px system-ui, sans-serif";
		ctx.textAlign = "center";
		ctx.textBaseline = "middle";
		ctx.fillText("Hover or segment an object to preview cutout", 160, 160);
	}
	updateCutoutPreview(cutoutRgbaBuffer, cropW, cropH) {
		if (!cutoutRgbaBuffer || cropW <= 0 || cropH <= 0) {
			this.clearCutoutPreview();
			return;
		}
		const cropCanvas = document.createElement("canvas");
		cropCanvas.width = cropW;
		cropCanvas.height = cropH;
		cropCanvas.getContext("2d").putImageData(new ImageData(new Uint8ClampedArray(cutoutRgbaBuffer), cropW, cropH), 0, 0);
		this.lastCutoutCropCanvas = cropCanvas;
		this.downloadCutoutBtn.disabled = false;
		const ctx = this.cutoutCtx;
		ctx.clearRect(0, 0, 320, 320);
		const scale = Math.min(280 / Math.max(16, cropW), 280 / Math.max(16, cropH));
		const drawW = cropW * scale;
		const drawH = cropH * scale;
		const dx = (320 - drawW) * .5;
		const dy = (320 - drawH) * .5;
		ctx.save();
		ctx.shadowColor = "rgba(56, 189, 248, 0.55)";
		ctx.shadowBlur = 12;
		ctx.drawImage(cropCanvas, 0, 0, cropW, cropH, dx, dy, drawW, drawH);
		ctx.restore();
	}
	stopVideoMode() {
		this.isVideoMode = false;
		if (this.videoAnimationFrameId !== null) {
			cancelAnimationFrame(this.videoAnimationFrameId);
			this.videoAnimationFrameId = null;
		}
		if (this.webcamVideo && this.webcamVideo.srcObject) {
			this.webcamVideo.srcObject.getTracks().forEach((t) => t.stop());
			this.webcamVideo.srcObject = null;
		}
	}
	startVideoSource(videoEl) {
		this.stopVideoMode();
		this.webcamVideo = videoEl;
		this.isVideoMode = true;
		this.loadedSourceImage = null;
		const renderLoop = () => {
			if (!this.isVideoMode) return;
			if (videoEl.readyState >= 2) {
				const ctx = this.baseCtx;
				ctx.fillStyle = "#0f172a";
				ctx.fillRect(0, 0, MODEL_IMG_SIZE, MODEL_IMG_SIZE);
				const srcW = videoEl.videoWidth || 512;
				const srcH = videoEl.videoHeight || 512;
				const scale = Math.max(MODEL_IMG_SIZE / srcW, MODEL_IMG_SIZE / srcH);
				const sw = MODEL_IMG_SIZE / scale;
				const sh = MODEL_IMG_SIZE / scale;
				const sx = (srcW - sw) * .5;
				const sy = (srcH - sh) * .5;
				const mirrorToggle = document.getElementById("mirror-camera-toggle");
				const shouldMirror = this.cameraFacingMode === "user" && (mirrorToggle?.checked ?? true);
				ctx.save();
				if (shouldMirror) {
					ctx.translate(MODEL_IMG_SIZE, 0);
					ctx.scale(-1, 1);
				}
				ctx.drawImage(videoEl, sx, sy, sw, sh, 0, 0, MODEL_IMG_SIZE, MODEL_IMG_SIZE);
				ctx.restore();
			}
			this.renderCanvas();
			this.videoAnimationFrameId = requestAnimationFrame(renderLoop);
		};
		this.videoAnimationFrameId = requestAnimationFrame(renderLoop);
		const backgroundDetectionLoop = async () => {
			if (!this.isVideoMode) return;
			if (videoEl.readyState >= 2) {
				this.baseImageData = this.baseCtx.getImageData(0, 0, MODEL_IMG_SIZE, MODEL_IMG_SIZE);
				this.currentImageVersion++;
				this.cachedEmbeddings = null;
				try {
					await this.analyzeCurrentImage("live video stream");
				} catch (err) {
					console.warn("Background AI inference error:", err);
				}
			}
			if (this.isVideoMode) setTimeout(backgroundDetectionLoop, 400);
		};
		backgroundDetectionLoop();
	}
	async loadPresetImage(preset) {
		if (preset === "desk") {
			const synthCanvas = this.createSyntheticDeskScene();
			this.loadedSourceImage = synthCanvas;
			this.drawSourceToBaseCanvas(synthCanvas);
			this.clearPrompts();
			if (this.samReady || this.yoloeReady) this.analyzeCurrentImage("Studio Desk");
			return;
		}
		const url = PRESET_URLS[preset] || PRESET_URLS.livingroom;
		try {
			const img = new Image();
			img.crossOrigin = "anonymous";
			await new Promise((resolve, reject) => {
				img.onload = () => resolve();
				img.onerror = () => reject(/* @__PURE__ */ new Error(`Failed to load ${url}`));
				img.src = url;
			});
			this.loadedSourceImage = img;
			this.drawSourceToBaseCanvas(img);
			this.clearPrompts();
			if (this.samReady || this.yoloeReady) this.analyzeCurrentImage(preset);
		} catch (err) {
			console.warn("Preset image load fell back to synthetic desk:", err);
			const synthCanvas = this.createSyntheticDeskScene();
			this.loadedSourceImage = synthCanvas;
			this.drawSourceToBaseCanvas(synthCanvas);
			this.clearPrompts();
			if (this.samReady || this.yoloeReady) this.analyzeCurrentImage("Studio Desk");
		}
	}
	loadUploadedFile(file) {
		const reader = new FileReader();
		reader.onload = () => {
			const img = new Image();
			img.onload = () => {
				this.loadedSourceImage = img;
				this.drawSourceToBaseCanvas(img);
				this.clearPrompts();
				if (this.samReady || this.yoloeReady) this.analyzeCurrentImage(`${file.name} (${img.naturalWidth}×${img.naturalHeight})`);
			};
			img.src = reader.result;
		};
		reader.readAsDataURL(file);
	}
	drawSourceToBaseCanvas(source) {
		const srcW = source instanceof HTMLImageElement ? source.naturalWidth : source.width;
		const srcH = source instanceof HTMLImageElement ? source.naturalHeight : source.height;
		const ctx = this.baseCtx;
		ctx.fillStyle = "#0f172a";
		ctx.fillRect(0, 0, MODEL_IMG_SIZE, MODEL_IMG_SIZE);
		if (this.fitContain) {
			const scale = Math.min(MODEL_IMG_SIZE / srcW, MODEL_IMG_SIZE / srcH);
			const dw = srcW * scale;
			const dh = srcH * scale;
			const dx = (MODEL_IMG_SIZE - dw) * .5;
			const dy = (MODEL_IMG_SIZE - dh) * .5;
			ctx.drawImage(source, 0, 0, srcW, srcH, dx, dy, dw, dh);
		} else {
			const scale = Math.max(MODEL_IMG_SIZE / srcW, MODEL_IMG_SIZE / srcH);
			const sw = MODEL_IMG_SIZE / scale;
			const sh = MODEL_IMG_SIZE / scale;
			const sx = (srcW - sw) * .5;
			const sy = (srcH - sh) * .5;
			ctx.drawImage(source, sx, sy, sw, sh, 0, 0, MODEL_IMG_SIZE, MODEL_IMG_SIZE);
		}
		this.baseImageData = ctx.getImageData(0, 0, MODEL_IMG_SIZE, MODEL_IMG_SIZE);
		this.currentImageVersion++;
		this.cachedEmbeddings = null;
		this.rawYoloeDetections = [];
		this.activeYoloeDetections = [];
		this.hoveredDetection = null;
		this.renderDetectionChips();
		this.renderCanvas();
	}
	createSyntheticDeskScene() {
		const c = document.createElement("canvas");
		c.width = MODEL_IMG_SIZE;
		c.height = MODEL_IMG_SIZE;
		const ctx = c.getContext("2d");
		const wallGrad = ctx.createLinearGradient(0, 0, 0, 320);
		wallGrad.addColorStop(0, "#1e293b");
		wallGrad.addColorStop(1, "#334155");
		ctx.fillStyle = wallGrad;
		ctx.fillRect(0, 0, 512, 320);
		const deskGrad = ctx.createLinearGradient(0, 320, 0, 512);
		deskGrad.addColorStop(0, "#78350f");
		deskGrad.addColorStop(1, "#451a03");
		ctx.fillStyle = deskGrad;
		ctx.fillRect(0, 320, 512, 192);
		ctx.fillStyle = "#94a3b8";
		ctx.beginPath();
		ctx.roundRect(70, 140, 210, 145, 10);
		ctx.fill();
		ctx.fillStyle = "#0ea5e9";
		ctx.fillRect(82, 152, 186, 120);
		ctx.fillStyle = "#cbd5e1";
		ctx.beginPath();
		ctx.moveTo(50, 315);
		ctx.lineTo(300, 315);
		ctx.lineTo(280, 285);
		ctx.lineTo(70, 285);
		ctx.closePath();
		ctx.fill();
		ctx.strokeStyle = "#dc2626";
		ctx.lineWidth = 12;
		ctx.beginPath();
		ctx.arc(395, 345, 24, -Math.PI * .45, Math.PI * .45);
		ctx.stroke();
		const mugGrad = ctx.createLinearGradient(320, 295, 395, 395);
		mugGrad.addColorStop(0, "#ef4444");
		mugGrad.addColorStop(1, "#991b1b");
		ctx.fillStyle = mugGrad;
		ctx.beginPath();
		ctx.roundRect(320, 295, 72, 92, [
			4,
			4,
			18,
			18
		]);
		ctx.fill();
		ctx.fillStyle = "#451a03";
		ctx.beginPath();
		ctx.ellipse(356, 297, 33, 8, 0, 0, Math.PI * 2);
		ctx.fill();
		ctx.fillStyle = "#d97706";
		ctx.beginPath();
		ctx.moveTo(405, 220);
		ctx.lineTo(475, 220);
		ctx.lineTo(463, 290);
		ctx.lineTo(417, 290);
		ctx.closePath();
		ctx.fill();
		for (const leaf of [
			{
				cx: 440,
				cy: 175,
				rx: 18,
				ry: 42,
				rot: 0
			},
			{
				cx: 412,
				cy: 188,
				rx: 16,
				ry: 38,
				rot: -.45
			},
			{
				cx: 468,
				cy: 188,
				rx: 16,
				ry: 38,
				rot: .45
			}
		]) {
			ctx.save();
			ctx.translate(leaf.cx, leaf.cy);
			ctx.rotate(leaf.rot);
			ctx.fillStyle = "#16a34a";
			ctx.beginPath();
			ctx.ellipse(0, 0, leaf.rx, leaf.ry, 0, 0, Math.PI * 2);
			ctx.fill();
			ctx.restore();
		}
		const orbGrad = ctx.createRadialGradient(135, 370, 6, 145, 380, 38);
		orbGrad.addColorStop(0, "#fde047");
		orbGrad.addColorStop(.5, "#f97316");
		orbGrad.addColorStop(1, "#c2410c");
		ctx.fillStyle = orbGrad;
		ctx.beginPath();
		ctx.arc(145, 380, 36, 0, Math.PI * 2);
		ctx.fill();
		return c;
	}
};
document.addEventListener("DOMContentLoaded", () => {
	new EfficientSamDesktopApp();
});
//#endregion
