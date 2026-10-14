// Loads the ONNX model and runs it on an image. Knows nothing about the UI.

import { letterboxParams, rgbaToChwFloat, unletterboxBox } from "./geometry.js";
import { DECODERS } from "./decoders.js";

// Candidates below this are thrown away straight after decoding. The
// confidence slider can't go lower than this without re-running the model.
export const CANDIDATE_FLOOR = 0.1;

function defaultCanvas(w, h) {
  if (typeof OffscreenCanvas !== "undefined") return new OffscreenCanvas(w, h);
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  return c;
}

export class Detector {
  /**
   * @param {object} opts
   * @param {object} opts.ort          onnxruntime-web module
   * @param {string} opts.modelBase    URL of the folder holding model.onnx + model.json
   * @param {function} [opts.createCanvas]  (w,h) => canvas; injectable for tests
   */
  constructor({ ort, modelBase, createCanvas = defaultCanvas }) {
    this.ort = ort;
    this.modelBase = modelBase.endsWith("/") ? modelBase : modelBase + "/";
    this.createCanvas = createCanvas;
    this.session = null;
    this.meta = null;
  }

  async load({ fetchFn = fetch, modelBytes } = {}) {
    const res = await fetchFn(this.modelBase + "model.json");
    if (!res.ok) throw new Error("MODEL_MISSING");
    this.meta = await res.json();
    if (!DECODERS[this.meta.decoder]) throw new Error(`Unknown decoder "${this.meta.decoder}" in model.json`);

    let bytes = modelBytes;
    if (!bytes) {
      const r = await fetchFn(this.modelBase + "model.onnx");
      if (!r.ok) throw new Error("MODEL_MISSING");
      bytes = new Uint8Array(await r.arrayBuffer());
    }
    this.session = await this.ort.InferenceSession.create(bytes, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    this.inputName = this.session.inputNames[0];

    // Warm-up run so the first real photo isn't slow.
    const s = this.meta.inputSize;
    const zeros = new this.ort.Tensor("float32", new Float32Array(3 * s * s), [1, 3, s, s]);
    await this.session.run({ [this.inputName]: zeros });
    return this.meta;
  }

  /** Letterbox the image onto a square canvas and return the input tensor. */
  prepare(image, srcW, srcH) {
    const size = this.meta.inputSize;
    const lb = letterboxParams(srcW, srcH, size);
    const canvas = this.createCanvas(size, size);
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    const g = this.meta.letterboxColor ?? 114;
    ctx.fillStyle = `rgb(${g},${g},${g})`;
    ctx.fillRect(0, 0, size, size);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(image, lb.padX, lb.padY, lb.w, lb.h);
    const { data } = ctx.getImageData(0, 0, size, size);
    const tensor = new this.ort.Tensor("float32", rgbaToChwFloat(data, size, size), [1, 3, size, size]);
    return { tensor, lb };
  }

  /**
   * Run the model. Returns every candidate above CANDIDATE_FLOOR in original
   * image pixels (unclamped); apply postprocess() for the final boxes.
   */
  async detect(image, srcW, srcH) {
    const t0 = performance.now();
    const { tensor, lb } = this.prepare(image, srcW, srcH);
    const t1 = performance.now();
    const out = await this.session.run({ [this.inputName]: tensor });
    const t2 = performance.now();
    const outputs = this.session.outputNames.map((n) => out[n]);
    const decoded = DECODERS[this.meta.decoder](outputs, CANDIDATE_FLOOR);
    const candidates = decoded.map((b) => unletterboxBox(b, lb));
    const t3 = performance.now();
    return { candidates, timings: { prepare: t1 - t0, inference: t2 - t1, decode: t3 - t2 } };
  }
}
