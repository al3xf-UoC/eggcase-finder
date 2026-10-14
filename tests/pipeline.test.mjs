// End-to-end check of the PWA's own JS modules, using the real onnxruntime-web
// (WASM backend, same as the browser) and node-canvas for drawing.
import fs from "node:fs";
import * as ort from "onnxruntime-web";
import { createCanvas, loadImage } from "canvas";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
const HERE = path.dirname(fileURLToPath(import.meta.url));
const APP = pathToFileURL(path.resolve(HERE, "../pwa")).href;
process.chdir(path.join(HERE, "fixtures"));
const { Detector } = await import(`${APP}/js/detector.js`);
const { postprocess, DECODERS } = await import(`${APP}/js/decoders.js`);
const { letterboxParams, unletterboxBox } = await import(`${APP}/js/geometry.js`);

ort.env.wasm.numThreads = 1;
const ref = JSON.parse(fs.readFileSync("ref.json"));
let failures = 0;
const check = (ok, msg) => { console.log((ok ? "PASS " : "FAIL ") + msg); if (!ok) failures++; };

function readNpy(path) {
  const buf = fs.readFileSync(path);
  const hlen = buf.readUInt16LE(8);
  const body = buf.subarray(10 + hlen);
  return new Float32Array(body.buffer.slice(body.byteOffset, body.byteOffset + body.length));
}
const fakeFetch = (files) => async (url) => {
  const name = url.split("/").pop();
  if (!(name in files)) return { ok: false };
  const v = files[name];
  return { ok: true, json: async () => v, arrayBuffer: async () => fs.readFileSync(v).buffer };
};
function compare(js, py, tol, label) {
  const sortK = (a) => [...a].sort((p, q) => q.score - p.score);
  const a = sortK(js), b = sortK(py);
  let maxErr = 0, clsOk = true;
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    for (const k of ["x1", "y1", "x2", "y2"]) maxErr = Math.max(maxErr, Math.abs(a[i][k] - b[i][k]));
    clsOk &&= a[i].cls === b[i].cls;
  }
  check(a.length === b.length && maxErr <= tol && clsOk,
    `${label}: js ${a.length} boxes vs python ${b.length}, max coord diff ${maxErr.toFixed(4)}px`);
}

const img = await loadImage("test_beach.jpg");
const W = img.width, H = img.height;
const lbJs = letterboxParams(W, H, 640);
check(lbJs.scale === ref.lb[0] && lbJs.padX === ref.lb[1] && lbJs.padY === ref.lb[2],
  `letterbox params match Python (scale ${lbJs.scale}, pad ${lbJs.padX},${lbJs.padY})`);

// ---- A: YOLO-shaped random net ----
const metaA = { name: "test", decoder: "yolo-v8", inputSize: 640, classes: ["eggcase"], countClasses: [0], iouThreshold: 0.5, letterboxColor: 114 };
const detA = new Detector({ ort, modelBase: "/m/", createCanvas });
await detA.load({ fetchFn: fakeFetch({ "model.json": metaA, "model.onnx": "yolo_like.onnx" }) });

// 1) preprocessing: JS canvas letterbox vs PIL letterbox
const { tensor } = detA.prepare(img, W, H);
const pyIn = readNpy("py_input.npy");
let sum = 0, mx = 0;
for (let i = 0; i < pyIn.length; i++) { const d = Math.abs(tensor.data[i] - pyIn[i]); sum += d; mx = Math.max(mx, d); }
const mad = sum / pyIn.length;
// pad region must be exactly 114/255
const pad = tensor.data[0 * 640 * 640 + 5 * 640 + 320];
check(Math.abs(pad - 114 / 255) < 1e-6, `padding pixel = 114/255 (got ${(pad * 255).toFixed(2)}/255)`);
check(mad < 0.02, `input tensor vs Python: mean abs diff ${mad.toFixed(4)} (resampler differences only; max ${mx.toFixed(3)})`);

// 2) decoder + NMS + un-letterbox, given the IDENTICAL input tensor
const out = await detA.session.run({ [detA.inputName]: new ort.Tensor("float32", pyIn, [1, 3, 640, 640]) });
const outputs = detA.session.outputNames.map((n) => out[n]);
check(outputs[0].dims.join(",") === "1,5,8400", `model output dims ${outputs[0].dims}`);
const pyOut = readNpy("py_outA.npy");
let omx = 0; for (let i = 0; i < pyOut.length; i++) omx = Math.max(omx, Math.abs(outputs[0].data[i] - pyOut[i]));
check(omx < 1e-3, `onnxruntime-web raw output vs onnxruntime-python: max diff ${omx.toExponential(2)}`);
const cands = DECODERS["yolo-v8"](outputs, 0.1).map((b) => unletterboxBox(b, lbJs));
for (const c of ["0.25", "0.35", "0.5"]) compare(postprocess(cands, { confidence: +c, iouThreshold: 0.5, width: W, height: H }), ref.A[c], 1e-3, `A decode+NMS @${c}`);

// 3) full detect() on the image (includes JS preprocessing)
const t0 = performance.now();
const resA = await detA.detect(img, W, H);
const ms = performance.now() - t0;
const nFull = postprocess(resA.candidates, { confidence: 0.35, iouThreshold: 0.5, width: W, height: H }).length;
console.log(`INFO full pipeline on test image: ${nFull} boxes @0.35 (python ${ref.A["0.35"].length}); ${ms.toFixed(0)} ms total, inference ${resA.timings.inference.toFixed(0)} ms (single-thread WASM, this machine)`);

// ---- B: known boxes ----
const metaB = { ...metaA, classes: ["eggcase", "coin"], countClasses: [0] };
const detB = new Detector({ ort, modelBase: "/m/", createCanvas });
await detB.load({ fetchFn: fakeFetch({ "model.json": metaB, "model.onnx": "known_boxes.onnx" }) });
const resB = await detB.detect(img, W, H);
for (const c of ["0.1", "0.25", "0.35"]) compare(postprocess(resB.candidates, { confidence: +c, iouThreshold: 0.5, width: W, height: H }), ref.B[c], 1e-3, `B known boxes @${c}`);
const counted = postprocess(resB.candidates, { confidence: 0.35, iouThreshold: 0.5, width: W, height: H }).filter((b) => metaB.countClasses.includes(b.cls));
check(counted.length === 4, `B count excludes coin and duplicate: ${counted.length} eggcases (expected 4)`);
const top = counted.find((b) => Math.abs(b.x1 - 140) < 1);
check(top && top.y1 === 0, `box overlapping padding is clamped to the image edge (y1=${top?.y1})`);

// ---- missing model is reported cleanly ----
const detC = new Detector({ ort, modelBase: "/m/", createCanvas });
let msg = ""; try { await detC.load({ fetchFn: fakeFetch({}) }); } catch (e) { msg = e.message; }
check(msg === "MODEL_MISSING", `missing model gives MODEL_MISSING (got "${msg}")`);

console.log(failures ? `\n${failures} FAILED` : "\nALL CHECKS PASSED");
process.exit(failures ? 1 : 0);
