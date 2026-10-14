// UI wiring: photo in, boxes and a count out.

import * as ort from "../vendor/ort/ort.wasm.min.mjs";
import { Detector } from "./detector.js";
import { postprocess } from "./decoders.js";

const MAX_SIDE = 2048; // downscale huge phone photos; plenty for a 640px model

const $ = (id) => document.getElementById(id);
const statusEl = $("status");
const resultEl = $("result");
const canvas = $("view");
const ctx = canvas.getContext("2d");
const confInput = $("conf");

// --- ONNX Runtime setup ------------------------------------------------------
// Files are self-hosted (no CDN) so it works offline and behind school filters.
ort.env.wasm.wasmPaths = new URL("../vendor/ort/", import.meta.url).href;
// Multi-threading needs cross-origin isolation, which GitHub Pages can't set.
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

const detector = new Detector({ ort, modelBase: new URL("../model/", import.meta.url).href });

const state = {
  image: null,        // ImageBitmap or HTMLImageElement at display resolution
  w: 0, h: 0,
  candidates: [],
  boxes: [],          // after postprocess, sorted for numbering
  excluded: new Set(),// indices into boxes the user crossed out
  busy: false,
};

function setStatus(text, kind = "") {
  statusEl.textContent = text;
  statusEl.className = "status" + (kind ? " " + kind : "");
  statusEl.hidden = !text;
}

// --- Loading -----------------------------------------------------------------
async function init() {
  try {
    const meta = await detector.load();
    setStatus("Ready. Take a photo of the strandline, or choose one you've already taken.");
    confInput.value = Math.round((meta.defaultConfidence ?? 0.35) * 100);
    $("model-info").textContent = `Model: ${meta.name} ${meta.version ?? ""} · ${meta.inputSize}px · ${meta.licence ?? ""}`;
    updateConfLabel();
  } catch (err) {
    console.error(err);
    if (err.message === "MODEL_MISSING") {
      setStatus("No detector model installed yet. Copy model.onnx and model.json from the training notebook into the app's model folder.", "error");
    } else {
      setStatus("Sorry, the detector couldn't start on this device: " + err.message, "error");
    }
    for (const i of document.querySelectorAll("input[type=file]")) i.disabled = true;
  }
}

// --- Image input -------------------------------------------------------------
async function loadImage(file) {
  let bmp;
  try {
    bmp = await createImageBitmap(file, { imageOrientation: "from-image" });
  } catch {
    // Older Safari: fall back to an <img>, which also applies EXIF rotation.
    bmp = await new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error("That file doesn't look like a photo this browser can open."));
      img.src = URL.createObjectURL(file);
    });
  }
  const w = bmp.naturalWidth || bmp.width;
  const h = bmp.naturalHeight || bmp.height;
  const s = Math.min(1, MAX_SIDE / Math.max(w, h));
  if (s < 1 && typeof createImageBitmap === "function") {
    const small = await createImageBitmap(bmp, { resizeWidth: Math.round(w * s), resizeHeight: Math.round(h * s), resizeQuality: "high" });
    bmp.close?.();
    return { image: small, w: small.width, h: small.height };
  }
  return { image: bmp, w, h };
}

async function handleFile(file) {
  if (!file || state.busy || !detector.session) return;
  state.busy = true;
  document.body.classList.add("busy");
  setStatus("Looking for eggcases…");
  try {
    const { image, w, h } = await loadImage(file);
    state.image?.close?.();
    Object.assign(state, { image, w, h });
    // Let the status repaint before the (blocking-ish) model run.
    await new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0)));
    const { candidates, timings } = await detector.detect(image, w, h);
    state.candidates = candidates;
    console.info("timings (ms)", timings);
    refresh();
    resultEl.hidden = false;
    setStatus("");
    resultEl.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    console.error(err);
    setStatus("Something went wrong with that photo: " + err.message, "error");
  } finally {
    state.busy = false;
    document.body.classList.remove("busy");
  }
}

for (const id of ["camera-input", "file-input"]) {
  $(id).addEventListener("change", (e) => {
    handleFile(e.target.files[0]);
    e.target.value = ""; // allow picking the same file again
  });
}

// --- Results -----------------------------------------------------------------
function updateConfLabel() {
  $("conf-out").textContent = confInput.value + "%";
}

function refresh() {
  const meta = detector.meta;
  const counted = new Set(meta.countClasses ?? [0]);
  state.boxes = postprocess(state.candidates, {
    confidence: confInput.value / 100,
    iouThreshold: meta.iouThreshold ?? 0.5,
    width: state.w,
    height: state.h,
  })
    .filter((b) => counted.has(b.cls))
    // number them in reading order: top-to-bottom bands, then left-to-right
    .sort((a, b) => {
      const band = (state.h || 1) / 8;
      return Math.floor(a.y1 / band) - Math.floor(b.y1 / band) || a.x1 - b.x1;
    });
  state.excluded.clear();
  draw();
}

function draw() {
  const { image, w, h, boxes, excluded } = state;
  canvas.width = w;
  canvas.height = h;
  ctx.drawImage(image, 0, 0, w, h);

  const lw = Math.max(2, Math.round(Math.max(w, h) / 400));
  const font = Math.max(14, Math.round(Math.max(w, h) / 45));
  ctx.font = `700 ${font}px system-ui, sans-serif`;
  ctx.textBaseline = "middle";
  ctx.textAlign = "center";

  boxes.forEach((b, i) => {
    const off = excluded.has(i);
    const bw = b.x2 - b.x1, bh = b.y2 - b.y1;
    ctx.lineWidth = lw * 2.2;
    ctx.strokeStyle = "rgba(0,0,0,0.55)";
    ctx.strokeRect(b.x1, b.y1, bw, bh);
    ctx.lineWidth = lw;
    ctx.strokeStyle = off ? "#9aa5ad" : "#ffd23f";
    ctx.setLineDash(off ? [lw * 3, lw * 2] : []);
    ctx.strokeRect(b.x1, b.y1, bw, bh);
    ctx.setLineDash([]);
    if (off) {
      ctx.beginPath();
      ctx.moveTo(b.x1, b.y1); ctx.lineTo(b.x2, b.y2);
      ctx.moveTo(b.x2, b.y1); ctx.lineTo(b.x1, b.y2);
      ctx.stroke();
    }
    // numbered tag
    const r = font * 0.8;
    const cx = Math.min(Math.max(b.x1, r), w - r);
    const cy = Math.min(Math.max(b.y1, r), h - r);
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fillStyle = off ? "#6b7780" : "#0f4c5c";
    ctx.fill();
    ctx.lineWidth = Math.max(2, lw * 0.7);
    ctx.strokeStyle = "#fff";
    ctx.stroke();
    ctx.fillStyle = "#fff";
    ctx.fillText(String(i + 1), cx, cy + 1);
  });

  const n = boxes.length - excluded.size;
  $("count").textContent = n;
  $("count-label").textContent = n === 1 ? "eggcase found" : "eggcases found";
  resultEl.classList.toggle("none", n === 0);
  if (boxes.length === 0) {
    $("count-label").textContent = "eggcases found. Try “Find more”, or a closer photo.";
  }
}

confInput.addEventListener("input", () => {
  updateConfLabel();
  if (state.image) refresh();
});

// Tap a box to cross it out (smallest box under the finger wins).
canvas.addEventListener("click", (e) => {
  if (!state.boxes.length) return;
  const rect = canvas.getBoundingClientRect();
  const x = ((e.clientX - rect.left) / rect.width) * canvas.width;
  const y = ((e.clientY - rect.top) / rect.height) * canvas.height;
  let hit = -1, hitArea = Infinity;
  state.boxes.forEach((b, i) => {
    const area = (b.x2 - b.x1) * (b.y2 - b.y1);
    if (x >= b.x1 && x <= b.x2 && y >= b.y1 && y <= b.y2 && area < hitArea) { hit = i; hitArea = area; }
  });
  if (hit < 0) return;
  state.excluded.has(hit) ? state.excluded.delete(hit) : state.excluded.add(hit);
  draw();
});

$("share-btn").addEventListener("click", async () => {
  const blob = await new Promise((r) => canvas.toBlob(r, "image/jpeg", 0.9));
  const n = state.boxes.length - state.excluded.size;
  const file = new File([blob], `eggcases-${n}.jpg`, { type: "image/jpeg" });
  if (navigator.canShare?.({ files: [file] })) {
    try { await navigator.share({ files: [file], title: `${n} eggcases found` }); } catch { /* cancelled */ }
  } else {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = file.name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  }
});

// --- Offline support ---------------------------------------------------------
if ("serviceWorker" in navigator && self.isSecureContext) {
  navigator.serviceWorker.register("sw.js").catch((e) => console.warn("Offline mode unavailable:", e));
}

init();
